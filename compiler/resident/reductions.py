"""Rewrite loop nests that only accumulate scalars so that the device can run them.

A nest such as

    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          p = abs(u(i,j,k)) / dx
          umax = max(umax, p)
          if (v(i,j,k) > 0) count = count + 1
        end do
      end do
    end do

leaves nothing behind but ``umax`` and ``count``. No array element records one
iteration's result, so the nest cannot become an ordinary entry. It is replaced
by nests that can, and a few host statements:

* a nest gives each of up to ``SLOTS`` device threads every ``SLOTS``-th
  iteration of the original nest, in the original order, and stores what the
  thread accumulated in one element of a partial-result array;
* a second nest folds those partial results into ``FINAL`` values;
* the host brings the ``FINAL`` values back and folds them into the variables.

A nest of at most ``DIRECT`` iterations skips the middle step: ``FINAL``
threads take every ``FINAL``-th iteration. Nests that directly follow each
other bring their values back in one trip, unless a later one reads what an
earlier one accumulates. The original nest stays in place for the time the
values are not on the device.

Maximum, minimum, integer sums and logical accumulation give the same result in
any order, so such nests are rewritten wherever they are found. A sum of reals
depends on the order of its terms: it is rewritten only where the source says
the order is free, with an OpenMP ``reduction`` clause (which the preparation
turns into the directive below) or with the directive itself:

    !$fort reduction(+: total) [private(p, q)]

The device order is fixed, so a run repeats its sums exactly, but they differ
in rounding from a serial host sum. A nest in the body of another loop is
rewritten only with the directive, whatever it accumulates: without it the
outer loop may be a nest whose every iteration accumulates for itself. ``private`` names scalars the nest
assigns that carry no value out of it; without it, such a scalar must be
declared in the procedure and not be named again after the nest.
"""
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase

from fparser.two.utils import walk

from compiler.frontend.syntax import kind_of, parse_fortran, part_of
from compiler.scopes.segments import REDUCTION_LINE, parse_license

from .anchors import _strip_inline_comment

SLOTS = 65536
FINAL = 256
# A nest with at most this many iterations stores FINAL partial results directly.
DIRECT = 65536

_STATEMENTS = {"Nonlabel_Do_Stmt", "End_Do_Stmt", "If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt",
               "If_Stmt", "Assignment_Stmt"}
_OPERATORS = {"+": "+", "-": "+", "max": "max", "min": "min", ".or.": ".or.", ".and.": ".and.",
              "ior": None, "iand": None, "ieor": None, "*": None, ".eqv.": None, ".neqv.": None}






def _span(node):
    """First and last source line of a statement or construct."""
    item = getattr(node, "item", None)
    if item is not None:
        return item.span
    spans = [span for span in (_span(child) for child in node.content if kind_of(child) != "Comment")
             if span is not None]
    # A part that holds nothing but comments has no statement to place it by.
    return (min(first for first, _ in spans), max(last for _, last in spans)) if spans else None


def _names(node):
    """Lower-case names in ``node``, in source order, without the fields selected from them."""
    if node is None or isinstance(node, str):
        return []
    if kind_of(node) == "Name":
        return [str(node).lower()]
    if kind_of(node) == "Data_Ref":
        # object(i)%field: the object and its subscripts are names of the scope; the field is not.
        return _names(node.items[0]) + [name for part in node.items[1:] if kind_of(part) != "Name"
                                        for name in _names(getattr(part, "items", (None, None))[1])]
    children = getattr(node, "items", None)
    if children is None:
        children = getattr(node, "content", ())
    return [name for child in children for name in _names(child)]


@dataclass
class _Scope:
    """A procedure with the scopes its names can come from, innermost first."""

    qualified: str
    node: object
    hosts: tuple = ()


def _heading(node):
    """The statement that opens a procedure or a main program."""
    return part_of(node, "Subroutine_Stmt") or part_of(node, "Function_Stmt") or part_of(node, "Program_Stmt")


def _units(text):
    """The first parts of the names procedures of ``text`` can be selected by.

    A module's procedures are ``module::name``, an external procedure is
    ``$external::name`` and a named main program ``$program::name``; internal
    procedures add ``::name`` to their host's.
    """
    return {"$external", "$program",
            *(match.group(1).lower() for match in re.finditer(r"(?im)^\s*module\s+([a-z_]\w*)\b", text))}


def _procedures(tree):
    def subprograms(nodes, prefix, hosts):
        for node in nodes:
            statement = _heading(node) if kind_of(node) in {"Subroutine_Subprogram", "Function_Subprogram"} else None
            if statement is None:
                continue
            scope = _Scope(prefix + "::" + str(statement.items[1]).lower(), node, hosts)
            yield scope
            yield from internal(scope)

    def internal(scope):
        container = part_of(scope.node, "Internal_Subprogram_Part")
        yield from subprograms(container.content if container is not None else (), scope.qualified,
                               (scope.node, *scope.hosts))

    for module in (node for node in walk(tree) if kind_of(node) == "Module"):
        name = str(part_of(module, "Module_Stmt").items[1]).lower()
        container = part_of(module, "Module_Subprogram_Part")
        yield from subprograms(container.content if container is not None else (), name, (module,))
    # Program units of their own: external procedures and the main program.
    yield from subprograms(tree.content, "$external", ())
    for program in (node for node in tree.content if kind_of(node) == "Main_Program"):
        if part_of(program, "Program_Stmt") is not None:
            scope = _Scope("$program::" + str(part_of(program, "Program_Stmt").items[1]).lower(), program, ())
            yield scope
            yield from internal(scope)


def _declaration(name, scope):
    """``(type text, attribute text, is an array)`` of a variable one scope declares itself, or None."""
    statement = part_of(scope, "Function_Stmt")
    if statement is not None:
        result = str(statement.items[3].items[0] if statement.items[3] is not None else statement.items[1]).lower()
        prefix = statement.items[0]
        typed = next((item for item in getattr(prefix, "items", ()) if kind_of(item) == "Intrinsic_Type_Spec"), None)
        if typed is None and kind_of(prefix) == "Intrinsic_Type_Spec":
            typed = prefix
        if result == name and typed is not None:
            return str(typed), "", False
    specification = part_of(scope, "Specification_Part")
    for node in (specification.content if specification is not None else ()):
        if kind_of(node) != "Type_Declaration_Stmt":
            continue
        dtype, attributes, entities = node.items
        for entity in entities.items:
            if str(entity.items[0]).lower() == name:
                attrs = str(attributes).lower() if attributes is not None else ""
                return str(dtype), attrs, entity.items[1] is not None or "dimension" in attrs
    return None


def _category(dtype):
    text = " ".join(dtype.lower().split())
    if text.startswith(("real", "double precision")):
        return "real"
    return {"integer": "integer", "logical": "logical"}.get(text)


def _operator(statement, name):
    """How one assignment to ``name`` accumulates, or None when it does something else."""
    value = statement.items[2]
    kind = kind_of(value)
    occurrences = _names(value).count(name)
    if kind == "Logical_Literal_Constant" and not occurrences:
        return ".or." if str(value).lower() == ".true." else ".and."
    if occurrences != 1:
        return None
    if kind == "Level_2_Expr":
        # name + a - b ...: the variable leads a chain of additions, or closes one addition.
        if kind_of(value.items[2]) == "Name" and str(value.items[2]).lower() == name:
            return "+" if value.items[1] == "+" else None
        node = value
        while kind_of(node) == "Level_2_Expr" and len(node.items) == 3:
            node = node.items[0]
        return "+" if kind_of(node) == "Name" and str(node).lower() == name else None
    if kind == "Intrinsic_Function_Reference" and str(value.items[0]).lower() in {"max", "min"}:
        arguments = getattr(value.items[1], "items", ())
        if any(kind_of(item) == "Name" and str(item).lower() == name for item in arguments):
            return str(value.items[0]).lower()
        return None
    if kind in {"Equiv_Operand", "Or_Operand"} and any(
            kind_of(item) == "Name" and str(item).lower() == name for item in (value.items[0], value.items[2])):
        return str(value.items[1]).lower()
    return None


def _rename(lines, mapping):
    """Replace whole names outside comments; a name after ``%`` is a field and stays."""
    if not mapping:
        return list(lines)
    pattern = re.compile(r"(?<![\w%])(?<!%\s)(" + "|".join(map(re.escape, mapping)) + r")(?!\w)", re.IGNORECASE)
    renamed = []
    for line in lines:
        code, comment = _strip_inline_comment(line.rstrip("\n"))
        renamed.append(pattern.sub(lambda match: mapping[match.group(1).lower()], code) + comment + "\n")
    return renamed


@dataclass
class _Nest:
    loops: list            # (variable, lower, upper, step) texts of the rectangular prefix, outermost first
    body: tuple            # first and last line of the innermost prefix loop's body
    reductions: dict       # variable -> (operator, category, type text)
    temporaries: dict      # scalar the nest assigns -> type text


def _analyse(nest, scope, operators, private, lines, last_line):
    """Describe ``nest`` when it only accumulates scalars the rewrite can handle; otherwise say why not."""
    assigned, counts, obstacle = {}, {}, None
    opening = _span(nest)[0]
    for node in walk(nest):
        kind = kind_of(node)
        if kind == "Comment":
            # A comment before the DO statement belongs to the construct, but not to the loop.
            if re.match(r"\s*!\$", str(node)) and node.item is not None and node.item.span[0] > opening:
                obstacle = obstacle or "a directive inside the nest"
        elif kind.endswith("_Stmt") and kind not in _STATEMENTS:
            obstacle = obstacle or "a statement other than assignments, IF and DO: " + kind
        elif kind == "Nonlabel_Do_Stmt" and (node.items[1] is None or node.items[1].items[1] is None):
            obstacle = obstacle or "a loop that is not counted"
        elif kind == "Assignment_Stmt":
            if kind_of(node.items[0]) != "Name":
                obstacle = obstacle or "an assignment to an array element or a field"
            else:
                assigned.setdefault(str(node.items[0]).lower(), []).append(node)
    for name in _names(nest):
        counts[name] = counts.get(name, 0) + 1

    scopes = (scope.node, *scope.hosts)

    def declared(name):
        return next(((found, index) for index, found in enumerate(_declaration(name, item) for item in scopes)
                     if found is not None), (None, None))

    reductions = {}
    for name, statements in assigned.items():
        found = {_operator(statement, name) for statement in statements}
        operator = _OPERATORS.get(next(iter(found))) if len(found) == 1 and None not in found else None
        # Every mention of the variable is its own accumulation: the target, and itself once on the right.
        mentions = sum(1 + _names(statement.items[2]).count(name) for statement in statements)
        if operator is not None and counts[name] == mentions:
            reductions[name] = operator
    if not reductions:
        return None
    if obstacle is not None:
        return obstacle
    described = {}
    for name, operator in reductions.items():
        (declaration, _), stated = declared(name), operators.get(name)
        if declaration is None or declaration[2] or re.search(r"\b(parameter|pointer|allocatable)\b", declaration[1]):
            return f"no scalar declaration of {name} in the procedure, its host or its module"
        category = _category(declaration[0])
        if category is None or (category == "logical") != (operator in {".or.", ".and."}):
            return f"{name} has a type its accumulation is not rewritten for: {declaration[0]}"
        if stated is not None and _OPERATORS.get(stated) != operator:
            return f"{name} accumulates with {operator}, not with the stated {stated}"
        if operator == "+" and category == "real" and stated is None:
            return (f"the sum {name} depends on the order of its terms; "
                    f"a reduction(+: {name}) directive or OpenMP clause says the order is free")
        described[name] = (operator, category, declaration[0])

    loops, node, bound = [], nest, set(assigned)
    while True:
        statement = part_of(node, "Nonlabel_Do_Stmt")
        variable, limits = statement.items[1].items[1]
        variable = str(variable).lower()
        if any(name in bound for limit in limits for name in _names(limit)):
            break  # A bound that the nest computes: this loop runs inside each iteration.
        loops.append((variable, *(str(limit) for limit in limits), *(["1"] if len(limits) == 2 else [])))
        bound.add(variable)
        inner = [item for item in node.content if kind_of(item) not in {"Comment", "Nonlabel_Do_Stmt", "End_Do_Stmt"}]
        if len(inner) != 1 or kind_of(inner[0]) != "Block_Nonlabel_Do_Construct":
            break
        node = inner[0]
    if not loops:
        return "loop bounds that the nest computes"
    if any(variable in assigned for variable, *_ in loops):
        return "an assignment to a loop variable"
    innermost = nest
    for _ in loops[1:]:
        innermost = next(item for item in innermost.content if kind_of(item) == "Block_Nonlabel_Do_Construct")
    body = (_span(part_of(innermost, "Nonlabel_Do_Stmt"))[1] + 1, _span(part_of(innermost, "End_Do_Stmt"))[0] - 1)

    inner_variables = {str(statement.items[1].items[1][0]).lower() for statement in walk(innermost)
                       if kind_of(statement) == "Nonlabel_Do_Stmt"} - {loops[-1][0]}
    temporaries = {}
    after = "".join(_strip_inline_comment(line.rstrip("\n"))[0] + "\n" for line in lines[_span(nest)[1]:last_line])
    for name in sorted((set(assigned) | inner_variables) - set(described)):
        declaration, depth = declared(name)
        if declaration is None or declaration[2] or _category(declaration[0]) is None:
            return f"no scalar declaration of {name} in the procedure, its host or its module"
        if name not in private:
            # The rewritten nest computes it in a copy, so the variable itself must not be needed afterwards.
            if depth != 0 or "save" in declaration[1]:
                return f"{name} is not local to the procedure and no private clause names it"
            if re.search(rf"(?i)(?<![\w%]){re.escape(name)}(?!\w)", after):
                return f"{name} is named again after the nest and no private clause names it"
        temporaries[name] = declaration[0]

    return _Nest(loops, body, described, temporaries)


def _lowered(nest, info, number, lines):
    """Text that replaces the nest, and the declarations it needs."""
    first, last = _span(nest)
    original = lines[first - 1:last]
    indent = original[0][:len(original[0]) - len(original[0].lstrip())]
    tag = f"fort_red{number}_"
    count = len(info.loops)
    reductions = list(info.reductions.items())
    copies = {name: f"{tag}v_{name}" for name in (*info.temporaries, *(loop[0] for loop in info.loops))}
    accumulators = {name: f"{tag}a{index}" for index, (name, _) in enumerate(reductions, 1)}

    declarations = [
        f"logical, parameter :: {tag}compiled = .true.",
        f"logical :: {tag}on, {tag}now",
        f"integer(selected_int_kind(18)) :: {tag}wide",
        f"integer :: {tag}slot, {tag}t, {tag}o, {tag}total, {tag}used, {tag}last",
        *(f"integer :: {tag}l{axis}, {tag}s{axis}, {tag}n{axis}, {copies[loop[0]]}"
          for axis, loop in enumerate(info.loops, 1)),
        *(f"{dtype} :: {copies[name]}" for name, dtype in info.temporaries.items())]
    start, fold, merge, store, initial = [], [], [], [], []
    needs_infinity = False
    for index, (name, (operator, category, dtype)) in enumerate(reductions, 1):
        accumulator, partial, final, identity = accumulators[name], f"{tag}p{index}", f"{tag}q{index}", f"{tag}i{index}"
        stored = "integer" if category == "logical" else dtype
        # A scalar that lives only inside a compiled nest is a one-byte logical there.
        declarations += [f"{'logical(1)' if category == 'logical' else dtype} :: {accumulator}",
                         f"{stored}, save, target :: {partial}({SLOTS}), {final}({FINAL})"]
        if operator in {"max", "min"}:
            declarations.append(f"{dtype} :: {identity}")
            if category == "real":
                needs_infinity = True
                sign = "fort_red_negative_inf" if operator == "max" else "fort_red_positive_inf"
                initial.append(f"{identity} = fort_red_ieee_value({name}, {sign})")
            else:
                initial.append(f"{identity} = " + (f"-huge({name}) - 1" if operator == "max" else f"huge({name})"))
            start.append(f"{accumulator} = {identity}")
            fold.append(f"{accumulator} = {operator}({accumulator}, {partial}({tag}t))")
            merge.append(f"{name} = {operator}({name}, {final}({tag}slot))")
        elif operator == "+":
            start.append(f"{accumulator} = 0")
            fold.append(f"{accumulator} = {accumulator} + {partial}({tag}t)")
            merge.append(f"{name} = {name} + {final}({tag}slot)")
        else:
            # A logical accumulates one way only: .or. can turn true, .and. can turn false.
            turned, rest = (".true.", ".false.") if operator == ".or." else (".false.", ".true.")
            test = "/= 0" if operator == ".or." else "== 0"
            start.append(f"{accumulator} = {rest}")
            fold.append(f"if ({partial}({tag}t) {test}) {accumulator} = {turned}")
            merge.append(f"if ({final}({tag}slot) {test}) {name} = {turned}")
        for target in (partial, final):
            if category == "logical":
                store.append((target, [f"if ({accumulator}) then", f"  {target}({tag}slot) = 1", "else",
                                       f"  {target}({tag}slot) = 0", "end if"]))
            else:
                store.append((target, [f"{target}({tag}slot) = {accumulator}"]))

    def put(depth, items):
        return [indent + "  " * depth + item + "\n" for item in items]

    def partial_results(depth, limit, stride, target):
        """The nest that gives each of ``limit`` slots every ``stride``-th iteration and stores what it accumulates."""
        nest_text = put(depth, [f"do {tag}slot = 1, {limit}", *("  " + item for item in start),
                                f"  do {tag}t = {tag}slot, {tag}total, {stride}"])
        if count == 1:
            nest_text += put(depth + 2, [f"{copies[info.loops[0][0]]} = {tag}l1 + ({tag}t - 1) * {tag}s1"])
        else:
            nest_text += put(depth + 2, [f"{tag}o = {tag}t - 1"])
            for axis in range(count, 1, -1):
                nest_text += put(depth + 2, [
                    f"{copies[info.loops[axis - 1][0]]} = {tag}l{axis} + mod({tag}o, {tag}n{axis}) * {tag}s{axis}",
                    f"{tag}o = {tag}o / {tag}n{axis}"])
            nest_text += put(depth + 2, [f"{copies[info.loops[0][0]]} = {tag}l1 + {tag}o * {tag}s1"])
        nest_text += _rename(lines[info.body[0] - 1:info.body[1]], {**copies, **accumulators})
        nest_text += put(depth + 1, ["end do"])
        for stored, items in store:
            if stored.startswith(f"{tag}{target}"):
                nest_text += put(depth + 1, items)
        return nest_text + put(depth, ["end do"])

    # The directive is all this text says about the domain, so the nests read as plain Fortran.
    setup = put(0, [f"{tag}wide = -1", f"{tag}now = .false.", f"!$fort active({tag}now)",
                    f"if ({tag}compiled) then", f"  if ({tag}now) then"])
    for axis, (_, lower, upper, step) in enumerate(info.loops, 1):
        setup += put(2, [f"{tag}l{axis} = {lower}", f"{tag}s{axis} = {step}",
                         f"{tag}n{axis} = max(0, (({upper}) - {tag}l{axis} + {tag}s{axis}) / {tag}s{axis})"])
    setup += put(2, [f"{tag}wide = " + " * ".join(
        [f"int({tag}n1, kind({tag}wide))", *(f"{tag}n{axis}" for axis in range(2, count + 1))])])
    setup += put(0, ["  end if", "end if",
                     # One default integer counts the iterations: a nest with more of them stays on the host.
                     f"{tag}on = {tag}wide > 0 .and. {tag}wide <= huge({tag}total)"])
    device = put(0, [f"if ({tag}on) then", f"  {tag}total = int({tag}wide)", f"  {tag}last = min({FINAL}, {tag}total)",
                     *("  " + item for item in initial), f"  if ({tag}total <= {DIRECT}) then"])
    # A short nest gives FINAL slots every FINAL-th iteration; the host folds what they store.
    device += partial_results(2, f"{tag}last", FINAL, "q")
    # A long one gives SLOTS slots every SLOTS-th iteration, and a second nest folds their results.
    device += put(1, ["else", f"  {tag}used = min({SLOTS}, {tag}total)"])
    device += partial_results(2, f"{tag}used", SLOTS, "p")
    device += put(2, [f"do {tag}slot = 1, {tag}last", *("  " + item for item in start),
                      f"  do {tag}t = {tag}slot, {tag}used, {FINAL}", *("    " + item for item in fold), "  end do"])
    for stored, items in store:
        if stored.startswith(f"{tag}q"):
            device += put(3, items)
    device += put(0, ["    end do", "  end if", "end if"])
    host = put(0, [f"if ({tag}on) then", f"  do {tag}slot = 1, {tag}last", "    !$fort host",
                   *("    " + item for item in merge), "  end do",
                   # A completed DO leaves its variable one step past the last iteration.
                   *(f"  {variable} = {tag}l{axis} + {tag}n{axis} * {tag}s{axis}"
                     for axis, (variable, *_) in enumerate(info.loops, 1)),
                   # With the values on the device and the nest not rewritten for them, the host needs them all back.
                   "else", f"  !$fort publish if({tag}wide /= 0)"])
    header = _span(part_of(nest, "Nonlabel_Do_Stmt"))[1] - first + 1
    host += [*original[:header], indent + "  !$fort host\n", *original[header:]]
    host += put(0, ["end if"])
    return {"setup": setup, "device": device, "host": host, "declarations": declarations,
            "infinity": needs_infinity, "on": f"{tag}on", "indent": indent,
            "folded": [f"{tag}q{index}" for index in range(1, len(reductions) + 1)]}


def lower_reductions(text, procedures):
    """Rewrite the accumulating loop nests of the selected procedures in one source.

    ``procedures`` are ``module::name`` values or globs, as in a selection.
    The new text reaches the residency domain through directives only.
    Returns it and one record per nest that only accumulates scalars: its ``procedure``
    and ``line``, and either ``reason``, why it stays as it is, or ``tag``,
    the prefix of the names the rewrite introduced, with ``variables``.
    ``!$fort reduction`` lines are consumed either way.
    """
    lines = text.splitlines(keepends=True)
    patterns = [pattern.lower() for pattern in procedures]
    if not any(fnmatchcase(unit, pattern.split("::", 1)[0]) for unit in _units(text) for pattern in patterns):
        return text, []
    tree = parse_fortran(text)
    edits, records, number = [], [], 0
    for scope in _procedures(tree):
        if not any(fnmatchcase(scope.qualified, pattern) for pattern in patterns):
            continue
        execution = part_of(scope.node, "Execution_Part")
        if execution is None:
            continue
        statement = _heading(scope.node)
        last_line = _span(scope.node.content[-1])[1]
        declarations, uses_infinity, lowered = [], False, False

        def runs(node, inside=False):
            """Loop nests that directly follow each other, in every statement sequence of the procedure.

            Each run comes with whether it stands in the body of a loop. A loop
            whose body holds a nest beside other statements may run around
            nests (an iteration, a time loop) or be a nest itself, with an
            accumulation of its own in every iteration; the source says which
            by stating the order of the accumulation there.
            """
            run = []
            for item in getattr(node, "content", ()):
                kind = kind_of(item)
                if kind == "Block_Nonlabel_Do_Construct":
                    run.append(item)
                    continue
                if kind == "Comment":
                    continue
                if run:
                    yield run, inside
                    run = []
                if kind in {"If_Construct", "Block_Construct", "Associate_Construct"}:
                    yield from runs(item, inside)
            if run:
                yield run, inside
            for item in getattr(node, "content", ()):
                if kind_of(item) == "Block_Nonlabel_Do_Construct":
                    body = [part for part in item.content
                            if kind_of(part) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"}]
                    if len(body) > 1 and any(kind_of(part) == "Block_Nonlabel_Do_Construct" for part in body):
                        yield from runs(item, True)

        def emit(group):
            """One replacement for nests whose results can come back together."""
            first, last = _span(group[0][0])[0], _span(group[-1][0])[1]
            parts = [item[2] for item in group]
            indent = parts[0]["indent"]
            replacement = [line for part in parts for line in part["setup"]]
            replacement += [line for part in parts for line in part["device"]]
            condition = " .or. ".join(part["on"] for part in parts)
            replacement.append(f"{indent}!$fort observe({', '.join(name for part in parts for name in part['folded'])})"
                               f" if({condition})\n")
            previous = first
            for (nest, _, part) in group:
                # Comments between the nests stay with the nest they precede.
                between = lines[previous - 1:_span(nest)[0] - 1]
                replacement += [line for line in between if line.strip() and not REDUCTION_LINE.match(line)]
                replacement += part["host"]
                previous = _span(nest)[1] + 1
            edits.append((first, last, replacement))

        for run, inside in runs(execution):
            group, accumulated = [], set()
            for nest in run:
                first, last = _span(nest)
                stated = first - 2
                while stated >= 0 and not lines[stated].strip():
                    stated -= 1
                marker = REDUCTION_LINE.match(lines[stated]) if stated >= 0 else None
                operators, private = parse_license(marker.group(0)) if marker else ({}, set())
                info = None
                if (re.match(r"(?i)\s*do\b", lines[first - 1]) and re.match(r"(?i)\s*end\s*do\b", lines[last - 1])
                        and (marker is not None or not inside)):
                    info = _analyse(nest, scope, operators, private, lines, last_line)
                record = {"procedure": scope.qualified, "line": first}
                if isinstance(info, str):
                    records.append({**record, "reason": info})
                if info is None or isinstance(info, str):
                    if group:
                        emit(group)
                    group, accumulated = [], set()
                    continue
                # Nests share one trip to the host unless a later one reads what an earlier one accumulates.
                if group and set(_names(nest)) & accumulated - set(info.reductions):
                    emit(group)
                    group, accumulated = [], set()
                number += 1
                part = _lowered(nest, info, number, lines)
                group.append((nest, info, part))
                accumulated |= set(info.reductions)
                declarations += part["declarations"]
                uses_infinity |= part["infinity"]
                lowered = True
                records.append({**record, "tag": f"fort_red{number}_", "variables": sorted(info.reductions)})
            if group:
                emit(group)
        if lowered:
            indent = " " * (len(lines[_span(statement)[0] - 1]) - len(lines[_span(statement)[0] - 1].lstrip()) + 2)
            uses = []
            if uses_infinity:
                uses += ["use, intrinsic :: ieee_arithmetic, only: fort_red_ieee_value => ieee_value, &",
                         "  fort_red_negative_inf => ieee_negative_inf, fort_red_positive_inf => ieee_positive_inf"]
            heading = _span(statement)[1]
            specification = part_of(scope.node, "Specification_Part")
            ends = [_span(item)[1] for item in (specification.content if specification is not None else ())
                    if kind_of(item) != "Comment" and _span(item) is not None]
            declared_after = max([heading, *ends])
            if declared_after == heading:
                uses, declarations = uses + declarations, []
            if uses:
                edits.append((heading + 1, heading, [indent + item + "\n" for item in uses]))
            if declarations:
                edits.append((declared_after + 1, declared_after, [indent + item + "\n" for item in declarations]))
    # A stated order is used or not; either way the line has done its work.
    replaced = [(first, last) for first, last, _ in edits if last >= first]
    for index, line in enumerate(lines):
        if REDUCTION_LINE.match(line) and not any(first <= index + 1 <= last for first, last in replaced):
            edits.append((index + 1, index + 1, ["\n"]))
    for first, last, replacement in sorted(edits, key=lambda edit: (edit[0], edit[1]), reverse=True):
        if any(first <= other_first <= last and (other_first, other_last) != (first, last)
               for other_first, other_last, _ in edits if other_last >= other_first):
            # A marker line inside a replaced nest cannot occur: nests with directives are not rewritten.
            raise ValueError("Overlapping reduction rewrites")
        lines[first - 1:last] = replacement
    return "".join(lines), records
