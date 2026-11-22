"""Write array assignments that stand outside every loop as loop nests.

A statement such as

    total = total + field(1:nx,1:ny,1:nz) * weight

is a loop nest that the source does not spell out. Entries are made from loop
nests, so in a selected procedure such a statement is replaced by its nest:

    do fort_arr1_3 = lbound(total,3), ubound(total,3)
      do fort_arr1_2 = lbound(total,2), ubound(total,2)
        do fort_arr1_1 = lbound(total,1), ubound(total,1)
          total(fort_arr1_1, fort_arr1_2, fort_arr1_3) = total(fort_arr1_1, fort_arr1_2, fort_arr1_3) + &
            field(fort_arr1_1 - (lbound(total,1)) + (1), ...) * weight
        end do
      end do
    end do

An array assignment inside a loop needs none of this: the front end already
treats its sections as further loop dimensions.

The nest assigns element by element, in order, where the statement evaluates
its whole right-hand side first. The two agree unless the statement reads
elements it also assigns, so a statement is rewritten only when

* every name in it is declared in the project's sources, in the procedure, its
  hosts or a module that reaches it through USE statements, as a scalar or as
  an array of known rank, and no array is a pointer;
* the assigned array appears on the right only as exactly the same elements;
* the right-hand side combines scalars, whole arrays, array sections and array
  elements by operators and by intrinsic functions that apply element by
  element;
* an allocatable array that is assigned as a whole keeps its shape: it also
  appears on the right as a whole, or the right-hand side is a scalar.

Anything else stays as it is, with the reason in the record.
"""
import re
from fnmatch import fnmatchcase

from fparser.two.utils import walk

from compiler.frontend.syntax import kind_of, parse_fortran, part_of
from compiler.resident.procedures import (
    declarations_end,
    declarations_of,
    heading_of,
    holds_selected,
    line_span,
    normalized,
    procedures_of,
)

from .anchors import _strip_inline_comment

# Intrinsic functions whose result for arrays is the function of each element.
_ELEMENTAL = frozenset({
    "abs", "sqrt", "exp", "log", "log10", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
    "min", "max", "mod", "modulo", "sign", "real", "dble", "int", "nint", "floor", "ceiling", "aint", "anint", "dim",
    "merge", "conjg", "aimag", "cmplx"})
_OPERATIONS = frozenset({"Level_2_Expr", "Level_2_Unary_Expr", "Add_Operand", "Mult_Operand", "Parenthesis",
                         "Level_4_Expr", "And_Operand", "Or_Operand", "Equiv_Operand"})
# Statements that give a name a shape or storage outside its type declaration.
_ELSEWHERE = frozenset({"Dimension_Stmt", "Allocatable_Stmt", "Pointer_Stmt", "Target_Stmt", "Common_Stmt",
                        "Equivalence_Stmt"})
_WIDTH = 110


class _Unchanged(Exception):
    """Why a statement stays as it is."""


def _subscripts(reference):
    listed = reference.items[1]
    return list(listed.items) if kind_of(listed) == "Section_Subscript_List" else [listed]


def _imports(node):
    """The USE statements of a scope: ``(module, only, {local name: name there})`` each."""
    found = []
    specification = part_of(node, "Specification_Part")
    for statement in (specification.content if specification is not None else ()):
        if kind_of(statement) != "Use_Stmt":
            continue
        _nature, _colons, module, only, listed = statement.items
        names = {}
        for item in (listed.items if listed is not None else ()):
            # Operators and assignments are no names of variables.
            local, there = ((str(item.items[1]), str(item.items[2])) if kind_of(item) == "Rename"
                            else (str(item), str(item)))
            if re.fullmatch(r"\w+", local) and re.fullmatch(r"\w+", there):
                names[local.lower()] = there.lower()
        found.append((str(module).lower(), "ONLY" in str(only or "").upper(), names))
    return found


class _Modules:
    """What the modules of a project declare, read from their sources when a name is looked for."""

    def __init__(self, own, others):
        # The other sources are read only when a name leads out of this one.
        self.own, self.others, self.read = own, others, {}

    def _module(self, module):
        if module not in self.read:
            self.read[module] = None
            pattern = re.compile(rf"(?im)^[ \t]*module[ \t]+{re.escape(module)}[ \t]*(?:!.*)?$")
            for text in (self.own, *([] if pattern.search(self.own) else self.others)):
                if pattern.search(text) is None:
                    continue
                # The whole source: the tree of a text that the passes read is parsed once.
                node = next((item for item in walk(parse_fortran(text)) if kind_of(item) == "Module"
                             and str(part_of(item, "Module_Stmt").items[1]).lower() == module), None)
                if node is None:
                    continue
                specification = part_of(node, "Specification_Part")
                elsewhere = {str(name).lower() for statement in (specification.content if specification else ())
                             if kind_of(statement) in _ELSEWHERE for name in walk(statement) if kind_of(name) == "Name"}
                self.read[module] = (declarations_of(node), _imports(node), elsewhere)
                break
        return self.read[module]

    def find(self, module, name, seen=()):
        """The declaration of ``name`` as ``module`` offers it, its own or one it passes on; None if unknown."""
        entry = self._module(module)
        if entry is None or module in seen:
            return None
        declarations, imports, elsewhere = entry
        if name in elsewhere:
            return None
        if name in declarations:
            return declarations[name]
        return self.through(imports, name, (*seen, module))

    def through(self, imports, name, seen=()):
        for module, only, names in imports:
            if name in names:
                found = self.find(module, names[name], seen)
            elif only or name in names.values():
                continue  # Not on the list, or renamed away.
            else:
                found = self.find(module, name, seen)
            if found is not None:
                return found
        return None


class _Names:
    """What the scopes of one procedure declare, and what reaches them from modules."""

    def __init__(self, scope, modules):
        self.declared, self.elsewhere = {}, set()
        self.modules = modules
        # Innermost first: the procedure's own USE statements, then its hosts'.
        self.imports = [item for node in (scope.node, *scope.hosts) for item in _imports(node)]
        self.own = declarations_of(scope.node)
        heading = heading_of(scope.node)
        # The arrays of a main program live as long as the run, like a module's.
        self.program = kind_of(heading) == "Program_Stmt"
        # A main program has no arguments.
        self.dummies = {str(item).lower() for item in getattr(heading.items[2], "items", ())} \
            if kind_of(heading) != "Program_Stmt" else set()
        for node in (*reversed(scope.hosts), scope.node):
            self.declared.update(declarations_of(node))
            specification = part_of(node, "Specification_Part")
            for statement in (specification.content if specification is not None else ()):
                if kind_of(statement) in _ELSEWHERE:
                    self.elsewhere |= {str(name).lower() for name in walk(statement) if kind_of(name) == "Name"}

    def find(self, name):
        """``(type, attributes, bounds, initialized)`` of ``name``, or None when the sources do not say."""
        name = name.lower()
        if name in self.elsewhere:
            return None
        if name in self.declared:
            return self.declared[name]
        return self.modules.through(self.imports, name)

    def array(self, name):
        found = self.find(name)
        return found is not None and found[2] is not None

    def rank(self, name):
        """The rank of ``name``, 0 for a scalar; raises when the sources do not say."""
        name = name.lower()
        found = self.find(name)
        if found is None:
            raise _Unchanged(f"{name} is not declared in the project's sources with all its attributes")
        _dtype, attributes, bounds, _initialized = found
        if bounds is None:
            return 0
        if "pointer" in attributes:
            raise _Unchanged(f"{name} is a pointer and may share storage with another array")
        if "*" in bounds:
            raise _Unchanged(f"{name} has no declared extent")
        depth, rank = 0, 1
        for character in bounds:
            depth += (character == "(") - (character == ")")
            rank += character == "," and depth == 0
        return rank

    def allocatable(self, name):
        return "allocatable" in self.find(name)[1]

    def private(self, name):
        """Whether ``name`` is an array of the procedure's own, with no life beyond one call.

        Such an array is working storage of the host statements around it; a
        nest that assigns it could not become an entry while they read it.
        """
        found = self.own.get(name.lower())
        return (found is not None and name.lower() not in self.dummies and not self.program
                and not re.search(r"\b(allocatable|pointer|target|save)\b", found[1]) and not found[3])


def _operands(node, names, found):
    """Collect the array-valued references of an expression into ``found``; raise when it has other parts."""
    kind = kind_of(node)
    if isinstance(node, str) or kind.endswith("_Literal_Constant") or kind.startswith("Signed_"):
        return
    if kind == "Name":
        if names.rank(str(node)):
            found.append(node)
    elif kind == "Part_Ref":
        base = str(node.items[0])
        if not names.rank(base):
            raise _Unchanged(f"{base} is not an array; a function result has no known shape")
        if any(kind_of(item) == "Subscript_Triplet" for item in _subscripts(node)):
            found.append(node)
        for item in _subscripts(node):
            if kind_of(item) != "Subscript_Triplet":
                _scalar(item, names)
    elif kind == "Intrinsic_Function_Reference":
        function = str(node.items[0]).lower()
        if function not in _ELEMENTAL:
            raise _Unchanged(f"the intrinsic {function} is not applied element by element")
        for item in getattr(node.items[1], "items", ()) if node.items[1] is not None else ():
            if kind_of(item) == "Actual_Arg_Spec":
                raise _Unchanged(f"the intrinsic {function} is called with a keyword")
            _operands(item, names, found)
    elif kind in _OPERATIONS:
        for item in node.items:
            if item is not None:
                _operands(item, names, found)
    else:
        raise _Unchanged("a part of the statement is neither a scalar nor an array of this file: "
                         + " ".join(str(node).split()))


def _scalar(node, names):
    found = []
    _operands(node, names, found)
    if found:
        raise _Unchanged("an array is used as a subscript: " + " ".join(str(node).split()))


def _sections(reference, names):
    """``(first, last, step)`` texts of each dimension an array-valued reference runs over."""
    if kind_of(reference) == "Name":
        name = str(reference)
        return [(f"lbound({name},{axis})", f"ubound({name},{axis})", None)
                for axis in range(1, names.rank(name) + 1)]
    name, result = str(reference.items[0]), []
    subscripts = _subscripts(reference)
    if len(subscripts) != names.rank(name):
        raise _Unchanged(f"{name} is not given a subscript for each dimension")
    for axis, item in enumerate(subscripts, 1):
        if kind_of(item) == "Subscript_Triplet":
            first, last, step = item.items
            result.append((f"lbound({name},{axis})" if first is None else str(first),
                           f"ubound({name},{axis})" if last is None else str(last),
                           None if step is None else str(step)))
    return result


def _element(reference, positions, names):
    """The element of an array-valued reference that belongs to the given position in each dimension."""
    if kind_of(reference) == "Name":
        return f"{reference}({', '.join(positions)})"
    positions = iter(positions)
    return f"{reference.items[0]}(" + ", ".join(
        next(positions) if kind_of(item) == "Subscript_Triplet" else str(item) for item in _subscripts(reference)) + ")"


def _position(variable, target, source):
    """Where a section stands when the assigned section stands at ``variable``."""
    (first, _last, step), (other_first, _other_last, other_step) = target, source
    if normalized(step or "1") == normalized(other_step or "1"):
        if normalized(first) == normalized(other_first):
            return variable
        return f"{variable} - ({first}) + ({other_first})"
    return f"({other_first}) + (({variable} - ({first})) / ({step or 1})) * ({other_step or 1})"


def _wrapped(text, indent):
    """One statement as lines no longer than the free-form limit allows."""
    if "'" in text or '"' in text:
        return [indent + text + "\n"]  # A blank inside a character constant is no place to continue.
    lines, current = [], indent
    for word in text.split(" "):
        if len(current) + len(word) + 3 > _WIDTH and current.strip():
            lines.append(current.rstrip() + " &\n")
            current = indent + "  "
        current += word + " "
    return [*lines, current.rstrip() + "\n"]


def _nest(statement, names, number, indent):
    """The loop nest for one array assignment, or None when the statement assigns no array of this file."""
    target, _, value = statement.items
    kind = kind_of(target)
    if kind == "Name":
        if not names.array(str(target)):
            return None
    elif kind == "Part_Ref":
        if (not names.array(str(target.items[0]))
                or not any(kind_of(item) == "Subscript_Triplet" for item in _subscripts(target))):
            return None
    else:
        return None
    name = str(target if kind == "Name" else target.items[0]).lower()
    if names.private(name):
        return None
    _operands(target, names, [])
    operands = []
    _operands(value, names, operands)
    whole = kind == "Name"
    for operand in operands:
        own = str(operand if kind_of(operand) == "Name" else operand.items[0]).lower()
        if own == name and normalized(operand) != normalized(target):
            raise _Unchanged(f"it reads other elements of {name} than it assigns")
    if whole and names.allocatable(name) and operands and not any(normalized(item) == name for item in operands):
        raise _Unchanged(f"it may give {name} a new shape")
    ranges = _sections(target, names)
    variables = [f"fort_arr{number}_{axis}" for axis in range(1, len(ranges) + 1)]
    text = " ".join(str(value).split())
    replaced = {}
    for operand in operands:
        own = _sections(operand, names)
        if len(own) != len(ranges):
            raise _Unchanged("its arrays do not have the same number of dimensions: " + " ".join(str(operand).split()))
        key = " ".join(str(operand).split())
        replaced[key] = _element(operand, [_position(variable, mine, other) for variable, mine, other
                                           in zip(variables, ranges, own, strict=True)], names)
        if key not in text:
            raise _Unchanged("the statement could not be taken apart")
    if replaced:
        # Longer references first: a section holds its array's name.
        pattern = "|".join(r"(?<![\w%])" + re.escape(key) + (r"(?!\w)(?!\s*\()" if re.fullmatch(r"\w+", key) else "")
                           for key in sorted(replaced, key=len, reverse=True))
        text = re.sub(pattern, lambda match: replaced[match.group(0)], text)
    lines = []
    for depth, (variable, (first, last, step)) in enumerate(reversed(list(zip(variables, ranges, strict=True)))):
        lines += _wrapped(f"do {variable} = {first}, {last}" + (f", {step}" if step else ""), indent + "  " * depth)
    lines += _wrapped(f"{_element(target, variables, names)} = {text}", indent + "  " * len(ranges))
    lines += [indent + "  " * depth + "end do\n" for depth in reversed(range(len(ranges)))]
    arrays = sorted({name, *(str(item if kind_of(item) == "Name" else item.items[0]).lower() for item in operands)})
    return lines, variables, arrays


def _assignments(nodes):
    """``(statement node, condition, assignment)`` for the assignments that stand in no loop nest.

    Those are the ones outside every loop, in IF branches too, and those in the
    body of a loop that holds a loop nest beside other statements: such a loop
    runs around nests, and an array assignment in its body is one more of them.
    """
    for node in nodes:
        kind = kind_of(node)
        if kind == "If_Construct":
            yield from _assignments(node.content)
        elif kind == "Block_Nonlabel_Do_Construct":
            body = [item for item in node.content if kind_of(item) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"}]
            if len(body) > 1 and any(kind_of(item) == "Block_Nonlabel_Do_Construct" for item in body):
                yield from _assignments(node.content)
        elif kind == "If_Stmt" and kind_of(node.items[1]) == "Assignment_Stmt":
            yield node, node.items[0], node.items[1]
        elif kind == "Assignment_Stmt":
            yield node, None, node


def lower_array_statements(text, procedures, sources=()):
    """Rewrite the array assignments outside loops in the selected procedures of one source.

    ``procedures`` are ``module::name`` values or globs, as in a selection.
    ``sources`` are the texts of the project's other sources: a name that
    reaches the procedure through USE statements is looked up in the module
    that declares it.
    Returns the new text and one record per array assignment found: its
    ``procedure`` and ``line``, and the ``reason`` it stays, or ``lowered``
    with ``tag``, the prefix of the nest's loop variables, the ``arrays`` it
    names and its ``text`` as the audit of host statements writes it. An
    array that is working storage of the procedure itself (a local array
    without ALLOCATABLE, SAVE or TARGET) is not looked at.
    """
    patterns = [pattern.lower() for pattern in procedures]
    if not holds_selected(text, patterns):
        return text, []
    lines = text.splitlines(keepends=True)
    edits, records, number = [], [], 0
    modules = _Modules(text, sources)
    for scope in procedures_of(parse_fortran(text)):
        if not any(fnmatchcase(scope.qualified, pattern) for pattern in patterns):
            continue
        execution = part_of(scope.node, "Execution_Part")
        if execution is None:
            continue
        names, added = _Names(scope, modules), []
        for node, condition, statement in _assignments(execution.content):
            first, last = line_span(node)
            record = {"procedure": scope.qualified, "line": first}
            indent = lines[first - 1][:len(lines[first - 1]) - len(lines[first - 1].lstrip())]
            try:
                if getattr(node.item, "label", None) is not None or any(
                        ";" in _strip_inline_comment(line.rstrip("\n"))[0] for line in lines[first - 1:last]):
                    if _nest(statement, names, 0, indent) is None:
                        continue
                    raise _Unchanged("it is labelled or shares its line with another statement")
                made = _nest(statement, names, number + 1, indent + ("  " if condition is not None else ""))
            except _Unchanged as reason:
                records.append({**record, "reason": str(reason)})
                continue
            if made is None:
                continue
            number += 1
            nest, variables, arrays = made
            if condition is not None:
                nest = [*_wrapped(f"if ({condition}) then", indent), *nest, indent + "end if\n"]
            edits.append((first, last, nest))
            added.extend(variables)
            records.append({**record, "lowered": True, "tag": f"fort_arr{number}_", "arrays": arrays,
                            "text": " ".join(str(node).split())})
        if added:
            heading = line_span(heading_of(scope.node))[1]
            after = declarations_end(scope.node)
            margin = " " * (len(lines[heading - 1]) - len(lines[heading - 1].lstrip()) + 2)
            edits.append((after + 1, after, _wrapped("integer :: " + ", ".join(added), margin)))
    for first, last, replacement in sorted(edits, key=lambda edit: (edit[0], edit[1]), reverse=True):
        lines[first - 1:last] = replacement
    return "".join(lines), records
