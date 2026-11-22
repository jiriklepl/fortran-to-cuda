"""Outline source-proven numerical loops without moving their owning storage.

This module constructs a numerical candidate, not a placement decision.  The
ordinary frontend must still prove numerical legality.  Allocation guards and
saved declarations remain in the original procedure; only the reached loop is
borrowed through ordinary array arguments and explicit original lower bounds.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from fparser.two import Fortran2003 as F
from fparser.two.utils import Base, walk

from compiler.frontend.component_bindings import component_access, references, source_scope_for
from compiler.frontend.source_effects import Binding, _children, _kind, _part
from compiler.ir import CompilationError
from compiler.scopes.numerical import Parameter
from compiler.scopes.segments import directive, fortran_lines, grouped_nodes, statement_span


@dataclass(frozen=True)
class RegionExtraction:
    span: tuple[int, int]
    source: str
    entry: str
    parameters: tuple[Parameter, ...]
    bindings: tuple[Binding, ...]
    written_resources: frozenset[str]
    source_identity: str
    runtime_guards: tuple[str, ...] = ()
    # Loops without statements that leave the DO variables read after the region with their final values.
    final_iterators: tuple[str, ...] = ()

    def write(self, directory: Path) -> Path:
        """Publish once, rejecting an accidental identity/content collision."""
        path = Path(directory) / (self.entry.split("::")[0] + ".f90")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != self.source:
            raise CompilationError("inline numerical artifact identity changed")
        path.write_text(self.source)
        return path

    def public(self):
        return {"source_identity": self.source_identity, "entry": self.entry,
                "first_line": self.span[0], "last_line": self.span[1],
                "resources": [binding.public() for binding in self.bindings],
                "parameters": [{"name": item.name, "resource": item.resource, "rank": item.rank,
                                "lower_bound_dimension": item.lower_bound_dimension} for item in self.parameters],
                "written_resources": sorted(self.written_resources),
                "runtime_guards": list(self.runtime_guards), "final_iterators": list(self.final_iterators)}


def _variable_names(node):
    """Skip syntax names such as intrinsic keywords and DO construct labels."""
    if isinstance(node, (tuple, list)):
        for child in node:
            yield from _variable_names(child)
        return
    kind = _kind(node)
    if kind == "Name":
        yield node
        return
    if kind in {"Actual_Arg_Spec", "Intrinsic_Function_Reference", "Nonlabel_Do_Stmt"}:
        children = (node.items[1],)
    elif kind in {"End_Do_Stmt", "Comment"}:
        children = ()
    else:
        children = _children(node)
    for child in children:
        yield from _variable_names(child)


def _named(analysis, routine, node, arrays):
    """Roots of the scalars, or of the arrays, that ``node`` may read.

    Where the statement uses something its references cannot be resolved for
    (an object whose association is uncertain, say), every name in it counts:
    more than it reads, which is the safe side for liveness.
    """
    try:
        return {binding.root for binding, _ in references(analysis, routine.scope, node) if bool(binding.rank) == arrays}
    except CompilationError:
        found = (analysis._binding(routine.scope, name) for name in walk(node, F.Name))
        return {binding.root for binding in found if binding is not None and bool(binding.rank) == arrays}


def _scalar_reads(analysis, routine, node):
    result = _named(analysis, routine, node, False)
    # Passing a lexical procedure to a native call may expose its captures even
    # though no scalar occurs in the actual argument text.
    for name in walk(node, F.Name):
        helper = _helper(analysis, routine, name)
        if helper is not None:
            result.update(_host_associated_reads(analysis, routine, helper))
    return result


def _host_associated_reads(analysis, routine, helper, active=frozenset()):
    """Conservative lexical captures, including callbacks and entry bounds.

    This is liveness only. It neither executes a callee nor grants an effect or
    numerical proof. Potential hidden writes are conservatively live too.
    """
    owner = helper.scope.parent
    while owner is not None and owner is not routine.scope:
        owner = owner.parent
    if owner is None:
        return set()
    if helper.qualified in active or len(active) >= analysis.depth_limit:
        raise CompilationError('inline scalar liveness needs a bounded nonrecursive lexical closure')
    active = active | {helper.qualified}
    roots = {binding.root for binding in routine.scope.bindings.values() if not binding.rank}
    found, children = set(), {}
    for part in (_part(helper.scope.node, 'Specification_Part'), helper.execution):
        if part is None:
            continue
        try:
            found.update([binding.root for binding, _ in references(analysis, helper.scope, part)
                          if binding.root in roots])
        except CompilationError:
            # Something in it cannot be resolved: every name of the host that it mentions counts as read.
            named = (analysis._binding(helper.scope, name) for name in walk(part, F.Name))
            found.update(binding.root for binding in named if binding is not None and binding.root in roots)
        for name in walk(part, F.Name):
            child = _helper(analysis, helper, name)
            if child is not None:
                children[child.qualified] = child
    for child in children.values():
        found.update(_host_associated_reads(analysis, routine, child, active))
    return found


def _element_names(routine, binding):
    """The variable, or for an array of rank two or more each of its elements, as assignment targets."""
    if binding.rank < 2:
        return [binding.name]
    shape = [tuple(int(value) for value in axis.replace("(-2147483647 - 1)", str(-(2**31))).split(":"))
             for axis in _fixed_shape(routine, binding)]
    subscripts = [()]
    for lower, upper in reversed(shape):
        subscripts = [(index, *rest) for index in range(lower, upper + 1) for rest in subscripts]
    return [f"{binding.name}({','.join(str(index) for index in subscript)})" for subscript in subscripts]


def _openmp_private(loop):
    """The scalars that the ``!$fort private`` comment of the loop's DO statement names.

    The preparation writes it there where it removes the loop's OpenMP directive.
    """
    opening = next(item for item in loop.content if _kind(item) == "Nonlabel_Do_Stmt")
    names = set()
    for item in loop.content:
        if _kind(item) == "Comment" and getattr(item, "item", None) is not None \
                and item.item.span[0] == opening.item.span[0]:
            match = re.search(r"(?i)!\$fort\s+private\s*\(([^)]*)\)", str(item))
            if match:
                names |= {name.strip().lower() for name in match.group(1).split(",") if name.strip()}
    return names


def _logical_field(binding):
    """A default LOGICAL field of the elements of a list: the entry reads the words it is stored in."""
    return "element_field" in binding.attributes and binding.dtype == "logical"


def _helper(analysis, routine, name):
    candidates = analysis._candidates(routine.scope, name)
    if len(candidates) != 1:
        return None
    return analysis.numerical_helpers.get(candidates[0])


# Statements a PURE procedure may not hold, or that act on storage beyond the procedure's own.
_IMPURE = frozenset({"Write_Stmt", "Print_Stmt", "Read_Stmt", "Open_Stmt", "Close_Stmt", "Inquire_Stmt",
                     "Backspace_Stmt", "Rewind_Stmt", "Endfile_Stmt", "Flush_Stmt", "Wait_Stmt", "Stop_Stmt",
                     "Error_Stop_Stmt", "Pointer_Assignment_Stmt", "Allocate_Stmt", "Deallocate_Stmt",
                     "Nullify_Stmt", "Entry_Stmt", "Data_Stmt", "Goto_Stmt", "Computed_Goto_Stmt"})


def _declared_pure(helper):
    statement = _part(helper.scope.node, "Subroutine_Stmt") or _part(helper.scope.node, "Function_Stmt")
    return any(str(prefix).lower() in {"pure", "elemental"} for prefix in _children(statement.items[0]))


def _pure(analysis, helper, visiting=()):
    """Whether a helper does what a PURE procedure may, whether or not it says so.

    It changes nothing but its own locals and its dummies that have INTENT(OUT)
    or INTENT(INOUT), keeps nothing from one call to the next, does no input
    or output, does not stop the program, and calls only procedures that are
    pure in the same sense or intrinsic. Every dummy has an INTENT, as in a
    PURE procedure. Such a helper can be inlined like one declared PURE.
    """
    if _declared_pure(helper):
        return True
    if helper.qualified in visiting or len(visiting) >= analysis.depth_limit:
        return False
    visiting = (*visiting, helper.qualified)
    own = helper.qualified + "::"
    for name, binding in helper.scope.bindings.items():
        if name in helper.arguments:
            if binding.intent is None and not binding.attributes & {"value"}:
                return False
        elif binding.root.startswith(own) and binding.attributes & {"save", "pointer", "target", "allocatable"}:
            return False
    specification = _part(helper.scope.node, "Specification_Part")
    for declaration in walk(specification, F.Entity_Decl) if specification is not None else ():
        name = str(declaration.items[0]).lower()
        binding = helper.scope.bindings.get(name)
        # An initialized local keeps its value from one call to the next.
        if declaration.items[3] is not None and binding is not None and "parameter" not in binding.attributes:
            return False
    if any(_kind(node) in _IMPURE for node in walk(helper.execution)):
        return False
    for statement in walk(helper.execution, F.Assignment_Stmt):
        target = statement.items[0]
        while _kind(target) in {"Part_Ref", "Data_Ref"}:
            target = target.items[0]
        binding = analysis._binding(helper.scope, target) if _kind(target) == "Name" else None
        if binding is None:
            return False
        if binding.root.startswith("argument::"):
            if binding.intent not in {"out", "inout"}:
                return False
        elif not binding.root.startswith(own):
            return False
    for node in walk(helper.execution):
        if _kind(node) not in {"Call_Stmt", "Part_Ref", "Function_Reference"} or _kind(node.items[0]) != "Name":
            continue
        child = _helper(analysis, helper, node.items[0])
        if child is not None:
            if not _pure(analysis, child, visiting):
                return False
        elif _kind(node) == "Call_Stmt":
            return False
    return True


def _helper_actuals(analysis, routine, node, *, allow_omitted_optional=False):
    """Resolve ordinary numerical calls without evaluating their actuals."""
    if _kind(node) == "Call_Stmt":
        name, arguments = node.items
    elif _kind(node) in {"Part_Ref", "Function_Reference", "Structure_Constructor"}:
        name, arguments = node.items[:2]
    else:
        return None, ()
    if _kind(name) != "Name":
        return None, ()
    helper = _helper(analysis, routine, name)
    if helper is None:
        return None, ()
    actuals, seen = {}, set()
    positional = 0
    for value in _children(arguments):
        keyword = _kind(value) == "Actual_Arg_Spec"
        if keyword:
            formal, value = str(value.items[0]).lower(), value.items[1]
        else:
            if seen or positional >= len(helper.arguments):
                raise CompilationError("inline numerical helper argument association is unsupported")
            formal = helper.arguments[positional]
            positional += 1
        if formal not in helper.arguments or formal in actuals:
            raise CompilationError("inline numerical helper argument association is ambiguous")
        actuals[formal] = value
        if keyword:
            seen.add(formal)
    missing = set(helper.arguments) - set(actuals)
    if missing:
        if not allow_omitted_optional or any(
                name not in helper.scope.bindings or "optional" not in helper.scope.bindings[name].attributes
                for name in missing):
            raise CompilationError("inline numerical helper requires all original arguments")
        actuals.update((name, None) for name in missing)
    return helper, tuple(actuals[name] for name in helper.arguments)


def _element_count(binding, reference):
    """Whether ``reference`` names an array of derived type only to ask for its SIZE."""
    parent = getattr(getattr(reference, "parent", None), "parent", None)
    return (binding.rank == 1 and str(binding.dtype).lower().startswith("type") and _kind(reference) == "Name"
            and _kind(parent) == "Intrinsic_Function_Reference" and str(parent.items[0]).lower() == "size"
            and len(_children(parent.items[1])) == 1)


def _helper_closure(analysis, routine, loops):
    """Borrow only reached, explicitly pure source helpers and their captures."""
    result, active = {}, []

    def visit(owner, body):
        # A field subscript is syntactically a Part_Ref too. Its component
        # name must not resolve as a same-spelling lexical procedure.
        component_selectors = {id(part) for reference in walk(body, F.Data_Ref) for part in reference.items}
        for item in walk(body):
            if id(item) in component_selectors:
                continue
            if _kind(item) not in {"Call_Stmt", "Part_Ref", "Function_Reference", "Structure_Constructor"}:
                continue
            helper, _ = _helper_actuals(analysis, owner, item)
            if helper is None:
                if _kind(item) == "Call_Stmt":
                    raise CompilationError("unsupported inline numerical statement: unresolved source helper")
                continue
            if helper.qualified in active:
                raise CompilationError("recursive inline numerical helper closure is unsupported")
            if helper.qualified in result:
                continue
            if not _pure(analysis, helper):
                raise CompilationError("inline numerical helper changes or keeps more than its own arguments and locals")
            if len(active) + 1 >= analysis.depth_limit or len(result) >= analysis.procedure_limit:
                raise CompilationError("inline numerical helper closure budget exceeded")
            result[helper.qualified] = helper
            active.append(helper.qualified)
            visit(helper, helper.execution)
            active.pop()

    for loop in loops:
        visit(routine, loop)
    return result


def _exception_observers(analysis):
    """Resolve intrinsic IEEE observers across configured source inputs.

    Device arithmetic does not publish the CPU exception flags. Until that
    effect can be represented, even an observer in another configured caller
    conservatively prevents admission of these new numerical closures. A user
    procedure with the same spelling is not an intrinsic observer.
    """
    observers = []
    procedures = {**analysis.routines, **analysis.numerical_helpers}
    for qualified, routine in procedures.items():
        for node in walk(routine.execution):
            if _kind(node) != "Call_Stmt" or _kind(node.items[0]) != "Name":
                continue
            for target in analysis._candidates(routine.scope, node.items[0]):
                parts = target.split("::")
                if (len(parts) == 3 and parts[0] == "$intrinsic"
                        and parts[1] in {"ieee_arithmetic", "ieee_exceptions"}
                        and parts[2] in {"ieee_get_flag", "ieee_get_status"}):
                    observers.append(qualified + " calls " + target)
    return tuple(sorted(set(observers)))


def _validate_real_varargs(analysis, routine, node):
    """The supported native contract proves unordered two-operand MIN/MAX.

    Longer real expressions can be reassociated by the native compiler. Until
    that ordering is characterized, new source closures retain native execution
    unless every operand is an explicitly finite numeric literal.
    """
    def dtype(value):
        kind = _kind(value)
        if kind == "Real_Literal_Constant":
            return "real"
        if kind == "Int_Literal_Constant":
            return "integer"
        if kind in {"Name", "Part_Ref"}:
            binding = analysis._binding(routine.scope, value if kind == "Name" else value.items[0])
            return binding.dtype if binding is not None else None
        if kind == "Intrinsic_Function_Reference":
            name = str(value.items[0]).lower()
            if name in {"size", "lbound", "ubound", "int", "nint", "ceiling", "floor", "kind"}:
                return "integer"
            if name in {"real", "sqrt", "sin", "cos", "acos", "exp", "log"}:
                return "real"
            children = _children(value.items[1])
        else:
            children = _children(value)
        types = {dtype(child) for child in children if not isinstance(child, (str, int))}
        return "real" if "real" in types else "integer" if types == {"integer"} else None

    def finite_literal(value):
        kind = _kind(value)
        if kind in {"Real_Literal_Constant", "Int_Literal_Constant"}:
            return math.isfinite(float(value.items[0].lower().replace("d", "e")))
        items = getattr(value, "items", ())
        if len(items) == 2 and str(items[0]) in {"+", "-"}:
            return finite_literal(items[1])
        if kind == "Parenthesis":
            return finite_literal(items[1])
        return False

    for item in walk(node):
        if _kind(item) != "Intrinsic_Function_Reference" or str(item.items[0]).lower() not in {"min", "max"}:
            continue
        arguments = tuple(value.items[1] if _kind(value) == "Actual_Arg_Spec" else value
                          for value in _children(item.items[1]))
        if len(arguments) > 2 and any(dtype(value) != "integer" for value in arguments) and not all(finite_literal(value) for value in arguments):
            raise CompilationError("inline real MIN/MAX with more than two operands needs proven finite values or a supported native ordering contract")


def _intrinsic_shadowed(analysis, routine, name):
    scope = routine.scope
    while scope is not None:
        if name in scope.externals or name in scope.procedure_arguments:
            return True
        scope = scope.parent
    return bool(analysis._binding(routine.scope, name) or analysis._candidates(routine.scope, F.Name(name)))


def _iterator(node):
    header = next((item for item in _children(node) if _kind(item) == "Nonlabel_Do_Stmt"), None)
    control = header.items[1] if header is not None else None
    counted = control.items[1] if control is not None and _kind(control) == "Loop_Control" else None
    if not counted or len(counted) != 2 or _kind(counted[0]) != "Name":
        raise CompilationError("inline numerical regions require counted DO loops")
    return str(counted[0]).lower(), tuple(counted[1])


def _loop_control(analysis, routine, loops, live, iterators, changed):
    """The loops of a region without their statements, for the DO variables that are read after it.

    A counted loop leaves its variable with a value that only its bounds
    decide, so the loops alone, run on the host after the entry, leave the
    variables as the region would. That holds for a variable that only DO
    statements set (``iterators``), whose loops are not under a condition, and
    whose bounds read nothing the region changes (``changed``). Loops that
    hold no such variable are left out. Returns the lines, or raises when a
    variable in ``live`` is not of that kind.
    """
    refusal = CompilationError("inline loop-written scalar is live after the region")
    if live - iterators:
        raise refusal

    def variable(loop):
        return analysis._binding(source_scope_for(analysis, loop, routine.scope), _iterator(loop)[0]).root

    def control(loop, enclosing, depth):
        name, bounds = _iterator(loop)
        root, inner = variable(loop), []
        for child in _children(loop):
            if _kind(child) == "Block_Nonlabel_Do_Construct":
                inner += control(child, enclosing | {root}, depth + 1)
            elif any(variable(item) in live for item in walk(child, F.Block_Nonlabel_Do_Construct)):
                raise refusal  # Whether that loop runs is decided by the statements left out.
        if not inner and root not in live:
            return []
        read = {binding.root for bound in bounds for binding, _ in references(analysis, routine.scope, bound)}
        if read & (changed | (iterators - enclosing)):
            raise refusal
        margin = "  " * depth
        return [f"{margin}do {name} = {', '.join(str(bound) for bound in bounds)}", *inner, f"{margin}end do"]

    return tuple(line for loop in loops for line in control(loop, frozenset(), 0))


def _fixed_shape(routine, binding):
    from compiler.ir import SourceLocation
    if not binding.rank or len(binding.shape_nodes) != binding.rank:
        raise CompilationError("inline private arrays require constant explicit bounds")
    result, elements = [], 1
    scope = binding.declaring_scope or routine.scope
    for axis in binding.shape_nodes:
        if _kind(axis) != "Explicit_Shape_Spec" or axis.items[1] is None:
            raise CompilationError("inline private arrays require constant explicit bounds")
        try:
            lower = scope.kinds.integer(axis.items[0], SourceLocation(str(scope.path))) if axis.items[0] is not None else 1
            upper = scope.kinds.integer(axis.items[1], SourceLocation(str(scope.path)))
        except CompilationError as error:
            raise CompilationError("inline private arrays require constant explicit bounds") from error
        if not -(2**31) <= lower < 2**31 or not -(2**31) <= upper < 2**31:
            raise CompilationError("inline private array bounds exceed the INTEGER ABI")
        elements *= max(0, upper - lower + 1)
        def integer_spelling(value):
            return "(-2147483647 - 1)" if value == -(2**31) else str(value)
        result.append(f"{integer_spelling(lower)}:{integer_spelling(upper)}")
    if elements > 256:
        raise CompilationError("inline private array exceeds the bounded element budget")
    return tuple(result)


def _untile(analysis, routine, loop, helpers):
    """Prove a perfect rectangular tile nest, with reached-point runtime guards.

    Only the coordinate enumeration changes. The ordinary dependence proof
    still establishes independence after this normalization; this function
    never authorizes a stencil recurrence or an index-as-data tile coordinate.
    """
    chain, current = [], loop
    while _kind(current) == "Block_Nonlabel_Do_Construct":
        chain.append(current)
        body = tuple(child for child in _children(current)
                     if _kind(child) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"})
        if len(body) != 1 or _kind(body[0]) != "Block_Nonlabel_Do_Construct":
            break
        current = body[0]
    if len(chain) < 2 or len(chain) % 2:
        return loop, (), ()
    dimensions = len(chain) // 2
    outer, inner = chain[:dimensions], chain[dimensions:]
    domains, guards = [], []

    def check(value):
        """A bound that is not a variable is a constant of the INTEGER ABI."""
        from compiler.ir import SourceLocation
        binding = analysis._binding(routine.scope, value) if _kind(value) == "Name" else None
        if binding is not None and "parameter" not in binding.attributes:
            return
        constant = routine.scope.kinds.integer(value, SourceLocation(str(routine.scope.path)))
        if not -(2**31) <= constant < 2**31:
            raise CompilationError("inline tile bound exceeds the INTEGER ABI")
    for tile, cells in zip(outer, inner, strict=True):
        tile_name, tile_bounds = _iterator(tile)
        index_name, cell_bounds = _iterator(cells)
        if len(tile_bounds) != 3 or len(cell_bounds) != 2:
            return loop, (), ()
        lower, upper, step = tile_bounds
        for value in (lower, upper, step):
            if _kind(value) == "Name":
                continue
            try:
                from compiler.ir import SourceLocation
                constant = routine.scope.kinds.integer(value, SourceLocation(str(routine.scope.path)))
                if not -(2**31) <= constant < 2**31:
                    return loop, (), ()
            except CompilationError:
                return loop, (), ()
        if str(cell_bounds[0]).lower() != tile_name:
            return loop, (), ()
        ceiling = cell_bounds[1]
        if _kind(ceiling) != "Intrinsic_Function_Reference" or str(ceiling.items[0]).lower() != "min":
            return loop, (), ()
        if _intrinsic_shadowed(analysis, routine, "min"):
            raise CompilationError("inline tile clipping intrinsic is shadowed")
        arguments = _children(ceiling.items[1])
        expected = {str(F.Level_2_Expr(f"{tile_name}+{step}-1")).lower(), str(upper).lower()}
        if len(arguments) != 2 or {str(value).lower() for value in arguments} != expected:
            return loop, (), ()
        for value in (lower, upper, step):
            if _kind(value) == "Name":
                binding = analysis._binding(routine.scope, value)
                if binding is None or binding.signature() != ("integer", 4, 0):
                    raise CompilationError("inline tile bounds require default INTEGER scalars")
                if str(value).lower() == "c_int64_t":
                    raise CompilationError("inline tile guard INTEGER kind namespace conflicts")
        if _intrinsic_shadowed(analysis, routine, "int"):
            raise CompilationError("inline tile guard conversion intrinsic is shadowed")
        predicates = [f"int({step},kind=c_int64_t) > 0_c_int64_t",
                      f"int({upper},kind=c_int64_t) + int({step},kind=c_int64_t) <= 2147483647_c_int64_t"]
        guards += predicates
        for value in (step, upper, lower):
            check(value)
        domains.append({"tile_iterator": tile_name, "iterator": index_name, "lower": str(lower), "upper": str(upper)})
    tile_names = {domain["tile_iterator"] for domain in domains}
    numerical_body = tuple(child for child in _children(inner[-1])
                           if _kind(child) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"})
    if any(str(name).lower() in tile_names for value in (*numerical_body, *(helper.execution for helper in helpers.values()))
           for name in _variable_names(value)):
        raise CompilationError("inline tile coordinates are observed by numerical work")
    # The flattened body has no tile-coordinate dependency. Drop only proved
    # outer enumeration and restore each original logical cell interval.
    result = numerical_body
    for cells, domain in reversed(tuple(zip(inner, domains, strict=True))):
        clone = copy.copy(cells)
        header = next(child for child in _children(cells) if _kind(child) == "Nonlabel_Do_Stmt")
        ending = next(child for child in _children(cells) if _kind(child) == "End_Do_Stmt")
        replacement = F.Nonlabel_Do_Stmt(f"do {domain['iterator']}={domain['lower']},{domain['upper']}")
        replacement.item = header.item
        clone.content = [replacement, *result, ending]
        result = (clone,)
    return result[0], tuple(guards), tuple(domains)


def _upward_reads(analysis, routine, nodes):
    """Conservative scalar liveness, including original DO final values."""
    defined, reads = set(), set()
    for node in nodes:
        kind = _kind(node)
        if kind == "Comment":
            continue
        if kind == "Assignment_Stmt":
            target, _, value = node.items
            own = _scalar_reads(analysis, routine, value)
            if _kind(target) != "Name":
                own |= _scalar_reads(analysis, routine, target)
            reads |= own - defined
            binding = analysis._binding(routine.scope, target) if _kind(target) == "Name" else None
            if binding and not binding.rank:
                defined.add(binding.root)
        elif kind == "Block_Nonlabel_Do_Construct":
            name, bounds = _iterator(node)
            reads |= set().union(*(_scalar_reads(analysis, routine, bound) for bound in bounds)) - defined
            index = analysis._binding(routine.scope, name)
            if index:
                defined.add(index.root)
            body = tuple(item for item in _children(node) if _kind(item) not in {
                "Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"})
            reads |= _upward_reads(analysis, routine, body) - defined
        elif kind == "Call_Stmt":
            helper, actuals = _helper_actuals(analysis, routine, node, allow_omitted_optional=True)
            if helper is None:
                reads |= _scalar_reads(analysis, routine, node) - defined
                continue
            reads |= _host_associated_reads(analysis, routine, helper) - defined
            outputs = set()
            for formal, actual in zip(helper.arguments, actuals, strict=True):
                if actual is None:
                    continue  # An absent optional has no caller storage to observe or define.
                binding = helper.scope.bindings.get(formal)
                if binding is None:
                    # Native continuations can take procedure arguments. This
                    # is liveness, not a numerical-callee proof: retain every
                    # possible actual/callback capture and infer no definition.
                    reads |= _scalar_reads(analysis, routine, actual) - defined
                    if _kind(actual) != "Name" or _helper(analysis, routine, actual) is None:
                        # A dynamic callback may capture any original local;
                        # a later coherent native fallback cannot reconstruct
                        # a loop iterator discarded by an earlier GPU worker.
                        reads |= {value.root for value in routine.scope.bindings.values()
                                  if not value.rank} - defined
                    continue
                target = analysis._binding(routine.scope, actual) if _kind(actual) == "Name" else None
                if binding.intent != "out":
                    reads |= _scalar_reads(analysis, routine, actual) - defined
                if binding.intent in {"out", "inout"} and target and not target.rank:
                    outputs.add(target.root)
            defined |= outputs
        elif kind == "If_Construct":
            body = []
            for child in _children(node):
                if _kind(child) in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                    reads |= _upward_reads(analysis, routine, body) - defined
                    body = []
                    reads |= _scalar_reads(analysis, routine, child) - defined
                else:
                    body.append(child)
            reads |= _upward_reads(analysis, routine, body) - defined
            # Branch-local definitions do not establish a definition after IF.
        else:
            # Unknown calls can read any referenced scalar; their conditional
            # writes do not establish a subsequent definition.
            reads |= _scalar_reads(analysis, routine, node) - defined
    return reads


def _upward_array_reads(analysis, routine, nodes):
    """Whole-private-array liveness across original following operations.

    A helper INTENT(OUT) defines a new private value before its later reads.
    Element writes alone do not establish a complete array definition here.
    The numerical frontend separately proves every used output element.
    """
    def reads(node):
        return _named(analysis, routine, node, True)
    defined, incoming = set(), set()
    for node in nodes:
        kind = _kind(node)
        if kind == "Comment":
            continue
        if kind == "Call_Stmt":
            helper, actuals = _helper_actuals(analysis, routine, node, allow_omitted_optional=True)
            if helper is None:
                incoming |= reads(node) - defined
                continue
            outputs = set()
            for formal, actual in zip(helper.arguments, actuals, strict=True):
                if actual is None:
                    continue
                parameter = helper.scope.bindings.get(formal)
                binding = analysis._binding(routine.scope, actual) if _kind(actual) == "Name" else None
                if parameter is None or parameter.intent != "out":
                    incoming |= reads(actual) - defined
                if parameter and parameter.intent == "out" and binding and binding.rank:
                    outputs.add(binding.root)
            defined |= outputs
        elif kind == "Assignment_Stmt":
            target, _, value = node.items
            incoming |= reads(value) - defined
            if _kind(target) == "Name":
                binding = analysis._binding(routine.scope, target)
                if binding and binding.rank:
                    defined.add(binding.root)
        elif kind == "Block_Nonlabel_Do_Construct":
            _, bounds = _iterator(node)
            incoming |= set().union(*(reads(bound) for bound in bounds)) - defined
            body = tuple(child for child in _children(node)
                         if _kind(child) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"})
            incoming |= _upward_array_reads(analysis, routine, body) - defined
        elif kind == "If_Construct":
            branch = []
            for child in _children(node):
                if _kind(child) in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                    incoming |= _upward_array_reads(analysis, routine, branch) - defined
                    branch = []
                    incoming |= reads(child) - defined
                else:
                    branch.append(child)
            incoming |= _upward_array_reads(analysis, routine, branch) - defined
        else:
            incoming |= reads(node) - defined
    return incoming


def _allocated_names(node):
    if _kind(node) != "Allocate_Stmt":
        return set()
    result = set()
    for item in walk(node):
        if _kind(item) == "Allocation":
            result.add(str(item.items[0]).lower())
    return result


def _check_allocation(analysis, routine, binding, preceding):
    """Require an original allocation site of a local array before the region, leaving its existence to a guard.

    A conditional allocation need not execute on this invocation.  Fresh
    ALLOCATED/bounds checks at the reached point establish the actual descriptor;
    no earlier allocation address, counter or proof is reused.  Allocation and
    unknown operations before this point remain outside this owning region.
    """
    if binding.rank == 0 or "allocatable" not in binding.attributes:
        raise CompilationError("inline local allocations require original allocatable array storage")
    if binding.root != routine.qualified + "::" + binding.name:
        raise CompilationError("inline allocation proof requires original procedure-local storage")
    if binding.attributes & {"target", "pointer", "optional", "volatile", "asynchronous"}:
        raise CompilationError("inline local allocation may escape or have uncertain association")
    if _intrinsic_shadowed(analysis, routine, "allocated"):
        raise CompilationError("inline allocation guard intrinsic is shadowed")
    if not any(_kind(item) == "Allocate_Stmt" and binding.name in _allocated_names(item)
               for node in preceding for item in walk(node)):
        raise CompilationError("inline local allocation lacks a preceding original allocation site: " + binding.root)


def extract_region(analysis, routine, node, *, preceding=(), following=(), cyclic_following=()):
    """Construct a bounded candidate from serial counted loops of the original procedure."""
    # The original procedure keeps its declarations and initialization. Only
    # declarations actually borrowed by this region need numerical support.
    # Implicit typing, interfaces and uncertain imports still cannot establish
    # an authority for an unresolved name below.
    specification_issues = [issue for issue in routine.issues
                            if issue != "specification effect unavailable: Derived_Type_Def"]
    if specification_issues:
        raise CompilationError("inline original specification is unsupported: " + "; ".join(specification_issues))
    if analysis._unknown_exports(routine.scope):
        raise CompilationError("inline intrinsic authority requires complete wildcard imports")
    if any(directive(item) is not None for item in walk(_part(routine.scope.node, "Specification_Part"))):
        # fparser may attach opening executable directives to specifications.
        # A detached DO alone does not prove either serial participation or the
        # complete parallel region; retain the original operation in that case.
        raise CompilationError("inline OpenMP association requires its complete original source region")
    nodes = tuple(node)
    if not nodes:
        raise CompilationError("inline numerical region is empty")
    selected_span = (min(statement_span(item)[0] for item in nodes),
                     max(statement_span(item)[1] for item in nodes))

    def complete_original_groups(sequence):
        for group in grouped_nodes(sequence):
            if isinstance(group, tuple):
                first, last = statement_span(group[0])[0], statement_span(group[-1])[1]
                if (selected_span[0] <= last and first <= selected_span[1]
                        and not (selected_span[0] <= first and last <= selected_span[1])):
                    raise CompilationError("inline OpenMP association requires its complete original source region")
            elif _kind(group) in {"If_Construct", "Associate_Construct"}:
                first, last = statement_span(group)
                if selected_span[0] <= last and first <= selected_span[1]:
                    complete_original_groups(group.content)

    # A detached DO can be a genuine original AST node while still belonging
    # to a larger team. Source identity alone must not turn it into serial work.
    complete_original_groups(_children(routine.execution))
    # fparser attaches an associated !$OMP DO prefix to its DO construct.
    # Peel only that original structural prefix; do not authorize a projected
    # routine or arbitrary foreign AST as source-backed numerical work.
    # Ordinary comments can precede an attached OpenMP prefix. They belong to
    # the exact original selection, but do not create an execution operation
    # beside the joined group. Keep that selection for completion authority.
    grouped = tuple(item for item in grouped_nodes(nodes)
                    if _kind(item) != "Comment" or directive(item) is not None)
    if len(grouped) == 1 and isinstance(grouped[0], tuple):
        nodes = grouped[0]
    elif len(grouped) == 1 or all(_kind(item) == "Block_Nonlabel_Do_Construct" for item in grouped):
        nodes = grouped
    if directive(nodes[0]) is not None:
        # A loop under an OpenMP directive is the work of a whole team; only serial loops are extracted.
        raise CompilationError("inline OpenMP association requires its complete original source region")
    if not all(_kind(item) == "Block_Nonlabel_Do_Construct" for item in nodes):
        raise CompilationError("inline numerical region requires serial counted DO loops")
    if len(nodes) > 1:
        # Membership and sorted line numbers do not prove a serial group:
        # selecting two loops must not discard an intervening native call,
        # assignment or branch. Require their exact consecutive siblings.
        wanted = tuple(id(item) for item in nodes)

        def consecutive(sequence):
            effective = tuple(item for item in sequence
                              if _kind(item) != "Comment" or directive(item) is not None)
            identities = tuple(id(item) for item in effective)
            if any(identities[start:start + len(wanted)] == wanted
                   for start in range(len(identities) - len(wanted) + 1)):
                return True
            return any(consecutive(_children(item)) for item in effective
                       if _kind(item) in {"If_Construct", "Associate_Construct", "Block_Construct",
                                          "Block_Nonlabel_Do_Construct"})

        if not consecutive(_children(routine.execution)):
            raise CompilationError("serial numerical group requires consecutive original sibling loops")
    loops = nodes
    original_nodes = {id(item) for item in walk(routine.execution)}
    for item in (*nodes, *preceding, *following, *cyclic_following):
        if id(item) in original_nodes:
            continue
        related = False
        if _kind(item) == "Block_Nonlabel_Do_Construct":
            for original in walk(routine.execution, F.Block_Nonlabel_Do_Construct):
                children = list(original.content)
                while children and _kind(children[0]) == "Comment":
                    children.pop(0)
                if (len(children) == len(item.content)
                        and all(left is right for left, right in zip(children, item.content, strict=True))):
                    related = True
                    break
        if not related:
            raise CompilationError("inline extraction nodes must belong to the original execution")
    helpers = _helper_closure(analysis, routine, loops)
    bodies = [(routine, loop) for loop in loops] + [(helper, helper.execution) for helper in helpers.values()]
    vector_reduction = any(_kind(item) == "Intrinsic_Function_Reference"
                           and str(item.items[0]).lower() in {"sum", "dot_product"}
                           for _owner, body in bodies for item in walk(body))
    if (helpers or vector_reduction) and (observers := _exception_observers(analysis)):
        raise CompilationError("source-observable floating-point exception flags prevent GPU helper closure: "
                               + "; ".join(observers))
    admitted = {"Execution_Part", "Block_Nonlabel_Do_Construct", "Nonlabel_Do_Stmt", "End_Do_Stmt", "Assignment_Stmt", "Call_Stmt",
                "If_Stmt", "If_Construct", "If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt", "Comment",
                "Continue_Stmt"}
    for owner, loop in bodies:
        if helpers or vector_reduction:
            _validate_real_varargs(analysis, owner, loop)
        for item in walk(loop):
            if ((hasattr(item, "content") or _kind(item).endswith("_Stmt")
                 or getattr(item, "item", None) is not None) and _kind(item) not in admitted):
                raise CompilationError("unsupported inline numerical statement: " + _kind(item))
            if directive(item) is not None:
                raise CompilationError("nested inline OpenMP directives require a synchronization proof")

    used = {}
    for owner, loop in bodies:
        for binding, _reference in references(analysis, owner.scope, loop):
            if (owner is not routine and "parameter" not in binding.attributes
                    and any(binding is own for own in owner.scope.bindings.values())):
                continue  # Helper dummies/private storage belong to its worker.
            if _element_count(binding, _reference):
                continue  # SIZE of an array of derived type is read from one of its element fields.
            boundary = analysis.resource_identity_boundary(binding)
            if boundary:
                raise CompilationError(boundary)
            used[binding.root] = binding
    for binding in used.values():
        # The device copy of an array is as large as the array, and this one does not say how large it is.
        if any(_kind(axis) == "Assumed_Size_Spec" for axis in binding.shape_nodes):
            raise CompilationError("an assumed-size array has no extent to keep on the device: " + binding.name)
    writes, scalar_writes, iterators, local_arrays, assigned_scalars = set(), set(), set(), set(), set()
    def write(binding, target):
        if binding is None:
            raise CompilationError("unresolved inline assignment target")
        if "parameter" in binding.attributes:
            raise CompilationError("inline region cannot write original PARAMETER storage")
        if (binding.root.startswith(routine.qualified + "::") and binding.rank
                and not hasattr(binding, "component_object")
                and not binding.attributes & {"save", "allocatable", "pointer", "target"}):
            local_arrays.add(binding.root)
        elif binding.rank:
            if (_kind(target) not in {"Name", "Part_Ref"} and not (_kind(target) == "Data_Ref"
                    and component_access(analysis, routine.scope, target).indices)):
                raise CompilationError("inline whole-array assignments require separate allocation/effect proofs")
            if binding.intent == "in":
                raise CompilationError("inline region writes original INTENT(IN) storage")
            writes.add(binding.root)
        else:
            scalar_writes.add(binding.root)
            assigned_scalars.add(binding.root)

    for owner, loop in bodies:
        for item in walk(loop):
            if _kind(item) == "Block_Nonlabel_Do_Construct":
                name, _ = _iterator(item)
                binding = analysis._binding(source_scope_for(analysis, item, owner.scope), name)
                if binding is None:
                    raise CompilationError("undeclared inline loop iterator")
                if owner is not routine and binding is owner.scope.bindings.get(name):
                    continue
                scalar_writes.add(binding.root)
                iterators.add(binding.root)
            elif _kind(item) == "Assignment_Stmt":
                target = item.items[0]
                name = target.items[0] if _kind(target) == "Part_Ref" else target
                binding = analysis._binding(source_scope_for(analysis, target, owner.scope), name)
                if owner is not routine and binding is owner.scope.bindings.get(str(name).lower()):
                    continue
                write(binding, target)
            elif _kind(item) == "Call_Stmt":
                helper, actuals = _helper_actuals(analysis, owner, item)
                for formal, actual in zip(helper.arguments, actuals, strict=True):
                    parameter = helper.scope.bindings.get(formal)
                    if parameter is None or parameter.intent not in {"in", "out", "inout"}:
                        raise CompilationError("inline numerical helper requires explicit formal intents")
                    if parameter.intent == "in":
                        continue
                    name = actual.items[0] if _kind(actual) == "Part_Ref" else actual
                    binding = analysis._binding(source_scope_for(analysis, actual, owner.scope), name) if _kind(name) in {"Name", "Data_Ref"} else None
                    if owner is not routine and binding is owner.scope.bindings.get(str(name).lower()):
                        continue
                    if (binding is not None and binding.rank and _kind(actual) == "Name"
                            and binding.root.startswith(routine.qualified + "::")
                            and binding.attributes & {"save", "allocatable", "pointer", "target"}):
                        raise CompilationError("inline helper array outputs require private storage in the original owner")
                    # A whole private fixed array is a per-item output, rather
                    # than an external allocation-changing array assignment.
                    if binding is not None and binding.rank and _kind(actual) == "Name" and not binding.root.startswith(routine.qualified + "::"):
                        if binding.intent == "in":
                            raise CompilationError("inline helper writes original INTENT(IN) storage")
                        writes.add(binding.root)
                    else:
                        write(binding, actual)
    for root in scalar_writes:
        binding = used[root]
        if not root.startswith(routine.qualified + "::") or "save" in binding.attributes or binding.rank:
            raise CompilationError("inline scalar effects must remain in the original owner: " + root)
        if binding.attributes & {"target", "pointer", "allocatable", "optional", "volatile", "asynchronous", "value"}:
            raise CompilationError("inline private scalar association is uncertain: " + root)
        if (binding.dtype, binding.kind) not in {("real", 4), ("real", 8), ("integer", 4), ("logical", 1),
                                                 ("logical", 4)}:
            raise CompilationError("unsupported inline private scalar type: " + root)
        if root in iterators and (binding.dtype, binding.kind) != ("integer", 4):
            raise CompilationError("inline loop iterators require default INTEGER: " + root)
    # A scalar the original OpenMP directive of a loop made private has no value when an iteration starts:
    # reading it before the iteration sets it reads nothing a previous iteration left. It is given one.
    undefined = {}
    for loop in loops:
        named = _openmp_private(loop)
        for root in scalar_writes & _upward_reads(analysis, routine, (loop,)):
            if used[root].name.lower() in named:
                undefined.setdefault(id(loop), []).append(used[root])
    if scalar_writes & _upward_reads(analysis, routine, loops) - {binding.root for items in undefined.values()
                                                                   for binding in items}:
        raise CompilationError("inline private scalar reads require a definition inside the region")
    # Following native statements need a read/liveness proof, not numerical
    # eligibility. Original fixed metadata fields can carry index reads here
    # without becoming captures or authorizing their own GPU computation.
    following_analysis = copy.copy(analysis)
    following_analysis._native_metadata = True
    live_scalars = (_upward_reads(following_analysis, routine, following)
                    | _upward_reads(following_analysis, routine, cyclic_following))
    # Nor has such a scalar a value after its loop: what follows does not read the region's.
    private_names = set().union(*(_openmp_private(loop) for loop in loops))
    live_scalars -= {root for root in scalar_writes if used[root].name.lower() in private_names}
    final_iterators = ()
    if scalar_writes & live_scalars:
        only_counted = iterators - assigned_scalars
        final_iterators = _loop_control(analysis, routine, loops, scalar_writes & live_scalars, only_counted,
                                        (scalar_writes - only_counted) | writes | local_arrays)
    # Read-only local fixed arrays can only be used if initialized inside the
    # reached numerical work. The ordinary frontend checks every scalarized
    # element's definition before use; they must not be captured as live arrays.
    written_locals = set(local_arrays)
    local_arrays |= {root for root, binding in used.items()
                     if root.startswith(routine.qualified + "::") and binding.rank
                     and not hasattr(binding, "component_object")
                     and not binding.attributes & {"save", "allocatable", "pointer", "target", "parameter"}}
    for root in local_arrays:
        binding = used[root]
        _fixed_shape(routine, binding)
        if binding.attributes & {"optional", "volatile", "asynchronous", "value"}:
            raise CompilationError("inline private array association is uncertain: " + root)
        if (binding.dtype, binding.kind) not in {("real", 4), ("real", 8), ("integer", 4)}:
            raise CompilationError("unsupported inline private array type: " + root)
        if binding.name.lower() not in private_names and root in (
                _upward_array_reads(following_analysis, routine, following)
                | _upward_array_reads(following_analysis, routine, cyclic_following)):
            raise CompilationError("inline private array is live after the region")
        for loop in loops:
            # An array the loop's OpenMP directive made private has no values when an iteration starts either:
            # each iteration gives it some where it may be read before it is set.
            if binding.name.lower() in _openmp_private(loop) and binding not in undefined.get(id(loop), ()) \
                    and root in _upward_array_reads(analysis, routine, (loop,)):
                undefined.setdefault(id(loop), []).append(binding)
    private = {root: used[root] for root in sorted(scalar_writes | local_arrays)}
    occupied = {binding.name for binding in used.values()}
    occupied.update(name for helper in helpers.values() for name in helper.scope.bindings)
    from compiler.scopes.constant_parameters import outline_constants
    constants = outline_constants(analysis, routine, used.values(), occupied, _fixed_shape)
    captures = {root: binding for root, binding in used.items()
                if root not in private and "parameter" not in binding.attributes}
    captures.update((binding.root, binding) for binding in constants.scalar_captures)
    # A short local array that the region only reads holds the values the procedure gave it before: they come
    # to the entry one scalar per element, and each iteration puts them into its own copy of the array.
    valued = {}
    for root in sorted(local_arrays - written_locals):
        binding = used[root]
        shape = [tuple(int(value) for value in axis.replace("(-2147483647 - 1)", str(-(2**31))).split(":"))
                 for axis in _fixed_shape(routine, binding)]
        count = math.prod(max(0, upper - lower + 1) for lower, upper in shape)
        if binding.name.lower() in private_names or count > 16 or binding.dtype not in {"real", "integer"}:
            continue
        subscripts = [()]
        for lower, upper in reversed(shape):
            subscripts = [(index, *rest) for index in range(lower, upper + 1) for rest in subscripts]
        values = []
        for subscript in subscripts:
            text = ",".join(str(index) for index in subscript)
            element = replace(binding, name=f"{binding.name}({text})", root=f"{root}({text})", rank=0,
                              attributes=frozenset(), lower_bounds=(), lower_bound_nodes=(), shape_nodes=())
            captures[element.root] = element
            values.append(element)
        valued[root] = (binding, values)
    for binding in captures.values():
        if binding.attributes & {"pointer", "optional", "volatile", "asynchronous", "value"}:
            raise CompilationError("inline capture association is uncertain: " + binding.root)
        if _logical_field(binding):
            # Its values come to the entry as the INTEGER words they are stored in, and are only read.
            if binding.root in writes:
                raise CompilationError("a LOGICAL field of a list is only read by an entry: " + binding.root)
        elif (binding.dtype, binding.kind, binding.rank) == ("logical", 4, 0):
            # A default LOGICAL scalar comes to the entry converted to the entry's own kind, so it is only read.
            if binding.root in writes:
                raise CompilationError("a default LOGICAL scalar is only read by an entry: " + binding.root)
        elif (binding.dtype, binding.kind) not in {("real", 4), ("real", 8), ("integer", 4), ("logical", 1)}:
            raise CompilationError("unsupported inline numerical capture type: " + binding.root)
        if "allocatable" in binding.attributes and binding.root.startswith(routine.qualified + "::"):
            _check_allocation(analysis, routine, binding, preceding)
    if not any(binding.rank for binding in captures.values()):
        raise CompilationError("inline numerical candidate has no array resources")
    if len(captures) > 64:
        raise CompilationError("inline numerical capture budget exceeded")
    capture_names = {root: binding.name for root, binding in captures.items()}
    for index, (root, binding) in enumerate(sorted(captures.items())):
        if hasattr(binding, "component_object") or "parameter" in binding.attributes or root.endswith(")"):
            name = ("fort_region_field_" if hasattr(binding, "component_object")
                    else "fort_region_value_" if root.endswith(")") else "fort_region_parameter_") + str(index)
            if name in occupied:
                raise CompilationError("inline field parameter namespace conflicts")
            occupied.add(name)
            capture_names[root] = name
    arrays = {capture_names[root]: binding for root, binding in captures.items() if binding.rank}
    lowers = {}
    for name, binding in arrays.items():
        for axis in range(1, binding.rank + 1):
            lower = f"fort_region_lb_{name}_{axis}"
            if lower in occupied:
                raise CompilationError("inline lower-bound parameter namespace conflicts")
            occupied.add(lower)
            lowers[name, axis] = lower

    helper_names = {name: "fort_helper_" + str(index) for index, name in enumerate(helpers)}
    if any(name in occupied for name in helper_names.values()):
        raise CompilationError("inline helper entry namespace conflicts with original source")
    # A helper may declare its array dummies with constant lower bounds. The
    # copied helper declares them from one and shifts each element reference,
    # which addresses the same element of the actual argument.
    helper_lowers = {}
    for helper in helpers.values():
        for argument in helper.arguments:
            binding = helper.scope.bindings.get(argument)
            if binding is None or not binding.rank or all(node is None for node in binding.lower_bound_nodes):
                continue
            from compiler.ir import SourceLocation
            try:
                helper_lowers[id(binding)] = tuple(
                    1 if node is None else helper.scope.kinds.integer(node, SourceLocation(str(helper.scope.path)))
                    for node in binding.lower_bound_nodes)
            except CompilationError as error:
                raise CompilationError("inline helper array lower bounds must be constant") from error
            for name in walk(helper.execution, F.Name):
                if (analysis._binding(helper.scope, name) is binding
                        and not (_kind(name.parent) == "Part_Ref" and name.parent.items[0] is name)):
                    raise CompilationError("inline helper arrays with lower bounds require element references")
    occupied.update(helper_names.values())
    section_variables, section_guards = [], []

    def normalized_indices(name, indices, owner):
        if len(indices) != arrays[name].rank:
            raise CompilationError("inline numerical array accesses require rank-preserving indices")
        rendered = []
        for axis, index in enumerate(indices, 1):
            lower = lowers[name, axis]

            if _kind(index) != "Subscript_Triplet":
                rendered.append(f"({transformed(index, owner)}) - {lower} + 1")
                continue
            from compiler.ir import SourceLocation
            start, stop, step = index.items
            try:
                stride = 1 if step is None else owner.scope.kinds.integer(step, SourceLocation(str(owner.scope.path)))
            except CompilationError as error:
                raise CompilationError("inline array slices require a constant nonzero stride") from error
            if not -(2**31) <= stride < 2**31 or not stride:
                raise CompilationError("inline array slices require a constant nonzero INTEGER stride")
            original_lower = F.Name(lower)
            original_upper = F.Level_2_Expr(f"({lower} + (SIZE({name},{axis}) - 1))")
            # Fortran uses declared bounds for omitted endpoints regardless of
            # stride. In particular, a(::-1) need not denote a reversed array.
            start = original_lower if start is None else start
            stop = original_upper if stop is None else stop
            # Omitted endpoints above already use normalized descriptor names;
            # transform only original expressions, never synthetic source names.
            first = (str(start) if start is original_lower or start is original_upper else str(transformed(start, owner)))
            last = (str(stop) if stop is original_lower or stop is original_upper else str(transformed(stop, owner)))
            rendered.append(f"({first}) - {lower} + 1 : ({last}) - {lower} + 1 : {stride}")
        return rendered

    def transformed(value, owner=routine):
        if isinstance(value, (tuple, list)):
            return type(value)(transformed(child, owner) for child in value)
        if not isinstance(value, Base):
            return value
        kind = _kind(value)
        if kind == "Assignment_Stmt" and owner is routine:
            target = value.items[0]
            base = target.items[0] if _kind(target) == "Part_Ref" else target
            target_binding = analysis._binding(owner.scope, base)
            bounded = (_kind(target) == "Part_Ref" and any(_kind(index) == "Subscript_Triplet"
                       and any(child is not None for child in index.items) for index in _children(target.items[1])))
            if not bounded and not (target_binding and target_binding.root in private):
                from compiler.scopes.section_operations import expand_full_sections
                expanded = expand_full_sections(analysis, routine, value, section_variables, section_guards)
                if expanded is not None:
                    return transformed(expanded, owner)
        if kind == "Data_Ref":
            access = component_access(analysis, source_scope_for(analysis, value, owner.scope), value)
            binding = access.binding
            if binding.root not in captures:
                raise CompilationError("inline field is not a proved original capture")
            name = capture_names[binding.root]
            if not access.indices:
                return F.Name(name)
            rendered = normalized_indices(name, access.indices, owner)
            if _logical_field(binding):
                return F.Parenthesis("(" + name + "(" + ",".join(rendered) + ") /= 0)")
            return F.Part_Ref(name + "(" + ",".join(rendered) + ")")
        if kind == "Comment":
            return value
        if kind == "Actual_Arg_Spec":
            result = copy.copy(value)
            result.items = (value.items[0], transformed(value.items[1], owner))
            return result
        if kind == "End_Do_Stmt":
            return copy.copy(value)
        if kind == "Nonlabel_Do_Stmt":
            result = copy.copy(value)
            result.items = (value.items[0], transformed(value.items[1], owner))
            return result
        if kind == "Intrinsic_Function_Reference":
            intrinsic = str(value.items[0]).lower()
            if _intrinsic_shadowed(analysis, owner, intrinsic):
                raise CompilationError("inline numerical intrinsic is shadowed: " + intrinsic)
        helper, _ = _helper_actuals(analysis, owner, value)
        if helper is not None:
            result = copy.copy(value)
            result.items = (F.Name(helper_names[helper.qualified]), transformed(value.items[1], owner), *value.items[2:])
            return result
        binding = analysis._binding(owner.scope, value.items[0]) if kind == "Part_Ref" else None
        captured_array = (binding is not None and binding.root in captures and binding.rank)
        private_array = binding is not None and binding.rank and (binding.root in private
                        or binding.root in constants.names or binding is owner.scope.bindings.get(binding.name))
        if kind == "Part_Ref" and not captured_array and not private_array:
            name = str(value.items[0]).lower()
            found = analysis._candidates(source_scope_for(analysis, value, owner.scope), value.items[0]) \
                if name == "ieee_is_nan" and len(_children(value.items[1])) == 1 else ()
            if found and all(item.removeprefix("$intrinsic::").startswith("ieee_arithmetic::") for item in found):
                # IEEE_IS_NAN of IEEE_ARITHMETIC, the intrinsic module or a source module that stands in
                # for it where the compiler has none: the comparison that only a NaN fails.
                operand = str(transformed(_children(value.items[1])[0], owner))
                return F.Expr(f"(({operand}) /= ({operand}))")
            if _intrinsic_shadowed(analysis, owner, name):
                raise CompilationError("inline numerical intrinsic is shadowed: " + name)
        if kind == "Part_Ref" and captured_array:
            name = capture_names[binding.root]
            indices = _children(value.items[1])
            rendered = normalized_indices(name, indices, owner)
            return F.Part_Ref(name + "(" + ",".join(rendered) + ")")
        if kind == "Part_Ref" and binding is not None and id(binding) in helper_lowers:
            indices = _children(value.items[1])
            if len(indices) != binding.rank or any(_kind(index) == "Subscript_Triplet" for index in indices):
                raise CompilationError("inline helper arrays with lower bounds require element references")
            rendered = [f"({transformed(index, owner)}) - ({lower}) + 1"
                        for index, lower in zip(indices, helper_lowers[id(binding)], strict=True)]
            return F.Part_Ref(str(value.items[0]) + "(" + ",".join(rendered) + ")")
        if kind == "Intrinsic_Function_Reference" and str(value.items[0]).lower() in {"lbound", "ubound", "size"}:
            intrinsic = str(value.items[0]).lower()
            if _intrinsic_shadowed(analysis, owner, intrinsic):
                raise CompilationError("inline array inquiry intrinsic is shadowed")
            arguments = _children(value.items[1])
            original_array = (analysis._binding(source_scope_for(analysis, arguments[0], owner.scope), arguments[0])
                              if arguments and _kind(arguments[0]) in {"Name", "Data_Ref"} else None)
            if original_array is not None and _element_count(original_array, arguments[0]):
                # Every element field has the extent of its array of derived type.
                fields = sorted(name for root, name in capture_names.items()
                                if getattr(captures[root], "component_object", None) is not None
                                and captures[root].component_object.root == original_array.root
                                and "element_field" in captures[root].attributes)
                if not fields:
                    raise CompilationError("SIZE of an array of derived type needs one of its fields in the region")
                return F.Intrinsic_Function_Reference(f"SIZE({fields[0]})")
            if original_array is None or original_array.root not in captures or not original_array.rank:
                raise CompilationError("inline array inquiry requires an original array descriptor")
            name = capture_names[original_array.root]
            if intrinsic == "size":
                # Full-layout shape is unchanged by coordinate rebasing.
                result = copy.copy(value)
                actuals = copy.copy(value.items[1])
                actuals.items = (F.Name(name), *transformed(arguments[1:], owner))
                result.items = (value.items[0], actuals)
                return result
            else:
                if len(arguments) != 2 or _kind(arguments[1]) != "Int_Literal_Constant":
                    raise CompilationError("inline bounds inquiries require a constant scalar DIM")
                axis = int(str(arguments[1]).split("_")[0])
                if (name, axis) not in lowers:
                    raise CompilationError("inline bounds inquiry DIM is outside the original rank")
                if intrinsic == "lbound":
                    return F.Name(lowers[name, axis])
                return F.Level_2_Expr(f"({lowers[name, axis]} + (SIZE({name},{axis}) - 1))")
        if kind == "Name":
            statement = _part(owner.scope.node, "Function_Stmt")
            if (statement is not None and statement.items[3] is None
                    and str(value).lower() == str(statement.items[1]).lower()):
                return F.Name(helper_names[owner.qualified])
            binding = analysis._binding(owner.scope, value)
            if binding and "parameter" in binding.attributes:
                if binding.root in constants.names:
                    return F.Name(constants.names[binding.root])
                if binding.root in captures:
                    return F.Name(capture_names[binding.root])
                if (owner is not routine and binding is owner.scope.bindings.get(binding.name)) or binding.dtype == "real":
                    return copy.copy(value)
                # Resolve kinds/constants in the original lexical scope; a new
                # module cannot silently acquire a same-spelling imported value.
                if binding.dtype != "integer" or binding.kind != 4:
                    raise CompilationError("inline numerical constants currently require default INTEGER")
                from compiler.ir import SourceLocation
                integer = owner.scope.kinds.integer(value, SourceLocation(str(owner.scope.path)))
                if not -(2**31) <= integer < 2**31:
                    raise CompilationError("inline constant exceeds the existing INTEGER ABI")
                spelling = "(-2147483647 - 1)" if integer == -(2**31) else str(integer)
                return F.Level_2_Expr(spelling)
        if kind in {"Real_Literal_Constant", "Int_Literal_Constant"} and value.items[1] is not None:
            from compiler.ir import SourceLocation
            selector = owner.scope.kinds.integer(value.items[1], SourceLocation(str(owner.scope.path)))
            result = copy.copy(value)
            result.items = (value.items[0], str(selector))
            return result
        result = copy.copy(value)
        for attribute in ("content", "items"):
            if hasattr(value, attribute):
                setattr(result, attribute, transformed(getattr(value, attribute), owner))
        return result

    span = (statement_span(nodes[0])[0], statement_span(nodes[-1])[1])
    helper_identity = "\0".join(qualified + "\0" + analysis.sources[str(helper.scope.path)]
                                  for qualified, helper in sorted(helpers.items()))
    identity = sha256((analysis.sources[str(routine.scope.path)] + "\0" + routine.qualified + "\0" +
                       str(span) + "\0" + "\n".join(map(str, nodes)) + "\0" + helper_identity
                       + ("\0" + constants.identity if constants.identity else "")).encode()).hexdigest()
    module, procedure = "fort_inline_" + identity[:12], "region"
    parameters, declarations = [], list(constants.declarations)

    def numerical_type(binding):
        # The numerical ABI uses default INTEGER and its C-bool LOGICAL model.
        # Their original kinds were checked above. Real temporaries must retain
        # their precision as well as the array/scalar captures.
        if _logical_field(binding):
            return "integer"
        return binding.dtype if binding.dtype in {"integer", "logical"} else f"{binding.dtype}({binding.kind})"

    for binding in sorted(captures.values(), key=lambda item: item.root):
        name = capture_names[binding.root]
        parameters.append(Parameter(name, binding.root, binding.rank))
        dtype = numerical_type(binding)
        shape = "(" + ",".join(":" for _ in range(binding.rank)) + ")" if binding.rank else ""
        intent = "inout" if binding.root in writes else "in"
        declarations.append(f"{dtype}, intent({intent}) :: {name}{shape}")
        if binding.rank:
            for axis in range(1, binding.rank + 1):
                lower = lowers[name, axis]
                parameters.append(Parameter(lower, binding.root, 0, axis))
                declarations.append(f"integer, intent(in) :: {lower}")
    declarations += [f"{numerical_type(binding)} :: {binding.name}" +
                     ("(" + ",".join(_fixed_shape(routine, binding)) + ")" if binding.rank else "")
                     for binding in sorted(private.values(), key=lambda item: item.name)]
    normalized = [_untile(analysis, routine, loop, helpers) for loop in loops]
    runtime_guards = tuple(dict.fromkeys(guard for _, guards, _ in normalized for guard in guards))
    body = []
    for (loop, _, _), original in zip(normalized, loops, strict=True):
        text = str(transformed(loop))
        given = [f"{element.name} = {capture_names[element.root]}" for _root, (array, values) in sorted(valued.items())
                 for element in values if re.search(rf"(?i)(?<![\w%]){re.escape(array.name)}\s*\(", text)]
        if id(original) in undefined or given:
            lines = text.split("\n")
            opening = next(index for index, line in enumerate(lines) if re.match(r"(?i)\s*do\b", line))
            zero = {binding.root: ".false." if binding.dtype == "logical" else "0"
                    for binding in undefined.get(id(original), ())}
            lines[opening + 1:opening + 1] = [
                f"{name} = {zero[binding.root]}" for binding in sorted(undefined.get(id(original), ()),
                                                                     key=lambda item: item.name)
                for name in _element_names(routine, binding)] + given
            text = "\n".join(lines)
        body.append(text)
    declarations += ["integer :: " + name for name in section_variables]
    runtime_guards = tuple(dict.fromkeys((*runtime_guards, *section_guards)))
    helper_sources = []
    for qualified, helper in helpers.items():
        statement = _part(helper.scope.node, "Subroutine_Stmt") or _part(helper.scope.node, "Function_Stmt")
        function = _kind(statement) == "Function_Stmt"
        header = copy.copy(statement)
        prefix = transformed(statement.items[0], helper)
        if not _declared_pure(helper):
            # The analysis found it pure; the entry's own copy says so.
            prefix = F.Prefix(("pure " + str(prefix)) if prefix is not None else "pure")
        header.items = (prefix, F.Name(helper_names[qualified]), *statement.items[2:])
        local_declarations = []
        for declaration in _children(_part(helper.scope.node, "Specification_Part")):
            if _kind(declaration) in {"Implicit_Part", "Comment", "Use_Stmt"}:
                # Original imported constants are resolved below. An unused
                # intrinsic IEEE import does not move its caller's guards.
                continue
            if _kind(declaration) != "Type_Declaration_Stmt":
                raise CompilationError("unsupported inline numerical helper specification: " + _kind(declaration))
            dtype, attributes, entities = declaration.items
            retained = [entity for entity in _children(entities)
                        if helper.scope.bindings[str(entity.items[0]).lower()].root not in constants.names]
            if not retained:
                continue
            entities = copy.copy(entities)
            entities.items = tuple(retained)
            own = [helper.scope.bindings[str(entity.items[0]).lower()] for entity in _children(entities)]
            if any(id(binding) in helper_lowers for binding in own):
                # A DIMENSION attribute gives its shape to each entity, which then says it itself.
                shared = next((item for item in _children(attributes) if _kind(item) == "Dimension_Attr_Spec"), None) \
                    if attributes is not None else None
                rebased = []
                for entity, binding in zip(retained, own, strict=True):
                    if id(binding) in helper_lowers:
                        entity = copy.copy(entity)
                        entity.items = (entity.items[0], F.Array_Spec(",".join(":" * binding.rank)), *entity.items[2:])
                    elif entity.items[1] is None and shared is not None:
                        entity = copy.copy(entity)
                        entity.items = (entity.items[0], shared.items[1], *entity.items[2:])
                    elif entity.items[1] is None and binding.rank:
                        raise CompilationError("inline helper array declaration is not understood")
                    rebased.append(entity)
                entities.items = tuple(rebased)
                if shared is not None:
                    kept = [item for item in _children(attributes) if item is not shared]
                    attributes = F.Attr_Spec_List(", ".join(str(item) for item in kept)) if kept else None
            if any(binding.attributes & {"save", "allocatable", "pointer", "optional", "target", "volatile", "asynchronous"} for binding in own):
                raise CompilationError("inline numerical helper storage association is unsupported")
            # A helper's own variables are not passed to the entry: a LOGICAL of any kind is a flag in it.
            if any((binding.dtype, binding.kind) not in {("real", 4), ("real", 8), ("integer", 4), ("logical", 1),
                                                         ("logical", 4)} for binding in own):
                raise CompilationError("unsupported inline numerical helper declaration type")
            if any(binding.signature()[:2] != own[0].signature()[:2] for binding in own):
                raise CompilationError("inline numerical helper declaration has inconsistent type")
            updated = copy.copy(declaration)
            updated.items = (F.Intrinsic_Type_Spec(numerical_type(own[0])), transformed(attributes, helper), transformed(entities, helper))
            local_declarations.append(str(updated))
        ending = "end function" if function else "end subroutine"
        helper_sources += [str(header), *local_declarations, str(transformed(helper.execution, helper)), ending]
    source = "\n".join(fortran_lines([f"module {module}", "implicit none", "contains",
                                   f"subroutine {procedure}(" + ",".join(item.name for item in parameters) + ")",
                                   *declarations, *"\n".join(body).splitlines(),
                                   *(["contains", *"\n".join(helper_sources).splitlines()] if helper_sources else []),
                                   "end subroutine", "end module"])) + "\n"
    return RegionExtraction(span, source, module + "::" + procedure, tuple(parameters),
                            tuple(sorted(captures.values(), key=lambda item: item.root)), frozenset(writes),
                            identity, runtime_guards=runtime_guards, final_iterators=final_iterators)
