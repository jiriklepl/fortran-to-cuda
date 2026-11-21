"""Numerical entries for a caller-owned residency domain.

This producer deliberately has no placement or lifetime policy. It exports
source-backed numerical regions through the same extraction used by checked
source scopes. The owner retains scalar control flow, native effects and
allocation epochs. Every exported entry still goes through ordinary numerical
lowering and dependence analysis before it can execute on a device.
"""
import re
from fnmatch import fnmatchcase
from hashlib import sha256
from pathlib import Path

from fparser.two.Fortran2003 import (
    Allocate_Stmt,
    Allocation,
    Assignment_Stmt,
    Call_Stmt,
    Data_Ref,
    Deallocate_Stmt,
    Intrinsic_Function_Reference,
    Name,
    Part_Ref,
    Read_Stmt,
)
from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_original_call
from compiler.frontend.component_bindings import component_access
from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError
from compiler.scopes.region_context import original_region_context
from compiler.scopes.regions import extract_region
from compiler.scopes.segments import directive, grouped_nodes, statement_span

_INQUIRIES = {"lbound", "ubound", "size", "shape", "allocated", "associated", "present", "kind", "rank",
              "storage_size", "is_contiguous"}
_CONSTRUCT_HEADERS = {"If_Then_Stmt", "Else_If_Stmt", "Associate_Stmt"}


def _residency_directive(node):
    return str(node).lstrip().lower().startswith("!$fort")


def _entry_groups(sequence, acknowledged=None):
    """Share an entry across adjacent serial nests in one control region.

    Calls, assignments, branch boundaries, OpenMP constructs and residency
    directives end the group; a nest with a directive inside it stays native.
    This only changes the numerical extraction boundary: ordinary dependence
    analysis still schedules every source pass and proves any later fusion.
    ``acknowledged`` collects the identities of the groups that directly follow
    a residency directive.
    """
    pending, follows_directive = [], False

    def mark(group):
        nonlocal follows_directive
        if follows_directive and acknowledged is not None:
            acknowledged.add(id(group[0] if isinstance(group, tuple) else group))
        follows_directive = False
        return group

    for group in grouped_nodes(sequence):
        if not isinstance(group, tuple) and _kind(group) == "Comment" and directive(group) is None:
            if _residency_directive(group):
                if pending:
                    yield mark(tuple(pending))
                    pending = []
                follows_directive = True
            continue
        if (not isinstance(group, tuple) and _kind(group) == "Block_Nonlabel_Do_Construct"
                and not any(directive(node) is not None or _residency_directive(node)
                            for node in walk(group) if _kind(node) == "Comment")):
            pending.append(group)
            continue
        if pending:
            yield mark(tuple(pending))
            pending = []
        yield mark(group)
    if pending:
        yield mark(tuple(pending))


def _holds_a_sequence(loop):
    """Whether a loop's body is more than one nest: a loop nest beside other statements or nests."""
    body = [item for item in _children(loop) if _kind(item) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"}]
    return len(body) > 1 and any(_kind(item) == "Block_Nonlabel_Do_Construct" for item in body)


def _text(node):
    return " ".join(str(node).split())


# The inquiry that says whether an array with the attribute exists.
_EXISTS = {"allocatable": "allocated", "pointer": "associated", "optional": "present"}
_LEAVING = {"Return_Stmt", "Goto_Stmt", "Computed_Goto_Stmt", "Stop_Stmt", "Error_Stop_Stmt", "Exit_Stmt",
            "Cycle_Stmt"}


def _runs(loop):
    """A condition under which a counted loop with no stated step runs at all, or None."""
    if _kind(loop) != "Block_Nonlabel_Do_Construct":
        return None
    control = next(item for item in loop.content if _kind(item) == "Nonlabel_Do_Stmt").items[1]
    counted = control.items[1] if control is not None and _kind(control) == "Loop_Control" else None
    if not counted or len(counted) != 2 or len(counted[1]) != 2:
        return None
    return f"({_text(counted[1][1])}) >= ({_text(counted[1][0])})"


def _statement_form(nodes):
    """What generated host access around host statements has to know about their shape.

    ``kind`` is the kind of a single statement or ``group``. A one-line IF
    comes with its ``condition``, the names in it and its ``action``, so that
    the access can go inside the condition. ``leaves`` says that control can
    leave from inside (RETURN, GOTO, EXIT ...), past whatever follows.
    """
    form = {"kind": _kind(nodes[0]) if len(nodes) == 1 else "group",
            "leaves": any(_kind(item) in _LEAVING for node in nodes for item in walk(node))}
    if form["kind"] == "If_Stmt":
        condition, action = nodes[0].items[0], nodes[0].items[1]
        form.update(condition=_text(condition), action=_text(action),
                    condition_names=sorted({str(name).lower() for name in walk(condition, Name)}))
    return form


def _structure_fields(analysis, scope, node):
    """The element fields of arrays of derived type that ``node`` reads or writes, by resource."""
    fields = {}
    for reference in walk(node, Data_Ref):
        try:
            access = component_access(analysis, scope, reference)
        except CompilationError:
            continue
        if access is not None and "element_field" in access.binding.attributes:
            fields[access.binding.root] = access
    return fields


def _structure_objects(analysis, scope, node):
    return frozenset(access.root_binding.root for access in _structure_fields(analysis, scope, node).values())


def _indirect_writes(analysis, routine, loops):
    """Arrays written through index values, and what a caller must check before running the nests.

    A nest such as ``a(idx(i)) = a(idx(i)) + v(i)`` is independent across
    iterations when its subscript tuple addresses a different element each
    time. That depends on the index values, so it is left to a check the
    caller makes: each requirement names the array, the DO statements around
    one subscript tuple and its subscripts, all in original source text. The
    caller checks all requirements of an array together: no two tuples, in
    any iterations of any of the nests, may address the same element. Then the
    nests are also independent of each other. An array gets requirements only
    when every nest that writes it is checkable: it is written through each of
    its subscript tuples, all of which go through index values, with no scalar
    the nest itself computes.
    """
    found, refused = {}, set()
    for loop in loops:
        computed = {str(statement.items[0]).lower() for statement in walk(loop, Assignment_Stmt)
                    if _kind(statement.items[0]) == "Name"}
        accesses = {}
        for reference in walk(loop, Part_Ref):
            if _kind(reference.parent) == "Data_Ref" and reference.parent.items[0] is reference:
                continue  # objects(i)%field: the element, not an access to a numeric array
            binding = analysis._binding(routine.scope, reference.items[0])
            if binding is None or not binding.rank or str(binding.dtype).lower().startswith("type"):
                continue
            item = accesses.setdefault(binding.root, {"name": binding.name, "tuples": {}, "written": set()})
            item["tuples"].setdefault(_text(reference.items[1]).lower(), reference)
        for statement in walk(loop, Assignment_Stmt):
            target = statement.items[0]
            binding = analysis._binding(routine.scope, target.items[0]) if _kind(target) == "Part_Ref" else None
            if binding is not None and binding.root in accesses:
                accesses[binding.root]["written"].add(_text(target.items[1]).lower())
        for root, item in accesses.items():
            if not item["written"]:
                continue
            uses_computed, by_tuple = False, []
            for reference in item["tuples"].values():
                subscripts, indexed = _children(reference.items[1]), {}
                uses_computed |= any(str(name).lower() in computed for part in subscripts for name in walk(part, Name))
                for part in subscripts:
                    indexed.update((root_, access.binding.name)
                                   for root_, access in _structure_fields(analysis, routine.scope, part).items())
                    for inner in walk(part, Part_Ref):
                        index = analysis._binding(routine.scope, inner.items[0])
                        if (index is not None and index.rank and not str(index.dtype).lower().startswith("type")
                                and not (_kind(inner.parent) == "Data_Ref" and inner.parent.items[0] is inner)):
                            indexed[index.root] = index.name
                by_tuple.append((reference, indexed))
            if not any(indexed for _, indexed in by_tuple) and not uses_computed:
                continue  # Subscripts of loop variables only: the ordinary dependence proof decides.
            if uses_computed or not all(indexed for _, indexed in by_tuple) or set(item["tuples"]) != item["written"]:
                # A tuple that is only read names elements the check says nothing about.
                refused.add(root)
                continue
            for reference, indexed in by_tuple:
                enclosing, node = [], reference
                while node is not None:
                    if _kind(node) == "Block_Nonlabel_Do_Construct":
                        enclosing.append(_text(next(part for part in node.content
                                                    if _kind(part) == "Nonlabel_Do_Stmt")))
                    if node is loop:
                        break
                    node = getattr(node, "parent", None)
                requirement = {"resource": root, "array": item["name"], "loops": enclosing[::-1],
                               "indices": [_text(part) for part in _children(reference.items[1])],
                               "index_resources": sorted(indexed)}
                if requirement not in found.setdefault(root, []):
                    found[root].append(requirement)
    return [requirement for root, requirements in sorted(found.items()) if root not in refused
            for requirement in requirements]


def _other_names(node):
    """The ids of the names in ``node`` that are components or argument keywords, not variables of the scope."""
    found = set()
    for item in walk(node):
        kind = _kind(item)
        if kind == "Data_Ref":
            found.update(id(part.items[0] if _kind(part) == "Part_Ref" else part) for part in item.items[1:])
        elif kind == "Procedure_Designator":
            found.add(id(item.items[2]))
        elif kind in {"Actual_Arg_Spec", "Component_Spec"}:
            found.add(id(item.items[0]))
    return found


def _array_uses(analysis, routine, nodes):
    """The arrays ``nodes`` name, the procedures they call, and whether an array is used outside a call.

    A whole array that is an actual argument is the called procedure's
    business. A section or an expression of it is evaluated here, so it counts
    as a use, like any other mention. Asking for an array's bounds or
    allocation, or allocating it, reads none of its values. Also returns the
    arrays the statements may change (an assignment target, input, an argument
    of a procedure that does not take it INTENT(IN)) and those they deallocate.
    """
    arrays, calls, passed, elsewhere = {}, set(), set(), False
    written, released, others = set(), set(), set()

    def named(node):
        found = (analysis._binding(routine.scope, name) for name in walk(node, Name) if id(name) not in others)
        return {binding.root for binding in found if binding is not None and binding.rank}

    def may_write(procedure, arguments):
        """The arrays among the actual arguments that the called procedure may change."""
        callee = analysis.routines.get(procedure) or analysis.numerical_helpers.get(procedure)
        for position, argument in enumerate(_children(arguments) if arguments is not None else ()):
            keyword = str(argument.items[0]).lower() if _kind(argument) == "Actual_Arg_Spec" else None
            value = argument.items[1] if keyword is not None else argument
            formal = None
            if callee is not None:
                formal = keyword if keyword is not None else (
                    callee.arguments[position] if position < len(callee.arguments) else None)
            binding = callee.scope.bindings.get(formal) if formal is not None else None
            if binding is None or binding.intent != "in":
                written.update(named(value))

    def whole_arguments(arguments):
        for argument in (_children(arguments) if arguments is not None else ()):
            value = argument.items[1] if _kind(argument) == "Actual_Arg_Spec" else argument
            if _kind(value) == "Name":
                passed.add(id(value))

    for node in nodes:
        # A component or a keyword may be spelled like an array of the scope; it is not that array.
        others = _other_names(node)
        for call in walk(node, Call_Stmt):
            whole_arguments(call.items[1])
            try:
                procedure = resolve_original_call(analysis, routine.scope, call).procedure
                calls.add(procedure)
            except CompilationError:
                procedure = None
                calls.update(analysis._candidates(routine.scope, call.items[0]) or ("$unresolved",))
            may_write(procedure, call.items[1])
        for reference in walk(node, Part_Ref):
            candidates = analysis._candidates(routine.scope, reference.items[0]) or ()
            if len(candidates) == 1 and candidates[0] in analysis.functions \
                    and analysis._binding(routine.scope, reference.items[0]) is None:
                calls.add(candidates[0])
                whole_arguments(reference.items[1])
                may_write(candidates[0], reference.items[1])
        for statement in walk(node, Assignment_Stmt):
            target = statement.items[0]
            while _kind(target) in {"Part_Ref", "Data_Ref"}:
                target = target.items[0]
            written.update(named(target))
        for statement in walk(node, Read_Stmt):
            written.update(named(statement))
        for statement in walk(node, Deallocate_Stmt):
            released.update(named(statement))
        inquired = {id(reference.items[1].items[0]) for reference in walk(node, Intrinsic_Function_Reference)
                    if str(reference.items[0]).lower() in _INQUIRIES and reference.items[1] is not None
                    and reference.items[1].items}
        inquired |= {id(allocation.items[0]) for allocation in walk(node, Allocation)}
        for name in walk(node, Name):
            binding = analysis._binding(routine.scope, name)
            if binding is not None and binding.rank and id(name) not in inquired and id(name) not in others:
                arrays[binding.root] = binding
                elsewhere |= id(name) not in passed
    return arrays, calls, elsewhere, written & set(arrays), released & set(arrays)


def _temporaries(analysis, routine, procedure):
    """The temporary arrays of a procedure: locals it allocates and deallocates once, in its own statements.

    Between the two statements such an array belongs to one call, and its
    values are dead before and after. That is all the domain needs to know to
    lend storage that stays from call to call (see ``resident_splice.lend_storage``),
    which takes a pointer in place of the allocatable. The exchange is exact
    only where the array is used like any array: by its elements and sections,
    by its bounds, or passed whole to a procedure that takes an ordinary array.
    Listed are REAL arrays of rank three, the shape the domain lends. A
    procedure with a RETURN, with internal procedures, or with a temporary used
    in any other way gets none.
    """
    node = routine.scope.node
    execution = routine.execution
    if any(_kind(item) == "Internal_Subprogram_Part" for item in _children(node)) \
            or any(_kind(item) in {"Return_Stmt", "Goto_Stmt", "Computed_Goto_Stmt"} for item in walk(execution)):
        return []
    candidates = {name: binding for name, binding in routine.scope.bindings.items()
                  if binding.rank == 3 and binding.dtype == "real" and binding.kind in {4, 8} and binding.intent is None
                  and "allocatable" in binding.attributes
                  and not binding.attributes & {"save", "pointer", "target", "optional", "parameter"}
                  and name not in routine.arguments}
    if not candidates:
        return []
    statements = {"Allocate_Stmt": {}, "Deallocate_Stmt": {}}
    top = {id(item) for item in _children(execution)}
    allowed, refused = set(), set()
    for statement in walk(execution, (Allocate_Stmt, Deallocate_Stmt)):
        kind = _kind(statement)
        if kind == "Allocate_Stmt":
            objects = [item for item in walk(statement, Allocation)]
            named = [str(item.items[0]).lower() for item in objects]
            plain = statement.items[0] is None and statement.items[2] is None and all(
                item.items[1] is not None and len(_children(item.items[1])) == 3 for item in objects)
        else:
            listed = statement.items[0]
            parts = [listed] if _kind(listed) == "Name" else list(_children(listed))
            named = [str(item).lower() for item in parts]
            plain = statement.items[1] is None and all(_kind(item) == "Name" for item in parts)
        mine = [name for name in named if name in candidates]
        if not mine:
            continue
        # One statement allocates or deallocates temporaries only, at the top of the procedure's statements.
        if not plain or len(mine) != len(named) or id(statement) not in top or any(
                name in statements[kind] for name in mine):
            refused.update(mine)
            continue
        for name in mine:
            statements[kind][name] = statement
        allowed.update(id(item) for item in walk(statement, Name))
    for reference in walk(execution, Part_Ref):
        allowed.add(id(reference.items[0]))
    for reference in walk(execution, Intrinsic_Function_Reference):
        if str(reference.items[0]).lower() in {"lbound", "ubound", "size", "shape"} and reference.items[1] is not None:
            allowed.update(id(item) for item in _children(reference.items[1])[:1])
    for call in walk(execution, Call_Stmt):
        try:
            callee = analysis.routines.get(resolve_original_call(analysis, routine.scope, call).procedure)
        except CompilationError:
            callee = None
        for position, argument in enumerate(_children(call.items[1]) if call.items[1] is not None else ()):
            formal = callee.arguments[position] if callee is not None and position < len(callee.arguments) else None
            binding = callee.scope.bindings.get(formal) if formal is not None else None
            if _kind(argument) == "Name" and binding is not None and binding.rank \
                    and not binding.attributes & {"allocatable", "pointer"}:
                allowed.add(id(argument))
    for name in walk(execution, Name):
        if str(name).lower() in candidates and id(name) not in allowed:
            refused.add(str(name).lower())
    found = []
    for name, binding in sorted(candidates.items()):
        allocate, deallocate = statements["Allocate_Stmt"].get(name), statements["Deallocate_Stmt"].get(name)
        if name in refused or allocate is None or deallocate is None \
                or statement_span(allocate)[1] >= statement_span(deallocate)[0]:
            continue
        found.append({"procedure": procedure, "source": str(routine.scope.path), "name": binding.name,
                      "resource": binding.root, "kind": binding.kind,
                      "allocate": list(statement_span(allocate)), "deallocate": list(statement_span(deallocate))})
    # Arrays that share a statement go together or not at all.
    kept = {item["name"].lower() for item in found}
    for kind in statements:
        for name, statement in statements[kind].items():
            partners = {other for other, node in statements[kind].items() if node is statement}
            if name in kept and not partners <= kept:
                kept -= partners
    return [item for item in found if item["name"].lower() in kept]


def _forwarders(analysis, selected, host_statements):
    """Procedures outside the selection that host statements call, and whether each only passes arrays on.

    A wrapper that hands its arrays to other procedures and does nothing else
    with them is as safe as what it calls. Each record has ``only_calls`` and
    the procedures it ``calls``; the audit follows them.
    """
    found, pending = {}, {call for statement in host_statements for call in statement["calls"]}
    while pending:
        name = pending.pop()
        routine = analysis.routines.get(name) or analysis.numerical_helpers.get(name)
        if name in found or name in selected or routine is None or routine.execution is None:
            continue
        _arrays, calls, elsewhere, _written, _released = _array_uses(analysis, routine, tuple(_children(routine.execution)))
        found[name] = {"only_calls": not elsewhere, "calls": sorted(calls)}
        pending.update(calls)
    return found


def _resolve_procedures(analysis, procedures, *, allow_patterns=False):
    available = sorted(analysis.routines)
    if procedures is None:
        return available
    selected = []
    seen = set()
    for spec in procedures:
        pattern = spec.lower()
        is_pattern = allow_patterns or any(ch in pattern for ch in "*?[")
        if is_pattern:
            matches = [name for name in available if fnmatchcase(name, pattern)]
            if not matches:
                raise CompilationError("Resident procedure pattern matched no routines: " + spec)
            for name in matches:
                if name not in seen:
                    seen.add(name)
                    selected.append(name)
        else:
            if pattern not in analysis.routines:
                raise CompilationError("Resident procedure is unavailable: " + spec)
            if pattern not in seen:
                seen.add(pattern)
                selected.append(pattern)
    return selected


def export_resident_entries(paths, procedures, directory, *, analysis_sources=None, allow_patterns=False):
    """Export resident numerical entries for explicit procedures or glob patterns.

    Besides ``entries`` and ``rejected``, the result lists ``host_statements``:
    the statements of the selected procedures that stay host code and name
    arrays, each with its span, the arrays and whether a residency directive
    directly precedes it or sits inside it (``acknowledged``). They run beside
    the entries, so each one that touches an array an entry binds needs the
    host to hold current values.

    A loop whose body holds a loop nest beside other statements may be a nest
    itself, or a loop the host has to run around nests (an iteration, a time
    loop). When it cannot be exported, its body is exported in its place. When
    it can, its entry is marked ``descends`` and its body is exported too:
    the entries, rejections and host statements of the body carry
    ``alternative_of``, the spans of the loops they replace, outermost first,
    and count only where none of those loops compiled as an entry (see
    ``compile_resident_entries``).
    """
    # A selected procedure can be long, the more so with calls replaced by their statements; the
    # default bound on statements per procedure is sized for proofs this export does not ask for.
    # A main program is analysed as a procedure only when it is asked for by name.
    programs = tuple(sorted({name.lower() for name in procedures or ()
                             if re.fullmatch(r"\$program::[a-z][a-z0-9_]*", name.lower())}))
    analysis = SourceEffects(paths, analysis_sources=analysis_sources, operations=4096, program_roots=programs)
    analysis.element_fields = True
    # The body of a module function holds loop nests like any subroutine's.
    for name, routine in analysis.numerical_helpers.items():
        if routine.source_kind == "function" and name not in analysis.routines:
            analysis.routines[name] = routine
            analysis._source_roles[name] = (routine.execution, analysis._routine_signature(routine),
                                            str(routine.execution))
    # So does the body of a main program that was asked for. Its variables live as long as the run, like a
    # module's; the analysis keeps a program frame for call ownership only until it is selected here.
    for name in programs:
        routine = analysis.routines.get(name)
        if routine is not None:
            routine.issues[:] = [issue for issue in routine.issues if not issue.startswith("original program frame")]
            # A scalar of the frame is working storage of its nests like a procedure's local: nothing
            # returns to find it changed, and a later read in the frame is seen by the liveness check.
            for binding in routine.scope.bindings.values():
                if not binding.rank:
                    binding.attributes = binding.attributes - {"save"}
            analysis._source_roles[name] = (routine.execution, analysis._routine_signature(routine),
                                            str(routine.execution))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    resolved_procedures = _resolve_procedures(analysis, procedures, allow_patterns=allow_patterns)
    entries, rejected, grouping_rejections, host_statements, temporaries = [], [], [], [], []
    for procedure in resolved_procedures:
        routine = analysis.routines[procedure]

        def emit(nodes, sequence, outer=()):
            if len(nodes) > 1:
                # Nests over different arrays of derived type bind, and are skipped when empty, separately.
                objects = [_structure_objects(analysis, routine.scope, node) for node in nodes]
                if len(set(objects)) > 1:
                    start = 0
                    for index in range(1, len(nodes) + 1):
                        if index == len(nodes) or objects[index] != objects[start]:
                            emit(nodes[start:index], sequence, outer)
                            start = index
                    return
            first = min(statement_span(node)[0] for node in nodes)
            last = max(statement_span(node)[1] for node in nodes)
            # Grouping can peel directive prefixes. Recover the actual
            # source nodes instead of supplying manufactured syntax.
            selected = tuple(node for node in sequence
                             if first <= statement_span(node)[0]
                             and statement_span(node)[1] <= last)
            loops = [node for node in nodes if _kind(node) == "Block_Nonlabel_Do_Construct"]
            try:
                context = original_region_context(routine, selected, analysis=analysis)
                extracted = extract_region(analysis, routine, selected,
                                           preceding=context.preceding, following=context.following,
                                           cyclic_following=context.cyclic_following)
            except CompilationError as error:
                # A larger interface can expose an unsupported association
                # even when its individual source nests remain exportable.
                if len(nodes) > 1 and len(loops) == len(nodes):
                    grouping_rejections.append({"procedure": procedure, "first_line": first,
                                                "last_line": last, "reason": str(error)})
                    for node in nodes:
                        emit((node,), sequence, outer)
                    return
                if len(nodes) == 1 and _holds_a_sequence(nodes[0]):
                    # Not a nest but a loop around nests, such as an iteration or a time loop: the
                    # host runs it, and what stands in its body is looked at in its own right.
                    grouping_rejections.append({"procedure": procedure, "first_line": first,
                                                "last_line": last, "reason": str(error)})
                    within(nodes[0], outer)
                    return
                rejected.append({"procedure": procedure, "first_line": first, "last_line": last,
                                 "reason": str(error), **alternative(outer), **host_code(nodes)})
                return
            path = extracted.write(directory)
            structures = []
            for root in sorted(_structure_objects(analysis, routine.scope, nodes)):
                name = root.rsplit("::", 1)[-1]
                # When every nest runs once per element, an empty array leaves the entry nothing to do.
                counted = all(_text(next(part for part in node.content if _kind(part) == "Nonlabel_Do_Stmt")).lower()
                              .replace(" ", "").endswith(f",size({name})") for node in loops)
                structures.append({"name": name, "resource": root, "counted": bool(loops) and counted})
            entries.append({"procedure": procedure, "source": str(routine.scope.path),
                            "input_path": str(path.resolve()), **extracted.public(),
                            "source_loop_spans": [list(statement_span(node)) for node in loops],
                            "structures": structures, "injective": _indirect_writes(analysis, routine, loops),
                            # For the case that the entry does not compile and the nests stay host code.
                            "host_code": host_code(nodes),
                            **alternative(outer)})
            if len(nodes) == 1 and _holds_a_sequence(nodes[0]):
                # Whether the loop runs its iterations independently is decided when the entry is
                # compiled. In case it does not, its body is exported as well, to be used then.
                entries[-1]["descends"] = True
                within(nodes[0], (*outer, (first, last)))

        def alternative(outer):
            """Marks what is used only where the loops of the given spans did not compile as entries."""
            return {"alternative_of": [list(span) for span in outer]} if outer else {}

        def within(loop, outer):
            statement = next(item for item in loop.content if _kind(item) == "Nonlabel_Do_Stmt")
            host((statement,), False, outer)
            scan(tuple(item for item in _children(loop) if _kind(item) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt"}),
                 outer)

        def touched(arrays, written, released):
            return [{"name": binding.name, "resource": root, "type": binding.dtype, "kind": binding.kind,
                     "rank": binding.rank, "field": hasattr(binding, "component_object"),
                     # What must hold for the array to exist when the statements start.
                     "exists": sorted(_EXISTS[name] for name in binding.attributes if name in _EXISTS),
                     "access": "release" if root in released else "write" if root in written else "read"}
                    for root, binding in sorted(arrays.items())]

        def host_code(nodes):
            """How a nest that stays on the host touches arrays, for the access it then needs; {} if unknown."""
            try:
                arrays, _calls, _elsewhere, written, released = _array_uses(analysis, routine, nodes)
            except CompilationError:
                return {}
            return {"arrays": touched(arrays, written, released), **_statement_form(nodes),
                    **({"runs": _runs(nodes[0])} if len(nodes) == 1 and _runs(nodes[0]) else {})}

        def host(nodes, acknowledged, outer=()):
            """Record host code that names arrays: it runs beside entries that may hold newer values."""
            arrays, calls, elsewhere, written, released = _array_uses(analysis, routine, nodes)
            if arrays:
                first = min(statement_span(node)[0] for node in nodes)
                last = max(statement_span(node)[1] for node in nodes)
                text = " ".join(" ".join(str(node).split()) for node in nodes)
                host_statements.append({"procedure": procedure, "source": str(routine.scope.path),
                                        "first_line": first, "last_line": last,
                                        "text": text if len(text) <= 240 else text[:237] + "...",
                                        # Stable while the statement and its procedure keep their text.
                                        "id": sha256((procedure + "\n" + text).encode()).hexdigest()[:16],
                                        "arrays": touched(arrays, written, released),
                                        **_statement_form(nodes),
                                        "calls": sorted(calls), "acknowledged": acknowledged,
                                        "only_calls": not elsewhere, **alternative(outer)})

        def scan(sequence, outer=()):
            follows = set()
            for group in _entry_groups(sequence, follows):
                nodes = tuple(group) if isinstance(group, tuple) else (group,)
                carries = any(_residency_directive(item) for node in nodes for item in walk(node)
                              if _kind(item) == "Comment")
                if len(nodes) == 1 and _kind(nodes[0]) in {"If_Construct", "Associate_Construct", "Block_Construct"}:
                    # The construct's own statements (its conditions, selectors) are host code as well.
                    host(tuple(item for item in nodes[0].content if _kind(item) in _CONSTRUCT_HEADERS),
                         id(nodes[0]) in follows or carries, outer)
                    scan(tuple(_children(nodes[0])), outer)
                elif any(_kind(node) == "Block_Nonlabel_Do_Construct" for node in nodes) and not carries:
                    emit(nodes, sequence, outer)
                elif len(nodes) == 1 and _kind(nodes[0]) == "Block_Nonlabel_Do_Construct" \
                        and _holds_a_sequence(nodes[0]):
                    # A loop around nests, with directives between them: the host runs it.
                    within(nodes[0], outer)
                else:
                    # A nest that carries a residency directive is host code that manages its own access.
                    host(nodes, id(nodes[0]) in follows or carries, outer)

        scan(tuple(_children(routine.execution)))
        temporaries += _temporaries(analysis, routine, procedure)
    analysis.verify_source()
    return {"version": 2, "kind": "caller_owned_resident_entries", "source_inputs": analysis.sources,
            "entries": entries, "rejected": rejected, "grouping_rejections": grouping_rejections,
            "host_statements": host_statements, "temporaries": temporaries,
            "forwarders": _forwarders(analysis, set(resolved_procedures), host_statements),
            "lifetime_authority": "caller storage epoch; numerical export grants no native-effect authority"}


def pass_scalars_by_value(extracted, directory, *, resources=(), names=None):
    """Rewrite exported entries so chosen INTENT(IN) scalars are passed by VALUE.

    Lower-bound parameters always qualify: the caller evaluates them after it
    binds the reached allocation. ``resources`` names further scalars, as
    ``module::name``, that the caller guarantees are defined before any entry
    runs. ``names(entry)`` may return the names of scalar parameters that are
    defined whenever that one entry runs. VALUE is a language-level input
    contract; other scalars keep their guarded reference ABI. Returns the
    export with the rewritten paths.
    """
    import re
    from hashlib import sha256

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    resources = set(resources)
    entries = []
    for original in extracted["entries"]:
        own = set(names(original)) if names is not None else set()
        values = [parameter["name"] for parameter in original["parameters"]
                  if parameter["lower_bound_dimension"] is not None or parameter["resource"] in resources
                  or (not parameter["rank"] and parameter["name"] in own)]
        source = Path(original["input_path"])
        text = source.read_text()
        for name in values:
            pattern = (
                r"(?im)^([ \t]*(?:integer(?:\s*\([^)]+\))?|real(?:\s*\([^)]+\))?|double\s+precision)"
                r"(?:(?!\bvalue\b)[^\n])*?),\s*intent\s*\(\s*in\s*\)\s*::\s*("
                + re.escape(name)
                + r")\s*$"
            )
            if re.search(rf"(?im)^[ \t]*[^\n]*\bvalue\b[^\n]*::\s*{re.escape(name)}\s*$", text):
                continue
            text, count = re.subn(pattern, r"\1, value, intent(in) :: \2", text)
            if count != 1:
                raise CompilationError("Unexpected resident scalar declaration: " + name)
        path = directory / source.name
        path.write_text(text)
        entries.append({**original, "input_path": str(path), "value_inputs": values,
                        "value_contract": "lower bounds of the bound arrays and scalars the caller defines before any entry",
                        "original_numerical_input": {"path": str(source.resolve()),
                                                     "sha256": sha256(source.read_bytes()).hexdigest()}})
    return {**extracted, "entries": entries}


def guard_entries(extracted, directory, condition):
    """Make every exported entry do its work only where ``condition`` holds.

    The condition is wrapped around the body of each innermost loop, where the
    dependence analysis reads it as a fact about the loop bounds. It also joins
    the entry's runtime guards, so a caller that reaches the entry while the
    condition is false stops instead of silently skipping work. Use it for a
    property that the caller's activation implies and the loop bounds do not
    show, such as a minimum extent that keeps a halo copy from overlapping.
    A condition that is a conjunction is applied part by part: an entry gets
    the parts whose variables it has as parameters, and none if it has none of
    them. Returns the export with the rewritten paths.
    """
    import re
    from hashlib import sha256

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    parts, depth, start = [], 0, 0
    for match in re.finditer(r"\(|\)|\.and\.", condition, re.IGNORECASE):
        if match.group() in "()":
            depth += 1 if match.group() == "(" else -1
        elif depth == 0:
            parts.append(condition[start:match.start()].strip())
            start = match.end()
    parts.append(condition[start:].strip())
    whole = condition
    entries = []
    for original in extracted["entries"]:
        available = {parameter["name"].lower() for parameter in original["parameters"]}
        condition = " .and. ".join(
            part for part in parts
            if {name.lower() for name in re.findall(r"[A-Za-z_]\w*", re.sub(r"\.\w+\.|\b\d\w*", " ", part))}
            - {"min", "max", "abs", "mod", "size"} <= available)
        if not condition:
            entries.append(original)
            continue
        if len(parts) == 1:
            condition = whole
        source = Path(original["input_path"])
        lines = source.read_text().splitlines(keepends=True)
        open_loops, bodies = [], []
        for index, line in enumerate(lines):
            statement = line.strip().upper()
            if re.match(r"DO\b", statement):
                open_loops.append([index, True])
            elif re.match(r"END\s*DO\b", statement):
                first, innermost = open_loops.pop()
                if open_loops:
                    open_loops[-1][1] = False
                if innermost:
                    bodies.append((first + 1, index))
        if open_loops or not bodies:
            raise CompilationError("Cannot guard an entry without complete loops: " + original["entry"])
        for first, last in reversed(bodies):
            indent = lines[first][:len(lines[first]) - len(lines[first].lstrip())]
            lines[first:last] = [indent + "if (" + condition + ") then\n", *lines[first:last], indent + "endif\n"]
        path = directory / source.name
        path.write_text("".join(lines))
        entries.append({**original, "input_path": str(path),
                        "runtime_guards": [*original["runtime_guards"], condition],
                        "guard": {"condition": condition,
                                  "original": {"path": str(source.resolve()),
                                               "sha256": sha256(source.read_bytes()).hexdigest()}}})
    return {**extracted, "entries": entries}


def compile_resident_entry(source, entry, directory, *, host_threads=4, collective=False, injective=()):
    """Compile one entry for forced device placement and write its sources.

    The directory receives the entry's sources, the runtime headers they
    include and ``compiler-report.json``. Returns that report; its ``scoped``
    member is what a caller needs to bind arguments and call the entry.
    """
    import json

    from compiler.driver.kernel_entry import (
        compile_kernel_entry,
        kernel_entry_outputs,
        kernel_entry_report,
        write_kernel_entry,
    )
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig

    try:
        config = OffloadConfig(host_threads, collective, "direct")
    except ValueError as error:
        raise CompilationError(str(error)) from error
    options = CompilerOptions(opt_level=1, fallback="error", scope_transfers="direct",
                              injective_arrays=tuple(injective))
    _function, plan, sources = compile_kernel_entry(Path(source).resolve(), entry, options=options,
                                                    offload_config=config, common_header="runtime.hpp")
    outputs = kernel_entry_outputs(sources, common_header="runtime.hpp")
    report = kernel_entry_report(entry, plan, sources, outputs)
    directory = write_kernel_entry(outputs, directory)
    (directory / "compiler-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compile_resident_entries(extracted, directory, *, host_threads=4):
    """Compile every exported entry; keep those whose regions all run on the device.

    An entry that the compiler rejects, or that would leave a region on the
    host, joins the export's rejections with its source span, so the caller
    keeps that span native. For a loop marked ``descends`` the entries of its
    body are compiled in its place: the loop is listed under ``descended``,
    and the host statements of its body under ``host_statements``. Writes and
    returns ``device-entries.json``.
    """
    import json

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    entries, rejected, descended = [], [], []
    opened = set()

    def counts(item):
        """Not an alternative, or one of loops that all turned out not to be entries."""
        return all((item["procedure"], *span) in opened for span in item.get("alternative_of", ()))

    for region in extracted["entries"]:
        if not counts(region):
            continue
        target = directory / region["source_identity"][:16]
        span = {"procedure": region["procedure"], "first_line": region["first_line"],
                "last_line": region["last_line"]}
        checked = {requirement["resource"] for requirement in region.get("injective", ())}
        unavailable = ()
        try:
            report = compile_resident_entry(
                region["input_path"], region["entry"], target, host_threads=host_threads,
                injective=[parameter["name"] for parameter in region["parameters"]
                           if parameter["rank"] and parameter["resource"] in checked])
            unavailable = [item for item in report["scoped"]["region_execution"] if not item["gpu_available"]]
            failure = {"reason": "Resident GPU entry contains native-only regions",
                       "regions": unavailable} if unavailable else None
        except CompilationError as error:
            failure = {"reason": str(error)}
        if failure is None:
            entries.append({"region": region, "directory": str(target), "entry": report["scoped"]})
        elif region.get("descends"):
            # The host runs this loop; the nests in its body were exported for this case.
            opened.add((span["procedure"], span["first_line"], span["last_line"]))
            descended.append({**span, **failure})
        else:
            rejected.append({**span, **failure, **region.get("host_code", {})})
    rejected = [item for item in extracted["rejected"] if counts(item)] + rejected
    record = {"version": 1, "kind": "resident_device_entries", "source_inputs": extracted["source_inputs"],
              "entries": entries, "rejected": rejected, "descended": descended,
              "host_statements": [item for item in extracted.get("host_statements", ())
                                  if item.get("alternative_of") and counts(item)],
              "gpu_execution_established": False}
    (directory / "device-entries.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
