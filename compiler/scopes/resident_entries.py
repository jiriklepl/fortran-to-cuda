"""Numerical entries for a caller-owned residency domain.

This producer deliberately has no placement or lifetime policy. It exports
source-backed numerical regions through the same extraction used by checked
source scopes. The owner retains scalar control flow, native effects and
allocation epochs. Every exported entry still goes through ordinary numerical
lowering and dependence analysis before it can execute on a device.
"""
import re
from dataclasses import replace
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
from compiler.frontend.source_effects import Binding, SourceEffects, _children, _kind
from compiler.ir import CompilationError
from compiler.scopes.numerical import resource_binding
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


def _counted(loop):
    """The variable and the bounds of a counted DO construct; (None, ()) for any other loop."""
    statement = next((item for item in _children(loop) if _kind(item) == "Nonlabel_Do_Stmt"), None)
    control = statement.items[1] if statement is not None else None
    counted = control.items[1] if control is not None and len(control.items) > 1 else None
    if not counted:
        return None, ()
    return str(counted[0]).lower(), tuple(counted[1])


def _carries(loop):
    """Whether the iterations of a loop may depend on each other through an array its body assigns.

    A cheap reading of the subscripts, to decide which loops are worth trying
    as loops the host runs around a nest; the dependence analysis has the say
    on whether a nest compiles. The iterations of a loop are apart when every
    assigned array is addressed through the loop's variable (directly or
    through scalars and inner loop bounds computed from it), and wherever the
    body names that array it has the same subscripts in those places. They may
    depend on each other when an assigned array is not addressed through the
    variable at all, as under the steps of an iteration or of time, or when it
    is named with other subscripts there, as in a sweep through planes that
    each use the one before. A body that calls procedures is not looked into.
    """
    variable, _bounds = _counted(loop)
    if variable is None or walk(loop, Call_Stmt):
        return False
    depends, grown = {variable}, True
    assignments = list(walk(loop, Assignment_Stmt))
    inner = [_counted(item) for item in walk(loop) if _kind(item) == "Block_Nonlabel_Do_Construct" and item is not loop]

    def named(node):
        return {str(name).lower() for name in walk(node, Name)}

    while grown:
        grown = False
        for statement in assignments:
            target = statement.items[0]
            if _kind(target) == "Name" and str(target).lower() not in depends and named(statement.items[2]) & depends:
                depends.add(str(target).lower())
                grown = True
        for name, bounds in inner:
            if name is not None and name not in depends and any(named(bound) & depends for bound in bounds):
                depends.add(name)
                grown = True

    def subscripts(reference):
        listed = reference.items[1]
        return tuple(_children(listed)) if _kind(listed) == "Section_Subscript_List" else (listed,)

    uses = {}
    for reference in walk(loop, Part_Ref):
        uses.setdefault(str(reference.items[0]).lower(), []).append(subscripts(reference))
    for statement in assignments:
        target = statement.items[0]
        if _kind(target) != "Part_Ref":
            continue
        written = subscripts(target)
        places = [index for index, item in enumerate(written) if named(item) & depends]
        if not places:
            return True
        for other in uses[str(target.items[0]).lower()]:
            if len(other) != len(written) or any(_text(other[index]) != _text(written[index]) for index in places):
                return True
    return False


def _around_nests(loop):
    """Whether a loop may be one that the host runs around nests, and not the outer loop of a nest.

    Either its body is more than one nest (a loop nest beside other statements
    or nests), or its body is one nest and its iterations may depend on each
    other (see ``_carries``).
    """
    body = [item for item in _children(loop) if _kind(item) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"}]
    if not any(_kind(item) == "Block_Nonlabel_Do_Construct" for item in body):
        return False
    return len(body) > 1 or _carries(loop)


def _text(node):
    return " ".join(str(node).split())


# The inquiry that says whether an array with the attribute exists.
_EXISTS = {"allocatable": "allocated", "pointer": "associated", "optional": "present"}
_LEAVING = {"Return_Stmt", "Goto_Stmt", "Computed_Goto_Stmt", "Stop_Stmt", "Error_Stop_Stmt", "Exit_Stmt",
            "Cycle_Stmt"}


def _runs(analysis, routine, nodes):
    """A condition under which any of the given counted loops runs at all, or None.

    Each loop has no stated step. The condition is asked before the first of
    them: a later loop's bounds can only have changed where an earlier loop
    ran. Bounds that read an array's values are not asked that early, since
    the values on the host may not be the newest.
    """
    conditions = []
    for loop in nodes:
        if _kind(loop) != "Block_Nonlabel_Do_Construct":
            return None
        control = next(item for item in loop.content if _kind(item) == "Nonlabel_Do_Stmt").items[1]
        counted = control.items[1] if control is not None and _kind(control) == "Loop_Control" else None
        if not counted or len(counted) != 2 or len(counted[1]) != 2:
            return None
        for bound in counted[1]:
            inquired = {id(reference.items[1].items[0]) for reference in walk(bound, Intrinsic_Function_Reference)
                        if str(reference.items[0]).lower() in _INQUIRIES and reference.items[1] is not None
                        and reference.items[1].items}
            for name in walk(bound, Name):
                binding = analysis._binding(routine.scope, name)
                if binding is not None and binding.rank and id(name) not in inquired:
                    return None
        lower, upper = _text(counted[1][0]), _text(counted[1][1])
        if lower.isdigit() and upper.isdigit() and int(upper) >= int(lower):
            # A loop that always runs: there is nothing to ask.
            return None
        conditions.append(f"({upper}) >= ({lower})")
    return " .or. ".join(dict.fromkeys(conditions)) or None


def _statement_form(nodes):
    """What generated host access around host statements has to know about their shape.

    ``kind`` is the kind of a single statement, ``group`` for the statements
    that open the parts of one construct, and ``statements`` for several
    whole statements. A one-line IF comes with its ``condition``, the names
    in it and its ``action``, so that the access can go inside the condition.
    ``leaves`` says that control can leave from inside (RETURN, GOTO, EXIT
    ...), past whatever follows.
    """
    several = "group" if any(_kind(node) in _CONSTRUCT_HEADERS for node in nodes) else "statements"
    form = {"kind": _kind(nodes[0]) if len(nodes) == 1 else several,
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


def _bound(analysis):
    """``_bound_procedures`` of an analysis, worked out once."""
    if not hasattr(analysis, "bound_procedures"):
        analysis.bound_procedures = _bound_procedures(analysis)
    return analysis.bound_procedures


def _bound_procedures(analysis):
    """The procedures the derived types of the sources bind, by type and binding name, and each type's parent.

    A type and a binding name map to the procedures bound under the name in
    types of that name, each with whether the object is passed as its first
    argument, or to None when one of them is not known here (bound to a
    procedure of another module, or with a PASS attribute that names another
    argument). A generic binding stands for its specific ones, which
    ``_reached`` resolves.
    """
    found, generics, parents = {}, {}, {}
    for module, scope in analysis.modules.items():
        for definition in walk(scope.node):
            if _kind(definition) != "Derived_Type_Def":
                continue
            header = definition.content[0]
            owner = str(header.items[1]).lower()
            extended = re.search(r"\bextends\s*\(\s*(\w+)\s*\)", str(header.items[0] or "").lower())
            if extended:
                parents.setdefault(owner, set()).add(extended[1])
            for binding in walk(definition):
                if _kind(binding) == "Generic_Binding":
                    names = binding.items[2]
                    generics.setdefault((owner, str(binding.items[1]).lower()), set()).update(
                        str(item).lower() for item in (_children(names) if _kind(names) != "Name" else (names,)))
                    continue
                if _kind(binding) != "Specific_Binding":
                    continue
                key = owner, str(binding.items[3]).lower()
                attributes = {item.split("(")[0].strip() for item in str(binding.items[1] or "").lower().split(",")}
                if "deferred" in attributes:
                    # Its implementations are the bindings of the same name in the extending types.
                    found.setdefault(key, set())
                    continue
                procedure = module + "::" + str(binding.items[4] or binding.items[3]).lower()
                known = procedure in analysis.routines or procedure in analysis.numerical_helpers
                if not known or "pass" in attributes or found.get(key, set()) is None:
                    found[key] = None
                else:
                    found.setdefault(key, set()).add((procedure, "nopass" not in attributes))
    return {"specific": found, "generic": generics, "parents": parents}


def _reached(analysis, declared, name):
    """What a call through the binding ``name`` of an object declared ``declared`` may reach, or None if unknown.

    The object may be of the declared type, where the binding may be inherited
    from any type it extends, or of a type that extends it and binds the name
    anew. With no declared type (the object is a component, say) every type of
    the sources counts. That is more than the call reaches and never less.
    """
    bound = _bound(analysis)
    owners = {owner for owner, _name in (*bound["specific"], *bound["generic"])}
    if declared is not None:
        family, pending = set(), [declared]
        while pending:
            owner = pending.pop()
            if owner not in family:
                family.add(owner)
                pending += bound["parents"].get(owner, ())
        pending = [declared]
        while pending:
            owner = pending.pop()
            children = [child for child, above in bound["parents"].items() if owner in above and child not in family]
            family.update(children)
            pending += children
        owners &= family
    targets, names, seen = set(), [name], set()
    while names:
        current = names.pop()
        if current in seen:
            continue
        seen.add(current)
        for owner in owners:
            names += bound["generic"].get((owner, current), ())
            specific = bound["specific"].get((owner, current), set())
            if specific is None:
                return None
            targets |= specific
    return targets


def _array_uses(analysis, routine, nodes):
    """The arrays ``nodes`` name, the procedures they call, and the arrays that are used outside a call.

    A whole array that is an actual argument is the called procedure's
    business. A section or an expression of it is evaluated here, so it counts
    as a use, like any other mention. Asking for an array's bounds or
    allocation, or allocating it, reads none of its values. Also returns the
    arrays the statements may change (an assignment target, input, an argument
    of a procedure that does not take it INTENT(IN)) and those they deallocate.
    """
    arrays, calls, passed, elsewhere = {}, set(), set(), set()
    written, released, others = set(), set(), set()

    def named(node):
        found = (analysis._binding(routine.scope, name) for name in walk(node, Name) if id(name) not in others)
        return {binding.root for binding in found if binding is not None and binding.rank}

    def may_write(procedure, arguments, first=0):
        """The arrays among the actual arguments that the called procedure may change.

        ``first`` is the position of the first actual argument among the dummies: one
        where the object of a type-bound call is passed ahead of them.
        """
        callee = analysis.routines.get(procedure) or analysis.numerical_helpers.get(procedure)
        for position, argument in enumerate(_children(arguments) if arguments is not None else (), first):
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
            if _kind(call.items[0]) == "Procedure_Designator":
                # Through a type-bound procedure: every procedure bound under the name, if they are known.
                base = call.items[0].items[0]
                base = base.items[0] if _kind(base) == "Part_Ref" else base
                declared = analysis._binding(routine.scope, base) if _kind(base) == "Name" else None
                declared = re.fullmatch(r"(?:type|class)\((\w+)\)", (declared.dtype or "").replace(" ", "")) \
                    if declared is not None else None
                targets = _reached(analysis, declared[1] if declared else None, str(call.items[0].items[2]).lower())
                if not targets:
                    calls.update(("$unresolved", "$opaque"))
                    may_write(None, call.items[1])
                for procedure, object_first in targets or ():
                    calls.add(procedure)
                    may_write(procedure, call.items[1], int(object_first))
                continue
            try:
                procedure = resolve_original_call(analysis, routine.scope, call).procedure
                calls.add(procedure)
            except CompilationError:
                procedure = None
                candidates = analysis._candidates(routine.scope, call.items[0])
                calls.update(candidates or ("$unresolved",))
                if not candidates and _may_be_ours(analysis, routine, str(call.items[0]).lower()):
                    calls.add("$opaque")
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
                arrays[binding.root] = _spelled(binding, name)
                if id(name) not in passed:
                    elsewhere.add(binding.root)
    return arrays, calls, elsewhere, written & set(arrays), released & set(arrays)


def _spelled(binding, name):
    """The binding under the name the source gives the array here: a USE may have renamed it."""
    if binding.name.lower() == str(name).lower() or hasattr(binding, "component_object"):
        return binding
    return replace(binding, name=str(name))


def _may_be_ours(analysis, routine, name):
    """Whether a call that did not resolve may still reach a procedure of the sources.

    A dummy procedure may be any of them. Another name is one of them only if
    the sources define a procedure of that name; else it is an intrinsic or a
    library's, which sees no array it is not handed.
    """
    if not hasattr(analysis, "procedure_names"):
        analysis.procedure_names = {key.rsplit("::", 1)[-1] for key in (*analysis.routines, *analysis.numerical_helpers)}
    return name in routine.scope.procedure_arguments or name in analysis.procedure_names


def _host_procedures(analysis, selected):
    """What procedures outside the selection do with arrays: a function from a name to a record, and all records.

    The function answers None for a selected procedure, which sees to its
    own arrays, and for one that is no procedure of the sources: a library
    procedure sees only what it is handed. A record has the arrays the
    procedure ``used`` other than by passing them whole, by resource; the
    procedures it ``calls``; ``opaque`` where a call of it may reach a
    procedure of the sources that is not known; and under ``outside`` the
    arrays it touches that are not its own dummies or locals, each with
    whether it is only ``passed`` whole to procedures, may be ``written`` or
    ``released``, and the ``host`` procedure it belongs to if it is no module
    array (an internal procedure sees the variables of its host).
    """
    records = {}

    def scan(name):
        if name in records:
            return records[name]
        routine = analysis.routines.get(name) or analysis.numerical_helpers.get(name)
        if name in selected or routine is None or routine.execution is None:
            return None
        try:
            arrays, calls, elsewhere, written, released = _array_uses(
                analysis, routine, tuple(_children(routine.execution)))
        except CompilationError:
            arrays, calls, elsewhere, written, released = {}, {"$opaque"}, set(), set(), set()
        outside = {}
        for root, binding in arrays.items():
            entity = root.rsplit("::", 1)[-1]
            own = root.startswith(name + "::") or (root.startswith("argument::") and entity in routine.arguments)
            if own:
                continue
            module = root.split("::", 1)[0]
            host = None if root.count("::") == 1 and module in analysis.modules and "%" not in root \
                else name.rsplit("::", 1)[0]
            outside[root] = {"binding": binding, "host": host, "passed": root not in elsewhere,
                             "written": root in written, "released": root in released}
        records[name] = {"used": sorted(elsewhere), "calls": sorted(calls - {"$opaque"}),
                         "opaque": "$opaque" in calls, "outside": outside}
        return records[name]

    return scan, records


_CONDITIONAL = {"If_Construct", "If_Stmt", "Case_Construct", "Where_Construct", "Where_Stmt", "Forall_Construct",
                "Forall_Stmt", "Select_Type_Construct"}
_PLAIN = {"Name", "Int_Literal_Constant", "Level_2_Expr", "Level_2_Unary_Expr", "Add_Operand", "Mult_Operand",
          "Parenthesis", "Section_Subscript_List", "str", "NoneType"}


def _fixed_elements(analysis, routine, nodes):
    """Elements of arrays that nests only read, at subscripts that stay the same while the nests run.

    Such an element is one value for the whole entry: the call can read it on
    the host and pass it, and the array needs no place on the device. That
    holds for an array every reference to which, in ``nodes``, is an element
    whose subscripts are made of integer constants and of scalars the nests
    do not assign, and that the nests neither assign nor pass on. The read
    moves ahead of the nests, so the element must be reached in every
    iteration (it stands under no condition) and the array must always be
    there (not allocatable, a pointer or optional). Returns the references,
    grouped by array resource.
    """
    try:
        arrays, _calls, elsewhere, written, released = _array_uses(analysis, routine, nodes)
    except CompilationError:
        return {}
    assigned = set()
    for node in nodes:
        for statement in walk(node, Assignment_Stmt):
            if _kind(statement.items[0]) == "Name":
                assigned.add(str(statement.items[0]).lower())
        for loop in walk(node):
            if _kind(loop) == "Block_Nonlabel_Do_Construct":
                assigned.add(_counted(loop)[0])
    found, refused = {}, set()
    for node in nodes:
        for reference in walk(node, Part_Ref):
            binding = analysis._binding(routine.scope, reference.items[0]) if _kind(reference.items[0]) == "Name" else None
            if binding is None or not binding.rank:
                continue
            root = binding.root
            subscripts = reference.items[1]
            parts = list(_children(subscripts)) if _kind(subscripts) == "Section_Subscript_List" else [subscripts]
            fixed = len(parts) == binding.rank and all(
                _kind(item) in _PLAIN and (_kind(item) != "Name" or (
                    str(item).lower() not in assigned and (scalar := analysis._binding(routine.scope, item)) is not None
                    and not scalar.rank and scalar.dtype == "integer"))
                for part in parts for item in walk(part))
            above, conditional = reference.parent, False
            while above is not None and above not in nodes:
                conditional = conditional or _kind(above) in _CONDITIONAL
                above = getattr(above, "parent", None)
            if fixed and not conditional and above is not None:
                found.setdefault(root, []).append(reference)
            else:
                refused.add(root)
    return {root: references for root, references in found.items()
            if root not in refused and root not in written and root not in released
            and (arrays[root].dtype, arrays[root].kind) in {("real", 4), ("real", 8), ("integer", 4)}
            and not hasattr(arrays[root], "component_object")
            and not arrays[root].attributes & {"allocatable", "pointer", "optional"}
            # Every mention of the array is one of these elements.
            and sum(1 for node in nodes for name in walk(node, Name)
                    if (seen := analysis._binding(routine.scope, name)) is not None and seen.root == root) == len(references)}


def _as_values(analysis, routine, elements):
    """Put a scalar in place of each of the ``elements`` in the parse tree; returns how to undo it, and the scalars.

    The scalars are local to the procedure for as long as they stand in the
    tree: an entry exported meanwhile takes them as inputs. Each maps, by
    resource, to the element it stands for as the source writes it.
    """
    changes, values, texts = [], {}, {}
    for _root, references in sorted(elements.items()):
        for reference in references:
            text = _text(reference)
            name = texts.setdefault(text.lower().replace(" ", ""), f"fort_fixed_{len(texts) + 1}")
            binding = analysis._binding(routine.scope, reference.items[0])
            routine.scope.bindings[name] = Binding(name, routine.scope.qualified + "::" + name, binding.dtype, binding.kind, 0)
            values[routine.scope.qualified + "::" + name] = text
            parent, new = reference.parent, Name(name)
            new.parent = parent
            if isinstance(getattr(parent, "items", None), tuple) and any(item is reference for item in parent.items):
                before = parent.items
                parent.items = tuple(new if item is reference else item for item in before)
                changes.append((parent, "items", before))
            else:
                before = list(parent.content)
                parent.content[:] = [new if item is reference else item for item in before]
                changes.append((parent, "content", before))

    def undo():
        for parent, attribute, before in reversed(changes):
            if attribute == "items":
                parent.items = before
            else:
                parent.content[:] = before
        for name in texts.values():
            routine.scope.bindings.pop(name, None)

    return undo, values


def _readable(analysis, routine, parameter):
    """Whether a scalar input can be read wherever its nest is reached, whatever the nest then does with it.

    A scalar that only a branch of a nest uses may be undefined where no
    iteration takes the branch. Reading an undefined variable early is still
    only a load of a value nothing uses, as long as the variable is there. An
    allocatable scalar may not be; pointers and optional arguments are no
    inputs of a nest at all. A component is not looked into.
    """
    if parameter.rank or parameter.lower_bound_dimension is not None or "%" in parameter.resource:
        return False
    name = parameter.resource.rsplit("::", 1)[-1]
    binding = analysis._binding(routine.scope, name)
    if binding is None or binding.root != parameter.resource:
        try:
            binding = resource_binding(analysis, routine, parameter.resource)
        except CompilationError:
            return False
    return not binding.attributes & {"allocatable", "pointer", "optional"}


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


def export_resident_entries(paths, procedures, directory, *, allow_patterns=False):
    """Export resident numerical entries for explicit procedures or glob patterns.

    Besides ``entries`` and ``rejected``, the result lists ``host_statements``:
    the statements of the selected procedures that stay host code and name
    arrays, each with its span, the arrays and whether a residency directive
    directly precedes it or sits inside it (``acknowledged``). They run beside
    the entries, so each one that touches an array an entry binds needs the
    host to hold current values.

    A loop whose body holds a loop nest beside other statements, or one nest
    while its iterations may depend on each other (see ``_carries``), may be a
    nest itself, or a loop the host has to run around nests (an iteration, a
    time loop, a sweep). When it cannot be exported, its body is exported in
    its place. When
    it can, its entry is marked ``descends`` and its body is exported too:
    the entries, rejections and host statements of the body carry
    ``alternative_of``, the spans of the loops they replace, outermost first,
    and count only where none of those loops compiled as an entry (see
    ``compile_resident_entries``). An entry that such a loop shares with other
    nests is marked the same way, with each of its nests as the alternative.
    """
    # A selected procedure can be long, the more so with calls replaced by their statements; the
    # default bound on statements per procedure is sized for proofs this export does not ask for.
    # A main program is analysed as a procedure only when it is asked for by name.
    programs = tuple(sorted({name.lower() for name in procedures or ()
                             if re.fullmatch(r"\$program::[a-z][a-z0-9_]*", name.lower())}))
    analysis = SourceEffects(paths, operations=4096, program_roots=programs)
    analysis.element_fields = True
    # The body of a module function holds loop nests like any subroutine's.
    for name, routine in analysis.numerical_helpers.items():
        if routine.source_kind == "function" and name not in analysis.routines:
            analysis.routines[name] = routine
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
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    resolved_procedures = _resolve_procedures(analysis, procedures, allow_patterns=allow_patterns)
    entries, rejected, grouping_rejections, host_statements, temporaries = [], [], [], [], []
    host_procedure, host_procedures = _host_procedures(analysis, set(resolved_procedures))
    def export(procedure, entries, rejected, grouping_rejections, host_statements, temporaries, splitting=None):
        """Export what one procedure holds. With ``splitting``, only the nests of those spans, each on its own."""
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
            if splitting is not None:
                if (first, last) in splitting and len(loops) > 1:
                    for node in loops:
                        emit((node,), sequence, (*outer, (first, last)))
                    return
                if not outer:
                    return
            # Elements that are one value for the whole entry are read at the call and passed as scalars.
            fixed = _fixed_elements(analysis, routine, selected) if len(loops) == len(nodes) else {}
            undo, values = _as_values(analysis, routine, fixed)
            try:
                context = original_region_context(routine, selected, analysis=analysis)
                extracted = extract_region(analysis, routine, selected,
                                           preceding=context.preceding, following=context.following,
                                           cyclic_following=context.cyclic_following)
                path = extracted.write(directory)
                public = extracted.public()
                readable = sorted(parameter.name for parameter in extracted.parameters
                                  if _readable(analysis, routine, parameter))
            except CompilationError as error:
                undo()
                # A larger interface can expose an unsupported association
                # even when its individual source nests remain exportable.
                if len(nodes) > 1 and len(loops) == len(nodes):
                    grouping_rejections.append({"procedure": procedure, "first_line": first,
                                                "last_line": last, "reason": str(error)})
                    for node in nodes:
                        emit((node,), sequence, outer)
                    return
                if len(nodes) == 1 and _around_nests(nodes[0]):
                    # Not a nest but a loop around nests, such as an iteration or a time loop: the
                    # host runs it, and what stands in its body is looked at in its own right.
                    grouping_rejections.append({"procedure": procedure, "first_line": first,
                                                "last_line": last, "reason": str(error)})
                    within(nodes[0], outer)
                    return
                rejected.append({"procedure": procedure, "source": str(routine.scope.path),
                                 "first_line": first, "last_line": last,
                                 "reason": str(error), **alternative(outer), **host_code(nodes)})
                return
            undo()
            arrays_named = {root: str(references[0].items[0]) for root, references in fixed.items()}
            structures = []
            # Variables that a USE renamed: the call names them as the nests do. An element passed as a
            # value is named as the nests write it.
            spelled = dict(values)
            for node in nodes:
                for name in walk(node, Name):
                    binding = analysis._binding(routine.scope, name)
                    if binding is not None and _spelled(binding, name) is not binding:
                        spelled[binding.root] = str(name)
            for root in sorted(_structure_objects(analysis, routine.scope, nodes)):
                structures.append({"name": spelled.get(root, root.rsplit("::", 1)[-1]), "resource": root})
            runs = _runs(analysis, routine, nodes)
            entries.append({"procedure": procedure, "source": str(routine.scope.path),
                            "input_path": str(path.resolve()), **public,
                            "source_loop_spans": [list(statement_span(node)) for node in loops],
                            "structures": structures, "injective": _indirect_writes(analysis, routine, loops),
                            # For the case that the entry does not compile and the nests stay host code.
                            "host_code": host_code(nodes),
                            # Where no nest has an iteration, there is nothing to bind and nothing to launch.
                            **({"runs": runs} if runs else {}),
                            **({"spelled": spelled} if spelled else {}),
                            # Arrays whose elements the call reads on the host: their values must be there.
                            **({"observed": sorted({arrays_named[root] for root in fixed})} if fixed else {}),
                            "readable_scalars": readable,
                            **alternative(outer)})
            if len(nodes) == 1 and _around_nests(nodes[0]):
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

        def touched(arrays, written, released, elsewhere):
            return [{"name": binding.name, "resource": root, "type": binding.dtype, "kind": binding.kind,
                     "rank": binding.rank, "field": hasattr(binding, "component_object"),
                     # All the code does with the array is pass it whole to procedures.
                     "handed": root not in elsewhere,
                     # What must hold for the array to exist when the statements start.
                     "exists": sorted(_EXISTS[name] for name in binding.attributes if name in _EXISTS),
                     "access": "release" if root in released else "write" if root in written else "read"}
                    for root, binding in sorted(arrays.items())]

        def uses(nodes):
            """The arrays host code touches, its calls, and the arrays the host procedures it calls may reach.

            ``reached`` lists every array that such a procedure, or one it
            calls, touches without being handed it: a module array, or a
            variable of this procedure seen by its internal procedures. Each
            comes with the name it has here, with the USE that gives it a
            name here, or as ``unnameable``. Which of them count is decided
            where the selected and the kept arrays are known.
            """
            arrays, calls, elsewhere, written, released = _array_uses(analysis, routine, nodes)
            found, pending, seen = {}, sorted(calls), set()
            while pending:
                name = pending.pop()
                record = host_procedure(name) if name not in seen else None
                seen.add(name)
                if record is None:
                    continue
                pending += record["calls"]
                for root, item in record["outside"].items():
                    # A variable of another procedure is that procedure's business.
                    if item["host"] is None or (procedure + "::").startswith(item["host"] + "::"):
                        found[root] = item["binding"]
            reached = []
            for root, binding in sorted(found.items()):
                entity = root.rsplit("::", 1)[-1]
                module = root.split("::", 1)[0]
                here = analysis._binding(routine.scope, entity)
                item = {"name": entity, "resource": root, "type": binding.dtype, "kind": binding.kind,
                        "rank": binding.rank, "field": False, "handed": False,
                        "exists": sorted(_EXISTS[name] for name in binding.attributes if name in _EXISTS)}
                if root in arrays:
                    item["name"] = arrays[root].name
                elif here is not None and here.root == root:
                    pass
                elif module in analysis.modules and root.count("::") == 1 and not procedure.startswith(module + "::") \
                        and analysis._exported(analysis.modules[module], entity):
                    alias = f"fort_{module}_{entity}"
                    item.update(name=alias if len(alias) <= 63 else "fort_" + sha256(root.encode()).hexdigest()[:16],
                                use=[module, entity])
                else:
                    # A private array, or one a local name hides: it cannot be named here.
                    item["unnameable"] = True
                reached.append(item)
            return {"arrays": touched(arrays, written, released, elsewhere), "calls": sorted(calls - {"$opaque"}),
                    **({"reached": reached} if reached else {}), **({"opaque": True} if "$opaque" in calls else {})}

        def host_code(nodes):
            """How a nest that stays on the host touches arrays, for the access it then needs; {} if unknown."""
            try:
                return {**uses(nodes), **_statement_form(nodes),
                        **({"runs": runs} if (runs := _runs(analysis, routine, nodes)) else {})}
            except CompilationError:
                return {}

        def host(nodes, acknowledged, outer=()):
            """Record host code that touches arrays: it runs beside entries that may hold newer values."""
            if splitting is not None and not outer:
                return
            used = uses(nodes)
            if used["arrays"] or used.get("reached") or used.get("opaque"):
                first = min(statement_span(node)[0] for node in nodes)
                last = max(statement_span(node)[1] for node in nodes)
                text = " ".join(" ".join(str(node).split()) for node in nodes)
                host_statements.append({"procedure": procedure, "source": str(routine.scope.path),
                                        "first_line": first, "last_line": last,
                                        "text": text if len(text) <= 240 else text[:237] + "...",
                                        # Stable while the statement and its procedure keep their text.
                                        "id": sha256((procedure + "\n" + text).encode()).hexdigest()[:16],
                                        **used, **_statement_form(nodes), "acknowledged": acknowledged,
                                        **alternative(outer)})

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
                        and _around_nests(nodes[0]):
                    # A loop around nests, with directives between them: the host runs it.
                    within(nodes[0], outer)
                else:
                    # A nest that carries a residency directive is host code that manages its own access.
                    host(nodes, id(nodes[0]) in follows or carries, outer)

        scan(tuple(_children(routine.execution)))
        temporaries += _temporaries(analysis, routine, procedure)

    for procedure in resolved_procedures:
        export(procedure, entries, rejected, grouping_rejections, host_statements, temporaries)

    def retry(procedure, first, last):
        """The nests of a span that was exported as one entry, each as an entry of its own."""
        more = {"entries": [], "rejected": [], "grouping_rejections": [], "host_statements": []}
        export(procedure, *more.values(), [], splitting={(first, last)})
        return {"source_inputs": analysis.sources, **more}

    return {"version": 2, "kind": "caller_owned_resident_entries", "source_inputs": analysis.sources,
            "entries": entries, "rejected": rejected, "grouping_rejections": grouping_rejections,
            "host_statements": host_statements, "temporaries": temporaries, "retry": retry,
            "forwarders": {name: {**record, "outside": {
                root: {key: value for key, value in item.items() if key != "binding"}
                for root, item in record["outside"].items()}} for name, record in sorted(host_procedures.items())},
            "lifetime_authority": "caller storage epoch; numerical export grants no native-effect authority"}


def pass_scalars_by_value(extracted, directory):
    """Rewrite exported entries so that the scalar inputs that can be read early are passed by VALUE.

    Those are the lower bounds of the bound arrays, which the caller evaluates
    after it binds the reached allocation, and the INTEGER and REAL scalars
    whose storage always exists (``readable_scalars`` of the export). Other
    scalar inputs keep the reference ABI, which reads one only where the
    original statements would; a nest in which only a branch reads such a
    scalar stays on the host. Returns the export with the rewritten paths.
    """
    import re
    from hashlib import sha256

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for original in extracted["entries"]:
        bounds = {parameter["name"] for parameter in original["parameters"]
                  if parameter["lower_bound_dimension"] is not None}
        source = Path(original["input_path"])
        text = source.read_text()
        values = []
        for name in (*sorted(bounds), *original.get("readable_scalars", ())):
            pattern = (
                r"(?im)^([ \t]*(?:integer(?:\s*\([^)]+\))?|real(?:\s*\([^)]+\))?|double\s+precision)"
                r"(?:(?!\bvalue\b)[^\n])*?),\s*intent\s*\(\s*in\s*\)\s*::\s*("
                + re.escape(name)
                + r")\s*$"
            )
            if re.search(rf"(?im)^[ \t]*[^\n]*\bvalue\b[^\n]*::\s*{re.escape(name)}\s*$", text):
                values.append(name)
                continue
            text, count = re.subn(pattern, r"\1, value, intent(in) :: \2", text)
            if count == 1:
                values.append(name)
            elif name in bounds:
                raise CompilationError("Unexpected resident scalar declaration: " + name)
        path = directory / source.name
        path.write_text(text)
        entries.append({**original, "input_path": str(path), "value_inputs": values,
                        "value_contract": "lower bounds of the bound arrays and scalars whose storage always exists",
                        "original_numerical_input": {"path": str(source.resolve()),
                                                     "sha256": sha256(source.read_bytes()).hexdigest()}})
    return {**extracted, "entries": entries}


def compile_resident_entry(source, entry, directory, *, host_threads=4, injective=()):
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
        config = OffloadConfig(host_threads)
    except ValueError as error:
        raise CompilationError(str(error)) from error
    options = CompilerOptions(opt_level=1, injective_arrays=tuple(injective))
    _function, plan, sources = compile_kernel_entry(Path(source).resolve(), entry, options=options,
                                                    offload_config=config, common_header="runtime.hpp")
    outputs = kernel_entry_outputs(sources, common_header="runtime.hpp")
    report = kernel_entry_report(entry, plan, sources, outputs)
    directory = write_kernel_entry(outputs, directory)
    (directory / "compiler-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compile_resident_entries(extracted, directory, *, host_threads=4, expand=None):
    """Compile every exported entry; keep those whose regions all run on the device.

    An entry that the compiler rejects, or that would leave a region on the
    host, joins the export's rejections with its source span, so the caller
    keeps that span native. For a span marked ``descends`` (a loop the host
    may run around nests, or nests that share an entry with one) the entries
    exported for what it holds are compiled in its place, and they are kept
    when at least one of them compiles: the span is then listed under
    ``descended``, and the host statements of a loop's body under
    ``host_statements``. When none compiles the span is rejected as a whole,
    which hands its arrays to the host once instead of once per iteration; its
    record says under ``inside`` why what it holds did not compile either.
    Nests that share an entry which does not compile are tried one by one in
    the same way: ``expand(region)`` exports them when that happens (see the
    ``retry`` of an export), and returns their entries, rejections and host
    statements. Writes and returns ``device-entries.json``.
    """
    import json

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    entries, rejected, tried = [], [], {}
    pending, expanded = list(extracted["entries"]), {"rejected": [], "host_statements": []}

    def counts(item, opened):
        """Not an alternative, or one of spans that all turned out not to be entries."""
        return all((item["procedure"], *span) in opened for span in item.get("alternative_of", ()))

    while pending:
        region = pending.pop(0)
        if not counts(region, tried):
            continue
        target = directory / region["source_identity"][:16]
        span = {"procedure": region["procedure"], "source": region["source"], "first_line": region["first_line"],
                "last_line": region["last_line"], **{key: region[key] for key in ("alternative_of",) if key in region}}
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
            # The host may run this loop; what its body holds was exported for this case.
            tried[span["procedure"], span["first_line"], span["last_line"]] = (
                {**span, **failure}, region.get("host_code", {}))
        elif expand is not None and len(region["source_loop_spans"]) > 1:
            # One nest may be all that stands in the way of the others: each is exported on its own now.
            more = expand(region)
            tried[span["procedure"], span["first_line"], span["last_line"]] = (
                {**span, **failure}, region.get("host_code", {}))
            pending[:0] = more["entries"]
            for key in expanded:
                expanded[key] += more[key]
        else:
            rejected.append({**span, **failure, **region.get("host_code", {})})
    # A span is worth opening when an entry compiled somewhere inside it.
    fruitful = {(item["region"]["procedure"], *span) for item in entries
                for span in item["region"].get("alternative_of", ())}
    opened = {key for key in tried if key in fruitful}
    failed = [*extracted["rejected"], *expanded["rejected"], *rejected]

    def inside(key):
        """Why what a span holds did not compile either, for the report."""
        return sorted({": ".join([item["reason"], *(region["reason"] for region in item.get("regions", ())
                                                    if region.get("reason"))])
                       for item in (*failed, *(record for record, _code in tried.values()))
                       if item["procedure"] == key[0] and list(key[1:]) in item.get("alternative_of", ())})

    rejected = [item for item in failed if counts(item, opened)]
    rejected += [{**record, **code, "inside": inside(key)} for key, (record, code) in tried.items()
                 if key not in opened and counts(record, opened)]
    rejected.sort(key=lambda item: (item["source"], item["first_line"], item["last_line"]))
    descended = [{name: value for name, value in record.items() if name != "alternative_of"}
                 for key, (record, _code) in tried.items() if key in opened]
    record = {"version": 1, "kind": "resident_device_entries", "source_inputs": extracted["source_inputs"],
              "entries": entries, "rejected": rejected, "descended": descended,
              "host_statements": [item for item in (*extracted.get("host_statements", ()), *expanded["host_statements"])
                                  if item.get("alternative_of") and counts(item, opened)],
              "gpu_execution_established": False}
    (directory / "device-entries.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
