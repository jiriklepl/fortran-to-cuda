"""Link supported source tasks to checked per-call allocation forwarding.

This is a projection of CheckedBorrowedCall receipts, not another call-proof
issuer. It adds no source, layout, definition, alias or lifetime authority.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256

from compiler.frontend.atomic_worksharing_team import AtomicWorksharingTeamProof
from compiler.frontend.immutable_graph import original_type
from compiler.frontend.line_snapshot_team import OriginalLineSnapshotTeamProof
from compiler.ir import CompilationError
from compiler.scopes.conditional_return import CheckedBorrowedCall
from compiler.scopes.segments import statement_span


def _fail(message):
    raise CompilationError("source task call projection " + message)


def _digest(record):
    return sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _projectable(task):
    """Only existing direct, unpriced line or atomic whole-team execution."""
    return (
        (type(getattr(task.proof, "team", None)) is OriginalLineSnapshotTeamProof
         or type(task.proof) is AtomicWorksharingTeamProof)
        and task._transport == "direct"
        and task.pricing is None
        and task.placement is None
    )


def needs_call_receipt(member):
    """Retain only paths that actually contain supported source work."""
    pending, seen = [member], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if (any(_projectable(task) for task in current.source_device_tasks)
                or any(operation.native_opaque_accesses for unit in current.units for operation in unit.scope.native)):
            return True
        for receipt in getattr(current, "source_companion_calls", ()):
            if original_type(receipt) is not CheckedBorrowedCall:
                _fail("requires an exact retained checked CALL receipt")
            pending.append(receipt.callee)
    return False


def _emitted_calls(source, entry):
    """Each compiler-generated companion call, including trace-control copies."""
    compact = "".join(source.lower().split()).replace("&", "")
    marker = "call" + entry.lower() + "("
    starts = [match.end() for match in re.finditer(re.escape(marker), compact)]
    if not starts:
        _fail("lost its emitted companion CALL")
    result = []
    for start in starts:
        depth = 1
        end = start
        while depth and end < len(compact):
            depth += (compact[end] == "(") - (compact[end] == ")")
            end += 1
        if depth:
            _fail("has an incomplete emitted companion CALL")
        result.append(compact[start:end])
    return tuple(result)


def _field_roots(nodes, edges, procedure, field_slots):
    """Finite per-slot effect closure, never enumerate joint CALL paths.

    Inputs here are projections already validated by call_projection. This helper
    issues no authority. Joint associations remain on the exact checked edges;
    overlapping possible-root sets from different invocations are not aliases.
    """
    incoming = {name: [] for name in nodes}
    for edge in edges:
        if edge["caller"] not in nodes or edge["callee"] not in nodes:
            _fail("lost a member of its owning projection")
        incoming[edge["callee"]].append(edge)
    memo, active = {}, set()
    for slot in field_slots.values():
        pending = [((procedure, slot), False)]
        while pending:
            key, finish = pending.pop()
            if key in memo:
                continue
            name, current_slot = key
            if name not in nodes or current_slot not in nodes[name]["resource_slots"].values():
                _fail("lost a slot of its owning projection")
            node = nodes[name]
            if not node["parameterized"]:
                reverse = {number: node["owner_resources"].get(root) for root, number in node["resource_slots"].items()}
                if reverse[current_slot] is None:
                    _fail("lost its final owning allocation")
                memo[key] = frozenset((reverse[current_slot],))
                continue
            dependencies = []
            for edge in incoming[name]:
                actual = edge["handle_bindings"].get(str(current_slot))
                if actual is None:
                    _fail("lost a forwarded slot of its owning projection")
                dependencies.append((edge["caller"], actual))
            if not dependencies:
                _fail("has no complete owning projection")
            if finish:
                memo[key] = frozenset().union(*(memo[parent] for parent in dependencies))
                active.remove(key)
            else:
                if key in active:
                    _fail("contains a cycle in its owning projection")
                active.add(key)
                pending.append((key, True))
                pending.extend((parent, False) for parent in dependencies if parent not in memo)
    return {field: sorted(memo[(procedure, slot)]) for field, slot in field_slots.items()}


def checked_call_inventory(member):
    """Existing exact companion receipts and local slots, without task authority."""
    from compiler.scopes.module_owner import ModuleOwner

    owner = member.owner
    if owner.members.get(member.routine.qualified) is not member:
        _fail("lost its exact checked member")
    nodes, edges, active = {}, {}, set()
    changed = False

    def visit(current):
        nonlocal changed
        name = current.routine.qualified
        if name in active:
            _fail("contains a cycle")
        if name in nodes:
            return
        if current.owner is not owner or owner.members.get(name) is not current:
            _fail("contains a foreign owning invocation")
        active.add(name)
        namespace = current.resource_slot_namespace
        parameterized = namespace is current
        if namespace is not None and not parameterized:
            _fail("requires an exact original helper namespace")
        if parameterized and original_type(current) is not ModuleOwner:
            _fail("requires an existing parameterized module companion")
        if parameterized:
            for _root, (binding, slot) in current.resources.items():
                canonical = current.canonical(binding).root
                formals = [formal for formal, actual in current.canonical_bindings.items() if actual.root == canonical]
                expected_root = formals[0] if len(formals) == 1 else canonical
                if len(formals) > 1 or current.resource_slot_roots.get(slot) != expected_root:
                    _fail("changed its separate helper storage identity namespace")
        slots = {root: slot for root, (_binding, slot) in current.resources.items()}
        if len(set(slots.values())) != len(slots):
            _fail("has ambiguous helper resource slots")
        node = {
            "procedure": name,
            "source_graph": current.graph.identity,
            "source": str(current.routine.scope.path),
            "source_sha256": current.builder.analysis.sources[str(current.routine.scope.path)],
            "parameterized": parameterized,
            "resource_slots": slots,
        }
        nodes[name] = node
        if not parameterized:
            canonical = {}
            for _root, (binding, slot) in current.resources.items():
                original = current.canonical(binding).root
                if original not in owner.resources or owner.resources[original][1] != slot:
                    _fail("terminal namespace differs from owning registrations")
                canonical[_root] = original
            node["owner_resources"] = canonical
        else:
            sites = tuple(current.call_sites)
            if not sites:
                _fail("parameterized member has no checked original callers")
            expected = {root: binding.root for root, binding in current.canonical_bindings.items()}
            retained_sites = []
            for caller in owner.members.values():
                retained = tuple(getattr(caller, "source_companion_calls", ()))
                if any(original_type(receipt) is not CheckedBorrowedCall for receipt in retained):
                    _fail("requires an exact retained checked CALL receipt")
                matches = [receipt for receipt in retained if receipt.callee is current]
                markers = {
                    marker
                    for marker, (callee, _context) in getattr(caller, "external_control_placeholders", {}).items()
                    if callee is current
                }
                if {receipt.marker for receipt in matches} != markers:
                    _fail("does not cover every emitted companion control")
                retained_sites.extend((caller.routine.qualified, *statement_span(receipt.node)) for receipt in matches)
            declared_sites = [(site["caller"], site["first_line"], site["last_line"]) for site in sites]
            if sorted(retained_sites) != sorted(declared_sites):
                _fail("does not cover every retained checked CALL")
            for site in sites:
                caller = owner.members.get(site["caller"])
                if caller is None:
                    _fail("lost an original caller")
                if caller.resource_slot_namespace is not None and caller.resource_slot_namespace is not caller:
                    _fail("requires an exact original caller namespace")
                span = site["first_line"], site["last_line"]
                retained = tuple(getattr(caller, "source_companion_calls", ()))
                if any(original_type(receipt) is not CheckedBorrowedCall for receipt in retained):
                    _fail("requires an exact retained checked CALL receipt")
                receipts = [
                    receipt
                    for receipt in retained
                    if receipt.callee is current and statement_span(receipt.node) == span
                ]
                if len(receipts) != 1:
                    _fail("requires exactly one retained checked CALL receipt")
                (receipt,) = receipts
                receipt.validate(caller)
                if caller.builder.resolved_calls.get((caller.routine.qualified, span)) != receipt.resolved.public():
                    _fail("differs from the retained resolved source CALL")
                mapping = {str(local): actual for local, actual in receipt.handle_bindings}
                if site.get("handle_bindings") != mapping:
                    _fail("changed its helper handle forwarding")
                if len(set(mapping.values())) != len(mapping):
                    _fail("aliases slots in an original CALL projection")
                actuals = {
                    "argument::" + item.formal: caller.canonical(item.binding).root
                    for item in receipt.resolved.mappings
                    if item.formal_binding.rank
                }
                if site.get("resource_mappings") != actuals:
                    _fail("changed its original canonical actual association")
                controls = _emitted_calls(receipt.source, current.entry)
                storage = {}
                for local, actual in receipt.handle_bindings:
                    value = caller.registration_identity(actual, kind=caller.kind)
                    for control in (
                        f"fort_buffer_{local}=fort_buffer_{actual}",
                        f"fort_storage_identity_{local}={value}",
                    ):
                        if any(
                            len(re.findall(r"(?:^|,)" + re.escape(control.lower()) + r"(?=[,)])", call)) != 1
                            for call in controls
                        ):
                            _fail("changed its emitted handle or storage identity forwarding")
                    storage[str(local)] = {
                        "kind": "forwarded_slot" if caller.resource_slot_namespace is caller else "owner_registration",
                        "slot": actual,
                    }
                changed |= actuals != expected
                graph, selected = caller.builder.analysis._selected_source(caller.routine.qualified, (receipt.node,))
                if graph is not caller.graph or not selected:
                    _fail("lost original source membership")
                if set(map(int, mapping)) != set(current.handle_numbers()):
                    _fail("has incomplete forwarded helper slots")
                if not set(mapping.values()) <= {slot for _binding, slot in caller.resources.values()}:
                    _fail("forwards an unknown caller slot")
                key = caller.routine.qualified, name, *span
                if key in edges:
                    _fail("duplicates an original CALL edge")
                edges[key] = {
                    "caller": caller.routine.qualified,
                    "callee": name,
                    "first_line": span[0],
                    "last_line": span[1],
                    "selected_node_ids": list(selected),
                    "caller_graph": graph.identity,
                    "handle_bindings": mapping,
                    "storage_identities": storage,
                    "resource_mappings": actuals,
                    "source_call": receipt.resolved.public(),
                }
                visit(caller)
        active.remove(name)

    visit(member)
    return nodes, edges, changed


def call_projection(member, task):
    """Revalidate the complete original companion DAG before publishing it."""
    if not _projectable(task):
        return None
    task.proof.validate(member.builder.analysis, member.routine, task.nodes)
    nodes, edges, changed = checked_call_inventory(member)
    owner = member.owner
    if not changed:
        return None  # Preserve the existing single-association record unchanged.
    fields = tuple(binding.root for binding, _written in task.proof.fields)
    local_slots = tuple(member.resources[root][1] for root in fields)
    field_slots = dict(zip(fields, local_slots, strict=True))
    field_roots = _field_roots(nodes, edges.values(), member.routine.qualified, field_slots)
    forbidden = member.host_read_roots() | owner.scalar_element_resources.keys()
    all_roots = {root for values in field_roots.values() for root in values}
    if all_roots & forbidden:
        _fail("conflicts with an opaque original storage reservation")
    actions = {}
    for binding, written in task.proof.fields:
        for root in field_roots[binding.root]:
            canonical = owner.resources[root][0]
            if (binding.dtype, binding.kind, binding.rank) != (canonical.dtype, canonical.kind, canonical.rank):
                _fail("changed a source field type or rank")
            actions[root] = actions.get(root, False) or written
    record = {
        "version": 1,
        "kind": "original_source_task_companion_projection",
        "owner": owner.routine.qualified,
        "procedure": member.routine.qualified,
        "source_proof_identity": task.proof.identity,
        "field_slots": field_slots,
        "members": [nodes[name] for name in sorted(nodes)],
        "calls": [edges[key] for key in sorted(edges)],
        "field_roots": field_roots,
        "root_registrations": {root: owner.resources[root][1] for root in sorted(all_roots)},
        "projection_semantics": "per-field effect closure; joint associations remain in the checked CALL DAG",
        "conservative_effects": {
            "reads": sorted(actions),
            "may_writes": sorted(root for root, written in actions.items() if written),
            "metadata_reads": sorted(actions),
            "must_writes": [],
            "overwrites": [],
        },
        "definition_authorized": False,
        "lifetime_authorized": False,
        "alias_authorized": False,
        "value_specialization_authorized": False,
        "runtime_validation": "original reached registration and existing source access transaction validate each invocation",
    }
    return {"identity": _digest(record), **record}


def source_task_actions(member, task):
    """All possible original allocations, with conservative source field roles."""
    projection = call_projection(member, task)
    if projection is None:
        return {member.canonical(binding).root: written for binding, written in task.proof.fields}
    effects = projection["conservative_effects"]
    writes = set(effects["may_writes"])
    return {root: root in writes for root in effects["reads"]}
