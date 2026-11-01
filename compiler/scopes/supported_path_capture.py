"""One synchronous supported-path capture transaction, never a proof cache."""
from __future__ import annotations

import json
from dataclasses import fields

from compiler.frontend.entry_preservation import _native_attachments, _projection_attachments
from compiler.frontend.optional_scalar_control import _same_scope_state, _scope_state
from compiler.ir import CompilationError


def _encoded(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CompilationError("supported capture metadata changed during its transaction") from error


def _attached(value):
    return tuple((part.name, getattr(value, part.name)) for part in fields(value))


def _binding_state(binding):
    return (_attached(binding), _encoded(binding.public()),
            tuple(binding.shape_nodes), tuple(binding.lower_bound_nodes),
            getattr(binding, "component_object", None), getattr(binding, "associate_selector", None))


def _same_binding(binding, state):
    attached, public, axes, lowers, root, selector = state
    return (all(getattr(binding, name) is value for name, value in attached)
            and _encoded(binding.public()) == public and len(binding.shape_nodes) == len(axes)
            and all(a is b for a, b in zip(binding.shape_nodes, axes, strict=True))
            and len(binding.lower_bound_nodes) == len(lowers)
            and all(a is b for a, b in zip(binding.lower_bound_nodes, lowers, strict=True))
            and getattr(binding, "component_object", None) is root
            and getattr(binding, "associate_selector", None) is selector)


def _kind_state(scope):
    result, seen, current = [], set(), scope.kinds
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        result.append((current, current.parent, current.values, tuple(current.values.items()),
                       getattr(current, "intrinsic_kinds", None),
                       _encoded(getattr(current, "intrinsic_kinds", None))))
        current = current.parent
    return scope.kinds, tuple(result)


def _same_kinds(scope, state):
    head, values = state
    return (scope.kinds is head and all(current.parent is parent and current.values is mapping
            and len(mapping) == len(items) and all(mapping.get(name) is value for name, value in items)
            and getattr(current, "intrinsic_kinds", None) is intrinsic
            and _encoded(intrinsic) == encoded for current, parent, mapping, items, intrinsic, encoded in values))


def capture_many(epoch, builder, bindings, *, _consumer_check=None):
    """Authenticate everything on both sides of detached, pure fact filtering.

    All receipts exist only on this synchronous stack. In particular a later
    capture batch, standalone capture or plan check performs its full proof
    validation again. Baselines precede the first charged validation callback.
    """
    if (not isinstance(bindings, tuple) or len({id(binding) for binding in bindings}) != len(bindings)
            or len({binding.root for binding in bindings}) != len(bindings)):
        raise CompilationError("supported capture requires distinct exact original bindings")
    owner, analysis, routine, config = epoch._owner, builder.analysis, builder.entry, builder.config
    graph, facts, captures = owner.graph, builder.facts, builder.facts["captures"]
    selection = builder._native_ancestor_selection
    selection_fields = _attached(selection)
    members, original_members = owner.members, tuple(owner.members.items())
    owner_fields = tuple((name, getattr(owner, name)) for name in
                         ("closed_selection", "native_ancestor_selection", "eligibility_allowed", "supported_path", "supported_gate"))
    epoch_fields, facts_record = _attached(epoch), _encoded(facts)
    config_fields, profile_record = _attached(config), _encoded(config.profile)
    originals = tuple((binding, binding.root, _binding_state(binding), captures.get(binding.root),
                       _encoded(captures.get(binding.root))) for binding in bindings)
    retained, seen = [], set()

    def retain(proof, registry_name):
        if id(proof) in seen:
            return
        seen.add(id(proof))
        registry = getattr(analysis, registry_name, None)
        if not isinstance(registry, dict) or registry.get(proof.identity) is not proof:
            raise CompilationError("supported capture lost an exact retained source receipt")
        retained.append((proof, _attached(proof), registry_name, registry, proof.identity))

    descriptor = epoch._descriptor
    retain(selection, "_native_ancestor_selections")
    retain(epoch._path, "_supported_path_selections")
    retain(epoch._gate, "_allocation_gates")
    retain(descriptor, "_supported_path_storage")
    for _node, proof in (*descriptor._components, *descriptor._exposures, *descriptor._outside):
        retain(proof, "_descriptor_window_proofs")
    for _node, _original, clock, proof in descriptor._metadata:
        retain(clock, "_scalar_clock_proofs")
        retain(proof, "_descriptor_window_proofs")
    entry, position = epoch._entry_preservation, epoch._reached_position
    if entry is not None:
        retain(entry, "_entry_preservations")
        retain(position, "_original_reached_positions")
        for proof, _attached_fields, name, _registry, _identity in (
                *_native_attachments(analysis, entry._preservations),
                *_projection_attachments(analysis, entry._projection)):
            retain(proof, name)
    owners = tuple((name, value) for name, value in analysis.__dict__.items() if name.endswith("_owner"))
    tables = tuple((name, getattr(analysis, name), tuple(getattr(analysis, name).items()))
                   for name in ("routines", "numerical_helpers", "modules", "_block_scopes"))
    scopes = {}
    for _name, _table, items in tables:
        for _key, value in items:
            current = getattr(value, "scope", value)
            while current is not None and id(current) not in scopes:
                scopes[id(current)] = current
                current = current.parent
    for association in getattr(analysis, "_associate_scopes", {}).values():
        if association.available:
            scopes[id(association.scope)] = association.scope
    scope_states = tuple((_scope_state(scope), _kind_state(scope),
                          tuple((binding, _binding_state(binding)) for binding in scope.bindings.values()))
                         for scope in scopes.values())

    def checkpoint():
        if _consumer_check is not None:
            _consumer_check()
        if (any(getattr(epoch, name) is not value for name, value in epoch_fields)
                or owner.builder is not builder or owner.position_epoch is not epoch
                or owner.routine is not routine or owner.graph is not graph
                or builder.analysis is not analysis or builder.entry is not routine or builder.config is not config
                or builder._native_ancestor_selection is not selection
                or any(getattr(selection, name) is not value for name, value in selection_fields)
                or owner.members is not members or len(members) != len(original_members)
                or any(members.get(name) is not member for name, member in original_members)
                or any(getattr(owner, name) is not value for name, value in owner_fields)
                or builder.facts is not facts or facts.get("captures") is not captures
                or any(getattr(config, name) is not value for name, value in config_fields)
                or _encoded(config.profile) != profile_record
                or config.policy != "sections" or config.scope_execution != "graph"
                or any(getattr(analysis, name, None) is not value for name, value in owners)
                or any(getattr(analysis, name, None) is not table or len(table) != len(items)
                       or any(table.get(key) is not value for key, value in items) for name, table, items in tables)
                or any(getattr(analysis, name, None) is not registry or registry.get(identity) is not proof
                       or any(getattr(proof, field) is not value for field, value in attached)
                       for proof, attached, name, registry, identity in retained)
                or any(not _same_scope_state(state) or not _same_kinds(state[0], kinds)
                       or any(not _same_binding(binding, saved) for binding, saved in original_bindings)
                       for state, kinds, original_bindings in scope_states)
                or any(not _same_binding(binding, saved) for binding, _root, saved, _fact, _record in originals)):
            raise CompilationError("supported capture source, policy or receipt attachment changed during transaction")

    epoch.validate(builder)
    checkpoint()
    # Preserve the scalar resource_state contract: every retained edge must
    # agree on each original declaration. These are detached validated states.
    states = [dict((state["resource"], state) for state in proof.public()["resources"])
              for _node, proof in descriptor._components]
    results = []
    for binding, root, _saved, supplied, record in originals:
        checked = builder.capture_fact(binding)
        checkpoint()
        if (checked is not supplied or captures.get(root) is not supplied or _encoded(checked) != record
                or not states or any(root not in component for component in states)
                or any(component[root] != states[0][root] for component in states[1:])):
            raise CompilationError("supported capture substituted facts or disagreed on original descriptor state")
        results.append(epoch._capture_result(binding, json.loads(record)))
        checkpoint()
    epoch.validate(builder)
    analysis.verify_publication()
    analysis.verify_source()
    analysis.inputs.verify_live()
    checkpoint()
    if _encoded(facts) != facts_record:
        raise CompilationError("supported capture original facts changed at publication")
    return tuple(results)
