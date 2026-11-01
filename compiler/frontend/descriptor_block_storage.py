"""A complete joined source closure for one ordinary-graph BLOCK boundary.

Numerical grouping is irrelevant. The existing team, declaration and SAME
issuers independently authenticate every original body and callee. This local
composition grants no execution, access, definition or completion permission.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from hashlib import sha256

from compiler.frontend.descriptor_associations import _same_syntax, _syntax_attachments
from compiler.frontend.descriptor_blocks import FixedScalarBlock, _same_declaration, original_fixed_block
from compiler.frontend.descriptor_intrinsics import (
    intrinsic_scope_attachments,
    intrinsic_source_attachments,
    same_intrinsic_scope_attachments,
    same_intrinsic_source_attachments,
)
from compiler.frontend.descriptor_teams import DescriptorTeamProof
from compiler.frontend.optional_scalar_control import _same_scope_state, _scope_state
from compiler.frontend.storage_mutation_effects import (
    StorageMutationEffects,
    _preserves_record,
    prove_storage_mutation_effects,
)
from compiler.frontend.structured_effects import _thaw
from compiler.ir import CompilationError

DESCRIPTOR_WINDOW_JOINED_BLOCK_VERSION = 20
_ISSUER, _ABSENT = object(), object()


def _attached(value):
    return tuple((item.name, getattr(value, item.name)) for item in fields(value))


def _same(value, attached):
    return all(getattr(value, name) is item for name, item in attached)


def _encoded(value):
    return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _team_syntax(originals):
    return tuple(row for original in originals for row in _syntax_attachments(original))


def _receipt(analysis, proof, registry_name):
    registry = getattr(analysis, registry_name, None)
    if not isinstance(registry, dict) or registry.get(proof.identity) is not proof:
        raise CompilationError("joined BLOCK lost its exact registered source receipt")
    return (proof, _attached(proof), _encoded(proof._record), registry_name, registry,
            getattr(analysis, registry_name + "_owner", _ABSENT), proof.identity)


def _same_receipt(analysis, receipt):
    proof, attached, encoded, name, registry, owner, identity = receipt
    return (_same(proof, attached) and _encoded(proof._record) == encoded
            and getattr(analysis, name, None) is registry and registry.get(identity) is proof
            and getattr(analysis, name + "_owner", _ABSENT) is owner)


def _immutable_scope_syntax(analysis, scopes):
    """Use the existing sealed source facts; never cache a validation result."""
    if getattr(analysis, "_source_session", None) is None:
        return None
    context = analysis.source_authority_context()
    snapshot = analysis._source_syntax
    if not snapshot.immutable:
        raise CompilationError("joined BLOCK requires its sealed original syntax authority")
    nodes = tuple(scope.node for scope in scopes)
    membership = snapshot.checkpoint()
    if not all(membership.unchanged(node) for node in nodes):
        raise CompilationError("joined BLOCK namespace has foreign original syntax")
    return context, snapshot, nodes


def _same_immutable_scope_syntax(analysis, saved):
    if saved is None:
        return True
    context, snapshot, nodes = saved
    context.require(analysis)
    membership = snapshot.checkpoint()
    return (analysis._source_syntax is snapshot and snapshot.immutable
            and all(membership.unchanged(node) for node in nodes))


def _scopes(analysis):
    """Stack-local complete original namespace receipt, never a lookup cache."""
    tables = tuple((name, getattr(analysis, name), tuple(getattr(analysis, name).items())) for name in
                   ("routines", "numerical_helpers", "modules", "_associate_scopes", "_block_scopes"))
    objects, scopes = {}, {}
    for _name, _table, items in tables:
        for _key, value in items:
            objects[id(value)] = (value, _attached(value))
            scope = getattr(value, "scope", value)
            if not hasattr(scope, "bindings"):
                continue
            while scope is not None and id(scope) not in scopes:
                scopes[id(scope)] = scope
                scope = scope.parent
    states = tuple((scope, _attached(scope), getattr(scope, "qualified", _ABSENT),
                    tuple((binding, _attached(binding), tuple(binding.shape_nodes), tuple(binding.lower_bound_nodes),
                           getattr(binding, "component_object", _ABSENT), getattr(binding, "associate_selector", _ABSENT))
                          for binding in scope.bindings.values())) for scope in scopes.values())
    type_maps = tuple((scope, getattr(scope, "component_types", _ABSENT),
                       tuple(getattr(scope, "component_types", {}).items())) for scope in scopes.values())
    schemas = tuple((schema, _attached(schema), schema.members, tuple(schema.members.items()))
                    for _scope, _mapping, items in type_maps for _name, schema in items)
    # Creating a fresh authority context serializes source policy. Keep raw
    # namespace/container entries before that callback can alter any input.
    sealed = getattr(analysis, "_source_session", None) is not None
    namespaces = tuple(intrinsic_scope_attachments(scope, syntax=not sealed)
                       for scope in scopes.values())
    source = intrinsic_source_attachments(analysis)
    immutable_syntax = _immutable_scope_syntax(analysis, scopes.values())
    raw = (tables, tuple(objects.values()), states, type_maps, schemas, namespaces,
           source, immutable_syntax)
    if not _same_scope_fields(analysis, (*raw, ())):
        raise CompilationError("joined BLOCK namespace changed before canonical publication")
    # Canonical namespace serialization calls Binding.public. Save every raw
    # attachment before any serializer can mutate an earlier consumed scope.
    resolutions = tuple(_scope_state(scope) for scope in scopes.values())
    return (*raw, resolutions)


def _same_scope_fields(analysis, saved):
    tables, objects, states, type_maps, schemas, namespaces, source, immutable_syntax, _resolutions = saved
    return (all(getattr(analysis, name, None) is table and len(table) == len(items)
                and all(table.get(key) is value for key, value in items) for name, table, items in tables)
            and all(_same(value, attached) for value, attached in objects)
            and all(getattr(scope, "component_types", _ABSENT) is mapping
                    and (mapping is _ABSENT or len(mapping) == len(items)
                         and all(mapping.get(name) is value for name, value in items))
                    for scope, mapping, items in type_maps)
            and all(_same(schema, attached) and schema.members is members and len(members) == len(items)
                    and all(members.get(name) is value for name, value in items)
                    for schema, attached, members, items in schemas)
            and all(same_intrinsic_scope_attachments(state) for state in namespaces)
            and same_intrinsic_source_attachments(analysis, source)
            and _same_immutable_scope_syntax(analysis, immutable_syntax)
            and all(_same(scope, attached) and getattr(scope, "qualified", _ABSENT) == qualified
                    and all(_same(binding, fields_) and len(binding.shape_nodes) == len(axes)
                            and all(a is b for a, b in zip(binding.shape_nodes, axes, strict=True))
                            and len(binding.lower_bound_nodes) == len(lowers)
                            and all(a is b for a, b in zip(binding.lower_bound_nodes, lowers, strict=True))
                            and getattr(binding, "component_object", _ABSENT) is root
                            and getattr(binding, "associate_selector", _ABSENT) is selector
                            for binding, fields_, axes, lowers, root, selector in bindings)
                    for scope, attached, qualified, bindings in states))


def _same_scopes(analysis, saved):
    return (_same_scope_fields(analysis, saved)
            and all(_same_scope_state(state) for state in saved[-1])
            and _same_scope_fields(analysis, saved))


def _original(analysis, routine, graph, node_id, original, team):
    if (type(team) is not DescriptorTeamProof or team._analysis is not analysis
            or team.procedure != getattr(routine, "qualified", None)
            or analysis._require_original(routine.qualified) is not routine
            or graph is not analysis.structure(routine.qualified)):
        raise CompilationError("joined BLOCK requires its exact original routine and same-analysis team")
    selected_graph, selected = analysis._selected_source(routine.qualified, team._originals)
    if node_id not in graph.nodes:
        raise CompilationError("joined BLOCK requires its original graph boundary")
    originals = graph.source_nodes(node_id)
    if (selected_graph is not graph or selected != team.selected_node_ids
            or graph.nodes[node_id].kind != "boundary"
            or len(originals) != 1 or originals[0] is not original
            or type(original).__name__ != "Block_Construct"
            or sum(node is original for node, _children in _team_syntax(team._originals)) != 1):
        raise CompilationError("joined BLOCK requires exact complete-team membership and original boundary")


def _public_record(routine, graph, node_id, team, block, storage):
    return {"version": 1, "kind": "original_joined_block_storage", "procedure": routine.qualified,
            "source_graph": graph.identity, "original_node_id": node_id,
            "team_storage": team.public(), "block_scope": block.public(), "storage_mutation": storage.public(),
            "complete_original_group_required": True, "complete_original_block_required": True,
            "descriptor_evidence_only": True, "entry_definition_authority": False,
            "payload_access_authorized": False, "alias_authority": False, "execution_authorized": False,
            "motion_authorized": False, "numerical_completion_authorized": False}


@dataclass(frozen=True, slots=True)
class DescriptorJoinedBlockStorage:
    identity: str
    _analysis: object = field(repr=False)
    _routine: object = field(repr=False)
    _graph: object = field(repr=False)
    _node_id: str
    _original: object = field(repr=False)
    _team: object = field(repr=False)
    _block: object = field(repr=False)
    _storage: object = field(repr=False)
    _syntax: object = field(repr=False)
    _record: str = field(repr=False)
    _issuer: object = field(repr=False)

    def checkpoint(self):
        return (_attached(self), tuple(_receipt(self._analysis, proof, name) for proof, name in (
            (self._team, "_descriptor_team_proofs"), (self._block, "_descriptor_blocks"),
            (self._storage, "_storage_mutation_effects"))), _scopes(self._analysis))

    def require_checkpoint(self, saved):
        attached, receipts, namespaces = saved
        analysis = self._analysis

        def unchanged():
            return (_same(self, attached) and _same_syntax(self._syntax)
                    and all(_same_receipt(analysis, receipt) for receipt in receipts))

        if not unchanged():
            raise CompilationError("joined BLOCK consumed source receipt changed at final publication")
        declaration = _same_declaration(analysis, self._block._declaration)
        scopes = _same_scopes(analysis, namespaces)
        if not declaration or not scopes or not unchanged():
            raise CompilationError("joined BLOCK consumed source receipt changed at final publication")

    def validate(self, *, scope_validator=None, tick=None):
        attached = _attached(self)
        if (type(self) is not DescriptorJoinedBlockStorage or self._issuer is not _ISSUER
                or type(self._team) is not DescriptorTeamProof or type(self._block) is not FixedScalarBlock
                or type(self._storage) is not StorageMutationEffects
                or any(proof._analysis is not self._analysis for proof in (self._team, self._block, self._storage))
                or self._block._routine is not self._routine or self._block._original is not self._original
                or sha256(self._record.encode()).hexdigest() != self.identity):
            raise CompilationError("joined BLOCK requires its exact independent source composition")
        analysis, routine, graph, node_id, original = (self._analysis, self._routine, self._graph,
                                                       self._node_id, self._original)
        team, block, storage, syntax = self._team, self._block, self._storage, self._syntax
        receipts = tuple(_receipt(analysis, proof, name) for proof, name in (
            (team, "_descriptor_team_proofs"), (block, "_descriptor_blocks"),
            (storage, "_storage_mutation_effects")))
        namespaces = _scopes(analysis)
        _original(analysis, routine, graph, node_id, original, team)
        team.validate(analysis, routine.qualified, team.selected_node_ids)
        block.validate(scope_validator=scope_validator, tick=tick)
        storage.validate(analysis, routine.qualified)
        if (storage._selection != team.selected_node_ids or storage._joined_storage is not team
                or not _preserves_record(storage._record)
                or _encoded(_public_record(routine, graph, node_id, team, block, storage)) != self._record):
            raise CompilationError("joined BLOCK requires complete unchanged whole-team storage effects")
        _original(analysis, routine, graph, node_id, original, team)
        analysis.verify_publication()
        analysis.verify_source()
        analysis.inputs.verify_live()
        source_identity = analysis.inputs.identity()
        declaration = _same_declaration(analysis, block._declaration)
        scopes = _same_scopes(analysis, namespaces)
        if (not _same(self, attached) or not _same_syntax(syntax)
                or any(not _same_receipt(analysis, receipt) for receipt in receipts)
                or source_identity != _thaw(storage._record["source_inputs"])
                or not declaration or not scopes):
            raise CompilationError("joined BLOCK source or retained receipt changed at publication")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def original_joined_block_storage(analysis, routine, graph, node_id, original, team,
                                  *, scope_validator=None, tick=None):
    """Check every original team statement; never substitute a numerical unit."""
    if type(team) is not DescriptorTeamProof or team._analysis is not analysis:
        raise CompilationError("joined BLOCK requires its exact same-analysis issued source team")
    receipts = [_receipt(analysis, team, "_descriptor_team_proofs")]
    syntax = _team_syntax(team._originals)
    namespaces = _scopes(analysis)

    def checkpoint():
        if (not _same_syntax(syntax) or not _same_scopes(analysis, namespaces)
                or any(not _same_receipt(analysis, receipt) for receipt in receipts)):
            raise CompilationError("joined BLOCK consumed source receipt changed before publication")

    _original(analysis, routine, graph, node_id, original, team)
    checkpoint()
    team.validate(analysis, routine.qualified, team.selected_node_ids)
    checkpoint()
    block = original_fixed_block(analysis, routine, original, scope_validator=scope_validator, tick=tick)
    receipts.append(_receipt(analysis, block, "_descriptor_blocks"))
    checkpoint()
    storage = prove_storage_mutation_effects(analysis, routine.qualified,
        selection=team.selected_node_ids, joined_storage=team)
    receipts.append(_receipt(analysis, storage, "_storage_mutation_effects"))
    checkpoint()
    if not _preserves_record(storage._record):
        raise CompilationError("joined BLOCK whole-team source has mutation, escape or unknown closure")
    record = _encoded(_public_record(routine, graph, node_id, team, block, storage))
    proof = DescriptorJoinedBlockStorage(sha256(record.encode()).hexdigest(), analysis, routine, graph,
        node_id, original, team, block, storage, syntax, record, _ISSUER)
    attached = _attached(proof)
    result = proof.validate(scope_validator=scope_validator, tick=tick)
    checkpoint()
    if result is not proof or not _same(proof, attached):
        raise CompilationError("joined BLOCK issuer changed its exact final composition")
    return proof
