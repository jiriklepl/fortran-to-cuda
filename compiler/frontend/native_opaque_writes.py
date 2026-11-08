"""Compose native-only fixed-array writes from existing source authorities.

The complete original effect inventory is retained. It is not a whole-output
definition proof and supplies no managed payload, entry, alias or device credit.
No scope consumer accepts this token until the separate typed role integration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from hashlib import sha256

from compiler.frontend.authority_context import context_for
from compiler.frontend.call_bindings import _fixed_array_bounds, _original_call_control
from compiler.frontend.descriptor_teams import DescriptorTeamProof
from compiler.frontend.native_completion import (
    NativeCompletionProof,
    NativeGraphCompletionProof,
    _formal_array_call_rows,
    _graph_completion_attachments,
)
from compiler.frontend.native_demand import _Declarations, _owned_storage
from compiler.frontend.resource_reachability import (
    VERSION as REACHABILITY_VERSION,
)
from compiler.frontend.resource_reachability import (
    NativeEffectGraph,
    ResourceReachabilityProof,
    _authenticate_binding,
    _digest,
)
from compiler.frontend.storage_mutation_effects import StorageMutationEffects, source_registry_budget
from compiler.ir import CompilationError

NATIVE_OPAQUE_WRITE_VERSION = 1
_ISSUER = object()
_UNSAFE = {'allocatable', 'pointer', 'target', 'optional', 'value', 'volatile',
           'asynchronous', 'parameter', 'codimension'}
_NO_AUTHORITY = (
    'managed_payload_access_authorized', 'managed_definition_authorized',
    'entry_definition_authority', 'declaration_initialization_credit',
    'capture_authorized', 'device_replica_authorized', 'alias_authority',
    'initial_quiescence_authority', 'ownership_or_lifetime_grant',
    'motion_authority', 'gpu_legality_established', 'cost_authority',
    'execution_authorized', 'whole_output_definedness_authorized',
)


def _same_nodes(left, right):
    return (type(left) is tuple and type(right) is tuple and len(left) == len(right)
            and all(a is b for a, b in zip(left, right, strict=True)))


def _prerequisites(analysis, reachability, storage):
    """Snapshot only the bounded receipts consumed by this composition."""
    if (type(reachability) is not ResourceReachabilityProof
            or type(storage) is not StorageMutationEffects
            or type(reachability.native_graph) is not NativeEffectGraph):
        raise CompilationError('native opaque write requires typed original source prerequisites')
    graph = reachability.native_graph
    owners = [reachability, storage, graph, *graph.templates, *graph.instances]
    bindings = {id(binding): binding for binding in reachability.demanded}
    entries = [('_resource_reachability_proofs', reachability.identity, reachability),
               ('_storage_mutation_effects', storage.identity, storage),
               ('_native_effect_graphs', graph.identity, graph)]
    for access in reachability.accesses:
        owners.extend((access, access.storage))
        bindings[id(access.storage.binding)] = access.storage.binding
    for obligation in reachability.obligations:
        owners.extend((obligation, obligation.native))
    for edge in reachability.calls:
        owners.append(edge)
    attachments = []
    for procedure, selected, completion in reachability._completion:
        if type(completion) is NativeCompletionProof:
            # Complete CALL inventories can retain an ordinary original join.
            # Keep its exact token and registry alongside the graph's checked
            # source-budget receipt; this is not fixed-array write authority.
            completion.validate(analysis, procedure, selected)
            if (tuple(selected) != completion.selected_node_ids or not any(
                    name == procedure and tuple(identities) == tuple(selected) and proof is completion
                    for template in graph.templates for name, identities, proof in template.completion)):
                raise CompilationError('native prerequisites lost their retained ordinary join')
            owners.append(completion)
            entries.append(('_joined_completions', completion.identity, completion))
            continue
        if type(completion) is not NativeGraphCompletionProof:
            raise CompilationError('native prerequisites require exact original completion proofs')
        attachments.extend(_graph_completion_attachments(completion))
        entries.append(('_native_graph_completions', completion.identity, completion))
        for team in completion._reduction_teams:
            if type(team) is not DescriptorTeamProof:
                raise CompilationError('native opaque write requires exact issued joined array teams')
            entries.append(('_descriptor_team_proofs', team.identity, team))
            for association in team._fixed_array_associations:
                entries.append(('_native_fixed_array_associations', association.identity, association))
                bindings[id(association.actual)] = association.actual
                bindings[id(association.formal)] = association.formal
    if storage._source_fact is not None:
        owners.extend((storage._source_fact, storage._source_fact.derived, *storage._source_fact.dependencies))
    for call in storage._calls:
        entries.append(('_original_call_associations', (id(analysis), id(call._original)), call))
    syntax = analysis._source_syntax
    source_nodes = list(reachability._selection)
    scopes = {id(binding.declaring_scope): binding.declaring_scope for binding in bindings.values()}
    owners.extend((*bindings.values(), *scopes.values()))
    namespaces = [(binding.declaring_scope, 'bindings', binding.name, binding) for binding in bindings.values()]
    namespaces.extend((analysis, 'modules', scope.module, scope) for scope in scopes.values()
                      if analysis.modules.get(scope.module) is scope)
    for name, routine in reachability._routines:
        registry_name = 'routines' if analysis.routines.get(name) is routine else 'numerical_helpers'
        namespaces.append((analysis, registry_name, name, routine))
    for binding in bindings.values():
        source_nodes.extend((*binding.shape_nodes, *binding.lower_bound_nodes))
    for scope in scopes.values():
        source_nodes.extend(part for part in getattr(scope.node, 'content', ())
                            if type(part).__name__ == 'Specification_Part')
    for owner in {id(owner): owner for owner in owners}.values():
        attachments.extend((owner, part.name, object.__getattribute__(owner, part.name))
                           for part in fields(type(owner)))
    registries = tuple((name, getattr(analysis, name, None), key, value)
                       for name, key, value in entries)
    return (tuple(attachments), registries, syntax,
            tuple({id(node): node for node in source_nodes if node is not None}.values()), tuple(namespaces))


def _require_prerequisites(analysis, saved, *, same_registries=False):
    attachments, registries, syntax, source_nodes, namespaces = saved
    if (any(object.__getattribute__(owner, name) is not value for owner, name, value in attachments)
            or any((same_registries and getattr(analysis, name, None) is not registry)
                   or type(getattr(analysis, name, None)) is not dict
                   or getattr(analysis, name).get(key) is not value
                   for name, registry, key, value in registries)):
        raise CompilationError('native opaque write prerequisite attachments changed')
    if any(getattr(owner, name).get(key) is not expected for owner, name, key, expected in namespaces):
        raise CompilationError('native opaque write original namespace membership changed')
    if (analysis._source_syntax is not syntax
            or any(not syntax.original_subtree_unchanged(node) for node in source_nodes)):
        raise CompilationError('native opaque write consumed original declaration or control changed')


def _derive(analysis, procedure, selection, binding, reachability, storage):
    saved = _prerequisites(analysis, reachability, storage)
    _require_prerequisites(analysis, saved, same_registries=True)
    context = context_for(analysis)
    session, arena = context.session, context.derivation_arena(analysis)
    original_call = _original_call_control(selection)
    if original_call is None or type(procedure) is not str:
        raise CompilationError('native opaque write requires one complete original CALL control')
    caller = analysis._require_original(procedure)
    parent = session.lexical_parent(original_call)
    if type(parent).__name__ == 'If_Stmt' and selection[0] is not parent:
        raise CompilationError('native opaque write must retain the whole original IF predicate')
    source_graph, selected = analysis._selected_source(procedure, selection)
    # Retain the existing graph and caller attachments as well as the syntax
    # rows; never accept a replacement graph selection after the last callback.
    graph_attachments = tuple((owner, part.name, object.__getattribute__(owner, part.name))
                              for owner in (source_graph, caller)
                              for part in fields(type(owner)))
    graph_entry = (analysis, '_live_structures', (procedure, source_graph.authority_identity), source_graph)
    saved = (saved[0]+graph_attachments, *saved[1:4], (*saved[4], graph_entry))
    _authenticate_binding(analysis, binding)
    scope = binding.declaring_scope
    if (analysis.modules.get(getattr(scope, 'module', None)) is not scope
            or binding.rank != 1 or binding.dtype not in {'real', 'integer'}
            or type(binding.kind) is not int or binding.kind not in {4, 8}
            or binding.attributes & _UNSAFE or not _owned_storage(analysis, binding, _Declarations())):
        raise CompilationError('native opaque write requires ordinary fixed module primitive storage')
    bounds, = _fixed_array_bounds(analysis, binding, element_limit=None)
    count = max(0, bounds[1]-bounds[0]+1)
    if count < 1 or count > ((1 << 63)-1)//binding.kind:
        raise CompilationError('native opaque write storage byte count exceeds its ABI')
    if not any(item is binding for item in reachability.demanded):
        raise CompilationError('native opaque write is absent from the exact source demand')
    reachability.validate(analysis, procedure, selection, reachability.demanded)
    graph = reachability.native_graph
    graph.require_payload_projection()
    if reachability.barriers or not graph.completion_eligible or graph.lifetime_obligations:
        raise CompilationError('native opaque write lacks a complete payload and normal-return graph')
    # Authenticate the existing issuer's effect identity, including OUT undefine.
    # This is a consistency check of that inventory, not another source scan.
    effect_identity = _digest({'version': REACHABILITY_VERSION, 'procedure': procedure,
        'sources': analysis.inputs.identity(), 'selection': list(map(id, selection)),
        'demanded': list(map(id, reachability.demanded)),
        'accesses': [item.public() for item in reachability.accesses],
        'barriers': [item.public() for item in reachability.barriers], 'work': dict(reachability._work)})
    if effect_identity != reachability.identity:
        raise CompilationError('native opaque write original effect inventory changed')
    teams, completions = [], []
    for name, identities, completion in reachability._completion:
        if (type(completion) is not NativeGraphCompletionProof
                or completion._graph is not graph or completion._call_storage is not None
                or not completion._reduction_teams):
            raise CompilationError('native opaque write requires the same strong array completion graph')
        completion.validate(analysis, name, identities)
        completions.append(completion)
        for team in completion._reduction_teams:
            if not any(old is team for old in teams):
                teams.append(team)
    if (not teams or not _same_nodes(storage._joined_storage, tuple(teams))
            or storage._native_graph is not graph or not _same_nodes(storage._selection, selection)):
        raise CompilationError('native opaque write requires identical graph, selection and joined storage receipts')
    storage.validate(analysis, procedure, selection=storage._selection, joined_storage=storage._joined_storage)
    record = storage.public()
    if (not record['complete'] or record['frontiers'] or record['mutations'] or record['escapes']
            or not storage.descriptor_unchanged(binding.root)):
        raise CompilationError('native opaque write requires complete stable storage without escapes')
    matched = []
    for team in teams:
        for association in team._fixed_array_associations:
            if (association.actual is binding and association._caller.qualified == procedure
                    and _same_nodes(association._selection, selection)
                    and not any(old is association for old in matched)):
                matched.append(association)
    if not matched:
        raise CompilationError('native opaque write needs its exact incoming fixed-array association')
    incoming = _formal_array_call_rows(analysis, graph, tuple(teams))
    if not all(any(row['association_identity'] == item.identity for row in incoming) for item in matched):
        raise CompilationError('native opaque write association is absent from the same instantiated graph')
    target = tuple(item for item in reachability.accesses if item.storage.binding is binding)
    actions = {item.kind for item in target}
    if ('write' not in actions or actions - {'read', 'write', 'undefine', 'association', 'descriptor'}
            or any(item.storage.components for item in target)):
        raise CompilationError('native opaque write requires complete original primitive write effects')
    mode = 'RW' if 'read' in actions else 'W'
    item = selection[0].item
    span = getattr(item, 'fort_original_span', item.span)
    result = {
        'schema_version': NATIVE_OPAQUE_WRITE_VERSION, 'proof_role': 'original_native_opaque_array_write',
        'procedure': procedure, 'source_session': session.identity, 'source_context': context._intern_key,
        'source_inputs': analysis.inputs.identity(), 'source_graph': source_graph.identity,
        'selected_node_ids': list(selected), 'source': str(caller.scope.path),
        'first_line': span[0], 'last_line': span[1], 'original_selection': str(selection[0]),
        'original_call': str(original_call), 'resource': binding.root, 'descriptor': binding.public(),
        'bounds': list(bounds), 'element_count': count, 'storage_bytes': count*binding.kind, 'mode': mode,
        'native_graph_identity': graph.identity, 'reachability_identity': reachability.identity,
        'storage_identity': storage.identity, 'completion_identities': [p.identity for p in completions],
        'joined_team_identities': [p.identity for p in teams],
        'association_identities': [p.identity for p in matched], 'incoming_instances': incoming,
        'original_effect_inventory': [access.public() for access in reachability.accesses],
        'target_actions': sorted(actions),
        'storage_disjointness_obligations': [item.public() for item in reachability.obligations],
        'original_definition_kills': record['definition_kills'],
        'complete_write_effect_inventory': True, 'completion_scope': 'normal return only',
        'execution': 'unchanged original control after separate alias and managed host transactions',
        **dict.fromkeys(_NO_AUTHORITY, False),
    }
    encoded = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)
    analysis.verify_publication()
    _require_prerequisites(analysis, saved, same_registries=True)
    if (getattr(analysis, '_source_authority_context', None) is not context
            or getattr(analysis, '_source_session', None) is not session or context.session is not session
            or getattr(analysis, '_source_derivation_arena', None) is not arena):
        raise CompilationError('native opaque write issuing source context changed')
    return context, mode, bounds, encoded, saved


@dataclass(frozen=True)
class NativeOnlyArrayWrite:
    identity: str
    mode: str
    binding: object = field(repr=False, compare=False)
    bounds: tuple[int, int]
    _analysis: object = field(repr=False, compare=False)
    _procedure: str
    _selection: tuple = field(repr=False, compare=False)
    _reachability: object = field(repr=False, compare=False)
    _storage: object = field(repr=False, compare=False)
    _context: object = field(repr=False, compare=False)
    _arena: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _prerequisites: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def _require_handle(self, analysis):
        self._context.require(analysis)
        arena = self._context.derivation_arena(analysis)
        if not arena.accepts(self._arena):
            raise CompilationError('native opaque write belongs to a discarded generation')
        arena.require_handle(('native_opaque_write', id(analysis), self.identity), self)

    def _validate_source(self, analysis, procedure, selection, binding):
        if (type(self) is not NativeOnlyArrayWrite or self._issuer is not _ISSUER
                or analysis is not self._analysis or procedure != self._procedure or binding is not self.binding
                or not _same_nodes(selection, self._selection)
                or getattr(analysis, '_native_opaque_write_proofs', {}).get(self.identity) is not self):
            raise CompilationError('native opaque write requires its exact issued original source')
        _require_prerequisites(analysis, self._prerequisites)
        context, mode, bounds, encoded, _ = _derive(
            analysis, procedure, selection, binding, self._reachability, self._storage)
        if (context is not self._context or mode != self.mode or bounds != self.bounds
                or encoded != self._record or sha256(encoded.encode()).hexdigest() != self.identity):
            raise CompilationError('native opaque write authority record changed')

    def validate(self, analysis, procedure, selection, binding):
        attached = _attachments(self)
        operation_prerequisites = _prerequisites(analysis, self._reachability, self._storage)
        registry = getattr(analysis, '_native_opaque_write_proofs', None)
        context, session, arena = self._context, self._context.session, analysis._source_derivation_arena
        identity = self.identity
        self._require_handle(analysis)
        self._validate_source(analysis, procedure, selection, binding)
        _final_fence(self, attached, operation_prerequisites, analysis, registry, context, session, arena, identity)
        self._require_handle(analysis)
        _final_fence(self, attached, operation_prerequisites, analysis, registry, context, session, arena, identity)
        return self

    def public(self):
        return {'identity': self.identity, **json.loads(self._record)}


def _attachments(proof):
    return tuple((part.name, object.__getattribute__(proof, part.name)) for part in fields(NativeOnlyArrayWrite))


def _final_fence(proof, attached, operation_prerequisites, analysis, registry, context, session, arena, identity):
    if (any(object.__getattribute__(proof, name) is not value for name, value in attached)
            or getattr(analysis, '_native_opaque_write_proofs', None) is not registry
            or type(registry) is not dict or registry.get(identity) is not proof
            or getattr(analysis, '_source_authority_context', None) is not context
            or getattr(analysis, '_source_session', None) is not session or context.session is not session
            or getattr(analysis, '_source_derivation_arena', None) is not arena):
        raise CompilationError('native opaque write attachments changed before publication')
    _require_prerequisites(analysis, proof._prerequisites)
    _require_prerequisites(analysis, operation_prerequisites, same_registries=True)


def prove_native_opaque_write(analysis, procedure, original_selection, binding, *, reachability, storage):
    """Compose native write permission; retain every independent prerequisite."""
    context, mode, bounds, encoded, prerequisites = _derive(
        analysis, procedure, original_selection, binding, reachability, storage)
    identity = sha256(encoded.encode()).hexdigest()
    registry = getattr(analysis, '_native_opaque_write_proofs', None)
    if registry is None:
        registry = analysis._native_opaque_write_proofs = {}
    if identity in registry:
        proof = registry[identity]
        if type(proof) is not NativeOnlyArrayWrite:
            raise CompilationError('native opaque write registry contains foreign evidence')
        return proof.validate(analysis, procedure, original_selection, binding)
    if len(registry) >= source_registry_budget(analysis):
        raise CompilationError('native opaque write registry budget exhausted')
    arena = context.derivation_arena(analysis)
    proof = NativeOnlyArrayWrite(identity, mode, binding, bounds, analysis, procedure, original_selection,
        reachability, storage, context, arena, encoded, prerequisites, _ISSUER)
    attached, session = _attachments(proof), context.session
    registry[identity] = proof
    try:
        proof._validate_source(analysis, procedure, original_selection, binding)
        if context.derivation_arena(analysis) is not arena:
            raise CompilationError('native opaque write issuing arena changed')
        _final_fence(proof, attached, prerequisites, analysis, registry, context, session, arena, identity)
        arena.register_handle(('native_opaque_write', id(analysis), identity), proof)
        return proof
    except BaseException:
        if registry.get(identity) is proof:
            del registry[identity]
        raise
