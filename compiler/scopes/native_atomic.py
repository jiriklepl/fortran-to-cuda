"""Bounded effect unions for one unchanged, complete native OpenMP operation.

Each original worksharing unit has its own source proof. Joining those effects
does not move or parallelize code, infer a whole overwrite, or authorize a cut
inside the team. Definition/lifetime events and calls require another proof.
"""

import json
from copy import deepcopy
from hashlib import sha256

from compiler.ir import CompilationError


def summarize(analysis, procedure, selected, completion):
    graph, identities = analysis._selected_source(procedure, selected)
    completion.validate(analysis, procedure, identities)
    group = graph.native_group_for_selection(selected)
    if group is not None:
        graph, identities, group, summaries = analysis.native_group_summaries(procedure, selected, completion)
    else:
        if not selected or len(selected) > analysis.operation_limit:
            raise CompilationError("native atomic operation exceeds its bounded source-unit budget")
        summaries = tuple(analysis.segment_summary(procedure, (node,), capture_locals=True) for node in selected)
    if not summaries or len(summaries) > analysis.operation_limit:
        raise CompilationError("native atomic operation exceeds its bounded source-unit budget")
    effects, environment, predicates = {}, [], []
    allowed = {"read", "write", "overwrite", "descriptor_read"}
    for proof in summaries:
        if not proof['complete'] or proof.get('definition_changes') or proof.get('definition_diagnostics'):
            raise CompilationError("native atomic unit lacks complete effects: " + '; '.join(proof['reasons']))
        for requirement in proof.get('native_predicate_requirements', ()):
            token = analysis._native_predicates.get(requirement.get('proof_identity'))
            if token is None:
                raise CompilationError("native atomic predicate lacks registered original source authority")
            token.validate(analysis, procedure, token._expression)
            if {key: value for key, value in requirement.items() if key != 'guard_frames'} != token.public():
                raise CompilationError("native atomic predicate differs from original source authority")
            predicates.append(deepcopy(requirement))
            if len(predicates) > analysis.operation_limit:
                raise CompilationError("native atomic predicate union exceeds the operation budget")
        for operation in proof['operations']:
            if operation['kind'] == 'native_environment':
                token = analysis._native_environments.get(operation.get('proof_identity'))
                if token is None:
                    raise CompilationError("native atomic environment lacks registered original source authority")
                token.validate(analysis, procedure, token._call)
                if operation.get('proof') != token.public() or operation.get('effects') != token.public()['effects']:
                    raise CompilationError("native atomic environment effects differ from original source authority")
                environment.append((operation, token))
                continue
            if operation['kind'] not in allowed:
                raise CompilationError("native atomic unit contains a call, definition or control boundary")
            # Preserve all possible reads and writes. A writer must preserve
            # unmentioned data; even a local full overwrite cannot establish
            # a whole-entry definition through an enclosing original guard.
            action = 'write' if operation['kind'] == 'overwrite' else operation['kind']
            key = operation['resource'], operation['rank'], action
            effects.setdefault(key, {**operation, 'kind': action, 'guard': ()})
        if len(effects) + len(environment) > analysis.operation_limit:
            raise CompilationError("native atomic effect union exceeds the operation budget")
    expected = [item['proof_identity'] for item in completion.public().get('native_environment_operations', ())]
    if [token.identity for _operation, token in environment] != expected:
        raise CompilationError("native atomic environment order lacks whole original completion authority")
    record = {'schema_version': 1, 'role': 'complete unchanged native operation',
              'structured_identity': graph.identity, 'completion_identity': completion.identity,
              'source_units': [item['demand_identity'] for item in summaries],
              'unit_operation_counts': [len(item['operations']) for item in summaries],
              'effect_count': len(effects), 'operation_limit': analysis.operation_limit,
              'definition_policy': 'no unproved calls or lifetime events; no inferred whole overwrites',
              'communication': 'conservative whole managed resources unless independently refined'}
    if group is not None:
        record['deferred_native_group'] = group.public()
    if predicates:
        record.update(native_predicate_requirements=predicates, native_only=True,
                      internal_cuts_authorized=False, numerical_lowering_authorized=False)
    ordered_environment = []
    if environment:
        record.update(native_environment_operations=[token.public() for _operation, token in environment],
                      environment_operation_count=len(environment),
                      requires_completed_device_work=True, native_only=True, internal_cuts_authorized=False)
        for operation, token in environment:
            ordered_environment.extend({**effect, 'guard': operation.get('guard', ()),
                                        'native_environment_identity': token.identity}
                                       for effect in token.public()['effects'])
    identity = sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    summary = deepcopy(summaries[0])
    summary.update(operations=[*effects.values(), *(operation for operation, _token in environment)],
                   ordered_effects=ordered_environment, definition_changes=[],
                   definition_diagnostics=[], guaranteed_whole_overwrites=[],
                   native_predicate_requirements=predicates,
                   selected_node_ids=list(identities), demand_identity=identity, summary_identity=identity,
                   native_atomic=record, native_completion=completion.public(),
                   effect_composition={'available': False, 'operations': len(ordered_environment),
                       'reason': 'managed effects are an unordered atomic union; native environment effects retain source order'})
    if predicates:
        summary['cloneable'] = False
    return summary
