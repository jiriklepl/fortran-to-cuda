"""Bounded effect unions for one unchanged, complete native OpenMP operation.

Each original worksharing unit has its own source proof. Joining those effects
does not move or parallelize code, infer a whole overwrite, or authorize a cut
inside the team. Definition/lifetime events and calls require another proof.
"""

from copy import deepcopy
from hashlib import sha256
import json

from compiler.ir import CompilationError


def summarize(analysis, procedure, selected, completion):
    graph, identities = analysis._selected_source(procedure, selected)
    completion.validate(analysis, procedure, identities)
    if not selected or len(selected) > analysis.operation_limit:
        raise CompilationError("native atomic operation exceeds its bounded source-unit budget")
    summaries, effects = [], {}
    allowed = {"read", "write", "overwrite", "descriptor_read"}
    for node in selected:
        proof = analysis.segment_summary(procedure, (node,), capture_locals=True)
        if not proof['complete'] or proof.get('definition_changes') or proof.get('definition_diagnostics'):
            raise CompilationError("native atomic unit lacks complete effects: " + '; '.join(proof['reasons']))
        for operation in proof['operations']:
            if operation['kind'] not in allowed:
                raise CompilationError("native atomic unit contains a call, definition or control boundary")
            # Preserve all possible reads and writes. A writer must preserve
            # unmentioned data; even a local full overwrite cannot establish
            # a whole-entry definition through an enclosing original guard.
            action = 'write' if operation['kind'] == 'overwrite' else operation['kind']
            key = operation['resource'], operation['rank'], action
            effects.setdefault(key, {**operation, 'kind': action, 'guard': ()})
        if len(effects) > analysis.operation_limit:
            raise CompilationError("native atomic effect union exceeds the operation budget")
        summaries.append(proof)
    record = {'schema_version': 1, 'role': 'complete unchanged native operation',
              'structured_identity': graph.identity, 'completion_identity': completion.identity,
              'source_units': [item['demand_identity'] for item in summaries],
              'unit_operation_counts': [len(item['operations']) for item in summaries],
              'effect_count': len(effects), 'operation_limit': analysis.operation_limit,
              'definition_policy': 'no calls or lifetime events; no inferred whole overwrites',
              'communication': 'conservative whole managed resources unless independently refined'}
    identity = sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    summary = deepcopy(summaries[0])
    summary.update(operations=list(effects.values()), ordered_effects=[], definition_changes=[],
                   definition_diagnostics=[], guaranteed_whole_overwrites=[],
                   selected_node_ids=list(identities), demand_identity=identity, summary_identity=identity,
                   native_atomic=record, native_completion=completion.public(),
                   effect_composition={'available': False, 'operations': 0,
                       'reason': 'atomic native union is not an ordered call summary'})
    return summary
