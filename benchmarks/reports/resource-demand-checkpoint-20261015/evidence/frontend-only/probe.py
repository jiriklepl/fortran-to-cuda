"""One unsealed SourceEffects-only diagnostic; no generation or execution authority."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/jirka/research/fortran-abomination')
HERE = Path(__file__).resolve().parent
SEED = ROOT / 'benchmarks/results/checked-plan-application-20261014'
START = time.monotonic()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def emit(phase, **values):
    print(json.dumps({'phase': phase, 'seconds': time.monotonic() - START, **values}), flush=True)


def inputs():
    request = json.loads((SEED / 'request.json').read_text())
    bundle = json.loads((SEED / 'input-bundle.json').read_text())
    analysis = json.loads(Path(request['analysis_sources']).read_text())
    facts = json.loads(Path(request['capture_facts']).read_text())
    expected = {**analysis['source_inputs'], **analysis['dependencies'],
                **{row['path']: row['sha256'] for row in analysis['entries']}}
    expected[request['capture_facts']] = bundle['capture_facts_sha256']
    expected[request['analysis_sources']] = bundle['analysis_sha256']
    for case in bundle['cases'].values():
        expected.update({str(Path(case['path']) / name): digest for name, digest in case['files'].items()})
    for name in ('request.json', 'input-bundle.json'):
        expected[str(SEED / name)] = sha(SEED / name)
    changed = [path for path, digest in expected.items() if sha(path) != digest]
    if changed or facts['sources'] != analysis['source_inputs']:
        raise RuntimeError('Authenticated source inputs differ: ' + repr(changed))
    return request, analysis, facts, expected


def inventory():
    return {str(path.relative_to(ROOT)): sha(path)
            for path in sorted((ROOT / 'compiler').rglob('*.py'))
            if 'tests' not in path.relative_to(ROOT / 'compiler').parts}


def selections(analysis):
    from fparser.two.utils import walk
    from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence

    specs = (
        ('dynamics::subgridstresses', {
            'computeuvwfluxeswm', 'boundviscosity', 'computetdiff'}),
        ('dynamics::convection', {
            'cdu', 'cdv', 'cdw', 'cd4divu', 'cd4divv', 'cd4divw',
            'filterustar', 'coriolisforce', 'buoyancyforce', 'resistanceforce',
            'linear_forcing', 'rayleigh_damping_uvw', 'stressboundaryflux',
            'momentumdiffusion_nobranch_2ord', 'momentumdiffusion_variable_z_nobranch_2ord',
            'momentumdiffusion_4ord_5point'}),
    )
    result = []
    for procedure, names in specs:
        routine = analysis.routines[procedure]
        for node in walk(routine.execution):
            if type(node).__name__ == 'Call_Stmt' and str(node.items[0]).lower() in names:
                result.append((procedure, str(node), (node,)))
    sgs = analysis.routines['dynamics::subgridstresses']
    for node in sgs.execution.content:
        if type(node).__name__.endswith('Do_Construct') and 'tibpoint_viscosity' in str(node).lower():
            result.append((sgs.qualified, 'IBM viscosity interpolation loop', (node,)))
    convection = analysis.routines['dynamics::convection']
    for node in convection.execution.content:
        if type(node).__name__ == 'If_Construct' and 'rk_stage > 1' in str(node.content[0]).lower():
            result.append((convection.qualified, 'stage initialization branch', (node,)))
    for group in grouped_original_sequence(convection.execution.content, {}):
        if isinstance(group, NativeGroupCandidate) and 'beta(rk_stage)' in ''.join(
                str(node) for node in group.original_nodes).replace(' ', '').lower():
            result.append((convection.qualified, 'joined accumulation team', group.original_nodes))
    # The simple genuinely disjoint demand is first; no source operation is reordered.
    result.sort(key=lambda row: ('stressboundaryflux' not in row[1].lower(), specs.index(
        next(spec for spec in specs if spec[0] == row[0]))))
    return result


def node_record(node):
    from fparser.two.utils import walk
    spans = [getattr(item, 'fort_original_span', item.span)
             for child in walk(node) if (item := getattr(child, 'item', None)) is not None]
    return {'kind': type(node).__name__, 'original_span': [min(x[0] for x in spans), max(x[1] for x in spans)]
            if spans else None, 'syntax_sha256': hashlib.sha256(str(node).encode()).hexdigest()}


def worker():
    from compiler.frontend.source_effects import SourceEffects
    from compiler.frontend.resource_reachability import prove_resource_reachability

    request, analysis_sources, facts, expected = inputs()
    emit('inputs_verified', input_count=len(expected), configured_sources=len(analysis_sources['entries']))
    analysis = SourceEffects([request['source'], *request['source_files']], analysis_sources=analysis_sources)
    emit('source_effects_constructed', sealed_session=False, identity=analysis.session_identity,
         budgets={'depth': analysis.depth_limit, 'procedures': analysis.procedure_limit,
                  'operations': analysis.operation_limit})
    chosen = selections(analysis)
    rows = []
    for procedure, label, selection in chosen:
        scope = analysis.routines[procedure].scope
        demand = tuple(analysis._binding(scope, name) for name in ('u', 'v', 'w', 'viscosity'))
        if any(binding is None for binding in demand):
            raise RuntimeError('Unavailable demanded source binding for ' + procedure)
        record = {'procedure': procedure, 'label': label, 'selection': [node_record(node) for node in selection],
                  'source': str(scope.path), 'source_sha256': analysis.sources[str(scope.path)],
                  'demanded': [binding.public() for binding in demand]}
        started = time.monotonic()
        emit('resource_demand_start', procedure=procedure, label=label)
        try:
            token = prove_resource_reachability(analysis, procedure, tuple(selection), demand)
            token.validate(analysis, procedure, tuple(selection), demand)
            record['demand_result'] = token.public()
        except Exception as error:
            record['demand_error'] = {'type': type(error).__name__, 'reason': str(error)}
        record['demand_seconds'] = time.monotonic() - started
        rows.append(record)
        emit('resource_demand', **record)
    # Compare only after every demanded classification has been attempted.
    for record, (_procedure, _label, selection) in zip(rows, chosen, strict=True):
        started = time.monotonic()
        if time.monotonic() - START >= 45:
            record['full_summary'] = {'status': 'unmeasured', 'reason': 'Reserve remaining cap for final source authentication'}
            emit('full_summary_unmeasured', procedure=record['procedure'], label=record['label'])
            continue
        try:
            summary = analysis.segment_summary(record['procedure'], selection)
            baseline = {key: summary.get(key) for key in ('complete', 'reasons', 'summary_identity',
                'native_completion', 'native_sections', 'definition_diagnostics')}
        except Exception as error:
            baseline = {'error_type': type(error).__name__, 'reason': str(error)}
        record['full_summary'] = baseline
        record['full_summary_seconds'] = time.monotonic() - started
        emit('full_summary', procedure=record['procedure'], label=record['label'],
             result=baseline, elapsed_seconds=record['full_summary_seconds'])
    analysis.verify_source()
    analysis.verify_publication()
    result = {'schema_version': 1, 'kind': 'unsealed source-bound frontend-only resource-demand diagnostic',
              'immutable_application_pipeline': False, 'source_session_sealed': False,
              'capture_facts_validated_but_not_applied': True,
              'source_session_identity': analysis.session_identity, 'cases': request['configurations'],
              'source_repair': 'Same authenticated six INTENT declaration corrections as existing controls',
              'cross_procedure_ownership_proved': False, 'execution_authorized': False,
              'generated': False, 'application_executed': False, 'performance_measured': False,
              'rows': rows, 'changed_inputs': [p for p, digest in expected.items() if sha(p) != digest]}
    write(HERE / 'result.json', result)
    emit('publication_authentication', valid=True, selections=len(rows), changed_inputs=result['changed_inputs'])


def main():
    request, _analysis, _facts, expected = inputs()
    before = inventory()
    write(HERE / 'input-inventory.json', expected)
    write(HERE / 'compiler-before.json', before)
    command = ['timeout', '--signal=TERM', '--kill-after=5s', '60s', 'taskset', '-c', '4,5,10,11',
               str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__).resolve()), '--worker']
    record = {'command': command, 'cwd': str(ROOT), 'cap_seconds': 60, 'termination_grace_seconds': 5,
              'runner_sha256': sha(__file__), 'request': str(SEED / 'request.json'),
              'request_sha256': sha(SEED / 'request.json'), 'input_count': len(expected),
              'compiler_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'pipeline_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'elmm-pipeline', text=True).strip(),
              'compiler_bytes_are_authoritative': True, 'source_session_sealed': False,
              'setup': 'SourceEffects directly; no ScopeBuilder, no source-session sealing', 'default_analysis_budgets_unchanged': True}
    write(HERE / 'launch.json', record)
    started = time.monotonic()
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT)}
    with (HERE / 'output.jsonl').open('w') as output:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=False)
    after = inventory()
    write(HERE / 'compiler-after.json', after)
    record.update(exit_code=completed.returncode, elapsed_seconds=time.monotonic() - started,
                  changed_compiler_files=[p for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)],
                  changed_inputs=[p for p, digest in expected.items() if sha(p) != digest],
                  output_sha256=sha(HERE / 'output.jsonl'), result_written=(HERE / 'result.json').exists())
    write(HERE / 'completion.json', record)
    print(json.dumps(record), flush=True)
    sys.exit(completed.returncode)


if __name__ == '__main__':
    worker() if '--worker' in sys.argv else main()
