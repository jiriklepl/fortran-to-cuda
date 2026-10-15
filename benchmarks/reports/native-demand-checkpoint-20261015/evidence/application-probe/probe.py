"""One diagnostic source session; no code generation or execution authority."""
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

    choices = (("dynamics::convection", "stressboundaryflux", None),
               ("dynamics::subgridstresses", "computeuvwfluxeswm", None),
               ("dynamics::subgridstresses", "boundviscosity", "viscosity"),
               ("dynamics::subgridstresses", "computetdiff", None))
    result = []
    for procedure, target, first_actual in choices:
        routine = analysis.routines[procedure]
        found = [node for node in walk(routine.execution)
                 if type(node).__name__ == "Call_Stmt" and str(node.items[0]).lower() == target
                 and (first_actual is None or str(node.items[1].items[0]).lower() == first_actual)]
        if len(found) != 1:
            raise RuntimeError("Expected one exact original query call: " + repr((procedure, target, len(found))))
        result.append((procedure, str(found[0]), (found[0],)))
    return result


def node_record(node):
    from fparser.two.utils import walk
    spans = [getattr(item, 'fort_original_span', item.span)
             for child in walk(node) if (item := getattr(child, 'item', None)) is not None]
    return {'kind': type(node).__name__, 'original_span': [min(x[0] for x in spans), max(x[1] for x in spans)]
            if spans else None, 'syntax_sha256': hashlib.sha256(str(node).encode()).hexdigest()}


def worker():
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.scopes.source import ScopeBuilder
    from compiler.frontend.resource_reachability import prove_resource_reachability
    from compiler.frontend.native_demand import prove_native_demand

    request, analysis_sources, facts, expected = inputs()
    emit("inputs_verified", inputs=len(expected), configured_sources=len(analysis_sources["entries"]),
         cases=request["configurations"])
    builder = ScopeBuilder([request["source"], *request["source_files"]], request["entry"], facts=facts,
        options=CompilerOptions(gpu_policy="auto", memory_model="scoped"),
        config=OffloadConfig(policy="auto", host_threads=4, scope_execution="reached"),
        analysis_sources=analysis_sources)
    analysis = builder.analysis
    modules = {name: {"source": str(scope.path), "sha256": sha(scope.path)}
               for name, scope in sorted(analysis.modules.items())}
    write(HERE / "loaded-modules.json", modules)
    emit("source_session", identity=analysis.session_identity, loaded_modules=len(modules),
         loaded_modules_sha256=sha(HERE / "loaded-modules.json"),
         budgets={"depth": analysis.depth_limit, "procedures": analysis.procedure_limit,
                  "operations": analysis.operation_limit})
    rows = []
    for procedure, label, selection in selections(analysis):
        scope = analysis.routines[procedure].scope
        demand = tuple(analysis._binding(scope, name) for name in ("u", "v", "w", "viscosity"))
        if any(binding is None for binding in demand):
            raise RuntimeError("Unavailable demanded original binding for " + procedure)
        row = {"procedure": procedure, "label": label, "selection": [node_record(node) for node in selection],
               "source": str(scope.path), "source_sha256": sha(scope.path),
               "demanded": [binding.public() for binding in demand]}
        emit("query_start", **row)
        started = time.monotonic()
        try:
            proof = prove_resource_reachability(analysis, procedure, selection, demand)
            proof.validate(analysis, procedure, selection, demand)
            row["reachability"] = proof.public()
        except Exception as error:
            row["reachability_error"] = {"type": type(error).__name__, "reason": str(error)}
        row["reachability_seconds"] = time.monotonic() - started
        emit("reachability", **row)
        started = time.monotonic()
        if "reachability" in row:
            try:
                native = prove_native_demand(analysis, procedure, selection, demand)
                native.validate(analysis, procedure, selection, demand)
                row["native_demand"] = native.public()
            except Exception as error:
                row["native_demand_error"] = {"type": type(error).__name__, "reason": str(error)}
        row["native_demand_seconds"] = time.monotonic() - started
        rows.append(row)
        emit("native_demand", **row)
    analysis.verify_publication()
    result = {"schema_version": 1, "kind": "bounded actual-source native demand query",
              "source_session_identity": analysis.session_identity, "cases": request["configurations"],
              "source_repair": "Same authenticated six INTENT declaration corrections as existing controls",
              "shared_configured_source_all_original_branches": True,
              "whole_interval_inventory_proved": False, "automatic_placement_proved": False,
              "generated": False, "application_executed": False, "performance_measured": False,
              "full_effect_summaries_requested": False,
              "rows": rows, "changed_inputs": [p for p, digest in expected.items() if sha(p) != digest]}
    write(HERE / "result.json", result)
    emit("publication_authentication", valid=True, completed_queries=len(rows), changed_inputs=result["changed_inputs"])


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
              'compiler_bytes_are_authoritative': True, 'default_analysis_budgets_unchanged': True, 'full_effect_summaries_requested': False,
              'generated': False, 'application_executed': False, 'performance_measured': False}
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
