"""Capped fresh SGS control generation from the shared authenticated seed."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path('/home/jirka/research/fortran-abomination')
OUT = Path(__file__).resolve().parent
CHECKPOINT = OUT.parent
PYTHON = ROOT / '.venv/bin/python'
ENTRY = 'subgrid::sgs_sigma_uniform'
OLD_FACTS = ROOT / 'benchmarks/results/generic-residency-30pct-20261009/correctness/cuda-numerical-contract-02-current-application/inputs/facts/sgs.json'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(name, value):
    (OUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')

def inventory():
    request = read(OUT / 'request.json')
    return {path: digest(path) for path in request['input_sha256']}

def prepare():
    if (OUT / 'completion.json').exists() or (OUT / 'request.json').exists():
        raise ValueError('generation attempt is immutable')
    parent = read(CHECKPOINT / 'request.json')
    analysis = read(parent['analysis_sources'])
    facts = read(OLD_FACTS)
    normalized = lambda rows: {'/'.join(Path(p).parts[-3:]).lower(): h for p,h in rows.items()}
    assert len(facts['sources']) == len(analysis['source_inputs']) == 69
    assert normalized(facts['sources']) == normalized(analysis['source_inputs'])
    assert analysis['source_inputs'] == parent['source_sha256']
    original_sources = facts['sources']
    facts['sources'] = analysis['source_inputs']
    write('capture-facts.json', facts)
    source = next(p for p in analysis['source_inputs'] if p.endswith('/ELMM/src/subgrid.f90'))
    inputs = {**analysis['source_inputs'], **analysis['dependencies'],
              **{row['path']: row['sha256'] for row in analysis['entries']}}
    for path in (Path(parent['analysis_sources']), OLD_FACTS, OUT / 'capture-facts.json',
                 CHECKPOINT / 'request.json', CHECKPOINT / 'compiler.tar', CHECKPOINT / 'pipeline.tar'):
        inputs[str(path)] = digest(path)
    for snapshot in ('compiler','pipeline'):
        for path in (CHECKPOINT / snapshot).rglob('*'):
            if path.is_file(): inputs[str(path)] = digest(path)
        assert digest(CHECKPOINT / (snapshot + '.tar')) == parent['archives'][snapshot]['sha256']
    for path, expected in inputs.items(): assert digest(path) == expected, path
    request = {'schema_version':1,'entry':ENTRY,'source':source,
        'source_files':[p for p in analysis['source_inputs'] if p != source],
        'analysis_sources':parent['analysis_sources'],'capture_facts':str(OUT/'capture-facts.json'),
        'capture_provenance':{'historical_path':str(OLD_FACTS),'historical_sha256':digest(OLD_FACTS),
            'historical_sources':original_sources,'all_69_relative_source_hashes_equal':True,
            'assertions_unchanged':True,'only_reissued_field':'sources'},
        'options':{'gpu_policy':'sections','scope_execution':'bounded','scope_transfers':'direct','host_threads':4},
        'commits':parent['commits'],'archives':parent['archives'],'configurations':parent['configurations'],
        'input_sha256':inputs,'generation_cap_seconds':300,'virtual_memory_bytes':3*1024**3,
        'cpu_affinity':[0,1,6,7],'application_built':False,'application_executed':False,'performance_measured':False}
    write('request.json',request)
    write('source-before.json',inventory())

def worker():
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    sys.path.insert(0,str(CHECKPOINT/'pipeline'))
    from elmm_pipeline.local_adapter import transform_scopes
    request=read(OUT/'request.json')
    started=time.monotonic()
    result=transform_scopes(Path(request['source']),CHECKPOINT/'compiler',PYTHON,OUT/'artifacts',
        entry=ENTRY,source_files=[Path(p) for p in request['source_files']],
        capture_facts=read(request['capture_facts']),gpu_options=request['options'],
        analysis_sources=Path(request['analysis_sources']))
    manifest=result['scope_manifest']
    assert manifest['scope_count'] > 0
    assert set(manifest['sources']) == {request['source']}
    assert result['gpu_sources']
    summary={'elapsed_seconds':time.monotonic()-started,'scope_count':manifest['scope_count'],
        'gpu_leaves':[s.get('gpu_leaves') for s in manifest['scopes']],
        'source_replacements':list(manifest['sources']),
        'runtime_id':manifest.get('runtime',{}).get('runtime_id'),
        'manifest_sha256':digest(OUT/'artifacts/scope-manifest.json'),
        'cuda_units':len(result['gpu_sources']),
        'application_built':False,'application_executed':False,'performance_measured':False}
    write('worker-result.json',summary)
    print(json.dumps(summary),flush=True)

def parent():
    prepare()
    command=['taskset','-c','0,1,6,7',str(PYTHON),str(OUT/'run.py'),'--worker']
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE'}
    write('command.json',{'argv':command,'cwd':str(ROOT),'environment_overrides':{
        k:env[k] for k in ('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OMP_DYNAMIC')},
        'cap_seconds':300,'runner_sha256':digest(__file__)})
    started=time.monotonic()
    with (OUT/'generation.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('running.json',{'pid':process.pid,'started_epoch':time.time(),'cap_seconds':300})
        print(json.dumps({'phase':'running','pid':process.pid,'cap_seconds':300,'output':str(OUT)}),flush=True)
        timeout=False
        try: code=process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timeout=True
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
    after=inventory();write('source-after.json',after)
    before=read(OUT/'source-before.json')
    changed=sorted(p for p in before.keys()|after.keys() if before.get(p)!=after.get(p))
    finish={'exit_code':code,'timed_out':timeout,'elapsed_seconds':time.monotonic()-started,
        'changed_inputs':changed,'source_unchanged':not changed,
        'manifest_exists':(OUT/'artifacts/scope-manifest.json').exists(),
        'worker_result_exists':(OUT/'worker-result.json').exists(),'log_sha256':digest(OUT/'generation.log')}
    write('completion.json',finish)
    print(json.dumps(finish),flush=True)
    return code

if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:sys.exit(parent())
