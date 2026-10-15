"""One capped complete-diffusion generation diagnostic; no application run."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path('/home/jirka/research/fortran-abomination')
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'benchmarks/results/checked-plan-application-20261014'
PYTHON = ROOT / '.venv/bin/python'
ENTRY = 'momentumdiffusion::momentumdiffusion_nobranch_2ord'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')

def git(cwd, *args):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()

def inventory():
    result = {}
    for cwd, prefix, selection in ((ROOT, '', ['compiler']), (ROOT/'elmm-pipeline', 'elmm-pipeline/', [])):
        for name in git(cwd, 'ls-files', *selection).splitlines():
            path = cwd / name
            if path.is_file():
                result[prefix + name] = digest(path)
    prior = json.loads((BASE / 'automatic-generation/completion.json').read_text())['input_sha256']
    assert len(prior) == 178
    for name, expected in prior.items():
        result[name] = digest(name)
        assert result[name] == expected, name
    return result

def prepare():
    original = json.loads((BASE/'request.json').read_text())
    current_facts = json.loads(Path(original['capture_facts']).read_text())
    configuration = json.loads(Path(original['analysis_sources']).read_text())
    assert current_facts['sources'] == configuration['source_inputs'] == original['source_inputs']
    historical = ROOT/'benchmarks/results/generic-residency-30pct-20261009/m1/joined-team-application-source-05/artifacts/scope-manifest.json'
    historical_manifest = json.loads(historical.read_text())
    source = next(path for path in original['source_inputs'] if path.endswith('/momentum_diffusion.f90'))
    old_source = historical_manifest['scopes'][0]['path']
    assert historical_manifest['source_inputs'][old_source] == original['source_inputs'][source]
    mapping = {f'argument::{name}': f'argument::{name}' for name in ('u','v','w')}
    mapping.update({f'argument::{name}2':f'argument::{name}star' for name in ('u','v','w')})
    mapping.update({name:name for name in ('parameters::viscosity','tiling::tilenx','tiling::tileny','tiling::tilenz')})
    facts = {key:value for key,value in current_facts.items() if key != 'captures'}
    facts['captures'] = {formal: current_facts['captures'][actual] for formal,actual in mapping.items()}
    write('capture-facts.json', facts)
    revisions = {'compiler':git(ROOT,'rev-parse','HEAD'),'pipeline':git(ROOT/'elmm-pipeline','rev-parse','HEAD')}
    assert not git(ROOT,'diff','--name-only','HEAD','--','compiler')
    assert not git(ROOT/'elmm-pipeline','diff','--name-only','HEAD')
    for name,cwd,paths in [('compiler',ROOT,['compiler']),('pipeline',ROOT/'elmm-pipeline',[])]:
        archive = OUT/(name+'.tar')
        with archive.open('wb') as stream:
            subprocess.run(['git','archive','HEAD',*paths],cwd=cwd,stdout=stream,check=True)
        destination = OUT/name
        destination.mkdir(exist_ok=False)
        with tarfile.open(archive) as bundle:
            bundle.extractall(destination,filter='data')
    request = {
        'schema_version':1, 'entry':ENTRY, 'source':source,
        'source_files':[p for p in original['source_inputs'] if p != source],
        'capture_facts':str(OUT/'capture-facts.json'),
        'analysis_sources':original['analysis_sources'],
        'original_request':str(BASE/'request.json'),
        'original_request_sha256':digest(BASE/'request.json'),
        'capture_mapping':mapping,
        'capture_mapping_authority':'Original Convection call passes Ustar,Vstar,Wstar,U,V,W to diffusion U2,V2,W2,U,V,W. Values are copied from existing frozen caller facts; no definition or allocation facts added.',
        'captured_input_scope':'Diagnostic complete invocation; source/configuration unchanged. This does not prove caller-suffix admission.',
        'historical_recipe':'/tmp/m1-joined-team-application-source.py',
        'historical_recipe_sha256':digest('/tmp/m1-joined-team-application-source.py'),
        'historical_manifest':str(historical), 'historical_manifest_sha256':digest(historical),
        'source_sha256':original['source_inputs'], 'configurations':original['configurations'],
        'options':{'gpu_policy':'sections','scope_execution':'reached','scope_transfers':'direct','host_threads':4,'scope_measurement':'resident'},
        'commits':revisions,'archives':{name:{'sha256':digest(OUT/(name+'.tar'))} for name in ('compiler','pipeline')},
        'generation_cap_seconds':300,'virtual_memory_bytes':3*1024**3,'cpu_affinity':[4,5,10,11],
        'application_built':False,'application_executed':False,'performance_measured':False,
    }
    write('request.json',request)
    write('source-before.json',inventory())

def worker():
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    sys.path.insert(0,str(OUT/'pipeline'))
    from elmm_pipeline.local_adapter import transform_scopes
    request=json.loads((OUT/'request.json').read_text())
    started=time.monotonic()
    result=transform_scopes(Path(request['source']),OUT/'compiler',PYTHON,OUT/'artifacts',entry=ENTRY,
        source_files=[Path(p) for p in request['source_files']],
        capture_facts=json.loads(Path(request['capture_facts']).read_text()),
        gpu_options=request['options'],analysis_sources=Path(request['analysis_sources']))
    manifest=result['scope_manifest']
    teams=[segment['operations']['joined_team'] for owner in manifest['scopes']
        for segment in owner.get('planning_segments',[]) if 'joined_team' in segment['operations']]
    summary={'elapsed_seconds':time.monotonic()-started,'scope_count':manifest['scope_count'],
        'joined_teams':len(teams),'numerical_units':[len(t['numerical_units']) for t in teams],
        'gpu_leaves':[s['gpu_leaves'] for s in manifest['scopes']],
        'owner_boundaries':[s.get('boundaries',[]) for s in manifest['scopes']],
        'diagnostic_boundaries':manifest['boundaries'],'manifest_sha256':digest(OUT/'artifacts/scope-manifest.json'),
        'automatic_estimate_available':manifest.get('automatic_estimate_available'),
        'application_built':False,'application_executed':False,'performance_measured':False}
    write('worker-result.json',summary)
    print(json.dumps(summary),flush=True)

def parent():
    prepare()
    command=['taskset','-c','4,5,10,11',str(PYTHON),str(OUT/'run.py'),'--worker']
    environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE'}
    write('command.json',{'argv':command,'cwd':str(ROOT),'environment_overrides':{
        k:environment[k] for k in ('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OMP_DYNAMIC')},
        'cap_seconds':300,'runner_sha256':digest(__file__)})
    started=time.monotonic()
    with (OUT/'generation.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write('running.json',{'pid':process.pid,'started_epoch':time.time(),'cap_seconds':300})
        print(json.dumps({'phase':'running','pid':process.pid,'cap_seconds':300,'output':str(OUT)}),flush=True)
        timeout=False
        try:
            code=process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timeout=True
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
            code=124
    after=inventory();write('source-after.json',after)
    before=json.loads((OUT/'source-before.json').read_text())
    changed=sorted(p for p in before.keys()|after.keys() if before.get(p)!=after.get(p))
    finish={'exit_code':code,'timed_out':timeout,'elapsed_seconds':time.monotonic()-started,
        'changed_source_inputs_or_repositories':changed,'source_unchanged':not changed,
        'manifest_exists':(OUT/'artifacts/scope-manifest.json').exists(),
        'worker_result_exists':(OUT/'worker-result.json').exists(),'log_sha256':digest(OUT/'generation.log'),
        'compiler_snapshot_sha256':digest(OUT/'compiler.tar'),'pipeline_snapshot_sha256':digest(OUT/'pipeline.tar')}
    write('completion.json',finish)
    print(json.dumps(finish),flush=True)
    return code

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:worker()
    else:sys.exit(parent())
