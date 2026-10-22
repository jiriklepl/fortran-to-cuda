"""Prepared private one-object TOP reduction diagnostic; run only on parent lease."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import hashlib

ROOT=Path('/tmp/fortran-native-top-reference-serial-01')
PLAN=ROOT/'build-plan.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');args=p.parse_args()
plan=json.loads(PLAN.read_text())
def check():
    for path,expected in plan['original_inputs_sha256'].items():
        if sha(path)!=expected:raise RuntimeError('changed original private-build input: '+path)
check()
if not args.execute:
    print('CHECKED ONLY; no compilation',PLAN,sha(PLAN));raise SystemExit()
none=Path('/tmp/fortran-fftw-native-no-reference-full-01/launch-execution-17.json')
control=json.loads(none.read_text())
if control['status']!='completed_controlled_diagnostic' or not control['all20_fields_byte_identical'] or not control['strict_full20_passed'] or not control['vtk_validation']['passed']:
    raise RuntimeError('required full supported NONE native control not byte-identical/strict/VTK passed')
receipt=ROOT/'build-receipt.json'
if receipt.exists() or (ROOT/'build').exists():raise RuntimeError('preserve existing private diagnostic build')
(ROOT/'build/modules').mkdir(parents=True)
result={'schema_version':1,'status':'running','diagnostic_only':True,'official_qualification':False,'performance_sample':False,
        'build_plan_sha256':sha(PLAN),'driver_sha256':sha(__file__),'none_control':{'path':str(none),'sha256':sha(none)},'commands':[]}
start=time.monotonic()
try:
    for name in ['compile_recipe','link_recipe']:
        check();recipe=plan[name];env=dict(os.environ)
        for key,value in recipe['environment'].items():
            if value is None:env.pop(key,None)
            else:env[key]=value
        timeout=min(plan['per_command_cap_seconds'],plan['outer_phase_cap_seconds']-(time.monotonic()-start))
        if timeout<=0:raise TimeoutError('private diagnostic outer240 cap')
        log=ROOT/(name+'.log')
        with log.open('w') as target:
            process=subprocess.run(recipe['argv'],cwd=recipe['cwd'],env=env,stdout=target,stderr=subprocess.STDOUT,timeout=timeout)
        result['commands'].append({'recipe':name,'argv':recipe['argv'],'returncode':process.returncode,'log':str(log),'log_sha256':sha(log)})
        if process.returncode:raise RuntimeError('private diagnostic command failed: '+name)
    check()
    result.update(status='built_private_diagnostic_only',outputs_sha256={str(p):sha(p) for p in (ROOT/'build').rglob('*') if p.is_file()})
except BaseException as error:
    result.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
finally:
    result['diagnostic_build_wall_seconds']=time.monotonic()-start
    receipt.write_text(json.dumps(result,indent=2)+'\n')
print('PRIVATE TOP BUILT',receipt,sha(receipt))
