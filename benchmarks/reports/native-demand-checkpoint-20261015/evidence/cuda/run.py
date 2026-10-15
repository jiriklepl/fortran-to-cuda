from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path('/home/jirka/research/fortran-abomination')
HERE=Path(__file__).resolve().parent

def inventory():
 return {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'compiler').rglob('*.py')) if 'tests' not in p.relative_to(ROOT/'compiler').parts}
def write(name,value): (HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
before=inventory();write('compiler-before.json',before)
command=['taskset','-c','4,5,10,11',str(ROOT/'.venv/bin/python'),'-m','pytest','-q','compiler/tests/test_native_demand_cuda.py','--basetemp',str(HERE/'pytest')]
environment={**os.environ,'CUDA_VISIBLE_DEVICES':'GPU-d16b0d42-ee00-3af3-0c7e-d757b0cf9515','OMP_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','OMP_SCHEDULE':'static','PYTHONDONTWRITEBYTECODE':'1'}
record={'command':command,'cwd':str(ROOT),'compiler_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'environment':{k:environment[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OMP_DYNAMIC','OMP_SCHEDULE')},'test_sha256':sha256((ROOT/'compiler/tests/test_native_demand_cuda.py').read_bytes()).hexdigest(),'kind':'corrected executable CUDA/host integration check; no calibration or performance claim'}
write('launch.json',record);start=time.monotonic()
with (HERE/'pytest.log').open('w') as log: result=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=300,check=False)
after=inventory();write('compiler-after.json',after)
record.update(exit_code=result.returncode,seconds=time.monotonic()-start,changed_compiler_files=[p for p in sorted(before.keys()|after.keys()) if before.get(p)!=after.get(p)])
write('completion.json',record);print(json.dumps(record),flush=True)
