from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
from compiler.tests.test_native_demand_cuda import DRIVER
ROOT=Path('/home/jirka/research/fortran-abomination')
HERE=Path(__file__).resolve().parent
ORIGINAL=HERE/'pytest/test_3d_native_demand_keeps_cu0'
BUILD=ORIGINAL/'build'
OUT=HERE/'driver-resume';OUT.mkdir(exist_ok=True)
commands=json.loads((ORIGINAL/'commands.json').read_text())
old=BUILD/'driver.f90';(OUT/'original-driver.f90').write_bytes(old.read_bytes());old.write_text(DRIVER)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'GPU-d16b0d42-ee00-3af3-0c7e-d757b0cf9515','OMP_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','OMP_SCHEDULE':'static'}
objects=[row['argv'][-1] for row in commands]
link=[shutil.which('nvcc'),'-ccbin',shutil.which('g++-14'),'-Xcompiler=-fopenmp',*objects,'-lgfortran','-o',str(BUILD/'verify')]
records=[]
for name,argv,extra in [('driver',commands[-1]['argv'],{}),('link',link,{}),('traced',[str(BUILD/'verify')],{'FORT_RUNTIME_TRACE':'1'}),('untraced',[str(BUILD/'verify')],{}),('fallback',[str(BUILD/'verify')],{'FORT_RUNTIME_TRACE':'1','CUDA_VISIBLE_DEVICES':''})]:
 result=subprocess.run(argv,cwd=BUILD,env={**env,**extra},capture_output=True,text=True,timeout=120)
 (OUT/(name+'.stdout')).write_text(result.stdout);(OUT/(name+'.stderr')).write_text(result.stderr)
 records.append({'name':name,'argv':argv,'cwd':str(BUILD),'environment':extra,'exit_code':result.returncode})
 (OUT/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 print(name,result.returncode,result.stdout,result.stderr[-1000:] if result.returncode else '',flush=True)
 if result.returncode: raise SystemExit(result.returncode)
 if name in {'traced','untraced','fallback'}:
  expected=(BUILD/'native-fields.bin').read_bytes();candidate=(BUILD/'candidate-fields.bin').read_bytes()
  assert expected==candidate
  print('fields',len(expected),sha256(expected).hexdigest(),flush=True)
