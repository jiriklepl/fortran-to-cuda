"""Full application diagnostics for operation-local inspector reuse, not qualification."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT=Path('/home/jirka/research/fortran-abomination')
HERE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'benchmarks/results/owner-measurement-checkpoint-20261015-retry'
OLD=ROOT/'benchmarks/results/diffusion-complete-checkpoint-20261015'
PYTHON=ROOT/'.venv/bin/python'
GPU='GPU-d16b0d42-ee00-3af3-0c7e-d757b0cf9515'
MODES={
 'native':Path('/tmp/fortran-residency-e2e-20261014/repaired-native-checkpoint-gfortran14/build/ELMM'),
 'sgs-direct':OLD/'sgs-control/build/ELMM',
 'resident-before':PREVIOUS/'resident/candidate/build/ELMM',
 'resident-reuse':HERE/'resident/candidate/build/ELMM',
}

def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def run(*args):subprocess.run([str(PYTHON),*map(str,args)],cwd=ROOT,check=True)
def execute(case,mode,binary,out,reference,trace=False):
 args=[HERE/'run_case.py','--case',case,'--binary',binary,'--mode',mode,'--output',out,
       '--reference',reference,'--gpu',GPU,'--timeout',str(120 if case=='channel-simple' else 1200)]
 if trace:args.append('--trace')
 run(*args)

def main():
 assert read(HERE/'resident/candidate/build-generation.json')['status']=='built_not_executed'
 for binary in MODES.values():assert binary.is_file(),binary
 fields=HERE/'full-fields'
 run(HERE/'full_field_diagnostic.py','plan','--mode','resident-reuse',
     '--build-receipt',HERE/'resident/candidate/build/build-receipts.json','--output',fields,
     '--pipeline-root',HERE/'resident/pipeline')
 run(HERE/'full_field_diagnostic.py','build','--plan',fields/'plan.json')
 reference=OLD/'runs/fields/channel-simple/native'
 modes=list(MODES)
 observations=[]
 # One excluded warmup per mode, then three rotated rounds; each process is
 # complete and untraced. This is a capability checkpoint, never seven-round qualification.
 for round_number in range(4):
  order=modes[round_number%len(modes):]+modes[:round_number%len(modes)]
  for mode in order:
   output=HERE/'runs/untraced/channel-simple'/f'round-{round_number}'/mode
   execute('channel-simple',mode,MODES[mode],output,reference)
   record=read(output/'run.json')
   rows=[json.loads(line.split(' ',1)[1]) for line in (output/'stderr.log').read_text().splitlines()
         if line.startswith('FORT_SCOPE_MEASUREMENT ')]
   if mode.startswith('resident-'):
    assert len(rows)==327 and all(r['status']=='complete' for r in rows)
   else:assert not rows
   observations.append({'round':round_number,'excluded_warmup':round_number==0,'mode':mode,
     'seconds':record['diagnostic_complete_process_seconds'],'run':str(output/'run.json'),
     'binary_sha256':record['binary_sha256'],'observed_calls':len(rows),
     'owner_seconds':sum(r['seconds'] for r in rows) if rows else None,
     'qualification':False})
   write(HERE/'observations.json',observations)
 for case,steps in [('channel-simple',109),('channel',2458)]:
  output=HERE/'runs/fields'/case/'resident-reuse'
  native=OLD/'runs/fields'/case/'native'
  execute(case,'resident-reuse-fields',fields/'ELMM-fields',output,native,trace=case=='channel-simple')
  run(HERE/'full_field_diagnostic.py','compare','--native',native,'--candidate',output,
      '--reference-vtk',native/'output','--expected-steps',steps,
      '--output',HERE/(case+'-full-fields-comparison.json'),'--pipeline-root',HERE/'resident/pipeline')
  if case=='channel-simple':
   run(OLD/'transfer_ledger.py','--log',output/'stderr.log',
       '--manifest',HERE/'resident/artifacts/scope-manifest.json',
       '--runtime-source',HERE/'resident/artifacts/scoped_runtime.cu',
       '--pipeline-root',HERE/'resident/pipeline','--returncode','0','--output',HERE/'transfer-ledger.json')
 write(HERE/'checkpoint-completion.json',{'completed':True,'qualification':False,'cases':['channel-simple','channel'],
       'performance_rounds':3,'excluded_warmup_rounds':1,'automatic_residency':False})
 print(json.dumps({'completed':True,'qualification':False}),flush=True)

if __name__=='__main__':main()
