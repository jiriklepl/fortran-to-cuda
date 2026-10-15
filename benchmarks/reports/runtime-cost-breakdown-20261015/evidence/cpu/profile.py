"""One capped full-duration profiler run of the existing diffusion diagnostic."""
import hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/jirka/research/fortran-abomination')
OUT=Path(__file__).resolve().parent
BASE=ROOT/'benchmarks/results/owner-measurement-checkpoint-20261015-retry'
BINARY=BASE/'resident/candidate/build/ELMM'
CASE=ROOT/'benchmarks/results/checked-plan-application-20261014/cases/channel-simple'
sys.path.insert(0,str(ROOT/'elmm-pipeline'))
from elmm_pipeline.campaigns.run import environment
from elmm_pipeline.validation import validate_log,compare_vtk,read_vtk

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,data):(OUT/name).write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')

def main():
    run=OUT/'case';shutil.copytree(CASE,run)
    source={str(p.relative_to(CASE)):sha(p) for p in sorted(CASE.rglob('*')) if p.is_file()}
    assert len(os.sched_getaffinity(0))==4
    env=environment(4,traced=False);env.pop('OMP_PLACES',None);env.pop('GOMP_CPU_AFFINITY',None)
    env.update(OMP_PROC_BIND='false',CUDA_VISIBLE_DEVICES='GPU-d16b0d42-ee00-3af3-0c7e-d757b0cf9515')
    command=['/usr/local/cuda/bin/nsys','profile','--trace=cuda','--sample=process-tree','--backtrace=dwarf','--cpuctxsw=process-tree',
        '--export=sqlite','--force-overwrite=false','--output='+str(OUT/'profile'),str(BINARY),"output_dir='output/'"]
    record={'schema_version':1,'kind':'separate diagnostic CPU and CUDA profile','qualification':False,
        'performance_measured':False,'automatic_placement_claim':False,'argv':command,'cwd':str(run),
        'binary':str(BINARY),'binary_sha256':sha(BINARY),'configuration':source,'configuration_overrides':{},
        'threads':4,'affinity':sorted(os.sched_getaffinity(0)),'timeout_seconds':120,
        'environment':{k:v for k,v in env.items() if k.startswith(('OMP_','GOMP_','CUDA_','FORT_'))},
        'compiler_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'pipeline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'elmm-pipeline',text=True).strip(),
        'profiler_version':subprocess.check_output(['/usr/local/cuda/bin/nsys','--version'],text=True).strip(),
        'runner_sha256':sha(Path(__file__)),'completed':False}
    write('command.json',record)
    start=time.monotonic()
    with (OUT/'stdout.log').open('w') as stdout,(OUT/'stderr.log').open('w') as stderr:
        p=subprocess.Popen(command,cwd=run,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        record['pid']=p.pid;write('running.json',record)
        try:record['exit_code']=p.wait(timeout=120);record['timed_out']=False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            record.update(exit_code=124,timed_out=True)
    record['profile_process_seconds']=time.monotonic()-start
    try:
        assert record['exit_code']==0
        assert sha(BINARY)==record['binary_sha256']
        assert all(sha(run/k)==v and sha(CASE/k)==v for k,v in source.items())
        record['completion']=validate_log((OUT/'stdout.log').read_text(),expected_completed_steps=109)
        assert record['completion']['natural_end']
        record['vtk']={name:compare_vtk(read_vtk(BASE/'runs/untraced/native/output'/name),read_vtk(run/'output'/name)) for name in ('out.vtk','avg.vtk')}
        assert all(r['passed'] for r in record['vtk'].values())
        record['artifacts']={p.name:sha(p) for p in OUT.glob('profile.*') if p.suffix in ('.sqlite','.nsys-rep')}
        assert len(record['artifacts'])==2
        record['completed']=True
    except BaseException as e:record['failure']=str(e)
    write('completion.json',record)
    print(json.dumps({k:record.get(k) for k in ('completed','exit_code','profile_process_seconds','completion','failure')}))
    return 0 if record['completed'] else 1

if __name__=='__main__':raise SystemExit(main())
