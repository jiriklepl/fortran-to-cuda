import hashlib, json, os, pathlib, subprocess, sys, tarfile, time
ROOT=pathlib.Path('/home/jirka/research/fortran-abomination')
OUT=ROOT/'benchmarks/results/application-path-20261014'
OUT.mkdir(exist_ok=False)
compiler=OUT/'compiler'; compiler.mkdir()
with (OUT/'compiler.tar').open('wb') as f:
 subprocess.run(['git','-C',str(ROOT),'archive','HEAD','compiler'],stdout=f,check=True)
with tarfile.open(OUT/'compiler.tar') as f: f.extractall(compiler,filter='data')
sys.path.insert(0,str(ROOT/'elmm-pipeline'))
from elmm_pipeline.local_adapter import transform_scopes
analysis_path=ROOT/'benchmarks/results/automatic-residency-20261014/preflight/inputs/analysis-sources.json'
analysis=json.loads(analysis_path.read_text())
digest=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for p, h in {**analysis['source_inputs'],**analysis['dependencies']}.items(): assert digest(p)==h,p
for row in analysis['entries']:assert digest(row['path'])==row['sha256'],row['path']
facts_path=ROOT/'benchmarks/results/generic-residency-30pct-20261009/correctness/original-convection-application-01/convection-partial-viscosity-facts-02.json'
facts=json.loads(facts_path.read_text());facts['sources']=analysis['source_inputs']
source=ROOT/'elmm-pipeline/.cache/repos/elmm/src/dynamics.f90'
request={'parent_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'pipeline_commit':subprocess.check_output(['git','-C',str(ROOT/'elmm-pipeline'),'rev-parse','HEAD'],text=True).strip(),'source':str(source),'source_inputs':analysis['source_inputs'],'analysis_sources':str(analysis_path),'original_capture_facts':str(facts_path),'original_capture_sha256':digest(facts_path),'capture_facts':facts,'entry':'dynamics::convection','options':{'gpu_policy':'sections','scope_transfers':'direct','scope_execution':'reached','host_threads':4},'source_changes':'none; pristine upstream INTENT declarations','status':'running','started':time.time()}
(OUT/'request.json').write_text(json.dumps(request,indent=2)+'\n')
started=time.monotonic()
try:
 metadata=transform_scopes(source,compiler,ROOT/'.venv/bin/python',OUT/'generated',entry=request['entry'],source_files=[pathlib.Path(p) for p in analysis['source_inputs'] if pathlib.Path(p)!=source],capture_facts=facts,gpu_options=request['options'],analysis_sources=analysis_path,summary_cache=OUT/'summary-cache')
 manifest=metadata['scope_manifest']
 result={'status':'generated','seconds':time.monotonic()-started,'scope_count':manifest['scope_count'],'manifest':str(OUT/'generated/scope-manifest.json'),'scope_summaries':[{k:scope.get(k) for k in ['owner','first_line','last_line','gpu_leaves','mode','estimate_available','planning_reason','boundaries']} for scope in manifest['scopes']],'boundaries':manifest.get('boundaries'),'automatic_estimate_available':manifest.get('automatic_estimate_available')}
except BaseException as e:
 result={'status':'failed','seconds':time.monotonic()-started,'error':str(e),'error_type':type(e).__name__}
 raise
finally:
 if 'result' in locals():(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result)[:8000],flush=True)
