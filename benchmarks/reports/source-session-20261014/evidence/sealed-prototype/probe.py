import hashlib, json, pathlib, subprocess, sys, tarfile, time
ROOT=pathlib.Path('/home/jirka/research/fortran-abomination')
EXPERIMENT=pathlib.Path('/tmp/fortran-residency-e2e-20261014')
OUT=ROOT/'benchmarks/results/application-path-sealed-20261014'
OUT.mkdir(exist_ok=False)
digest=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
compiler=OUT/'compiler'; compiler.mkdir()
commit=subprocess.check_output(['git','-C',str(EXPERIMENT),'rev-parse','HEAD'],text=True).strip()
with (OUT/'compiler.tar').open('wb') as f:subprocess.run(['git','-C',str(EXPERIMENT),'archive','HEAD','compiler'],stdout=f,check=True)
with tarfile.open(OUT/'compiler.tar') as f:f.extractall(compiler,filter='data')
changed=['compiler/frontend/sealed_syntax.py','compiler/frontend/source_syntax.py','compiler/scopes/source.py','compiler/tests/test_sealed_syntax.py']
overlay={}
for relative in changed:
 source=EXPERIMENT/relative;target=compiler/relative;before=digest(source);target.write_bytes(source.read_bytes());assert digest(source)==digest(target)==before;overlay[relative]=before
compiler_hashes={str(p.relative_to(compiler)):digest(p) for p in (compiler/'compiler').rglob('*') if p.is_file()}
pipeline=OUT/'pipeline';pipeline.mkdir()
pipeline_commit='3c799dc0d266abf6aa6eb0d4b3ce39234cd80bcd'
with (OUT/'pipeline.tar').open('wb') as f:subprocess.run(['git','-C',str(ROOT/'elmm-pipeline'),'archive',pipeline_commit,'elmm_pipeline'],stdout=f,check=True)
with tarfile.open(OUT/'pipeline.tar') as f:f.extractall(pipeline,filter='data')
sys.path.insert(0,str(pipeline))
from elmm_pipeline.local_adapter import transform_scopes
old_path=ROOT/'benchmarks/results/application-path-20261014/request.json';old=json.loads(old_path.read_text())
analysis_path=pathlib.Path(old['analysis_sources']);analysis=json.loads(analysis_path.read_text())
for p,h in {**analysis['source_inputs'],**analysis['dependencies']}.items():assert digest(p)==h,p
for row in analysis['entries']:assert digest(row['path'])==row['sha256'],row['path']
request={**old,'parent_commit':commit,'pipeline_commit':pipeline_commit,'compiler_overlay':overlay,'compiler_snapshot_sha256':compiler_hashes,'experimental':True,'prototype':'sealed original AST/item syntax; copies thaw; immutable syntax identity checks','same_request_as':str(old_path),'same_request_sha256':digest(old_path),'source_changes':'none; same69 pristine original source inputs/captures/options','started':time.time()}
(OUT/'request.json').write_text(json.dumps(request,indent=2)+'\n')
source=pathlib.Path(old['source']);started=time.monotonic()
try:
 metadata=transform_scopes(source,compiler,ROOT/'.venv/bin/python',OUT/'generated',entry=old['entry'],source_files=[pathlib.Path(p) for p in old['source_inputs'] if pathlib.Path(p)!=source],capture_facts=old['capture_facts'],gpu_options=old['options'],analysis_sources=analysis_path,summary_cache=OUT/'summary-cache')
 manifest=metadata['scope_manifest']
 result={'status':'generated','experimental':True,'seconds':time.monotonic()-started,'scope_count':manifest['scope_count'],'manifest':str(OUT/'generated/scope-manifest.json'),'scope_summaries':[{k:scope.get(k) for k in ['owner','first_line','last_line','gpu_leaves','mode','estimate_available','planning_reason','boundaries']} for scope in manifest['scopes']],'boundaries':manifest.get('boundaries'),'automatic_estimate_available':manifest.get('automatic_estimate_available')}
except BaseException as e:
 result={'status':'failed','experimental':True,'seconds':time.monotonic()-started,'error':str(e),'error_type':type(e).__name__}
 raise
finally:
 if 'result' in locals():(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result)[:10000],flush=True)
