import json,pathlib,sys,time
ROOT=pathlib.Path('/home/jirka/research/fortran-abomination');OUT=ROOT/'benchmarks/results/application-path-20261014'
sys.path.insert(0,str(ROOT/'elmm-pipeline'))
from elmm_pipeline.local_adapter import transform_scopes
r=json.loads((OUT/'request.json').read_text());source=pathlib.Path(r['source']);started=time.monotonic()
try:
 m=transform_scopes(source,OUT/'compiler',pathlib.Path('/tmp/application_path_profile_python'),OUT/'profile-generated',entry=r['entry'],source_files=[pathlib.Path(p) for p in r['source_inputs'] if pathlib.Path(p)!=source],capture_facts=r['capture_facts'],gpu_options=r['options'],analysis_sources=pathlib.Path(r['analysis_sources']),summary_cache=OUT/'summary-cache')
 result={'status':'generated','seconds':time.monotonic()-started,'scope_count':m['scope_manifest']['scope_count']}
except BaseException as e:
 result={'status':'failed','seconds':time.monotonic()-started,'error':str(e)}
finally:(OUT/'profile-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
