from hashlib import sha256
import json
from pathlib import Path
import subprocess
from compiler.tests.test_native_demand_cuda import SOURCE
from compiler.tests.test_native_demand_owner import generate
ROOT=Path('/home/jirka/research/fortran-abomination')
HERE=Path(__file__).resolve().parent
FIXTURE=HERE.parent/'corrected/pytest/test_3d_native_demand_keeps_cu0'
BUILD=FIXTURE/'build'
def digest(x):return sha256(x).hexdigest()
def inventory():return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted((ROOT/'compiler').rglob('*.py')) if 'tests' not in p.relative_to(ROOT/'compiler').parts}
def write(name,x):(HERE/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
before=inventory();write('compiler-before.json',before)
original=json.loads((BUILD/'scope-manifest.json').read_text())
assert (FIXTURE/'owner.f90').read_text()==SOURCE
path,builder,outputs,current=generate(FIXTURE,SOURCE)
assert path==FIXTURE/'owner.f90'
selected={item['path'] for item in original['build_sources']}|{item['path'] for item in current['build_sources']}
selected|={name for name in set(outputs)|set(original['artifacts_sha256']) if Path(name).suffix.lower() in {'.h','.hpp','.cuh','.f90','.cu','.cpp','.c'}}
rows=[]
for name in sorted(selected):
 old=(BUILD/name).read_bytes() if (BUILD/name).exists() else None
 new=outputs[name].encode() if name in outputs else None
 rows.append({'path':name,'recorded_sha256':digest(old) if old is not None else None,'current_sha256':digest(new) if new is not None else None,'identical':old==new})
 if old!=new:
  target=HERE/'current'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(new or b'')
after=inventory();write('compiler-after.json',after)
result={'source':str(path),'source_sha256':digest(path.read_bytes()),'policy':'sections','compiler_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checked_compiled_sources_and_headers':rows,'all_identical':all(r['identical'] for r in rows),'changed_compiler_files':[p for p in sorted(before.keys()|after.keys()) if before.get(p)!=after.get(p)],'comparison':'exact bytes; no normalization; only manifest/JSON diagnostics omitted','new_test_transfer_count_assertions':'2 uploads/3 downloads checked against both archived branch ledgers; no application rerun','binary_sha256':digest((BUILD/'verify').read_bytes())}
write('comparison.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checked_compiled_sources_and_headers'}),flush=True)
print('checked',len(rows),'differences',[r['path'] for r in rows if not r['identical']],flush=True)
