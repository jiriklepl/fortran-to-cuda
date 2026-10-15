"""Freeze source-bound application evidence after the diagnostic processes stop."""
from pathlib import Path
import gzip
import hashlib
import json
import statistics
import subprocess

ROOT=Path('/home/jirka/research/fortran-abomination')
HERE=Path(__file__).resolve().parent
REPORT=ROOT/'benchmarks/reports/indirect-reuse-checkpoint-20261015'
OLD=ROOT/'benchmarks/results/owner-measurement-checkpoint-20261015-retry'
NATIVE=ROOT/'benchmarks/results/diffusion-complete-checkpoint-20261015/runs/fields'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')

def main():
 assert (HERE/'checkpoint-completion.json').is_file(), 'checkpoint must finish before report freeze'
 assert read(HERE/'checkpoint-completion.json')['completed']
 REPORT.mkdir(parents=True,exist_ok=True)
 inventory={}
 def archive(source,name=None):
  if name is None:name=str(source.relative_to(HERE))
  data=source.read_bytes();original=hashlib.sha256(data).hexdigest()
  if len(data)>65536 or source.suffix=='.log':
   name+='.gz';data=gzip.compress(data,mtime=0)
  destination=REPORT/'evidence'/name;destination.parent.mkdir(parents=True,exist_ok=True)
  if destination.exists():assert destination.read_bytes()==data,destination
  else:destination.write_bytes(data)
  inventory[str(destination.relative_to(REPORT))]={'source':str(source),'source_sha256':original,
       'archived_sha256':hashlib.sha256(data).hexdigest(),'compressed':name.endswith('.gz')}
  return str(destination.relative_to(REPORT))
 for name in ['observations.json','checkpoint-completion.json','checkpoint.py','run_case.py',
              'full_field_diagnostic.py','freeze_report.py','cuda-tests.json','cuda-tests.log',
              'indirect-access-reuse-validation-20261015.json','indirect-access-reuse-host-20261015.log',
              'indirect-access-reuse-size-20261015.json','transfer-ledger.json',
              'channel-simple-full-fields-comparison.json','channel-full-fields-comparison.json']:
  archive(HERE/name)
 for name in ['run.py','build_candidate.py','request.json','capture-facts.json','command.json','completion.json',
              'worker-result.json','generation.log','artifacts/scope-manifest.json','artifacts/scope-transformation.json',
              'candidate/build-generation.json','candidate/composition.json','candidate/native-command-comparison.json',
              'candidate/build/build-receipts.json','candidate/build/build-command.json']:
  archive(HERE/'resident'/name)
 for name in ['plan.json','build.json','build.log','dump-main.f90']:
  archive(HERE/'full-fields'/name)
 for case in ['channel-simple','channel']:
  for name in ['run.json','command.json']:
   archive(NATIVE/case/'native'/name,'native-field-references/'+case+'/'+name)
 for run in sorted((HERE/'runs').rglob('run.json')):
  for name in ['command.json','run.json','stdout.log','stderr.log']:
   archive(run.parent/name)
 archive(OLD/'transfer-ledger.json','before/transfer-ledger.json')
 archive(OLD/'resident/candidate/build-generation.json','before/build-generation.json')
 archive(OLD/'resident/artifacts/scope-manifest.json','before/scope-manifest.json')
 rows=read(HERE/'observations.json')
 assert len(rows)==16
 performance={}
 for mode in ['native','sgs-direct','resident-before','resident-reuse']:
  measured=[r for r in rows if r['mode']==mode and not r['excluded_warmup']]
  assert len(measured)==3
  values=[r['seconds'] for r in measured]
  performance[mode]={'samples':values,'median':statistics.median(values),'min':min(values),'max':max(values),
      'binary_sha256':measured[0]['binary_sha256'],
      'owner_seconds':[r['owner_seconds'] for r in measured]}
 native=performance['native']['median']
 for mode in performance:performance[mode]['native_relative_median']=performance[mode]['median']/native
 before=performance['resident-before']['median'];after=performance['resident-reuse']['median']
 old_trace=read(OLD/'transfer-ledger.json');new_trace=read(HERE/'transfer-ledger.json')
 assert not new_trace['issues'] and not new_trace['definition_failures']
 fields={}
 for case in ['channel-simple','channel']:
  comparison=read(HERE/(case+'-full-fields-comparison.json'))
  assert comparison['status']=='passed' and len(comparison['comparisons'])==20
  assert all(c['passed'] for c in comparison['comparisons'].values())
  run=read(HERE/'runs/fields'/case/'resident-reuse/run.json')
  fields[case]={'status':'passed','field_records':20,'completion':run['completion'],
      'max_absolute_error':max(c['max_absolute_error'] for c in comparison['comparisons'].values()),
      'bounds_and_halos':True,'atol':1e-6,'rtol':1e-5,'traced':case=='channel-simple',
      'performance_qualification':False}
 build=read(HERE/'resident/candidate/build-generation.json')
 assert build['original_compilation_flags_match_native'] and build['source_unchanged']
 manifest=read(HERE/'resident/artifacts/scope-manifest.json')
 report={'schema_version':1,'kind':'operation-local inspector reuse checkpoint','qualification':False,
      'automatic_residency_complete':False,'performance_targets_met':False,'commits':build['commits'],
      'binary':build['binary'],'prior_resident_compiler_commit':'dd8ea33d55328d7ea7919ae328440f6a047a2cdf',
      'generation_seconds':read(HERE/'resident/completion.json')['elapsed_seconds'],
      'build_seconds':build['elapsed_seconds'],'original_compilation_flags_match_native':True,
      'automatic_estimate_available':manifest.get('automatic_estimate_available'),
      'performance':performance,'reuse_median_reduction_fraction':1-after/before,
      'comparison_protocol':{'case':'channel-simple','threads':4,'affinity':[4,5,10,11],
           'excluded_warmup_per_mode':1,'measured_rotated_rounds':3,'natural_completed_steps':109,
           'timing':'complete process; untraced; includes observer logging in both resident modes',
           'resident_modes_sgs_placement':'native','target_seconds':9},
      'fields':fields,'transfers':{'before_observed':old_trace['observed_submissions'],
           'after_observed':new_trace['observed_submissions'],
           'expected':'unchanged exact sections; no compatible automatic transfer-cost estimate',
           'equal':old_trace['observed_submissions']==new_trace['observed_submissions'],
           'issues':new_trace['issues'],'successful_contexts':sum(c['successful_close_observed'] for c in new_trace['contexts'])},
      'tests':{'host':57,'cuda':2,'full_required_suites_run':False},
      'evidence':inventory}
 write(REPORT/'report.json',report)
 print(json.dumps({'medians':{k:v['median'] for k,v in performance.items()},
                   'reuse_reduction':report['reuse_median_reduction_fraction'],
                   'fields':fields,'transfers_equal':report['transfers']['equal']}))

if __name__=='__main__':main()
