"""Archive this bounded measurement checkpoint; no application execution."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics

ROOT = Path('/home/jirka/research/fortran-abomination')
HERE = Path(__file__).resolve().parent
FIRST = ROOT/'benchmarks/results/owner-measurement-checkpoint-20261015'
OUT = ROOT/'benchmarks/reports/owner-measurement-checkpoint-20261015'

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert not (OUT/'report.json').exists()
    obs=read(HERE/'observations.json')
    fields=read(HERE/'full-fields-comparison.json')
    ledger=read(HERE/'transfer-ledger.json')
    assert fields['status']=='passed' and len(fields['comparisons'])==20
    assert not ledger['issues'] and ledger['status']=='complete_observed_trace'
    inputs={}
    def archive(path,name):
        data=path.read_bytes()
        destination=OUT/'evidence'/name
        if len(data)>250_000:
            destination=destination.with_name(destination.name+'.gz')
            archived=gzip.compress(data,mtime=0)
        else:archived=data
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(archived)
        inputs[str(path)]={'path':str(destination.relative_to(OUT)),
            'source_sha256':hashlib.sha256(data).hexdigest(),'archive_sha256':sha(destination)}
    for attempt,base in (('initial',FIRST),('final',HERE)):
        for variant in ('native','resident'):
            for name in ('run.py','request.json','capture-facts.json','command.json','completion.json','generation.log',
                         'source-before.json','source-after.json'):
                archive(base/variant/name,f'{attempt}/{variant}/{name}')
    for variant in ('native','resident'):
        base=HERE/variant
        for name in ('build_candidate.py','worker-result.json','artifacts/scope-manifest.json',
                     'candidate/build-generation.json','candidate/composition.json',
                     'candidate/native-command-comparison.json','candidate/build/build-receipts.json'):
            archive(base/name,f'final/{variant}/{name}')
        for name in ('plan.json','build.json','dump-main.f90'):
            archive(HERE/'full-fields'/variant/name,f'full-fields/{variant}/{name}')
        for kind in ('untraced','fields'):
            for name in ('command.json','run.json','stdout.log','stderr.log'):
                archive(HERE/'runs'/kind/variant/name,f'runs/{kind}/{variant}/{name}')
    for name in ('checkpoint.py','checkpoint.log','run_case.py','full_field_diagnostic.py',
                 'observations.json','full-fields-comparison.json','transfer-ledger.json','freeze_report.py'):
        archive(HERE/name,name)
    for name in ('scope-measurement-root-tests.log','scope-measurement-gpu-recheck.log',
                 'scope-measurement-worker-fix.log','scope-measurement-final-tests.log'):
        archive(Path('/tmp')/name,'validation/'+name)
    for name in ('receipt.json','final.log'):
        archive(ROOT/'elmm-pipeline/work/owner-measurement-final-validation-20261015'/name,
                'validation/pipeline-'+name)
    request=read(HERE/'native/request.json')
    manifests={v:read(HERE/v/'artifacts/scope-manifest.json') for v in ('native','resident')}
    identities={v:manifests[v]['scopes'][0]['owner_measurement']['identity'] for v in manifests}
    assert len(set(identities.values()))==1
    summaries={}
    for variant in ('native','resident'):
        row=obs['untraced/'+variant]
        samples=[r['seconds'] for r in row['samples']]
        run=read(HERE/'runs/untraced'/variant/'run.json')
        receipt=read(HERE/variant/'candidate/build-generation.json')
        comparisons=read(HERE/variant/'candidate/native-command-comparison.json')
        assert len(comparisons)==72 and all(r['compiler_equal'] and r['normalized_arguments_equal'] for r in comparisons)
        assert receipt['source_unchanged'] and run['completed'] and row['count']==327
        summaries[variant]={'binary':receipt['binary'],'build_receipt_identity':receipt['build_receipts']['identity'],
            'complete_process_seconds':run['diagnostic_complete_process_seconds'],
            'owner_calls':len(samples),'owner_seconds_total':sum(samples),
            'owner_seconds_median':statistics.median(samples),'owner_seconds_range':[min(samples),max(samples)],
            'run_kind':'one full-process diagnostic; no warmup; no qualification',
            'configuration_sha256':run['configuration_sha256'],'runtime_environment':run['environment'],
            'affinity':run['affinity'],'threads':run['threads'],'completed_steps':run['completion']['completed_steps']}
    report={'schema_version':1,'kind':'bounded compiler-owned call observation checkpoint',
        'compatible_commits':request['commits'],'initial_compiler_commit':read(FIRST/'native/request.json')['commits']['compiler'],
        'source_inputs':request['source_sha256'],'configurations':request['configurations'],
        'archives':request['archives'],'measurement_identity':identities['native'],
        'measurement_contract':manifests['resident']['scopes'][0]['owner_measurement'],
        'automatic_estimate_available':False,'calibration_identity':None,'qualification':False,
        'shared_producer_diffusion_accumulation_window':False,
        'case':'channel-simple','channel_candidate_executed':False,'modes':summaries,
        'native_outside_measured_diffusion':True,'sgs_offloaded':False,
        'field_validation':{'status':fields['status'],'records':20,'driver':fields['driver']},
        'observed_resident_transfers':ledger['observed_submissions'],'prediction_available':False,
        'successful_contexts':len(ledger['contexts']),'trace_attribution_issues':ledger['issues'],
        'stop_condition':'Current measurement slice complete; no further capability or qualification campaign.',
        'remaining':['compiler-selected shared producer/diffusion/accumulation ownership',
            'workload-keyed calibrated automatic interval pricing and selector',
            'full channel GPU validation and both seven-round primary qualifications'],
        'archived_inputs':inputs}
    (OUT/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    scope=OUT/'scope-map.svg'
    scope.write_text(scope.read_text().replace('Compiler 2f36657','Compiler dd8ea33'))
    os.environ['MPLCONFIGDIR']='/tmp/owner-measurement-matplotlib'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.7,1]})
    for variant,color in (('native','#526b89'),('resident','#b85d35')):
        samples=[1000*r['seconds'] for r in obs['untraced/'+variant]['samples']]
        axes[0].plot(range(1,len(samples)+1),samples,'.',ms=3,alpha=.6,color=color,
            label=f'{variant}: total {sum(samples)/1000:.3f}s; median {statistics.median(samples):.2f}ms')
        axes[0].axhline(statistics.median(samples),color=color,lw=1,alpha=.7)
    axes[0].set(xlabel='Original diffusion invocation (327 per process)',ylabel='Complete call interval (ms)',
        title='One untraced full-duration process per variant')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    traffic=ledger['observed_submissions']
    values=[traffic['upload_bytes']/2**30,traffic['download_bytes']/2**30]
    bars=axes[1].bar(['Upload','Download'],values,color=['#637f9b','#b85d35'])
    for bar,value,count in zip(bars,values,[traffic['upload_events'],traffic['download_events']]):
        axes[1].text(bar.get_x()+bar.get_width()/2,value+.1,f'{value:.3f} GiB\n{count:,} events',ha='center',fontsize=9)
    axes[1].set(ylabel='Observed GiB',ylim=(0,max(values)*1.28),title='Separate resident field/transfer run')
    axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Measurement capability works; isolated diffusion still regresses',fontsize=14)
    fig.text(.5,.015,'Diagnostic only · No calibration/workload key · No SGS offloading in these modes · Predictions unavailable',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    fig.savefig(OUT/'observation-diagnostics.svg')
    fig.savefig(OUT/'observation-diagnostics.png',dpi=150)
    print(json.dumps({'report':str(OUT),'archived_inputs':len(inputs),'modes':summaries,'traffic':traffic},indent=2))

if __name__=='__main__':main()
