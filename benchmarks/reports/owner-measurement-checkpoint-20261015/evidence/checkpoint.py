"""One full-process observation pair plus separate field/transfer diagnostics."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/jirka/research/fortran-abomination')
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'benchmarks/results/diffusion-complete-checkpoint-20261015'
PYTHON = ROOT / '.venv/bin/python'
GPU = 'GPU-d16b0d42-ee00-3af3-0c7e-d757b0cf9515'

def run(*args):
    subprocess.run([str(PYTHON), *map(str,args)], cwd=ROOT, check=True)

def read(path):
    return json.loads(path.read_text())

def main():
    records = {}
    for variant in ('native','resident'):
        build = HERE/variant/'candidate/build'
        assert read(HERE/variant/'candidate/build-generation.json')['status']=='built_not_executed'
        diagnostic = HERE/'full-fields'/variant
        run(HERE/'full_field_diagnostic.py','plan','--mode',variant,
            '--build-receipt',build/'build-receipts.json','--output',diagnostic,
            '--pipeline-root',HERE/variant/'pipeline')
        run(HERE/'full_field_diagnostic.py','build','--plan',diagnostic/'plan.json')
    for kind in ('untraced','fields'):
        for variant in ('native','resident'):
            binary = (HERE/variant/'candidate/build/ELMM' if kind=='untraced' else
                      HERE/'full-fields'/variant/'ELMM-fields')
            output=HERE/'runs'/kind/variant
            command=[HERE/'run_case.py','--case','channel-simple','--binary',binary,
                '--mode','diffusion-measurement-'+variant,'--output',output,
                '--reference',OLD/'runs/fields/channel-simple/native','--gpu',GPU,'--timeout','120']
            if kind=='fields' and variant=='resident':command.append('--trace')
            run(*command)
            manifest=read(HERE/variant/'artifacts/scope-manifest.json')
            observer=manifest['scopes'][0]['owner_measurement']
            rows=[json.loads(line.removeprefix('FORT_SCOPE_MEASUREMENT '))
                for line in (output/'stderr.log').read_text().splitlines()
                if line.startswith('FORT_SCOPE_MEASUREMENT ')]
            assert len(rows)==327, (kind,variant,len(rows))
            for row in rows:
                assert row['identity']==observer['identity'] and row['variant']==variant
                assert row['call_site']==observer['call_sites'][0]['identity']
                assert row['status']=='complete' and math.isfinite(row['seconds']) and row['seconds']>=0
                assert row['clock_rate']>0 and row['clock_finish']>=row['clock_start']
                elapsed=(row['clock_finish']-row['clock_start'])/row['clock_rate']
                assert math.isclose(row['seconds'],elapsed,rel_tol=1e-12,abs_tol=1e-12)
            log=output/'stderr.log'
            records[kind+'/'+variant]={'count':len(rows),'seconds_total':sum(r['seconds'] for r in rows),
                'samples':rows,'manifest_identity':observer['identity'],
                'log':str(log),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
                'traced':kind=='fields' and variant=='resident','qualification':False}
    run(HERE/'full_field_diagnostic.py','compare','--native',HERE/'runs/fields/native',
        '--candidate',HERE/'runs/fields/resident','--reference-vtk',OLD/'runs/fields/channel-simple/native/output',
        '--expected-steps','109','--output',HERE/'full-fields-comparison.json',
        '--pipeline-root',HERE/'resident/pipeline')
    run(OLD/'transfer_ledger.py','--log',HERE/'runs/fields/resident/stderr.log',
        '--manifest',HERE/'resident/artifacts/scope-manifest.json',
        '--runtime-source',HERE/'resident/artifacts/scoped_runtime.cu',
        '--pipeline-root',HERE/'resident/pipeline','--returncode','0','--output',HERE/'transfer-ledger.json')
    (HERE/'observations.json').write_text(json.dumps(records,sort_keys=True,indent=2)+'\n')
    print(json.dumps({key:{k:v for k,v in row.items() if k!='samples'} for key,row in records.items()}))

if __name__=='__main__':main()
