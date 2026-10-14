"""One bounded automatic-generation checkpoint from committed compiler inputs."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time

ROOT = Path('/home/jirka/research/fortran-abomination')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'automatic-generation'
PYTHON = ROOT / '.venv/bin/python'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def checked_inputs():
    request = json.loads((HERE / 'request.json').read_text())
    bundle = json.loads((HERE / 'input-bundle.json').read_text())
    facts_path = Path(request['capture_facts'])
    request['capture_facts'] = json.loads(facts_path.read_text())
    analysis = json.loads(Path(request['analysis_sources']).read_text())
    expected = {**analysis['source_inputs'], **analysis['dependencies'],
                **{row['path']: row['sha256'] for row in analysis['entries']}}
    expected[str(facts_path)] = bundle['capture_facts_sha256']
    expected[request['analysis_sources']] = bundle['analysis_sha256']
    for case in bundle['cases'].values():
        expected.update({str(Path(case['path']) / name): sha for name, sha in case['files'].items()})
    for name in ['request.json', 'input-bundle.json']:
        expected[str(HERE / name)] = digest(HERE / name)
    for path, sha in expected.items():
        if digest(path) != sha:
            raise ValueError('Input differs from fresh configured export: ' + path)
    if request['capture_facts']['sources'] != analysis['source_inputs']:
        raise ValueError('Reissued capture facts differ from configured export')
    return request, expected


def worker():
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    sys.path.insert(0, str(OUT / 'pipeline'))
    from elmm_pipeline.local_adapter import transform_scopes
    request, inputs = checked_inputs()
    started = time.monotonic()
    try:
        metadata = transform_scopes(Path(request['source']), OUT / 'compiler', PYTHON,
            OUT / 'generated', entry=request['entry'],
            source_files=[Path(path) for path in request['source_files']],
            capture_facts=request['capture_facts'], gpu_options=request['options'],
            analysis_sources=Path(request['analysis_sources']), summary_cache=OUT / 'summary-cache')
        manifest = metadata['scope_manifest']
        result = {'status': 'generated', 'scope_count': manifest['scope_count'],
            'automatic_estimate_available': manifest['automatic_estimate_available'],
            'scope_summaries': [{key: scope.get(key) for key in
                ['owner', 'first_line', 'last_line', 'gpu_leaves', 'mode', 'estimate_available',
                 'planning_reason', 'automatic_preflight', 'boundaries']} for scope in manifest['scopes']],
            'boundaries': manifest['boundaries'], 'manifest_sha256': digest(OUT / 'generated/scope-manifest.json')}
    except BaseException as error:
        result = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
        raise
    finally:
        if 'result' in locals():
            result['seconds'] = time.monotonic() - started
            result['changed_inputs'] = [path for path, sha in inputs.items() if digest(path) != sha]
            write(OUT / 'worker-result.json', result)
            print(json.dumps(result), flush=True)


def main():
    OUT.mkdir(exist_ok=False)
    request, inputs = checked_inputs()
    identities = {}
    for name, repo, pathspec in [('compiler', ROOT, 'compiler'),
                                ('pipeline', ROOT / 'elmm-pipeline', 'elmm_pipeline')]:
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        data = subprocess.check_output(['git', '-C', str(repo), 'archive', commit, pathspec])
        archive = OUT / (name + '.tar')
        archive.write_bytes(data)
        destination = OUT / name
        destination.mkdir()
        with tarfile.open(fileobj=io.BytesIO(data)) as stream:
            stream.extractall(destination, filter='data')
        identities[name] = {'commit': commit, 'archive_sha256': digest(archive),
            'files': {str(path.relative_to(destination)): digest(path)
                      for path in sorted(destination.rglob('*')) if path.is_file()}}
    command = ['timeout', '--kill-after=5', '300', 'taskset', '-c', '4,5,10,11',
               str(PYTHON), str(Path(__file__).resolve()), '--worker']
    record = {'schema_version': 1, 'kind': 'bounded automatic application generation diagnostic',
        'request_sha256': digest(HERE / 'request.json'), 'input_bundle_sha256': digest(HERE / 'input-bundle.json'),
        'runner_sha256': digest(__file__), 'identities': identities, 'input_sha256': inputs,
        'command': command, 'cwd': str(ROOT), 'cap_seconds': 300, 'virtual_memory_bytes': 3 * 1024**3,
        'policy': request['options']['gpu_policy'], 'profile': 'none; missing estimates must select native',
        'application_executed': False, 'performance_measured': False, 'qualification': False}
    write(OUT / 'launch.json', record)
    started = time.monotonic()
    with (OUT / 'generation.log').open('w') as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    record.update(exit_code=process.returncode, elapsed_seconds=time.monotonic() - started,
        log_sha256=digest(OUT / 'generation.log'),
        status='timeout' if process.returncode == 124 else 'finished' if process.returncode == 0 else 'failed',
        manifest_emitted=(OUT / 'generated/scope-manifest.json').exists(),
        changed_inputs=[path for path, sha in inputs.items() if digest(path) != sha])
    write(OUT / 'completion.json', record)
    print(json.dumps({key: record[key] for key in ['status', 'exit_code', 'elapsed_seconds', 'manifest_emitted', 'changed_inputs']}), flush=True)
    sys.exit(process.returncode)


if __name__ == '__main__':
    worker() if '--worker' in sys.argv else main()
