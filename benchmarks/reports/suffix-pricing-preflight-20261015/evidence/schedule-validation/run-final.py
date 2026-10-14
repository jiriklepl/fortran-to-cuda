from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path('/home/jirka/research/fortran-abomination')
MODE = sys.argv[1]
if MODE not in {'host', 'cuda'}:
    raise SystemExit('choose host or cuda')
OUT = Path(__file__).resolve().parent / MODE
OUT.mkdir()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()

def inventory():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in git('ls-files', 'compiler').splitlines()}

os.sched_setaffinity(0, {0, 1, 6, 7} if MODE == 'host' else {4, 5, 10, 11})
before = inventory()
write('source-inventory.json', before)
(OUT / 'working-tree.patch').write_text(git('diff', '--', 'compiler') + '\n')
selection = ('compiler/tests/test_exact_window.py' if MODE == 'host' else
    'compiler/tests/test_exact_window_teams_cuda.py::test_effective_runtime_schedule_selects_before_ownership_with_unchanged_environment')
command = [str(ROOT / '.venv/bin/python'), '-m', 'pytest', '-q', selection,
           '--basetemp=' + str(OUT / 'pytest')]
receipt = {'schema_version': 1, 'kind': 'focused final ' + MODE + ' correctness diagnostic',
           'parent_base_commit': git('rev-parse', 'HEAD'),
           'pipeline_commit': git('rev-parse', 'HEAD', cwd=ROOT / 'elmm-pipeline'),
           'command': command, 'cwd': str(ROOT), 'affinity': sorted(os.sched_getaffinity(0)),
           'source_inventory_sha256': hashlib.sha256((OUT / 'source-inventory.json').read_bytes()).hexdigest(),
           'working_tree_patch_sha256': hashlib.sha256((OUT / 'working-tree.patch').read_bytes()).hexdigest(),
           'performance_claim': False, 'application_admission_claim': False}
write('command.json', receipt)
start = time.monotonic()
with (OUT / 'pytest.log').open('w') as log:
    try:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        receipt.update(exit_code=result.returncode, timed_out=False)
    except subprocess.TimeoutExpired:
        receipt.update(exit_code=None, timed_out=True)
receipt['elapsed_seconds'] = time.monotonic() - start
after = inventory()
receipt['inventory_stable'] = before == after
receipt['changed_paths'] = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
receipt['log_sha256'] = hashlib.sha256((OUT / 'pytest.log').read_bytes()).hexdigest()
write('final.json', receipt)
print(json.dumps(receipt, sort_keys=True), flush=True)
raise SystemExit(0 if receipt.get('exit_code') == 0 and receipt['inventory_stable'] else 1)
