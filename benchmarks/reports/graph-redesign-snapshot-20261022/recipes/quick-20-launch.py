from pathlib import Path
import os
import runpy
import sys

Path('/tmp/fortran-quick-20.pid').write_text(str(os.getpid()) + '\n')
entry = '/home/jirka/research/fortran-abomination/containers/probe_prepared_e2e.py'
sys.path.insert(0, str(Path(entry).parent))
sys.argv = [entry, *sys.argv[1:]]
runpy.run_path(entry, run_name='__main__')
