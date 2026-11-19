"""Shared pytest setup for the compiler test suite.

Put the repository on the import path so the suite runs from any working
directory, without PYTHONPATH.
"""

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]

if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
