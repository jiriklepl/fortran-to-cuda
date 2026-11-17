"""Shared pytest setup for the compiler test suite.

The tests import ``compiler`` and, for consumer-contract checks, the
``elmm_pipeline`` package from the ``elmm-pipeline`` submodule. Put both on the
import path here so the suite runs from any working directory, without
PYTHONPATH.
"""

import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
PIPELINE = REPOSITORY / "elmm-pipeline"

for _path in (PIPELINE, REPOSITORY):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_PIPELINE_MISSING = "the elmm-pipeline submodule is not checked out (git submodule update --init elmm-pipeline)"


def _skip_without_pipeline(error):
    if error.name == "elmm_pipeline" or (error.name or "").startswith("elmm_pipeline."):
        pytest.skip(_PIPELINE_MISSING)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except ModuleNotFoundError as error:
        _skip_without_pipeline(error)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except ModuleNotFoundError as error:
        _skip_without_pipeline(error)
        raise


def pytest_ignore_collect(collection_path, config):
    """Without the submodule, skip modules that import the pipeline at import time."""
    if (PIPELINE / "elmm_pipeline").is_dir() or collection_path.suffix != ".py":
        return None
    try:
        head = collection_path.read_text(encoding="utf-8")
    except OSError:
        return None
    return True if any(line.startswith(("from elmm_pipeline", "import elmm_pipeline")) for line in head.splitlines()) else None
