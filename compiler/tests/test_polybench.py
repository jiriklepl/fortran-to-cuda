"""PolyBench/Fortran 1.0: every benchmark is described, prepared, built and run, and its GPU build prints the
checksums its native build prints.

The suite's archive is downloaded once per run, or taken from ``POLYBENCH_ARCHIVE``; without it the tests skip.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from compiler.resident.application import build, load, prepare
from compiler.tests.resident_application_support import cuda_toolchain, fortran_compiler

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("polybench_describe", ROOT / "benchmarks/polybench/describe.py")
describe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(describe)

# What the compiler takes of each kernel: entries, the source nests they cover, and the loops it leaves to the host.
COVERAGE = {
    "2mm": (1, 2, 0),
    "3mm": (1, 3, 0),
    "adi": (0, 0, 1),
    "atax": (1, 2, 0),
    "bicg": (1, 2, 0),
    "cholesky": (1, 1, 1),
    "correlation": (1, 4, 0),
    "covariance": (1, 3, 0),
    "doitgen": (1, 1, 0),
    "durbin": (2, 2, 1),
    "dynprog": (0, 0, 1),
    "fdtd-2d": (1, 4, 0),
    "fdtd-apml": (1, 1, 0),
    "floyd-warshall": (1, 1, 0),
    "gemm": (1, 1, 0),
    "gemver": (1, 4, 0),
    "gesummv": (1, 1, 0),
    "gramschmidt": (1, 1, 0),
    "jacobi-1d-imper": (1, 2, 0),
    "jacobi-2d-imper": (1, 2, 0),
    "lu": (1, 2, 0),
    "ludcmp": (1, 2, 2),
    "mvt": (1, 2, 0),
    "reg_detect": (0, 0, 1),
    "seidel-2d": (0, 0, 1),
    "symm": (1, 1, 0),
    "syr2k": (1, 2, 0),
    "syrk": (1, 2, 0),
    "trisolv": (0, 0, 1),
    "trmm": (0, 0, 1),
}
# The benchmarks whose GPU build launches kernels at the size the tests run.
ON_DEVICE = {"2mm", "3mm", "cholesky", "correlation", "covariance", "doitgen", "durbin", "fdtd-2d", "fdtd-apml", "gemm",
             "gemver", "gesummv", "jacobi-1d-imper", "jacobi-2d-imper", "lu", "ludcmp", "mvt", "syr2k", "syrk"}
SIZE = "mini"


@pytest.fixture(scope="session")
def archive(tmp_path_factory):
    given = os.environ.get("POLYBENCH_ARCHIVE")
    if given:
        path = Path(given)
    else:
        path = tmp_path_factory.mktemp("polybench") / "polybench-fortran-1.0.tar.gz"
        try:
            with urllib.request.urlopen(describe.URL, timeout=60) as response:
                path.write_bytes(response.read())
        except (urllib.error.URLError, TimeoutError) as error:
            pytest.skip(f"PolyBench/Fortran is not available from {describe.URL}: {error}")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == describe.SHA256
    return path


@pytest.fixture(scope="session")
def descriptions(archive, tmp_path_factory):
    return describe.describe(archive, tmp_path_factory.mktemp("descriptions"))


def test_the_suite_has_the_benchmarks_the_tests_expect(descriptions):
    assert set(descriptions) == set(COVERAGE)
    assert set(COVERAGE) >= ON_DEVICE


@pytest.mark.skipif(fortran_compiler() is None, reason="GNU Fortran is required for preprocessing")
@pytest.mark.parametrize("name", sorted(COVERAGE))
def test_every_benchmark_prepares_what_the_compiler_takes_of_its_kernel(name, archive, descriptions, tmp_path):
    prepare(load(descriptions[name]), {"polybench": archive}, tmp_path / "build", variant=SIZE,
            fortran=fortran_compiler())
    summary, = json.loads((tmp_path / "build/resident/prepared.json").read_text())["selections"]
    assert (summary["entries"], summary["nests"], len(summary["rejected"])) == COVERAGE[name]


@pytest.mark.cuda
@pytest.mark.parametrize("name", sorted(COVERAGE))
def test_every_benchmark_prints_its_native_checksums_from_the_device(name, archive, descriptions, tmp_path):
    nvcc, host, fortran, architecture = cuda_toolchain()
    application = load(descriptions[name])
    given = {"polybench": archive}
    prepared = prepare(application, given, tmp_path / "build", variant=SIZE, fortran=fortran)
    built = build(application, prepared, given, tmp_path / "build", fortran=fortran, cxx=host, nvcc=nvcc,
                  architecture=architecture, jobs=2)

    def run(executable, **variables):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("FORT_")}
        result = subprocess.run([built["binaries"][executable]["path"]], cwd=tmp_path, capture_output=True,
                                text=True, timeout=300, check=False, env={**environment, **variables})
        assert result.returncode == 0, result.stderr[-4000:]
        checksums = [line for line in result.stdout.splitlines() if line.startswith("checksum")]
        assert checksums
        return checksums, result.stderr

    native, _ = run(name + "-native")
    resident, trace = run(name, FORT_RUNTIME_TRACE="1")
    assert resident == native
    launches = sum(line.startswith(("FORT_SCOPED launch ", "FORT_SCOPED graph_kernel ")) for line in trace.splitlines())
    assert (launches > 0) == (name in ON_DEVICE)
