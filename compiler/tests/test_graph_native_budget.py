"""Complete native teams use input work, not numerical search-window limits."""
import os

import pytest

from compiler.tests.test_graph_source_owner import PRIVATE_SCRATCH_SOURCE, emit
from compiler.tests.test_source_scopes import run
from compiler.tests.test_storage_epoch_runtime import build_binaries


def test_large_nowait_team_retains_one_native_transaction_and_residency(tmp_path):
    original = "scratch(i)=b(i)+mask(1)\na(i)=scratch(i)+b(i)"
    work = "scratch(1)=b(i)+mask(1)\na(i)=scratch(1)\n" + "a(i)=a(i)+b(i)\n" * 100
    source = PRIVATE_SCRATCH_SOURCE.replace(original, work).replace("!$omp end do", "!$omp end do nowait")
    path, outputs, report = emit(tmp_path, source)
    owner, = report["scopes"]
    assert not owner["boundaries"]
    native = [item for segment in owner["planning_segments"] for item in segment["operations"]["native_operations"]]
    assert any(item["source_constants"] for item in native)
    generated = outputs[report["sources"][str(path)]["replacement"]]
    # The checked native branch and the ordinary source fallback each retain
    # their original complete team; runtime tests check only one executes.
    assert generated.count("!$omp end do nowait") == 2
    # The original automatic scratch remains private native storage. No
    # caller capture or synthetic GPU definition is granted by a larger walk.
    assert all(not item["resource"].endswith("::scratch") for item in owner["resource_bindings"]["resources"])


@pytest.fixture(scope="module")
def large_native_binaries(tmp_path_factory):
    original = "scratch(i)=b(i)+mask(1)\na(i)=scratch(i)+b(i)"
    work = "scratch(1)=b(i)+mask(1)\na(i)=scratch(1)\nx(i)=x(i)+1\n" + "a(i)=a(i)+b(i)\n" * 100
    source = PRIVATE_SCRATCH_SOURCE.replace(original, work).replace("!$omp end do", "!$omp end do nowait")
    source = source.replace("a(i)=b(i)+1.0_8", "a(i)=a(i)+b(i)+1.0_8")
    directory = tmp_path_factory.mktemp("large_native_team")
    path, outputs, report = emit(directory / "input", source)
    assert not report["scopes"][0]["boundaries"]
    driver = """program exercise
use graph_fixture
implicit none
real(8)::a(-3:4),b(-3:4)
integer::x(-8:-1),i
do i=-3,4
a(i)=real(i+4,8)
enddo
b=-99
x=0
call advance(a,b,x,8)
if(any(x/=1))error stop 'native team must execute once'
do i=-3,4
if(a(i)/=204*real(i+4,8))error stop 'large native output'
if(b(i)/=2*real(i+4,8))error stop 'GPU producer'
enddo
print *, 'LARGE_NATIVE_FIELDS_OK'
end program
"""
    return build_binaries(directory, path, outputs, report, driver)


@pytest.mark.native
def test_large_native_team_original_fallback_executes_once(large_native_binaries):
    directory, candidate, native = large_native_binaries
    for binary in (native, candidate):
        result = run([str(binary)], cwd=directory,
                     env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"})
        assert "LARGE_NATIVE_FIELDS_OK" in result.stdout


@pytest.mark.cuda
def test_large_native_team_keeps_checked_gpu_children_and_mirrors(large_native_binaries):
    directory, candidate, _ = large_native_binaries
    result = run([str(candidate)], cwd=directory,
                 env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"})
    (directory / "gpu.stdout").write_text(result.stdout)
    (directory / "gpu.stderr").write_text(result.stderr)
    assert "LARGE_NATIVE_FIELDS_OK" in result.stdout
    lines = result.stderr.splitlines()
    assert sum(line.startswith("FORT_SCOPED launch ") for line in lines) == 2, result.stderr
    assert sum(line.startswith("FORT_SCOPED domain ") and '"event":"domain_retire"' in line for line in lines) == 1
