"""Fixed module metadata may remain native without acquiring device authority."""

import os

import pytest

from compiler.tests.test_graph_source_owner import NATIVE_HELPER_SOURCE, emit
from compiler.tests.test_source_scopes import run
from compiler.tests.test_storage_epoch_runtime import build_binaries

SOURCE = NATIVE_HELPER_SOURCE.replace("integer,allocatable::types(:),other_types(:)",
                                      "integer::types(8),other_types(8)")


def test_fixed_native_inputs_keep_domain_and_reserve_each_original_module_array(tmp_path):
    path, outputs, report = emit(tmp_path, SOURCE)
    owner, = report["scopes"]
    assert not owner["boundaries"]
    helper = next(item for item in owner["module_coordinators"] if item["procedure"].endswith("::smooth"))
    assert not helper["boundaries"]
    native = [op for segment in helper["planning_segments"] for op in segment["operations"]["native_operations"]]
    assert any(op["native_opaque_accesses"] for op in native)
    epoch = owner["checked_execution_plan"]["storage_epoch"]
    reservations = [resource for resource in epoch["resources"] if resource["resource"] in {
        "graph_fixture::types", "graph_fixture::other_types"}]
    assert len(reservations) == 2
    assert all(resource["device_replica_authorized"] is False for resource in reservations)
    assert all(resource["initialization_authorized"] is False for resource in reservations)
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert "shape(types, kind=c_size_t)" in text
    assert "shape(other_types, kind=c_size_t)" in text
    assert "FORT_SCOPE_BYTES" in text


@pytest.mark.parametrize("declaration", [
    "integer,target::types(8),other_types(8)",
    "integer,pointer::types(:),other_types(:)",
    "integer,volatile::types(8),other_types(8)",
    "integer::types(8),other_types(8)\ncommon /shared/ types",
    "integer::types(8),other_types(8)\n!$omp threadprivate(types)",
])
def test_uncertain_native_storage_still_cuts_before_original_execution(tmp_path, declaration):
    source = SOURCE.replace("integer::types(8),other_types(8)", declaration)
    _path, _outputs, report = emit(tmp_path, source)
    assert not report["scopes"] or any(owner["boundaries"] or any(
        child["boundaries"] for child in owner["module_coordinators"]) for owner in report["scopes"])


def test_fixed_native_reservation_never_authorizes_a_later_write(tmp_path):
    source = SOURCE.replace("call smooth(b,other_types,n)", "call smooth(b,other_types,n)\ntypes(1)=7")
    _path, _outputs, report = emit(tmp_path, source)
    owner, = report["scopes"]
    assert owner["boundaries"]


@pytest.fixture(scope="module")
def fixed_host_binaries(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fixed_host_arrays")
    path, outputs, report = emit(directory / "input", SOURCE)
    driver = """program exercise
use graph_fixture
implicit none
real(8)::a(8),b(8)
integer::x(8),i
types=0
other_types=1
do i=1,8
a(i)=real(i*i,8)
enddo
b=-99
x=0
call advance(a,b,x,8)
do i=1,8
if(a(i)/=2*real(i*i,8)+1)error stop 'fixed metadata result'
if(b(i)/=2*real(i*i,8))error stop 'fixed metadata producer'
enddo
if(any(types/=0).or.any(other_types/=1))error stop 'metadata changed'
print *, 'FIXED_HOST_ARRAYS_OK'
end program
"""
    return build_binaries(directory, path, outputs, report, driver)


@pytest.mark.native
def test_original_and_hidden_cuda_keep_fixed_native_values(fixed_host_binaries):
    directory, candidate, native = fixed_host_binaries
    for binary in (native, candidate):
        result = run([str(binary)], cwd=directory,
                     env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"})
        assert "FIXED_HOST_ARRAYS_OK" in result.stdout


@pytest.mark.cuda
def test_fixed_native_inputs_between_gpu_children_keep_the_domain(fixed_host_binaries):
    directory, candidate, _native = fixed_host_binaries
    result = run([str(candidate)], cwd=directory,
                 env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"})
    (directory / "gpu.stdout").write_text(result.stdout)
    (directory / "gpu.stderr").write_text(result.stderr)
    assert "FIXED_HOST_ARRAYS_OK" in result.stdout
    lines = result.stderr.splitlines()
    assert sum(line.startswith("FORT_SCOPED launch ") for line in lines) == 2
    assert sum(line.startswith("FORT_SCOPED domain ") and '"event":"domain_retire"' in line for line in lines) == 1
