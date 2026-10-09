"""Qualified numerical companions reuse an existing team without changing run."""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig
from compiler.tests.test_scoped_planning_entries import calibration
from compiler.tests.test_structured_offload import SOURCE

NESTED_SOURCE = '''module nested_case
contains
subroutine advance(a,n,outer,inner)
real(8),intent(inout)::a(:)
integer,intent(in)::n
logical,intent(in)::outer,inner
integer::i
if(outer) then
 if(inner) then
  do i=1,n
   a(i)=a(i)+1
  enddo
 else
  do i=1,n
   a(i)=a(i)+2
  enddo
 endif
else
 do i=1,n
  a(i)=a(i)+3
 enddo
endif
end subroutine
end module
'''

PROTECTED_SOURCE = '''module guarded_case
contains
subroutine advance(a,n,flag,protected)
real(8),intent(inout)::a(:)
integer,intent(in)::n
logical,intent(in)::flag
real(8),intent(in)::protected
integer::i
do i=1,n
 if(flag) a(i)=protected
enddo
do i=1,n
 a(i)=a(i)+1
enddo
end subroutine
end module
'''


def emit(tmp_path, source=SOURCE, *, collective=True, profile=None, threads=4):
    path = tmp_path / "source.f90"
    path.write_text(source)
    function, plan = prepare_function(lower_file(path, "advance"), options=CompilerOptions(gpu_policy="sections"))
    return generate_scoped(function, plan, OffloadConfig("sections", profile, threads, collective),
                           "common_functions.cuh", runtime_id=read_scoped_runtime()[1]["runtime_id"])


def team_text(emission):
    text = emission.cuda.split('extern "C" int ' + emission.report["team"]["entry"], 1)[1].split(
        'extern "C" int ' + emission.report["planning"]["entry"], 1)[0]
    return re.sub(r"fort_v\d+_", "", text)


def serial_text(emission):
    value = emission.cuda.split('extern "C" int ' + emission.report["entry"], 1)[1]
    delimiter = "\n    return FORT_SCOPE_OK;\n}\n"
    return value.split(delimiter, 1)[0]


def state_text(emission):
    value = emission.cuda.split("_team_state {", 1)[1].split("\n};", 1)[0]
    return re.sub(r"fort_v\d+_", "", value)


def test_team_companion_is_additive_and_does_not_change_serial_execution(tmp_path):
    serial = emit(tmp_path, collective=False)
    qualified = emit(tmp_path)
    assert "team" not in serial.report
    assert "run_team" not in serial.fortran
    assert serial.report["entry"] == qualified.report["entry"]
    assert serial.report["fortran_module"] == qualified.report["fortran_module"]
    assert serial_text(serial) == serial_text(qualified)
    assert serial.report["planning"]["units"] == qualified.report["planning"]["units"]
    assert qualified.cuda.count("__global__") == serial.cuda.count("__global__")
    contract = qualified.report["team"]
    assert contract["entry"] == qualified.report["entry"] + "_team"
    assert contract["participation"] == "qualified_full_team"
    assert contract["expected_omp_level"] == 1
    assert contract["coordinator"] == "master"
    assert "same context" in contract["captures"]
    assert "public :: run, run_team, plan, choose" in qualified.fortran
    assert "function run_team(" in qualified.fortran
    assert "bind(C, name='" + contract["entry"] + "')" in qualified.fortran


def test_team_metadata_precedes_barriers_and_descriptor_payload_queries(tmp_path):
    emission = emit(tmp_path, threads=36)
    text = team_text(emission)
    guard = "omp_get_level() != 1 || omp_get_num_threads() != 36"
    assert text.index(guard) < text.index("#pragma omp single copyprivate(shared)")
    assert text.index("#pragma omp single copyprivate(shared)") < text.index("fort_scope_layout_get(")
    assert text.index("#pragma omp barrier") < text.index("const double &scale = *fort_scalar_scale")
    assert "requires OpenMP support" in text
    assert "#pragma omp parallel" not in text
    assert "new " not in text
    assert "shared = &local_state" in text
    assert text.rfind("#pragma omp barrier") < text.rfind("return fort_result;")
    assert "offload::thread_id(), offload::team_size()" in text
    assert "cpu_0(" in text
    assert "cpu_1(" in text
    assert "if (omp_in_parallel())" not in text


def test_shared_preparation_and_status_are_published_before_workers(tmp_path):
    emission = emit(tmp_path)
    state = state_text(emission)
    assert "double coefficient{}" in state
    assert "int limit{}" in state
    assert "scale" not in state
    text = team_text(emission)
    assert "auto &coefficient = shared->coefficient" in text
    assert "auto &limit = shared->limit" in text
    assert "shared->status = [&]() -> int" in text
    # A late participant must enter each status guard before master changes
    # status; the first barrier is as important as its publication barrier.
    assert "#pragma omp barrier\n            #pragma omp master" in text
    assert text.index("shared->cpu_active = true") < text.index("cpu_0(")
    assert text.index("cpu_0(") < text.index("shared->status = fort_access.finish()")
    assert text.count("shared->status = fort_access.finish()") == 2


def test_nested_branch_choices_use_distinct_shared_fields(tmp_path):
    emission = emit(tmp_path, NESTED_SOURCE)
    text = team_text(emission)
    assert "shared->condition_0 = outer" in text
    assert "shared->condition_1 = inner" in text
    assert "if (shared->condition_0)" in text
    assert "if (shared->condition_1)" in text
    assert "shared->condition =" not in text
    assert text.count("offload::thread_id(), offload::team_size()") == 3


def test_protected_scalar_is_borrowed_and_only_guarded_worker_remains_native(tmp_path):
    emission = emit(tmp_path, PROTECTED_SOURCE)
    assert emission.report["region_execution"][0]["protected_scalars"] == ["protected"]
    text = team_text(emission)
    assert "const double &protected = *fort_scalar_protected" in text
    assert "native-scoped-protected-scalar" in text
    assert "bool fort_gpu = false" in text
    assert "bool fort_gpu = fort_gpu_requested" in text
    assert "const volatile double &protected" in re.sub(r"fort_v\d+_", "", emission.cuda)
    state = state_text(emission)
    assert "protected" not in state


def test_serial_calibration_cannot_enable_collective_automatic_estimates(tmp_path):
    profile = calibration()
    serial = emit(tmp_path, collective=False, profile=profile)
    qualified = emit(tmp_path, profile=profile)
    assert serial.report["automatic_estimate_available"]
    assert not qualified.report["automatic_estimate_available"]
    assert qualified.report["planning"]["query_available"]
    assert not qualified.report["planning"]["profile_available"]
    assert qualified.report["automatic_reason"] == "collective synchronization calibration is unavailable"
    assert "costs.valid = 1" not in qualified.cuda


def test_definition_event_runs_once_in_coordinator_initialization(tmp_path):
    source = PROTECTED_SOURCE.replace("intent(inout)::a", "intent(out)::a")
    emission = emit(tmp_path, source)
    text = team_text(emission)
    assert text.count("fort_scope_forget_definition(") == 1
    assert text.index("shared->status = [&]() -> int") < text.index("fort_scope_forget_definition(")
    assert text.index("fort_scope_forget_definition(") < text.index("auto *a = shared->a")


@pytest.mark.native
def test_team_public_fortran_interface_compiles(tmp_path):
    compiler = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not compiler:
        pytest.skip("Fortran compiler unavailable")
    emission = emit(tmp_path, PROTECTED_SOURCE)
    runtime, _ = read_scoped_runtime()
    for name, source in (("fort_scoped_memory.f90", runtime["fort_scoped_memory.f90"]),
                         ("shared_interface.f90", emission.fortran)):
        path = tmp_path / name
        path.write_text(source)
        result = subprocess.run([compiler, "-std=f2018", "-fopenmp", "-c", str(path)],
                                cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
