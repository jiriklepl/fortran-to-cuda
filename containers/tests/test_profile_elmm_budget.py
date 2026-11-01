"""The diagnostic wall partition must survive real nested Fortran/OpenMP callbacks."""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "elmm-pipeline"))
from elmm_pipeline.campaigns.cpu_budget import analyze


@pytest.fixture(scope="module", params=[False, True], ids=["all-callbacks", "coarse-filter"])
def observed(tmp_path_factory, request):
    if not shutil.which("gfortran-15") or not shutil.which("g++-14"):
        pytest.skip("actual GCC15 Fortran/GCC14 C++ diagnostic toolchain required")
    root = tmp_path_factory.mktemp("budget-callback-proof")
    source = root / "probe.f90"
    source.write_text('''program callback_proof
use iso_c_binding
implicit none
interface
  subroutine begin_check() bind(c)
  end subroutine
  integer(c_int) function end_check() bind(c)
    import c_int
  end function
end interface
integer :: failures
call begin_check()
call outer()
if (end_check()/=0) stop 1
failures=0
!$omp parallel reduction(+:failures)
call begin_check()
call outer()
if (end_check()/=0) failures=failures+1
!$omp end parallel
if (failures/=0) stop 2
print *, 'CALLBACK_FP_ERRNO_OK'
contains
subroutine outer()
  call inner()
end subroutine
subroutine inner()
  integer :: i, n
  n=0
  do i=1,20000
    n=mod(n+i,99991)
  end do
  if (n<0) stop 3
end subroutine
end program
''')
    check = root / "check.cpp"
    check.write_text('''#include <cfenv>
#include <cerrno>
extern "C" void begin_check() {
  std::feclearexcept(FE_ALL_EXCEPT); std::fesetround(FE_DOWNWARD);
  std::feraiseexcept(FE_DIVBYZERO); errno=EDOM;
}
extern "C" int end_check() {
  return errno!=EDOM || std::fegetround()!=FE_DOWNWARD || std::fetestexcept(FE_ALL_EXCEPT)!=FE_DIVBYZERO;
}
''')
    commands = [
        ["g++-14", "-std=c++17", "-O2", "-fno-instrument-functions", "-c", str(HERE / "profiling/budget_timer.cpp"), "-o", str(root / "timer.o")],
        ["g++-14", "-std=c++17", "-O2", "-fno-instrument-functions", "-c", str(check), "-o", str(root / "check.o")],
        ["gfortran-15", "-O3", "-fopenmp", "-finstrument-functions", str(source), str(root / "timer.o"), str(root / "check.o"), "-lstdc++", "-o", str(root / "probe")],
    ]
    if request.param:
        commands.insert(2, ["g++-14", "-std=c++17", "-O2", "-fno-instrument-functions", "-c", str(HERE / "profiling/budget_filter.cpp"), "-o", str(root / "filter.o")])
        commands[-1].extend([str(root / "filter.o"), "-Wl,--wrap=__cyg_profile_func_enter", "-Wl,--wrap=__cyg_profile_func_exit"])
    for argv in commands:
        subprocess.run(argv, check=True, capture_output=True, timeout=120)
    csv_path, symbols = root / "events.csv", root / "symbols.txt"
    symbols.write_bytes(subprocess.check_output(["nm", "-an", str(root / "probe")]))
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "CUDA_VISIBLE_DEVICES": "", "FORT_BUDGET_OUTPUT": str(csv_path)}
    if request.param:
        table = [line.split() for line in symbols.read_text().splitlines() if len(line.split()) == 3]
        anchor = next(row[0] for row in table if row[2] == "fort_budget_anchor")
        selected = [row[0] for row in table if row[2] == "main" or any(word in row[2] for word in ("MAIN__", "outer.", "inner."))]
        selected_file = root / "selection.txt"
        selected_file.write_text("\n".join([anchor, *selected]) + "\n")
        environment["FORT_BUDGET_FILTER"] = str(selected_file)
    result = subprocess.run([str(root / "probe")], env=environment,
                            capture_output=True, text=True, check=True, timeout=30)
    assert "CALLBACK_FP_ERRNO_OK" in result.stdout
    return csv_path, symbols


def test_instrumented_fortran_preserves_fp_errno_and_partitions_main_wall(observed):
    budget, timeline = analyze(*observed)
    main = budget["threads"][str(budget["main_tid"])]
    assert len(budget["threads"]) == 4
    assert sum(row["exclusive_wall_ns"] for row in main["categories"].values()) == main["root_wall_ns"]
    assert all(row["exclusive_cpu_ns"] >= 0 for thread in budget["threads"].values() for row in thread["categories"].values())
    assert any(row["parent"] != 2**64-1 for row in (event["args"] for event in timeline["traceEvents"]))
    assert budget["performance_sample"] is False


def test_incomplete_timer_or_wrong_elf_never_qualifies(observed, tmp_path):
    csv_path, symbols = observed
    changed = tmp_path / "events.csv"
    changed.write_text(csv_path.read_text().replace("clock_errors=0", "clock_errors=1", 1))
    with pytest.raises(ValueError, match="failed clocks"):
        analyze(changed, symbols)
    wrong = tmp_path / "symbols.txt"
    wrong.write_text("")
    with pytest.raises(ValueError, match="ELF anchor"):
        analyze(csv_path, wrong)


def test_typed_forwarding_rejects_unsupported_callback_signature(tmp_path):
    spec = importlib.util.spec_from_file_location("budget_signatures", HERE / "profiling/cabi_signatures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source.cu"
    source.write_text('extern "C" int fort_scope_bad(void (*callback)(int)) { return 0; }')
    with pytest.raises(ValueError, match="unsupported"):
        module.exports(source)
