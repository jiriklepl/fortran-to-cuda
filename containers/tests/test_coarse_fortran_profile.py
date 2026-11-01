"""Compile-time exclusions must remove inlined helper callbacks without changing FP state."""
import csv
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_original_dwarf_excludes_inlined_fortran_helpers_before_codegen(tmp_path):
    if not shutil.which("gfortran-15") or not shutil.which("g++-14"):
        pytest.skip("actual diagnostic Fortran/C++ toolchain required")
    spec = importlib.util.spec_from_file_location("coarse_fortran", ROOT / "containers/profiling/coarse_fortran.py")
    coarse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coarse)
    budget_spec = importlib.util.spec_from_file_location("budget", ROOT / "elmm-pipeline/elmm_pipeline/campaigns/cpu_budget.py")
    budget = importlib.util.module_from_spec(budget_spec)
    budget_spec.loader.exec_module(budget)
    source = tmp_path / "probe.f90"
    source.write_text("""module original_phases
implicit none
contains
pure elemental real(8) function scalar_helper(x) result(y)
real(8), intent(in) :: x
y=x+0.5_8*x
end function
subroutine convection(total)
real(8), intent(out) :: total
integer :: i
total=0
do i=1,2048
total=total+scalar_helper(real(i,8))
enddo
end subroutine
end module
program original_main
use original_phases
use iso_c_binding
implicit none
interface
subroutine begin_check() bind(c)
end subroutine
integer(c_int) function end_check() bind(c)
import c_int
end function
end interface
real(8) :: total
call begin_check()
call convection(total)
if(end_check()/=0) stop 1
print '(Z16.16)',transfer(total,0_c_int64_t)
end program
""")
    check = tmp_path / "check.cpp"
    check.write_text("""#include <cfenv>
#include <cerrno>
extern "C" void begin_check() {std::feclearexcept(FE_ALL_EXCEPT);std::fesetround(FE_DOWNWARD);std::feraiseexcept(FE_DIVBYZERO);errno=EDOM;}
extern "C" int end_check() {return errno!=EDOM||std::fegetround()!=FE_DOWNWARD||std::fetestexcept(FE_ALL_EXCEPT)!=FE_DIVBYZERO;}
""")
    commands = [
        ["gfortran-15", "-g", "-O3", "-c", str(source), "-o", str(tmp_path / "original.o")],
        ["g++-14", "-std=c++17", "-O2", "-c", str(check), "-o", str(tmp_path / "check.o")],
        ["g++-14", "-std=c++17", "-O2", "-c", str(ROOT / "containers/profiling/budget_timer.cpp"), "-o", str(tmp_path / "timer.o")],
    ]
    for argv in commands:
        subprocess.run(argv, cwd=tmp_path, capture_output=True, check=True, timeout=120)
    selection = coarse.selection(source, tmp_path / "original.o", budget.category)
    assert "convection" in selection["retained"]
    assert "scalar_helper" in selection["excluded"]
    assert any(row["name"] == "scalar_helper" and row["inlined"] for row in selection["procedures"])
    outputs, counts = [], []
    for name, options in (("plain", []), ("all", ["-finstrument-functions"]),
                          ("coarse", ["-finstrument-functions", "-finstrument-functions-exclude-function-list=" + ",".join(selection["excluded"])])):
        binary = tmp_path / name
        subprocess.run(["gfortran-15", "-g", "-O3", *options, str(source), str(tmp_path / "check.o"),
                        str(tmp_path / "timer.o"), "-lstdc++", "-o", str(binary)], cwd=tmp_path,
                       capture_output=True, check=True, timeout=120)
        path = tmp_path / (name + ".csv")
        result = subprocess.run([str(binary)], cwd=tmp_path, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "FORT_BUDGET_OUTPUT": str(path)},
                                capture_output=True, check=True, timeout=30)
        outputs.append(result.stdout)
        if name == "plain":
            continue
        lines = path.read_text().splitlines()
        counts.append(len(list(csv.DictReader(line for line in lines[1:] if not line.startswith("#")))))
        symbols = tmp_path / (name + ".nm")
        symbols.write_bytes(subprocess.check_output(["nm", "-an", str(binary)]))
        budget.analyze(path, symbols)
    assert len(set(outputs)) == 1
    assert counts == [2051, 3]


def test_interface_only_original_object_is_reused_without_instrumentation(tmp_path):
    if not shutil.which("gfortran-15"):
        pytest.skip("actual diagnostic Fortran toolchain required")
    spec = importlib.util.spec_from_file_location("coarse_fortran", ROOT / "containers/profiling/coarse_fortran.py")
    coarse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coarse)
    source, original = tmp_path / "interfaces.f90", tmp_path / "interfaces.o"
    source.write_text("module interfaces\ninterface\nsubroutine external_routine()\nend subroutine\nend interface\nend module\n")
    subprocess.run(["gfortran-15", "-g", "-O3", "-c", str(source), "-o", str(original)], cwd=tmp_path,
                   capture_output=True, check=True, timeout=120)
    chosen = coarse.selection(source, original, lambda _symbol: None)
    assert chosen["compile_instrumentation"] is False
    assert chosen["original_object_has_no_code"] is True
