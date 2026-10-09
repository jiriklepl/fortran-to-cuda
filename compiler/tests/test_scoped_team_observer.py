"""Offline exclusive observations cannot charge peer waits as coordination."""

import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
SOURCE = r"""
#define FORT_SCOPE_TEAM_OBSERVER_IMPLEMENTATION
#include "scoped_team_observer.hpp"
#include <omp.h>
#include <chrono>
#include <thread>
#include <cassert>
#include <cstdio>
int main() {
 assert(fort_scope_team_fortran_compatible_v1("GCC15","-O3 -fopenmp -I /first -J/one -o a.o", "GCC15","-O3\x1f-fopenmp"));
 assert(!fort_scope_team_fortran_compatible_v1("GCC14","-O3 -fopenmp", "GCC15","-O3\x1f-fopenmp"));
 assert(!fort_scope_team_fortran_compatible_v1("GCC15","-O2 -fopenmp", "GCC15","-O3\x1f-fopenmp"));
#ifdef FORT_SCOPE_CALIBRATION
 assert(fort_scope_team_observer_enabled_v1()==1);
 assert(!fort_scope_team_observer_reset_v1(4));
 #pragma omp parallel num_threads(4)
 {
  fort_scoped::TeamObservation owner(FORT_SCOPE_TEAM_OBSERVE_OWNER);
  if (omp_get_thread_num()==0)
  {
   fort_scoped::TeamObservation api(FORT_SCOPE_TEAM_OBSERVE_API);
   std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }
  #pragma omp barrier
 }
 double owner=0,api=0;uint64_t calls=0;
 assert(!fort_scope_team_observer_read_v1(FORT_SCOPE_TEAM_OBSERVE_OWNER,4,&owner,&calls));assert(calls==1);
 assert(!fort_scope_team_observer_read_v1(FORT_SCOPE_TEAM_OBSERVE_API,4,&api,&calls));assert(calls==1);
 assert(api>0.02&&owner<api*0.5); // Waiting peers must not reintroduce API time.
 assert(!fort_scope_team_observer_reset_v1(4));
 fort_scope_team_observe_begin_v1(FORT_SCOPE_TEAM_OBSERVE_ENTRY);
 assert(fort_scope_team_observer_reset_v1(4));
 fort_scope_team_observe_end_v1(FORT_SCOPE_TEAM_OBSERVE_CPU_WORKER);
 assert(fort_scope_team_observer_read_v1(FORT_SCOPE_TEAM_OBSERVE_ENTRY,4,&owner,&calls));
#else
 assert(fort_scope_team_observer_enabled_v1()==0);
 fort_scoped::TeamObservation disabled(FORT_SCOPE_TEAM_OBSERVE_OWNER);
 double seconds=0;uint64_t calls=0;
 assert(fort_scope_team_observer_read_v1(1,4,&seconds,&calls));
#endif
 std::puts("OBSERVER_OK");
}
"""


@pytest.mark.parametrize("calibration", [False, True])
def test_nested_exclusive_offline_observer_and_production_compileaway(tmp_path, calibration):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ OpenMP compiler unavailable")
    source = tmp_path / "observer.cpp"
    source.write_text(SOURCE)
    binary = tmp_path / "observer"
    flags = ["-DFORT_SCOPE_CALIBRATION"] if calibration else []
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fopenmp",
            *flags,
            "-I",
            str(RUNTIME),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OBSERVER_OK" in result.stdout
