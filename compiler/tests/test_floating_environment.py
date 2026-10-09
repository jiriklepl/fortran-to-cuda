"""Runtime initialization preserves the original thread's flags and trap mask."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.native
def test_runtime_environment_restores_flags_rounding_and_traps(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("requires a native C++ compiler")
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    source = tmp_path / "floating.cpp"
    source.write_text(r'''
#include "floating_environment.hpp"
#include <cassert>
#include <stdexcept>
int main() {
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INEXACT);
    std::fesetround(FE_DOWNWARD);
#if defined(__GLIBC__)
    const int original_traps = feenableexcept(FE_INVALID | FE_DIVBYZERO);
#endif
    const int original_flags = std::fetestexcept(FE_ALL_EXCEPT);
    assert(!fort_runtime::numerical_environment_supported());
    try {
        fort_runtime::HostFloatingEnvironment state;
        assert(state.valid());
        assert(std::fetestexcept(FE_ALL_EXCEPT) == 0);
#if defined(__GLIBC__)
        assert(fegetexcept() == 0);
#endif
        // CUDA initialization and decision arithmetic are foreign to the
        // numerical source. Neither their flags nor their mode changes escape.
        std::feraiseexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
        std::fesetround(FE_UPWARD);
        { fort_runtime::HostFloatingEnvironment nested; assert(nested.valid()); }
        throw 1;
    } catch (int) {}
    assert(std::fegetround() == FE_DOWNWARD);
    assert(std::fetestexcept(FE_ALL_EXCEPT) == original_flags);
#if defined(__GLIBC__)
    assert(fegetexcept() == (original_traps | FE_INVALID | FE_DIVBYZERO));
    fedisableexcept(FE_ALL_EXCEPT);
#endif
    std::fesetround(FE_TONEAREST);
#if defined(__GLIBC__)
    assert(fort_runtime::numerical_environment_supported());
#endif
}
''')
    binary = tmp_path / "floating"
    for command in ([compiler, "-std=c++17", "-O2", "-I", str(runtime), str(source), "-o", str(binary)],
                    [str(binary)]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
