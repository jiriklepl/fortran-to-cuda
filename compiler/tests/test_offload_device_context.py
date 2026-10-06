"""Section dispatch retains the selected CUDA device across OpenMP workers."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.cuda.offload import generate_offload
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig


@pytest.mark.parametrize("collective", [False, True])
def test_section_emission_captures_device_before_execution_team(tmp_path, collective):
    source = tmp_path / "source.f90"
    source.write_text("""module device_case
contains
subroutine advance(a,b,n)
real(8),intent(inout)::a(:)
real(8),intent(in)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
 a(i)=b(i)*2.0_8
end do
end subroutine
end module
""")
    function, plan = prepare_function(
        lower_file(source, "advance"), options=CompilerOptions(gpu_policy="sections")
    )
    generated = generate_offload(function, plan, OffloadConfig("sections", host_threads=2, collective=collective))
    gpu = generated.helpers.split("_gpu(", 1)[1].split("_team(", 1)[0]
    assert gpu.index("offload::DeviceScope device_scope(d.device)") < gpu.index("offload::Allocation")
    team = generated.helpers.split("_team(", 1)[1]
    assert team.index("if (!prepared)") < team.index("cudaGetDevice(&shared->device)")
    assert "if (!prepared) delete shared;" in team
    body = "\n".join(generated.body)
    if collective:
        assert "#pragma omp parallel" not in body
        assert "#pragma omp barrier" in body
        assert "&check" not in body
    else:
        assert body.index("cudaGetDevice(&check.device)") < body.index("#pragma omp parallel num_threads(2)")
        assert "_team(" in body and ", &check);" in body


@pytest.mark.native
def test_device_scope_restores_each_executor_and_empty_units_stay_native(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++ compiler with OpenMP is required")
    runtime = Path(__file__).resolve().parents[1] / "runtime/offload.hpp"
    source = tmp_path / "device_context.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#define __CUDACC__
#define __CUDACC_VER_MAJOR__ 13
#define __CUDACC_VER_MINOR__ 0
#define __CUDACC_VER_BUILD__ 0
constexpr int cudaSuccess = 0;
constexpr int cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
struct cudaDeviceProp {
    struct { char bytes[16]; } uuid{};
    int major = 8, minor = 6;
};
thread_local int current_device = 0;
thread_local int switches = 0;
int cudaGetDevice(int *device) { *device = current_device; return cudaSuccess; }
int cudaSetDevice(int device) {
    assert(device == 0 || device == 1);
    current_device = device;
    ++switches;
    return cudaSuccess;
}
int cudaGetDeviceProperties(cudaDeviceProp *properties, int) {
    *properties = cudaDeviceProp{}; return cudaSuccess;
}
int cudaRuntimeGetVersion(int *version) { *version = 13000; return cudaSuccess; }
int cudaDriverGetVersion(int *version) { *version = 13000; return cudaSuccess; }
int cudaMemcpy(void *, const void *, std::size_t, int) { return cudaSuccess; }
int cudaMemcpy2D(void *, std::size_t, const void *, std::size_t,
                 std::size_t, std::size_t, int) { return cudaSuccess; }
#define CUCH(call) do { if ((call) != cudaSuccess) std::abort(); } while (0)
namespace generated_kernels::storage {
enum class AllocationPolicy { pooled };
namespace allocation_detail {
struct DeviceAllocation {
    DeviceAllocation(std::size_t, AllocationPolicy) {}
    void *get() const { return nullptr; }
    void release_completed() {}
};
}
[[noreturn]] void fail(const char *) { std::abort(); }
void trace(const char *, std::size_t) {}
}
"""
        + f'#include "{runtime}"\n'
        + r"""
int main() {
    using namespace generated_kernels::offload;
    omp_set_dynamic(0);
    Data selected;
    CUCH(cudaSetDevice(1));
    CUCH(cudaGetDevice(&selected.device)); // Serial caller, before its team.
    #pragma omp parallel num_threads(2) shared(selected)
    {
        assert(omp_get_num_threads() == 2);
        const int executor_device = omp_get_thread_num();
        CUCH(cudaSetDevice(executor_device));
        {
            DeviceScope scope(selected.device);
            assert(current_device == 1);
            {
                DeviceScope nested(0);
                assert(current_device == 0);
            }
            assert(current_device == 1);
        }
        assert(current_device == executor_device);
        try {
            DeviceScope scope(1 - executor_device);
            assert(current_device == 1 - executor_device);
            throw 7;
        } catch (int value) {
            assert(value == 7);
        }
        assert(current_device == executor_device);
        const int before = switches;
        { DeviceScope already_current(executor_device); }
        { DeviceScope no_selected_device(-1); }
        assert(switches == before);
    }
    // A protected scalar in an empty unit must never reach the numerical
    // value ABI merely because another independent unit has active work.
    Profile profile;
    profile.valid = true;
    Data data;
    data.units.resize(2);
    data.units[0].iterations = 0;
    data.units[1].iterations = 10;
    data.units[1].flops = 1000;
    profile.cpu_flops = 1;
    profile.gpu_flops = 1000;
    for (bool automatic : {false, true}) {
        const auto result = select(data, profile, automatic);
        assert(!result.has_gpu() && result.choices.size() == 2);
    }
    data.units[0].iterations = 1;
    assert(select(data, profile, false).has_gpu());
}
"""
    )
    binary = tmp_path / "device_context"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-fopenmp", str(source), "-o", str(binary)],
        capture_output=True, text=True, timeout=60,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run(
        [str(binary)], env=dict(os.environ, OMP_DYNAMIC="FALSE"),
        capture_output=True, text=True, timeout=30,
    )
    assert executed.returncode == 0, executed.stdout + executed.stderr
