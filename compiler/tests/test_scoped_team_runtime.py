"""Real public team entries execute once and fail uniformly without replay."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from compiler.emission.common.resources import read_scoped_runtime

ROOT = Path(__file__).resolve().parents[2]

SOURCE = '''module numerical_team_case
contains
subroutine advance(a,b,configuration,n,outer,inner,protected)
real(8),intent(inout)::a(:)
real(8),intent(in)::b(:),protected
integer,intent(in)::configuration(:),n
logical,intent(in)::outer,inner
logical::choose_inner
integer::i,limit
real(8)::coefficient
limit=configuration(1)
coefficient=2.0_8
choose_inner=inner
if(outer) then
 if(choose_inner.and.limit>0) then
  do i=-2,n-5
   a(i+4)=a(i+4)+b(i+4)*coefficient+real(i,8)
  enddo
 else
  do i=-2,n-5
   a(i+4)=a(i+4)+b(i+4)*coefficient-real(i,8)
  enddo
 endif
endif
do i=2,min(2,n-1)
 a(i)=a(i)+1
enddo
do i=2,n-1
 a(i)=a(i)+1
enddo
end subroutine
subroutine guarded(a,n,flag,protected)
real(8),intent(inout)::a(:)
integer,intent(in)::n
logical,intent(in)::flag
real(8),intent(in)::protected
integer::i
do i=2,n-1
 if(flag) a(i)=protected
enddo
do i=2,n-1
 a(i)=a(i)+1
enddo
end subroutine
end module
'''

REFERENCE = '''subroutine native_reference(a,b,configuration,n,outer,inner,protected) bind(C)
use iso_c_binding
use numerical_team_case,only:advance
integer(c_int),intent(in)::n,configuration(1)
real(c_double),intent(inout)::a(n)
real(c_double),intent(in)::b(n),protected
logical(c_bool),intent(in)::outer,inner
logical::native_outer,native_inner
native_outer=outer
native_inner=inner
call advance(a,b,configuration,n,native_outer,native_inner,protected)
end subroutine
subroutine native_guarded_reference(a,n,flag,protected) bind(C)
use iso_c_binding
use numerical_team_case,only:guarded
integer(c_int),intent(in)::n
real(c_double),intent(inout)::a(n)
real(c_double),intent(in)::protected
logical(c_bool),intent(in)::flag
logical::native_flag
native_flag=flag
call guarded(a,n,native_flag,protected)
end subroutine
'''

DRIVER = r'''#include <cuda_runtime.h>
#include <omp.h>
#include <sys/mman.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include "scoped_runtime.h"
extern "C" int @TEAM@(fort_scope_t,int,fort_buffer_t,fort_buffer_t,fort_buffer_t,
                      const int*,const bool*,const bool*,const double*);
extern "C" int @PLAN@(fort_scope_t,fort_buffer_t,fort_buffer_t,fort_buffer_t,
                      const int*,const bool*,const bool*);
extern "C" int @CHOOSE@(fort_scope_t,fort_scope_plan_decision*);
extern "C" int @GUARDED_TEAM@(fort_scope_t,int,fort_buffer_t,const int*,const bool*,const double*);
extern "C" void native_reference(double*,const double*,const int*,const int*,
                                 const bool*,const bool*,const double*);
extern "C" void native_guarded_reference(double*,const int*,const bool*,const double*);
static void check(int status) {
    if (status) { std::fprintf(stderr,"runtime status %d: %s\n",status,fort_scope_error()); std::exit(2); }
}
static void require(bool value,const char *reason) {
    if (!value) { std::fprintf(stderr,"assertion: %s\n",reason); std::exit(3); }
}
static fort_buffer_t bind(fort_scope_t context,void *pointer,size_t count,
                          uint64_t identity,uint32_t type,int initialized) {
    const int64_t lower=-4;
    const fort_scope_layout layout{1,type,type==FORT_SCOPE_REAL64?sizeof(double):sizeof(int),
                                   pointer,&count,&lower,1};
    fort_buffer_t result=0;
    check(fort_scope_register(context,identity,1,&layout,initialized,&result));
    return result;
}
static void install_test_plan(fort_scope_t context,fort_buffer_t ha,fort_buffer_t hb,int n,
                             uint64_t first_unit,bool inconsistent) {
    // Synthetic costs select a known first worker solely for failure testing.
    // No timing, hardware estimate, or performance claim uses this schedule.
    size_t lo=1,hi=static_cast<size_t>(n-1);
    const fort_scope_section section{&lo,&hi};
    fort_scope_access a{},b{};
    a.read_count=a.write_count=a.overwrite_count=1;
    a.reads=a.writes=a.overwrites=&section;
    b.read_count=1;b.reads=&section;
    size_t face_lo=1,face_hi=2;
    const fort_scope_section face{&face_lo,&face_hi};
    fort_scope_access thin=a;thin.reads=thin.writes=thin.overwrites=&face;
    const fort_scope_plan_binding first[]={{ha,a},{hb,b}},middle[]={{ha,thin}},last[]={{ha,a}};
    check(fort_scope_plan_reset(context));
    check(fort_scope_plan_add(context,FORT_SCOPE_PLAN_WORKER,first_unit,first,2,1e12,1e6,1));
    check(fort_scope_plan_add(context,FORT_SCOPE_PLAN_WORKER,inconsistent?999ULL:@MIDDLE_UNIT@ULL,middle,1,1,8,1));
    if (!inconsistent)
        check(fort_scope_plan_add(context,FORT_SCOPE_PLAN_WORKER,@LAST_UNIT@ULL,last,1,1e12,1e6,1));
    check(fort_scope_plan_validate(context));
    fort_scope_plan_costs costs{};costs.version=FORT_SCOPE_PLANNING_ABI_VERSION;costs.valid=1;
    costs.max_allocation_bytes=1<<24;costs.cpu_flops=costs.cpu_bandwidth=1e6;
    costs.gpu_flops=costs.gpu_bandwidth=costs.h2d_bandwidth=costs.d2h_bandwidth=1e12;
    costs.create_seconds=costs.register_seconds=costs.host_access_seconds=costs.device_access_seconds=1e-9;
    costs.gpu_setup_seconds=costs.cold_driver_startup_seconds=costs.allocation_seconds=costs.release_seconds=1e-9;
    costs.wait_seconds=costs.planning_operation_seconds=1e-9;
    costs.launch_enqueue_seconds=1e-3;
    fort_scope_plan_decision decision{};
    check(fort_scope_plan_select(context,&costs,1,&decision));
    require(decision.gpu_units==(inconsistent?1:2),"synthetic correctness schedule must select GPU workers");
    require(decision.cpu_units==1,"thin middle worker must use the team CPU path");
}
int main(int argc,char **argv) {
    require(argc==2,"one mode required");
    const std::string label=argv[1];
    const bool gpu=label=="gpu"||label=="mixed"||label=="protected"||label=="badplan";
    int devices=0;
    if (gpu && (cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)) return 77;
    omp_set_dynamic(0);omp_set_max_active_levels(2);
    unsigned long long launches=0,uploads=0,downloads=0;
    int repetitions=0;
    for (int n:{0,19,27}) for (int repetition=0;repetition<2;++repetition) {
        if (!n && (label=="undefined"||label=="badplan")) continue;
        std::vector<double> a(n),b(n),reference(n),initial(n);
        for (int k=0;k<n;++k) { a[k]=k-20+repetition;b[k]=k+1; }
        reference=initial=a;
        int configuration=3;
        bool outer=label!="protected",inner=repetition==0;
        double protected_value=1.5;
        if (label=="protected") native_guarded_reference(reference.data(),&n,&outer,&protected_value);
        else native_reference(reference.data(),b.data(),&configuration,&n,&outer,&inner,&protected_value);
        void *guard=nullptr;
        const double *protected_input=&protected_value;
        if (label=="protected") {
            guard=mmap(nullptr,4096,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
            require(guard!=MAP_FAILED,"guarded inaccessible capture");
            protected_input=static_cast<const double*>(guard);
        }
        fort_scope_t context=0;check(fort_scope_create(0,&context));
        auto ha=bind(context,a.data(),n,1,FORT_SCOPE_REAL64,1);
        auto hb=bind(context,b.data(),n,2,FORT_SCOPE_REAL64,label=="undefined"?0:1);
        auto hc=bind(context,&configuration,1,3,FORT_SCOPE_INTEGER32,1);
        int mode=label=="cpu"||label=="undefined"?0:1;
        if (label=="cpu"||label=="auto"||label=="gpu"||label=="resource"||(label=="mixed"&&!n)) {
            check(fort_scope_plan_reset(context));
            check(@PLAN@(context,ha,hb,hc,&n,&outer,&inner));
            check(fort_scope_plan_validate(context));
        }
        if (label=="protected") {
            // The existing checked compiler query deliberately rejects a
            // protected body scalar. This explicit test contract has whole
            // initialized storage and literal/n bounds, so conservative full
            // effects validate definitions without evaluating that scalar.
            check(fort_scope_plan_reset(context));
            fort_scope_access all{};all.flags=FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL;
            const fort_scope_plan_binding binding{ha,all};
            check(fort_scope_plan_add(context,FORT_SCOPE_PLAN_WORKER,@GUARDED_FIRST_UNIT@ULL,&binding,1,1,1,0));
            check(fort_scope_plan_add(context,FORT_SCOPE_PLAN_WORKER,@GUARDED_LAST_UNIT@ULL,&binding,1,1,1,1));
            check(fort_scope_plan_validate(context));
        }
        if (label=="auto") {
            mode=2;
            fort_scope_plan_decision decision{};check(@CHOOSE@(context,&decision));
            require(!decision.available&&!decision.gpu_units,"unprofiled collective choice must be native");
        }
        if (label=="resource") check(fort_scope_set_device_budget(context,0));
        if (label=="mixed") {
            mode=2;
            if (n) install_test_plan(context,ha,hb,n,inner?@FIRST_UNIT@ULL:@ELSE_UNIT@ULL,false);
            else { fort_scope_plan_decision decision{};check(@CHOOSE@(context,&decision)); }
        }
        if (label=="badplan") { mode=2;inner=true;install_test_plan(context,ha,hb,n,@FIRST_UNIT@ULL,true); }
        if (label=="argument") mode=9;
        const int threads=label=="budget"?3:4;
        std::vector<int> statuses(threads,-1);
        if (label=="serial") {
            statuses.assign(1,@TEAM@(context,mode,ha,hb,hc,&n,&outer,&inner,protected_input));
        } else if (label=="nested") {
            #pragma omp parallel num_threads(1)
            {
                #pragma omp parallel num_threads(4)
                {
                    statuses[omp_get_thread_num()]=@TEAM@(context,mode,ha,hb,hc,&n,&outer,&inner,protected_input);
                }
            }
        } else {
            #pragma omp parallel num_threads(threads)
            {
                const int tid=omp_get_thread_num();
                // Exercise a late participant on failing preparation paths.
                if (tid==threads-1) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                statuses[tid]=label=="protected"?@GUARDED_TEAM@(context,mode,ha,&n,&outer,protected_input):
                    @TEAM@(context,mode,ha,hb,hc,&n,&outer,&inner,protected_input);
            }
        }
        const int expected=(label=="serial"||label=="budget"||label=="nested")?FORT_SCOPE_BOUNDARY:
                           label=="undefined"?FORT_SCOPE_UNINITIALIZED:
                           label=="argument"?FORT_SCOPE_ARGUMENT:
                           label=="badplan"?FORT_SCOPE_EXECUTION:FORT_SCOPE_OK;
        for (int status:statuses) require(status==expected,"all participants must report the same expected status");
        if (label=="badplan") {
            // The first kernel has started. The failed entry cannot be replayed.
            require(fort_scope_wait(context)==FORT_SCOPE_EXECUTION,"later operations must see poisoned scope");
            check(fort_scope_abandon(context));
        } else {
            fort_scope_access all{};all.flags=FORT_SCOPE_READ_ALL;
            check(fort_scope_host_begin(context,ha,&all));check(fort_scope_host_end(context,ha));
            fort_scope_stats stats{};check(fort_scope_stats_get(context,&stats));
            launches+=stats.launches;uploads+=stats.upload_bytes;downloads+=stats.download_bytes;
            check(fort_scope_close(context));
            const auto &wanted=expected==FORT_SCOPE_OK?reference:initial;
            for (int k=0;k<n;++k) require(std::abs(a[k]-wanted[k])<1e-12,"full fields and preserved halos agree");
            if (label=="cpu"||label=="auto"||label=="resource"||expected) {
                require(!stats.launches&&!stats.upload_bytes&&!stats.download_bytes,"native/failure path must have zero CUDA work");
            } else {
                require(stats.launches==(!n?0:label=="protected"?1:label=="mixed"?2:3),"each eligible active GPU region launches once");
            }
        }
        if (guard) require(munmap(guard,4096)==0,"release protected guard");
        ++repetitions;
    }
    std::printf("PASS %s repetitions=%d launches=%llu upload_bytes=%llu download_bytes=%llu\n",
                label.c_str(),repetitions,launches,uploads,downloads);
}
'''


def _memory_guard():
    minimum = int(os.environ.get("FORT_TEST_MIN_AVAILABLE_BYTES", "0"))
    if minimum:
        available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        assert available >= minimum, "memory guard stopped numerical-team acceptance before a new process"


def _run(command, directory, *, env=None, timeout=180):
    _memory_guard()
    result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def compiled_team(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA/OpenMP/Fortran toolchain unavailable")
    directory = tmp_path_factory.mktemp("numerical_team")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    original = directory / "original.f90"
    original.write_text(SOURCE)
    env = {**os.environ, "PYTHONPATH": str(checkout)}
    reports, outputs = {}, {}
    for entry in ("advance", "guarded"):
        output = directory / entry
        result = _run([sys.executable, "-m", "compiler", "--input", str(original), "--kernel", entry,
                       "--fallback", "host", "--gpu-policy", "sections", "--gpu-collective", "--host-threads", "4",
                       "--memory-model", "scoped", "--json", "--output-dir", str(output)], checkout, env=env)
        reports[entry] = json.loads(result.stdout)
        assert reports[entry]["supported"], reports[entry]
        outputs[entry] = output
    public = reports["advance"]["scoped"]
    guarded = reports["guarded"]["scoped"]
    assert public["team"]["participation"] == "qualified_full_team"
    assert not public["team"]["automatic_estimate_available"]
    assert public["planning"]["query_available"]
    assert not guarded["planning"]["query_available"]
    # The one shared runtime object is linked only after proving every public
    # runtime source/header/interface byte equal between independent clients.
    for name in read_scoped_runtime()[0]:
        assert (outputs["advance"] / name).read_bytes() == (outputs["guarded"] / name).read_bytes()
    replacements = {"@TEAM@": public["team"]["entry"], "@PLAN@": public["planning"]["entry"],
                    "@CHOOSE@": public["planning"]["selector"],
                    "@GUARDED_TEAM@": guarded["team"]["entry"],
                    "@FIRST_UNIT@": str(public["planning"]["units"][0]["id"]),
                    "@ELSE_UNIT@": str(public["planning"]["units"][1]["id"]),
                    "@MIDDLE_UNIT@": str(public["planning"]["units"][2]["id"]),
                    "@LAST_UNIT@": str(public["planning"]["units"][3]["id"]),
                    "@GUARDED_FIRST_UNIT@": str(guarded["planning"]["units"][0]["id"]),
                    "@GUARDED_LAST_UNIT@": str(guarded["planning"]["units"][1]["id"])}
    driver = DRIVER
    for placeholder, value in replacements.items():
        driver = driver.replace(placeholder, value)
    (directory / "driver.cu").write_text(driver)
    (directory / "reference.f90").write_text(REFERENCE)
    _run([fortran, "-O0", "-fopenmp", "-fcheck=all,array-temps", "-c", str(original),
          str(directory / "reference.f90")], directory)
    flags = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp"]
    objects = []
    for label, source in (("advance", outputs["advance"] / "shared_entry.cu"),
                          ("guarded", outputs["guarded"] / "shared_entry.cu"),
                          ("runtime", outputs["advance"] / "scoped_runtime.cu"),
                          ("driver", directory / "driver.cu")):
        target = directory / (label + ".o")
        _run([*flags, "-I", str(outputs["advance"]), "-c", str(source), "-o", str(target)], directory)
        objects.append(str(target))
    executable = directory / "team"
    _run([*flags, *objects, str(directory / "original.o"), str(directory / "reference.o"),
          "-lgfortran", "-o", str(executable)], directory)
    (directory / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")
    return directory, executable, public


def execute(compiled, mode, *, trace=True):
    _memory_guard()
    directory, executable, public = compiled
    env = {**os.environ, "OMP_DYNAMIC": "FALSE"}
    env.pop("FORT_RUNTIME_TRACE", None)
    if trace:
        env["FORT_RUNTIME_TRACE"] = "1"
    result = subprocess.run([str(executable), mode], cwd=directory, env=env,
                            capture_output=True, text=True, timeout=30)
    (directory / (mode + ("-traced" if trace else "-untraced") + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Fortran runtime warning" not in result.stderr
    repetitions = 4 if mode in {"undefined", "badplan"} else 6
    assert "repetitions=" + str(repetitions) in result.stdout
    if not trace:
        assert "FORT_SCOPED" not in result.stderr
    return result.stdout, result.stderr, public


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["cpu", "auto", "gpu", "mixed", "protected", "resource"])
def test_existing_team_matches_native_fields_without_duplicate_work(compiled_team, mode):
    stdout, stderr, _ = execute(compiled_team, mode)
    assert "PASS " + mode in stdout
    expected = 12 if mode == "gpu" else 8 if mode == "mixed" else 4 if mode == "protected" else 0
    assert sum(line.startswith("FORT_SCOPED launch ") for line in stderr.splitlines()) == expected
    execute(compiled_team, mode, trace=False)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["serial", "budget", "nested", "argument", "undefined"])
def test_invalid_context_or_preparation_returns_uniformly_before_cuda(compiled_team, mode):
    stdout, stderr, _ = execute(compiled_team, mode)
    assert "PASS " + mode in stdout
    assert "FORT_SCOPED launch " not in stderr
    assert "FORT_SCOPED upload " not in stderr
    assert "FORT_SCOPED download " not in stderr


@pytest.mark.cuda
def test_failure_after_first_gpu_worker_poisoned_uniformly_without_replay(compiled_team):
    stdout, stderr, _ = execute(compiled_team, "badplan")
    assert "PASS badplan" in stdout
    assert sum(line.startswith("FORT_SCOPED launch ") for line in stderr.splitlines()) == 4
