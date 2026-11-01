"""Generated Fortran owner on the host recording runtime, without CUDA."""
import json
import os
import shutil
import subprocess

import pytest

from compiler.tests.test_provider_only_lifetime import compile_parent

PROVIDER = r'''
#include "scoped_runtime.h"
#include <cstdint>
#include <cstdlib>
#include <new>
struct Config { int64_t extent; double coefficient; };
struct Provider { Config config; fort_scope_domain_t domain; uint64_t lease; };
static int creates=0,executes=0,destroys=0;
static uint64_t generations[8]{};
static size_t uploaded=0,downloaded=0;
extern "C" int provider_create(fort_scope_domain_t domain,const Config *config,void **handle) {
 ++creates; *handle=nullptr;
 if(std::getenv("FORT_PROVIDER_REJECT"))return FORT_SCOPE_RESOURCE;
 if(!config||config->extent!=8)return FORT_SCOPE_BOUNDARY;
 auto *p=new(std::nothrow)Provider{*config,domain,0};if(!p)return FORT_SCOPE_RESOURCE;
 int status=fort_scope_domain_workspace_reserve_v1(domain,32,&p->lease);
 if(status){delete p;return status;} *handle=p;return FORT_SCOPE_OK;
}
static int work(void *opaque,const fort_scope_external_view_v1 *views,size_t count,void *,int *started) {
 auto &p=*static_cast<Provider *>(opaque);*started=0;
 if(count!=2||views[0].generation!=views[1].generation||views[0].rank!=1||views[1].rank!=1||
    views[0].extents[0]!=8||views[1].extents[0]!=8)return FORT_SCOPE_ARGUMENT;
 if(executes>=8)return FORT_SCOPE_ARGUMENT;
 generations[executes++]=views[0].generation; *started=1;
 auto *a=static_cast<const double *>(views[0].device);auto *b=static_cast<double *>(views[1].device);
 for(int i=0;i<8;++i)b[i]=p.config.coefficient*a[i];
 if(std::getenv("FORT_PROVIDER_FATAL"))return FORT_SCOPE_EXECUTION;
 return FORT_SCOPE_OK;
}
extern "C" int provider_execute(fort_scope_t context,void *handle,fort_buffer_t input,fort_buffer_t output,
                                  fort_scope_external_report_v1 *report) {
 if(!handle)return FORT_SCOPE_ARGUMENT;auto &p=*static_cast<Provider *>(handle);
 size_t lower[]={0},upper[]={8};fort_scope_section box{lower,upper};
 fort_scope_access read{0,1,&box,0,nullptr,0,nullptr},overwrite{0,0,nullptr,1,&box,1,&box};
 fort_scope_external_binding_v1 bindings[2]={{input,FORT_SCOPE_REAL64,1,8,read},{output,FORT_SCOPE_REAL64,1,8,overwrite}};
 fort_scope_external_call_v1 call{1,0,bindings,2,p.lease,32};
 return fort_scope_external_execute_v1(context,&call,work,&p,report);
}
extern "C" int provider_destroy(void **handle) {
 if(!handle||!*handle)return FORT_SCOPE_ARGUMENT;auto *p=static_cast<Provider *>(*handle);
 fort_scope_stats stats{};int status=fort_scope_domain_stats_get_v1(p->domain,&stats);if(status)return status;
 if(stats.allocated_bytes!=0)return FORT_SCOPE_ARGUMENT;
 uploaded+=stats.upload_bytes;downloaded+=stats.download_bytes;
 status=fort_scope_domain_workspace_release_v1(p->domain,p->lease);if(status)return status;
 ++destroys;delete p;*handle=nullptr;return FORT_SCOPE_OK;
}
extern "C" int provider_check(int phase) {
 if(std::getenv("FORT_PROVIDER_REJECT"))return creates==phase&&executes==0&&destroys==0?0:101;
 if(phase==1)return creates==1&&executes==2&&destroys==1&&generations[0]==1&&generations[1]==2&&
                   uploaded==128&&downloaded==128?0:102;
 return creates==2&&executes==3&&destroys==2&&generations[2]==1&&uploaded==192&&downloaded==192?0:103;
}
'''
DRIVER = '''program exercise
use iso_c_binding
use native_parent
implicit none
interface
 integer(c_int) function provider_check(phase) bind(C)
 import c_int
 integer(c_int),value::phase
 end function
end interface
real(8),allocatable::a(:),b(:)
real(8)::original(8)
integer::i,status
allocate(a(8),b(8))
original=[(real(i,8),i=1,8)]
a=-100.0_8
b=original
call owner(a,b,8)
if(any(a/=3.0_8*(3.0_8*original+10.0_8)))error stop 'second full output or first publication'
if(any(b/=3.0_8*original+10.0_8))error stop 'unmanaged host work not preserved'
status=provider_check(1)
if(status/=0)then
print *,status
error stop 'owner must retain one plan with fresh registrations'
endif
! Unknown/public dispatch receives no parent slot and publishes standalone.
call middle(a,b,8)
if(any(a/=3.0_8*b))error stop 'standalone full output'
status=provider_check(2)
if(status/=0)then
print *,status
error stop 'standalone lifetime must not inherit retired owner'
endif
print *, 'PROVIDER_ONLY_FIELDS_OK'
end program
'''


@pytest.fixture(scope="module", params=[False, True], ids=["same_storage", "changed_storage"])
def binaries(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp("provider_only_" + str(request.param))
    build, objects, report = compile_parent(directory, replace_storage=request.param)
    cpp = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    assert cpp
    assert fortran
    provider, driver = build / "recording_provider.cpp", build / "driver.f90"
    provider.write_text(PROVIDER)
    driver.write_text(DRIVER)
    commands = [[cpp, "-std=c++17", "-fopenmp", "-pthread", "-DFORT_SCOPE_CPU_TEST", "-x", "c++", "-I", str(build),
                 "-c", str(build / "scoped_runtime.cu"), "-o", str(build / "runtime.o")],
                [cpp, "-std=c++17", "-fopenmp", "-pthread", "-I", str(build), "-c", str(provider), "-o", str(build / "provider.o")],
                [fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-I", str(build), str(driver),
                 *map(str, objects), str(build / "runtime.o"), str(build / "provider.o"), "-lstdc++", "-pthread", "-o", str(build / "exercise")]]
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=build, text=True, capture_output=True, timeout=90)
        (build / f"runtime-{index}.command.json").write_text(json.dumps(command))
        (build / f"runtime-{index}.stdout").write_text(result.stdout)
        (build / f"runtime-{index}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stderr
    return build, report


@pytest.mark.native
@pytest.mark.parametrize("mode", ["enabled", "preaccept_reject", "accepted_fatal"])
def test_original_parent_executes_with_private_plan_reuse_and_complete_field_publication(binaries, mode):
    build, _ = binaries
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("FORT_", "ELMM_", "CUDA_")):
            env.pop(key)
    if mode == "preaccept_reject":
        env["FORT_PROVIDER_REJECT"] = "1"
    elif mode == "accepted_fatal":
        env["FORT_PROVIDER_FATAL"] = "1"
    env["FORT_RUNTIME_TRACE"] = "1"
    result = subprocess.run([str(build / "exercise")], cwd=build, env=env, capture_output=True, text=True, timeout=15)
    (build / f"{mode}.stdout").write_text(result.stdout)
    (build / f"{mode}.stderr").write_text(result.stderr)
    (build / f"{mode}.completion.json").write_text(json.dumps({"returncode": result.returncode}))
    if mode == "accepted_fatal":
        assert result.returncode != 0
        assert "external device work failed; replay prohibited" in result.stderr
        assert "PROVIDER_ONLY_FIELDS_OK" not in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PROVIDER_ONLY_FIELDS_OK" in result.stdout


MAIN_CONTROLS = r'''
static void verify_main_retirement() {
 if(std::getenv("FORT_PROVIDER_FATAL"))return;
 bool rejected=std::getenv("FORT_PROVIDER_REJECT");
 if(creates!=1||executes!=(rejected?0:4)||destroys!=(rejected?0:1))std::_Exit(119);
 std::puts("MAIN_PLAN_RETIRED");
}
extern "C" int main_provider_check(int stage) {
 if(std::getenv("FORT_PROVIDER_REJECT"))return creates==1&&executes==0&&destroys==0?0:111;
 if(creates!=1||destroys!=0||executes!=2*stage||!main_active)return 112;
 for(int i=0;i<executes;++i)if(generations[i]!=static_cast<uint64_t>(i+1))return 113;
 fort_scope_stats stats{};
 int status=fort_scope_domain_stats_get_v1(main_active->domain,&stats);
 if(status||stats.allocated_bytes!=0||stats.upload_bytes!=128*stage||stats.download_bytes!=128*stage)return 114;
 return 0;
}
'''


@pytest.fixture(scope="module")
def main_binary(tmp_path_factory):
    directory = tmp_path_factory.mktemp("provider_main_owner")
    build, objects, report = compile_parent(directory, program=True)
    cpp = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    assert cpp
    assert fortran
    code = PROVIDER.replace('#include <new>', '#include <new>\n#include <cstdio>')
    code = code.replace('static int creates=0,executes=0,destroys=0;',
        'static int creates=0,executes=0,destroys=0;\nstatic Provider *main_active=nullptr;\nstatic void verify_main_retirement();')
    code = code.replace(' ++creates; *handle=nullptr;', ' if(creates==0)std::atexit(verify_main_retirement);\n ++creates; *handle=nullptr;')
    code = code.replace('*handle=p;return FORT_SCOPE_OK;', '*handle=p;main_active=p;return FORT_SCOPE_OK;')
    code = code.replace('++destroys;delete p;', '++destroys;main_active=nullptr;delete p;')
    provider = build / "main_recording_provider.cpp"
    provider.write_text(code + MAIN_CONTROLS)
    commands = [[cpp, "-std=c++17", "-fopenmp", "-pthread", "-DFORT_SCOPE_CPU_TEST", "-x", "c++", "-I", str(build),
                 "-c", str(build / "scoped_runtime.cu"), "-o", str(build / "runtime.o")],
                [cpp, "-std=c++17", "-fopenmp", "-pthread", "-I", str(build), "-c", str(provider), "-o", str(build / "provider.o")],
                [fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-I", str(build),
                 *map(str, objects), str(build / "runtime.o"), str(build / "provider.o"), "-lstdc++", "-pthread", "-o", str(build / "exercise")]]
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=build, text=True, capture_output=True, timeout=90)
        (build / f"main-{index}.command.json").write_text(json.dumps(command))
        (build / f"main-{index}.stdout").write_text(result.stdout)
        (build / f"main-{index}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stderr
    return build, report


@pytest.mark.native
@pytest.mark.parametrize("mode", ["enabled", "preaccept_reject", "accepted_fatal"])
def test_main_program_retains_plan_across_timesteps_with_zero_fields_between(main_binary, mode):
    build, _ = main_binary
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("FORT_", "ELMM_", "CUDA_")):
            env.pop(key)
    if mode == "preaccept_reject":
        env["FORT_PROVIDER_REJECT"] = "1"
    elif mode == "accepted_fatal":
        env["FORT_PROVIDER_FATAL"] = "1"
    env["FORT_RUNTIME_TRACE"] = "1"
    result = subprocess.run([str(build / "exercise")], cwd=build, env=env, capture_output=True, text=True, timeout=15)
    (build / f"main-{mode}.stdout").write_text(result.stdout)
    (build / f"main-{mode}.stderr").write_text(result.stderr)
    (build / f"main-{mode}.completion.json").write_text(json.dumps({"returncode": result.returncode}))
    if mode == "accepted_fatal":
        assert result.returncode != 0
        assert "external device work failed; replay prohibited" in result.stderr
        assert "MAIN_PROVIDER_FIELDS_OK" not in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "MAIN_PROVIDER_FIELDS_OK" in result.stdout
        assert "MAIN_PLAN_RETIRED" in result.stdout
