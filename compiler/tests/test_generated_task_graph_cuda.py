"""Execute compiler-generated graph groups with exact shared-buffer effects."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import read_common_header
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig

SOURCE = """module graph_generated
contains
subroutine entry(a,b,out,n,shift)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),out(:)
integer,intent(in)::n
real(8),intent(in)::shift
integer::i,j
do i=2,n-1
b(i)=a(i)+1.d0
enddo
do j=2,n-1
out(j)=b(j-1)+b(j)+shift
enddo
end subroutine
end module
"""

DRIVER = r"""
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" int ENTRY(fort_scope_t,int,ARRAY_PARAMETERS,const int*,const double*);
void check(int status) {
    if(status) { std::fprintf(stderr,"status=%d: %s\n",status,fort_scope_error()); std::abort(); }
}
int main(int argc,char **argv) {
    assert(argc==2);
    const bool empty=!std::strcmp(argv[1],"empty");
    const bool budget=!std::strcmp(argv[1],"budget");
    if(!empty) {
        int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess || !count) return 77;
    }
    double a[12],b[12],out[12];
    for(int i=0;i<12;++i) { a[i]=3*i+2; b[i]=-99; out[i]=-7; }
    b[0]=100;
    size_t extent[1]={12}; int64_t lower[1]={1};
    fort_scope_t context=0; fort_buffer_t handles[3]{};
    check(fort_scope_create(0,&context));
    if(budget) check(fort_scope_set_device_budget(context,1));
    double *values[3]={a,b,out};
    for(int i=0;i<3;++i) {
        fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,values[i],extent,lower,1};
        if(i==1) {
            size_t lo[1]={0},hi[1]={1}; fort_scope_section halo{lo,hi};
            check(fort_scope_register_sections(context,i+1,1,&layout,&halo,1,&handles[i]));
        } else check(fort_scope_register(context,i+1,1,&layout,1,&handles[i]));
    }
    VIEW_SETUP
    int n=empty?0:12; double shift=0.25;
    check(ENTRY(context,1,ARRAY_ARGUMENTS,&n,&shift));
    fort_scope_stats stats{}; check(fort_scope_stats_get(context,&stats));
    fort_scope_graph_stats_v1 graph{}; check(fort_scope_graph_stats_get_v1(context,&graph));
    fort_scope_graph_queue_stats_v1 queued{};
    assert(!stats.downloads);
    if(empty) {
        assert(!stats.allocations && !stats.launches && !graph.creations && !graph.submissions);
        for(int i=0;i<12;++i) assert(out[i]==-7);
        check(fort_scope_close(context)); return 0;
    }
    if(budget) {
        assert(!stats.allocations && !stats.launches && !graph.submissions);
    } else {
        assert(stats.allocations==3 && !stats.launches && !stats.uploads);
        assert(!graph.creations && !graph.submissions);
        assert(b[1]==-99); // Intermediate has not been published.
    }
    // These entry-local values no longer exist when the queue is submitted.
    // The accepted task must own the original scalar values and domain.
    n=0; shift=999;
    check(fort_scope_observe_v1(context,&handles[2],1));
    check(fort_scope_stats_get(context,&stats));
    check(fort_scope_graph_stats_get_v1(context,&graph));
    check(fort_scope_graph_queue_stats_get_v1(context,&queued));
    if(!budget) {
        assert(stats.launches==2 && stats.uploads==2 && stats.upload_bytes==88);
        assert(!graph.creations && !graph.submissions && !graph.logical_kernels);
        assert(queued.direct_submissions==1 && queued.direct_kernels==2);
    }
    assert(out[0]==-7 && out[11]==-7);
    for(int i=1;i<11;++i) {
        double previous=i==1?100:a[i-1]+1;
        assert(out[i]==previous+a[i]+1+0.25);
    }
    n=9; shift=4.5;
    check(ENTRY(context,1,ARRAY_ARGUMENTS,&n,&shift));
    n=0; shift=999;
    check(fort_scope_observe_v1(context,&handles[2],1));
    for(int i=1;i<8;++i) {
        double previous=i==1?100:a[i-1]+1;
        assert(out[i]==previous+a[i]+1+4.5);
    }
    check(fort_scope_stats_get(context,&stats));
    check(fort_scope_graph_stats_get_v1(context,&graph));
    check(fort_scope_graph_queue_stats_get_v1(context,&queued));
    if(!budget) {
        assert(stats.allocations==3 && stats.uploads==2 && stats.launches==4);
        assert(graph.creations==1 && graph.submissions==1 && !graph.replays);
        assert(!graph.parameter_updates && graph.logical_kernels==2);
        assert(queued.direct_submissions==1 && queued.direct_kernels==2);
    }
    n=10; shift=-1.75;
    check(ENTRY(context,1,ARRAY_ARGUMENTS,&n,&shift));
    n=0; shift=999;
    check(fort_scope_observe_v1(context,&handles[2],1));
    for(int i=1;i<9;++i) {
        double previous=i==1?100:a[i-1]+1;
        assert(out[i]==previous+a[i]+1-1.75);
    }
    check(fort_scope_stats_get(context,&stats));
    check(fort_scope_graph_stats_get_v1(context,&graph));
    if(!budget) {
        assert(stats.allocations==3 && stats.uploads==2 && stats.launches==6);
        assert(graph.creations==1 && graph.submissions==2 && graph.replays==1);
        assert(graph.parameter_updates==2 && graph.logical_kernels==4);
    }
    check(fort_scope_close(context));
    return 0;
}
"""


def _run(command, directory, *, timeout=120, env=None):
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=timeout, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module", params=[False, True], ids=["roots", "views"])
def executable(request, tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit and host C++ compiler required")
    directory = tmp_path_factory.mktemp("generated_graph_views" if request.param else "generated_graph")
    source = directory / "original.f90"
    source.write_text(SOURCE)
    options = CompilerOptions(gpu_policy="sections", memory_model="scoped", scope_execution="graph")
    function, plan = prepare_function(lower_file(source, "entry"), options=options)
    assert len(plan.regions) == 2
    generated = generate_scoped(
        function,
        plan,
        OffloadConfig(policy="sections", scope_execution="graph"),
        "common_functions.cuh",
        root_views=request.param,
    )
    artifacts, _manifest = read_scoped_runtime()
    for name, content in artifacts.items():
        (directory / name).write_text(content)
    (directory / "common_functions.cuh").write_text(read_common_header())
    (directory / "entry.cu").write_text(generated.cuda)
    (directory / "entry-report.json").write_text(json.dumps(generated.report, indent=2))
    assert [region for group in generated.report["graph_execution"]["groups"] for region in group["regions"]] == [0, 1]
    driver = DRIVER.replace("ENTRY", generated.report["entry"])
    if request.param:
        driver = driver.replace("ARRAY_PARAMETERS", ",".join(["const fort_scope_view_v1*"] * 3))
        driver = driver.replace("ARRAY_ARGUMENTS", "&views[0],&views[1],&views[2]")
        driver = driver.replace(
            "VIEW_SETUP",
            """size_t origins[1]={0};
        fort_scope_view_v1 views[3]{};
        for(int i=0;i<3;++i) views[i]={1,1,handles[i],1,origins,extent,lower};""",
        )
    else:
        driver = driver.replace("ARRAY_PARAMETERS", "fort_buffer_t,fort_buffer_t,fort_buffer_t")
        driver = driver.replace("ARRAY_ARGUMENTS", "handles[0],handles[1],handles[2]")
        driver = driver.replace("VIEW_SETUP", "")
    (directory / "driver.cu").write_text(driver)
    target = directory / "test"
    command = [
        nvcc,
        "-std=c++17",
        "-ccbin",
        host,
        "-Xcompiler=-fopenmp",
        "-Xcompiler=-ffp-contract=off",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
        "entry.cu",
        "scoped_runtime.cu",
        "driver.cu",
        "-o",
        str(target),
    ]
    _run(command, directory, timeout=180)
    return target


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["gpu", "budget", "empty"])
def test_generated_graph_reuses_parameters_and_defines_intermediates(executable, mode):
    env = dict(os.environ)
    if mode == "empty":
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([str(executable), mode], capture_output=True, text=True, env=env, timeout=30)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
