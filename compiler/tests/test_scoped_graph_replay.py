"""Ordered graph-cache acceptance with an explicitly simulated CUDA backend."""
# ruff: noqa: F811 - imported pytest fixture.
from __future__ import annotations

import ctypes as c
import shutil
import subprocess

import pytest

from compiler.tests.test_scoped_access_batch import Binding, pack
from compiler.tests.test_scoped_runtime import RUNTIME, SIZE, TOKEN, Scope, access, runtime  # noqa: F401


class Kernel(c.Structure):
    _fields_ = [("version", c.c_uint32), ("reserved", c.c_uint32), ("function", c.c_void_p),
               ("grid", c.c_uint32 * 3), ("block", c.c_uint32 * 3), ("shared_bytes", SIZE),
               ("arguments", c.POINTER(c.c_void_p)), ("argument_count", SIZE)]


class GraphStats(c.Structure):
    _fields_ = [("version", c.c_uint32), ("reserved", c.c_uint32)] + [(name, TOKEN) for name in (
        "creations", "instantiations", "rebuilds", "submissions", "replays", "parameter_updates",
        "logical_kernels", "live_graphs", "destructions")]


class Task(c.Structure):
    _fields_ = [("kernel", Kernel), ("bindings", c.POINTER(Binding)), ("binding_count", SIZE)]


@pytest.fixture
def graphs(runtime):
    runtime.fort_scope_graph_submit_v1.argtypes = [TOKEN, TOKEN, c.POINTER(Kernel), SIZE]
    runtime.fort_scope_graph_stats_get_v1.argtypes = [TOKEN, c.POINTER(GraphStats)]
    runtime.fort_scope_graph_execute_v1.argtypes = [TOKEN, TOKEN, c.POINTER(Task), SIZE]
    runtime.fort_scope_device_reserve_v1.argtypes = [TOKEN, TOKEN, c.POINTER(c.c_void_p)]
    runtime.fort_scope_observe_v1.argtypes = [TOKEN, c.POINTER(TOKEN), SIZE]
    runtime.fort_scope_discard_v1.argtypes = runtime.fort_scope_retire_v1.argtypes = [TOKEN, TOKEN]
    return runtime


def graph_stats(scope):
    result = GraphStats()
    scope.check(scope.lib.fort_scope_graph_stats_get_v1(scope.handle, c.byref(result)))
    assert result.version == 1
    return result


def kernel():
    return Kernel(1, 0, 1, (c.c_uint32 * 3)(1, 1, 1), (c.c_uint32 * 3)(32, 1, 1), 0, None, 0)


@pytest.mark.native
def test_cpu_context_and_noop_group_create_no_graphs(graphs):
    scope = Scope(graphs)
    scope.check(graphs.fort_scope_graph_submit_v1(scope.handle, 0, None, 0))
    descriptor = kernel()
    assert graphs.fort_scope_graph_submit_v1(scope.handle, 1, c.byref(descriptor), 1) == 4
    assert scope.stats().launches == scope.stats().allocations == scope.stats().waits == 0
    result = graph_stats(scope)
    assert all(getattr(result, name) == 0 for name, _ in result._fields_ if name != "version")
    scope.close()


@pytest.mark.native
@pytest.mark.parametrize("invalid", ["key", "version", "reserved", "function", "grid", "block", "arguments"])
def test_invalid_groups_fail_before_cuda(graphs, invalid):
    scope = Scope(graphs)
    descriptor = kernel()
    key = 0 if invalid == "key" else 1
    if invalid in {"version", "reserved", "function"}:
        setattr(descriptor, invalid, 0 if invalid != "reserved" else 1)
    elif invalid in {"grid", "block"}:
        getattr(descriptor, invalid)[0] = 0
    elif invalid == "arguments":
        descriptor.argument_count = 1
    assert graphs.fort_scope_graph_submit_v1(scope.handle, key, c.byref(descriptor), 1) == 1
    assert graph_stats(scope).creations == scope.stats().launches == 0
    scope.close()


@pytest.mark.native
def test_observation_batches_partial_defined_resources(graphs):
    scope = Scope(graphs)
    arrays = []
    middle = [([2], [6])]
    for identity in range(1, 4):
        host = (c.c_double * 8)(*range(8))
        status, handle = scope.register(host, [8], identity=identity, initialized=False)
        scope.check(status)
        device = scope.gpu(handle, access(write=middle, overwrite=middle))
        for i in range(2, 6):
            device[i] = identity * 100 + i
        scope.gpu_end(handle)
        arrays.append((host, handle))
    handles = (TOKEN * 3)(*(handle.value for _, handle in arrays))
    before = scope.stats()
    scope.check(graphs.fort_scope_observe_v1(scope.handle, handles, 3))
    assert scope.stats().waits == before.waits + 1
    assert scope.stats().downloads == before.downloads + 3
    for identity, (host, handle) in enumerate(arrays, 1):
        assert list(host) == [0, 1, *[identity * 100 + i for i in range(2, 6)], 6, 7]
        assert graphs.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 7
    scope.close()


@pytest.mark.native
def test_invalid_observation_does_not_publish_an_earlier_resource(graphs):
    scope = Scope(graphs)
    host = (c.c_double * 8)(*range(8))
    status, handle = scope.register(host, [8])
    scope.check(status)
    device = scope.gpu(handle, access(flags=3))
    device[0] = 1000
    scope.gpu_end(handle)
    before = bytes(scope.stats())
    for other, expected in [(2**63, 2), (handle.value, 3)]:
        assert graphs.fort_scope_observe_v1(scope.handle, (TOKEN * 2)(handle.value, other), 2) == expected
        assert bytes(scope.stats()) == before
        assert host[0] == 0
    scope.check(graphs.fort_scope_observe_v1(scope.handle, (TOKEN * 1)(handle.value), 1))
    assert host[0] == 1000
    scope.close()


@pytest.mark.native
def test_discard_and_retire_avoid_dead_value_publication(graphs):
    scope = Scope(graphs)
    host = (c.c_double * 8)(*range(8))
    status, handle = scope.register(host, [8])
    scope.check(status)
    device = scope.gpu(handle, access(flags=3))
    device[0] = 1000
    scope.gpu_end(handle)
    before = scope.stats()
    scope.check(graphs.fort_scope_discard_v1(scope.handle, handle))
    assert scope.stats().downloads == before.downloads
    assert scope.stats().waits == before.waits
    assert graphs.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 7
    device = scope.gpu(handle, access(flags=6))
    for i in range(8):
        device[i] = 2000 + i
    scope.gpu_end(handle)
    assert scope.stats().allocations == 1
    scope.check(graphs.fort_scope_retire_v1(scope.handle, handle))
    assert scope.stats().downloads == 0
    assert scope.stats().allocated_bytes == 0
    assert list(host) == list(range(8))
    assert graphs.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 2
    scope.close()


@pytest.mark.native
def test_retirement_and_discard_reject_active_access_and_empty_observation_is_independent(graphs):
    scope = Scope(graphs)
    host = (c.c_double * 8)(*range(8))
    status, handle = scope.register(host, [8])
    scope.check(status)
    scope.gpu(handle, access(flags=1))
    assert graphs.fort_scope_discard_v1(scope.handle, handle) == 8
    assert graphs.fort_scope_retire_v1(scope.handle, handle) == 8
    scope.gpu_end(handle)
    before = scope.stats().waits
    scope.check(graphs.fort_scope_observe_v1(scope.handle, None, 0))
    assert scope.stats().waits == before
    scope.check(graphs.fort_scope_retire_v1(scope.handle, handle))
    assert scope.stats().waits == before + 1
    scope.close()


@pytest.mark.native
def test_ordered_task_simulation_defines_intermediate_privately(graphs):
    scope = Scope(graphs)
    handles = []
    for identity in range(1, 4):
        values = (c.c_double * 8)(*range(8))
        status, handle = scope.register(values, [8], identity=identity, initialized=identity == 1)
        scope.check(status)
        handles.append(handle)
    a, b, out = handles
    produce = pack((a, access(flags=1)), (b, access(flags=6)))
    consume = pack((b, access(flags=1)), (out, access(flags=6)))
    ordered = (Task * 2)(Task(kernel(), produce, 2), Task(kernel(), consume, 2))
    before = bytes(scope.stats())
    # RESOURCE is the CPU backend's post-validation outcome: consumer's
    # intermediate read is legal only when producer effects were simulated.
    assert graphs.fort_scope_graph_execute_v1(scope.handle, 8, ordered, 2) == 4
    assert bytes(scope.stats()) == before
    assert graphs.fort_scope_host_begin(scope.handle, b, c.byref(access(flags=1))) == 7
    assert graphs.fort_scope_host_begin(scope.handle, out, c.byref(access(flags=1))) == 7
    reverse = (Task * 2)(ordered[1], ordered[0])
    assert graphs.fort_scope_graph_execute_v1(scope.handle, 8, reverse, 2) == 7
    assert bytes(scope.stats()) == before
    scope.close()


@pytest.mark.native
def test_reservation_changes_only_storage_and_failed_late_task_commits_nothing(graphs):
    scope = Scope(graphs)
    host = (c.c_double * 8)()
    status, handle = scope.register(host, [8], initialized=False)
    scope.check(status)
    pointer, again = c.c_void_p(), c.c_void_p()
    scope.check(graphs.fort_scope_device_reserve_v1(scope.handle, handle, c.byref(pointer)))
    scope.check(graphs.fort_scope_device_reserve_v1(scope.handle, handle, c.byref(again)))
    assert pointer.value == again.value
    assert scope.stats().allocations == 1
    assert scope.stats().uploads == scope.stats().downloads == 0
    assert graphs.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 7
    produce = pack((handle, access(flags=6)))
    bad = pack((TOKEN(2**63), access(flags=1)))
    tasks = (Task * 2)(Task(kernel(), produce, 1), Task(kernel(), bad, 1))
    before = bytes(scope.stats())
    assert graphs.fort_scope_graph_execute_v1(scope.handle, 9, tasks, 2) == 2
    assert bytes(scope.stats()) == before
    assert graphs.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 7
    scope.close()


FAKE_CUDA = r'''
#pragma once
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
using cudaError_t=int;
constexpr int cudaSuccess=0, cudaErrorMemoryAllocation=2, cudaErrorInvalidValue=1;
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
struct cudaKernelNodeParams {
    void *func=nullptr; dim3 gridDim,blockDim; unsigned sharedMemBytes=0;
    void **kernelParams=nullptr, **extra=nullptr;
};
struct Node { void *function=nullptr; int *output=nullptr; int value=0; dim3 grid,block; unsigned shared=0; };
struct Graph { std::vector<Node *> nodes; };
struct Execution { std::vector<Node> nodes; std::vector<Node *> originals; };
using cudaGraph_t=Graph *; using cudaGraphExec_t=Execution *;
using cudaGraphNode_t=Node *; using cudaStream_t=void *;
inline int add_identity=0,multiply_identity=0;
inline std::string failure;
inline int updates=0, live_graphs=0, live_execs=0;
inline std::vector<std::vector<Node>> pending;
inline std::vector<std::string> events;
inline Node value(const cudaKernelNodeParams *p) {
    return {p->func,*static_cast<int **>(p->kernelParams[0]),*static_cast<int *>(p->kernelParams[1]),
            p->gridDim,p->blockDim,p->sharedMemBytes};
}
inline void drain() {
    for(const auto &group:pending) for(const auto &node:group)
        if(node.function==&add_identity) *node.output+=node.value;
        else { assert(node.function==&multiply_identity); *node.output*=node.value; }
    pending.clear();
}
inline cudaError_t cudaGraphCreate(cudaGraph_t *out,unsigned) {
    events.push_back("create"); if(failure=="create") return 2;
    *out=new Graph; ++live_graphs; return 0;
}
inline cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *out,cudaGraph_t graph,const cudaGraphNode_t *deps,
                                         size_t count,const cudaKernelNodeParams *params) {
    events.push_back("add");
    if(failure=="node" && !graph->nodes.empty()) return 2;
    assert(count==(graph->nodes.empty()?0u:1u));
    if(count) assert(deps && *deps==graph->nodes.back());
    *out=new Node(value(params)); graph->nodes.push_back(*out); return 0;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t *out,cudaGraph_t graph,void *,void *,size_t) {
    events.push_back("instantiate"); if(failure=="instantiate") return 2;
    *out=new Execution; ++live_execs;
    for(auto node:graph->nodes) { (*out)->nodes.push_back(*node); (*out)->originals.push_back(node); }
    return 0;
}
inline cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t exec,cudaGraphNode_t node,
                                                   const cudaKernelNodeParams *params) {
    events.push_back("update"); ++updates;
    if(failure=="update" && updates%2==0) return 2;
    auto found=std::find(exec->originals.begin(),exec->originals.end(),node);
    assert(found!=exec->originals.end()); exec->nodes[found-exec->originals.begin()]=value(params); return 0;
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t exec,cudaStream_t) {
    events.push_back("launch"); if(failure=="launch") return 1;
    pending.push_back(exec->nodes); return 0;
}
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
    events.push_back("destroy_exec"); delete exec; --live_execs; return 0;
}
inline cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    events.push_back("destroy_graph"); for(auto node:graph->nodes) delete node; delete graph; --live_graphs; return 0;
}
'''

CLIENT = r'''
#include "scoped_graphs.hpp"
#include <cstring>
#include <iostream>
using fort_scoped::graphs::Cache;
struct Arguments {
    int *pointer; int scalar;
    void *values[2]; fort_scope_kernel_v1 descriptor;
    Arguments(void *function,int *out,int v):pointer(out),scalar(v),values{&pointer,&scalar},
        descriptor{1,0,function,{1,1,1},{32,1,1},0,values,2} {}
};
void check(cudaError_t status) { assert(status==cudaSuccess); }
int main(int argc,char **argv) {
    assert(argc==2); const std::string mode=argv[1];
    Cache cache; fort_scope_graph_stats_v1 stats{}; stats.version=1;
    int first=0,second=1; Arguments a(&add_identity,&first,2), b(&multiply_identity,&first,3);
    fort_scope_kernel_v1 group[2]={a.descriptor,b.descriptor}; bool attempted=false;
    if(mode=="replay") {
        check(cache.submit(7,group,2,nullptr,stats,attempted)); assert(attempted);
        a.pointer=b.pointer=&second; a.scalar=4; b.scalar=5;
        group[0].grid[0]=8; group[1].block[0]=64; group[1].shared_bytes=128;
        check(cache.submit(7,group,2,nullptr,stats,attempted));
        a.scalar=b.scalar=999; // Values were borrowed only during submission.
        drain(); assert(first==6 && second==25);
        assert(stats.creations==1 && stats.instantiations==1 && stats.replays==1);
        assert(stats.parameter_updates==2 && stats.submissions==2 && stats.logical_kernels==4);
    } else if(mode=="topology") {
        check(cache.submit(7,group,1,nullptr,stats,attempted));
        check(cache.submit(7,group,2,nullptr,stats,attempted));
        group[0]=b.descriptor;
        check(cache.submit(7,group,1,nullptr,stats,attempted));
        drain(); assert(first==36);
        assert(stats.creations==3 && stats.rebuilds==2 && stats.destructions==2);
    } else if(mode=="observe") {
        bool warmup=false;
        check(cache.observe(7,group,2,stats,warmup)); assert(warmup);
        check(cache.observe(8,group,2,stats,warmup)); assert(warmup);
        assert(events.empty() && !stats.creations && !stats.live_graphs);
        a.pointer=b.pointer=&second; a.scalar=4; b.scalar=5;
        check(cache.observe(7,group,2,stats,warmup)); assert(!warmup);
        check(cache.submit(7,group,2,nullptr,stats,attempted)); drain();
        assert(first==0 && second==25 && stats.live_graphs==1);
        group[0]=b.descriptor;
        check(cache.observe(7,group,1,stats,warmup)); assert(warmup);
        assert(!stats.live_graphs && stats.destructions==1);
        b.scalar=2;
        check(cache.observe(7,group,1,stats,warmup)); assert(!warmup);
        check(cache.submit(7,group,1,nullptr,stats,attempted)); drain();
        assert(second==50 && stats.live_graphs==1 && stats.creations==2 && stats.rebuilds==1);
    } else if(mode=="keys") {
        check(cache.submit(7,group,2,nullptr,stats,attempted));
        check(cache.submit(8,group,2,nullptr,stats,attempted));
        assert(stats.creations==2 && stats.live_graphs==2);
        drain(); assert(first==24);
    } else {
        if(mode=="update" || mode=="launch") {
            check(cache.submit(7,group,2,nullptr,stats,attempted)); drain(); assert(first==6);
        }
        failure=mode;
        assert(cache.submit(7,group,2,nullptr,stats,attempted)!=cudaSuccess);
        assert(attempted==(mode=="launch"));
        assert(pending.empty());
        failure.clear();
        check(cache.submit(7,group,2,nullptr,stats,attempted)); drain();
        assert(first==((mode=="update" || mode=="launch")?24:6));
    }
    check(cache.clear(stats));
    assert(!stats.live_graphs && !live_graphs && !live_execs);
    assert(stats.creations==stats.destructions);
    assert(events.size()>=2 && events[events.size()-2]=="destroy_exec" && events.back()=="destroy_graph");
    std::cout << "simulated CUDA graph cache accepted: " << mode << '\n';
}
'''


@pytest.fixture(scope="module")
def simulated_cuda(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    directory = tmp_path_factory.mktemp("simulated_graph_cuda")
    (directory / "cuda_runtime.h").write_text(FAKE_CUDA)
    source = directory / "client.cpp"
    source.write_text(CLIENT)
    target = directory / "run"
    result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(directory),
                             "-I", str(RUNTIME), str(source), "-o", str(target)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return target


@pytest.mark.native
@pytest.mark.parametrize("mode", ["replay", "topology", "observe", "keys", "create", "node", "instantiate", "update", "launch"])
def test_simulated_cuda_graph_cache(simulated_cuda, mode):
    result = subprocess.run([str(simulated_cuda), mode], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
