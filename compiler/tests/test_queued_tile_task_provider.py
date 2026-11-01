"""Real generated queue transport, compiled against a host-only runtime double."""
import ctypes as c
import json
import shutil
import subprocess
from copy import copy
from pathlib import Path

import pytest

from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.ir import CompilationError
from compiler.tests.test_tile_private import SOURCE
from compiler.tests.test_tile_private_team import team_fixture

ROOT = Path(__file__).resolve().parents[1]


class QueueReport(c.Structure):
    _fields_ = [("version", c.c_uint32), ("status", c.c_int32),
                ("accepted", c.c_uint32), ("committed", c.c_uint32)]


def provider_at(path, *, mapping="cooperative", sealed=False, text=SOURCE, **options):
    _, analysis, routine, nodes = team_fixture(path, text)
    if sealed:
        analysis.seal_session()
    proof = source_device_task(analysis, routine, nodes)
    provider = emit_tile_task_provider(proof.team, mapping=mapping, readable_scalars=proof,
                                      access_envelopes=True, queued_single=True, **options)
    return proof, provider


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("mapping", ["thread", "cooperative"])
def test_actual_source_proof_has_separate_backend_identity_and_no_definition_credit(tmp_path, sealed, mapping):
    proof, provider = provider_at(tmp_path, mapping=mapping, sealed=sealed)
    record = provider.public()
    assert record["kind"] == "queued_single_original_tile_private_team"
    assert record["backend"] == "fort_scope_graph_enqueue_v1+fort_scope_graph_flush_v1"
    assert provider.entry.endswith("_boxes_q1")
    assert record["team_identity"] == proof.team.identity
    assert record["scalar_readability_identity"] == proof.identity
    assert record["effects"]["overwrites_authorized"] is False
    assert record["effects"]["initialization_authorized"] is False
    assert record["definitions_authorized"] is False
    assert record["consumer_wiring"] is False
    assert "source_execute" not in provider.cuda_source
    assert "<<<" not in provider.cuda_source
    assert "fort_tile_queue_report_v1" in provider.fortran_source
    assert "fort_scope_external_report_v1" not in provider.fortran_source
    regenerated = emit_tile_task_provider(proof.team, mapping=mapping, readable_scalars=proof,
        access_envelopes=True, queued_single=True)
    assert regenerated == provider
    assert regenerated.public()["code_key"] == record["code_key"]
    text = provider.cuda_source
    order = ["if (expected!=1)", "fort_scope_graph_queue_stats_get_v1", "status=prepare_tile_access",
             "status=fort_scope_scratch_acquire_v1", "status=fort_scope_device_reserve_many_v1",
             "status=fort_scope_graph_enqueue_v1", "report->accepted=1", "status=fort_scope_graph_flush_v1",
             "report->committed=1", "status=fort_scope_scratch_release_v1"]
    assert [text.index(x) for x in order] == sorted(text.index(x) for x in order)


def test_queued_choice_rejects_copied_or_stale_authority_and_multiunit_team(tmp_path):
    proof, _ = provider_at(tmp_path)
    with pytest.raises(CompilationError):
        emit_tile_task_provider(copy(proof.team), access_envelopes=True, queued_single=True)
    path = tmp_path / "second"
    path.mkdir()
    first = SOURCE.index("!$omp do ")
    last = SOURCE.index("!$omp end do", first) + len("!$omp end do\n")
    text = SOURCE[:last] + SOURCE[first:last] + SOURCE[last:]
    _, analysis, routine, nodes = team_fixture(path, text)
    multi = source_device_task(analysis, routine, nodes)
    with pytest.raises(CompilationError, match="one unpriced"):
        emit_tile_task_provider(multi.team, readable_scalars=multi, access_envelopes=True, queued_single=True)
    original = tmp_path / "source.f90"
    original.write_text(original.read_text() + "! changed\n")
    with pytest.raises(CompilationError):
        emit_tile_task_provider(proof.team, readable_scalars=proof, access_envelopes=True, queued_single=True)


@pytest.mark.parametrize("options", [{"queued_single": 1}, {"queued_single": True},
                                    {"queued_single": True, "access_envelopes": True, "batch_target_bytes": 4194304},
                                    {"queued_single": True, "access_envelopes": True, "pricing": object()}])
def test_unqualified_or_priced_queue_choice_rejects_before_codegen(tmp_path, options):
    _, analysis, routine, nodes = team_fixture(tmp_path)
    proof = source_device_task(analysis, routine, nodes)
    with pytest.raises(CompilationError):
        emit_tile_task_provider(proof.team, readable_scalars=proof, **options)


STUB = r'''
#include "scoped_runtime.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
static int mode=0,calls[8]={};static std::uint64_t code=0;static std::size_t argcopied=0,bytecopied=0,lastcount=0;
static std::vector<std::vector<unsigned char>> owned;
extern "C" void reset(int m){mode=m;for(auto& c:calls)c=0;code=0;argcopied=bytecopied=lastcount=0;owned.clear();}
extern "C" int counter(int n){return calls[n];}
extern "C" std::uint64_t key(){return code;}
extern "C" std::size_t arg_count(){return argcopied;}
extern "C" std::size_t arg_bytes(){return bytecopied;}
extern "C" std::size_t tile_count(){return lastcount;}
extern "C" int fort_scope_serial_caller(){return mode!=9;}
extern "C" int fort_scope_numerical_environment_supported(){return 1;}
extern "C" int fort_scope_graph_queue_stats_get_v1(fort_scope_t,fort_scope_graph_queue_stats_v1* out){
 ++calls[0];*out={};out->version=1;out->queue_enabled=mode!=2;out->pending_tasks=mode==3;return 0;}
extern "C" int fort_scope_layout_get(fort_scope_t,fort_buffer_t,fort_scope_layout* out){
 ++calls[1];static std::size_t extent=64;extent=mode==1?2:64;static std::int64_t lower=-3;
 *out={1,mode==10?FORT_SCOPE_REAL32:FORT_SCOPE_REAL64,8,nullptr,&extent,&lower,7};return 0;}
extern "C" int fort_scope_scratch_acquire_v1(fort_scope_t,std::size_t n,fort_scope_scratch_lease_v1* out){
 ++calls[2];if(mode==4)return FORT_SCOPE_RESOURCE;*out={1,0,123,reinterpret_cast<void*>(0x30000),n,n};return 0;}
extern "C" int fort_scope_scratch_release_v1(fort_scope_t,std::uint64_t token){
 ++calls[6];assert(token==123);return mode==8||mode==11?FORT_SCOPE_RESOURCE:0;}
extern "C" int fort_scope_device_reserve_many_v1(fort_scope_t,const fort_buffer_t* h,std::size_t n,void** p){
 ++calls[3];assert(n==2&&h[0]==11&&h[1]==12);if(mode==5||mode==11)return FORT_SCOPE_RESOURCE;
 p[0]=reinterpret_cast<void*>(0x10000);p[1]=reinterpret_cast<void*>(0x20000);return 0;}
extern "C" int fort_scope_graph_enqueue_v1(fort_scope_t,std::uint64_t k,const fort_scope_graph_task_v1* task,const std::size_t* sizes){
 ++calls[4];assert(task&&task->binding_count==2&&k);code=k;
 assert(task->kernel.function&&task->kernel.grid[0]>0&&task->kernel.block[0]==256);
 for(std::size_t i=0;i<2;++i){const auto& access=task->bindings[i].access;
  assert(task->bindings[i].buffer==11+i&&access.flags==0&&access.overwrite_count==0);
  assert(access.read_count>0);assert(i?access.write_count>0:access.write_count==0);}
 if(mode==6)return FORT_SCOPE_UNINITIALIZED;
 argcopied=task->kernel.argument_count;
 for(std::size_t i=0;i<argcopied;++i){assert(sizes[i]>0&&sizes[i]<=8);bytecopied+=sizes[i];
  owned.emplace_back(sizes[i]);std::memcpy(owned.back().data(),task->kernel.arguments[i],sizes[i]);}
 assert(sizes[argcopied-1]==sizeof(std::size_t));std::memcpy(&lastcount,owned.back().data(),sizeof(lastcount));
 return 0;}
extern "C" int fort_scope_graph_flush_v1(fort_scope_t){++calls[5];assert(!owned.empty());return mode==7?FORT_SCOPE_RESOURCE:0;}
extern "C" int fort_scope_execution_error(fort_scope_t,const char*){++calls[7];return FORT_SCOPE_EXECUTION;}
'''


def checked(argv, directory, name):
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=120, cwd=directory)
    (directory / (name + ".command.json")).write_text(json.dumps(argv, indent=2))
    (directory / (name + ".stdout")).write_text(completed.stdout)
    (directory / (name + ".stderr")).write_text(completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture(scope="module", params=["thread", "cooperative"])
def compiled(request, tmp_path_factory):
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    assert nvcc, "Required host CUDA toolkit absent; this controlled gate may not skip"
    assert host, "Required host C++ toolkit absent; this controlled gate may not skip"
    assert fc, "Required Fortran toolkit absent; this controlled gate may not skip"
    d = tmp_path_factory.mktemp("queued-" + request.param)
    _, provider = provider_at(d, mapping=request.param)
    (d / "provider.cu").write_text(provider.cuda_source)
    (d / "provider.f90").write_text(provider.fortran_source)
    (d / "stub.cpp").write_text(STUB)
    checked([nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "--fmad=false", "--ftz=false",
             "--prec-div=true", "--prec-sqrt=true", "-shared", "-Xcompiler=-fPIC", "-I", str(ROOT / "runtime"),
             "-I", str(ROOT / "emission/common/templates"), str(d / "provider.cu"), str(d / "stub.cpp"),
             "-o", str(d / "provider.so")], d, "cuda-host-link")
    checked([fc, "-c", "-J", str(d), str(d / "provider.f90"), "-o", str(d / "provider.o")], d, "fortran-interface")
    lib = c.CDLL(str(d / "provider.so"))
    function = getattr(lib, provider.entry)
    function.argtypes = [c.c_int64, c.c_int64, c.c_int64, *[c.POINTER(c.c_int) for _ in provider.parameters], c.POINTER(QueueReport)]
    function.restype = c.c_int
    for name in ("arg_count", "arg_bytes", "tile_count"):
        getattr(lib, name).restype = c.c_size_t
    lib.key.restype = c.c_uint64
    return provider, lib, function


def invoke(compiled, *, mode=0, n=13, upper=5, missing=None):
    provider, lib, function = compiled
    values = {("scalar", "n", None): n, ("scalar", "t", None): 3,
              ("external_lower", "a", 1): -3, ("external_lower", "b", 1): -3,
              ("scratch_lower", "tmp", 1): -1, ("scratch_upper", "tmp", 1): upper}
    scalars = [None if (p.role, p.binding.name, p.axis) == missing else
               c.c_int(values[p.role, p.binding.name, p.axis]) for p in provider.parameters]
    lib.reset(mode)
    report = QueueReport()
    status = function(1, 11, 12, *[None if v is None else c.byref(v) for v in scalars], c.byref(report))
    assert report.version == 1
    assert report.status == status
    return status, report, [lib.counter(i) for i in range(8)]


def test_real_compiled_generated_task_has_one_complete_kernel_and_existing_atomic_commit(compiled):
    provider, lib, _ = compiled
    status, report, calls = invoke(compiled)
    assert (status, report.accepted, report.committed) == (0, 1, 1)
    assert calls == [1, 2, 1, 1, 1, 1, 1, 0]
    assert lib.key() == int(provider.public()["code_key"])
    assert lib.arg_count() == provider.public()["launch_abi"]["parameters"]
    assert 0 < lib.arg_bytes() <= provider.public()["launch_abi"]["bytes"]
    assert lib.tile_count() == 5  # All tiles, not just the first one, in this kernel.


def test_zero_domain_reads_no_descriptor_and_calls_no_runtime(compiled):
    status, report, calls = invoke(compiled, n=0)
    assert (status, report.accepted, report.committed) == (0, 0, 0)
    assert calls == [0] * 8


def test_multilaunch_rejects_before_any_runtime_or_scratch(compiled):
    status, report, calls = invoke(compiled, n=19, upper=196613)
    assert (status, report.accepted, report.committed) == (4, 0, 0)
    assert calls == [0] * 8


@pytest.mark.parametrize(("mode", "status", "calls"), [
    (1, 1, [1, 2, 0, 0, 0, 0, 0, 0]),
    (2, 5, [1, 0, 0, 0, 0, 0, 0, 0]),
    (3, 5, [1, 0, 0, 0, 0, 0, 0, 0]),
    (4, 4, [1, 2, 1, 0, 0, 0, 0, 0]),
    (5, 4, [1, 2, 1, 1, 0, 0, 1, 0]),
    (6, 7, [1, 2, 1, 1, 1, 0, 1, 0]),
    (9, 5, [0, 0, 0, 0, 0, 0, 0, 0]),
    (10, 1, [1, 1, 0, 0, 0, 0, 0, 0]),
])
def test_preaccept_declines_release_only_owned_scratch_and_do_not_publish_work(compiled, mode, status, calls):
    result, report, observed = invoke(compiled, mode=mode)
    assert (result, report.accepted, report.committed) == (status, 0, 0)
    assert observed == calls


@pytest.mark.parametrize(("mode", "committed", "released"), [(7, 0, 0), (8, 1, 1)])
def test_accepted_flush_or_cleanup_failure_poison_and_require_abandon_no_native_replay(compiled, mode, committed, released):
    status, report, calls = invoke(compiled, mode=mode)
    assert (status, report.accepted, report.committed) == (6, 1, committed)
    assert calls[5] == calls[7] == 1
    assert calls[6] == released


@pytest.mark.parametrize("missing", [("scalar", "t", None), ("external_lower", "a", 1)])
def test_missing_original_values_decline_before_runtime(compiled, missing):
    status, report, calls = invoke(compiled, missing=missing)
    assert (status, report.accepted, report.committed) == (1, 0, 0)
    assert calls == [0] * 8


def test_even_preaccept_cleanup_failure_is_fatal_not_native_fallback(compiled):
    status, report, calls = invoke(compiled, mode=11)
    assert (status, report.accepted, report.committed) == (6, 0, 0)
    assert calls == [1, 2, 1, 1, 0, 0, 1, 1]
    assert status not in compiled[0].public()["restriction"]["recoverable_statuses"]
