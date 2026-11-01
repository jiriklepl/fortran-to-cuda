"""Three original units and eighteen queued kernels; host transport controls only."""
import ctypes as c
import shutil

import pytest

from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.ir import CompilationError
from compiler.tests.test_queued_tile_task_provider import STUB, QueueReport, checked
from compiler.tests.test_tile_private import SOURCE
from compiler.tests.test_tile_private_team import team_fixture


def three_units():
    first = SOURCE.index("!$omp do ")
    last = SOURCE.index("!$omp end do", first) + len("!$omp end do\n")
    return SOURCE[:last] + SOURCE[first:last] * 2 + SOURCE[last:]


def generated(path, *, mapping="cooperative", sealed=False):
    _, analysis, routine, nodes = team_fixture(path, three_units())
    if sealed:
        analysis.seal_session()
    proof = source_device_task(analysis, routine, nodes)
    provider = emit_tile_task_provider(proof.team, readable_scalars=proof,
                                      access_envelopes=True, queued_team=True, mapping=mapping)
    return proof, provider


@pytest.mark.parametrize("sealed", [False, True])
def test_complete_original_three_unit_plan_has_no_definition_or_old_backend_credit(tmp_path, sealed):
    proof, provider = generated(tmp_path, sealed=sealed)
    record = provider.public()
    assert len(proof.team.units) == len(record["unit_proof_identities"]) == 3
    assert record["kind"] == "queued_original_tile_private_team"
    assert record["unit_proof_identities"] == [unit.identity for unit in proof.team.units]
    assert len(record["launch_abi"]["units"]) == len(set(record["code_keys"])) == 3
    assert record["effects"]["overwrites_authorized"] is False
    assert record["effects"]["initialization_authorized"] is False
    assert record["definitions_authorized"] is record["placement_authorized"] is False
    assert record["consumer_wiring"] is False
    assert record["restriction"]["metadata_budget_bytes"] == 16 * 1024 * 1024
    assert "expected_launches" not in record["restriction"]
    assert provider.entry.endswith("_boxes_qt")
    assert "fort_scope_source_execute" not in provider.cuda_source
    assert "<<<" not in provider.cuda_source
    text = provider.cuda_source
    order = ["status=prepare_tile_access", "if (metadata_bytes>", "tasks.reserve(expected)",
             "status=fort_scope_scratch_acquire_v1", "status=fort_scope_device_reserve_many_v1",
             "tasks.push_back", "status=fort_scope_graph_enqueue_v1", "report->accepted=1",
             "status=fort_scope_graph_flush_v1", "report->committed=1"]
    assert [text.index(value) for value in order] == sorted(text.index(value) for value in order)
    assert text.count("status=fort_scope_graph_flush_v1") == 1
    assert "report->accepted ? fatal() : decline(status)" in text
    assert emit_tile_task_provider(proof.team, readable_scalars=proof, access_envelopes=True,
                                   queued_team=True) == provider


@pytest.mark.parametrize("options", [{"queued_team": 1}, {"queued_team": True},
                                    {"queued_team": True, "access_envelopes": True, "queued_single": True},
                                    {"queued_team": True, "access_envelopes": True, "pricing": object()}])
def test_no_unchecked_or_priced_whole_team_transport(tmp_path, options):
    _, analysis, routine, nodes = team_fixture(tmp_path, three_units())
    proof = source_device_task(analysis, routine, nodes)
    with pytest.raises(CompilationError):
        emit_tile_task_provider(proof.team, readable_scalars=proof, **options)


@pytest.fixture(scope="module", params=["thread", "cooperative"])
def compiled(request, tmp_path_factory):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    assert nvcc, "Required controlled CUDA/GNU host tools must not skip"
    assert host, "Required controlled CUDA/GNU host tools must not skip"
    assert fc, "Required controlled CUDA/GNU host tools must not skip"
    directory = tmp_path_factory.mktemp("queued-team-" + request.param)
    _, provider = generated(directory, mapping=request.param)
    (directory / "provider.cu").write_text(provider.cuda_source)
    (directory / "provider.f90").write_text(provider.fortran_source)
    # Same established double, with larger descriptor capacity and an Nth
    # acceptance failure. It never launches a CUDA kernel or acquires a device.
    stub = STUB.replace("static std::size_t extent=64;extent=mode==1?2:64;",
                        "static std::size_t extent=2000000000;extent=mode==1?2:2000000000;")
    stub = stub.replace("++calls[1];static", "assert(!calls[4]);++calls[1];static")
    stub = stub.replace("++calls[2];if", "assert(!calls[4]);++calls[2];if")
    stub = stub.replace("++calls[3];assert", "assert(!calls[4]);++calls[3];assert")
    stub = stub.replace("if(mode==6)return FORT_SCOPE_UNINITIALIZED;",
                        "if(mode==6)return FORT_SCOPE_UNINITIALIZED;\n if(mode==12&&calls[4]==2)return FORT_SCOPE_RESOURCE;\n if(mode==13&&calls[4]==18)return FORT_SCOPE_RESOURCE;")
    (directory / "stub.cpp").write_text(stub)
    checked([nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "--fmad=false", "--ftz=false",
             "--prec-div=true", "--prec-sqrt=true", "-shared", "-Xcompiler=-fPIC", "-I", str(root / "runtime"),
             "-I", str(root / "emission/common/templates"), str(directory / "provider.cu"),
             str(directory / "stub.cpp"), "-o", str(directory / "provider.so")], directory, "cuda-host-link")
    checked([fc, "-c", "-J", str(directory), str(directory / "provider.f90"), "-o", str(directory / "provider.o")],
            directory, "fortran-interface")
    library = c.CDLL(str(directory / "provider.so"))
    function = getattr(library, provider.entry)
    function.argtypes = [c.c_int64, c.c_int64, c.c_int64,
                         *[c.POINTER(c.c_int) for _ in provider.parameters], c.POINTER(QueueReport)]
    function.restype = c.c_int
    library.arg_bytes.restype = c.c_size_t
    return provider, library, function


def invoke(compiled, *, mode=0, n=13):
    provider, library, function = compiled
    values = {("scalar", "n", None): n, ("scalar", "t", None): 3,
              ("external_lower", "a", 1): -3, ("external_lower", "b", 1): -3,
              ("scratch_lower", "tmp", 1): -1, ("scratch_upper", "tmp", 1): 5}
    scalars = [c.c_int(values[p.role, p.binding.name, p.axis]) for p in provider.parameters]
    library.reset(mode)
    report = QueueReport()
    status = function(1, 11, 12, *[c.byref(value) for value in scalars], c.byref(report))
    assert report.version == 1
    assert report.status == status
    return status, report, [library.counter(i) for i in range(8)]


@pytest.mark.parametrize(("n", "kernels"), [(13, 3), (1348164, 18)])
def test_all_units_and_batches_accept_before_one_complete_flush(compiled, n, kernels):
    _, library, _ = compiled
    status, report, calls = invoke(compiled, n=n)
    assert status == 0
    assert report.accepted == report.committed == 1
    assert calls == [1, 2, 1, 1, kernels, 1, 1, 0]
    assert library.arg_bytes() == kernels * 116


def test_zero_common_domain_reads_no_descriptor_or_runtime(compiled):
    status, report, calls = invoke(compiled, n=0)
    assert status == 0
    assert report.accepted == report.committed == 0
    assert calls == [0] * 8


@pytest.mark.parametrize(("mode", "status", "enqueues", "release"),
                         [(2, 5, 0, 0), (3, 5, 0, 0), (4, 4, 0, 0), (5, 4, 0, 1), (6, 7, 1, 1)])
def test_first_acceptance_prework_only_can_decline(compiled, mode, status, enqueues, release):
    result, report, calls = invoke(compiled, mode=mode, n=1348164)
    assert result == status
    assert report.accepted == report.committed == 0
    assert calls[4] == enqueues
    assert calls[5] == calls[7] == 0
    assert calls[6] == release


@pytest.mark.parametrize(("mode", "enqueues", "flush", "committed", "release"),
                         [(12, 2, 0, 0, 0), (13, 18, 0, 0, 0), (7, 18, 1, 0, 0), (8, 18, 1, 1, 1)])
def test_any_later_enqueue_flush_or_cleanup_error_is_irrevocable(compiled, mode, enqueues, flush, committed, release):
    status, report, calls = invoke(compiled, mode=mode, n=1348164)
    assert status == 6
    assert report.accepted == 1
    assert report.committed == committed
    assert calls[4:] == [enqueues, flush, release, 1]


def test_metadata_byte_resource_budget_declines_without_truncating_or_acquiring(compiled):
    status, report, calls = invoke(compiled, n=1500000000)
    assert status == 4
    assert report.accepted == report.committed == 0
    assert calls[2:] == [0] * 6
