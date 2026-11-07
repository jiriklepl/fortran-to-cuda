"""Versioned live CPU alternatives preserve the original team and its replicas."""

import json
import shutil
import subprocess
from copy import deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.offload.config import OffloadConfig
from compiler.offload.source_protocol import profile_with_source_measurements
from compiler.offload.source_task_pricing import source_task_pricing
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_source_native_calibration import native_rows
from compiler.tests.test_source_protocol_calibration import MAX_BYTES, observations, teams
from compiler.tests.test_source_task_pricing import measured_fixture
from compiler.tests.test_tile_private import SOURCE
from compiler.tests.test_tile_private_team import team_fixture


def live_profile():
    return profile_with_source_measurements(measured_fixture(), observations() + native_rows(),
        runtime_id=read_scoped_runtime()[1]["runtime_id"], max_bytes=MAX_BYTES, team_records=teams())


def live_issued(directory):
    _, analysis, routine, nodes = team_fixture(directory)
    proof = source_device_task(analysis, routine, nodes)
    return proof, source_task_pricing(proof.team, live_profile(), read_scoped_runtime()[1]["runtime_id"],
                                     native_live=True)


def live_emitted(directory, child, *, transfers="direct", with_worker=False):
    source = SOURCE
    if with_worker:
        source = source.replace("!$omp parallel private", "call prepare(a,b,n)\n!$omp parallel private", 1)
        source = source.rsplit("end module", 1)[0] + """subroutine prepare(a,b,n)
integer,intent(in)::n
real(8),intent(in)::a(-3:)
real(8),intent(inout)::b(-3:)
integer::i
do i=1,n
 b(i)=b(i)+a(i)*0.25_8
enddo
end subroutine
end module
"""
    if child:
        source = source.rsplit("end module", 1)[0] + """subroutine outer(left,right,n,t)
integer,intent(in)::n,t
real(8),intent(in)::left(-3:)
real(8),intent(inout)::right(-3:)
call advance(left,right,n,t)
end subroutine
end module
"""
    path = directory / "source.f90"
    path.write_text(source)
    names = ("left", "right") if child else ("a", "b")
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: deepcopy(FACT) for name in names}}
    return path, ScopeBuilder([path], "stencil::" + ("outer" if child else "advance"), facts=facts,
        options=CompilerOptions(), config=OffloadConfig(policy="auto", scope_execution="graph",
                                                        profile=live_profile(), scope_transfers=transfers)).run()


def test_live_pricing_requires_separate_measured_management_and_retains_historical_endpoint(tmp_path):
    proof, pricing = live_issued(tmp_path)
    old = source_task_pricing(proof.team, live_profile(), read_scoped_runtime()[1]["runtime_id"])
    assert old.public()["version"] == 1
    assert pricing.public()["version"] == 2
    assert old.identity != pricing.identity
    assert "remain live" in pricing.public()["endpoints"]["native"]
    assert "no terminal retirement" in pricing.public()["endpoints"]["gpu"]
    assert pricing.public()["native_transaction_costs"]["prepare_start_complete_seconds"] > 0
    with pytest.raises(ValueError, match="native source protocol"):
        source_task_pricing(proof.team, measured_fixture(), read_scoped_runtime()[1]["runtime_id"], native_live=True)


def test_live_provider_quotes_before_gpu_fence_and_prepares_the_same_typed_bindings(tmp_path):
    proof, pricing = live_issued(tmp_path)
    default = emit_tile_task_provider(proof.team, readable_scalars=proof)
    provider = emit_tile_task_provider(proof.team, readable_scalars=proof, pricing=pricing)
    assert default == emit_tile_task_provider(proof.team, readable_scalars=proof)
    record, code = provider.public(), provider.cuda_source
    assert record["version"] == 4
    assert record["native_transaction_api"] == {name: "fort_scope_source_host_" + name + "_v1"
                                               for name in ("prepare", "start", "complete", "cancel")}
    assert code.index("fort_scope_source_quote_v2(") < code.index("status=fort_scope_wait(context)")
    assert "request.flags=FORT_SCOPE_SOURCE_QUOTE_NATIVE_LIVE|" in code
    assert "NATIVE_CLOSE_DOMAIN" not in code
    assert code.count("fort_scope_external_binding_v1 bindings[]=") == 2
    native = code[code.index('extern "C" int ' + record["native_prepare_entry"]):]
    assert "fort_scope_source_host_prepare_v1(context,&call,token)" in native
    assert "cuda" not in native
    assert "FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL" in native


def test_unqualified_gpu_protocol_is_explicit_native_only_and_never_a_zero_cost(tmp_path):
    proof, _ = live_issued(tmp_path)
    profile = live_profile()
    profile["source_tasks"]["available"] = False
    pricing = source_task_pricing(proof.team, profile, read_scoped_runtime()[1]["runtime_id"], native_live=True)
    record = pricing.public()
    assert record["version"] == 3
    assert record["gpu_protocol_available"] is False
    assert record["source_costs"]["costs"]["source_submit_seconds"] > 0
    assert record["native_transaction_costs"]["prepare_start_complete_seconds"] > 0
    provider = emit_tile_task_provider(proof.team, readable_scalars=proof, pricing=pricing)
    assert provider.public()["version"] == 4
    assert "source.valid=0; // GPU source protocol failed independent coverage" in provider.cuda_source
    assert "fort_scope_source_quote_v2(" in provider.cuda_source
    assert "fort_scope_source_host_prepare_v1(" in provider.cuda_source


@pytest.mark.parametrize("child", [False, True])
def test_emitted_live_alternative_brackets_exact_original_team_and_keeps_canonical_registration(tmp_path, child):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    path, (outputs, report) = live_emitted(tmp_path, child)
    validate_checked_execution_plans(report)
    task, = report["scopes"][0]["source_device_tasks"]
    assert task["version"] == 4
    assert task["kind"] == "scheduled_original_tile_private_team_live"
    assert task["source_proof"]["version"] == 1
    assert task["execution_binding"]["provider"]["metadata"]["version"] == 6
    assert set(task["execution_binding"]["registration_slots"].values()) == {1, 2}
    code = outputs[report["sources"][str(path)]["replacement"]]
    start = code.index("fort_scope_source_host_start_v1(")
    original = code.index("!$omp parallel private(bi,i,li,tmp)", start)
    complete = code.index("fort_scope_source_host_complete_v1(", original)
    fallback_close = code.index("fort_scope_resident_close_v1(", complete)
    assert start < original < complete < fallback_close
    assert "replay prohibited" in code[start:original]
    assert "replay prohibited" in code[complete:fallback_close]
    assert "fort_source_registered = .true." in code
    assert report["scopes"][0]["estimate_available"] is False
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        artifact = build / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("transfers", ["direct", "auto"])
def test_mixed_graph_keeps_priced_source_teams_with_automatic_transfers(tmp_path, child, transfers):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    path, (outputs, report) = live_emitted(tmp_path, child, transfers=transfers, with_worker=True)
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    validate_checked_execution_plans(report)
    scope, = report["scopes"]
    task, = scope["source_device_tasks"]
    assert task["kind"] == "scheduled_original_tile_private_team_live"
    assert task["pricing"]["endpoints"]["native"].startswith("typed host transaction")
    assert task["execution_binding"]["provider"]["metadata"]["version"] == 6
    # The ordinary worker configures the owning context once. The source task
    # consumes that same context and never installs a separate copy policy.
    code = outputs[report["sources"][str(path)]["replacement"]]
    assert "fort_configure(" in code
    assert "fort_scope_source_host_start_v1(" in code
    assert scope["estimate_available"] is False


@pytest.mark.native
def test_live_provider_and_generated_root_child_compile_against_actual_interfaces(tmp_path):
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    if not all((nvcc, host, fc)):
        pytest.skip("CUDA and GNU compilers unavailable")
    for child in (False, True):
        directory = tmp_path / ("nested" if child else "root")
        directory.mkdir()
        path, (outputs, report) = live_emitted(directory, child)
        for name, text in outputs.items():
            artifact = directory / name
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(text)
        task, = report["scopes"][0]["source_device_tasks"]
        provider = task["execution_binding"]["provider"]
        commands = [[nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp",
                     "-I", str(directory), "-c", str(directory / (provider["entry"] + ".cu")),
                     "-o", str(directory / "provider.o")]]
        for index, name in enumerate(("fort_scoped_memory.f90", provider["module"] + ".f90",
                                       report["sources"][str(path)]["replacement"])):
            commands.append([fc, "-fopenmp", "-fcheck=all", "-J", str(directory), "-I", str(directory),
                             "-c", str(directory / name), "-o", str(directory / f"source-{index}.o")])
        for index, command in enumerate(commands):
            (directory / f"command-{index}.json").write_text(json.dumps(command))
            result = subprocess.run(command, text=True, capture_output=True, timeout=120)
            (directory / f"command-{index}.stdout").write_text(result.stdout)
            (directory / f"command-{index}.stderr").write_text(result.stderr)
            assert result.returncode == 0, result.stdout + result.stderr
