"""Source pricing is independently bound and cannot authorize numerical work."""

from __future__ import annotations

import json
import shutil
import subprocess
from copy import copy, deepcopy
from dataclasses import replace

import pytest

from compiler.driver.options import CompilerOptions
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.offload.source_protocol import profile_with_source_measurements
from compiler.offload.source_task_pricing import source_task_pricing
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_offload_profile import scoped_profile
from compiler.tests.test_scoped_runtime import RUNTIME
from compiler.tests.test_source_protocol_calibration import MAX_BYTES, observations, teams
from compiler.tests.test_tile_private import SOURCE
from compiler.tests.test_tile_private_team import team_fixture


def measured_fixture():
    # Synthetic input observations exercise model/schema code; no performance
    # evidence or actual hardware identity is claimed by these unit tests.
    runtime_id = read_scoped_runtime()[1]["runtime_id"]
    profile = scoped_profile()
    profile["scoped"]["runtime_id"] = runtime_id
    profile["toolchain"].update(nvcc_version="Cuda compilation tools, release 13.4, V13.4.92",
                                host_cxx_version="g++ (Debian) 14.4.0")
    return profile_with_source_measurements(profile, observations(), runtime_id=runtime_id,
        max_bytes=MAX_BYTES, team_records=teams())


def issued(tmp_path, *, profile=None, source=SOURCE):
    _, analysis, routine, nodes = team_fixture(tmp_path, source)
    proof = source_device_task(analysis, routine, nodes)
    pricing = source_task_pricing(proof.team, measured_fixture() if profile is None else profile,
                                  read_scoped_runtime()[1]["runtime_id"])
    return proof, pricing


def test_pricing_binds_exact_source_work_and_distinct_measured_protocol(tmp_path):
    proof, pricing = issued(tmp_path)
    record = pricing.public()
    assert pricing.validate() is pricing
    assert record["team_identity"] == proof.team.identity
    assert record["work"]["team_identity"] == proof.team.identity
    assert record["native_protocol"]["rank"] == record["native_protocol"]["collapse"] == 1
    assert record["native_protocol"]["units"] == 1
    assert record["native_protocol"]["seconds"] > 0
    assert record["whole_owner_estimate_available"] is False
    assert record["full_wall_prediction"] is False
    assert all(record[key] is False for key in ("definitions_authorized", "execution_authorized", "motion_authorized"))
    assert "domain remains live" in record["endpoints"]["gpu"]
    assert "PRIVATE allocation" in record["exclusions"]
    assert proof.public()["pricing"] == "unpriced; explicit sections only"


@pytest.mark.parametrize("case", ["profile_missing", "runtime", "precision", "threads", "holdouts", "native",
                                 "unsupported_schedule"])
def test_missing_or_incompatible_prices_never_become_zero_costs(tmp_path, case):
    profile, source = measured_fixture(), SOURCE
    if case == "profile_missing":
        profile = None
    elif case == "runtime":
        profile["scoped"]["runtime_id"] = "a" * 64
    elif case == "precision":
        profile["precision_bits"] = 32
    elif case == "threads":
        profile["cpu_threads"] = 3
    elif case == "holdouts":
        profile["source_tasks"]["available"] = False
    elif case == "native":
        profile["source_tasks"]["original_teams"]["costs"] = []
    elif case == "unsupported_schedule":
        source = source.replace("schedule(runtime)", "schedule(dynamic)")
    _, analysis, routine, nodes = team_fixture(tmp_path, source)
    proof = source_device_task(analysis, routine, nodes)
    with pytest.raises(ValueError, match="profile|source|precision|thread"):
        source_task_pricing(proof.team, profile, read_scoped_runtime()[1]["runtime_id"])
    assert not getattr(analysis, "_source_task_pricings", {})


def test_forged_or_changed_pricing_and_source_identity_are_rejected(tmp_path):
    proof, pricing = issued(tmp_path)
    for forged in (copy(pricing), replace(pricing), replace(pricing, identity="0" * 64)):
        with pytest.raises(CompilationError, match="registered cost attachment"):
            forged.validate()
    other = tmp_path / "other"
    other.mkdir()
    foreign, _ = issued(other)
    with pytest.raises(CompilationError, match="exact team's"):
        emit_tile_task_provider(foreign.team, readable_scalars=foreign, pricing=pricing)
    # Caller edits cannot mutate the immutable profile captured by the quote.
    original = pricing.profile()
    original["source_tasks"]["costs"]["source_submit_seconds"] = 0
    assert pricing.profile()["source_tasks"]["costs"]["source_submit_seconds"] > 0
    assert pricing.team is proof.team


def test_default_sections_provider_remains_unchanged_and_auto_ordering_is_explicit(tmp_path):
    proof, pricing = issued(tmp_path)
    before = emit_tile_task_provider(proof.team, readable_scalars=proof)
    provider = emit_tile_task_provider(proof.team, readable_scalars=proof, pricing=pricing)
    after = emit_tile_task_provider(proof.team, readable_scalars=proof)
    assert before == after
    assert before.public()["version"] == 2
    assert provider.public()["version"] == 3
    assert provider.public()["pricing"]["identity"] == pricing.identity
    assert len(provider.module) <= 63
    code = provider.cuda_source
    wait = code.index("int status=fort_scope_wait(context)")
    quote = code.index("status=fort_scope_source_quote_v1(", wait)
    reject = code.index("if(!quote.available||!quote.gpu_preferred)", quote)
    assert wait < quote < reject < code.index("fort_scope_scratch_acquire_v1(", quote)
    assert code.index("fort_scope_external_binding_v1 bindings[]") < quote
    assert "request.compute.known_backends=FORT_SCOPE_COMPUTE_NATIVE_FORTRAN|FORT_SCOPE_COMPUTE_GPU;" in code
    assert "request.compute.known_backends=FORT_SCOPE_COMPUTE_ALL" not in code
    assert "FORT_SCOPE_SOURCE_QUOTE_WAIT_BEFORE_LEASE" in code
    assert "omp_get_level()==0&&omp_get_thread_limit()>=4&&omp_get_max_active_levels()>0" in code
    assert code.index("fort_runtime::HostFloatingEnvironment quote_environment;") < code.index("request.compute.native_fortran_seconds=")
    assert code.index("if(!quote.available||!quote.gpu_preferred)") < code.index("!offload::compatible(profile,")


@pytest.mark.native
def test_priced_provider_compiles_with_the_actual_cpp_cuda_and_fortran_interfaces(tmp_path):
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    if not nvcc or not host or not fc:
        pytest.skip("supported CUDA and GNU compilers unavailable")
    proof, pricing = issued(tmp_path)
    provider = emit_tile_task_provider(proof.team, readable_scalars=proof, pricing=pricing)
    (tmp_path / "provider.cu").write_text(provider.cuda_source)
    (tmp_path / "provider.f90").write_text(provider.fortran_source)
    commands = [[nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp", "-I", str(RUNTIME),
                 "-I", str(RUNTIME.parent / "emission/common/templates"), "-c", str(tmp_path / "provider.cu"),
                 "-o", str(tmp_path / "provider.o")],
                [fc, "-fopenmp", "-J", str(tmp_path), "-c", str(RUNTIME / "scoped_memory.f90"),
                 "-o", str(tmp_path / "memory.o")],
                [fc, "-fopenmp", "-J", str(tmp_path), "-I", str(tmp_path), "-c", str(tmp_path / "provider.f90"),
                 "-o", str(tmp_path / "fortran.o")]]
    for index, command in enumerate(commands):
        (tmp_path / f"command-{index}.json").write_text(json.dumps(command))
        result = subprocess.run(command, text=True, capture_output=True, timeout=120)
        (tmp_path / f"command-{index}.stdout").write_text(result.stdout)
        (tmp_path / f"command-{index}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("child", [False, True])
def test_auto_graph_retains_the_unmodified_source_proof_and_independent_cost_attachment(tmp_path, child):
    from hashlib import sha256

    source = SOURCE
    if child:
        source = source.rsplit("end module", 1)[0] + """subroutine outer(a,b,n,t)
integer,intent(in)::n,t
real(8),intent(in)::a(-3:)
real(8),intent(inout)::b(-3:)
call advance(a,b,n,t)
end subroutine
end module
"""
    path = tmp_path / "source.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: deepcopy(FACT) for name in ("a", "b")}}
    builder = ScopeBuilder([path], "stencil::" + ("outer" if child else "advance"), facts=facts,
        options=CompilerOptions(), config=OffloadConfig(policy="auto", scope_execution="graph", profile=measured_fixture()))
    outputs, report = builder.run()
    scope, = report["scopes"]
    task, = scope["source_device_tasks"]
    assert task["version"] == 2
    assert task["source_proof"]["version"] == 1
    assert task["pricing"]["team_identity"] == task["source_proof"]["team"]["identity"]
    assert task["execution_binding"]["provider"]["metadata"]["version"] == 3
    assert scope["estimate_available"] is False
    assert scope["native_owner_hint"]["eligible"] is False
    assert "source task" in scope["native_owner_hint"]["reason"]
    assert sha256(outputs[task["pricing"]["profile_artifact"]].encode()).hexdigest() == task["pricing"]["profile_identity"]
    replacement = outputs[report["sources"][str(path)]["replacement"]]
    decline = replacement.index("if (.not. fort_source_done) then")
    close = replacement.index("fort_close_status = fort_scope_resident_close_v1(", decline)
    native = replacement.index("!$omp parallel private(bi,i,li,tmp)", close)
    assert decline < close < native
    assert "!$omp parallel private(bi,i,li,tmp)" in replacement
