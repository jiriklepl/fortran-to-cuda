"""Transfer configuration precedes placement and keeps missing costs explicit."""

from __future__ import annotations

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import form_source_scopes
from compiler.tests.test_collective_source_scopes import SOURCE as TEAM_SOURCE
from compiler.tests.test_scoped_planning_entries import calibration
from compiler.tests.test_source_continuation import TREE
from compiler.tests.test_source_scopes import FACT, PROGRAM
from compiler.tests.test_structured_offload import SOURCE


def emit(tmp_path, transfers, *, profile=None, collective=False):
    source = tmp_path / "source.f90"
    source.write_text(SOURCE)
    options = CompilerOptions(gpu_policy="auto", memory_model="scoped", scope_transfers=transfers)
    function, plan = prepare_function(lower_file(source, "advance"), options=options)
    return generate_scoped(function, plan, OffloadConfig("auto", profile, 4, collective, scope_transfers=transfers),
                           "common_functions.cuh", runtime_id=read_scoped_runtime()[1]["runtime_id"])


@pytest.mark.parametrize(("requested", "selected", "reason", "constant"), [
    ("direct", "direct", None, "DIRECT"), ("pinned", "pinned", None, "PINNED"),
    ("auto", "direct", "transfer_estimates_unavailable", "AUTO"),
    ("pipelined", "direct", "pipelined_not_available", "PIPELINED"),
])
def test_public_configuration_retains_requested_mode_without_initializing_cuda(tmp_path, requested, selected, reason, constant):
    emitted = emit(tmp_path, requested)
    configuration = emitted.report["transfer_configuration"]
    assert configuration["requested"] == requested
    assert configuration["selected"] == selected
    assert configuration["reason"] == reason
    assert configuration["execution"] == "synchronous"
    assert configuration["pinned_budget_bytes"] == 64 * 1024 * 1024
    assert configuration["argument_order"] == ["context"]
    text = emitted.cuda.split('extern "C" int ' + configuration["entry"], 1)[1].split("}", 1)[0]
    assert "return fort_scope_set_transfers(fort_context, FORT_SCOPE_TRANSFERS_" + constant + ");" in text
    assert "cuda" not in text.lower()
    assert emitted.cuda.count("fort_scope_set_transfers(") == 1
    assert "function configure(fort_context)" in emitted.fortran
    assert "bind(C, name='" + configuration["entry"] + "')" in emitted.fortran
    assert emitted.report["entry_abi_version"] == 2


def test_pinned_bandwidth_in_v1_profile_cannot_estimate_staging_or_select_gpu(tmp_path):
    value = calibration()
    assert value["rates"]["h2d_pinned"]["bandwidth_bytes_per_second"] > 0
    emitted = emit(tmp_path, "pinned", profile=value)
    assert not emitted.report["automatic_estimate_available"]
    assert emitted.report["automatic_reason"] == "transfer_estimates_unavailable"
    assert not emitted.report["planning"]["profile_available"]
    assert emitted.report["planning"]["query_available"]
    assert not emitted.report["transfer_configuration"]["placement_estimate_available"]
    selector = emitted.cuda.split('extern "C" int ' + emitted.report["planning"]["selector"], 1)[1]
    assert "costs.valid = 1" not in selector
    assert "costs.h2d_bandwidth =" not in selector
    assert "!preview.available ||" in selector


@pytest.mark.parametrize("transfers", ["auto", "pipelined"])
def test_direct_transfer_fallback_retains_calibrated_direct_placement(tmp_path, transfers):
    emitted = emit(tmp_path, transfers, profile=calibration())
    assert emitted.report["automatic_estimate_available"]
    assert emitted.report["transfer_configuration"]["selected"] == "direct"
    assert emitted.report["transfer_configuration"]["reason"]
    assert "costs.valid = 1" in emitted.cuda


def source_scope(tmp_path, transfers, *, collective=False, structured=False, policy="sections", profile=None):
    source = tmp_path / "original.f90"
    text = TEAM_SOURCE if collective else TREE if structured else PROGRAM
    source.write_text(text)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(source): sha256(source.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}}
    if collective:
        lines = text.splitlines(keepends=True)
        first = next(index for index, line in enumerate(lines, 1) if line.startswith("call renamed_step"))
        facts.update(schema_version=2,
                     captures={**{"argument::" + name: {**FACT, "association": "shared_whole_storage",
                                                         "descriptor_uniform": True} for name in ("a", "b", "c")},
                               "argument::n": {**FACT, "association": "shared_immutable_control"}},
                     participation={"kind": "omp_full_team", "dispatch": "qualified_companion",
                                    "entry": "operators::step", "host_threads": 4, "expected_omp_level": 1,
                                    "call_sites": [{"source": str(source), "caller": "callers::qualified",
                                                    "first_line": first, "last_line": first,
                                                    "span_sha256": sha256(lines[first-1].encode()).hexdigest(),
                                                    "team_first_line": first-1, "team_last_line": first+1,
                                                    "uniform_guard": "unconditional"}]})
    outputs, report = form_source_scopes([source], "operators::step" if collective else "step", facts=facts,
                                        options=CompilerOptions(gpu_policy=policy, memory_model="scoped", scope_transfers=transfers),
                                        config=OffloadConfig(policy, profile, 4, collective, scope_transfers=transfers))
    assert report["scope_count"] == 1, report["boundaries"]
    scope, = report["scopes"]
    replacement = outputs[report["sources"][str(source)]["replacement"]]
    owner = replacement.split("subroutine " + scope["owner"], 1)[1].split("end subroutine " + scope["owner"], 1)[0]
    return outputs, report, owner


@pytest.mark.parametrize("collective", [False, True])
@pytest.mark.parametrize("transfers", ["direct", "pinned", "pipelined", "auto"])
def test_owners_configure_once_before_registration_and_planning(tmp_path, collective, transfers):
    _, report, owner = source_scope(tmp_path, transfers, collective=collective)
    scope, = report["scopes"]
    assert scope["transfer_configuration"]["requested"] == transfers
    if transfers == "direct":
        assert "fort_configure" not in owner
        return
    assert owner.count("= fort_configure(") == 1
    configure = owner.index("= fort_configure(")
    assert owner.index("fort_scope_set_device_budget(") < configure < owner.index("fort_scope_register(")
    assert configure < owner.index("fort_scope_plan_reset(")
    if collective:
        assert owner.rfind("!$omp master", 0, configure) > owner.rfind("!$omp end master", 0, configure)
        assert "fort_state%status = fort_configure(fort_state%context)" in owner


@pytest.mark.parametrize("structured", [False, True])
def test_pinned_automatic_source_keeps_native_when_old_profile_has_only_direct_costs(tmp_path, structured):
    from compiler.offload.numerical_calibration import profile_from_compute_measurements
    from compiler.tests.test_numerical_calibration_v2 import observations_v2

    profile = profile_from_compute_measurements(calibration(), observations_v2(), calibration={})
    outputs, report, owner = source_scope(tmp_path, "pinned", policy="auto", profile=profile, structured=structured)
    assert not report["automatic_estimate_available"]
    scope, = report["scopes"]
    assert scope["planning_reason"] == "transfer_estimates_unavailable"
    assert scope["transfer_configuration"]["placement_estimate_reason"] == "transfer_estimates_unavailable"
    preflight = scope["automatic_preflight"]
    assert preflight["successful"] and preflight["selection"] == "native"
    assert "transfer_estimates_unavailable" in preflight["reason"]
    assert preflight["runtime_decision_inputs"] == []
    assert preflight["contexts_created"] == preflight["registrations"] == preflight["queries_constructed"] == 0
    assert "Successful static native selection:" in owner
    assert "fort_scope_create(" not in owner
    assert "fort_scope_register(" not in owner
    assert "fort_scope_plan_reset" not in owner
    assert "fort_configure(" not in owner
