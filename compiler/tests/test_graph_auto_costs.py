"""Graph placement heuristics cannot replace numerical or storage authority."""

from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_source
from compiler.offload.config import OffloadConfig
from compiler.offload.graph_costs import apply_graph_hardware_costs, graph_work_proxy
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_scoped_planning_entries import calibration
from compiler.tests.test_source_compute_costs import SOURCE, source_analysis


def emit(source=SOURCE, *, profile=None, collective=False, numerical_environment=False):
    function, plan = prepare_function(lower_source(source, "evaluate", source_name="graph_auto.f90"),
        options=CompilerOptions(gpu_policy="auto"))
    function = replace(function, requires_numerical_environment=numerical_environment)
    config = OffloadConfig("auto", profile, 4, collective,
                           native_participation="existing_team" if collective else "serial")
    return generate_scoped(function, plan, config, "common_functions.cuh",
                           runtime_id=read_scoped_runtime()[1]["runtime_id"])


def test_generic_static_proxy_prices_intrinsics_without_family_attestation():
    _, _, analysis = source_analysis()
    unit, = analysis.units
    assert unit.work_per_iteration is None
    proxy, uncertainty = graph_work_proxy(unit.region)
    assert proxy >= 72  # Explicit sqrt/cos proxy weights, not measured latency.
    assert uncertainty == ()
    units, diagnostics = apply_graph_hardware_costs({unit.region.id: unit})
    estimated = units[unit.region.id]
    assert estimated.footprints is unit.footprints
    assert estimated.region is unit.region
    assert estimated.compute_model is None
    assert not diagnostics[unit.region.id]["original_fortran_family_attestation"]
    assert not diagnostics[unit.region.id]["speedup_guaranteed"]


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("numerical_environment", [False, True])
def test_graph_auto_uses_hardware_rates_but_keeps_exact_effects_and_environment_checks(conditional, numerical_environment):
    source = SOURCE.replace("a(i)=sqrt", "if(a(i)>0) a(i)=sqrt") if conditional else SOURCE
    value = emit(source, profile=calibration(), numerical_environment=numerical_environment)
    assert value.report["automatic_estimate_available"]
    assert value.report["planning"]["query_available"]
    assert value.report["planning"]["available"]
    costs = value.report["graph_auto_costs"]
    assert costs["heuristic"]
    assert not costs["cost_authorizes_execution"]
    assert "costs.valid = 1" in value.cuda
    assert "scoped_host_compatible(profile)" in value.cuda
    assert "offload::compatible(profile" in value.cuda
    assert "fort_access.add(" in value.cuda
    assert "invalid checked planning footprint" in value.cuda
    if numerical_environment:
        assert "if (!fort_runtime::numerical_environment_supported())" in value.cuda
        assert "unsupported selection numerical environment" in value.cuda
    if conditional:
        assert any(item["original_work_conditional_or_upper_bound"] for item in costs["units"].values())


def test_graph_auto_allows_unpriced_team_overhead_but_preserves_actual_team_protocol():
    value = emit(profile=calibration(), collective=True)
    assert value.report["automatic_estimate_available"]
    assert value.report["graph_auto_costs"]["unpriced_team_overhead"]
    assert "fort_scope_set_team_costs_v1(fort_context, nullptr, 0)" in value.cuda
    assert "if (!fort_team_ready) costs.valid=0" not in value.cuda
    assert value.report["team"]["participation"] == "qualified_full_team"
    assert "omp_get_level() != 1 || omp_get_num_threads() != 4" in value.cuda
    assert "#pragma omp barrier" in value.cuda
    assert value.report["team"]["automatic_estimate_available"]


@pytest.mark.parametrize("profile", [None, "old_runtime", "wrong_precision"])
def test_hardware_profile_applicability_is_still_required(profile):
    value = None if profile is None else calibration()
    if profile == "old_runtime":
        value["scoped"]["runtime_id"] = "0" * 64
    if profile == "wrong_precision":
        value["precision_bits"] = 32
    generated = emit(profile=value)
    assert not generated.report["automatic_estimate_available"]
    assert "costs.valid = 1" not in generated.cuda


def test_ineligible_sequential_recurrence_never_acquires_a_gpu_implementation():
    source = SOURCE.replace("do i=1,n", "do i=2,n").replace(
        "a(i)=sqrt(a(i)*a(i)+1.0_8)+cos(a(i))", "a(i)=a(i-1)+1.0_8")
    function, plan = prepare_function(lower_source(source, "evaluate", source_name="recurrence.f90"),
        options=CompilerOptions(gpu_policy="auto", fallback="host"))
    assert not plan.regions
    assert function.body.statements


def test_original_source_owner_can_rank_without_original_fortran_family_evidence(tmp_path):
    path = tmp_path / "source.f90"
    path.write_text(SOURCE)
    builder = ScopeBuilder([path], "generic_source_compute::evaluate",
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
               "captures": {"argument::a": dict(FACT)}},
        options=CompilerOptions(), config=OffloadConfig("auto", calibration()))
    outputs, report = builder.run()
    owner, = report["scopes"]
    assert owner["planning_reason"] is None
    assert owner["gpu_leaves"]
    assert owner["planning_segments"][0]["compute_identity"]["required"] is False
    replacement = outputs[report["sources"][str(path)]["replacement"]]
    assert "fort_scope_plan_reset_mode" in replacement
    assert "fort_decision" in replacement
    assert "=> choose" in replacement
