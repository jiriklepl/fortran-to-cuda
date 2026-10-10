"""Native Fortran placement cannot borrow generated-worker calibration."""

from dataclasses import replace

import pytest

from compiler.analysis import build_execution_plan
from compiler.emission import generate_sources
from compiler.emission.common.resources import read_scoped_runtime
from compiler.frontend import lower_source
from compiler.offload.analysis import analyze_offload
from compiler.offload.config import OffloadConfig
from compiler.offload.numerical_calibration import profile_from_compute_measurements
from compiler.offload.source_compute import (apply_source_compute_costs,
    fork_join_schedule_compatible, native_participation)
from compiler.tests.test_numerical_calibration import observations
from compiler.tests.test_numerical_calibration_v2 import observations_v2
from compiler.tests.test_offload_profile import scoped_profile

SOURCE = """module generic_source_compute
contains
subroutine evaluate(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=sqrt(a(i)*a(i)+1.0_8)+cos(a(i))
enddo
end subroutine
end module
"""


def source_analysis():
    function = lower_source(SOURCE, "evaluate", source_name="generic_source_compute.f90")
    plan = build_execution_plan(function)
    return function, plan, analyze_offload(function, plan)


def profile_v2():
    profile = scoped_profile()
    profile["scoped"]["runtime_id"] = read_scoped_runtime()[1]["runtime_id"]
    profile["toolchain"].update(host_cxx_version="GCC 14.4.0", nvcc_version="NVCC V13.4.92")
    return profile_from_compute_measurements(profile, observations_v2(), calibration={})


def original_source_effects(tmp_path, directive="parallel do", *, guarded=False):
    from compiler.frontend.source_effects import SourceEffects

    source = SOURCE.replace("do i=1,n", "!$omp " + directive + "\ndo i=1,n")
    source = source.replace("enddo", "enddo\n!$omp end parallel do")
    if guarded:
        source = source.replace("!$omp " + directive, "if (n>0) then\n!$omp " + directive, 1)
        source = source.replace("!$omp end parallel do", "!$omp end parallel do\nendif")
    path = tmp_path / "generic_source_compute.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    procedure = "generic_source_compute::evaluate"
    return analysis, procedure, tuple(analysis.routines[procedure].execution.content)


@pytest.mark.parametrize("schedule", ["", " schedule(static)"])
@pytest.mark.parametrize("guarded", [False, True])
def test_original_static_joined_teams_have_calibrated_fork_join_participation(tmp_path, schedule, guarded):
    analysis, procedure, nodes = original_source_effects(tmp_path, "parallel do" + schedule, guarded=guarded)
    assert fork_join_schedule_compatible(nodes)
    assert native_participation(analysis, procedure) == "fork_join"


@pytest.mark.parametrize("schedule", ["schedule(runtime)", "SCHEDULE ( RUNTIME )"])
@pytest.mark.parametrize("guarded", [False, True])
def test_runtime_schedules_retain_completion_but_not_automatic_compute_estimates(tmp_path, schedule, guarded):
    from compiler.scopes.segments import grouped_nodes

    analysis, procedure, nodes = original_source_effects(tmp_path, "parallel do " + schedule, guarded=guarded)
    # Completion authenticates the retained native body, independently of its
    # compatibility with the static offline performance fixture.
    selected = nodes
    if guarded:
        branch, = grouped_nodes(nodes)
        selected = tuple(branch.content[1:-1])
    assert analysis.joined_completion(procedure, selected).public()["available"]
    assert not fork_join_schedule_compatible(nodes)
    participation = native_participation(analysis, procedure)
    assert participation == "fork_join_runtime"
    _, _, numerical = source_analysis()
    unit, = apply_source_compute_costs(numerical, profile_v2(), participation).units
    assert unit.compute_model is None
    assert "runtime schedule validation is missing" in unit.work_estimate_reason


def test_schedule_compatibility_is_local_to_the_selected_original_region(tmp_path):
    from compiler.frontend.source_effects import SourceEffects
    from compiler.scopes.segments import grouped_nodes

    analysis, procedure, nodes = original_source_effects(tmp_path, "parallel do schedule(static)")
    path = analysis.routines[procedure].scope.path
    source = path.read_text()
    first = source[source.index("!$omp parallel"):source.index("!$omp end parallel do") + len("!$omp end parallel do")]
    path.write_text(source.replace(first, first + "\n" + first.replace("schedule(static)", "schedule(runtime)")))
    analysis = SourceEffects([path])
    static, runtime = grouped_nodes(tuple(analysis.routines[procedure].execution.content))
    assert fork_join_schedule_compatible(static)
    assert not fork_join_schedule_compatible(runtime)
    assert native_participation(analysis, procedure) == "unknown"


@pytest.mark.parametrize("schedule", ["static,1", "static,4", "dynamic", "guided"])
def test_static_chunked_and_other_schedules_do_not_borrow_contiguous_costs(tmp_path, schedule):
    from compiler.offload.source_compute import fork_join_participation
    _, _, nodes = original_source_effects(tmp_path, "parallel do schedule(" + schedule + ")")
    assert fork_join_participation(nodes) == "unknown"


@pytest.mark.parametrize("flags", [{"exact": False}, {"full_read": True}, {"full_write": True}])
def test_piecewise_memory_unknown_physical_sections_decline_only_the_estimate(flags):
    _, _, analysis = source_analysis()
    original, = analysis.units
    footprint, = original.footprints
    unit = replace(original, footprints=(replace(footprint, **flags),))
    altered = replace(analysis, units=(unit,))
    result = apply_source_compute_costs(altered, profile_v2(), "serial")
    assert result.available == altered.available
    assert result.units[0].region is unit.region
    assert result.units[0].footprints == unit.footprints
    assert result.units[0].compute_model is None
    assert "exact physical working-set sections" in result.units[0].work_estimate_reason


def test_source_models_restore_known_work_and_keep_original_backend_distinct():
    _, _, analysis = source_analysis()
    original, = analysis.units
    assert original.work_per_iteration is None
    modeled = apply_source_compute_costs(analysis, profile_v2(), "serial")
    unit, = modeled.units
    assert unit.work_per_iteration == original.arithmetic_work_per_iteration
    assert unit.compute_model["native_fortran"]["backend_identity"] == "native_serial"
    assert unit.compute_model["generated_cpu"]["backend_identity"] == "generated_cpu"
    assert unit.workload_features == original.workload_features
    assert unit.region is original.region
    assert unit.footprints == original.footprints


def test_cpp_only_evidence_does_not_establish_original_fortran_costs():
    from compiler.offload.numerical_calibration import profile_from_measurements

    _, _, analysis = source_analysis()
    profile = profile_from_measurements(scoped_profile(), observations(), calibration={})
    unit, = apply_source_compute_costs(analysis, profile, "serial").units
    assert unit.work_per_iteration is None
    assert unit.compute_model is None
    assert "native Fortran" in unit.work_estimate_reason


def test_missing_existing_team_and_shared_fork_join_contracts_remain_unavailable():
    _, _, analysis = source_analysis()
    for participation in ("unknown", "existing_team"):
        unit, = apply_source_compute_costs(analysis, profile_v2(), participation).units
        assert unit.compute_model is None
        assert "participation unavailable" in unit.work_estimate_reason
    multiple = replace(analysis, units=analysis.units * 2)
    assert all("multi-loop cost contract" in unit.work_estimate_reason
               for unit in apply_source_compute_costs(multiple, profile_v2(), "fork_join").units)


def test_public_scoped_queries_supply_three_totals_with_runtime_range_and_placement_checks():
    function, plan, _ = source_analysis()
    generated = generate_sources(function, plan,
        offload_config=OffloadConfig(policy="auto", profile=profile_v2(), native_participation="serial"),
        memory_model="scoped")
    public = generated.scoped
    assert public["planning"]["available"]
    assert public["compute_estimates"]["native_backend"] == "original Fortran"
    unit, = public["planning"]["units"]
    assert unit["workload_class"] == "scalar_expression_v2"
    assert unit["workload_features"]["schema_version"] == 2
    assert unit["compute_model"]["item_range"] == [65536, 1048576]
    source = next(text for name, text in generated.artifacts.items() if name.endswith(".cu") and "record_compute(" in text)
    assert ".record_compute(" in source
    assert "fort_compute.native_fortran_seconds" in source
    assert "fort_compute.generated_cpu_seconds" in source
    assert "fort_compute.gpu_seconds" in source
    assert "iterations >= 65536ULL && unit.units[0].iterations <= 1048576ULL" in source
    assert "sched_getaffinity" in source
    assert "CPU_COUNT(&actual) != 4" in source


def test_unavailable_original_object_identity_is_an_explicit_native_requirement(tmp_path, monkeypatch):
    from compiler.scopes.compute_identity import ComputeIdentityGuard
    from compiler.tests.test_scoped_planning_sources import generate

    monkeypatch.setattr("compiler.scopes.compute_identity.compute_identity_guard",
        lambda *args, **kwargs: ComputeIdentityGuard(available=False, required=True,
            expression=".false.", reason="original child identity unavailable"))
    outputs, manifest = generate(tmp_path, monkeypatch)
    scope, = manifest["scopes"]
    assert not scope["estimate_available"]
    assert not manifest["automatic_estimate_available"]
    assert scope["planning_reason"] == "original child identity unavailable"
    assert not scope["compute_identity"]["available"]
    source = next(text for name, text in outputs.items() if name.startswith("sources/"))
    caller = source[source.index("subroutine step"):source.index("end subroutine", source.index("subroutine step"))]
    assert "if (.false.) then" in caller
    assert caller.index("else\ncall producer(") < caller.index("call consumer(")
