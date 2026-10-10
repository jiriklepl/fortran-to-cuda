"""Reachable original IEEE state restricts numerical work before outlining."""

from hashlib import sha256
from types import SimpleNamespace

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.lexical import immutable_numerical_unit
from compiler.scopes.segments import Segment
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_lexical_source_owner import MODULE_SOURCE, SOURCE, emit
from compiler.tests.test_reached_counted_loops import builder_for
from compiler.tests.test_reached_stage_prerequisites_cuda import sources
from compiler.tests.test_source_scopes import FACT


def numerical_regions(report):
    registries = [report["inline_numerical_regions"], *report["child_numerical_regions"]]
    return [region for registry in registries for region in registry["regions"] if region["used"]]


def numerical_segments(owner):
    members = [owner, *owner["module_coordinators"], *owner["internal_coordinators"]]
    return [segment for member in members for unit in member["planning_segments"]
            for segment in unit["operations"]["planning_segments"]]


def trap_source():
    # With a=+Inf, the producer writes -Inf while traps are disabled. The
    # original consumer must retain the restored invalid-operation trap.
    return (SOURCE.replace("subroutine step(a,b,out,n,escape)",
                           "subroutine step(a,b,out,n,escape)\nuse ieee_exceptions")
            .replace("visits=visits+1", "visits=visits+1\ncall ieee_set_halting_mode(ieee_invalid,.false.)")
            .replace("b(i)=2*a(i)+real(i,8)", "b(i)=-a(i)")
            .replace("if(escape) then\ncall opaque(b,n)\nendif",
                     "call ieee_set_halting_mode(ieee_invalid,.true.)"))


@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("observer", ["ieee_get_flag", "peek"])
def test_plain_arithmetic_cannot_lose_original_exception_flags(tmp_path, precision, observer):
    import_name = "ieee_get_flag" if observer == "ieee_get_flag" else "peek=>ieee_get_flag"
    path = tmp_path / "observed.f90"
    path.write_text(f"""module observed
contains
subroutine step(a,b,out,n,seen)
use,intrinsic::ieee_exceptions,only:{import_name},ieee_invalid
real({precision}),intent(in)::a(:),b(:)
real({precision}),intent(inout)::out(:)
integer,intent(in)::n
logical,intent(out)::seen
integer::i
do i=1,n
out(i)=a(i)*b(i)
enddo
call {observer}(ieee_invalid,seen)
end subroutine
end module
""")
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::"+name: FACT for name in ("a", "b", "out")}}
    outputs, report = ScopeBuilder([path], "observed::step", facts=facts,
        options=CompilerOptions(), config=OffloadConfig(policy="sections", scope_execution="reached")).run()
    assert not numerical_regions(report)
    assert "source-observable floating-point exception flags" in str(report["boundaries"])
    assert not any(name.endswith("shared_entry.cu") for name in outputs)


def test_plain_arithmetic_user_observer_spelling_is_not_intrinsic_authority(tmp_path):
    path = tmp_path / "ordinary.f90"
    path.write_text("""module ordinary
contains
subroutine ieee_get_flag(seen)
logical,intent(out)::seen
seen=.false.
end subroutine
subroutine step(a,out,n,seen)
real(8),intent(in)::a(:)
real(8),intent(inout)::out(:)
integer,intent(in)::n
logical,intent(out)::seen
integer::i
do i=1,n
out(i)=2*a(i)
enddo
call ieee_get_flag(seen)
end subroutine
end module
""")
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::"+name: FACT for name in ("a", "out")}}
    _outputs, report = ScopeBuilder([path], "ordinary::step", facts=facts,
        options=CompilerOptions(), config=OffloadConfig(policy="sections", scope_execution="reached")).run()
    assert len(numerical_regions(report)) == 1


def test_plain_arithmetic_after_restored_trap_has_reached_native_fallback(tmp_path):
    path, outputs, report = emit(tmp_path, trap_source())
    owner, = report["scopes"]
    regions = numerical_regions(report)
    assert len(regions) == 2
    assert all(region["numerical_environment"]["required"] for region in regions)
    assert all("fort_scope_numerical_environment_supported() /= 0" in segment["semantic_guards"]
               for segment in numerical_segments(owner))
    text = outputs[report["sources"][str(path)]["replacement"]]
    restored = text.index("call ieee_set_halting_mode(ieee_invalid,.true.)")
    suffix = text[restored:]
    checked = suffix.index("fort_scope_numerical_environment_supported()")
    guarded = suffix[checked:]
    assert "fort_scope_plan_reset_mode" in guarded
    assert "fort_scope_host_begin" in guarded
    assert "fort_scope_host_end" in guarded
    assert "FORT_SCOPE_PLAN_NATIVE" in guarded
    assert all(segment["guard_failure"] == "original native operation with coherence hooks; earlier work is retained"
               for segment in numerical_segments(owner))
    assert not owner["boundaries"]
    # The generated public query and execution entry also receive the flag;
    # guards are part of the artifact identity, not a late metadata mutation.
    cuda = [value for name, value in outputs.items() if name.endswith("shared_entry.cu")]
    assert len(cuda) == 2
    assert all("numerical_environment_supported()" in value for value in cuda)


def test_child_ieee_state_guards_parent_regions_outlined_before_and_after_borrow(tmp_path):
    source = (MODULE_SOURCE.replace("subroutine adjust(x,n,escape)",
                                   "subroutine adjust(x,n,escape)\nuse ieee_exceptions")
              .replace("child_visits=child_visits+1", "child_visits=child_visits+1\n"
                       "call ieee_set_halting_mode(ieee_invalid,.false.)")
              .replace("if(escape) then\ncall opaque(x,n)\nendif",
                       "call ieee_set_halting_mode(ieee_invalid,.true.)"))
    _, _, report = emit(tmp_path, source)
    owner, = report["scopes"]
    assert len(report["child_numerical_regions"]) == 1
    regions = numerical_regions(report)
    assert len(regions) == 3
    assert all(region["numerical_environment"]["required"] for region in regions)
    assert all("fort_scope_numerical_environment_supported() /= 0" in segment["semantic_guards"]
               for segment in numerical_segments(owner))


def test_renamed_intrinsic_setter_still_requires_environment_guard(tmp_path):
    source = (trap_source().replace("use ieee_exceptions", "use ieee_exceptions,only: "
                                    "set_mode=>ieee_set_halting_mode,ieee_invalid")
              .replace("call ieee_set_halting_mode", "call set_mode"))
    _, _, report = emit(tmp_path, source)
    assert all(region["numerical_environment"]["required"] for region in numerical_regions(report))


def test_original_team_checks_every_participant_before_uniform_native_fallback(tmp_path):
    source = (trap_source().replace("do i=-2,n-3\nb(i)=-a(i)",
                                   "!$omp parallel private(i)\n!$omp do\ndo i=-2,n-3\nb(i)=-a(i)")
              .replace("b(i)=-a(i)\nenddo", "b(i)=-a(i)\nenddo\n!$omp end do\n!$omp end parallel"))
    path, outputs, report = emit(tmp_path, source)
    owner, = report["scopes"]
    assert len(owner["gpu_leaves"]) == 2
    assert not owner["boundaries"]
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.count("!$omp parallel private(i)") == 1
    team = text[text.index("!$omp parallel private(i)"):]
    aggregate = ("!$omp master\nfort_numerical_guard = .true.\n!$omp end master\n!$omp barrier\n"
                 "if (fort_native_ready) then\nif (fort_scope_numerical_environment_supported() == 0) then\n"
                 "!$omp atomic write\nfort_numerical_guard = .false.\nendif\nendif\n!$omp barrier")
    assert aggregate in team
    decision = team[team.index(aggregate)+len(aggregate):]
    assert decision.index("if (fort_numerical_guard) then") < decision.index("if (fort_team_run) then")
    assert "fort_numerical_guard = .true." not in decision[:decision.index("if (fort_team_run) then")]
    assert "else\n!$omp do\ndo i=-2,n-3\nb(i)=-a(i)\nenddo\n!$omp end do\nendif" in decision
    assert sum(region["execution_participation"] == "qualified_original_team"
               for region in numerical_regions(report)) == 1
    team_entries = [value for name, value in outputs.items()
                    if name.endswith("shared_entry.cu") and "shared->environment_supported" in value]
    assert len(team_entries) == 1
    assert "shared->environment_supported = false;" in team_entries[0]


def test_source_without_original_ieee_calls_keeps_existing_numerical_contract(tmp_path):
    _, _, report = emit(tmp_path, SOURCE)
    assert all(not region["numerical_environment"]["required"] for region in numerical_regions(report))


def test_disconnected_ieee_routine_does_not_change_whole_team_numerical_contract(tmp_path):
    source = (SOURCE.replace("do i=-2,n-3\nb(i)=2*a(i)+real(i,8)",
                             "!$omp parallel private(i)\n!$omp do\ndo i=-2,n-3\nb(i)=2*a(i)+real(i,8)")
              .replace("b(i)=2*a(i)+real(i,8)\nenddo", "b(i)=2*a(i)+real(i,8)\nenddo\n"
                       "!$omp end do\n!$omp end parallel"))
    source += """module disconnected
contains
subroutine set_modes()
use ieee_exceptions
call ieee_set_halting_mode(ieee_invalid,.true.)
end subroutine
end module
"""
    _, _, report = emit(tmp_path, source)
    regions = numerical_regions(report)
    assert len(regions) == 2
    assert all(not region["numerical_environment"]["required"] for region in regions)
    team_region, = [region for region in regions if region["completion"].get("has_openmp_in_closure")]
    assert team_region["execution_participation"] == "serial_coordinator"


def test_reachable_source_cycle_scan_never_materializes_effect_closures(tmp_path):
    source = SOURCE.replace("call opaque(b,n)", "call first()") + """subroutine first()
call second()
end subroutine
subroutine second()
call first()
end subroutine
"""
    builder = builder_for(tmp_path, source)
    assert not builder.analysis.summaries
    assert not builder.analysis._closures
    assert not builder.inline.native_environment_required()
    assert not builder.analysis.summaries
    assert not builder.analysis._closures


def test_reachable_call_budget_exhaustion_requires_guard_conservatively(tmp_path):
    builder = builder_for(tmp_path, SOURCE.replace("call opaque(b,n)", "call opaque(b,n)\ncall other()"))
    builder.analysis.operation_limit = 1
    assert builder.inline.native_environment_required()


@pytest.mark.parametrize("precision", [4, 8])
def test_counted_borrow_with_immutable_and_ieee_fallback_keeps_reached_owner(tmp_path, precision):
    # Use the exact native/CUDA fixture's original sources without calibration,
    # compilers or execution. Its guarded child must not discard the whole owner.
    paths = sources(tmp_path, precision)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {"argument::"+name: FACT for name in ("a", "b", "out")}}
    outputs, report = ScopeBuilder(paths, "owner_lib::advance", facts=facts,
        options=CompilerOptions(), config=OffloadConfig(policy="sections", scope_execution="reached")).run()
    owner, = report["scopes"]
    child, = owner["module_coordinators"]
    assert owner["counted_controls"]
    assert len(owner["gpu_leaves"]) == 2
    assert not child["boundaries"]
    assert all("immutable operands require" not in boundary["reason"] for boundary in report["boundaries"])
    assert all(region["numerical_environment"]["required"] for region in numerical_regions(report))
    child_segments = [segment for unit in child["planning_segments"]
                      for segment in unit["operations"]["planning_segments"]]
    assert len(child_segments) == 1
    assert child_segments[0]["semantic_guards"]
    fallbacks = [operation for unit in child["planning_segments"]
                 for operation in unit["operations"]["native_operations"]
                 if operation["kind"] == "guarded original numerical fallback"]
    assert len(fallbacks) == 1
    assert fallbacks[0]["immutable_inputs"]
    child_text = outputs[report["sources"][str(paths[1])]["replacement"]]
    child_region, = report["child_numerical_regions"][0]["regions"]
    capture, = child_region["scalar_element_captures"]
    activation = "if (" + capture["activation_guards"][0] + ") then"
    assert child_text.index(activation) < child_text.index("fort_can = is_contiguous(a)")
    assert child_text.index(capture["index_guard"]) < child_text.index("fort_inline_query(")
    assert "fort_scope_numerical_environment_supported()" in child_text
    # Coefficient payload remains reached run-only work, after the original
    # nonempty and index guards; the native fallback retains original math.
    assert "weights(stage)" in child_text
    assert "out(j)=a(j)*weights(stage)+real(j,"+str(precision)+")" in child_text
    assert any(region["execution_participation"] == "qualified_original_team"
               for region in numerical_regions(report))


def test_immutable_unit_accepts_only_identical_registered_fallback():
    call = SimpleNamespace(node=object())
    fallback = SimpleNamespace(kind="guarded original numerical fallback")
    scope = SimpleNamespace(calls=[call], tree=[Segment([call])], native=[fallback],
                            guarded={id(call.node): fallback})
    assert immutable_numerical_unit(scope)
    scope.native.append(SimpleNamespace(kind="native source"))
    assert not immutable_numerical_unit(scope)
    scope.native = [SimpleNamespace(kind=fallback.kind)]
    assert not immutable_numerical_unit(scope)
    scope.native = [fallback]
    scope.guarded = {id(object()): fallback}
    assert not immutable_numerical_unit(scope)
