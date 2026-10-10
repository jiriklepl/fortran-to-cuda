"""IEEE calls preserve original thread environments and native LOGICAL state."""

from copy import copy
from dataclasses import replace

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.native_environment import intrinsic_export
from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError

ENTRY = "renamed_environment::advance"


def analyze(tmp_path, *, array=True, declaration=None, body=None, use="use, intrinsic :: ieee_exceptions",
            module_spec="", child="", other_sources=()):
    state = "logical :: modes(size(ieee_all))" if array else "logical :: modes"
    flag = "ieee_all" if array else "ieee_invalid"
    source = tmp_path / "renamed.f90"
    source.write_text(f"""module renamed_environment
{module_spec}
contains
subroutine advance(a)
{use}
implicit none
real(8),intent(inout)::a(:)
{declaration or state}
{body or f'call ieee_get_halting_mode({flag}, modes)' + chr(10) + f'call ieee_set_halting_mode({flag}, .false.)' + chr(10) + f'call ieee_set_halting_mode({flag}, modes)'}
{('contains' + chr(10) + child) if child else ''}
end subroutine
end module
""")
    analysis = SourceEffects([*other_sources, source])
    routine = analysis.routines[ENTRY]
    calls = tuple(node for node in walk(routine.execution) if _kind(node) == "Call_Stmt")
    return source, analysis, routine, calls


@pytest.mark.parametrize("array", [False, True])
def test_original_typed_state_and_ordered_environment_effects(tmp_path, array):
    path, analysis, routine, calls = analyze(tmp_path, array=array)
    original = path.read_bytes()
    proofs = [analysis.native_environment(ENTRY, call) for call in calls]
    assert all(proof.validate(analysis, ENTRY, call) is proof for proof, call in zip(proofs, calls, strict=True))
    assert proofs[0].written_roots == (ENTRY + "::modes",)
    assert proofs[1].state_roots == proofs[1].written_roots == ()
    assert proofs[2].state_roots == (ENTRY + "::modes",)
    state = analysis.native_environment_state(ENTRY, routine.scope.bindings["modes"])
    assert state is proofs[0].native_states[0] is proofs[2].native_states[0]
    assert state.public()["type"] == "logical"
    assert state.public()["kind"] == 4
    assert state.public()["rank"] == int(array)
    assert not state.public()["device_capture"]
    assert not state.public()["managed_definitions"]
    assert not state.public()["representation_conversion"]
    if array:
        assert "extent or payload evaluation" in state.public()["shape_authority"]
    else:
        assert state.public()["shape_authority"] == "scalar"
    summary = analysis.summarize(ENTRY)
    assert summary["complete"], summary["reasons"]
    assert not summary["cloneable"]
    assert [effect["kind"] for effect in summary["ordered_effects"]] == [
        "environment_read", "write", "environment_write", "environment_write", "read"]
    assert [operation["kind"] for operation in summary["operations"]] == ["native_environment"] * 3
    assert path.read_bytes() == original


def test_private_projection_uses_registered_original_token(tmp_path):
    _path, analysis, _routine, calls = analyze(tmp_path)
    first = analysis.native_environment(ENTRY, calls[0])
    summary = analysis.segment_summary(ENTRY, (calls[0],), capture_locals=True)
    assert summary["complete"], summary["reasons"]
    assert summary["operations"][0]["proof_identity"] == first.identity
    assert summary["ordered_effects"][0]["native_environment_identity"] == first.identity
    assert analysis.native_environment(ENTRY, calls[0]) is first


def test_source_parent_retains_environment_effects_without_clone_authority(tmp_path):
    path, _analysis, _routine, _calls = analyze(tmp_path, array=False)
    parent = """subroutine parent(a)
real(8),intent(inout)::a(:)
call advance(a)
end subroutine
"""
    path.write_text(path.read_text().replace("end module", parent + "end module"))
    analysis = SourceEffects([path])
    summary = analysis.summarize("renamed_environment::parent")
    assert summary["complete"], summary["reasons"]
    assert not summary["cloneable"]
    assert [effect["kind"] for effect in summary["ordered_effects"]] == [
        "environment_read", "write", "environment_write", "environment_write", "read"]


def test_persistent_summary_cache_never_issues_environment_tokens(tmp_path):
    path, _analysis, _routine, _calls = analyze(tmp_path)
    cache = tmp_path / "cache"
    first = SourceEffects([path], summary_cache=cache)
    assert first.summarize(ENTRY)["complete"]
    second = SourceEffects([path], summary_cache=cache)
    assert second.summarize(ENTRY)["complete"]
    assert second._native_environments
    assert not {id(token) for token in first._native_environments.values()}.intersection(
        id(token) for token in second._native_environments.values())
    token = next(iter(first._native_environments.values()))
    with pytest.raises(CompilationError, match="registered original"):
        token.validate(second, ENTRY, token._call)


def test_intrinsic_renames_keywords_and_reexports(tmp_path):
    exported = tmp_path / "exports.f90"
    exported.write_text("""module renamed_exports
use, intrinsic :: ieee_exceptions, only: query_mode=>ieee_get_halting_mode, &
 change_mode=>ieee_set_halting_mode, exceptions=>ieee_all
end module
""")
    _path, analysis, routine, calls = analyze(tmp_path, other_sources=(exported,),
        use="use renamed_exports, only: query_mode, change_mode, exceptions",
        declaration="logical::modes(size(exceptions))",
        body="call query_mode(halting=modes,flag=exceptions)\ncall change_mode(flag=exceptions,halting=modes)")
    assert intrinsic_export(analysis, routine.scope, "query_mode") == "$intrinsic::ieee_exceptions::ieee_get_halting_mode"
    assert [analysis.native_environment(ENTRY, call).intrinsic.rsplit("::", 1)[1] for call in calls] == [
        "ieee_get_halting_mode", "ieee_set_halting_mode"]


def test_duplicate_standard_reexports_are_one_intrinsic(tmp_path):
    _path, analysis, routine, calls = analyze(tmp_path, use="use ieee_arithmetic\nuse ieee_exceptions")
    assert intrinsic_export(analysis, routine.scope, "ieee_get_halting_mode") == "$intrinsic::ieee_exceptions::ieee_get_halting_mode"
    assert analysis.native_environment(ENTRY, calls[0]).state_roots == (ENTRY + "::modes",)


def test_unrestricted_rename_excludes_original_name(tmp_path):
    _path, analysis, routine, calls = analyze(tmp_path, array=False,
        use="use ieee_exceptions, query=>ieee_get_halting_mode",
        body="call query(ieee_invalid,modes)\ncall ieee_get_halting_mode(ieee_invalid,modes)")
    assert intrinsic_export(analysis, routine.scope, "query") is not None
    assert intrinsic_export(analysis, routine.scope, "ieee_get_halting_mode") is None
    with pytest.raises(CompilationError, match="unproved source use"):
        analysis.native_environment(ENTRY, calls[0])


@pytest.mark.parametrize(("use", "module_spec"), [
    ("use, non_intrinsic :: ieee_exceptions", ""),
    ("use ieee_features", ""),
    ("use ieee_exceptions\nuse unavailable", ""),
    ("use ieee_exceptions,only: get=>ieee_get_halting_mode\nuse ieee_exceptions,only: get=>ieee_set_halting_mode", ""),
])
def test_unproved_intrinsic_exports_never_gain_authority(tmp_path, use, module_spec):
    _path, analysis, _routine, calls = analyze(tmp_path, array=False, use=use, module_spec=module_spec,
        body="call get(ieee_invalid,modes)" if "get=>" in use else None)
    with pytest.raises(CompilationError, match="resolution|source use"):
        analysis.native_environment(ENTRY, calls[0])


def test_source_module_named_like_intrinsic_is_not_intrinsic(tmp_path):
    application = tmp_path / "impostor.f90"
    application.write_text("""module ieee_exceptions
integer,parameter::ieee_invalid=1
contains
subroutine ieee_get_halting_mode(flag,mode)
integer,intent(in)::flag
logical,intent(out)::mode
mode=.false.
end subroutine
end module
""")
    _path, analysis, _routine, calls = analyze(tmp_path, array=False, use="use ieee_exceptions",
        other_sources=(application,), body="call ieee_get_halting_mode(ieee_invalid,modes)")
    with pytest.raises(CompilationError, match="intrinsic resolution"):
        analysis.native_environment(ENTRY, calls[0])


def test_local_generic_interface_cannot_acquire_intrinsic_authority(tmp_path):
    _path, analysis, _routine, calls = analyze(tmp_path, array=False,
        declaration="logical::modes\ninterface ieee_get_halting_mode\nmodule procedure impostor\nend interface")
    with pytest.raises(CompilationError, match="intrinsic resolution|source use"):
        analysis.native_environment(ENTRY, calls[0])


@pytest.mark.parametrize("declaration", [
    "logical,save::modes", "logical::modes=.false.", "logical,target::modes", "logical,pointer::modes",
    "logical,allocatable::modes", "logical,volatile::modes", "logical(1)::modes", "logical::modes(5)",
    "logical::modes(2:size(ieee_all))", "logical::modes(size(ieee_usual))", "real(8)::modes",
])
def test_unproved_local_storage_remains_native_boundary(tmp_path, declaration):
    _path, analysis, _routine, calls = analyze(tmp_path, declaration=declaration)
    with pytest.raises(CompilationError, match="LOGICAL|bounds"):
        analysis.native_environment(ENTRY, calls[0])
    assert not analysis._native_environments
    assert not analysis._native_environment_states


@pytest.mark.parametrize("extra", ["call opaque(modes)", "a(1)=merge(1.d0,0.d0,modes)", "print *,modes"])
def test_any_other_state_use_rejects_native_only_state(tmp_path, extra):
    body = "call ieee_get_halting_mode(ieee_invalid,modes)\n" + extra
    _path, analysis, _routine, calls = analyze(tmp_path, array=False, body=body)
    with pytest.raises(CompilationError, match="unproved source use or escape"):
        analysis.native_environment(ENTRY, calls[0])


def test_unused_internal_child_host_association_is_not_hidden(tmp_path):
    child = "subroutine observer()\nprint *,modes\nend subroutine"
    _path, analysis, _routine, calls = analyze(tmp_path, array=False, child=child)
    with pytest.raises(CompilationError, match="unproved source use or escape"):
        analysis.native_environment(ENTRY, calls[0])


@pytest.mark.parametrize("child", [False, True])
def test_specification_expression_cannot_hide_a_native_state_read(tmp_path, child):
    declaration = "logical::modes"
    observer = "subroutine observer()\nreal::scratch(merge(2,3,modes))\nend subroutine"
    if not child:
        declaration += "\nreal::scratch(merge(2,3,modes))"
    _path, analysis, _routine, calls = analyze(tmp_path, array=False, declaration=declaration,
                                              child=observer if child else "")
    with pytest.raises(CompilationError, match="unproved source use or escape"):
        analysis.native_environment(ENTRY, calls[0])


@pytest.mark.parametrize("spec", ["common /shared/ modes", "logical::other\nequivalence(modes,other)",
                                  "!$omp threadprivate(modes)"])
def test_storage_association_and_threadprivate_decline(tmp_path, spec):
    _path, analysis, _routine, calls = analyze(tmp_path, array=False, declaration="logical::modes\n" + spec)
    with pytest.raises(CompilationError, match="association|THREADPRIVATE"):
        analysis.native_environment(ENTRY, calls[0])


@pytest.mark.parametrize("body", [
    "call ieee_get_halting_mode(ieee_invalid,.false.)",
    "call ieee_get_halting_mode(ieee_all,modes)",
    "call ieee_set_halting_mode(ieee_invalid,.false._4)",
    "call ieee_get_halting_mode(ieee_invalid,halting=modes,flag=ieee_invalid)",
    "call ieee_get_halting_mode(halting=modes,ieee_invalid)",
    "call ieee_get_halting_mode(flag=ieee_invalid)",
])
def test_unproved_actuals_are_not_evaluated_or_rewritten(tmp_path, body):
    _path, analysis, _routine, calls = analyze(tmp_path, array=False, body=body)
    with pytest.raises(CompilationError, match="HALTING|shape|association|actuals"):
        analysis.native_environment(ENTRY, calls[0])


def test_registered_tokens_cannot_be_copied_forged_or_reused_after_source_change(tmp_path):
    path, analysis, routine, calls = analyze(tmp_path)
    proof = analysis.native_environment(ENTRY, calls[0])
    state = proof.native_states[0]
    with pytest.raises(CompilationError, match="registered original"):
        copy(proof).validate(analysis, ENTRY, calls[0])
    with pytest.raises(CompilationError, match="registered original"):
        replace(state).validate(analysis, ENTRY)
    with pytest.raises(CompilationError, match="exact original"):
        analysis.native_environment(ENTRY, F.Call_Stmt(str(calls[0])))
    with pytest.raises(CompilationError, match="registered original"):
        proof.validate(analysis, ENTRY, calls[1])
    path.write_text(path.read_text() + "! changed source\n")
    with pytest.raises(CompilationError):
        proof.validate(analysis, ENTRY, calls[0])
    assert routine.scope.bindings["modes"].dtype == "logical"


@pytest.mark.parametrize("private", [False, True])
def test_complete_original_teams_keep_environment_operations_and_forbid_cuts(tmp_path, private):
    body = """!$omp parallel default(none) """ + ("private(modes)" if private else "shared(modes)") + """
call ieee_get_halting_mode(ieee_all,modes)
call ieee_set_halting_mode(ieee_all,.false.)
call ieee_set_halting_mode(ieee_all,modes)
!$omp end parallel
"""
    path, analysis, routine, _calls = analyze(tmp_path, body=body)
    original = path.read_bytes()
    selected = tuple(_children(routine.execution))
    proof = analysis.joined_completion(ENTRY, selected)
    facts = proof.public()
    assert facts["native_environment_contract"] == "original-ieee-halting-mode-v1"
    assert facts["native_only"]
    assert not facts["internal_cuts_authorized"]
    assert facts["requires_completed_device_work"]
    assert not facts["gpu_legality_established"]
    assert len(facts["native_environment_operations"]) == 3
    assert facts["private_resources"] == ([ENTRY + "::modes"] if private else [])
    from compiler.frontend.native_completion import _joined_completion_facts
    with pytest.raises(CompilationError, match="worksharing cuts"):
        _joined_completion_facts(analysis, ENTRY, selected, worksharing=[])
    with pytest.raises(CompilationError, match="source helper|synchronous source helper"):
        analysis.numerical_joined_completion(ENTRY, selected)
    assert path.read_bytes() == original


def test_missing_original_team_join_remains_boundary(tmp_path):
    _path, analysis, routine, _calls = analyze(tmp_path,
        body="!$omp parallel\ncall ieee_get_halting_mode(ieee_all,modes)")
    with pytest.raises(CompilationError, match="joined END PARALLEL"):
        analysis.joined_completion(ENTRY, tuple(_children(routine.execution)))


def test_set_scalar_state_broadcast_retains_original_read(tmp_path):
    _path, analysis, _routine, calls = analyze(tmp_path, array=False,
        body="call ieee_get_halting_mode(ieee_invalid,modes)\ncall ieee_set_halting_mode(ieee_all,modes)")
    proof = analysis.native_environment(ENTRY, calls[1])
    assert proof.state_roots == (ENTRY + "::modes",)
    assert [effect["kind"] for effect in proof.public()["effects"]] == ["environment_write", "read"]


def test_guarded_calls_do_not_publish_or_evaluate_native_state(tmp_path):
    body = """if(size(a)>0) then
call ieee_get_halting_mode(ieee_invalid,modes)
call ieee_set_halting_mode(ieee_invalid,modes)
endif
"""
    _path, analysis, _routine, calls = analyze(tmp_path, array=False, body=body)
    summary = analysis.summarize(ENTRY)
    assert summary["complete"], summary["reasons"]
    for call in calls:
        public = analysis.native_environment(ENTRY, call).public()
        assert public["requires_original_execution"]
        assert public["requires_original_thread_participation"]
        assert not public["logical_representation_conversion"]
    assert [operation["guard"] for operation in summary["operations"]
            if operation["kind"] == "native_environment"] == [("SIZE(a) > 0",)] * 2
    assert [operation["resource"] for operation in summary["operations"]
            if operation["kind"] == "descriptor_read"] == ["argument::a"]
    assert not any(operation["kind"] in {"read", "write", "overwrite"}
                   and operation.get("resource") == ENTRY + "::modes"
                   for operation in summary["operations"])


@pytest.mark.parametrize("mixed", [False, True])
def test_atomic_native_union_retains_authenticated_environment_order(tmp_path, mixed):
    from compiler.scopes.native_atomic import summarize

    body = """!$omp parallel default(none) shared(a,modes) private(i)
call ieee_get_halting_mode(ieee_all,modes)
call ieee_set_halting_mode(ieee_all,.false.)
"""
    if mixed:
        body += "!$omp do\ndo i=1,size(a)\na(i)=a(i)+sum(a(i:i))\nenddo\n!$omp end do nowait\n"
    body += "call ieee_set_halting_mode(ieee_all,modes)\n!$omp end parallel\n"
    path, analysis, routine, _calls = analyze(tmp_path, declaration="logical::modes(size(ieee_all))\ninteger::i", body=body)
    before = path.read_bytes()
    selected = tuple(_children(routine.execution))
    completion = analysis.joined_completion(ENTRY, selected)
    summary = summarize(analysis, ENTRY, selected, completion)
    assert summary["complete"]
    operations = [operation for operation in summary["operations"] if operation["kind"] == "native_environment"]
    assert [item["proof_identity"] for item in operations] == [
        item["proof_identity"] for item in completion.public()["native_environment_operations"]]
    assert [effect["kind"] for effect in summary["ordered_effects"]] == [
        "environment_read", "write", "environment_write", "environment_write", "read"]
    assert not summary["guaranteed_whole_overwrites"]
    assert summary["native_atomic"]["requires_completed_device_work"]
    assert not summary["native_atomic"]["internal_cuts_authorized"]
    assert path.read_bytes() == before


def test_atomic_environment_rejects_unregistered_serialized_facts(tmp_path, monkeypatch):
    from compiler.scopes.native_atomic import summarize

    body = "!$omp parallel\ncall ieee_set_halting_mode(ieee_invalid,.false.)\n!$omp end parallel\n"
    _path, analysis, routine, _calls = analyze(tmp_path, array=False, body=body)
    selected = tuple(_children(routine.execution))
    completion = analysis.joined_completion(ENTRY, selected)
    original = analysis.segment_summary

    def forged(*args, **kwargs):
        summary = original(*args, **kwargs)
        for operation in summary["operations"]:
            if operation["kind"] == "native_environment":
                operation["proof_identity"] = "0" * 64
        return summary

    monkeypatch.setattr(analysis, "segment_summary", forged)
    with pytest.raises(CompilationError, match="registered original source authority"):
        summarize(analysis, ENTRY, selected, completion)
