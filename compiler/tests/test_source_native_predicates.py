"""Native predicate metadata does not replace original guarded data effects."""

import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects


def source(tmp_path, *, expression="ieee_is_nan(p)", declaration="real(8),intent(in)::p",
           use="use,intrinsic::ieee_arithmetic,only:ieee_is_nan", cache=None):
    path = tmp_path / "predicates.f90"
    path.write_text(f"""module flow
contains
subroutine leaf(p,flag,result)
{use}
{declaration}
logical,intent(in)::flag
logical,intent(inout)::result
if(flag) result={expression}
end subroutine
subroutine step(a,flag,result)
{declaration.replace('::p', '::a')}
logical,intent(in)::flag
logical,intent(inout)::result
if(flag) call leaf(a,flag,result)
end subroutine
end module
""")
    return SourceEffects([path], summary_cache=cache)


@pytest.mark.parametrize("expression", ["ieee_is_nan(p)", "ieee_is_nan(x=p)"])
def test_original_guarded_reads_and_native_only_requirements_propagate(tmp_path, expression):
    analysis = source(tmp_path, expression=expression)
    summary = analysis.summarize("flow::step")
    assert summary["complete"], summary["reasons"]
    requirement, = summary["native_predicate_requirements"]
    assert requirement["intrinsic"] == "$intrinsic::ieee_arithmetic::ieee_is_nan"
    assert requirement["guard_frames"] == [
        {"procedure": "flow::step", "condition": "flag"},
        {"procedure": "flow::leaf", "condition": "flag"}]
    assert requirement["native_only"]
    assert not requirement["gpu_legality_established"]
    assert not summary["cloneable"]
    assert not analysis.summarize("flow::leaf")["cloneable"]
    reads = [effect for effect in summary["ordered_effects"] if effect["resource"] == "argument::a"]
    assert len(reads) == 1
    assert reads[0]["kind"] == "read"
    assert reads[0]["guard_frames"] == requirement["guard_frames"]
    assert {operation["kind"] for operation in analysis.summarize("flow::leaf")["operations"]} <= {
        "read", "write", "overwrite"}


def test_predicate_source_requirements_are_reissued_after_cache_boundary(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    analysis = source(tmp_path, cache=cache)
    first = analysis.report("flow::step")
    import compiler.frontend.native_intrinsics as predicates
    original, issued = predicates.prove_native_predicate, []

    def checked(*args):
        issued.append(args)
        return original(*args)

    monkeypatch.setattr(predicates, "prove_native_predicate", checked)
    second = SourceEffects(analysis.inputs.paths, summary_cache=cache).report("flow::step")
    assert first["complete"]
    assert second["complete"]
    assert issued
    assert second["summary_cache"]["hits"] == 0
    assert analysis._summary_authority()["native_predicate_version"] == 1


def test_predicate_does_not_authorize_actual_storage_lifetime(tmp_path):
    analysis = source(tmp_path, declaration="real(8),pointer,intent(in)::p")
    summary = analysis.summarize("flow::leaf")
    assert not summary["complete"]
    assert any("storage lifetime requires capture proof" in reason for reason in summary["reasons"])
    assert summary["native_predicate_requirements"]


def test_native_predicate_metadata_survives_exact_reached_projection(tmp_path):
    analysis = source(tmp_path)
    routine = analysis.routines["flow::leaf"]
    assignment = next(node for node in walk(routine.execution) if type(node).__name__ == "Assignment_Stmt")
    summary = analysis.segment_summary("flow::leaf", (assignment,))
    assert summary["complete"], summary["reasons"]
    requirement, = summary["native_predicate_requirements"]
    assert requirement["procedure"] == "flow::leaf"
    assert requirement["source_expression"].lower() == "ieee_is_nan(p)"
    assert requirement["guard_frames"] == [{"procedure": "flow::leaf", "condition": "flag"}]
    assert not summary["cloneable"]


def test_nested_demanded_segment_keeps_all_outer_original_guards(tmp_path):
    original = source(tmp_path)
    path = original.inputs.paths[0]
    path.write_text(path.read_text().replace("logical,intent(in)::flag", "logical,intent(in)::flag\ninteger::i")
                    .replace("if(flag) result=", "if(flag) then\ndo i=1,2\nresult=")
                    .replace("result=ieee_is_nan(p)", "result=ieee_is_nan(p)\nenddo\nendif"))
    analysis = SourceEffects([path])
    assignment = next(node for node in walk(analysis.routines["flow::leaf"].execution)
                      if type(node).__name__ == "Assignment_Stmt")
    summary = analysis.segment_summary("flow::leaf", (assignment,))
    assert summary["complete"], summary["reasons"]
    frames = summary["native_predicate_requirements"][0]["guard_frames"]
    assert len(frames) == 2
    assert frames[0] == {"procedure": "flow::leaf", "condition": "flag"}
    assert frames[1]["condition"].lower() == "do i = 1, 2"
    reads = [effect for effect in summary["ordered_effects"] if effect["resource"] == "argument::p"]
    assert reads[0]["guard_frames"] == frames


def test_failed_private_projection_cannot_register_predicate_authority_on_original_analysis(tmp_path):
    from compiler.scopes.segments import _native_analysis

    original = source(tmp_path, declaration="real(8),pointer,intent(in)::p")
    native = _native_analysis(original)
    assignment = next(node for node in walk(native.routines["flow::leaf"].execution)
                      if type(node).__name__ == "Assignment_Stmt")
    summary = native.segment_summary("flow::leaf", (assignment,))
    assert not summary["complete"]
    assert native._native_predicates
    assert original._native_predicates == {}
    assert native._native_predicates is not original._native_predicates


def test_same_spelling_unproved_module_predicate_remains_unavailable(tmp_path):
    analysis = source(tmp_path, use="use,non_intrinsic::ieee_arithmetic,only:ieee_is_nan")
    summary = analysis.summarize("flow::leaf")
    assert not summary["complete"]
    assert summary["native_predicate_requirements"] == []
