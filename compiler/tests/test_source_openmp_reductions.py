"""Only original joined clauses can authorize parallel sum combination order."""

from dataclasses import replace

import pytest

from compiler.frontend.omp_reductions import analyze_openmp_reduction
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

PROCEDURE = "renamed_parallel_state::run"


def fixture(tmp_path, body=None, *, dtype="real", width=8, opening=None,
            ending="end parallel do", array_attributes="", module_storage=""):
    path = tmp_path / "parallel_sum.f90"
    body = body or "do i=-2,n\ntmp=a(i,3)*a(i,3)\ntotal=total+tmp\nenddo"
    opening = opening or "parallel do private(i,tmp) reduction(+:total)"
    path.write_text(f"""module renamed_parallel_state
implicit none
{module_storage}
contains
subroutine run(a,out,n,m)
{dtype}({width}),intent(in){array_attributes}::a(-2:,3:)
{dtype}({width}),intent(out)::out
integer,intent(in)::n,m
integer::i,j
{dtype}({width})::total,tmp
total=5
continue
!$omp {opening}
{body}
!$omp {ending}
out=total
end subroutine
end module
""")
    analysis = SourceEffects([path])
    nodes = tuple(node for node in analysis.routines[PROCEDURE].execution.content
                  if type(node).__name__ not in {"Assignment_Stmt", "Continue_Stmt"})
    return path, analysis, nodes


def test_unrelated_module_thread_storage_does_not_poison_local_reduction(tmp_path):
    storage = "real(8)::hidden\n!$omp threadprivate(hidden)"
    _path, analysis, nodes = fixture(tmp_path, module_storage=storage)
    proof = analysis.openmp_reduction(PROCEDURE, nodes)
    assert proof.available, proof.reason


def test_accessed_module_thread_storage_retains_an_explicit_boundary(tmp_path):
    storage = "real(8)::hidden\n!$omp threadprivate(hidden)"
    body = "do i=-2,n\ntmp=a(i,3)*hidden\ntotal=total+tmp\nenddo"
    _path, analysis, nodes = fixture(tmp_path, body, module_storage=storage)
    proof = analysis.openmp_reduction(PROCEDURE, nodes)
    assert not proof.available
    assert "module thread ownership" in proof.reason


@pytest.mark.parametrize("width", [4, 8])
def test_original_parallel_sum_retains_initial_value_private_state_and_completion(tmp_path, width):
    _path, analysis, nodes = fixture(tmp_path, width=width)
    proof = analysis.openmp_reduction(PROCEDURE, nodes)
    assert proof.available, proof.reason
    record = proof.public()
    assert record["source_form"] == "original_openmp_clause_reduction"
    assert record["source_analysis_available"] and not record["execution_supported"]
    assert record["initial_value"]["resource"] == PROCEDURE + "::total"
    assert record["initial_value"]["requires_original_definition"]
    assert record["private_initializer"] == {"value": 0, "type": "real", "kind": width}
    assert record["contribution"]["value"] == "tmp"
    assert record["canonical_reads"] == ["argument::a"]
    assert set(record["scalar_reads"]) == {"argument::n", PROCEDURE + "::total"}
    assert set(record["private_resources"]) == {PROCEDURE + "::" + item for item in ("i", "tmp")}
    assert not record["reduction_private_copies"]["original_application_writes_before_commit"]
    assert record["native_completion"]["retains_original_team_and_directives"]
    assert not record["native_completion"]["ordinary_completion_token"]
    assert record["scalar_publication"]["combines_original_value_once"]
    assert record["numerical_contract"]["serial_sum_permission"] is False
    assert record["numerical_contract"]["field_tolerances"] == "unchanged"
    assert record["ordered_effects"]
    total_effects = [item for item in record["ordered_effects"] if item.get("resource") == PROCEDURE + "::total"]
    assert [item["kind"] for item in total_effects] == ["read", "write"]
    assert total_effects[-1]["position"] == "original proven region completion"
    assert proof.validate(analysis) is proof
    with pytest.raises(CompilationError, match="unsupported joined native OpenMP clause"):
        analysis.joined_completion(PROCEDURE, nodes)


def test_separate_parallel_and_worksharing_nowait_keep_the_outer_join(tmp_path):
    body = "!$omp do reduction(+:total)\ndo i=-2,n\ntotal=total+a(i,3)\nenddo\n!$omp end do nowait"
    _path, analysis, nodes = fixture(tmp_path, body, opening="parallel private(i)", ending="end parallel")
    proof = analyze_openmp_reduction(analysis, PROCEDURE, nodes)
    assert proof.available, proof.reason
    assert proof.public()["scalar_publication"]["position"] == "original proven region completion"


def test_rectangular_collapsed_loop_preserves_original_negative_coordinates(tmp_path):
    body = "do i=-2,n\ndo j=3,m\ntotal=total+a(i,j)\nenddo\nenddo"
    _path, analysis, nodes = fixture(tmp_path, body,
        opening="parallel do private(i,j) shared(a,n,m) default(none) collapse(2) reduction(+:total)")
    proof = analyze_openmp_reduction(analysis, PROCEDURE, nodes)
    assert proof.available, proof.reason
    assert [(item["lower"], item["upper"]) for item in proof.public()["loop_domains"]] == [("- 2", "n"), ("3", "m")]
    assert proof.public()["contribution"]["value"] == "a(i, j)"


def test_source_descriptor_bounds_are_metadata_and_remain_at_original_loop(tmp_path):
    body = "do i=lbound(a,1),ubound(a,1)\ntotal=total+a(i,3)\nenddo"
    _path, analysis, nodes = fixture(tmp_path, body, opening="parallel do private(i) reduction(+:total)")
    proof = analyze_openmp_reduction(analysis, PROCEDURE, nodes)
    assert proof.available, proof.reason
    assert proof.public()["descriptor_reads"] == ["argument::a"]
    assert proof.public()["scalar_reads"] == [PROCEDURE + "::total"]


@pytest.mark.parametrize("body,options,reason", [
    ("do i=-2,n\ntotal=total+a(i,3)\nenddo", {"dtype": "integer", "width": 4}, "every partial sum is representable"),
    ("do i=-2,n\ntmp=total\ntotal=total+a(i,3)\nenddo", {}, "observed outside"),
    ("do i=-2,n\ntotal=total+total\nenddo", {}, "exactly once"),
    ("do i=-2,n\ntotal=total+tmp\nenddo", {}, "before its per-item definition"),
    ("do i=-2,n\ncall opaque(a)\ntotal=total+a(i,3)\nenddo", {}, "without calls, exits or array writes"),
    ("do i=-2,n\na(i,3)=0\ntotal=total+a(i,3)\nenddo", {}, "without calls, exits or array writes"),
    ("do i=-2,n\ncycle\ntotal=total+a(i,3)\nenddo", {}, "without calls, exits or array writes"),
    ("do i=-2,n\ntotal=total+unknown(a(i,3))\nenddo", {}, "unsupported call"),
    ("do i=-2,n\ndo j=3,i\ntotal=total+a(i,j)\nenddo\nenddo", {}, "independent rectangular"),
    ("do i=-2,n\ntotal=total+a(i,3)\nenddo", {"array_attributes": ",target"}, "association or identity"),
    ("do i=-2,n\ntotal=total+a(i,3)\nenddo", {"ending": "end parallel"}, "complete original joined region"),
    ("do i=-2,n\ntotal=total+a(i,3)\nenddo", {"opening": "parallel do private(i) reduction(inscan,+:total)"}, "one builtin scalar"),
])
def test_unproved_scalar_effect_alias_lifetime_or_completion_remains_native(tmp_path, body, options, reason):
    _path, analysis, nodes = fixture(tmp_path, body, **options)
    proof = analyze_openmp_reduction(analysis, PROCEDURE, nodes)
    assert not proof.available
    assert reason in proof.reason
    assert not proof.public()["execution_supported"]


def test_module_threadprivate_inputs_have_no_single_shared_gpu_state(tmp_path):
    _path, analysis, nodes = fixture(tmp_path, "do i=-2,n\ntotal=total+g(i)\nenddo",
        module_storage="real(8)::g(-2:100)\n!$omp threadprivate(g)")
    proof = analyze_openmp_reduction(analysis, PROCEDURE, nodes)
    assert not proof.available
    assert "thread ownership is unproved" in proof.reason


def test_copied_or_partial_group_and_stale_source_cannot_supply_reduction_authority(tmp_path):
    path, analysis, nodes = fixture(tmp_path)
    proof = analyze_openmp_reduction(analysis, PROCEDURE, nodes)
    assert proof.available, proof.reason
    with pytest.raises(CompilationError, match="complete-group authority"):
        replace(proof).validate(analysis)
    loop = next(node for node in nodes if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    partial = analyze_openmp_reduction(analysis, PROCEDURE, (loop,))
    assert not partial.available
    path.write_text(path.read_text().replace("reduction(+:total)", "reduction(*:total)"))
    with pytest.raises(CompilationError, match="changed"):
        proof.validate(analysis)


def test_public_source_report_contains_only_explicitly_requested_original_reductions(tmp_path):
    _path, analysis, nodes = fixture(tmp_path)
    report = analysis.report(PROCEDURE)
    assert report["openmp_reductions"]["analysis"] == "not_requested"
    assert report["openmp_reductions"]["records"] == []
    proof = analysis.openmp_reduction(PROCEDURE, nodes)
    assert proof.available, proof.reason
    requested = analysis.report(PROCEDURE)["openmp_reductions"]
    assert requested["analysis"] == "requested"
    assert requested["source_analysis_available"] and not requested["execution_supported"]
    assert requested["records"] == [proof.public()]
    requested["records"][0]["ordered_effects"].clear()
    assert analysis.report(PROCEDURE)["openmp_reductions"]["records"][0]["ordered_effects"]
