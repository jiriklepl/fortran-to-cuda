"""Reduction source forms and effects keep scalar publication separate."""

from __future__ import annotations

import copy

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.reductions import analyze_reduction
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def fixture(tmp_path, body, *, dtype="integer", width=4, target=None, imports="", extra="", helpers=""):
    path = tmp_path / "renamed_reduce.f90"
    target = target or f"{dtype}({width})"
    path.write_text("module changed_names\n" + imports + "\nimplicit none\ncontains\n"
        "subroutine evaluate(a,mask,result,n,flag)\n" + f"{dtype}({width}),intent(in)::a(-3:,:)\n"
        + "logical,intent(in)::mask(:,:),flag\n" + target + ",intent(out)::result\ninteger,intent(in)::n\n"
        + extra + "\n" + body + "\nend subroutine\n" + helpers + "\nend module\n")
    analysis = SourceEffects([path])
    return path, analysis, walk(analysis.routines["changed_names::evaluate"].execution, F.Assignment_Stmt)[-1]


@pytest.mark.parametrize("name,identity", [("minval", 2147483647), ("maxval", -2147483647)])
def test_integer_intrinsic_reduction_has_exact_empty_identity_and_one_publication(tmp_path, name, identity):
    _path, analysis, original = fixture(tmp_path, f"result={name}(a)")
    proof = analyze_reduction(analysis, "changed_names::evaluate", original)
    assert proof.available, proof.reason
    record = proof.public()
    assert record["source_form"] == "intrinsic_array_reduction"
    assert record["operator"] == name
    assert record["empty_identity"] == {"type": "integer", "kind": 4, "value": identity}
    assert record["numerical_contract"]["reassociation"] == "exact"
    assert record["numerical_contract"]["requires_nonempty_tag"]
    assert record["source_analysis_available"]
    assert record["execution_supported"] is False
    assert record["canonical_reads"][0]["resource"] == "argument::a"
    assert record["canonical_reads"][0]["sections"]["logical_lower_bounds"] == [-3, 1]
    assert record["scalar_publication"]["resource"] == "argument::result"
    assert not record["scalar_publication"]["repeats_procedure_entry_definition_event"]
    assert record["speculative_effects"]["application_writes"] == []
    assert proof.validate(analysis) is proof


@pytest.mark.parametrize("name,identity", [("all", True), ("any", False)])
def test_logical_reduction_remains_exact_with_empty_sections(tmp_path, name, identity):
    _path, analysis, original = fixture(tmp_path, f"result={name}(a(-3:n,:))", dtype="logical")
    proof = analyze_reduction(analysis, "changed_names::evaluate", original)
    assert proof.available, proof.reason
    record = proof.public()
    assert record["empty_identity"] == {"type": "logical", "value": identity}
    assert record["value"]["type"] == "logical"
    assert record["numerical_contract"]["native_predicate_contract_required"] is False
    sections = record["canonical_reads"][0]["sections"]
    assert sections["reads"][0]["axes"][0]["upper"] == {"kind": "scalar", "resource": "argument::n"}


def test_scalar_dim_rectangular_mask_and_original_keyword_order(tmp_path):
    _path, analysis, original = fixture(tmp_path, "result=minval(mask=mask(1:n,1),array=a(-3:n-4,1),dim=1)")
    proof = analyze_reduction(analysis, "changed_names::evaluate", original)
    assert proof.available, proof.reason
    record = proof.public()
    assert [item["formal"] for item in record["original_argument_order"]] == ["mask", "array", "dim"]
    assert record["dim"] == {"source": "1", "value": 1}
    assert record["mask"]["resource"] == "argument::mask"
    assert record["value"]["logical_rank"] == 1
    assert {item["resource"] for item in record["canonical_reads"]} == {"argument::a", "argument::mask"}
    assert "runtime MASK extent conformity before speculative work" in record["requirements"]


def test_scalar_mask_is_read_at_original_guard_and_result_is_not_written_by_partials(tmp_path):
    _path, analysis, original = fixture(tmp_path, "if(flag) then\nresult=maxval(a,mask=flag)\nendif")
    proof = analyze_reduction(analysis, "changed_names::evaluate", original)
    assert proof.available, proof.reason
    record = proof.public()
    assert record["guard"] == ["flag"]
    assert record["mask"] == {"kind": "scalar", "source": "flag", "resource": "argument::flag"}
    assert record["scalar_reads"] == ["argument::flag"]
    assert record["scalar_publication"]["position"] == "one commit at the original assignment"
    assert record["speculative_effects"] == {"application_writes": [], "scratch_only": True}


def test_positional_mask_overload_uses_original_source_type(tmp_path):
    _path, analysis, original = fixture(tmp_path, "result=maxval(a,flag)")
    proof = analyze_reduction(analysis, "changed_names::evaluate", original)
    assert proof.available, proof.reason
    assert proof.public()["dim"] is None
    assert proof.public()["mask"]["resource"] == "argument::flag"
    assert [item["formal"] for item in proof.public()["original_argument_order"]] == ["array", "mask"]


@pytest.mark.parametrize("import_text,call", [
    ("use,intrinsic::ieee_arithmetic,only:ieee_is_nan", "ieee_is_nan(a)"),
    ("use,intrinsic::ieee_arithmetic,only:renamed=>ieee_is_nan", "renamed(a)"),
])
def test_nan_predicate_requires_proven_intrinsic_identity_and_native_policy_gate(tmp_path, import_text, call):
    _path, analysis, original = fixture(tmp_path, f"result=any({call})", dtype="real", width=8,
                                       target="logical", imports=import_text)
    proof = analyze_reduction(analysis, "changed_names::evaluate", original)
    assert proof.available, proof.reason
    record = proof.public()
    assert record["value"]["predicate"]["identity"] == "$intrinsic::ieee_arithmetic::ieee_is_nan"
    assert record["numerical_contract"]["native_predicate_contract_required"] is True
    assert any("observable exception policy" in requirement for requirement in record["requirements"])
    assert [item["resource"] for item in record["canonical_reads"]] == ["argument::a"]


@pytest.mark.parametrize("body,dtype,reason", [
    ("result=minval(a)", "real", "separate verified NaN"),
    ("result=maxval(a)", "real", "separate verified NaN"),
    ("result=sum(a)", "real", "reassociation contract"),
    ("result=sum(a)", "integer", "reassociation contract"),
    ("result=minval(a,dim=2)", "integer", "scalar result from a rank-one"),
    ("result=min(result,a(-3,1))", "integer", "not an admitted reduction intrinsic"),
    ("result=minval(a(n*n,:))", "integer", "footprint unavailable"),
])
def test_unproved_numerical_or_shape_contract_remains_native(tmp_path, body, dtype, reason):
    _path, analysis, original = fixture(tmp_path, body, dtype=dtype)
    proof = analyze_reduction(analysis, "changed_names::evaluate", original)
    assert not proof.available
    assert reason in proof.reason
    if body.startswith("result=sum"):
        assert proof.public()["numerical_contract"] == {"kind": "original_native_sum", "reassociation": "not authorized"}
    if body.startswith("result=min("):
        assert proof.public()["source_form"] == "scalar_extremum_assignment"


def test_application_function_named_like_reduction_never_establishes_contract(tmp_path):
    helper = """integer function minval(x)
integer,intent(in)::x(:,:)
minval=123
end function
"""
    _path, analysis, original = fixture(tmp_path, "result=minval(a)", helpers=helper)
    proof = analyze_reduction(analysis, "changed_names::evaluate", original)
    assert not proof.available
    assert "shadowed" in proof.reason


def test_reduction_cannot_publish_to_original_intent_in_scalar(tmp_path):
    path, _analysis, _original = fixture(tmp_path, "result=minval(a)")
    path.write_text(path.read_text().replace("integer(4),intent(out)::result", "integer(4),intent(in)::result"))
    analysis = SourceEffects([path])
    original = walk(analysis.routines["changed_names::evaluate"].execution, F.Assignment_Stmt)[0]
    proof = analyze_reduction(analysis, "changed_names::evaluate", original)
    assert not proof.available
    assert "publication association or type is unproved" in proof.reason


def test_ordinary_user_nan_function_is_not_intrinsic_classification(tmp_path):
    helper = """logical function ieee_is_nan(x)
real(8),intent(in)::x(:,:)
ieee_is_nan=.false.
end function
"""
    _path, analysis, original = fixture(tmp_path, "result=any(ieee_is_nan(a))", dtype="real", width=8,
                                       target="logical", helpers=helper)
    proof = analyze_reduction(analysis, "changed_names::evaluate", original)
    assert not proof.available
    assert proof.public()["value"] is None


def test_reduction_foreign_nodes_and_stale_source_proofs_reject(tmp_path):
    path, analysis, original = fixture(tmp_path, "result=minval(a)")
    for foreign in (copy.copy(original), F.Assignment_Stmt(str(original))):
        with pytest.raises(CompilationError, match="original source authority"):
            analyze_reduction(analysis, "changed_names::evaluate", foreign)
    proof = analyze_reduction(analysis, "changed_names::evaluate", original)
    with pytest.raises(CompilationError, match="original source authority"):
        copy.copy(proof).validate(analysis)
    via_id = analyze_reduction(analysis, "changed_names::evaluate", proof.node_id)
    assert via_id.identity == proof.identity
    detached = proof.public()
    detached["canonical_reads"].clear()
    assert proof.public()["canonical_reads"]
    path.write_text(path.read_text().replace("result=minval", "result=maxval"))
    with pytest.raises(CompilationError, match="changed"):
        proof.validate(analysis)


def test_reduction_candidates_are_lazy_bounded_local_source_records(tmp_path):
    _path, analysis, original = fixture(tmp_path,
        "result=minval(a)\nresult=maxval(a)\nresult=sum(a)\nresult=n")
    initial = analysis.report("changed_names::evaluate")
    assert initial["source_reductions"]["analysis"] == "not_requested"
    assert not initial["source_reductions"]["execution_supported"]
    selected = walk(analysis.routines["changed_names::evaluate"].execution, F.Assignment_Stmt)[1]
    reached = analysis.reduction_candidates("changed_names::evaluate", (selected,))
    assert reached["candidate_count"] == 1
    assert reached["source_analysis_available"]
    assert not reached["execution_supported"]
    assert reached["records"][0]["operator"] == "maxval"
    bounded = analysis.reduction_candidates("changed_names::evaluate", limit=2)
    assert bounded["candidate_count"] == 3
    assert bounded["truncated"] and len(bounded["records"]) == 2
    detached = analysis.reduction_candidates("changed_names::evaluate", (selected,))
    detached["records"].clear()
    assert analysis.reduction_candidates("changed_names::evaluate", (selected,))["records"]
    report = analysis.report("changed_names::evaluate")
    assert report["source_reductions"]["analysis"] == "requested"
    assert len(report["source_reductions"]["requests"]) == 2


def test_reduction_candidate_requests_reject_fabricated_source_and_do_not_expand_calls(tmp_path):
    helper = """subroutine child(a,value)
integer,intent(in)::a(:,:)
integer,intent(out)::value
value=minval(a)
end subroutine
"""
    _path, analysis, original = fixture(tmp_path, "call child(a,result)\nresult=n", helpers=helper)
    assert analysis.reduction_candidates("changed_names::evaluate")["candidate_count"] == 0
    with pytest.raises(CompilationError, match="original source authority"):
        analysis.reduction_candidates("changed_names::evaluate", (copy.copy(original),))
