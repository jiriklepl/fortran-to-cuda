"""Conditional native faces are source transfers, never whole-init facts."""

import json
from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend.native_definitions import prove_native_definitions
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "renamed_edges::touch"


def fixture(tmp_path, body, *, declaration="real(8),intent(inout)::a(-2:,-1:),b(-2:,-1:)", operations=256):
    path = tmp_path / "edges.f90"
    path.write_text("module renamed_edges\nimplicit none\ncontains\nsubroutine touch(a,b,n,flag,other)\n"
                    + declaration + "\ninteger,intent(inout)::n\nlogical,intent(in)::flag,other\n"
                    "integer::i,j\n" + body + "\nend subroutine\nend module\n")
    analysis = SourceEffects([path], operations=operations)
    selection = tuple(analysis.routines[ENTRY].execution.content)
    return path, analysis, selection


def prove(tmp_path, body, **options):
    path, analysis, selection = fixture(tmp_path, body, **options)
    return path, analysis, selection, prove_native_definitions(analysis, ENTRY, selection)


def resource(step, root="argument::a"):
    return next(item for item in step.sections.resources if item.resource == root)


def test_ordered_reads_precede_each_assignment_definition(tmp_path):
    _path, _analysis, _selection, proof = prove(tmp_path,
        "a(-2,-1)=a(-2,-1)+1\nb(-2,-1)=a(-2,-1)")
    result = proof.transfer()
    assert len(result.uncovered_reads) == 1
    read, = result.uncovered_reads
    assert read.node_id == proof.steps[0].node_id
    assert read.resource == "argument::a"
    assert len(result.definitions["argument::a"]) == 1
    defined = resource(proof.steps[0]).overwrites
    assert proof.transfer({"argument::a": defined}).reads_covered


def test_prior_partial_assignment_covers_only_its_original_face(tmp_path):
    path, analysis, selection, proof = prove(tmp_path,
        "a(-2,:)=1\na(18,:)=2\nb(0,:)=a(0,:)")
    before = path.read_bytes()
    result = proof.transfer()
    assert len(result.definitions["argument::a"]) == 2
    assert [box.axes[0].lower.value for box in result.definitions["argument::a"]] == [-2, 18]
    assert len(result.uncovered_reads) == 1
    assert result.uncovered_reads[0].rectangle.axes[0].lower.value == 0
    assert path.read_bytes() == before
    assert prove_native_definitions(analysis, ENTRY, selection) is proof
    public = proof.public()
    assert public["global_initialization_authority"] is False
    assert public["whole_definition_inferred"] is False
    assert public["bounds_evaluated"] is False
    assert public["execution_authorized"] is False
    assert json.loads(json.dumps(public)) == public


@pytest.mark.parametrize("otherwise", ["else\na(-2,:)=2", "else if(other)then\na(-2,:)=2\nelse\na(-2,:)=3"])
def test_identical_face_defined_in_every_if_arm_survives_join(tmp_path, otherwise):
    _path, _analysis, _selection, proof = prove(tmp_path,
        "if(flag)then\na(-2,:)=1\n" + otherwise + "\nendif\nb(-2,:)=a(-2,:)")
    result = proof.transfer()
    assert result.reads_covered
    assert len(result.definitions["argument::a"]) == 1
    branch = proof.steps[0]
    assert branch.kind == "branch"
    assert branch.alternatives[0][1].children[0].guard == ("flag",)
    assert branch.alternatives[1][1].children[0].guard[0] == ".not.(flag)"


@pytest.mark.parametrize("otherwise", ["", "else\na(18,:)=2", "else if(other)then\na(-2,:)=2"])
def test_conditional_may_write_union_does_not_define_missing_face(tmp_path, otherwise):
    _path, _analysis, _selection, proof = prove(tmp_path,
        "if(flag)then\na(-2,:)=1\n" + otherwise + "\nendif\nb(-2,:)=a(-2,:)")
    result = proof.transfer()
    assert result.definitions["argument::a"] == ()
    assert len(result.uncovered_reads) == 1
    reached_arm = proof.steps[0].alternatives[0][1].children[0]
    assert resource(reached_arm).writes == resource(reached_arm).overwrites


def test_single_line_if_preserves_statement_authority_and_implicit_empty_arm(tmp_path):
    _path, _analysis, _selection, proof = prove(tmp_path,
        "if(flag)a(-2,:)=1\nb(-2,:)=a(-2,:)")
    assert proof.steps[0].kind == "branch"
    assert proof.transfer().definitions["argument::a"] == ()
    assert len(proof.transfer().uncovered_reads) == 1


def test_elseif_condition_reads_keep_original_evaluation_guard(tmp_path):
    _path, _analysis, _selection, proof = prove(tmp_path,
        "if(flag)then\nb(-2,:)=1\nelse if(a(-2,-1)>0)then\nb(18,:)=2\nendif")
    read, = proof.transfer().uncovered_reads
    assert read.resource == "argument::a"
    assert read.guard == (".not.(flag)",)
    assert read.node_id == proof.steps[0].alternatives[1][0].node_id
    assert read.public()["requirement"].endswith("no early bound evaluation")


def test_matching_runtime_affine_loops_preserve_typed_bound_trees(tmp_path):
    _path, _analysis, _selection, proof = prove(tmp_path,
        "do j=0,n\ndo i=1,n\na(i,j)=1\nenddo\nenddo\n"
        "do j=0,n\ndo i=1,n\nb(i,j)=a(i,j)\nenddo\nenddo")
    assert proof.transfer().reads_covered
    box, = proof.transfer().definitions["argument::a"]
    assert len(box.iterations) == 2
    assert box.axes[0].upper.public() == {"kind": "scalar", "resource": "argument::n"}
    assert box.node is resource(proof.steps[0]).overwrites[0].node


@pytest.mark.parametrize("bound", ["n", "0"])
def test_emptyable_repeated_point_write_does_not_cover_unconditional_read(tmp_path, bound):
    _path, _analysis, _selection, proof = prove(tmp_path,
        f"do i=1,{bound}\na(0,-1)=1\nenddo\nb(0,-1)=a(0,-1)")
    assert len(proof.transfer().uncovered_reads) == 1
    assert resource(proof.steps[0]).overwrites[0].iterations


def test_mutable_scalar_index_invalidates_prior_symbolic_coverage(tmp_path):
    _path, _analysis, _selection, proof = prove(tmp_path,
        "a(n,-1)=1\nn=n+1\nb(n,-1)=a(n,-1)")
    result = proof.transfer()
    assert result.definitions["argument::a"] == ()
    assert len(result.uncovered_reads) == 1
    assert proof.steps[1].scalar_writes == ("argument::n",)


def test_scalar_mutation_on_one_branch_invalidates_common_symbolic_definition(tmp_path):
    _path, _analysis, _selection, proof = prove(tmp_path,
        "a(n,-1)=1\nif(flag)then\nn=n+1\nendif\nb(n,-1)=a(n,-1)")
    assert len(proof.transfer().uncovered_reads) == 1


def test_full_native_assignment_covers_later_partial_reads_without_whole_fact(tmp_path):
    _path, _analysis, _selection, proof = prove(tmp_path, "a=1\nb(-2:1,:)=a(-2:1,:)")
    assert proof.transfer().reads_covered
    assert proof.public()["whole_definition_inferred"] is False
    assert all(axis.lower is None for axis in proof.transfer().definitions["argument::a"][0].axes)


def test_literal_subset_coverage_is_not_a_multi_face_bounding_hull(tmp_path):
    _path, _analysis, _selection, proof = prove(tmp_path, "a(-2:2,-1)=1\nb(0,-1)=a(0,-1)")
    assert proof.transfer().reads_covered


def test_entry_out_kill_is_explicit_and_only_allowed_for_whole_original_body(tmp_path):
    _path, analysis, selection, proof = prove(tmp_path, "b(-2,-1)=a(-2,-1)",
        declaration="real(8),intent(out)::a(-2:,-1:)\nreal(8),intent(inout)::b(-2:,-1:)")
    entry = {"argument::a": resource(proof.steps[0]).reads}
    assert proof.transfer(entry).reads_covered
    full = prove_native_definitions(analysis, ENTRY, selection, include_entry=True)
    assert full.public()["entry_definition_changes"] == ["argument::a"]
    assert not full.transfer(entry).reads_covered


@pytest.mark.parametrize("body", [
    "call opaque(a)", "allocate(a(3,3))", "return", "a(:,1)=a(:,1)+unknown(n)",
    "do i=1,n\na(i,-1)=1\nb(i,-1)=a(i,-1)\nenddo",
    "do i=1,n\nif(flag)a(i,-1)=1\nenddo",
    "do while(flag)\na(-2,-1)=1\nenddo",
    "!$omp parallel do\ndo i=1,n\na(i,-1)=1\nenddo\n!$omp end parallel do",
    "do i=i,n\na(i,-1)=1\nenddo",
])
def test_unknown_or_unproved_control_never_produces_partial_definition_authority(tmp_path, body):
    _path, analysis, selection = fixture(tmp_path, body)
    with pytest.raises(CompilationError):
        prove_native_definitions(analysis, ENTRY, selection)
    assert not getattr(analysis, "_native_definition_proofs", {})


def test_noncontiguous_selection_cannot_skip_scalar_mutation(tmp_path):
    _path, analysis, selection = fixture(tmp_path, "a(n,-1)=1\nn=n+1\nb(n,-1)=a(n,-1)")
    with pytest.raises(CompilationError, match="contiguous"):
        prove_native_definitions(analysis, ENTRY, (selection[0], selection[-1]))
    with pytest.raises(CompilationError, match="complete original body"):
        prove_native_definitions(analysis, ENTRY, (selection[0],), include_entry=True)


def test_registered_source_identity_rejects_copied_foreign_tokens_and_changed_sources(tmp_path):
    path, analysis, selection, proof = prove(tmp_path, "a(-2,:)=1")
    with pytest.raises(CompilationError, match="registered exact"):
        replace(proof).validate(analysis, ENTRY, selection)
    with pytest.raises(CompilationError, match="registered exact"):
        proof.validate(copy(analysis), ENTRY, selection)
    with pytest.raises(CompilationError, match="registered exact"):
        proof.validate(analysis, ENTRY, tuple(copy(item) for item in selection))
    path.write_text(path.read_text().replace("=1", "=2"))
    with pytest.raises(CompilationError, match="changed"):
        proof.transfer()


def test_original_binding_mutation_rejects_transfer(tmp_path):
    _path, analysis, _selection, proof = prove(tmp_path, "a(-2,:)=1")
    analysis.routines[ENTRY].scope.bindings["a"].attributes |= {"pointer"}
    with pytest.raises(CompilationError):
        proof.transfer()


def test_public_metadata_and_entry_containers_cannot_mutate_source_token(tmp_path):
    _path, _analysis, _selection, proof = prove(tmp_path, "a(-2,:)=1")
    public = proof.public()
    public["steps"].clear()
    assert proof.public()["steps"]
    result = proof.transfer()
    with pytest.raises(TypeError):
        result.definitions["argument::a"] = ()
    with pytest.raises(CompilationError, match="original-resource"):
        proof.transfer({"foreign::a": ()})


def test_bounded_rectangle_union_rejects_without_truncated_definitions(tmp_path):
    _path, _analysis, _selection, proof = prove(tmp_path, "\n".join(f"a({i},:)=1" for i in range(33)))
    with pytest.raises(CompilationError, match="rectangle budget"):
        proof.transfer()


def test_source_operation_budget_does_not_admit_truncated_partial_proof(tmp_path):
    _path, analysis, selection = fixture(tmp_path, "a(-2,:)=1\na(-1,:)=2\na(0,:)=3", operations=2)
    with pytest.raises(CompilationError):
        prove_native_definitions(analysis, ENTRY, selection)
