"""Conditional native faces are source transfers, never whole-init facts."""

import json
from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.native_definitions import (
    intersect_rectangles,
    invalidate_rectangles,
    merge_rectangles,
    prove_native_definitions,
    rectangle_covers,
    rectangle_dependencies,
)
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


def module_fixture(tmp_path, body, *, declaration=""):
    path = tmp_path / "module_edges.f90"
    path.write_text("module original_storage\nimplicit none\n"
                    "real(8),allocatable::cells(:,:),other(:,:)\nend module\n"
                    "module renamed_edges\nuse original_storage,only:a=>cells,b=>other\n"
                    "implicit none\ncontains\nsubroutine touch(n,flag)\n"
                    "integer,intent(inout)::n\nlogical,intent(in)::flag\ninteger::i,j\n"
                    + declaration + "\n" + body + "\nend subroutine\nend module\n")
    analysis = SourceEffects([path], summary_cache=tmp_path / "cache")
    selection = tuple(analysis.routines[ENTRY].execution.content)
    graph = analysis.structure(ENTRY)
    ids = tuple(graph.node_id(node) for node in selection)
    descriptor = prove_descriptor_window(analysis, ENTRY, ids,
        module_roots=("original_storage::cells", "original_storage::other"))
    return path, analysis, selection, descriptor


def test_module_partial_definitions_use_exact_descriptor_without_initialization(tmp_path):
    path, analysis, selection, descriptor = module_fixture(tmp_path,
        "a(-2:2,-1)=1\nb(0,-1)=a(0,-1)")
    original = path.read_bytes()
    before = analysis.summarize(ENTRY)
    assert not before["complete"]
    with pytest.raises(CompilationError, match="complete original"):
        prove_native_definitions(analysis, ENTRY, selection)
    proof = prove_native_definitions(analysis, ENTRY, selection, descriptor_window=descriptor)
    result = proof.transfer()
    assert result.reads_covered
    assert set(result.definitions) == {"original_storage::cells", "original_storage::other"}
    assert len(result.definitions["original_storage::cells"]) == 1
    assert proof.public()["schema_version"] == 3
    assert proof.public()["descriptor_window_identity"] == descriptor.identity
    assert len(proof.public()["leaf_descriptor_windows"]) == 2
    assert not proof.public()["global_initialization_authority"]
    assert not descriptor.public()["entry_definition_authority"]
    assert analysis.summarize(ENTRY) == before
    assert analysis.stable_module_allocatables == frozenset()
    assert not analysis.window_module_allocatables(ENTRY)
    assert analysis._allocation_authorizations == {}
    assert path.read_bytes() == original


def test_module_descriptor_only_does_not_cover_a_prior_read(tmp_path):
    _path, analysis, selection, descriptor = module_fixture(tmp_path, "b(0,-1)=a(0,-1)")
    proof = prove_native_definitions(analysis, ENTRY, selection, descriptor_window=descriptor)
    read, = proof.transfer().uncovered_reads
    assert read.resource == "original_storage::cells"
    assert read.rectangle.axes[0].lower.value == 0


def test_module_branch_leaves_get_exact_source_descriptor_tokens(tmp_path):
    _path, analysis, selection, descriptor = module_fixture(tmp_path,
        "if(flag)then\na(-2,:)=1\nelse\na(-2,:)=2\nendif\nb(-2,:)=a(-2,:)")
    proof = prove_native_definitions(analysis, ENTRY, selection, descriptor_window=descriptor)
    assert proof.transfer().reads_covered
    demands = proof.public()["leaf_descriptor_windows"]
    assert len(demands) == 4  # Original condition, two writes, then read/write.
    assert all(item["proof_identity"] != descriptor.identity for item in demands)
    for item in demands:
        token = analysis._descriptor_window_proofs[item["proof_identity"]]
        assert token.public()["selected_node_ids"] == [item["node_id"]]


def test_module_runtime_bounds_remain_original_descriptor_trees(tmp_path):
    _path, analysis, selection, descriptor = module_fixture(tmp_path,
        "do j=lbound(a,2),ubound(a,2)\ndo i=lbound(a,1),ubound(a,1)\na(i,j)=1\nenddo\nenddo")
    proof = prove_native_definitions(analysis, ENTRY, selection, descriptor_window=descriptor)
    sections = resource(proof.steps[0], "original_storage::cells")
    assert sections.descriptor_lower_bounds
    assert sections.descriptor_origin == "allocation"
    box, = proof.transfer().definitions["original_storage::cells"]
    assert box.axes[0].lower.kind == "lbound"
    assert box.axes[0].upper.kind == "ubound"
    assert box.axes[0].lower.resource == "original_storage::cells"
    assert proof.public()["bounds_evaluated"] is False


def test_formal_descriptor_window_keeps_partial_payload_and_original_coordinates(tmp_path):
    _path, analysis, selection = fixture(tmp_path, "a(-2:2,-1)=1\nb(0,-1)=a(0,-1)",
        declaration="real(8),allocatable,intent(inout)::a(:,:),b(:,:)")
    graph = analysis.structure(ENTRY)
    ids = tuple(graph.node_id(node) for node in selection)
    descriptor = prove_descriptor_window(analysis, ENTRY, ids, ("argument::a", "argument::b"))
    proof = prove_native_definitions(analysis, ENTRY, selection, descriptor_window=descriptor)
    assert proof.transfer().reads_covered
    assert proof.public()["descriptor_window_identity"] == descriptor.identity
    assert resource(proof.steps[0]).descriptor_origin == "allocation"
    assert resource(proof.steps[0]).overwrites[0].axes[0].lower.value == -2


def test_descriptor_tokens_cannot_alias_neighbors_foreign_or_copied_selections(tmp_path):
    _path, analysis, selection, descriptor = module_fixture(tmp_path, "a(-2,:)=1\nb(-2,:)=1")
    for token in (replace(descriptor), object(), descriptor.public()):
        with pytest.raises(CompilationError):
            prove_native_definitions(analysis, ENTRY, selection, descriptor_window=token)
    with pytest.raises(CompilationError, match="registered original interval"):
        prove_native_definitions(analysis, ENTRY, selection[:1], descriptor_window=descriptor)
    other = SourceEffects(analysis.inputs.paths)
    with pytest.raises(CompilationError, match="registered original interval"):
        prove_native_definitions(other, ENTRY, tuple(other.routines[ENTRY].execution.content),
                                 descriptor_window=descriptor)
    assert not getattr(analysis, "_native_definition_proofs", {})


def test_descriptor_source_and_binding_mutations_reject_partial_transfer(tmp_path):
    path, analysis, selection, descriptor = module_fixture(tmp_path, "a(-2,:)=1")
    proof = prove_native_definitions(analysis, ENTRY, selection, descriptor_window=descriptor)
    analysis.modules["original_storage"].bindings["cells"].attributes |= {"target"}
    with pytest.raises(CompilationError):
        proof.transfer()
    analysis.modules["original_storage"].bindings["cells"].attributes -= {"target"}
    path.write_text(path.read_text().replace("a(-2,:)=1", "a(-2,:)=2"))
    with pytest.raises(CompilationError, match="changed"):
        proof.transfer()


def test_geometry_helpers_meet_and_merge_without_face_hulls(tmp_path):
    _path, _analysis, _selection, proof = prove(tmp_path,
        "a(-2:2,-1)=1\na(8:10,-1)=2\nb(0,-1)=3")
    left = resource(proof.steps[0]).overwrites
    opposite = resource(proof.steps[1]).overwrites
    point = resource(proof.steps[2], "argument::b").overwrites
    assert rectangle_covers(left[0], point[0])
    assert not rectangle_covers(opposite[0], point[0])
    assert merge_rectangles(left, left, opposite) == (*left, *opposite)
    assert intersect_rectangles(left, point) == point
    assert intersect_rectangles(left, opposite) == ()
    assert intersect_rectangles(left, ()) == ()
    assert rectangle_dependencies(left[0]) == ()


def test_scalar_write_invalidates_rectangles_for_root_not_accessed_by_leaf(tmp_path):
    _path, analysis, selection, proof = prove(tmp_path, "a(n,-1)=1\nn=n+1")
    prior = resource(proof.steps[0]).overwrites
    scalar = prove_native_definitions(analysis, ENTRY, selection[1:])
    assert scalar.transfer().definitions == {}
    assert rectangle_dependencies(prior[0]) == ("argument::n",)
    assert invalidate_rectangles(prior, scalar.steps[0].scalar_writes) == ()
    assert invalidate_rectangles(prior, ("argument::unrelated",)) == prior


def test_geometry_helpers_reject_unbounded_or_malformed_groups(tmp_path):
    _path, _analysis, _selection, proof = prove(tmp_path, "a(-2,:)=1")
    boxes = resource(proof.steps[0]).overwrites
    with pytest.raises(CompilationError, match="bounded"):
        merge_rectangles(boxes * 33)
    with pytest.raises(CompilationError, match="group budget"):
        merge_rectangles(*(boxes for _ in range(33)))
    with pytest.raises(CompilationError, match="rectangle tuples"):
        intersect_rectangles(list(boxes), boxes)
    with pytest.raises(CompilationError, match="invalidation"):
        invalidate_rectangles(boxes, ["argument::n"])
    with pytest.raises(CompilationError, match="rectangle tuples"):
        rectangle_covers(object(), boxes[0])
