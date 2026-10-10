"""Module allocation evidence belongs to one original interval, never a call."""

import json
from copy import copy
from dataclasses import replace

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.native_intrinsics import prove_native_predicate
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "driver::advance"
ROOT = "storage::backing"


def fixture(tmp_path, *, selected="call coordinate(n)", prefix="call unknown_prefix()",
            tail="call retire()", attributes="allocatable", module_extra="", worker=None):
    storage = tmp_path / "storage.f90"
    storage.write_text(f"""module storage
implicit none
real(8),{attributes}::backing(:)
{module_extra}
contains
subroutine retire()
if(allocated(backing)) deallocate(backing)
end subroutine
end module
""")
    helper = tmp_path / "helper.f90"
    worker = worker or """do i=1,n
local(lbound(local,1)+i-1)=local(lbound(local,1)+i-1)+1.0_8
end do"""
    helper.write_text(f"""module helper
use storage,only:local=>backing
contains
subroutine work(n)
integer,intent(in)::n
integer::i
{worker}
end subroutine
subroutine forwarded(a,n)
real(8),allocatable,intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(lbound(a,1)+i-1)=a(lbound(a,1)+i-1)+2.0_8
end do
end subroutine
subroutine coordinate(n)
integer,intent(in)::n
call work(n)
call forwarded(local,n)
end subroutine
end module
""")
    driver = tmp_path / "driver.f90"
    driver.write_text(f"""module driver
use storage,only:slot=>backing,retire
use helper,only:coordinate
contains
subroutine advance(n)
integer,intent(in)::n
integer::i
{prefix}
{selected}
{tail}
end subroutine
subroutine unrelated()
slot(lbound(slot,1))=3.0_8
end subroutine
end module
""")
    analysis = SourceEffects([storage, helper, driver], summary_cache=tmp_path / "cache")
    graph = analysis.structure(ENTRY)
    node = next(node for node in analysis.routines[ENTRY].execution.content
                if "".join(str(node).lower().split()).replace("()", "").startswith(
                    "".join(selected.splitlines()[0].lower().split()).replace("()", "")))
    selected_ids = (graph.node_id(node),)
    return analysis, graph, selected_ids, (storage, helper, driver)


def prove(analysis, selected):
    return prove_descriptor_window(analysis, ENTRY, selected, module_roots=(ROOT,))


def test_module_window_is_explicit_source_local_and_definition_free(tmp_path):
    analysis, graph, selected, paths = fixture(tmp_path)
    originals = {path: path.read_bytes() for path in paths}
    assert not analysis.summarize(ENTRY)["complete"]
    with pytest.raises(CompilationError, match="formals"):
        prove_descriptor_window(analysis, ENTRY, selected, (ROOT,))
    proof = prove(analysis, selected)
    assert proof is prove(analysis, selected)
    public = proof.public()
    state = proof.resource_state(ROOT)
    assert public["schema_version"] == 4
    assert proof.module_roots == (ROOT,)
    assert state["resource_kind"] == "original_module_allocatable"
    assert state["original_visible_spelling"] == "slot"
    assert state["original_binding"] == json.loads(json.dumps(analysis.modules["storage"].bindings["backing"].public()))
    assert state["requires_original_runtime_bounds"]
    assert state["requires_original_allocation_presence_and_alias_guards"]
    assert state["execution_requires_runtime_guard"]
    assert not state["entry_definition_authority"]
    assert not public["prefix_effects_proved"]
    assert not public["execution_authorized"]
    assert not public["gpu_legality_established"]
    assert public["requires_close_before_excluded_tail"]
    assert graph.identity == public["structured_identity"]
    assert analysis.stable_module_allocatables == frozenset()
    assert analysis._allocation_authorizations == {}
    assert {path: path.read_bytes() for path in paths} == originals


def test_private_projection_admits_only_checked_module_children_and_isolates_caches(tmp_path):
    analysis, _, selected, _ = fixture(tmp_path)
    before = analysis.summarize("helper::work")
    assert not before["complete"]
    proof = prove(analysis, selected)
    scoped = analysis.segment_summary(ENTRY, selected, descriptor_window=proof)
    assert scoped["complete"], scoped["reasons"]
    assert scoped["descriptor_window_identity"] == proof.identity
    assert any(item["resource"] == ROOT and item["authorized"]
               for item in scoped["capture_lifetime_requirements"])
    private, _, _ = analysis._segment_projection(ENTRY, selected, descriptor_window=proof)
    assert private.window_module_allocatables(ENTRY) == {ROOT}
    assert private.window_module_allocatables("helper::work") == {ROOT}
    assert private.window_module_allocatables("helper::forwarded") == {ROOT}
    assert private.window_module_allocatables("driver::unrelated") == frozenset()
    assert private.summarize("helper::work")["complete"]
    assert not private.summarize("driver::unrelated")["complete"]
    child_graph = private.structure("helper::work")
    child_loop = next(node for node in private.routines["helper::work"].execution.content
                      if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    unbound, _, _ = private._segment_projection("helper::work", (child_graph.node_id(child_loop),))
    assert not unbound.summarize("helper::work")["complete"]
    assert private._summary_authority()["selected_descriptor_window"]["proof_identity"] == proof.identity
    assert analysis._summary_authority()["selected_descriptor_window"] is None
    assert analysis.window_module_allocatables(ENTRY) == frozenset()
    assert analysis.summarize("helper::work") == before
    assert not analysis.segment_summary(ENTRY, selected)["complete"]
    assert not SourceEffects(list(analysis.inputs.paths), summary_cache=tmp_path / "cache").summarize("helper::work")["complete"]
    assert analysis.stable_module_allocatables == private.stable_module_allocatables == frozenset()
    assert analysis._allocation_authorizations == private._allocation_authorizations == {}


def test_original_module_descriptor_supplies_runtime_bounds_to_native_sections(tmp_path):
    selected = """do i=lbound(slot,1),ubound(slot,1)
slot(i)=slot(i)+1.0_8
end do"""
    analysis, _, ids, _ = fixture(tmp_path, selected=selected)
    assert not analysis.native_sections_for_nodes(ENTRY, ids).available
    proof = prove(analysis, ids)
    sections = analysis.native_sections_for_nodes(ENTRY, ids, descriptor_window=proof)
    assert sections.available, sections.reason
    resource = next(item for item in sections.resources if item.resource == ROOT)
    assert resource.descriptor_lower_bounds
    assert resource.descriptor_origin == "allocation"
    assert resource.lower_bounds == (None,)
    assert resource.reads
    assert resource.writes
    assert not analysis.native_sections_for_nodes(ENTRY, ids).available


@pytest.mark.parametrize("prefix", [
    "allocate(slot(-7:9))", "call move_alloc(slot,temporary)", "call retire()"])
def test_excluded_prefix_allocation_effects_never_become_freshness(tmp_path, prefix):
    analysis, _, selected, _ = fixture(tmp_path, prefix=prefix)
    public = prove(analysis, selected).public()
    assert not public["prefix_effects_proved"]
    assert not public["entry_definition_authority"]
    assert public["resources"][0]["requires_prior_noescape_and_current_allocation_proof"]


@pytest.mark.parametrize(("selected", "worker"), [
    ("call retire()", None),
    ("deallocate(slot)", None),
    ("allocate(slot(-2:5))", None),
    ("call move_alloc(slot,temporary)", None),
    ("slot=1.0_8", None),
    ("call unknown(slot)", None),
    ("call coordinate(n)", "call retire()"),
    ("call coordinate(n)", "deallocate(local)"),
])
def test_selected_allocation_and_unknown_call_effects_reject(tmp_path, selected, worker):
    analysis, _, ids, _ = fixture(tmp_path, selected=selected, worker=worker)
    with pytest.raises(CompilationError):
        prove(analysis, ids)
    assert not getattr(analysis, "_descriptor_window_proofs", {})
    assert analysis.stable_module_allocatables == frozenset()


@pytest.mark.parametrize(("attributes", "module_extra"), [
    ("allocatable,target", ""), ("pointer", ""),
    ("allocatable", "!$omp threadprivate(backing)")])
def test_module_alias_or_thread_storage_requires_separate_authority(tmp_path, attributes, module_extra):
    analysis, _, ids, _ = fixture(tmp_path, attributes=attributes, module_extra=module_extra)
    with pytest.raises(CompilationError, match="module|storage|allocation"):
        prove(analysis, ids)


def test_foreign_token_lookalike_binding_and_projection_mutations_reject(tmp_path):
    analysis, _, selected, _ = fixture(tmp_path)
    proof = prove(analysis, selected)
    foreign = SourceEffects(list(analysis.inputs.paths))
    for token in (proof.public(), replace(proof)):
        with pytest.raises(CompilationError):
            analysis.segment_summary(ENTRY, selected, descriptor_window=token)
    with pytest.raises(CompilationError):
        foreign.segment_summary(ENTRY, selected, descriptor_window=proof)
    with pytest.raises(CompilationError):
        analysis.segment_summary("driver::unrelated", selected, descriptor_window=proof)
    private, routine, _ = analysis._segment_projection(ENTRY, selected, descriptor_window=proof)
    routine.execution.content.append(F.Assignment_Stmt("slot(1)=5.0_8"))
    with pytest.raises(CompilationError, match="unchanged private"):
        private.summarize(ENTRY)
    original = analysis.modules["storage"].bindings["backing"]
    analysis.modules["storage"].bindings["backing"] = copy(original)
    with pytest.raises(CompilationError, match="module allocations"):
        proof.validate(analysis, ENTRY, selected)


@pytest.mark.parametrize("mutation", ["source", "imports", "binding"])
def test_module_window_stale_source_and_resolution_reject(tmp_path, mutation):
    analysis, _, selected, paths = fixture(tmp_path)
    proof = prove(analysis, selected)
    if mutation == "source":
        paths[0].write_text(paths[0].read_text().replace("implicit none", "implicit none\n!changed"))
    elif mutation == "imports":
        analysis.routines[ENTRY].scope.parent.imports["slot"] = ("foreign", "backing")
    else:
        analysis.modules["storage"].bindings["backing"].kind = 4
    with pytest.raises(CompilationError):
        proof.validate(analysis, ENTRY, selected)


@pytest.mark.parametrize(("depth", "accepted"), [(8, True), (9, False)])
def test_descriptor_depth_counts_original_controls_once(tmp_path, depth, accepted):
    loops = "\n".join(f"do j{i}=1,1" for i in range(depth))
    selected = loops + "\nslot(1)=2.0_8\n" + "\n".join("end do" for _ in range(depth))
    analysis, _, ids, _ = fixture(tmp_path, selected=selected)
    # Original explicit INTEGER declarations retain the ordinary semantic limit.
    path = analysis.inputs.paths[-1]
    text = path.read_text().replace("integer::i", "integer::i," + ",".join(f"j{i}" for i in range(depth)))
    path.write_text(text)
    analysis = SourceEffects(list(analysis.inputs.paths))
    graph = analysis.structure(ENTRY)
    first = next(node for node in analysis.routines[ENTRY].execution.content
                 if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    ids = (graph.node_id(first),)
    if accepted:
        assert prove(analysis, ids).public()["schema_version"] == 4
    else:
        with pytest.raises(CompilationError, match="depth|bounded|unavailable"):
            prove(analysis, ids)


def predicate_fixture(tmp_path):
    selected = """do i=lbound(slot,1),ubound(slot,1)
probe=slot(i)
if(ieee_is_nan(probe)) slot(i)=0.0_8
end do"""
    analysis, _, ids, paths = fixture(tmp_path, selected=selected)
    path = paths[-1]
    path.write_text(path.read_text().replace("module driver", "module driver\nuse,intrinsic::ieee_arithmetic,only:ieee_is_nan", 1)
                    .replace("integer::i", "integer::i\nreal(8)::probe"))
    analysis = SourceEffects(list(analysis.inputs.paths))
    graph = analysis.structure(ENTRY)
    original = next(node for node in analysis.routines[ENTRY].execution.content
                    if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    expression = next(node for node in walk(original)
                      if type(node).__name__ in {"Part_Ref", "Function_Reference", "Intrinsic_Function_Reference"}
                      and str(node.items[0]).lower() == "ieee_is_nan")
    return analysis, (graph.node_id(original),), expression


def test_descriptor_window_consumes_exact_native_predicate_without_execution_authority(tmp_path):
    analysis, selected, expression = predicate_fixture(tmp_path)
    with pytest.raises(CompilationError, match="registered"):
        prove(analysis, selected)
    predicate = prove_native_predicate(analysis, ENTRY, expression)
    proof = prove(analysis, selected)
    requirement, = proof.public()["native_predicate_requirements"]
    assert requirement["proof_identity"] == predicate.identity
    assert requirement["requires_original_execution"]
    assert requirement["exception_behavior"].startswith("retained by unchanged original")
    assert not requirement["gpu_legality_established"]
    assert not proof.public()["execution_authorized"]
    assert not proof.public()["entry_definition_authority"]


def test_descriptor_native_predicate_foreign_and_stale_tokens_reject(tmp_path):
    analysis, selected, expression = predicate_fixture(tmp_path)
    foreign = SourceEffects(list(analysis.inputs.paths))
    foreign_expression = next(node for node in walk(foreign.routines[ENTRY].execution)
                              if type(node).__name__ == "Part_Ref" and str(node.items[0]).lower() == "ieee_is_nan")
    alien = prove_native_predicate(foreign, ENTRY, foreign_expression)
    analysis._native_predicates[alien.identity] = alien
    with pytest.raises(CompilationError, match="registered"):
        prove(analysis, selected)
    analysis._native_predicates.clear()
    predicate = prove_native_predicate(analysis, ENTRY, expression)
    proof = prove(analysis, selected)
    analysis._native_predicates.pop(predicate.identity)
    with pytest.raises(CompilationError, match="registered"):
        proof.validate(analysis, ENTRY, selected)


@pytest.mark.parametrize("selected", [
    "if(n>0) slot(1)=2.0_8", "if(n>0) then\nslot(1)=2.0_8\nend if"])
def test_exact_registered_condition_is_separate_from_its_original_branch(tmp_path, selected):
    analysis, graph, ids, _ = fixture(tmp_path, selected=selected)
    branch = graph.nodes[ids[0]]
    assert branch.kind == "branch"
    condition = branch.alternatives[0][0]
    original = graph.source_nodes(condition)[0]
    condition_proof = prove(analysis, (condition,))
    assert condition_proof.public()["selection_role"] == "original condition"
    summary = analysis.segment_summary(ENTRY, (original,), descriptor_window=condition_proof)
    assert summary["complete"], summary["reasons"]
    assert not any(effect["kind"] in {"write", "overwrite"} for effect in summary["ordered_effects"])
    branch_proof = prove(analysis, ids)
    assert branch_proof.public()["selection_role"] == "original contiguous sequence"
    branch_original = graph.source_nodes(ids[0])[0]
    assert analysis._selected_source(ENTRY, (branch_original,))[1] == ids
    with pytest.raises(CompilationError):
        analysis.segment_summary(ENTRY, (branch_original,), descriptor_window=condition_proof)
    with pytest.raises(CompilationError):
        analysis.segment_summary(ENTRY, (copy(original),), descriptor_window=condition_proof)


def test_single_line_if_selection_keeps_its_unsafe_action(tmp_path):
    analysis, graph, ids, _ = fixture(tmp_path, selected="if(n>0) call unknown(slot)")
    original = graph.source_nodes(ids[0])[0]
    assert type(original).__name__ == "If_Stmt"
    assert analysis._selected_source(ENTRY, (original,))[1] == ids
    with pytest.raises(CompilationError):
        prove(analysis, ids)
    condition = graph.nodes[ids[0]].alternatives[0][0]
    assert prove(analysis, (condition,)).public()["selection_role"] == "original condition"
