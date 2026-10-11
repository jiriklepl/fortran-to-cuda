"""One conditional owner: close before a native replacement, never replay."""

from copy import copy, deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.descriptor_windows import DescriptorWindowUnsupported
from compiler.frontend.reached_windows import plan_reached_window, prove_reached_window
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

ENTRY = "renamed_stage::step"
ROOTS = ("argument::a", "argument::b")

PLANNED_SOURCE = """module renamed_stage
implicit none
integer::prefix_visits=0,tail_visits=0,bad_visits=0,good_visits=0,condition_visits=0
real(8)::prefix_total=0.0_8
contains
subroutine native_prefix(a)
real(8),intent(in)::a(:)
prefix_visits=prefix_visits+1
prefix_total=sum(a)
end subroutine
subroutine replace_storage(a)
real(8),allocatable,intent(inout)::a(:)
real(8),allocatable::temporary(:)
allocate(temporary(lbound(a,1):ubound(a,1)))
temporary=a+10.0_8
call move_alloc(temporary,a)
bad_visits=bad_visits+1
end subroutine
subroutine face_update(a)
real(8),intent(inout)::a(:)
a(1)=a(1)+3.0_8
good_visits=good_visits+1
end subroutine
subroutine step(a,b,n,iterations,bad)
real(8),allocatable,intent(inout)::a(:),b(:)
integer,intent(in)::n,iterations
logical,intent(in)::bad
real(8),allocatable::temporary(:)
integer::stage,i
call native_prefix(a)
do stage=1,iterations
do i=-2,n-3
b(i)=a(i)+real(i+4,8)
enddo
condition_visits=condition_visits+1
if(bad) then
call replace_storage(b)
else
call face_update(b)
endif
do i=-2,n-3
a(i)=2.0_8*b(i)
enddo
call move_alloc(a,temporary)
call move_alloc(temporary,a)
tail_visits=tail_visits+1
enddo
end subroutine
end module
"""


def planned_window_fixture(directory, source=PLANNED_SOURCE, *, policy="sections", device_budget=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "planned_stage.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {root: deepcopy(FACT) for root in ROOTS}}
    if device_budget is not None:
        facts["device_budget_bytes"] = device_budget
    builder = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy=policy, scope_execution="reached"))
    outer = next(node for node in builder.entry.execution.content
                 if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    originals = tuple(outer.content[1:5])
    proof = plan_reached_window(builder.analysis, ENTRY, originals, ROOTS, facts)
    return path, builder, proof, originals


def emit_planned_window(directory, source=PLANNED_SOURCE, *, policy="sections", device_budget=None):
    path, builder, proof, originals = planned_window_fixture(directory, source, policy=policy,
                                                            device_budget=device_budget)
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    return path, outputs, report, owner


def test_plain_window_still_requires_complete_all_branch_lifetime(tmp_path):
    _path, builder, _proof, originals = planned_window_fixture(tmp_path)
    with pytest.raises(CompilationError, match="unknown selected storage|whole allocatable"):
        prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)
    assert not builder.analysis.descriptor_stability(ENTRY)["complete"]


def test_planned_bad_arm_closes_and_good_arm_retains_active_coverage(tmp_path):
    _path, builder, proof, originals = planned_window_fixture(tmp_path)
    record = proof.public()
    planned = record["planned_closures"]
    point, = planned["close_points"]
    assert point["original_source_kind"] == "Call_Stmt"
    assert "replace_storage" in str(builder.analysis.structure(ENTRY).source_nodes(point["node_id"])[0])
    assert planned["closed_path_possible"]
    assert planned["active_path_may_return"]
    assert not planned["conditional_return_is_unconditional_definition_summary"]
    assert planned["mapped_child_plans"] == []  # Complete native face call needs no mapped variant.
    assert "partial" not in point["execution"]
    assert proof.validate(builder.analysis, ENTRY, originals) is proof
    for state in record["states"]:
        assert state["coverage_on_return"] == "unknown_after_close"
        assert state["active_path_coverage_on_return"] == "whole"
        assert state["active_path_return_is_conditional"]
        assert state["escapes"] is None
        assert state["allocation_changes"] is None
        assert state["closed_path_storage_authority"] == "unavailable"
    assert not record["execution_authorized"]
    assert not record["gpu_legality_established"]


def test_emitter_closes_at_exact_bad_call_before_native_suffix_once(tmp_path):
    path, outputs, report, owner = emit_planned_window(tmp_path)
    text = outputs[report["sources"][str(path)]["replacement"]]
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    assert len(owner.units) == 3  # Two numerical units and the supported native face call.
    assert len(scope["boundaries"]) == 1
    point, = scope["reached_window"]["planned_closures"]["close_points"]
    boundary, = scope["boundaries"]
    assert boundary["original_node_ids"] == [point["node_id"]]
    close = text.rindex("fort_scope_close(", 0, text.index("call replace_storage(b)"))
    assert text.index("if(bad) then") < close < text.index("call replace_storage(b)")
    assert text.index(f"{owner.enabled} = .false.", close) < text.index("call replace_storage(b)")
    assert text.count("call replace_storage(b)") == 1
    assert text.count("call native_prefix(a)") == 1
    assert text.count("condition_visits=condition_visits+1") == 1
    assert text.count("call move_alloc(a,temporary)") == 1
    assert text.count(f"{owner.spent} = .true.") == 1
    assert path.read_text() == PLANNED_SOURCE
    assert not scope["reopen_after_boundary"]


def test_closed_suffix_has_no_descriptor_or_query_authority(tmp_path):
    source = PLANNED_SOURCE.replace("if(bad) then\ncall replace_storage(b)\nelse\ncall face_update(b)\nendif",
                                    "call replace_storage(b)")
    # Select producer/counter/replacement/consumer; the later loop is retained
    # original native source, even if the replacement changes its allocation.
    path, builder, proof, originals = planned_window_fixture(tmp_path, source)
    owner = ReachedWindowOwner(builder, proof, originals)
    consumer = originals[-1]
    with pytest.raises(CompilationError, match="planned close|closed native"):
        owner.builder.segment_summary(builder.analysis, ENTRY, (consumer,), capture_locals=True)
    assert owner.run()
    outputs, report = owner.builder.finish()
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert len(owner.units) == 1
    native_suffix = text[text.index("call replace_storage(b)"):text.index("call move_alloc(a,temporary)")]
    assert "a(i)=2.0_8*b(i)" in native_suffix
    assert "fort_scope_create(" not in native_suffix
    assert "fort_scope_register" not in native_suffix
    assert "fort_scope_plan" not in native_suffix


def test_unknown_relevant_prefix_disables_planned_activation_and_plain_proof_stays_strict(tmp_path):
    source = PLANNED_SOURCE.replace("call native_prefix(a)", "call unknown_prefix(a)")
    _path, builder, proof, originals = planned_window_fixture(tmp_path, source)
    record = proof.public()
    prefix = record["prefix_eligibility"]
    assert prefix["disabled_path_possible"]
    assert not prefix["activation_path_possible"]
    point, = prefix["disable_points"]
    assert point["original_source_kind"] == "Call_Stmt"
    assert point["disabled_is_absorbing"]
    assert not point["context_exists"]
    assert not point["reopen"]
    assert not record["source_demands"]
    assert not record["planned_closures"]["active_node_ids"]
    owner = ReachedWindowOwner(builder, proof, originals)
    assert not owner.run()
    assert not owner.builder.edits
    assert not owner.builder.scopes
    assert not owner.units
    with pytest.raises(CompilationError, match="native prefix|complete.*effects|original source|unknown effects"):
        prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)


def test_foreign_or_forged_plan_cannot_emit(tmp_path):
    path, builder, proof, originals = planned_window_fixture(tmp_path)
    with pytest.raises(CompilationError, match="registered"):
        ReachedWindowOwner(builder, copy(proof), originals)
    foreign = ScopeBuilder([path], ENTRY, facts=builder.facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    with pytest.raises(CompilationError, match="registered"):
        ReachedWindowOwner(foreign, proof, originals)


@pytest.mark.parametrize("mutation", ["source", "binding", "declaring_scope", "imports"])
def test_mutable_source_authority_is_not_converted_into_native_close(tmp_path, mutation):
    path, builder, proof, originals = planned_window_fixture(tmp_path)
    if mutation == "source":
        path.write_text(PLANNED_SOURCE + "\n! source mutation\n")
    elif mutation == "binding":
        scope = builder.analysis.routines["renamed_stage::face_update"].scope
        scope.bindings["a"] = copy(scope.bindings["a"])
    elif mutation == "declaring_scope":
        builder.entry.scope.bindings["a"].declaring_scope = builder.analysis.modules["renamed_stage"]
    else:
        builder.entry.scope.imports["mystery"] = ("unknown", "mystery")
    with pytest.raises(CompilationError, match="changed|authority"):
        proof.validate(builder.analysis, ENTRY, originals)


def test_partial_or_excluded_close_nodes_are_not_accepted(tmp_path):
    _path, builder, _proof, originals = planned_window_fixture(tmp_path)
    condition = originals[2].content[0]
    tail = next(node for node in builder.entry.execution.content if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    tail = tail.content[-4]
    for wrong in ((condition,), (tail,), (copy(originals[2]),)):
        with pytest.raises(CompilationError, match="original|selected|construct"):
            plan_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts,
                                close_before=((wrong, "forced test cut"),))


def test_unknown_single_line_if_action_closes_before_whole_condition(tmp_path):
    source = PLANNED_SOURCE.replace("if(bad) then\ncall replace_storage(b)\nelse\ncall face_update(b)\nendif",
                                    "if(bad) call replace_storage(b)")
    path, outputs, report, owner = emit_planned_window(tmp_path, source)
    text = outputs[report["sources"][str(path)]["replacement"]]
    point, = report["scopes"][0]["reached_window"]["planned_closures"]["close_points"]
    assert point["original_source_kind"] == "If_Stmt"
    assert text.count("if(bad) call replace_storage(b)") == 1
    assert text.index("fort_scope_close(") < text.index("if(bad) call replace_storage(b)")
    assert len(owner.units) == 1


def test_rejected_staged_generation_preserves_original_containers(tmp_path):
    _path, builder, proof, originals = planned_window_fixture(tmp_path)
    original_edits = deepcopy(builder.edits)
    original_registry = dict(builder.analysis._reached_window_proofs)
    def reject():
        with GenerationTransaction(builder) as transaction:
            owner = ReachedWindowOwner(transaction.builder, proof, originals)
            assert owner.run()
            transaction.adopt(owner.builder)
            raise CompilationError("reject staged candidate")
    with pytest.raises(CompilationError, match="reject staged candidate"):
        reject()
    assert builder.edits == original_edits
    assert builder.analysis._reached_window_proofs == original_registry
    assert not builder.scopes


def test_public_discovery_adopts_planned_branch_without_full_child_summary(tmp_path):
    _path, builder, _proof, _originals = planned_window_fixture(tmp_path)
    _outputs, report = builder.run()
    discovery = report["reached_window_discovery"]
    assert discovery["selected"], discovery
    assert discovery["schema_version"] == 4
    scope, = report["scopes"]
    assert scope["reached_window"]["automatic_discovery"]
    assert scope["reached_window"]["planned_closures"]["closed_path_possible"]
    assert len(scope["gpu_leaves"]) == 2
    child = builder.analysis.summarize("renamed_stage::replace_storage")
    assert not child["complete"]


def test_source_validation_failure_is_hard_without_diagnostic_keyword_matching(tmp_path, monkeypatch):
    import compiler.frontend.descriptor_windows as descriptor

    _path, builder, proof, originals = planned_window_fixture(tmp_path)
    bad_id = proof.public()["planned_closures"]["close_points"][0]["node_id"]
    original = descriptor.prove_descriptor_window

    def fail(analysis, procedure, selection, *args, **kwargs):
        if selection == (bad_id,):
            raise CompilationError("unusual authentication rejection")
        return original(analysis, procedure, selection, *args, **kwargs)

    monkeypatch.setattr(descriptor, "prove_descriptor_window", fail)
    with pytest.raises(CompilationError, match="unusual authentication rejection"):
        proof.validate(builder.analysis, ENTRY, originals)


def test_local_expression_limit_can_close_without_spending_more_global_budget(tmp_path):
    deep = "a(0)"
    for _ in range(10):
        deep = "abs(" + deep + ")"
    source = PLANNED_SOURCE.replace("call replace_storage(b)", "b(0)=" + deep)
    _path, builder, proof, originals = planned_window_fixture(tmp_path, source)
    record = proof.public()
    point, = record["planned_closures"]["close_points"]
    assert "expression depth exhausted" in point["reason"]
    assert point["original_source_kind"] == "Assignment_Stmt"
    assert record["ordered_effect_work"] <= builder.analysis.operation_limit
    assert record["ordered_effects"] <= record["ordered_effect_work"]
    assert proof.validate(builder.analysis, ENTRY, originals) is proof


def test_global_descriptor_scan_exhaustion_does_not_become_a_cut(tmp_path, monkeypatch):
    import compiler.frontend.descriptor_windows as descriptor

    _path, builder, proof, originals = planned_window_fixture(tmp_path)
    bad_id = proof.public()["planned_closures"]["close_points"][0]["node_id"]
    original = descriptor.prove_descriptor_window

    def fail(analysis, procedure, selection, *args, **kwargs):
        if selection == (bad_id,):
            raise CompilationError("descriptor window original-source traversal budget exhausted")
        return original(analysis, procedure, selection, *args, **kwargs)

    monkeypatch.setattr(descriptor, "prove_descriptor_window", fail)
    with pytest.raises(CompilationError, match="traversal budget exhausted"):
        proof.validate(builder.analysis, ENTRY, originals)


def test_whole_branch_upgrade_discards_descendant_demand_metadata(tmp_path):
    source = PLANNED_SOURCE.replace("if(bad) then\ncall replace_storage(b)\nelse\ncall face_update(b)\nendif",
                                    "if(bad) call replace_storage(b)")
    _path, _builder, proof, _originals = planned_window_fixture(tmp_path, source)
    record = proof.public()
    assert len(record["planned_closures"]["close_points"]) == 1
    assert not record["planned_closures"]["active_path_may_return"]
    assert record["ordered_effect_work"] >= record["ordered_effects"]
    point = record["planned_closures"]["close_points"][0]
    assert point["node_id"] not in record["planned_closures"]["active_node_ids"]
    assert not any("replace_storage" in demand for demand in record["source_demands"])


def test_forced_nested_cut_cannot_be_silently_ignored_by_a_loop_unit(tmp_path):
    source = PLANNED_SOURCE.replace("b(i)=a(i)+real(i+4,8)", "b(i)=a(i)+real(i+4,8)\ncall face_update(b)")
    _path, builder, _proof, originals = planned_window_fixture(tmp_path, source)
    inner = next(node for node in originals[0].content if type(node).__name__ == "Call_Stmt")
    with pytest.raises(CompilationError, match="active complete source planning unit"):
        plan_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts,
                            close_before=(((inner,), "unsupported inner unit"),))


def test_later_generation_cut_revalidates_exact_emission_and_recounts_live_kernels(tmp_path, monkeypatch):
    from compiler.scopes.lexical import LexicalOwner
    from compiler.scopes.window_discovery import _numerical_calls

    path, builder, proof, originals = planned_window_fixture(tmp_path)
    admit = LexicalOwner.admit

    def fail_face(owner, nodes, **options):
        if len(nodes) == 1 and "call face_update" in str(nodes[0]).lower():
            owner.boundary(nodes, "separate mapped child continuation unavailable")
            return
        return admit(owner, nodes, **options)

    monkeypatch.setattr(LexicalOwner, "admit", fail_face)
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    scope, = report["scopes"]
    assert owner.window_proof.identity != proof.identity
    assert scope["reached_window"]["control_source_proof_identity"] == proof.identity
    assert scope["reached_window"]["validated_emission_proof_identity"] == owner.window_proof.identity
    assert len(scope["reached_window"]["planned_closures"]["close_points"]) == 2
    assert len(scope["gpu_leaves"]) == 1
    assert _numerical_calls(owner) == 1
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert "a(i)=2.0_8*b(i)" in text


def test_unsupported_ieee_observer_remains_native_without_gpu_authority(tmp_path):
    source = PLANNED_SOURCE.replace("implicit none", "use, intrinsic::ieee_exceptions\nimplicit none", 1)
    source = source.replace("logical,intent(in)::bad", "logical,intent(in)::bad\nlogical::raised")
    source = source.replace("call replace_storage(b)", "call ieee_get_flag(ieee_invalid,raised)")
    _path, builder, proof, originals = planned_window_fixture(tmp_path, source)
    owner = ReachedWindowOwner(builder, proof, originals)
    # The configured-source IEEE observer check still owns numerical legality.
    # A close plan is never permission to lose earlier GPU exception effects.
    assert not owner.run()
    assert not owner.leaves
    assert not owner.builder.edits


def test_plannable_failure_has_explicit_type_without_new_execution_authority():
    assert isinstance(DescriptorWindowUnsupported("local unsupported source"), CompilationError)


def test_unproved_complete_child_team_closes_before_whole_original_call(tmp_path):
    first = PLANNED_SOURCE.index("subroutine replace_storage(a)")
    last = PLANNED_SOURCE.index("subroutine face_update(a)", first)
    native = """subroutine replace_storage(a)
real(8),intent(inout)::a(:)
!$omp critical
a(1)=a(1)+10.0_8
!$omp end critical
bad_visits=bad_visits+1
end subroutine
"""
    source = PLANNED_SOURCE[:first] + native + PLANNED_SOURCE[last:]
    _path, builder, proof, originals = planned_window_fixture(tmp_path, source)
    point, = proof.public()["planned_closures"]["close_points"]
    assert point["original_source_kind"] == "Call_Stmt"
    assert "whole-closure authority" in point["reason"]
    with pytest.raises(CompilationError, match="whole-closure authority"):
        prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)
    # The cut closes outside the child's original team. It does not admit or
    # instrument any critical/parallel group or derive its completion effects.
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    text = outputs[report["sources"][str(_path)]["replacement"]]
    assert "!$omp critical\na(1)=a(1)+10.0_8\n!$omp end critical" in text
    assert len(owner.units) == 3
