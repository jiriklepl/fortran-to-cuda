"""An explicit original window owns runtime state once, after its native prefix."""

from copy import copy, deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.reached_windows import prove_reached_window
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

ENTRY = "renamed_stage::step"
ROOTS = ("argument::a", "argument::b")

SOURCE = """module renamed_stage
implicit none
integer::prefix_visits=0,tail_visits=0
real(8)::prefix_total=0.0_8
contains
subroutine native_prefix(a)
real(8),intent(in)::a(:)
prefix_visits=prefix_visits+1
prefix_total=sum(a)
end subroutine
subroutine step(a,b,n,iterations)
real(8),allocatable,intent(inout)::a(:),b(:)
integer,intent(in)::n,iterations
real(8),allocatable::temporary(:)
integer::stage,i
call native_prefix(a)
do stage=1,iterations
do i=-2,n-3
b(i)=a(i)+real(i+4,8)
enddo
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


def window_fixture(directory, source=SOURCE, *, policy="sections", roots=ROOTS, device_budget=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "stage.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {root: deepcopy(FACT) for root in ROOTS}}
    if device_budget is not None:
        facts["device_budget_bytes"] = device_budget
    builder = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy=policy, scope_execution="reached"))
    routine = builder.entry
    outer = next(node for node in routine.execution.content
                 if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    if "do stage=1,iterations" in source:
        originals = tuple(node for node in outer.content[1:-1]
                          if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    else:
        originals = tuple(node for node in routine.execution.content
                          if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    proof = prove_reached_window(builder.analysis, ENTRY, originals, roots, facts)
    return path, builder, proof, originals


def emit_window(directory, source=SOURCE, *, policy="sections", device_budget=None):
    path, builder, proof, originals = window_fixture(directory, source, policy=policy, device_budget=device_budget)
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    return path, outputs, report, owner


def replacement(path, outputs, report):
    return outputs[report["sources"][str(path)]["replacement"]]


def test_window_controls_are_declared_at_entry_and_spent_only_at_original_window(tmp_path):
    path, outputs, report, owner = emit_window(tmp_path)
    text = replacement(path, outputs, report)
    prefix = text.index("call native_prefix(a)")
    outer = text.index("do stage=1,iterations")
    activation = text.index(f"if (.not. {owner.spent}) then")
    assert text.index(f"logical :: {owner.enabled}, {owner.spent}") < prefix
    assert text.index(f"{owner.spent} = .false.") < prefix < outer < activation
    assert text.index(f"{owner.spent} = .true.") < text.index("fort_scope_serial_caller()")
    assert text.index("fort_scope_serial_caller()") < text.index("allocated(a)")
    assert "fort_scope_create(" not in text[:activation]
    assert text.count("call native_prefix(a)") == 1
    assert text.count(f"{owner.spent} = .false.") == 1
    assert text.count(f"{owner.spent} = .true.") == 1
    assert text.count("call move_alloc(a,temporary)") == 1
    close = text.rindex("fort_scope_close(", 0, text.index("call move_alloc(a,temporary)"))
    assert close > activation
    assert text.index(f"{owner.enabled} = .false.", close) < text.index("call move_alloc(a,temporary)")
    assert path.read_text() == SOURCE


def test_manifest_reports_source_proof_one_shot_lifetime_and_terminal_boundary(tmp_path):
    _path, _outputs, report, owner = emit_window(tmp_path)
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    assert scope["reached_window"]["proof_identity"] == owner.window_proof.identity
    assert scope["reached_window"]["first_traversal_controls"]
    assert scope["reached_window"]["managed_resources"] == list(ROOTS)
    assert scope["reached_window"]["automatic_discovery"] is False
    assert "spent" in scope["reached_window"]["lifecycle"]
    assert "backedge" in scope["reached_window"]["mandatory_close"]
    assert scope["ownership"]["lifetime"] == "one-shot original reached window"
    assert not scope["reopen_after_boundary"]


def test_window_does_not_require_excluded_tail_descriptor_stability(tmp_path):
    _path, builder, proof, originals = window_fixture(tmp_path)
    assert not builder.analysis.descriptor_stability(ENTRY)["complete"]
    original_facts = deepcopy(builder.facts)
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    assert builder.facts == original_facts
    assert builder.analysis.stable_module_allocatables == frozenset()
    assert owner.builder.capture(builder.entry.scope.bindings["a"])["initialized"] == "whole"


def test_top_level_window_keeps_declarations_before_native_prefix(tmp_path):
    source = SOURCE.replace("do stage=1,iterations\n", "").replace(
        "tail_visits=tail_visits+1\nenddo", "tail_visits=tail_visits+1")
    path, outputs, report, owner = emit_window(tmp_path, source)
    text = replacement(path, outputs, report)
    assert not report["scopes"][0]["reached_window"]["first_traversal_controls"]
    assert text.index(f"logical :: {owner.enabled}, {owner.spent}") < text.index("call native_prefix(a)")


def test_static_native_auto_selection_leaves_entire_original_source_unchanged(tmp_path):
    path, outputs, report, _owner = emit_window(tmp_path, policy="auto")
    text = replacement(path, outputs, report)
    assert text == SOURCE
    scope, = report["scopes"]
    assert scope["automatic_preflight"]["selection"] == "native"
    assert scope["automatic_preflight"]["contexts_created"] == 0
    assert scope["reached_window"]["proof_identity"]


def test_explicit_window_retains_iterator_definition_observed_by_excluded_tail(tmp_path):
    source = SOURCE.replace("tail_visits=tail_visits+1", "tail_visits=tail_visits+i")
    path, outputs, report, _owner = emit_window(tmp_path, source)
    text = replacement(path, outputs, report)
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 1
    assert "tail_visits=tail_visits+i" in text
    assert "a(i)=2.0_8*b(i)" in text
    assert "do i=-2,n-3" in text


def test_unregistered_or_foreign_proofs_cannot_emit_controls(tmp_path):
    path, builder, proof, originals = window_fixture(tmp_path)
    for wrong in (copy(proof), object()):
        with pytest.raises(CompilationError, match="registered"):
            ReachedWindowOwner(builder, wrong, originals)
    foreign = ScopeBuilder([path], ENTRY, facts=builder.facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    with pytest.raises(CompilationError, match="registered"):
        ReachedWindowOwner(foreign, proof, originals)


@pytest.mark.parametrize("selection", ["skip", "reorder", "extend"])
def test_prepared_selection_cannot_skip_reorder_or_extend_the_proved_window(tmp_path, selection):
    _path, builder, proof, originals = window_fixture(tmp_path)
    owner = ReachedWindowOwner(builder, proof, originals)
    wrong = {"skip": originals[:1], "reorder": tuple(reversed(originals)),
             "extend": (*originals, builder.entry.execution.content[0])}[selection]
    before = proof.public()
    with pytest.raises(CompilationError, match="differs"):
        owner.run(wrong)
    assert owner.window_proof is proof
    assert proof.public() == before
    assert not owner.builder.edits
    assert not owner.builder.outputs
    assert not owner.builder.scopes
    with pytest.raises(CompilationError, match="failed reached consumption candidate"):
        owner.run(originals)


@pytest.mark.parametrize("mutation", ["facts", "source"])
def test_later_source_or_capture_fact_mutation_invalidates_emission(tmp_path, mutation):
    path, builder, proof, originals = window_fixture(tmp_path)
    owner = ReachedWindowOwner(builder, proof, originals)
    if mutation == "facts":
        builder.facts["captures"][ROOTS[0]]["initialized"] = "none"
        reason = "facts|authority"
    else:
        path.write_text(SOURCE + "\n! changed\n")
        reason = "source changed"
    before = proof.public()
    with pytest.raises(CompilationError, match=reason):
        owner.run()
    builder.facts["captures"][ROOTS[0]]["initialized"] = "whole"
    path.write_text(SOURCE)
    assert proof.public() == before
    assert not owner.builder.edits
    assert not owner.builder.outputs
    with pytest.raises(CompilationError, match="failed reached consumption candidate"):
        owner.run()


def test_window_bound_capture_and_summary_cannot_escape_registered_roots_or_nodes(tmp_path):
    _path, builder, proof, originals = window_fixture(tmp_path)
    owner = ReachedWindowOwner(builder, proof, originals)
    unknown = copy(builder.entry.scope.bindings["a"])
    unknown.root = "argument::unproved"
    owner.builder.facts["captures"][unknown.root] = deepcopy(FACT)
    with pytest.raises(CompilationError, match="facts|authority|proved"):
        owner.builder.capture(unknown)
    del owner.builder.facts["captures"][unknown.root]
    prefix = builder.entry.execution.content[0]
    with pytest.raises(CompilationError, match="outside|selected|window"):
        owner.builder.segment_summary(builder.analysis, ENTRY, (prefix,), capture_locals=True)


def test_same_root_copy_cannot_inherit_original_window_storage_authority(tmp_path):
    _path, builder, proof, originals = window_fixture(tmp_path)
    owner = ReachedWindowOwner(builder, proof, originals)
    binding = copy(builder.entry.scope.bindings["a"])
    with pytest.raises(CompilationError, match="exact original storage"):
        owner.builder.capture(binding)
