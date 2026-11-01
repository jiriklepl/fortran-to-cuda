"""Private native allocation changes never become managed storage authority."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from compiler.frontend.descriptor_windows import DescriptorWindowProof, prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.source import ScopeBuilder
from compiler.scopes.storage_epoch import _private_allocation_inventory

SOURCE = """module private_scratch
implicit none
contains
subroutine work(a)
real(8),intent(inout)::a(:)
real(8),save,allocatable::scratch(:)
integer::i
scratch=a
a(1)=a(1)+sum(scratch)
contains
pure real(8) function unused(x)
real(8),intent(in)::x
unused=x*x
end function
end subroutine
subroutine step(a)
real(8),allocatable,intent(inout)::a(:)
call work(a)
end subroutine
end module
"""
ROOT = "private_scratch::work::scratch"


def fixture(tmp_path, *, source=SOURCE, sealed=False):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines["private_scratch::step"]
    graph, selected = analysis._selected_source(routine.qualified, tuple(routine.execution.content))
    return path, analysis, routine, graph, selected


def issue(parts):
    _path, analysis, routine, _graph, selected = parts
    return prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))


def inventory(parts, proof, *, registered=False):
    _path, analysis, routine, _graph, _selected = parts
    # A genuine source-issued token models the graph's complete descriptor
    # inventory. No execution or lifetime epoch is issued by this fixture.
    resources = {"argument::a": (routine.scope.bindings["a"], 1)}
    if registered:
        resources[ROOT] = (analysis.routines["private_scratch::work"].scope.bindings["scratch"], 2)
    return SimpleNamespace(position_epoch=SimpleNamespace(_descriptor=proof), resources=resources,
                           members={})


@pytest.mark.parametrize("sealed", [False, True])
def test_saved_scratch_mutation_preserves_only_unrelated_roots_conditionally(tmp_path, sealed):
    parts = fixture(tmp_path, sealed=sealed)
    before = parts[0].read_bytes()
    proof = issue(parts)
    public = proof.public()
    assert public["schema_version"] == 21
    assert public["requires_unregistered_private_allocations"] == [ROOT]
    assert [item["resource"] for item in public["resources"]] == ["argument::a"]
    assert public["resources"][0]["stable"] is True
    assert public["entry_definition_authority"] is False
    assert public["execution_authorized"] is False
    assert public["native_private_allocation_requirements"][0]["allocation_presence_authority"] is False
    assert parts[0].read_bytes() == before
    mutation = parts[1].storage_mutation_effects("private_scratch::work").public()
    assert any(item["resource"] == ROOT for item in mutation["mutations"])
    assert ROOT not in mutation["private_local_resources"]
    owner = inventory(parts, proof)
    requirements, finish = _private_allocation_inventory(owner)
    assert requirements == [{"descriptor_identity": proof.identity, "descriptor_procedure": proof.procedure,
        "unregistered_private_allocations": [{"resource": ROOT, "declaring_procedure": "private_scratch::work"}]}]
    assert proof.procedure == "private_scratch::step"
    finish()


@pytest.mark.parametrize("sealed", [False, True])
def test_original_scratch_cannot_itself_be_protected_across_reallocation(tmp_path, sealed):
    parts = fixture(tmp_path, sealed=sealed)
    analysis = parts[1]
    routine = analysis.routines["private_scratch::work"]
    _graph, selected = analysis._selected_source(routine.qualified, tuple(routine.execution.content))
    with pytest.raises(CompilationError, match="whole allocatable"):
        prove_descriptor_window(analysis, routine.qualified, selected, (ROOT,))


@pytest.mark.parametrize("mutation", [
    lambda s: s.replace("save,allocatable", "save,target,allocatable"),
    lambda s: s.replace("unused=x*x", "unused=sum(scratch)+x"),
    lambda s: s.replace("scratch=a", "scratch=a\ncall unknown(scratch)"),
])
def test_alias_or_contained_export_never_becomes_private_storage_authority(tmp_path, mutation):
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, source=mutation(SOURCE), sealed=True))


@pytest.mark.parametrize("sealed", [False, True])
def test_any_live_registration_including_a_host_reservation_rejects(tmp_path, sealed):
    parts = fixture(tmp_path, sealed=sealed)
    proof = issue(parts)
    owner = inventory(parts, proof, registered=True)
    with pytest.raises(CompilationError, match="overlaps a live canonical registration"):
        _private_allocation_inventory(owner)


@pytest.mark.parametrize("mutation", ["register", "binding", "file", "token", "registry", "parent", "parent_descriptor"])
@pytest.mark.parametrize("sealed", [False, True])
def test_final_source_callback_cannot_change_earlier_inventory_or_authority(tmp_path, monkeypatch, mutation, sealed):
    parts = fixture(tmp_path, sealed=sealed)
    path, analysis = parts[:2]
    proof = issue(parts)
    owner = inventory(parts, proof)
    _requirements, finish = _private_allocation_inventory(owner)
    original = DescriptorWindowProof.validate
    scratch = analysis.routines["private_scratch::work"].scope.bindings["scratch"]

    def changed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self is proof:
            if mutation == "register":
                owner.resources[ROOT] = (scratch, 2)
            elif mutation == "binding":
                if sealed:
                    # Closed arena source rejects the supported mutation API;
                    # unsafe object.__setattr__ is not a production API.
                    scratch.kind = 16
                else:
                    object.__setattr__(scratch, "kind", 16)
            elif mutation == "file":
                path.write_text(path.read_text() + "! changed after final descriptor callback\n")
            elif mutation == "token":
                object.__setattr__(self, "_roots", ())
            elif mutation == "registry":
                analysis._descriptor_window_proofs = dict(analysis._descriptor_window_proofs)
            elif mutation == "parent":
                owner.position_epoch = SimpleNamespace(_descriptor=proof)
            else:
                owner.position_epoch._descriptor = replace(proof)
        return result

    monkeypatch.setattr(DescriptorWindowProof, "validate", changed)
    with pytest.raises(CompilationError):
        finish()


def test_a_copied_descriptor_is_not_an_execution_inventory_obligation(tmp_path):
    parts = fixture(tmp_path)
    proof = issue(parts)
    copied = replace(proof)
    owner = inventory(parts, copied)
    with pytest.raises(CompilationError):
        _private_allocation_inventory(owner)


def test_fixed_scratch_does_not_change_the_legacy_descriptor_schema(tmp_path):
    source = SOURCE.replace("save,allocatable::scratch(:)", "dimension(4)::scratch")
    proof = issue(fixture(tmp_path, source=source))
    assert proof.public()["schema_version"] == 6
    assert "requires_unregistered_private_allocations" not in proof.public()


def test_conditional_effects_cannot_escape_into_an_unqualified_consumer(tmp_path):
    proof = issue(fixture(tmp_path))
    builder = object.__new__(ScopeBuilder)
    with pytest.raises(CompilationError, match="complete graph storage epoch"):
        builder._check_private_descriptor_window(proof)


def real_epoch(tmp_path):
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.graph_epochs import discover_graph_storage_epoch
    from compiler.tests.test_graph_storage_epochs import SOURCE as EPOCH_SOURCE
    from compiler.tests.test_graph_storage_epochs import builder_at

    source = EPOCH_SOURCE.replace("real(8),intent(out)::b(:)", "real(8),intent(inout)::b(:)", 1)
    source = source.replace("integer::i\ndo i=1,n", "integer::i\nreal(8),allocatable,save::scratch(:)\nscratch=b\ndo i=1,n", 1)
    source = source.replace("call relocate(left,right)", "call relocate(left,right)\ncall producer(right,n)")
    _path, builder = builder_at(tmp_path, source)
    owner = GraphOwner(builder)
    epoch = discover_graph_storage_epoch(owner)
    assert epoch is not None, owner.position_epoch_declines
    builder._graph_storage_epoch = epoch
    return builder, owner, epoch


def test_real_graph_epoch_carries_private_callee_identity_to_child_consumers(tmp_path):
    from compiler.scopes.segments import _native_analysis

    builder, owner, epoch = real_epoch(tmp_path)
    assert epoch._descriptor.public()["requires_unregistered_private_allocations"] == ["epochs::producer::scratch"]
    assert builder._check_private_descriptor_window(epoch._descriptor) is epoch._descriptor
    native = _native_analysis(builder.analysis)
    child = native.routines["epochs::producer"]
    _graph, selected = native._selected_source(child.qualified, tuple(child.execution.content))
    descriptor = prove_descriptor_window(native, child.qualified, selected, ("argument::b",))
    assert builder._check_private_descriptor_window(descriptor) is descriptor
    obligations, finish = _private_allocation_inventory(owner)
    assert obligations[0]["descriptor_procedure"] == "epochs::advance"
    assert obligations[0]["unregistered_private_allocations"] == [
        {"resource": "epochs::producer::scratch", "declaring_procedure": "epochs::producer"}]
    finish()


def test_real_epoch_cannot_consume_an_original_private_mutation_after_its_cut(tmp_path):
    builder, _owner, epoch = real_epoch(tmp_path)
    graph = builder.analysis.structure(builder.entry.qualified)
    loops = [node for node in builder.entry.execution.content if type(node).__name__ == "Block_Nonlabel_Do_Construct"]
    call = [node for node in loops[0].content if type(node).__name__ == "Call_Stmt"][-1]
    selected = (graph.node_id(call),)
    proof = prove_descriptor_window(builder.analysis, builder.entry.qualified, selected, epoch.roots)
    with pytest.raises(CompilationError, match="exceeds its complete original epoch"):
        builder._check_private_descriptor_window(proof)


def test_real_graph_epoch_rejects_a_same_text_foreign_source_domain(tmp_path):
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    builder, _owner, _epoch = real_epoch(tmp_path / "first")
    _foreign_builder, _foreign_owner, foreign = real_epoch(tmp_path / "second")
    with pytest.raises(CompilationError, match="original source domain"):
        builder._check_private_descriptor_window(foreign._descriptor)


@pytest.mark.parametrize("sealed", [False, True])
def test_complete_original_join_is_retained_around_private_scratch_use(tmp_path, sealed):
    source = SOURCE.replace("a(1)=a(1)+sum(scratch)", """!$omp parallel do private(i) schedule(static)
do i=1,size(a)
a(i)=a(i)+scratch(i)
end do
!$omp end parallel do""")
    public = issue(fixture(tmp_path, source=source, sealed=sealed)).public()
    assert public["requires_unregistered_private_allocations"] == [ROOT]
    assert public["native_joined_descriptor_requirements"]
    assert public["native_private_allocation_requirements"][0]["joined_storage"]
