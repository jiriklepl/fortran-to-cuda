"""Every retained edge is checked; an excluded edge has no retention grant."""
import sys
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.allocation_epoch_eligibility import prove_allocation_gate
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.supported_path_storage import prove_supported_path_storage
from compiler.frontend.supported_paths import prove_supported_path
from compiler.ir import CompilationError

SOURCE = """module supported_storage
implicit none
contains
subroutine advance(n,mode)
integer,intent(in)::n,mode
integer,save::called=0
logical,save::eligible=.false.
real(8),save,allocatable::a(:),b(:)
if(called==0)then
allocate(a(n))
allocate(b(n))
a=0
eligible=.true.
endif
called=called+1
call produce(a,b,n)
if(mode==1)then
eligible=.false.
call unavailable(a,b)
else
call selected(a,b,n)
endif
call finish(a,b,n)
end subroutine
subroutine produce(a,b,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
a(1:n)=1.0_8
b(1:n)=2.0_8
end subroutine
subroutine selected(a,b,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
a(1:n)=a(1:n)+b(1:n)
end subroutine
subroutine finish(a,b,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
b(1:n)=a(1:n)
end subroutine
subroutine unavailable(a,b)
real(8),intent(inout)::a(:),b(:)
call unknown_escape(a)
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, *, sealed=True):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines["supported_storage::advance"]
    originals = tuple(routine.execution.content[2:])
    roots = tuple("supported_storage::"+name for name in ("produce", "selected", "finish"))
    selection = prove_supported_path(analysis, routine, originals, roots)
    barriers = [node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt"
                and str(node.items[0]).lower() == "unavailable"]
    gate = prove_allocation_gate(analysis, routine, routine.scope.bindings["eligible"],
        [routine.scope.bindings[name] for name in ("a", "b")], barriers)
    resources = tuple("supported_storage::advance::"+name for name in ("a", "b"))
    return path, analysis, selection, gate, resources


def issue(parts):
    return prove_supported_path_storage(*parts[1:])


@pytest.mark.parametrize("sealed", [False, True])
def test_composition_covers_all_retained_nodes_and_binds_only_exact_barred_call(tmp_path, sealed):
    parts = fixture(tmp_path, sealed=sealed)
    _path, analysis, path, gate, resources = parts
    proof = issue(parts)
    assert issue(parts) is proof
    assert proof.validate() is proof
    record = proof.public()
    assert len(record["barred_calls"]) == 1
    assert len(record["components"]) == 5
    assert "supported_storage::unavailable" not in record["source_call_closure"]
    assert set(record["source_call_closure"]) >= {"supported_storage::produce", "supported_storage::selected", "supported_storage::finish"}
    assert record["additional_unproved_edge"] == "reject entire supported path"
    assert record["complete_retained_graph"] is True
    for flag in ("entry_definition_authority", "allocation_presence_authority", "payload_access_authorized",
                 "execution_authorized", "alias_authority", "cross_frontier_retention_authorized"):
        assert record[flag] is False
    for resource in resources:
        assert proof.resource_state(resource)["entry_definition_authority"] is False
    component = proof.descriptor_for((path._originals[0],))
    _graph, selected = analysis._selected_source(path._routine.qualified, (path._originals[0],))
    assert component.validate(analysis, path._routine.qualified, selected) is component
    with pytest.raises(CompilationError, match="excluded or foreign"):
        proof.descriptor_for(path._originals)
    with pytest.raises(CompilationError, match="excluded or foreign"):
        proof.descriptor_for(gate._barriers)


@pytest.mark.parametrize("statement", ["call system_clock(count=rate)", "write(*,*) 'native rate',rate"])
def test_original_native_boundary_requires_the_independent_descriptor_issuer(tmp_path, statement):
    source = SOURCE.replace("integer,save::called=0", "integer,save::called=0\ninteger::rate=0").replace(
        "call selected(a,b,n)", statement+"\ncall selected(a,b,n)")
    parts = fixture(tmp_path, source)
    proof = issue(parts)
    _path, analysis, selection, _gate, _resources = parts
    graph = analysis.structure(selection._routine.qualified)
    boundaries = [(node, token) for node, token in proof._components if graph.nodes[node].kind == "boundary"]
    assert len(boundaries) == 1
    identity, token = boundaries[0]
    requirements = ("native_scalar_clock_descriptor_requirements" if statement.startswith("call")
                    else "native_intrinsic_output_requirements")
    assert token.public()[requirements]
    original, = graph.source_nodes(identity)
    assert proof.descriptor_for((original,)).validate(analysis, selection._routine.qualified, (identity,))
    assert proof.public()["execution_authorized"] is False
    assert proof.public()["payload_access_authorized"] is False


@pytest.mark.parametrize("statement", ["call unknown_escape(a)", "deallocate(a)",
                                      "write(*,*,iostat=rate) 'unsafe output'"])
def test_native_boundary_classification_cannot_replace_storage_or_output_authority(tmp_path, statement):
    source = SOURCE.replace("integer,save::called=0", "integer,save::called=0\ninteger::rate=0").replace(
        "call selected(a,b,n)", statement+"\ncall selected(a,b,n)")
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, source))


@pytest.mark.parametrize("mutation", ["extra_call", "allocation", "unsafe_tail"])
def test_additional_unproved_edge_cannot_be_hidden_behind_the_one_barrier(tmp_path, mutation):
    source = SOURCE
    if mutation == "extra_call":
        source = source.replace("call finish(a,b,n)", "call unknown_escape(a)\ncall finish(a,b,n)")
    elif mutation == "allocation":
        source = source.replace("call finish(a,b,n)", "deallocate(b)\ncall finish(a,b,n)")
    else:
        source = source.replace("b(1:n)=a(1:n)", "b(1:n)=a(1:n)\ncall unknown_escape(b)")
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, source))


@pytest.mark.parametrize("mutation", ["token", "registry", "component", "source", "namespace"])
def test_composition_cannot_retag_mutated_source_or_component_registry(tmp_path, mutation):
    parts = fixture(tmp_path, sealed=False)
    path, analysis, selection, _gate, _resources = parts
    proof = issue(parts)
    if mutation == "token":
        proof = copy(proof)
    elif mutation == "registry":
        analysis._supported_path_storage.clear()
    elif mutation == "component":
        analysis._descriptor_window_proofs.clear()
    elif mutation == "source":
        path.write_text(path.read_text()+"! changed\n")
    else:
        selection._routine.scope.imports["changed"] = ("other", "x")
    with pytest.raises(CompilationError):
        proof.validate()


def test_unrelated_or_copied_gate_does_not_authorize_another_original_path(tmp_path):
    parts = fixture(tmp_path)
    _path, analysis, selection, gate, resources = parts
    with pytest.raises(CompilationError):
        prove_supported_path_storage(analysis, selection, copy(gate), resources)


@pytest.mark.parametrize("position", ["prefix", "suffix"])
@pytest.mark.parametrize("effect", ["escape", "mutation"])
def test_unselected_prefix_or_tail_cannot_expose_or_replace_private_allocations(tmp_path, position, effect):
    operation = "call unknown_escape(a)" if effect == "escape" else "deallocate(a)"
    if position == "prefix":
        source = SOURCE.replace("called=called+1\n", "called=called+1\n"+operation+"\n")
    else:
        source = SOURCE.replace("call finish(a,b,n)\nend subroutine", "call finish(a,b,n)\n"+operation+"\nend subroutine")
    # Explicitly keep the original selected Pre→Post interval; the new work
    # is outside it and still must be checked by the whole-frame composition.
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["supported_storage::advance"]
    calls = [node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt"]
    first = next(node for node in calls if str(node.items[0]).lower() == "produce")
    last = next(node for node in calls if str(node.items[0]).lower() == "finish")
    contents = tuple(routine.execution.content)
    originals = contents[contents.index(first):contents.index(last)+1]
    roots = tuple("supported_storage::"+name for name in ("produce", "selected", "finish"))
    selected = prove_supported_path(analysis, routine, originals, roots)
    barriers = tuple(node for node in calls if str(node.items[0]).lower() == "unavailable")
    gate = prove_allocation_gate(analysis, routine, routine.scope.bindings["eligible"],
        [routine.scope.bindings[name] for name in ("a", "b")], barriers)
    with pytest.raises(CompilationError):
        prove_supported_path_storage(analysis, selected, gate,
            tuple("supported_storage::advance::"+name for name in ("a", "b")))


def shared_fixture(tmp_path, *, effect="", outside=False):
    source = SOURCE.replace("implicit none\n", "implicit none\nreal(8),allocatable::shared(:)\n", 1)
    source = source.replace("call unknown_escape(a)", "b=a"+("\n"+effect if effect and not outside else ""))
    if outside:
        source = source.replace("call finish(a,b,n)\nend subroutine", "call finish(a,b,n)\n"+effect+"\nend subroutine")
    # The ordinary fixture's closed interval intentionally excludes this tail.
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    analysis.seal_session()
    routine = analysis.routines["supported_storage::advance"]
    barriers = tuple(node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt"
                     and str(node.items[0]).lower() == "unavailable")
    gate = prove_allocation_gate(analysis, routine, routine.scope.bindings["eligible"],
        [routine.scope.bindings[name] for name in ("a", "b")], barriers)
    resources = tuple("supported_storage::advance::"+name for name in ("a", "b"))
    contents = tuple(routine.execution.content)
    first = next(node for node in contents if type(node).__name__ == "Call_Stmt" and str(node.items[0]).lower() == "produce")
    last = next(node for node in contents if type(node).__name__ == "Call_Stmt" and str(node.items[0]).lower() == "finish")
    selection = prove_supported_path(analysis, routine, contents[contents.index(first):contents.index(last)+1],
        tuple("supported_storage::"+name for name in ("produce", "selected", "finish")))
    return path, analysis, selection, gate, resources


def test_excluded_edge_independently_covers_unused_shared_descriptor_guards(tmp_path):
    _path, analysis, selected, gate, resources = shared_fixture(tmp_path)
    proof = prove_supported_path_storage(analysis, selected, gate, resources,
                                         module_roots=("supported_storage::shared",))
    barrier, = proof.public()["barred_calls"]
    assert barrier["exposure_resources"] == ["supported_storage::shared"]
    assert barrier["exposure_descriptor_identity"] in analysis._descriptor_window_proofs
    assert proof._exposures[0][1] is analysis._descriptor_window_proofs[barrier["exposure_descriptor_identity"]]
    assert all("supported_storage::shared" in {item["resource"] for item in component.public()["resources"]}
               for _node, component in proof._outside)


@pytest.mark.parametrize("effect", ["deallocate(shared)", "call unknown_export(shared)"])
@pytest.mark.parametrize("outside", [False, True])
def test_private_gate_cannot_discharge_shared_input_exposure_inside_barrier_or_outside_path(tmp_path, effect, outside):
    _path, analysis, selected, gate, resources = shared_fixture(tmp_path, effect=effect, outside=outside)
    with pytest.raises(CompilationError):
        prove_supported_path_storage(analysis, selected, gate, resources,
                                     module_roots=("supported_storage::shared",))


def test_native_consumer_reissues_exact_local_and_descriptor_tokens_without_mutating_parent_registry(tmp_path):
    from compiler.scopes.segments import _native_analysis

    _path, analysis, selected, gate, resources = fixture(tmp_path)
    proof = prove_supported_path_storage(analysis, selected, gate, resources)
    parent_local = dict(analysis._local_allocation_proofs)
    parent_descriptor = dict(analysis._descriptor_window_proofs)
    native = _native_analysis(analysis)
    assert native._local_allocation_proofs == {}
    component = proof.descriptor_for((selected._originals[0],), analysis=native)
    assert component._analysis is native
    assert native._local_allocation_proofs
    assert all(token._analysis is native for token in native._local_allocation_proofs.values())
    assert analysis._local_allocation_proofs == parent_local
    assert analysis._descriptor_window_proofs == parent_descriptor
    for identity, token in parent_local.items():
        assert analysis._local_allocation_proofs[identity] is token
    with pytest.raises(CompilationError):
        component.validate(analysis, component.procedure, component._selected)


def test_registered_composition_cannot_be_rekeyed_under_a_false_content_identity(tmp_path):
    parts = fixture(tmp_path)
    proof = issue(parts)
    object.__setattr__(proof, "identity", "0"*64)
    parts[1]._supported_path_storage[proof.identity] = proof
    with pytest.raises(CompilationError):
        proof.validate()


def test_registry_hit_cannot_lend_an_original_composition_to_a_foreign_analysis_projection(tmp_path):
    _source, analysis, path, gate, resources = fixture(tmp_path)
    proof = prove_supported_path_storage(analysis, path, gate, resources)
    foreign = analysis.private_projection()
    assert foreign._supported_path_storage[proof.identity] is proof
    with pytest.raises(CompilationError, match="original source analysis"):
        prove_supported_path_storage(foreign, path, gate, resources)


@pytest.mark.parametrize("mutation", ["path", "gate", "analysis", "roots", "component_registry"])
def test_final_publication_cannot_replace_composition_attachments_with_equal_public_values(tmp_path, monkeypatch, mutation):
    parts = fixture(tmp_path)
    proof = issue(parts)
    analysis = parts[1]
    original = analysis.verify_publication

    def publish():
        original()
        frame = sys._getframe(1)
        if frame.f_code.co_name != "_derive" or not frame.f_code.co_filename.endswith("supported_path_storage.py"):
            return
        if mutation in {"path", "gate"}:
            object.__setattr__(proof, "_"+mutation, copy(getattr(proof, "_"+mutation)))
        elif mutation == "analysis":
            object.__setattr__(proof, "_analysis", analysis.private_projection())
        elif mutation == "roots":
            object.__setattr__(proof, "_roots", tuple(list(proof._roots)))
        else:
            token = proof._components[0][1]
            analysis._descriptor_window_proofs[token.identity] = copy(token)

    monkeypatch.setattr(analysis, "verify_publication", publish)
    with pytest.raises(CompilationError):
        proof.validate()


def test_issuer_cannot_publish_after_final_callback_detaches_its_source_gate(tmp_path, monkeypatch):
    parts = fixture(tmp_path)
    analysis, gate = parts[1], parts[3]
    original = analysis.verify_publication

    def publish():
        original()
        frame = sys._getframe(1)
        if frame.f_code.co_name == "_derive" and frame.f_code.co_filename.endswith("supported_path_storage.py"):
            object.__setattr__(gate, "_analysis", analysis.private_projection())

    monkeypatch.setattr(analysis, "verify_publication", publish)
    with pytest.raises(CompilationError, match="final publication"):
        issue(parts)


def test_later_gate_validation_cannot_reseed_a_changed_path_attachment(tmp_path, monkeypatch):
    parts = fixture(tmp_path)
    path, gate = parts[2:4]
    original = type(gate).validate

    def validate(self):
        result = original(self)
        if self is gate:
            object.__setattr__(path, "_originals", tuple(list(path._originals)))
        return result

    monkeypatch.setattr(type(gate), "validate", validate)
    with pytest.raises(CompilationError, match="final publication"):
        issue(parts)


def test_later_descriptor_validation_cannot_reseed_an_earlier_consumed_token(tmp_path, monkeypatch):
    parts = fixture(tmp_path)
    proof = issue(parts)
    first, later = proof._components[0][1], proof._components[-1][1]
    original = type(first).validate
    first_seen = False

    def validate(self, *args, **kwargs):
        nonlocal first_seen
        result = original(self, *args, **kwargs)
        if self is first:
            first_seen = True
        if self is later and first_seen:
            object.__setattr__(first, "_selected", tuple(list(first._selected)))
        return result

    monkeypatch.setattr(type(first), "validate", validate)
    with pytest.raises(CompilationError, match="final publication"):
        proof.validate()

