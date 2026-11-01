"""One fresh storage derivation, followed by its original publication fence."""
from copy import copy

import pytest

from compiler.frontend import storage_mutation_effects as storage
from compiler.frontend.call_bindings import resolve_original_call
from compiler.ir import CompilationError
from compiler.tests.test_storage_mutation_effects import analysis

HELPER = """subroutine helper(x)
real(8),intent(inout)::x(:)
x(:)=x(:)+1.0_8
end subroutine
"""


@pytest.mark.parametrize("sealed", [False, True])
def test_issuer_derives_once_and_public_validate_always_derives_fresh(tmp_path, monkeypatch, sealed):
    _path, source = analysis(tmp_path, "call helper(a)", helpers=HELPER)
    if sealed:
        source.seal_session()
    derive, calls = storage._derive, []

    def count(*args, **kwargs):
        calls.append(args[1])
        return derive(*args, **kwargs)

    monkeypatch.setattr(storage, "_derive", count)
    first = storage.prove_storage_mutation_effects(source, "storage::advance")
    assert len(calls) == 1
    assert storage.prove_storage_mutation_effects(source, "storage::advance") is first
    assert len(calls) == (1 if sealed else 2)
    assert first.validate() is first
    assert len(calls) == (1 if sealed else 3)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("mutation", ["namespace", "live_source", "input_metadata", "consumed_call", "token_registry", "call_registry"])
def test_pure_comparison_cannot_publish_after_source_or_association_tampering(
        tmp_path, monkeypatch, sealed, existing, mutation):
    path, source = analysis(tmp_path, "call helper(a)", helpers=HELPER)
    if sealed:
        source.seal_session()
    if existing:
        original = storage.prove_storage_mutation_effects(source, "storage::advance")
    routine = source.routines["storage::advance"]
    call = routine.execution.content[0]
    edge = resolve_original_call(source, routine.scope, call)
    freeze = storage._freeze_source_record
    fired = False

    def alter(analysis, record):
        nonlocal fired
        result = freeze(analysis, record)
        if fired:
            return result
        fired = True
        if mutation == "namespace":
            # A private copy has equal source text and no original namespace
            # role. Replace the registry, not the deeply sealed object itself.
            modules = dict(analysis.modules)
            modules["storage"] = copy(analysis.modules["storage"])
            analysis.modules = modules
        elif mutation == "live_source":
            path.write_text(path.read_text() + "! changed after fresh derivation\n")
        elif mutation == "input_metadata":
            inputs = copy(analysis.inputs)
            inputs.configuration = {"after_scan": True}
            analysis.inputs = inputs
        elif mutation == "consumed_call":
            object.__setattr__(edge, "actuals", (copy(edge.actuals[0]),))
        elif mutation == "token_registry":
            analysis._storage_mutation_effects = {}
        else:
            analysis._original_call_associations = {}
        return result

    monkeypatch.setattr(storage, "_freeze_source_record", alter)
    if sealed and existing:
        assert storage.prove_storage_mutation_effects(source, routine.qualified) is original
        assert not fired
        return
    with pytest.raises(CompilationError):
        storage.prove_storage_mutation_effects(source, routine.qualified)
    assert fired
    if not existing:
        assert not getattr(source, "_storage_mutation_effects", {})
    elif mutation != "token_registry":
        assert source._storage_mutation_effects.get(original.identity) is original


def test_reused_selected_query_can_have_a_new_container_but_not_new_original_nodes(tmp_path):
    _path, source = analysis(tmp_path, "a(:)=a(:)+1.0_8")
    routine = source.routines["storage::advance"]
    node = routine.execution.content[0]
    first = storage.prove_storage_mutation_effects(source, routine.qualified, selection=[node])
    assert storage.prove_storage_mutation_effects(source, routine.qualified, selection=[node]) is first
    graph, identities = source._selected_source(routine.qualified, [node])
    assert graph.identity == first.public()["original_selection"]["structured_identity"]
    assert storage.prove_storage_mutation_effects(source, routine.qualified, selection=identities) is first
    with pytest.raises(CompilationError):
        storage.prove_storage_mutation_effects(source, routine.qualified, selection=[copy(node)])


@pytest.mark.parametrize("replacement", ["foreign", "none", "other_original"])
def test_valid_fresh_query_cannot_repair_a_tampered_retained_selection(tmp_path, replacement):
    _path, source = analysis(tmp_path, "a(:)=a(:)+1.0_8\na(:)=a(:)+2.0_8")
    routine = source.routines["storage::advance"]
    node, other = routine.execution.content
    first = storage.prove_storage_mutation_effects(source, routine.qualified, selection=[node])
    retained = (copy(node),) if replacement == "foreign" else None if replacement == "none" else (other,)
    object.__setattr__(first, "_selection", retained)
    with pytest.raises(CompilationError):
        storage.prove_storage_mutation_effects(source, routine.qualified, selection=[node])


@pytest.mark.parametrize("consumer", ["issuer", "validate"])
def test_selection_attachment_cannot_change_during_pure_record_comparison(tmp_path, monkeypatch, consumer):
    _path, source = analysis(tmp_path, "a(:)=a(:)+1.0_8")
    routine = source.routines["storage::advance"]
    node = routine.execution.content[0]
    first = storage.prove_storage_mutation_effects(source, routine.qualified, selection=[node])
    freeze = storage._freeze_source_record

    def alter(analysis, record):
        result = freeze(analysis, record)
        object.__setattr__(first, "_selection", (copy(node),))
        return result

    monkeypatch.setattr(storage, "_freeze_source_record", alter)
    consume = (lambda: storage.prove_storage_mutation_effects(source, routine.qualified, selection=[node])) if consumer == "issuer" else first.validate
    with pytest.raises(CompilationError, match="exact registered source projection"):
        consume()
