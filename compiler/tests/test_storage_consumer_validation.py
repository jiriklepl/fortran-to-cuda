"""Storage questions authenticate their requested consumer in one fresh pass."""
from copy import copy

import pytest

from compiler.frontend import storage_mutation_effects as storage
from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.ir import CompilationError
from compiler.tests.test_storage_function_effects import fixture


@pytest.mark.parametrize("sealed", [False, True])
def test_contextual_preservation_is_fresh_once_per_consumption(tmp_path, monkeypatch, sealed):
    _path, source, caller, (expression,) = fixture(tmp_path)
    if sealed:
        source.seal_session()
    proof = source.storage_mutation_effects(caller.qualified, expression=expression)
    derive, calls = storage._derive, []

    def counted(*args, **kwargs):
        calls.append(args[1:3])
        return derive(*args, **kwargs)

    monkeypatch.setattr(storage, "_derive", counted)
    for count in (1, 2):
        assert proof.preserves_descriptors(source, caller.qualified, expression)
        assert len(calls) == (0 if sealed else count)
        if calls:
            assert calls[-1] == (caller.qualified, expression)


@pytest.mark.parametrize("wrong", ["analysis", "procedure", "foreign_expression", "other_expression",
                                   "whole_procedure", "selection", "joined_storage"])
def test_contextual_question_rejects_a_different_consumer(tmp_path, wrong):
    _path, source, caller, expressions = fixture(tmp_path, body="a(:)=calculate(a)\na(:)=calculate(a)")
    proof = source.storage_mutation_effects(caller.qualified, expression=expressions[0])
    owner, name, node, kwargs = source, caller.qualified, expressions[0], {}
    if wrong == "analysis":
        owner = copy(source)
    elif wrong == "procedure":
        name = "functions::calculate"
    elif wrong == "foreign_expression":
        node = copy(node)
    elif wrong == "other_expression":
        node = expressions[1]
    elif wrong == "whole_procedure":
        node = None
    elif wrong == "selection":
        kwargs["selection"] = ()
    else:
        kwargs["joined_storage"] = object()
    with pytest.raises(CompilationError, match="exact registered source projection"):
        proof.preserves_descriptors(owner, name, node, **kwargs)


@pytest.mark.parametrize("sealed", [False, True])
def test_window_consumes_one_fresh_preservation_answer(tmp_path, monkeypatch, sealed):
    _path, source, caller, (expression,) = fixture(tmp_path)
    if sealed:
        source.seal_session()
    projection = source.storage_mutation_effects(caller.qualified, expression=expression)
    derive, calls = storage._derive, []

    def counted(*args, **kwargs):
        calls.append(args[1:3])
        return derive(*args, **kwargs)

    monkeypatch.setattr(storage, "_derive", counted)
    _graph, selected = source._selected_source(caller.qualified, tuple(caller.execution.content))
    window = prove_descriptor_window(source, caller.qualified, selected, ("argument::a",))
    assert calls == ([] if sealed else [(caller.qualified, expression)])
    assert window.public()["native_storage_function_requirements"] == [projection.public()]
    assert window.public()["execution_authorized"] is False


@pytest.mark.parametrize("mutation", ["live_source", "registry", "namespace", "association"])
def test_contextual_answer_keeps_the_final_publication_checks(tmp_path, monkeypatch, mutation):
    path, source, caller, (expression,) = fixture(tmp_path)
    proof = source.storage_mutation_effects(caller.qualified, expression=expression)
    freeze, fired = storage._freeze_source_record, False

    def alter(analysis, record):
        nonlocal fired
        result = freeze(analysis, record)
        if not fired:
            fired = True
            if mutation == "live_source":
                path.write_text(path.read_text() + "! changed during comparison\n")
            elif mutation == "registry":
                analysis._storage_mutation_effects = {}
            elif mutation == "namespace":
                analysis.modules = dict(analysis.modules, functions=copy(analysis.modules["functions"]))
            else:
                object.__setattr__(proof._calls[0], "actuals", (copy(proof._calls[0].actuals[0]),))
        return result

    monkeypatch.setattr(storage, "_freeze_source_record", alter)
    with pytest.raises(CompilationError):
        proof.preserves_descriptors(source, caller.qualified, expression)
    assert fired
