"""Receiver declaration composition never supplies caller array lifetime."""
from copy import copy

import pytest

from compiler.frontend.private_library_state import private_library_state
from compiler.ir import CompilationError
from compiler.tests.test_metadata_constructor import SOURCE, proof


def fixture(tmp_path, source=SOURCE):
    parts, constructor = proof(tmp_path, source)
    _path, analysis, routine, _call, _resolved, _assignment, _contract, _qualification = parts
    opaque = constructor._opaque
    return parts, constructor, private_library_state(analysis, routine, opaque.binding, opaque, constructor)


def test_exact_private_receiver_and_once_only_metadata_are_not_array_lifetime(tmp_path):
    parts, constructor, declared = fixture(tmp_path)
    public = declared.public()
    assert declared.validate() is declared
    assert public["metadata_constructor"]["identity"] == constructor.identity
    assert public["opaque_state"]["identity"] == constructor._opaque.identity
    assert public["caller_array_descriptor_authorized"] is False
    assert public["caller_array_noescape_authorized"] is False
    assert public["execution_authorized"] is False
    assert public["entry_definition_authority"] is False
    _path, analysis, routine, *_ = parts
    with pytest.raises(CompilationError):
        private_library_state(analysis, routine, routine.scope.bindings["a"], constructor._opaque, constructor)


@pytest.mark.parametrize("case", ["opaque_copy", "constructor_copy", "wrong_constructor", "opaque_retag",
                                  "constructor_retag", "foreign_analysis", "call_actuals", "call_associations"])
def test_independent_registered_receiver_constructor_and_consumed_closure_are_exact(tmp_path, case):
    parts, constructor, declared = fixture(tmp_path)
    _path, analysis, routine, _node, resolved, *_ = parts
    opaque, target = constructor._opaque, analysis
    if case == "opaque_copy":
        opaque = copy(opaque)
    elif case == "constructor_copy":
        constructor = copy(constructor)
    elif case == "wrong_constructor":
        object.__setattr__(constructor, "_opaque", copy(opaque))
    elif case.endswith("retag"):
        item = opaque if case == "opaque_retag" else constructor
        registry = analysis._opaque_library_state_proofs if case == "opaque_retag" else analysis._metadata_constructor_proofs
        object.__setattr__(item, "identity", "b"*64)
        registry[item.identity] = item
    elif case == "foreign_analysis":
        target = analysis.private_projection()
    elif case == "call_actuals":
        object.__setattr__(resolved, "actuals", ())
    else:
        object.__setattr__(resolved, "associations", tuple(reversed(resolved.associations)))
    with pytest.raises(CompilationError):
        private_library_state(target, routine, declared.binding, opaque, constructor)


@pytest.mark.parametrize("extra", ["call escape(cache)", "print *,cache%count", "cache%count=1"])
def test_native_opaque_observer_or_escape_is_not_a_private_declaration_contract(tmp_path, extra):
    with pytest.raises(CompilationError):
        fixture(tmp_path, SOURCE.replace("call operation(cache,a)", "call operation(cache,a)\n"+extra))


@pytest.mark.parametrize("change", ["token", "actuals", "constructor_routine", "declaration_token"])
def test_late_publication_cannot_replace_an_independent_receiver_token(tmp_path, monkeypatch, change):
    parts, constructor, declared = fixture(tmp_path)
    _path, analysis, *_ = parts
    original = analysis.verify_publication

    def publish():
        original()
        opaque = constructor._opaque
        if change == "token":
            analysis._opaque_library_state_proofs[opaque.identity] = copy(opaque)
        elif change == "actuals":
            object.__setattr__(opaque._resolved, "actuals", ())
        elif change == "constructor_routine":
            object.__setattr__(constructor, "_routine", copy(constructor._routine))
        else:
            object.__setattr__(declared, "_opaque", copy(opaque))

    monkeypatch.setattr(analysis, "verify_publication", publish)
    with pytest.raises(CompilationError):
        declared.validate()


def test_constructor_validation_cannot_reseed_an_already_consumed_receiver_attachment(tmp_path, monkeypatch):
    _parts, constructor, declared = fixture(tmp_path)
    opaque = constructor._opaque
    original = type(constructor).validate

    def validate(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self is constructor:
            object.__setattr__(opaque, "_routine", copy(opaque._routine))
        return result

    monkeypatch.setattr(type(constructor), "validate", validate)
    with pytest.raises(CompilationError):
        declared.validate()
