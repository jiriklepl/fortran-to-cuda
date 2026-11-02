"""Native CALL coordinates never supply write, capture or managed shape credit."""
from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fparser.two import Fortran2003 as F

from compiler.frontend.call_bindings import _fixed_array_bounds, _function_array_mapping
from compiler.frontend.native_fixed_array_association import (
    NATIVE_FIXED_ARRAY_ASSOCIATION_VERSION,
    prove_native_fixed_array_association,
)
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.source import ScopeBuilder


def fixture(tmp_path, *, formal="real(8),intent(out)::field(1:3)", actual="real(8)::a(-2:0)",
            argument="a", calls=None, body="field=0", extra="", seal=True):
    path = tmp_path / "fixed_calls.f90"
    path.write_text(f"""module fixed_calls
implicit none
{extra}
contains
subroutine leaf(field,n)
integer,intent(in)::n
{formal}
{body}
end subroutine
subroutine step(n,enabled)
integer,intent(in)::n
logical,intent(in)::enabled
{actual}
{calls or 'call leaf(n=n,field=' + argument + ')'}
end subroutine
end module
""")
    analysis = SourceEffects([path])
    if seal:
        analysis.seal_session()
    owner = analysis.routines["fixed_calls::step"]
    selections = tuple(node for node in owner.execution.content if type(node).__name__ in {"Call_Stmt", "If_Stmt"})
    return path, analysis, owner, selections


def prove(analysis, owner, selection, formal="field"):
    return prove_native_fixed_array_association(analysis, owner.qualified, (selection,), formal)


@pytest.mark.parametrize("intent", ["in", "out", "inout"])
@pytest.mark.parametrize("dtype", ["real(4)", "real(8)", "integer(4)", "integer(8)", "logical(4)", "logical(8)"])
def test_exact_whole_fixed_shape_retains_physical_lower_bounds_without_effect_credit(tmp_path, intent, dtype):
    path, analysis, owner, (selection,) = fixture(tmp_path,
        formal=f"{dtype},intent({intent})::field(1:3)", actual=f"{dtype}::a(-2:0)", body="")
    before = path.read_bytes()
    summary = analysis.summarize(owner.qualified)
    token = prove(analysis, owner, selection)
    assert token.validate(analysis, owner.qualified, (selection,), "field") is token
    assert prove(analysis, owner, selection) is token
    assert token.formal is analysis.routines["fixed_calls::leaf"].scope.bindings["field"]
    assert token.actual is owner.scope.bindings["a"]
    assert token.formal_bounds == (1, 3)
    assert token.actual_bounds == (-2, 0)
    assert token.element_count == 3
    assert token.index_offset == -3
    record = token.public()
    assert record["schema_version"] == NATIVE_FIXED_ARRAY_ASSOCIATION_VERSION == 1
    assert record["proof_role"] == "native_whole_fixed_array_call_association"
    assert record["source_call"]["original_arguments"][0]["formal"] == "n"
    assert record["source_call"]["original_arguments"][1]["formal"] == "field"
    assert record["original_selection"] == str(selection)
    assert record["policy"]["element_limit"] is None
    assert record["requires_original_execution"] is True
    assert not any(value for key, value in record.items() if key.endswith(("authority", "authorized", "established")))
    assert analysis.summarize(owner.qualified) == summary
    assert path.read_bytes() == before


def test_parameter_bounds_use_their_separate_original_scopes_and_module_storage(tmp_path):
    path = tmp_path / "renamed.f90"
    path.write_text("""module library
integer,parameter::lo=-1,hi=1
contains
subroutine leaf(field)
real(8),intent(out)::field(lo:hi)
field=0
end subroutine
end module
module client
use library,only:renamed=>leaf
integer,parameter::lo=7,hi=9
real(8)::array(lo:hi)=0
contains
subroutine step()
call renamed(array)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    analysis.seal_session()
    owner = analysis.routines["client::step"]
    selection = owner.execution.content[0]
    token = prove(analysis, owner, selection)
    assert token.actual is analysis.modules["client"].bindings["array"]
    assert token.formal_bounds == (-1, 1)
    assert token.actual_bounds == (7, 9)
    assert token.index_offset == 8
    assert token.public()["source_call"]["target"] == "renamed"


def test_whole_if_predicate_and_original_call_site_are_inseparable(tmp_path):
    _path, analysis, owner, selections = fixture(tmp_path,
        calls="if(enabled) call leaf(a,n)\nif(.not.enabled) call leaf(a,n)\ncall leaf(a,n)")
    tokens = [prove(analysis, owner, selection) for selection in selections]
    assert len({token.identity for token in tokens}) == 3
    assert tokens[0].public()["original_selection"] == str(selections[0])
    assert tokens[0].public()["original_call"] == str(selections[0].items[1])
    with pytest.raises(CompilationError, match="complete original IF predicate"):
        prove(analysis, owner, selections[0].items[1])
    for foreign in (selections[1], selections[2], F.If_Stmt("if(.false.) call leaf(a,n)")):
        with pytest.raises(CompilationError, match="exact issued source selection"):
            tokens[0].validate(analysis, owner.qualified, (foreign,), "field")


@pytest.mark.parametrize(("formal", "actual", "argument"), [
    ("real(8),intent(out)::field(3)", "real(8)::a(4)", "a"),
    ("real(8),intent(out)::field(n)", "real(8)::a(3)", "a"),
    ("real(8),intent(out)::field(3)", "real(8)::a(n)", "a"),
    ("real(8),intent(out)::field(*)", "real(8)::a(3)", "a"),
    ("real(8),intent(out)::field(:)", "real(8)::a(3)", "a"),
    ("real(8),intent(out)::field(3)", "real(8)::a(257)", "a"),
    ("real(8),intent(out)::field(3)", "real(8)::a(3)", "a(:)"),
    ("real(8),intent(out)::field(3)", "real(8)::a(5)", "a(2:4)"),
    ("real(8),intent(out)::field(3)", "real(8)::a(3)", "a([1,2,3])"),
    ("real(8),intent(in)::field(3)", "real(8)::a(3)", "[1.d0,2.d0,3.d0]"),
    ("real(8),intent(in)::field(3)", "real(8)::a(3)", "(a)"),
    ("real(8),intent(out)::field(3)", "real(4)::a(3)", "a"),
    ("integer,intent(out)::field(3)", "real(8)::a(3)", "a"),
    ("real(8),intent(out)::field(3)", "real(8)::a(3,1)", "a"),
    ("real(8),intent(out)::field(3,1)", "real(8)::a(3,1)", "a"),
])
def test_sections_temporaries_runtime_extents_and_signature_mismatches_reject(tmp_path, formal, actual, argument):
    _path, analysis, owner, (selection,) = fixture(tmp_path, formal=formal, actual=actual, argument=argument)
    with pytest.raises(CompilationError):
        prove(analysis, owner, selection)
    assert not getattr(analysis, "_native_fixed_array_associations", {})


@pytest.mark.parametrize("attribute", ["optional", "pointer", "allocatable", "target", "volatile", "asynchronous", "value"])
@pytest.mark.parametrize("side", ["actual", "formal"])
def test_special_storage_association_is_not_inferred(tmp_path, attribute, side):
    formal = f"real(8),intent(out){',' + attribute if side == 'formal' else ''}::field(3)"
    actual = f"real(8){',' + attribute if side == 'actual' else ''}::a(3)"
    _path, analysis, owner, (selection,) = fixture(tmp_path, formal=formal, actual=actual)
    with pytest.raises(CompilationError):
        prove(analysis, owner, selection)


@pytest.mark.parametrize("mutation", ["copy", "registered_copy", "registry", "foreign_analysis", "call", "selection_list", "formal",
                                      "record", "bounds", "actual_binding", "call_receipt", "budget", "policy",
                                      "source", "issuer"])
def test_exact_identity_source_policy_and_publication_are_rechecked(tmp_path, mutation, monkeypatch):
    path, analysis, owner, (selection,) = fixture(tmp_path)
    token = prove(analysis, owner, selection)
    target_analysis, target_selection, target_formal = analysis, (selection,), "field"
    if mutation == "copy":
        token = copy(token)
    elif mutation == "registered_copy":
        token = copy(token)
        analysis._native_fixed_array_associations[token.identity] = token
    elif mutation == "registry":
        analysis._native_fixed_array_associations[token.identity] = replace(token)
    elif mutation == "foreign_analysis":
        target_analysis = SourceEffects([path])
        target_analysis.seal_session()
    elif mutation == "call":
        target_selection = (F.Call_Stmt(str(selection)),)
    elif mutation == "selection_list":
        target_selection = [selection]
    elif mutation == "formal":
        target_formal = "n"
    elif mutation == "record":
        object.__setattr__(token, "_record", "{}")
    elif mutation == "bounds":
        object.__setattr__(token, "actual_bounds", (1, 3))
    elif mutation == "actual_binding":
        object.__setattr__(token, "actual", token.formal)
    elif mutation == "call_receipt":
        analysis._original_call_associations[id(analysis), id(selection)] = copy(token._resolved)
    elif mutation == "budget":
        analysis.operation_limit += 1
    elif mutation == "policy":
        monkeypatch.setattr("compiler.frontend.native_fixed_array_association.NATIVE_FIXED_ARRAY_ASSOCIATION_VERSION", 2)
    elif mutation == "source":
        path.write_text(path.read_text() + "! changed after source sealing\n")
    else:
        object.__setattr__(token, "_issuer", object())
    with pytest.raises(CompilationError):
        token.validate(target_analysis, owner.qualified, target_selection, target_formal)


def test_publication_rechecks_proof_registry_after_live_source_validation(tmp_path, monkeypatch):
    _path, analysis, owner, (selection,) = fixture(tmp_path)
    token = prove(analysis, owner, selection)
    verify = analysis.verify_publication

    def replace_registry():
        verify()
        analysis._native_fixed_array_associations[token.identity] = copy(token)

    monkeypatch.setattr(analysis, "verify_publication", replace_registry)
    with pytest.raises(CompilationError, match="registry changed before publication"):
        token.validate(analysis, owner.qualified, (selection,), "field")


def test_unsealed_and_unknown_formal_cannot_issue_a_token(tmp_path):
    _path, analysis, owner, (selection,) = fixture(tmp_path, seal=False)
    with pytest.raises(CompilationError, match="sealed source session"):
        prove(analysis, owner, selection)
    analysis.seal_session()
    with pytest.raises(CompilationError, match="present array formal"):
        prove(analysis, owner, selection, "missing")
    with pytest.raises(CompilationError, match="rank-one"):
        prove(analysis, owner, selection, "n")


def test_association_cannot_complete_effects_or_supply_companion_or_immutable_shape_authority(tmp_path):
    _path, analysis, owner, (selection,) = fixture(tmp_path, body="call unresolved(field)")
    before = analysis.summarize(owner.qualified)
    assert not before["complete"]
    token = prove(analysis, owner, selection)
    callee = analysis.routines["fixed_calls::leaf"]
    checker = SimpleNamespace(analysis=analysis)
    with pytest.raises(CompilationError, match="explicit dummy extents"):
        ScopeBuilder.check_whole_view_formals(checker, callee)
    with pytest.raises(CompilationError, match="original association proof"):
        ScopeBuilder.check_whole_view_formals(checker, callee, immutable_inputs={token.formal.root: token})
    assert analysis.summarize(owner.qualified) == before
    assert not getattr(analysis, "_native_functions", {})
    assert not getattr(analysis, "_descriptor_functions", {})
    record = token.public()
    record["payload_definition_authority"] = True
    assert token.public()["payload_definition_authority"] is False


def test_immutable_actual_is_only_an_input_association_and_never_an_output_definition(tmp_path):
    _path, analysis, owner, (selection,) = fixture(tmp_path,
        formal="real(8),intent(in)::field(3)", actual="real(8),parameter::a(3)=0")
    token = prove(analysis, owner, selection)
    assert "parameter" in token.actual.attributes
    assert token.public()["entry_definition_authority"] is False
    other = tmp_path / "out"
    other.mkdir()
    _path, analysis, owner, (selection,) = fixture(other, actual="real(8),parameter::a(3)=0")
    with pytest.raises(CompilationError, match="ordinary whole rank-one"):
        prove(analysis, owner, selection)


def test_equal_empty_arrays_retain_declared_lower_origins_without_payload(tmp_path):
    _path, analysis, owner, (selection,) = fixture(tmp_path,
        formal="real(8),intent(out)::field(4:3)", actual="real(8)::a(-1:-2)")
    token = prove(analysis, owner, selection)
    assert token.element_count == 0
    assert token.formal_bounds == (4, 3)
    assert token.actual_bounds == (-1, -2)
    assert token.index_offset == -5


@pytest.mark.parametrize("count", [257, 1000000, 2147483647])
def test_metadata_association_does_not_enumerate_elements_or_lend_larger_function_workspace(tmp_path, count):
    _path, analysis, owner, (selection,) = fixture(tmp_path,
        formal=f"real(8),intent(out)::field(1:{count})", actual=f"real(8)::a(-3:{count - 4})")
    token = prove(analysis, owner, selection)
    assert token.element_count == count
    assert token.index_offset == -4
    assert token.public()["policy"]["element_limit"] is None
    # These existing function consumers still retain their independent work cap.
    with pytest.raises(CompilationError, match="256-element fixed storage budget"):
        _fixed_array_bounds(analysis, token.actual)
    with pytest.raises(CompilationError, match="256-element fixed storage budget"):
        _function_array_mapping(analysis, token.formal, token.actual, None)


@pytest.mark.parametrize("bounds", ["1:2147483648", "-2147483647:2147483647"])
def test_metadata_only_bounds_and_extent_still_require_integer_abi(tmp_path, bounds):
    _path, analysis, owner, (selection,) = fixture(tmp_path,
        formal=f"real(8),intent(out)::field({bounds})", actual=f"real(8)::a({bounds})")
    with pytest.raises(CompilationError):
        prove(analysis, owner, selection)


@pytest.mark.parametrize("limit", [True, 0, -1, "256"])
def test_helper_work_budget_requires_exact_nonnegative_context_input(tmp_path, limit):
    _path, analysis, owner, (_selection,) = fixture(tmp_path)
    with pytest.raises(CompilationError, match="positive exact integer or None"):
        _fixed_array_bounds(analysis, owner.scope.bindings["a"], element_limit=limit)


def test_registered_copy_cannot_become_an_issued_handle_through_factory_lookup(tmp_path):
    _path, analysis, owner, (selection,) = fixture(tmp_path)
    token = prove(analysis, owner, selection)
    analysis._native_fixed_array_associations = {token.identity: copy(token)}
    with pytest.raises(CompilationError, match="exact arena-issued handle"):
        prove(analysis, owner, selection)


def test_copying_the_registry_preserves_only_genuine_existing_handles(tmp_path):
    _path, analysis, owner, (selection,) = fixture(tmp_path)
    token = prove(analysis, owner, selection)
    analysis._native_fixed_array_associations = analysis._native_fixed_array_associations.copy()
    assert token.validate(analysis, owner.qualified, (selection,), "field") is token
    assert prove(analysis, owner, selection) is token


@pytest.mark.parametrize("copy_handle", [False, True])
def test_discarded_generation_handle_cannot_be_revived_by_copying_its_registry(tmp_path, copy_handle):
    _path, analysis, owner, (selection,) = fixture(tmp_path)
    context = analysis.source_authority_context()
    parent_arena = context.derivation_arena(analysis)
    parent = SimpleNamespace(analysis=analysis, entry=owner, _check_reached_publication=analysis.verify_publication)
    with GenerationTransaction(parent):
        token = prove(analysis, owner, selection)
        assert token.validate(analysis, owner.qualified, (selection,), "field") is token
        child_registry = analysis._native_fixed_array_associations.copy()
    assert context.derivation_arena(analysis) is parent_arena
    if copy_handle:
        child_registry[token.identity] = copy(token)
    analysis._native_fixed_array_associations = child_registry
    with pytest.raises(CompilationError, match="discarded generation arena"):
        child_registry[token.identity].validate(analysis, owner.qualified, (selection,), "field")
    with pytest.raises(CompilationError, match="discarded generation arena"):
        prove(analysis, owner, selection)


def test_original_issued_handle_survives_registry_detachment_and_rejected_generation(tmp_path):
    _path, analysis, owner, (selection,) = fixture(tmp_path)
    token = prove(analysis, owner, selection)
    registry = analysis._native_fixed_array_associations
    parent = SimpleNamespace(analysis=analysis, entry=owner, _check_reached_publication=analysis.verify_publication)
    with GenerationTransaction(parent):
        assert analysis._native_fixed_array_associations is not registry
        assert prove(analysis, owner, selection) is token
    assert analysis._native_fixed_array_associations is registry
    assert prove(analysis, owner, selection) is token


@pytest.mark.parametrize("mutation", ["replace", "delete", "own"])
def test_issuance_failure_preserves_original_exception_and_only_removes_own_entry(tmp_path, mutation, monkeypatch):
    _path, analysis, owner, (selection,) = fixture(tmp_path)
    retained = {}

    def fail_publication():
        registry = analysis._native_fixed_array_associations
        identity, provisional = next(iter(registry.items()))
        retained.update(identity=identity, provisional=provisional)
        if mutation == "replace":
            retained["replacement"] = copy(provisional)
            registry[identity] = retained["replacement"]
        elif mutation == "delete":
            del registry[identity]
        raise RuntimeError("original publication failure")

    monkeypatch.setattr(analysis, "verify_publication", fail_publication)
    with pytest.raises(RuntimeError, match="original publication failure"):
        prove(analysis, owner, selection)
    registry = analysis._native_fixed_array_associations
    if mutation == "replace":
        assert registry[retained["identity"]] is retained["replacement"]
    else:
        assert retained["identity"] not in registry
    arena = analysis.source_authority_context().derivation_arena(analysis)
    with pytest.raises(CompilationError, match="exact arena-issued handle"):
        arena.require_handle(("native_fixed_array_association", id(analysis), retained["identity"]), retained["provisional"])


def test_provisional_proof_is_not_an_issued_handle_and_failure_can_retry_cleanly(tmp_path, monkeypatch):
    _path, analysis, owner, (selection,) = fixture(tmp_path)
    verify = analysis.verify_publication
    retained = []

    def reject_provisional():
        verify()
        provisional, = analysis._native_fixed_array_associations.values()
        retained.append(provisional)
        with pytest.raises(CompilationError, match="exact arena-issued handle"):
            provisional.validate(analysis, owner.qualified, (selection,), "field")
        raise RuntimeError("no provisional publication")

    monkeypatch.setattr(analysis, "verify_publication", reject_provisional)
    with pytest.raises(RuntimeError, match="no provisional publication"):
        prove(analysis, owner, selection)
    monkeypatch.setattr(analysis, "verify_publication", verify)
    token = prove(analysis, owner, selection)
    assert token is not retained[0]
    analysis._native_fixed_array_associations[token.identity] = retained[0]
    with pytest.raises(CompilationError, match="exact arena-issued handle"):
        retained[0].validate(analysis, owner.qualified, (selection,), "field")


@pytest.mark.parametrize("phase", ["issuance", "validation"])
@pytest.mark.parametrize("mutation", ["actual", "public_role", "call_receipt", "identity", "context", "arena"])
def test_final_publication_callback_cannot_change_checked_proof_attachments(tmp_path, monkeypatch, phase, mutation):
    _path, analysis, owner, (selection,) = fixture(tmp_path)
    token = prove(analysis, owner, selection) if phase == "validation" else None
    verify = analysis.verify_publication
    retained = []

    def mutate_attachment():
        verify()
        provisional, = analysis._native_fixed_array_associations.values()
        retained.append((provisional.identity, provisional))
        if mutation == "actual":
            object.__setattr__(provisional, "actual", provisional.formal)
        elif mutation == "public_role":
            edited = provisional._record.replace('"payload_access_authorized":false', '"payload_access_authorized":true')
            assert edited != provisional._record
            object.__setattr__(provisional, "_record", edited)
        elif mutation == "call_receipt":
            object.__setattr__(provisional, "_resolved", copy(provisional._resolved))
        elif mutation == "identity":
            object.__setattr__(provisional, "identity", "edited-during-publication")
        elif mutation == "context":
            object.__setattr__(provisional, "_context", object())
        else:
            object.__setattr__(provisional, "_arena", provisional._context.arena)

    monkeypatch.setattr(analysis, "verify_publication", mutate_attachment)
    def checked_demand():
        if token is None:
            return prove(analysis, owner, selection)
        return token.validate(analysis, owner.qualified, (selection,), "field")
    with pytest.raises(CompilationError, match="changed before publication"):
        checked_demand()
    assert retained
    if phase == "issuance":
        original_identity, provisional = retained[-1]
        assert original_identity not in analysis._native_fixed_array_associations
        # Cleanup uses the retained original key even if the callback edited identity.
        arena = analysis.source_authority_context().derivation_arena(analysis)
        with pytest.raises(CompilationError, match="exact arena-issued handle"):
            arena.require_handle(("native_fixed_array_association", id(analysis), original_identity), provisional)


@pytest.mark.parametrize("phase", ["issuance", "validation"])
def test_final_registry_copy_is_not_a_publication_attachment(tmp_path, monkeypatch, phase):
    _path, analysis, owner, (selection,) = fixture(tmp_path)
    token = prove(analysis, owner, selection) if phase == "validation" else None
    verify = analysis.verify_publication
    replacements = []

    def replace_registry():
        verify()
        replacement = analysis._native_fixed_array_associations.copy()
        analysis._native_fixed_array_associations = replacement
        replacements.append(replacement)

    monkeypatch.setattr(analysis, "verify_publication", replace_registry)
    def checked_demand():
        if token is None:
            return prove(analysis, owner, selection)
        return token.validate(analysis, owner.qualified, (selection,), "field")
    with pytest.raises(CompilationError, match="attachments changed before publication"):
        checked_demand()
    assert analysis._native_fixed_array_associations is replacements[-1]
    assert replacements[-1]  # Failure cleanup does not delete the replacement entry.
