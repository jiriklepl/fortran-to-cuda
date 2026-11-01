"""BLOCK storage declarations keep exact source scopes and body obligations."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_blocks import original_fixed_block
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module block_source
contains
subroutine run(a)
real(8),intent(inout)::a(:)
integer::counter
counter=1
block
use,intrinsic::iso_fortran_env,only:real64
real(real64)::value
integer::counter
counter=2
value=a(counter)
a(counter)=value
end block
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, *, sealed=False):
    path = tmp_path / "block_source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines["block_source::run"]
    original = next(node for node in walk(routine.execution) if type(node).__name__ == "Block_Construct")
    return path, analysis, routine, original


@pytest.mark.parametrize("sealed", [False, True])
def test_real_original_block_scope_owns_locals_and_shadowing_without_execution_authority(tmp_path, sealed):
    path, analysis, routine, original = fixture(tmp_path, sealed=sealed)
    before = path.read_bytes()
    proof = original_fixed_block(analysis, routine, original)
    scope = analysis._block_scopes[id(original)]
    assert proof._scope is scope
    assert scope.parent is routine.scope
    assert scope.bindings["counter"] is not routine.scope.bindings["counter"]
    assert scope.bindings["counter"].root == scope.qualified + "::counter"
    assert scope.bindings["value"].kind == 8
    assert scope.bindings["value"].declaring_scope is scope
    assert id(scope) in analysis._source_function_scopes
    assert id(scope) in analysis._source_object_scopes
    reference = next(node for node in walk(original) if type(node).__name__ == "Part_Ref")
    assert analysis.source_scope_for(reference) is scope
    record = proof.public()
    assert record["requires_complete_original_body"] is True
    for name in ("entry_definition_authority", "payload_access_authorized", "alias_authority",
                 "descriptor_lifetime_authorized", "execution_authorized", "motion_authorized"):
        assert record[name] is False
    assert analysis._summary_authority()["descriptor_window_block_version"] == 16
    assert proof.validate() is proof
    assert path.read_bytes() == before


@pytest.mark.parametrize("declaration", ["real(8),pointer::value", "real(8),allocatable::value",
    "real(8),target::value", "real(8),save::value", "real(8)::value=0", "real(8)::value(2)",
    "real(16)::value", "type(unknown_type)::value", "real(8)::value\nnamelist /observe/value",
    "real(8)::value\nexternal hidden", "real(8)::value\ncommon /shared/value"])
def test_uncertain_local_lifecycles_arrays_types_and_associations_require_separate_authority(tmp_path, declaration):
    _path, analysis, routine, original = fixture(tmp_path, SOURCE.replace("real(real64)::value", declaration))
    with pytest.raises(CompilationError):
        original_fixed_block(analysis, routine, original)


def test_unknown_body_is_a_complete_independent_consumer_obligation(tmp_path):
    source = SOURCE.replace("a(counter)=value", "call unknown_escape(a)\ndeallocate(a)")
    _path, analysis, routine, original = fixture(tmp_path, source)
    proof = original_fixed_block(analysis, routine, original)
    assert proof.public()["requires_complete_original_body"] is True
    assert analysis.summarize(routine.qualified)["complete"] is False


@pytest.mark.parametrize("change", ["scope", "binding", "kind", "source", "occurrence", "issues"])
@pytest.mark.parametrize("callback", ["scope", "tick"])
def test_callbacks_cannot_reseed_consumed_block_declarations_or_source_occurrences(tmp_path, change, callback):
    _path, analysis, routine, original = fixture(tmp_path)
    scope = analysis._block_scopes[id(original)]
    changed = False

    def inspect(*_args):
        nonlocal changed
        if changed:
            return
        changed = True
        if change == "scope":
            analysis._block_scopes[id(original)] = copy(scope)
        elif change == "binding":
            scope.bindings["value"] = copy(scope.bindings["value"])
        elif change == "kind":
            scope.bindings["value"].kind = 4
        elif change == "source":
            original.content[2] = copy(original.content[2])
        elif change == "occurrence":
            analysis._source_scopes[id(original.content[2])] = routine.scope
        else:
            scope.issues.append("late declaration")

    kwargs = {"scope_validator" if callback == "scope" else "tick": inspect}
    with pytest.raises(CompilationError):
        original_fixed_block(analysis, routine, original, **kwargs)


@pytest.mark.parametrize("change", ["analysis", "routine", "original", "scope", "record"])
def test_retained_block_receipt_keeps_exact_analysis_routine_scope_and_source(tmp_path, change):
    _path, analysis, routine, original = fixture(tmp_path)
    proof = original_fixed_block(analysis, routine, original)
    if change == "record":
        object.__setattr__(proof, "_record", "{}")
    else:
        field = "_" + change
        object.__setattr__(proof, field, copy(getattr(proof, field)))
    with pytest.raises(CompilationError):
        proof.validate()


def test_original_block_registry_does_not_accept_a_copied_issued_token_or_identity_retag(tmp_path):
    _path, analysis, routine, original = fixture(tmp_path)
    proof = original_fixed_block(analysis, routine, original)
    with pytest.raises(CompilationError):
        copy(proof).validate()
    object.__setattr__(proof, "identity", "0" * 64)
    analysis._descriptor_blocks[proof.identity] = proof
    with pytest.raises(CompilationError):
        proof.validate()


@pytest.mark.parametrize("mutation", ["registry", "owner", "declaration"])
def test_final_publication_cannot_replace_the_issuing_registry_or_retained_declaration(tmp_path, monkeypatch, mutation):
    _path, analysis, routine, original = fixture(tmp_path)
    publish = analysis.verify_publication

    def inspect():
        publish()
        registry = getattr(analysis, "_descriptor_blocks", {})
        if not registry:
            return
        proof = next(iter(registry.values()))
        if mutation == "registry":
            registry[proof.identity] = copy(proof)
        elif mutation == "owner":
            analysis._descriptor_blocks_owner = copy(analysis)
        else:
            object.__setattr__(proof, "_declaration", tuple(list(proof._declaration)))

    monkeypatch.setattr(analysis, "verify_publication", inspect)
    with pytest.raises(CompilationError):
        original_fixed_block(analysis, routine, original)


def test_nested_original_blocks_have_distinct_real_local_scopes(tmp_path):
    text = SOURCE.replace("a(counter)=value", "block\ninteger::counter\ncounter=3\na(counter)=value\nend block")
    _path, analysis, routine, original = fixture(tmp_path, text, sealed=True)
    inner = next(node for node in walk(original) if type(node).__name__ == "Block_Construct" and node is not original)
    outer_scope, inner_scope = analysis._block_scopes[id(original)], analysis._block_scopes[id(inner)]
    assert inner_scope.parent is outer_scope
    assert inner_scope.bindings["counter"].root != outer_scope.bindings["counter"].root
    assert original_fixed_block(analysis, routine, original).public()["requires_complete_original_body"] is True
    assert original_fixed_block(analysis, routine, inner)._scope is inner_scope


def test_preexisting_source_index_retag_does_not_supply_original_block_name_resolution(tmp_path):
    _path, analysis, routine, original = fixture(tmp_path)
    reference = next(node for node in walk(original) if type(node).__name__ == "Part_Ref")
    analysis._source_scopes[id(reference)] = routine.scope
    with pytest.raises(CompilationError, match="lexical scope"):
        original_fixed_block(analysis, routine, original)


@pytest.mark.parametrize("sealed", [False, True])
def test_another_genuine_routines_original_block_cannot_be_relabelled(tmp_path, sealed):
    other = SOURCE[SOURCE.index("subroutine run(a)"):SOURCE.index("end subroutine") + len("end subroutine")]
    text = SOURCE.replace("end module", other.replace("run(a)", "other(a)") + "\nend module")
    _path, analysis, routine, _original = fixture(tmp_path, text, sealed=sealed)
    sibling = analysis.routines["block_source::other"]
    original = next(node for node in walk(sibling.execution) if type(node).__name__ == "Block_Construct")
    with pytest.raises(CompilationError, match="original syntax|owning procedure"):
        original_fixed_block(analysis, routine, original)


@pytest.mark.parametrize("sealed", [False, True])
def test_original_block_inside_a_genuine_association_keeps_owning_procedure(tmp_path, sealed):
    text = SOURCE.replace("block\n", "associate(idx=>counter)\nblock\n", 1).replace(
        "end block", "end block\nend associate", 1)
    _path, analysis, routine, original = fixture(tmp_path, text, sealed=sealed)
    proof = original_fixed_block(analysis, routine, original)
    assert proof._scope.parent.node.__class__.__name__.endswith("Associate_Construct")


@pytest.mark.parametrize("callback", ["scope", "tick"])
@pytest.mark.parametrize("mutation", ["declaration", "record", "registry_entry", "registry", "owner"])
def test_repeated_issuer_cannot_reseed_existing_token_before_its_validation(tmp_path, callback, mutation):
    _path, analysis, routine, original = fixture(tmp_path)
    proof = original_fixed_block(analysis, routine, original)
    changed = False

    def inspect(*_args):
        nonlocal changed
        if changed:
            return
        changed = True
        if mutation == "declaration":
            object.__setattr__(proof, "_declaration", tuple(list(proof._declaration)))
        elif mutation == "record":
            object.__setattr__(proof, "_record", "{}")
        elif mutation == "registry_entry":
            analysis._descriptor_blocks[proof.identity] = copy(proof)
        elif mutation == "registry":
            analysis._descriptor_blocks = dict(analysis._descriptor_blocks)
        else:
            analysis._descriptor_blocks_owner = copy(analysis)

    kwargs = {"scope_validator" if callback == "scope" else "tick": inspect}
    with pytest.raises(CompilationError):
        original_fixed_block(analysis, routine, original, **kwargs)
    assert changed


def test_issuer_final_validator_callback_cannot_replace_the_retained_proof_attachment(tmp_path, monkeypatch):
    from compiler.frontend.descriptor_blocks import FixedScalarBlock

    _path, analysis, routine, original = fixture(tmp_path)
    proof = original_fixed_block(analysis, routine, original)
    validate = FixedScalarBlock.validate

    def inspect(self, *args, **kwargs):
        result = validate(self, *args, **kwargs)
        object.__setattr__(self, "_declaration", tuple(list(self._declaration)))
        return result

    monkeypatch.setattr(FixedScalarBlock, "validate", inspect)
    with pytest.raises(CompilationError):
        original_fixed_block(analysis, routine, original)
    assert proof is analysis._descriptor_blocks[proof.identity]


@pytest.mark.parametrize("sealed", [False, True])
def test_same_public_foreign_analysis_cache_token_cannot_answer_an_original_query(tmp_path, sealed):
    path, analysis, routine, original = fixture(tmp_path, sealed=sealed)
    foreign = original_fixed_block(analysis, routine, original)
    consumer = SourceEffects([path])
    if sealed:
        consumer.seal_session()
    current_routine = consumer.routines[routine.qualified]
    current_block = next(node for node in walk(current_routine.execution)
                         if type(node).__name__ == "Block_Construct")
    current = original_fixed_block(consumer, current_routine, current_block)
    assert current.identity == foreign.identity
    consumer._descriptor_blocks[current.identity] = foreign
    with pytest.raises(CompilationError, match="consuming analysis"):
        original_fixed_block(consumer, current_routine, current_block)
