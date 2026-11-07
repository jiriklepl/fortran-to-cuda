"""Existing entry definitions survive only an authenticated original prefix."""
import sys
from copy import copy, deepcopy

import pytest

from compiler.frontend.entry_preservation import (
    prove_preserved_entry_definitions,
)
from compiler.ir import CompilationError
from compiler.scopes.call_groups import _candidates
from compiler.tests.test_owner_local_call_groups import SOURCE, fixture


def prepared(tmp_path, prefix="a(:)=a(:)+1.0_8", *, source=None, lifetime="invocation", initialized="whole"):
    source = SOURCE.replace("call unknown_prefix()", prefix) if source is None else source
    path, builder = fixture(tmp_path, source, lifetime=lifetime)
    facts = deepcopy(builder.facts)
    for fact in facts["captures"].values():
        fact["initialized"] = initialized
    originals = next(_candidates(builder))
    bindings = tuple(builder.entry.scope.bindings[name] for name in ("a", "b"))
    return path, builder.analysis, originals, bindings, facts


def prove(parts):
    _path, analysis, originals, bindings, facts = parts
    return prove_preserved_entry_definitions(analysis, "local_groups::run", originals, bindings, facts)


def test_payload_write_preserves_definedness_without_claiming_unchanged_values_or_new_definitions(tmp_path):
    parts = prepared(tmp_path)
    proof = prove(parts)
    assert proof is not None
    assert proof.validate(parts[1], "local_groups::run", parts[2], parts[3], parts[4]) is proof
    assert {binding.root for binding in proof.preserved} == {"argument::a", "argument::b"}
    record = proof.public()
    assert record["definition_source"].startswith("independent original invocation-entry WHOLE")
    for flag in ("new_definition_authority", "execution_authorized", "payload_access_authorized",
                 "descriptor_lifetime_authority", "alias_authority", "cross_exit_retention_authorized"):
        assert record[flag] is False
    record["preserved_resources"].clear()
    assert proof.public()["preserved_resources"]


@pytest.mark.parametrize("prefix", ["call unknown_prefix()", "deallocate(a)", "allocate(a(n))",
                                  "a=2.0_8", "if(flag)call unknown_prefix()"])
def test_unknown_effect_or_descriptor_replacement_does_not_preserve_entry_state(tmp_path, prefix):
    assert prove(prepared(tmp_path, prefix)) is None


@pytest.mark.parametrize(("lifetime", "initialized"), [("selected_window", "whole"), ("invocation", "none"),
                                                   ("selected_window", "none")])
def test_fact_at_selected_position_or_unknown_entry_grants_no_invocation_preservation(tmp_path, lifetime, initialized):
    assert prove(prepared(tmp_path, lifetime=lifetime, initialized=initialized)) is None


KILL = """subroutine kill(x)
real(8),intent(out)::x(:)
x(:)=0.0_8
end subroutine
"""


def test_original_out_entry_kill_is_not_hidden_by_later_full_definition(tmp_path):
    source = SOURCE.replace("call unknown_prefix()", "call kill(a)").replace("end module", KILL+"end module")
    proof = prove(prepared(tmp_path, source=source))
    assert proof is None or not any(binding.root == "argument::a" for binding in proof.preserved)


def test_unreached_sibling_body_is_excluded_but_its_original_predicate_is_retained(tmp_path):
    source = SOURCE.replace("call unknown_prefix()", "a(:)=a(:)+1.0_8")
    source = source.replace("if(flag)then\ncall first", "if(.not.flag)then\ncall kill(a)\nelse\ncall first")
    source = source.replace("end module", KILL+"end module")
    proof = prove(prepared(tmp_path, source=source))
    assert proof is not None
    assert any(binding.root == "argument::a" for binding in proof.preserved)
    assert all("KILL" not in str(node) for node in proof._prefix)
    assert any(type(node).__name__ == "If_Then_Stmt" for node in proof._prefix)


def nested_source(depth):
    source = SOURCE.replace("call unknown_prefix()", "a(:)=a(:)+1.0_8")
    source = source.replace("if(flag)then\ncall first", "if(flag)then\n" * depth + "call first")
    return source.replace("endif\ncall unknown_suffix()", "endif\n" * depth + "call unknown_suffix()")


def test_nested_guard_segments_are_original_ordered_and_never_flattened_into_one_guard(tmp_path):
    parts = prepared(tmp_path, source=nested_source(3))
    proof = prove(parts)
    assert proof is not None
    assert len(proof._segments) == 4
    assert proof.public()["version"] == 2
    assert len(proof.public()["segments"]) == 4
    assert all(proof._preservations[index]._selection == segment
               for index, segment in enumerate(proof._segments))
    assert {binding.root for binding in proof.preserved} == {"argument::a", "argument::b"}
    with pytest.raises(CompilationError, match="one reached guarded segment"):
        parts[1]._selected_source("local_groups::run", proof._prefix)


def test_complete_prior_joined_if_preserves_definedness_through_native_payload_writes(tmp_path):
    prefix = """if(flag)then
!$omp parallel private(i)
!$omp do
do i=1,n
a(i)=a(i)+1.0_8
enddo
!$omp end do nowait
!$omp do
do i=1,n
b(i)=b(i)+2.0_8
enddo
!$omp end do
!$omp end parallel
else
a(:)=a(:)+b(:)
endif"""
    source = SOURCE.replace("logical,intent(in)::flag", "logical,intent(in)::flag\ninteger::i")
    source = source.replace("call unknown_prefix()", prefix)
    proof = prove(prepared(tmp_path, source=source))
    assert proof is not None
    assert type(proof._segments[0][0]).__name__ == "If_Construct"
    assert all(item.complete for item in proof._preservations)
    assert {binding.root for binding in proof.preserved} == {"argument::a", "argument::b"}


def test_more_than_64_original_guard_segments_do_not_become_a_procedure_count_limit(tmp_path, monkeypatch):
    from functools import partial

    from compiler.scopes import source as scope_source

    # Exercise an explicit diagnostic cap independently of source-sized defaults.
    monkeypatch.setattr(scope_source, "SourceEffects", partial(scope_source.SourceEffects, procedures=64))
    parts = prepared(tmp_path, source=nested_source(65))
    proof = prove(parts)
    assert proof is not None
    assert len(proof._segments) == 66
    assert len(proof._segments) > parts[1].procedure_limit
    assert len(parts[1]._native_preservation_proofs) == 66
    assert proof.validate(parts[1], "local_groups::run", parts[2], parts[3], parts[4]) is proof


@pytest.mark.parametrize("mutation", ["segments", "native_proofs", "public_segment"])
def test_segment_or_native_proof_substitution_cannot_rebind_a_registered_token(tmp_path, mutation):
    parts = prepared(tmp_path, source=nested_source(2))
    proof = prove(parts)
    if mutation == "segments":
        object.__setattr__(proof, "_segments", tuple(reversed(proof._segments)))
    elif mutation == "native_proofs":
        object.__setattr__(proof, "_preservations", tuple(reversed(proof._preservations)))
    else:
        record = deepcopy(proof.public())
        record.pop("identity")
        record["segments"][0]["preservation_identity"] = "0" * 64
        import json
        object.__setattr__(proof, "_record", json.dumps(record, sort_keys=True, separators=(",", ":")))
    with pytest.raises(CompilationError):
        proof.validate(parts[1], "local_groups::run", parts[2], parts[3], parts[4])


def test_enclosing_original_out_formal_is_undefined_before_prefix(tmp_path):
    source = SOURCE.replace("call unknown_prefix()", "a(:)=a(:)+1.0_8")
    source = source.replace("real(8),allocatable,intent(inout)::a(:),b(:)", "real(8),allocatable,intent(out)::a(:),b(:)")
    assert prove(prepared(tmp_path, source=source)) is None


@pytest.mark.parametrize("declaration", ["type(payload)", "complex(8)", "character(len=4)"])
def test_non_numeric_array_facts_do_not_enter_the_numeric_preservation_slice(tmp_path, declaration):
    source = SOURCE.replace("module local_groups\ncontains", "module local_groups\n"
                            "type payload\nreal(8)::value\nend type\ncontains")
    source = source.replace("subroutine run(a,b,n,flag)", "subroutine run(a,b,n,flag,objects)")
    source = source.replace("real(8),allocatable,intent(inout)::a(:),b(:)",
                            f"{declaration},allocatable,intent(inout)::objects(:)\n"
                            "real(8),allocatable,intent(inout)::a(:),b(:)")
    _path, analysis, originals, _bindings, facts = prepared(tmp_path, source=source)
    binding = analysis.routines["local_groups::run"].scope.bindings["objects"]
    facts["captures"][binding.root] = dict(facts["captures"]["argument::a"])
    assert prove_preserved_entry_definitions(analysis, "local_groups::run", originals, (binding,), facts) is None


def test_previous_loop_traversal_cannot_reuse_entry_coverage_killed_by_original_out(tmp_path):
    source = SOURCE.replace("logical,intent(in)::flag", "logical,intent(in)::flag\ninteger::stage")
    source = source.replace("call unknown_prefix()\nif(flag)", "do stage=1,2\nif(flag)")
    source = source.replace("endif\ncall unknown_suffix()", "endif\nenddo\ncall unknown_suffix()")
    assert prove(prepared(tmp_path, source=source)) is None


def loop_source(*, suffix="", nested=False):
    source = SOURCE.replace("intent(out)", "intent(inout)")
    source = source.replace("logical,intent(in)::flag", "logical,intent(in)::flag\ninteger::stage,inner")
    source = source.replace("call unknown_prefix()", "a(:)=a(:)+1.0_8\ndo stage=1,n" +
                            ("\ndo inner=1,n" if nested else ""))
    return source.replace("endif\ncall unknown_suffix()", "endif\n" + suffix +
                          ("\nenddo" if nested else "") + "\nenddo\ncall unknown_suffix()")


@pytest.mark.parametrize("nested", [False, True])
def test_complete_loop_invariant_preserves_every_reached_iteration(tmp_path, nested):
    parts = prepared(tmp_path, source=loop_source(nested=nested))
    proof = prove(parts)
    assert proof is not None
    assert {binding.root for binding in proof.preserved} == {"argument::a", "argument::b"}
    assert proof.public()["version"] == 5
    assert len(proof._segments) == 2
    loop, = proof._segments[-1]
    assert type(loop).__name__ == "Block_Nonlabel_Do_Construct"
    assert "call third" in str(loop).lower()  # Includes selected calls and later iterations.
    assert all(token.complete for token in proof._preservations)
    assert proof.validate(parts[1], "local_groups::run", parts[2], parts[3], parts[4]) is proof


@pytest.mark.parametrize("suffix", ["call unknown_iteration_tail()", "deallocate(a)",
                                    "if(.not.flag)call kill(a)", "call kill(a)"])
def test_loop_tail_and_untaken_arm_cannot_reseed_prior_iteration_definitions(tmp_path, suffix):
    source = loop_source(suffix=suffix).replace("end module", KILL + "end module")
    proof = prove(prepared(tmp_path, source=source))
    assert proof is None or "argument::a" not in {binding.root for binding in proof.preserved}


def test_loop_proof_rejects_substituting_only_the_first_iteration_prefix(tmp_path):
    parts = prepared(tmp_path, source=loop_source())
    proof = prove(parts)
    object.__setattr__(proof, "_segments", proof._segments[:-1])
    with pytest.raises(CompilationError):
        proof.validate(parts[1], "local_groups::run", parts[2], parts[3], parts[4])


def test_side_effect_in_original_predicate_cannot_be_omitted_from_prefix(tmp_path):
    helper = """logical function erase(x)
real(8),intent(out)::x(:)
x(:)=0.0_8
erase=.true.
end function
"""
    source = SOURCE.replace("call unknown_prefix()", "a(:)=a(:)+1.0_8")
    source = source.replace("if(flag)then\ncall first", "if(erase(a))then\ncall first")
    source = source.replace("end module", helper+"end module")
    proof = prove(prepared(tmp_path, source=source))
    assert proof is None or not any(binding.root == "argument::a" for binding in proof.preserved)


@pytest.mark.parametrize("factory", ["projection", "native_prefix", "segment"])
def test_private_forks_must_reissue_both_exit_and_entry_definition_tokens(tmp_path, factory):
    _path, analysis, originals, bindings, facts = prepared(tmp_path)
    first = prove_preserved_entry_definitions(analysis, "local_groups::run", originals, bindings, facts)
    if factory == "projection":
        other = analysis.private_projection()
    elif factory == "native_prefix":
        from compiler.frontend.native_prefix import _fork

        other = _fork(analysis, ())
    else:
        from compiler.scopes.segments import _native_analysis

        other = _native_analysis(analysis)
    with pytest.raises(CompilationError):
        first.validate(other, "local_groups::run", originals, bindings, facts)
    assert not getattr(other, "_entry_preservations", {})
    assert not getattr(other, "_original_sequence_exits", {})
    if factory == "projection":
        # A raw projection intentionally leaves other native cache ownership
        # to its caller; only the new source-token registries are reset here.
        return
    second = prove_preserved_entry_definitions(other, "local_groups::run", originals, bindings, facts)
    assert second is not first
    assert second.validate(other, "local_groups::run", originals, bindings, facts) is second


def test_late_source_publication_mutation_is_not_hidden_by_registered_native_preservation(tmp_path, monkeypatch):
    path, analysis, originals, bindings, facts = prepared(tmp_path)
    original = analysis.verify_publication
    calls = 0

    def mutate():
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_text(path.read_text()+"! publication changed\n")
        return original()

    monkeypatch.setattr(analysis, "verify_publication", mutate)
    with pytest.raises(CompilationError):
        prove_preserved_entry_definitions(analysis, "local_groups::run", originals, bindings, facts)


def test_canonical_record_is_immutable_encoded_json_and_public_values_are_detached(tmp_path):
    import json
    from hashlib import sha256

    _path, analysis, originals, bindings, facts = prepared(tmp_path)
    proof = prove_preserved_entry_definitions(analysis, "local_groups::run", originals, bindings, facts)
    assert type(proof._record) is str
    assert sha256(proof._record.encode()).hexdigest() == proof.identity
    record = proof.public()
    assert {key: value for key, value in record.items() if key != 'identity'} == json.loads(proof._record)
    record['segments'].clear()
    record['source_inputs']['sources'].clear()
    assert proof.public()['segments']
    assert proof.public()['source_inputs']['sources']
    proof.validate(analysis, 'local_groups::run', originals, bindings, facts)


@pytest.mark.parametrize('record', [None, {}, '{}', '[]'])
def test_malformed_retained_encoded_record_fails_with_compilation_error(tmp_path, record):
    _path, analysis, originals, bindings, facts = prepared(tmp_path)
    proof = prove_preserved_entry_definitions(analysis, 'local_groups::run', originals, bindings, facts)
    object.__setattr__(proof, '_record', record)
    with pytest.raises(CompilationError):
        proof.validate(analysis, 'local_groups::run', originals, bindings, facts)


def test_independent_facts_mutation_after_publication_callback_cannot_be_published(tmp_path, monkeypatch):
    _path, analysis, originals, bindings, facts = prepared(tmp_path)
    original = analysis.verify_publication

    def mutate():
        original()
        facts["late_mutation"] = True

    monkeypatch.setattr(analysis, "verify_publication", mutate)
    with pytest.raises(CompilationError, match="facts|source"):
        prove_preserved_entry_definitions(analysis, "local_groups::run", originals, bindings, facts)


@pytest.mark.parametrize("attachment", ["preserved", "_originals", "_bindings", "_exit", "_record", "_issuer"])
def test_retained_token_attachment_mutation_after_publication_cannot_be_published(tmp_path, monkeypatch, attachment):
    _path, analysis, originals, bindings, facts = prepared(tmp_path)
    proof = prove_preserved_entry_definitions(analysis, "local_groups::run", originals, bindings, facts)
    original = analysis.verify_publication

    def mutate():
        original()
        if sys._getframe(1).f_code is type(proof).validate.__code__:
            replacement = () if attachment in {"preserved", "_originals", "_bindings"} else object()
            object.__setattr__(proof, attachment, replacement)

    monkeypatch.setattr(analysis, "verify_publication", mutate)
    with pytest.raises(CompilationError, match="attachment changed at publication"):
        proof.validate(analysis, "local_groups::run", originals, bindings, facts)


@pytest.mark.parametrize("fault", ["copy", "registry", "facts", "source", "binding", "selection"])
def test_issued_definition_filter_reauthenticates_source_bindings_and_independent_facts(tmp_path, fault):
    path, analysis, originals, bindings, facts = prepared(tmp_path)
    proof = prove_preserved_entry_definitions(analysis, "local_groups::run", originals, bindings, facts)
    if fault == "copy":
        proof = copy(proof)
    elif fault == "registry":
        analysis._entry_preservations.pop(proof.identity)
    elif fault == "facts":
        facts["captures"]["argument::a"]["initialized"] = "none"
    elif fault == "source":
        path.write_text(path.read_text()+"! live source changed\n")
    elif fault == "binding":
        bindings = tuple(reversed(bindings))
    else:
        originals = tuple(reversed(originals))
    with pytest.raises(CompilationError):
        proof.validate(analysis, "local_groups::run", originals, bindings, facts)
