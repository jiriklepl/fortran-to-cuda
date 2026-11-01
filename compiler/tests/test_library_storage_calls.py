"""Synthetic supplied facts exercise source/schema binding, not library qualification."""
import json
from copy import copy
from hashlib import sha256

import pytest
from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_original_call
from compiler.frontend.library_storage_calls import library_storage_call, library_storage_declaration
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_metadata_constructor import SOURCE


def ref(path):
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def fixture(tmp_path, source=SOURCE):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "source.f90"
    path.write_text(source)
    initial = SourceEffects([path])
    routine = initial.routines["opaque_fixture::advance"]
    node = next(n for n in walk(routine.execution) if type(n).__name__ == "Call_Stmt")
    assignment = next(n for n in walk(routine.execution) if type(n).__name__ == "Assignment_Stmt"
                      and str(n.items[0]).lower() == "cache")
    callee = initial.routines["opaque_fixture::operation"]
    evidence = tmp_path / "synthetic-evidence.json"
    evidence.write_text('{"schema_fixture_only":true,"execution_evidence":false}')
    export = tmp_path / "export.json"
    export.write_text("null")
    qualification = tmp_path / "qualification.json"
    value = {"schema_version": 1, "kind": "explicit_pinned_synchronous_private_library_storage",
        "operation": {"procedure": callee.qualified, "body_sha256": sha256(str(callee.scope.node).encode()).hexdigest(),
                      "source": ref(path)}, "receiver_formal": "d",
        "array_formals": [{"name": "a", "type": "real", "kind": 8, "rank": 1, "intent": "inout"}],
        "storage": {"caller_descriptor_mutations": False, "caller_address_retention": False,
            "caller_pointer_association": False, "outstanding_caller_borrower_at_return": False,
            "private_receiver_scratch_may_change": True,
            "receiver_scratch_origin": "fresh library allocator; no alias to live caller allocations",
            "hidden_managed_resources": [], "library_global_effects": "explicit fixture private workspace"},
        "library_sources": [ref(path)], "configured_export": ref(export),
        "external_abi": [{"soname": "synthetic-source-schema-fixture", "artifact": ref(evidence)}],
        "evidence": {key: ref(evidence) for key in ("source_audit", "independent_static_review", "abi_observation", "actual_receiver_declaration")},
        "premises": ["Synthetic explicit storage facts for source-binding tests only"],
        "authority": {"caller_descriptor_preservation": True, "caller_noescape": True,
            "caller_alias_separation": False, "payload_access": False, "initializedness": False,
            "numerical_equivalence": False, "execution": False, "placement": False},
        "qualification_scope": "schema fixture only", "status": "qualified"}
    qualification.write_text(json.dumps(value))
    contract = {"version": 1, "source": {"sha256": ref(path)["sha256"], "span": list(node.item.span),
        "call_sha256": sha256(str(node).encode()).hexdigest(), "callee": callee.qualified,
        "callee_body_sha256": value["operation"]["body_sha256"]},
        "receiver": {"formal": "d", "constructor": {"span": list(assignment.item.span),
            "statement_sha256": sha256(str(assignment).encode()).hexdigest()}, "qualification": ref(qualification)},
        "array_formals": ["a"], "qualification": ref(qualification)}
    analysis = SourceEffects([path], contracts={routine.qualified: {"library_storage_calls": [contract]}})
    routine = analysis.routines[routine.qualified]
    node = next(n for n in walk(routine.execution) if type(n).__name__ == "Call_Stmt")
    return path, qualification, value, analysis, routine, node


def issue(parts):
    return library_storage_call(*parts[3:])


def test_qualified_storage_uses_independent_original_receiver_and_no_device_task_prerequisite(tmp_path):
    parts = fixture(tmp_path)
    proof = issue(parts)
    assert issue(parts) is proof
    assert proof.validate() is proof
    assert library_storage_declaration(parts[3], parts[4], proof._declaration.binding) is proof
    assert not getattr(parts[3], "_external_device_tasks", {})
    assert proof.public()["caller_noescape"] is True
    for key in ("caller_alias_authority", "entry_definition_authority", "payload_access_authorized",
                "execution_authorized", "numerical_equivalence_authorized"):
        assert proof.public()[key] is False


@pytest.mark.parametrize("mutation", ["escape", "integer_bool", "authority", "rank", "operation", "source", "unknown", "pending", "abi", "export"])
def test_qualification_cannot_expand_storage_or_retag_unrelated_evidence(tmp_path, mutation):
    parts = fixture(tmp_path)
    _source, path, value, analysis, routine, _call = parts
    if mutation == "escape":
        value["storage"]["caller_address_retention"] = True
    elif mutation == "integer_bool":
        value["storage"]["caller_descriptor_mutations"] = 0
    elif mutation == "authority":
        value["authority"]["initializedness"] = True
    elif mutation == "rank":
        value["array_formals"][0]["rank"] = 1.0
    elif mutation == "operation":
        value["operation"]["body_sha256"] = "0"*64
    elif mutation == "source":
        value["library_sources"][0]["sha256"] = "0"*64
    elif mutation == "unknown":
        value["invented_authority"] = True
    elif mutation == "pending":
        value["status"] = "pending"
    elif mutation == "abi":
        value["external_abi"][0]["artifact"]["sha256"] = "0"*64
    else:
        value["configured_export"]["sha256"] = "0"*64
    path.write_text(json.dumps(value))
    contract = analysis.contracts[routine.qualified]["library_storage_calls"][0]
    contract["qualification"] = ref(path)
    contract["receiver"]["qualification"] = ref(path)
    with pytest.raises(CompilationError):
        issue(parts)


@pytest.mark.parametrize("mutation", ["copy", "identity", "registry", "contract", "actuals", "constructor"])
def test_cached_call_keeps_exact_original_token_source_contract_and_association(tmp_path, mutation):
    parts = fixture(tmp_path)
    proof = issue(parts)
    analysis = parts[3]
    if mutation == "copy":
        proof = copy(proof)
    elif mutation == "identity":
        object.__setattr__(proof, "identity", "0"*64)
        analysis._library_storage_calls[proof.identity] = proof
    elif mutation == "registry":
        analysis._library_storage_calls.clear()
    elif mutation == "contract":
        analysis.contracts[parts[4].qualified]["library_storage_calls"] = []
    elif mutation == "actuals":
        object.__setattr__(proof._resolved, "actuals", ())
    else:
        object.__setattr__(proof._declaration._constructor, "_opaque", copy(proof._declaration._opaque))
    with pytest.raises(CompilationError):
        proof.validate()


def test_foreign_projection_cannot_borrow_a_registered_library_call(tmp_path):
    parts = fixture(tmp_path)
    proof = issue(parts)
    foreign = parts[3].private_projection()
    assert foreign._library_storage_calls == {}
    copied = copy(proof)
    object.__setattr__(copied, "_analysis", foreign)
    with pytest.raises(CompilationError):
        copied.validate()
    original = resolve_original_call(foreign, parts[4].scope, parts[5])
    fresh = library_storage_call(foreign, parts[4], parts[5], original)
    assert fresh._analysis is foreign
    assert fresh is not proof


def test_late_published_call_cannot_detach_its_receiver_or_actuals(tmp_path, monkeypatch):
    parts = fixture(tmp_path)
    proof = issue(parts)
    original = parts[3].verify_publication

    def publication():
        original()
        object.__setattr__(proof._resolved, "actuals", ())

    monkeypatch.setattr(parts[3], "verify_publication", publication)
    with pytest.raises(CompilationError):
        proof.validate()


def test_same_selected_projection_consumes_only_qualified_storage_and_preserves_no_payload_authority(tmp_path):
    from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects

    parts = fixture(tmp_path)
    analysis, routine, call = parts[3:]
    _graph, selected = analysis._selected_source(routine.qualified, (call,))
    proof = prove_storage_mutation_effects(analysis, routine.qualified, selection=selected)
    public = proof.public()
    assert public["version"] == 5
    assert public["complete"] is True
    assert public["qualified_library_storage"][0]["caller_noescape"] is True
    assert proof.preserves_descriptors()
    assert public["payload_access_authorized"] is False
    assert public["entry_definition_authority"] is False
    assert public["execution_authorized"] is False


@pytest.mark.parametrize("unknown_tail", [False, True])
def test_descriptor_full_wrapper_keeps_private_constructor_and_independent_array_call_obligations(tmp_path, unknown_tail):
    from compiler.frontend.descriptor_windows import prove_descriptor_window

    source = SOURCE
    if unknown_tail:
        source = source.replace("call operation(cache,a)", "call operation(cache,a)\ncall unknown_escape(a)")
    source = source.replace("end module", "subroutine driver(a)\nreal(8),intent(inout)::a(:)\ncall advance(a)\nend subroutine\nend module")
    parts = fixture(tmp_path, source)
    analysis = parts[3]
    routine = analysis.routines["opaque_fixture::driver"]
    call = next(node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt")
    _graph, selected = analysis._selected_source(routine.qualified, (call,))
    if unknown_tail:
        with pytest.raises(CompilationError):
            prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))
    else:
        proof = prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))
        public = proof.public()
        assert public["schema_version"] == 14
        assert public["native_private_library_state_requirements"]
        assert public["native_library_storage_requirements"]
        assert public["entry_definition_authority"] is False
        assert public["execution_authorized"] is False
        assert analysis._summary_authority()["descriptor_window_library_storage_version"] == 14
