"""Separate read-only native function evidence cannot grant PURE/GPU authority."""
from copy import copy
from dataclasses import replace
from hashlib import sha256

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_windows import _derive
from compiler.frontend.source_effects import SourceEffects, _Closure
from compiler.frontend.source_functions import (
    SOURCE_FUNCTION_VERSION,
    prove_descriptor_source_function,
    prove_native_source_function,
)
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_window_proofs import ENTRY, ROOT, fixture, prove


def function_fixture(tmp_path, *, declaration="real(8),intent(in)::x", body="ok=x>0", extra="", prefix=""):
    path, analysis, graph, selected = fixture(tmp_path,
        child_decl="real(8),intent(inout)::a(:)\nreal(8)::value",
        child="value=a(1)\ndo while(check(value))\nvalue=value-1.0_8\nenddo\na(1)=value",
        extra=f"{prefix} logical function check(x) result(ok)\n{declaration}\n{body}\nend function\n{extra}")
    caller = analysis.routines["intervals::read_modify"]
    expression = next(node for node in walk(caller.execution)
                      if type(node).__name__ == "Part_Ref" and str(node.items[0]).lower() == "check")
    return path, analysis, graph, selected, caller, expression


def test_new_token_is_metadata_only_and_does_not_complete_unsupported_effect_summary(tmp_path):
    path, analysis, _graph, selected, caller, expression = function_fixture(tmp_path)
    summary = analysis.summarize(caller.qualified)
    assert not summary["complete"]
    before = path.read_bytes()
    with pytest.raises(CompilationError, match="registered original readonly"):
        _derive(analysis, ENTRY, selected, (ROOT,))
    public = prove(analysis, selected).public()
    token, = public["native_descriptor_source_function_requirements"]
    assert token["schema_version"] == 1
    assert token["proof_role"] == "descriptor_readonly_source_function"
    assert token["configured_purity"] == "not declared PURE in configured source"
    assert token["requires_original_execution"]
    for flag in ("execution_authorized", "payload_access_authorized", "entry_definition_authority",
                 "capture_authorized", "pure_scalar_authorized", "gpu_legality_established",
                 "numerical_lowering_authorized"):
        assert token[flag] is False
    assert not analysis._native_functions
    assert analysis.summarize(caller.qualified) == summary
    assert path.read_bytes() == before
    with pytest.raises(CompilationError, match="requires PURE"):
        prove_native_source_function(analysis, caller.qualified, expression, (), _Closure())


def test_old_pure_v2_token_and_record_remain_exact(tmp_path):
    _path, analysis, _graph, selected, caller, expression = function_fixture(tmp_path, prefix="pure")
    old, _summary = prove_native_source_function(analysis, caller.qualified, expression, (), _Closure())
    before = old.public()
    public = prove(analysis, selected).public()
    assert SOURCE_FUNCTION_VERSION == before["schema_version"] == 2
    assert before["proof_role"] == "native_readonly_source_function"
    assert public["native_source_function_requirements"] == [before]
    assert "native_descriptor_source_function_requirements" not in public
    assert not analysis._descriptor_functions
    assert old.public() == before


@pytest.mark.parametrize(("declaration", "body", "extra", "storage_stable"), [
    ("real(8),intent(in)::x", "print *,x\nok=x>0", "", True),
    ("real(8),intent(in)::x\ninteger,save::counter=0", "counter=counter+1\nok=x>0", "", True),
    ("real(8),intent(in)::x\nreal(8),allocatable::p(:)", "allocate(p(2))\nok=x>0", "", True),
    ("real(8),pointer::x", "ok=x>0", "", False),
    ("real(8),target,intent(in)::x", "ok=x>0", "", False),
    ("real(8),intent(out)::x", "x=1\nok=x>0", "", True),
    ("real(8),intent(inout)::x", "x=1\nok=x>0", "", True),
    ("real(8),intent(in)::x\nreal(8)::c\ncommon /shared/c", "ok=x+c>0", "", False),
    ("real(8),intent(in)::x\nreal(8)::p,q\nequivalence(p,q)", "ok=x>0", "", False),
    ("real(8),intent(in)::x\nreal(8)::p=1", "ok=x+p>0", "", True),
    ("real(8),intent(in)::x", "call absent(x)\nok=x>0", "", False),
    ("real(8),intent(in)::x", "ok=hidden(x)",
     "logical function hidden(x)\nreal(8),intent(in)::x\nprint *,x\nhidden=x>0\nend function", True),
    ("real(8),intent(in)::x", "call hidden(x)\nok=x>0",
     "subroutine hidden(x)\nreal(8),intent(in)::x\nprint *,x\nend subroutine", True),
])
def test_readonly_issuer_rejects_payload_effects_while_storage_projection_distinguishes_lifetime(
        declaration, body, extra, storage_stable, tmp_path):
    _path, analysis, _graph, selected, caller, expression = function_fixture(
        tmp_path, declaration=declaration, body=body, extra=extra)
    with pytest.raises(CompilationError):
        prove_descriptor_source_function(analysis, caller.qualified, expression)
    assert not analysis._descriptor_functions
    if storage_stable:
        public = prove(analysis, selected).public()
        assert public["schema_version"] == 9
        assert public["native_storage_function_requirements"]
        assert not public["execution_authorized"]
    else:
        with pytest.raises(CompilationError):
            prove(analysis, selected)


def test_module_write_in_hidden_child_rejects_complete_closure(tmp_path):
    path, analysis, _graph, _selected, caller, expression = function_fixture(tmp_path,
        body="call hidden(x)\nok=x>0", extra="subroutine hidden(x)\nreal(8),intent(in)::x\nstate=x\nend subroutine")
    text = path.read_text().replace("module intervals\n", "module intervals\nreal(8)::state\n")
    path.write_text(text)
    analysis = SourceEffects([path])
    caller = analysis.routines[caller.qualified]
    expression = next(node for node in walk(caller.execution)
                      if type(node).__name__ == "Part_Ref" and str(node.items[0]).lower() == "check")
    with pytest.raises(CompilationError, match="read-only"):
        prove_descriptor_source_function(analysis, caller.qualified, expression)


@pytest.mark.parametrize("mutation", ["registry", "expression", "namespace", "body", "limits", "source"])
def test_retained_token_reauthenticates_exact_original_source_and_namespace(tmp_path, mutation):
    path, analysis, _graph, _selected, caller, expression = function_fixture(tmp_path)
    token = prove_descriptor_source_function(analysis, caller.qualified, expression)
    target = expression
    if mutation == "registry":
        analysis._descriptor_functions[token.identity] = replace(token)
    elif mutation == "expression":
        target = copy(expression)
    elif mutation == "namespace":
        caller.scope.imports["check"] = ("absent", "check")
    elif mutation == "body":
        callee = analysis.numerical_helpers[token.callee]
        node = next(node for node in walk(callee.execution) if type(node).__name__ == "Name" and str(node) == "x")
        node.string = "missing"
    elif mutation == "limits":
        analysis.operation_limit += 1
    else:
        path.write_text(path.read_text() + "! changed\n")
    with pytest.raises(CompilationError):
        token.validate(analysis, caller.qualified, target)


def test_configured_removed_pure_is_not_raw_source_purity(tmp_path):
    path, _analysis, _graph, _selected, _caller, _expression = function_fixture(tmp_path, prefix="pure")
    source = path.read_text()
    path.write_text("#define pure\n" + source)
    prepared = tmp_path / "prepared.f90"
    prepared.write_text(source.replace("pure logical function", "logical function"))
    document = {"schema_version": 1, "source_inputs": {str(path): sha256(path.read_bytes()).hexdigest()},
        "preserves_source_order": True, "configuration": {"defines": []}, "dependencies": {},
        "entries": [{"source": str(path), "path": str(prepared),
                     "sha256": sha256(prepared.read_bytes()).hexdigest(),
                     "line_map": list(range(2, len(source.splitlines()) + 2))}]}
    analysis = SourceEffects([path], analysis_sources=document)
    caller = analysis.routines["intervals::read_modify"]
    expression = next(node for node in walk(caller.execution)
                      if type(node).__name__ == "Part_Ref" and str(node.items[0]).lower() == "check")
    with pytest.raises(CompilationError, match="requires PURE"):
        prove_native_source_function(analysis, caller.qualified, expression, (), _Closure())
    token = prove_descriptor_source_function(analysis, caller.qualified, expression)
    assert token.public()["configured_purity"] == "not declared PURE in configured source"
    assert not analysis._native_functions
    prepared.write_text(prepared.read_text().replace("logical function", "impure logical function"))
    with pytest.raises(CompilationError):
        token.validate(analysis, caller.qualified, expression)


def test_discovery_does_not_admit_selected_root_function_or_trim_hidden_effects(tmp_path):
    _path, analysis, _graph, selected = fixture(tmp_path,
        selected="do while(check(total))\nexit\nenddo",
        extra="logical function check(x)\nreal(8),intent(in)::x\ncheck=x>0\nend function")
    with pytest.raises(CompilationError):
        prove(analysis, selected)
    assert not analysis._descriptor_functions


def test_complete_readonly_subroutine_child_is_checked_without_pure_claim(tmp_path):
    _path, analysis, _graph, _selected, caller, expression = function_fixture(tmp_path,
        body="call helper(x)\nok=x>0",
        extra="subroutine helper(x)\nreal(8),intent(in)::x\nreal(8)::scratch\nscratch=x+1\nend subroutine")
    # Every original child is retained; private arithmetic grants no execution
    # or caller entry-definition authority.
    token = prove_descriptor_source_function(analysis, caller.qualified, expression)
    assert token.public()["complete_transitive_effects"]
    assert not analysis._native_functions


def test_post_summary_mutation_cannot_publish_old_child_effects(tmp_path, monkeypatch):
    path, analysis, _graph, _selected, caller, expression = function_fixture(tmp_path)
    original = SourceEffects.summarize

    def mutate(self, requested, *args, **kwargs):
        result = original(self, requested, *args, **kwargs)
        if requested == "intervals::check":
            path.write_text(path.read_text() + "! mid-demand mutation\n")
        return result

    monkeypatch.setattr(SourceEffects, "summarize", mutate)
    with pytest.raises(CompilationError, match="changed"):
        prove_descriptor_source_function(analysis, caller.qualified, expression)
    assert not analysis._descriptor_functions


def test_private_projection_does_not_lend_metadata_registry_mutation(tmp_path):
    _path, analysis, _graph, _selected, caller, expression = function_fixture(tmp_path)
    token = prove_descriptor_source_function(analysis, caller.qualified, expression)
    projected = analysis.private_projection()
    projected._descriptor_functions.clear()
    assert analysis._descriptor_functions == {token.identity: token}
    assert token.validate(analysis, caller.qualified, expression) is token


def test_nonpure_function_outside_old_discovery_has_only_storage_authority(tmp_path):
    _path, analysis, _graph, selected = fixture(tmp_path,
        child_decl="real(8),intent(inout)::a(:)\nreal(8)::value",
        child="value=a(1)\nif(check(value))a(1)=2",
        extra="logical function check(x)\nreal(8),intent(in)::x\ncheck=x>0\nend function")
    assert prove(analysis, selected).public()["schema_version"] == 9
    assert not analysis._descriptor_functions


def test_unsupported_loop_does_not_leak_discovery_to_later_sibling(tmp_path):
    _path, analysis, _graph, selected = fixture(tmp_path,
        child_decl="real(8),intent(inout)::a(:)\nreal(8)::value",
        child="do\nexit\nenddo\nvalue=a(1)\nif(check(value))a(1)=2",
        extra="logical function check(x)\nreal(8),intent(in)::x\ncheck=x>0\nend function")
    assert prove(analysis, selected).public()["schema_version"] == 9
    assert not analysis._descriptor_functions


def test_discovery_never_trims_unsafe_source_after_native_early_control(tmp_path):
    _path, analysis, _graph, selected, _caller, _expression = function_fixture(tmp_path,
        body="return\ncall absent(x)\nok=x>0")
    with pytest.raises(CompilationError):
        prove(analysis, selected)
    assert not analysis._descriptor_functions


@pytest.mark.parametrize("declaration", [
    "real(8)::p\ncommon /storage/p", "real(8)::p,q\nequivalence(p,q)",
    "real(8),target::p", "real(8),pointer::p", "real(8),allocatable::p(:)",
    "real(8),save::p", "real(8)::p=1.0_8",
])
def test_hidden_callee_storage_association_and_lifetime_are_not_erased(tmp_path, declaration):
    _path, analysis, _graph, _selected, caller, expression = function_fixture(tmp_path,
        body="call helper(x)\nok=x>0", extra=f"subroutine helper(x)\nreal(8),intent(in)::x\n{declaration}\nend subroutine")
    with pytest.raises(CompilationError):
        prove_descriptor_source_function(analysis, caller.qualified, expression)
    assert not analysis._descriptor_functions


def test_hidden_original_out_definition_is_not_authorized(tmp_path):
    _path, analysis, _graph, _selected, caller, expression = function_fixture(tmp_path,
        declaration="real(8),intent(in)::x\nreal(8)::scratch",
        body="call helper(x,scratch)\nok=scratch>0",
        extra="subroutine helper(x,y)\nreal(8),intent(in)::x\nreal(8),intent(out)::y\ny=x+1\nend subroutine")
    with pytest.raises(CompilationError):
        prove_descriptor_source_function(analysis, caller.qualified, expression)


@pytest.mark.parametrize("budget", ["syntax", "closure", "registry"])
def test_each_existing_issuer_budget_stays_bounded(tmp_path, budget):
    _path, analysis, _graph, _selected, caller, expression = function_fixture(tmp_path)
    if budget == "syntax":
        analysis.operation_limit = 3
    elif budget == "closure":
        analysis.procedure_limit = 0
    else:
        analysis._descriptor_functions.update((str(i), object()) for i in range(analysis.operation_limit))
    with pytest.raises(CompilationError, match="budget|unproved"):
        prove_descriptor_source_function(analysis, caller.qualified, expression)



def test_frontend_issues_only_exact_demands_then_complete_consumer_restarts(tmp_path, monkeypatch):
    _path, analysis, _graph, selected, _caller, _expression = function_fixture(tmp_path)
    calls = []
    original = SourceEffects.descriptor_source_function_call

    def requested(self, procedure, expression):
        calls.append((procedure, expression))
        return original(self, procedure, expression)

    monkeypatch.setattr(SourceEffects, "descriptor_source_function_call", requested)
    proof = prove(analysis, selected)
    assert len(calls) == 1
    assert calls[0][0] == "intervals::read_modify"
    assert calls[0][1] is next(iter(analysis._descriptor_functions.values()))._expression
    assert proof.validate(analysis, ENTRY, selected) is proof
    assert len(calls) == 1  # Retained/direct consumer validation cannot mint.


def test_repeated_unmet_demand_rejects_instead_of_unbounded_retry(tmp_path, monkeypatch):
    _path, analysis, _graph, selected, _caller, _expression = function_fixture(tmp_path)
    calls = []

    def no_progress(self, procedure, expression):
        calls.append((procedure, expression))
        return object()  # No issued token was actually registered.

    monkeypatch.setattr(SourceEffects, "descriptor_source_function_call", no_progress)
    with pytest.raises(CompilationError, match="no progress"):
        prove(analysis, selected)
    assert len(calls) == 1
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_unique_requested_functions_retain_normal_procedure_cap(tmp_path):
    functions = "\n".join(f"logical function check{i}(x)\nreal(8),intent(in)::x\ncheck{i}=x>{i}\nend function"
                          for i in range(3))
    _path, analysis, _graph, selected = fixture(tmp_path, procedures=2,
        child_decl="real(8),intent(inout)::a(:)\nreal(8)::value",
        child="value=a(1)\ndo\nif(check0(value))exit\nif(check1(value))exit\nif(check2(value))exit\nexit\nenddo",
        extra=functions)
    with pytest.raises(CompilationError, match="admission request budget"):
        prove(analysis, selected)
    assert len(analysis._descriptor_functions) == 2
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_admission_never_skips_later_original_native_lifetime_effects(tmp_path):
    _path, analysis, _graph, selected = fixture(tmp_path,
        child_decl="real(8),allocatable,intent(inout)::a(:)\nreal(8)::value",
        child="value=a(1)\ndo while(check(value))\nexit\nenddo\ndeallocate(a)",
        extra="logical function check(x)\nreal(8),intent(in)::x\ncheck=x>0\nend function")
    with pytest.raises(CompilationError):
        prove(analysis, selected)
    assert analysis._descriptor_functions  # Exact function proof did not hide the later effect.
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_original_token_admission_is_visible_through_authenticated_private_view(tmp_path):
    _path, analysis, _graph, selected, _caller, _expression = function_fixture(tmp_path)
    projected = analysis.private_projection()
    projected._descriptor_source = analysis
    proof = prove(projected, selected)
    public = proof.public()
    assert len(public["native_descriptor_source_function_requirements"]) == 1
    assert not projected._descriptor_functions  # No registry mutation lent across the view.
    assert len(analysis._descriptor_functions) == 1
    assert proof.validate(projected, ENTRY, selected) is proof
