"""Read-only descriptor tokens keep dynamic arrays separate from PURE mapping."""
from copy import copy
from dataclasses import replace

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_array_functions import prove_descriptor_array_function
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.source_functions import prove_descriptor_source_function
from compiler.ir import CompilationError


def fixture(tmp_path, *, inputs="real(8),intent(in)::a(0:)\ninteger,intent(in)::n",
            declarations="real(8)::scratch(2)", body="call fill(a,n,scratch)\nv=sum(scratch)",
            helper="", globals_="", prefix=""):
    path = tmp_path / "array_functions.f90"
    if not helper:
        helper = """subroutine fill(a,n,s)
real(8),intent(in)::a(0:)
integer,intent(in)::n
real(8),intent(out)::s(2)
s(1)=a(n)
s(2)=a(n-1)
end subroutine"""
    path.write_text(f"""module workspace
{globals_}
contains
subroutine driver(a)
real(8),intent(inout)::a(:)
a(1)=inspect(a,1)
end subroutine
{prefix} real(8) function inspect(a,n) result(v)
{inputs}
{declarations}
{body}
end function
{helper}
end module
""")
    analysis = SourceEffects([path])
    caller = analysis.routines["workspace::driver"]
    expression = next(node for node in walk(caller.execution)
                      if type(node).__name__ == "Part_Ref" and str(node.items[0]).lower() == "inspect")
    return path, analysis, caller, expression


def test_exact_assumed_shape_input_and_fixed_private_child_output_are_storage_only(tmp_path):
    path, analysis, caller, expression = fixture(tmp_path)
    before = path.read_bytes()
    original_summary = analysis.summarize(caller.qualified)
    with pytest.raises(CompilationError, match="constant explicit bounds"):
        prove_descriptor_source_function(analysis, caller.qualified, expression)
    token = prove_descriptor_array_function(analysis, caller.qualified, expression)
    assert token.validate(analysis, caller.qualified, expression) is token
    assert prove_descriptor_array_function(analysis, caller.qualified, expression) is token
    public = token.public()
    assert public["schema_version"] == 2
    assert public["proof_role"] == "descriptor_readonly_array_source_function"
    assert public["complete_original_closure_scanned"] is True
    assert {row["procedure"] for row in public["closure"]} == {"workspace::inspect", "workspace::fill"}
    assert public["source_call"]["effect_authority"] is False
    for flag in ("summary_completion_authorized", "execution_authorized", "payload_access_authorized",
                 "entry_definition_authority", "capture_authorized", "pure_scalar_authorized",
                 "gpu_legality_established", "numerical_lowering_authorized", "motion_authorized", "alias_authority"):
        assert public[flag] is False
    assert not analysis._native_functions
    assert analysis.summarize(caller.qualified) == original_summary
    assert path.read_bytes() == before


@pytest.mark.parametrize(("inputs", "declarations", "body", "helper", "globals_"), [
    ("real(8),intent(out)::a(0:)\ninteger,intent(in)::n", "real(8)::scratch(2)", "v=a(n)", "", ""),
    ("real(8),pointer,intent(in)::a(:)\ninteger,intent(in)::n", "", "v=a(n)", "", ""),
    ("real(8),target,intent(in)::a(0:)\ninteger,intent(in)::n", "", "v=a(n)", "", ""),
    ("real(8),optional,intent(in)::a(0:)\ninteger,intent(in)::n", "", "v=a(n)", "", ""),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8),allocatable::scratch(:)", "allocate(scratch(2))\nv=a(n)", "", ""),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8),save::scratch(2)", "v=a(n)", "", ""),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8)::scratch(n)", "v=a(n)", "", ""),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8)::scratch(2)=0", "v=a(n)", "", ""),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8)::scratch(2)", "a(n)=1\nv=a(n)", "", ""),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8)::scratch(2)", "state=a(n)\nv=state", "", "real(8)::state"),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8)::scratch(2)", "print *,a(n)\nv=a(n)", "", ""),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8)::scratch(2)", "v=a(n)\nreturn\ncall absent(a)", "", ""),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8)::scratch(2)", "v=a(n)\nif(n<0) deallocate(state)", "", "real(8),allocatable::state(:)"),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8)::scratch(2)", "call fill(a,n,a)\nv=a(n)", "", ""),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8)::scratch(3)", "call fill(a,n,scratch)\nv=a(n)", "", ""),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8)::scratch(2)", "call fill(a,n,scratch)\nv=a(n)",
     "subroutine fill(a,n,s)\nreal(8),intent(in)::a(0:)\ninteger,intent(in)::n\nreal(8),intent(out)::s(2)\nstate=1\ns=a(n)\nend subroutine", "real(8)::state"),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8)::scratch(2)", "v=hidden(a,n)",
     "real(8) function hidden(a,n)\nreal(8),intent(in)::a(:)\ninteger,intent(in)::n\nprint *,n\nhidden=a(n)\nend function", ""),
    ("real(8),intent(in)::a(0:)\ninteger,intent(in)::n", "real(8)::p,q\nequivalence(p,q)", "v=a(n)", "", ""),
])
def test_external_writes_aliases_escapes_and_hidden_tail_effects_reject(inputs, declarations, body, helper, globals_, tmp_path):
    _path, analysis, caller, expression = fixture(tmp_path, inputs=inputs, declarations=declarations,
                                                  body=body, helper=helper, globals_=globals_)
    with pytest.raises(CompilationError):
        prove_descriptor_array_function(analysis, caller.qualified, expression)
    assert not analysis._descriptor_functions


@pytest.mark.parametrize("mutation", ["registry", "expression", "namespace", "source", "body", "limits", "issuer"])
def test_retained_array_token_rechecks_whole_original_closure(tmp_path, mutation):
    path, analysis, caller, expression = fixture(tmp_path)
    token = prove_descriptor_array_function(analysis, caller.qualified, expression)
    actual = expression
    if mutation == "registry":
        analysis._descriptor_functions[token.identity] = replace(token)
    elif mutation == "expression":
        actual = copy(expression)
    elif mutation == "namespace":
        analysis.routines["workspace::fill"].scope.imports["a"] = ("unknown", "a")
    elif mutation == "source":
        path.write_text(path.read_text() + "! changed\n")
    elif mutation == "body":
        node = next(node for node in walk(analysis.routines["workspace::fill"].execution)
                    if type(node).__name__ == "Name" and str(node).lower() == "a")
        node.string = "missing"
    elif mutation == "limits":
        analysis.operation_limit += 1
    else:
        object.__setattr__(token, "_issuer", object())
    with pytest.raises(CompilationError):
        token.validate(analysis, caller.qualified, actual)


def test_original_counted_loops_branches_and_nested_readonly_function_are_all_scanned(tmp_path):
    _path, analysis, caller, expression = fixture(tmp_path, declarations="real(8)::scratch(2)\ninteger::i",
        body="call fill(a,n,scratch)\nv=0\ndo i=1,2\nif (scratch(i)>0) then\nv=v+magnitude(scratch(i))\nelse\nv=v-scratch(i)\nend if\nend do",
        helper="""subroutine fill(a,n,s)
real(8),intent(in)::a(0:)
integer,intent(in)::n
real(8),intent(out)::s(2)
s(1)=a(n)
s(2)=a(n-1)
end subroutine
pure real(8) function magnitude(x)
real(8),intent(in)::x
magnitude=abs(x)
end function""")
    proof = prove_descriptor_array_function(analysis, caller.qualified, expression)
    assert len(proof.public()["closure"]) == 3
    assert not analysis._native_functions


def test_module_readonly_allocatable_dependency_keeps_original_lifetime_obligation(tmp_path):
    _path, analysis, caller, expression = fixture(tmp_path, globals_="real(8),allocatable::spacing(:)",
        body="v=a(n)*spacing(n)")
    proof = prove_descriptor_array_function(analysis, caller.qualified, expression)
    row = next(row for row in proof.public()["readonly_storage"] if row["original_binding"]["resource"] == "workspace::spacing")
    assert row["requires_original_evaluation_and_lifetime"] is True
    assert row["payload_access_authorized"] is False


def test_frontend_uses_separate_array_token_without_changing_physical_mapping(tmp_path):
    _path, analysis, caller, expression = fixture(tmp_path)
    token = analysis.descriptor_source_function_call(caller.qualified, expression)
    assert token.public()["proof_role"] == "descriptor_readonly_array_source_function"
    with pytest.raises(CompilationError, match="constant explicit bounds"):
        prove_descriptor_source_function(analysis, caller.qualified, expression)
    assert not analysis._native_functions


def test_readonly_initialized_module_value_is_not_an_immutable_capture(tmp_path):
    _path, analysis, caller, expression = fixture(tmp_path, globals_="real(8)::coefficient=0.125_8",
        body="v=a(n)*coefficient")
    proof = prove_descriptor_array_function(analysis, caller.qualified, expression)
    row = next(row for row in proof.public()["readonly_storage"] if row["original_binding"]["resource"] == "workspace::coefficient")
    assert row["persistent_storage_read_only"] is True
    assert proof.public()["capture_authorized"] is False


@pytest.mark.parametrize("attribute", ["pointer", "target", "optional"])
def test_outer_original_actuals_need_the_same_nonassociated_readonly_storage(tmp_path, attribute):
    path, _analysis, _caller, _expression = fixture(tmp_path)
    path.write_text(path.read_text().replace("real(8),intent(inout)::a(:)", f"real(8),{attribute},intent(inout)::a(:)"))
    analysis = SourceEffects([path])
    caller = analysis.routines["workspace::driver"]
    expression = next(node for node in walk(caller.execution)
                      if type(node).__name__ == "Part_Ref" and str(node.items[0]).lower() == "inspect")
    with pytest.raises(CompilationError):
        prove_descriptor_array_function(analysis, caller.qualified, expression)


def test_writable_private_workspace_cannot_overlap_a_readonly_child_actual(tmp_path):
    _path, analysis, caller, expression = fixture(tmp_path,
        body="call fill(scratch,n,scratch)\nv=a(n)")
    with pytest.raises(CompilationError, match="overlaps"):
        prove_descriptor_array_function(analysis, caller.qualified, expression)


def test_complete_closure_retains_operation_budget(tmp_path):
    _path, analysis, caller, expression = fixture(tmp_path, body="\n".join(["v=a(n)"] * 257))
    with pytest.raises(CompilationError, match="operation budget"):
        prove_descriptor_array_function(analysis, caller.qualified, expression)


def test_complete_original_child_requests_token_at_counted_loop_expression(tmp_path):
    from compiler.frontend.descriptor_windows import _derive
    from compiler.tests.test_descriptor_window_proofs import ENTRY, ROOT, prove
    from compiler.tests.test_descriptor_window_proofs import fixture as window_fixture

    _path, analysis, _graph, selected = window_fixture(tmp_path,
        child_decl="real(8),intent(inout)::a(:)\ninteger::i",
        child="do i=2,size(a)\na(i)=inspect(a,i)\nenddo",
        extra="""real(8) function inspect(a,n)
real(8),intent(in)::a(:)
integer,intent(in)::n
inspect=a(n-1)
end function""")
    with pytest.raises(CompilationError, match="registered original readonly"):
        _derive(analysis, ENTRY, selected, (ROOT,))
    public = prove(analysis, selected).public()
    token, = public["native_descriptor_source_function_requirements"]
    assert token["proof_role"] == "descriptor_readonly_array_source_function"
    assert public["resources"][0]["stable"] is True
    assert public["entry_definition_authority"] is False
    assert not analysis._native_functions


def test_selected_root_function_uses_storage_projection_without_readonly_array_token(tmp_path):
    from compiler.tests.test_descriptor_window_proofs import fixture as window_fixture
    from compiler.tests.test_descriptor_window_proofs import prove

    _path, analysis, _graph, selected = window_fixture(tmp_path, selected="total=inspect(a,1)",
        extra="""real(8) function inspect(a,n)
real(8),intent(in)::a(:)
integer,intent(in)::n
inspect=a(n)
end function""")
    public = prove(analysis, selected).public()
    assert public["schema_version"] == 9
    assert public["native_storage_function_requirements"]
    assert not analysis._descriptor_functions


def test_assumed_shape_lower_bound_is_checked_as_original_declaration_work(tmp_path):
    _path, analysis, caller, expression = fixture(tmp_path,
        inputs="real(8),intent(in)::a(n:)\ninteger,intent(in)::n", body="v=a(n)")
    proof = prove_descriptor_array_function(analysis, caller.qualified, expression)
    assert proof.public()["complete_original_closure_scanned"] is True


def test_hidden_specification_function_cannot_evade_full_closure(tmp_path):
    _path, analysis, caller, expression = fixture(tmp_path,
        inputs="real(8),intent(in)::a(lower():)\ninteger,intent(in)::n", body="v=a(n)",
        helper="integer function lower()\nprint *,1\nlower=1\nend function")
    with pytest.raises(CompilationError):
        prove_descriptor_array_function(analysis, caller.qualified, expression)
    assert not analysis._descriptor_functions


def test_private_data_initialization_is_persistent_storage(tmp_path):
    _path, analysis, caller, expression = fixture(tmp_path,
        declarations="real(8)::scratch(2)\ndata scratch/0.0_8,0.0_8/", body="v=a(n)+scratch(1)")
    with pytest.raises(CompilationError, match="persistent|nonassociated"):
        prove_descriptor_array_function(analysis, caller.qualified, expression)


@pytest.mark.parametrize("fault", ["replaced_resolver", "empty_actuals", "reordered_actuals", "copied_actual",
                                  "copied_association_actual", "reordered_formals"])
def test_consumed_original_resolver_edges_cannot_be_substituted(tmp_path, fault):
    _path, analysis, caller, expression = fixture(tmp_path)
    proof = prove_descriptor_array_function(analysis, caller.qualified, expression)
    if fault == "replaced_resolver":
        object.__setattr__(proof, "resolved", replace(proof.resolved, actuals=()))
    elif fault == "empty_actuals":
        object.__setattr__(proof.resolved, "actuals", ())
    elif fault == "reordered_actuals":
        object.__setattr__(proof.resolved, "actuals", tuple(reversed(proof.resolved.actuals)))
    elif fault == "copied_actual":
        object.__setattr__(proof.resolved, "actuals", (copy(proof.resolved.actuals[0]), proof.resolved.actuals[1]))
    elif fault == "copied_association_actual":
        original = proof.resolved.associations
        object.__setattr__(proof.resolved, "associations", (replace(original[0], actual=copy(original[0].actual)), original[1]))
    else:
        object.__setattr__(proof.resolved, "formals", tuple(reversed(proof.resolved.formals)))
    with pytest.raises(CompilationError, match="authority"):
        proof.validate(analysis, caller.qualified, expression)
