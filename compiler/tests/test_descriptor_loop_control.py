"""Original native loops preserve descriptors without new execution authority."""
import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import _Closure
from compiler.frontend.source_functions import prove_native_source_function
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_window_proofs import ENTRY, fixture, prove


@pytest.mark.parametrize(("header", "mode"), [("do", "bare"), ("do while(i<3)", "while")])
def test_complete_native_loop_keeps_conditions_body_and_exact_local_exit(tmp_path, header, mode):
    path, analysis, _graph, selection = fixture(tmp_path,
        child_decl="real(8),intent(inout)::a(:)\ninteger::i",
        child="i=0\n" + header + "\ni=i+1\nif(i>2)exit\na(i)=a(i)+1.0_8\nenddo")
    before = path.read_bytes()
    proof = prove(analysis, selection)
    public = proof.public()
    loop, = public["native_loop_descriptor_requirements"]
    control, = public["native_loop_control_descriptor_requirements"]
    assert loop["mode"] == mode
    assert control["original_statement"] == "EXIT"
    assert control["target_loop_id"] == loop["original_node_id"]
    for item in (loop, control):
        assert item["complete_original_body_scanned"]
        assert not item["termination_authority"]
        assert not item["execution_authorized"]
        assert not item["entry_definition_authority"]
        assert not item["payload_access_authorized"]
    assert not public["reachability_authority"]
    assert proof.validate(analysis, ENTRY, selection) is proof
    assert path.read_bytes() == before


def test_named_and_default_exits_cycles_resolve_exact_same_callee_lexical_loops(tmp_path):
    _path, analysis, _graph, selection = fixture(tmp_path,
        child_decl="real(8),intent(inout)::a(:)\ninteger::i,j",
        child="i=0\nouter: do while(i<3)\ni=i+1\ninner: do j=1,3\n"
              "if(j==1) cycle\nif(j==2) cycle outer\nif(i==2)exit inner\nexit outer\nenddo inner\nenddo outer")
    public = prove(analysis, selection).public()
    items = public["native_loop_control_descriptor_requirements"]
    assert [(item["original_statement"], item["target_construct_name"]) for item in items] == [
        ("CYCLE", "inner"), ("CYCLE outer", "outer"), ("EXIT inner", "inner"), ("EXIT outer", "outer")]
    assert len({item["target_loop_id"] for item in items}) == 2


def test_while_condition_requires_separate_registered_storage_or_readonly_source_proof(tmp_path):
    extra = "logical pure function check(x) result(ok)\nreal(8),intent(in)::x\nok=x>0\nend function"
    _path, analysis, _graph, selection = fixture(tmp_path, extra=extra,
        child_decl="real(8),intent(inout)::a(:)\nreal(8)::value",
        child="value=a(1)\ndo while(check(value))\nvalue=value-1.0_8\nenddo\na(1)=value")
    # The schema9 frontend can now issue an independent descriptor projection
    # while retaining the old explicitly issued PURE token path below.
    public = prove(analysis, selection).public()
    assert public["schema_version"] == 9
    assert public["native_storage_function_requirements"]
    assert not public["execution_authorized"]
    child = analysis.routines["intervals::read_modify"]
    loop = next(node for node in child.execution.content if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    condition = loop.content[0].items[1].items[0]
    prove_native_source_function(analysis, child.qualified, condition, (), _Closure())
    public = prove(analysis, selection).public()
    assert public["native_source_function_requirements"]
    assert public["native_loop_descriptor_requirements"][0]["condition"] == "check(value)"


@pytest.mark.parametrize("selected", [
    "do\nexit\nenddo", "do while(flag)\nexit\nenddo",
    "do counter=1,3\nexit\nenddo", "do counter=1,3\ncycle\nenddo",
])
def test_selected_root_loop_control_does_not_gain_completion_authority(tmp_path, selected):
    _path, analysis, _graph, selection = fixture(tmp_path, selected=selected)
    with pytest.raises(CompilationError, match="complete.*callee"):
        prove(analysis, selection)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("control", ["exit absent", "cycle absent", "exit", "cycle"])
def test_missing_original_local_target_stays_closed(tmp_path, control):
    body = "do\n" + control + "\nenddo" if "absent" in control else control
    _path, analysis, _graph, selection = fixture(tmp_path, child=body)
    with pytest.raises(CompilationError, match="same-callee lexical target"):
        prove(analysis, selection)


def test_called_procedure_cannot_exit_a_callers_original_named_loop(tmp_path):
    extra = "subroutine escape(a)\nreal(8),intent(inout)::a(:)\nexit outer\nend subroutine"
    _path, analysis, _graph, selection = fixture(tmp_path, extra=extra,
        child="outer:do\ncall escape(a)\nexit outer\nenddo outer")
    with pytest.raises(CompilationError, match="same-callee lexical target"):
        prove(analysis, selection)


@pytest.mark.parametrize("body", [
    "do concurrent(i=1:3)\na(i)=1.0_8\nenddo",
    "do\ngoto 10\n10 continue\nexit\nenddo",
    "do 10 i=1,3\na(i)=1.0_8\n10 continue",
])
def test_concurrent_goto_and_labelled_loops_require_separate_control_proof(tmp_path, body):
    _path, analysis, _graph, selection = fixture(tmp_path, child=body,
        child_decl="real(8),intent(inout)::a(:)\ninteger::i")
    with pytest.raises(CompilationError):
        prove(analysis, selection)


@pytest.mark.parametrize("control", ["exit", "cycle"])
def test_original_openmp_control_cannot_be_exited_by_new_descriptor_slice(tmp_path, control):
    _path, analysis, _graph, selection = fixture(tmp_path,
        child_decl="real(8),intent(inout)::a(:)\ninteger::i",
        child="!$omp parallel do private(i)\ndo i=1,3\n" + control + "\na(i)=1.0_8\nenddo\n!$omp end parallel do")
    with pytest.raises(CompilationError, match="OpenMP control"):
        prove(analysis, selection)


@pytest.mark.parametrize(("control", "effect"), [
    ("exit", "deallocate(a)"), ("cycle", "allocate(a(3))"),
    ("exit", "a=[1.0_8,2.0_8]"), ("cycle", "call absent(a)"),
])
def test_every_lifetime_or_unknown_effect_after_early_control_is_still_checked(tmp_path, control, effect):
    _path, analysis, _graph, selection = fixture(tmp_path,
        child_decl="real(8),allocatable,intent(inout)::a(:)", child="do\n" + control + "\n" + effect + "\nenddo")
    with pytest.raises(CompilationError):
        prove(analysis, selection)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_unknown_while_condition_cannot_hide_effects(tmp_path):
    _path, analysis, _graph, selection = fixture(tmp_path, child="do while(absent(a(1)))\nexit\nenddo")
    with pytest.raises(CompilationError, match="source-function|unknown function"):
        prove(analysis, selection)


def test_deep_loop_nesting_uses_source_budget_and_retains_exact_exit_target(tmp_path):
    _path, analysis, _graph, selection = fixture(tmp_path,
        child="\n".join(["do"] * 10 + ["exit"] + ["enddo"] * 10))
    public = prove(analysis, selection).public()
    assert len(public["native_loop_descriptor_requirements"]) == 10
    control, = public["native_loop_control_descriptor_requirements"]
    assert control["target_loop_id"] == public["native_loop_descriptor_requirements"][-1]["original_node_id"]
    assert not public["execution_authorized"]
    assert analysis.depth_limit == 8


@pytest.mark.parametrize("mutation", ["construct", "target"])
def test_retained_control_proof_rechecks_original_construct_and_target_names(tmp_path, mutation):
    _path, analysis, _graph, selection = fixture(tmp_path, child="outer:do\nexit outer\nenddo outer")
    proof = prove(analysis, selection)
    child = analysis.routines["intervals::read_modify"]
    if mutation == "construct":
        child.execution.content[0].content[0].item.name = "changed"
    else:
        statement = next(node for node in walk(child.execution) if type(node).__name__ == "Exit_Stmt")
        statement.items[1].string = "changed"
    with pytest.raises(CompilationError):
        proof.validate(analysis, ENTRY, selection)
