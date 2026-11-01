"""Original native control uses source work; only CALL closure consumes depth."""
from pathlib import Path

import pytest

from compiler.frontend import descriptor_windows as windows
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_window_proofs import ENTRY, ROOT, fixture, prove

ORACLE = Path(__file__).with_name("fixtures") / "descriptor_control_recursive.source"


def recursive_oracle():
    namespace = vars(windows).copy()
    exec(compile(ORACLE.read_text(), str(ORACLE), "exec"), namespace)
    return namespace["visit"]


def nested_if(body, count=14):
    return "\n".join(["if(a(1)>0)then"] * count + [body] + ["endif"] * count)


@pytest.mark.parametrize("body", [
    "if(a(1)>0)then\na(1)=a(1)+1\nelse\na(1)=a(1)-1\nendif",
    "do i=1,3\nif(a(i)>0)then\na(i)=2*a(i)\nelse if(a(i)<0)then\na(i)=-a(i)\nendif\nenddo",
    "i=0\nouter:do while(i<3)\ni=i+1\nif(i==2)cycle outer\nif(i==3)exit outer\na(i)=1\nenddo outer",
    "if(a(1)>0)then\ncall helper(a)\nelse\ncall helper(a)\nendif",
])
def test_accepted_counts_order_records_and_identities_match_recursive_visitor(tmp_path, monkeypatch, body):
    extra = "subroutine helper(a)\nreal(8),intent(inout)::a(:)\na(1)=a(1)+2\nend subroutine"
    _path, analysis, _graph, selected = fixture(tmp_path, child=body,
        child_decl="real(8),intent(inout)::a(:)\ninteger::i", extra=extra)
    public = prove(analysis, selected).public()
    monkeypatch.setattr(windows._Check, "visit", recursive_oracle())
    old = windows._derive(analysis, ENTRY, selected, (ROOT,))
    assert {key: value for key, value in public.items() if key != "proof_identity"} == windows._thaw(windows._freeze(old))
    assert public["proof_identity"] == windows._hash(old)
    for key in ("source_operations", "source_traversal_entries", "resolution_validation"):
        assert public[key] == old[key]


def test_complete_deep_native_if_keeps_all_source_work_without_a_call_depth_increase(tmp_path, monkeypatch):
    path, analysis, _graph, selected = fixture(tmp_path, child=nested_if("a(1)=a(1)+1"))
    before = path.read_bytes()
    visitor = windows._Check.visit
    monkeypatch.setattr(windows._Check, "visit", recursive_oracle())
    with pytest.raises(CompilationError, match="control depth"):
        prove(analysis, selected)
    monkeypatch.setattr(windows._Check, "visit", visitor)
    proof = prove(analysis, selected)
    assert proof.validate(analysis, ENTRY, selected) is proof
    assert proof.public()["source_operations"] == 16  # Original call, fourteen conditions, original assignment.
    assert not proof.public()["execution_authorized"]
    assert analysis.depth_limit == 8
    assert path.read_bytes() == before


def test_deep_mixed_control_keeps_exact_same_callee_named_loop_targets(tmp_path):
    body = "i=0\nouter:do while(i<2)\ni=i+1\ninner:do j=1,3\n" + nested_if(
        "if(j==1)cycle outer\nif(j==2)exit inner\na(j)=a(j)+1") + "\nenddo inner\nenddo outer"
    _path, analysis, _graph, selected = fixture(tmp_path, child=body,
        child_decl="real(8),intent(inout)::a(:)\ninteger::i,j")
    public = prove(analysis, selected).public()
    controls = public["native_loop_control_descriptor_requirements"]
    assert [(item["original_statement"], item["target_construct_name"]) for item in controls] == [
        ("CYCLE outer", "outer"), ("EXIT inner", "inner")]
    assert not any(item["termination_authority"] for item in controls)


@pytest.mark.parametrize("unsafe", ["deallocate(a)", "call absent(a)"])
def test_all_deep_arms_and_nodes_after_native_exit_are_still_scanned(tmp_path, unsafe):
    body = "do\nexit\n" + nested_if("a(1)=1\nelse\n" + unsafe) + "\nenddo"
    _path, analysis, _graph, selected = fixture(tmp_path, child=body,
        child_decl="real(8),allocatable,intent(inout)::a(:)")
    with pytest.raises(CompilationError):
        prove(analysis, selected)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("unguarded_tail", [False, True])
def test_deep_original_present_guard_does_not_leak_to_later_optional_payload(tmp_path, unguarded_tail):
    from compiler.tests.test_descriptor_optional_outputs import SOURCE
    from compiler.tests.test_descriptor_optional_outputs import fixture as optional_fixture

    original = "if(present(vector))then\nvector=1.0_8\nendif"
    replacement = "if(present(vector))then\n" + nested_if("vector=1.0_8") + "\nendif"
    if unguarded_tail:
        replacement += "\nvector=2.0_8"
    data = optional_fixture(tmp_path, SOURCE.replace(original, replacement))
    _path, analysis, entry, selected = data
    if unguarded_tail:
        with pytest.raises(CompilationError, match="presence|optional|storage"):
            windows.prove_descriptor_window(analysis, entry, selected, (ROOT,))
    else:
        public = windows.prove_descriptor_window(analysis, entry, selected, (ROOT,)).public()
        assert public["native_optional_array_requirements"][0]["presence"] == "omitted"
        assert not public["entry_definition_authority"]


@pytest.mark.parametrize("failure", ["hard", "function_demand", "termination_demand"])
def test_aborted_nested_control_unwinds_loop_and_procedure_context(tmp_path, monkeypatch, failure):
    _path, analysis, _graph, _selected = fixture(tmp_path, child="do\n" + nested_if("a(1)=1") + "\nenddo")
    check = windows._Check(analysis)
    check.active.append(ENTRY)
    caller_frame = (ENTRY, "caller-loop", object(), "outer", "bare", None)
    check.loop_frames.append(caller_frame)
    routine = analysis.routines["intervals::read_modify"]
    seen = []
    expression_check = check.expression

    def fail(current, expression, *_args, **_kwargs):
        if not check.active or check.active[-1] != current.qualified:
            return expression_check(current, expression, *_args, **_kwargs)
        seen.append((tuple(check.active), tuple(check.loop_frames)))
        if failure == "hard":
            raise CompilationError("diagnostic hard source failure")
        if failure == "function_demand":
            raise windows._DescriptorFunctionRequired(current.qualified, expression)
        raise windows._DescriptorTerminationRequired(current.qualified, expression, "diagnostic unmet call")

    monkeypatch.setattr(check, "expression", fail)
    with pytest.raises(CompilationError):
        check.procedure(routine.qualified)
    assert seen
    assert seen[0][0] == (ENTRY, routine.qualified)
    assert len(seen[0][1]) == 2
    assert check.active == [ENTRY]
    assert check.loop_frames == [caller_frame]
    assert check.team_call_depth == 0
    assert routine.qualified not in check.checked


def test_source_control_worklist_keeps_existing_charged_scan_limit(tmp_path):
    _path, analysis, graph, selected = fixture(tmp_path, selected="if(flag)then\ncontinue\nendif")
    routine = analysis.routines[ENTRY]
    check = windows._Check(analysis)
    check.active.append(ENTRY)
    check.source_budget.reached(routine, graph, selected)
    while check.nodes < check.source_budget.limit - 1:
        check.tick()
    before = check.nodes
    with pytest.raises(CompilationError, match="original-source traversal budget"):
        check.visit(routine, graph, selected)
    assert check.nodes > before
    assert check.source_budget.limit == 65536
    assert check.loop_frames == []


def test_deep_source_if_cannot_hide_overdeep_or_recursive_procedure_calls(tmp_path):
    helpers = "\n".join(f"subroutine helper{i}(a)\nreal(8),intent(inout)::a(:)\ncall helper{i+1}(a)\nend subroutine"
                        for i in range(10)) + "\nsubroutine helper10(a)\nreal(8),intent(inout)::a(:)\na(1)=1\nend subroutine"
    _path, analysis, _graph, selected = fixture(tmp_path, child=nested_if("call helper0(a)"), extra=helpers)
    with pytest.raises(CompilationError, match="call closure budget"):
        prove(analysis, selected)
    assert analysis.depth_limit == 8
