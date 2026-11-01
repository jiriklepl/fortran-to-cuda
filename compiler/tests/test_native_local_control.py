"""Original callee-local control retains caller joins and complete may-effects."""
from copy import copy

import pytest

from compiler.frontend.resource_reachability import prove_resource_reachability
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module work
implicit none
real(8)::a(8),input(4),later(8)
contains
subroutine owner()
integer::i
!$omp parallel do private(i) schedule(guided,5)
do i=1,8
call helper(a(i),input)
enddo
!$omp end parallel do
end subroutine
subroutine helper(value,values)
real(8),intent(inout)::value
real(8),intent(in)::values(:)
integer::j,k
BODY
end subroutine
end module
"""


def setup(tmp_path, body, *, source=SOURCE, seal=True, selection=None):
    path = tmp_path / "source.f90"
    path.write_text(source.replace("BODY", body))
    analysis = SourceEffects([path])
    if seal:
        analysis.seal_session()
    routine = analysis.routines["work::owner"]
    originals = tuple(routine.execution.content) if selection is None else selection(analysis)
    demanded = tuple(analysis.modules["work"].bindings[name] for name in ("a", "input", "later"))
    proof = prove_resource_reachability(analysis, routine.qualified, originals, demanded)
    return path, analysis, originals, demanded, proof


@pytest.mark.parametrize("seal", [False, True])
@pytest.mark.parametrize("body", [
    "do\nvalue=value+values(1)\nif(value>3)exit\nenddo",
    "do while(value<3)\nvalue=value+values(1)\ncycle\nenddo",
    "do j=1,size(values)\nif(j==2)cycle\nvalue=value+values(j)\nif(value>3)exit\nenddo",
    "outer: do\ninner: do j=1,size(values)\nif(j==2)cycle inner\n"
    "value=value+values(j)\nif(value>3)exit outer\nenddo inner\nexit outer\nenddo outer",
    "do\nif(value<0)return\nvalue=value+values(1)\nexit\nenddo",
    "do\nvalue=value+values(1)\nenddo",  # no termination claim
])
def test_local_control_preserves_complete_original_graph_and_outer_join(tmp_path, body, seal):
    _path, analysis, originals, demanded, proof = setup(tmp_path, body, seal=seal)
    assert not proof.barriers, proof.public()["barriers"]
    assert proof.validate(analysis, "work::owner", originals, demanded) is proof
    actions = {(item.storage.resource, item.kind) for item in proof.accesses}
    assert ("work::a", "write") in actions
    assert ("work::input", "read") in actions
    completion, = [item for _name, _selected, item in proof._completion
                   if type(item).__name__ == "NativeGraphCompletionProof"]
    assert completion.public()["effect_graph_required"]
    assert not completion.public()["internal_cuts_authorized"]
    assert not completion.public()["gpu_legality_established"]
    assert not proof.public()["execution_authorized"]
    assert body in _path.read_text()


def test_effects_after_local_transfer_remain_conservative(tmp_path):
    _path, _analysis, _originals, _demanded, proof = setup(
        tmp_path, "do\nif(value<0)exit\nvalue=value+values(1)\nlater(1)=value\nexit\nenddo")
    assert not proof.barriers
    assert any(item.storage.resource == "work::later" and item.kind == "write" for item in proof.accesses)


@pytest.mark.parametrize("body", [
    "exit",
    "cycle",
    "do\nexit missing\nenddo",
    "do\ncycle missing\nenddo",
    "do concurrent(j=1:4)\nvalue=value+values(j)\nenddo",
    "do concurrent(j=1:4)\nexit\nenddo",
    "do\n!$omp task\nvalue=value+values(1)\n!$omp end task\nexit\nenddo",
    "outer: do\n!$omp parallel do private(j)\ndo j=1,4\nexit outer\nenddo\n"
    "!$omp end parallel do\nexit outer\nenddo outer",
    "do\ngoto 10\n10 continue\nexit\nenddo",
])
def test_unknown_escaping_or_concurrent_control_remains_a_barrier(tmp_path, body):
    _path, analysis, _originals, _demanded, proof = setup(tmp_path, body)
    assert proof.barriers
    assert not proof.native_graph.completion_eligible
    assert not getattr(analysis, "_native_graph_completions", {})


def test_callee_cannot_target_the_callers_named_outer_loop(tmp_path):
    source = SOURCE.replace("do i=1,8", "outer: do i=1,8").replace(
        "enddo\n!$omp end parallel do", "enddo outer\n!$omp end parallel do")
    _path, _analysis, _originals, _demanded, proof = setup(tmp_path, "exit outer", source=source)
    assert any("callee-local ordinary loop target" in item.reason for item in proof.barriers)


def test_root_selection_cannot_borrow_complete_callee_loop_authority(tmp_path):
    source = SOURCE.replace("!$omp parallel do private(i) schedule(guided,5)\ndo i=1,8\n"
                            "call helper(a(i),input)\nenddo\n!$omp end parallel do",
                            "do\na(1)=input(1)\nexit\nenddo")
    _path, _analysis, _originals, _demanded, proof = setup(tmp_path, "value=values(1)", source=source)
    assert any("non-counted loop" in item.reason for item in proof.barriers)
    assert any("callee-local ordinary loop target" in item.reason for item in proof.barriers)


@pytest.mark.parametrize("expression", ["norm2(values)", "norm2(values,dim=1)", "NORM2(values(1:4))"])
def test_original_norm2_reads_its_inputs_without_source_call_or_gpu_authority(tmp_path, expression):
    _path, _analysis, _originals, _demanded, proof = setup(tmp_path, "value=" + expression)
    assert not proof.barriers, proof.public()["barriers"]
    assert {call.callee for call in proof.calls} == {"work::helper"}
    assert any(item.storage.resource == "work::input" and item.kind == "read" for item in proof.accesses)
    assert not proof.public()["execution_authorized"]


def test_source_procedure_named_norm2_keeps_its_real_effects(tmp_path):
    source = SOURCE.replace("end module", """real(8) function norm2(v)
real(8),intent(in)::v(:)
later(1)=v(1)
norm2=v(1)
end function
end module""")
    _path, _analysis, _originals, _demanded, proof = setup(tmp_path, "value=norm2(values)", source=source)
    assert not proof.barriers, proof.public()["barriers"]
    assert "work::norm2" in {call.callee for call in proof.calls}
    assert any(item.storage.resource == "work::later" and item.kind == "write" for item in proof.accesses)


@pytest.mark.parametrize("name", ["norm2", "sum"])
@pytest.mark.parametrize("declaration", ["real(8),external::{name}", "real(8)::{name}\nexternal {name}"])
def test_external_reduction_is_not_treated_as_an_intrinsic(tmp_path, name, declaration):
    source = SOURCE.replace("integer::j,k", "integer::j,k\n" + declaration.format(name=name))
    _path, _analysis, _originals, _demanded, proof = setup(tmp_path, f"value={name}(values)", source=source)
    assert proof.barriers


@pytest.mark.parametrize("name", ["norm2", "sum"])
@pytest.mark.parametrize("declaration", ["intrinsic {name}", "real(8),intrinsic::{name}",
                                         "real(8)::{name}\nintrinsic {name}"])
def test_exact_original_explicit_intrinsic_declarations_keep_input_reads(tmp_path, name, declaration):
    source = SOURCE.replace("integer::j,k", "integer::j,k\n" + declaration.format(name=name))
    _path, _analysis, _originals, _demanded, proof = setup(tmp_path, f"value={name}(values)", source=source)
    assert not proof.barriers, proof.public()["barriers"]
    assert {call.callee for call in proof.calls} == {"work::helper"}
    assert any(item.storage.resource == "work::input" and item.kind == "read" for item in proof.accesses)


def test_local_intrinsic_declaration_shadows_host_source_function(tmp_path):
    source = SOURCE.replace("integer::j,k", "integer::j,k\nintrinsic norm2")
    source = source.replace("end module", """real(8) function norm2(v)
real(8),intent(in)::v(:)
later(1)=v(1)
norm2=v(1)
end function
end module""")
    _path, _analysis, _originals, _demanded, proof = setup(tmp_path, "value=norm2(values)", source=source)
    assert not proof.barriers, proof.public()["barriers"]
    assert "work::norm2" not in {call.callee for call in proof.calls}
    assert not any(item.storage.resource == "work::later" and item.kind == "write" for item in proof.accesses)


def test_use_associated_source_function_is_not_an_intrinsic(tmp_path):
    shadow = """module shadow
real(8)::observed
contains
real(8) function norm2(v)
real(8),intent(in)::v(:)
observed=v(1)
norm2=v(1)
end function
end module
"""
    source = shadow + SOURCE.replace("module work\n", "module work\nuse shadow,only:norm2\n")
    _path, _analysis, _originals, _demanded, proof = setup(tmp_path, "value=norm2(values)", source=source)
    assert not proof.barriers, proof.public()["barriers"]
    assert "shadow::norm2" in {call.callee for call in proof.calls}
    assert any(item.storage.resource == "shadow::observed" and item.kind == "write" for item in proof.accesses)


def test_local_external_cannot_borrow_a_host_intrinsic_binding(tmp_path):
    source = SOURCE.replace("implicit none\n", "implicit none\nreal(8),intrinsic::norm2\n")
    source = source.replace("subroutine helper(value,values)\n", "subroutine helper(value,values)\nimplicit real(8)(n)\n")
    source = source.replace("integer::j,k", "integer::j,k\nexternal norm2")
    _path, _analysis, _originals, _demanded, proof = setup(tmp_path, "value=norm2(values)", source=source)
    assert proof.barriers


@pytest.mark.parametrize("name", ["norm2", "sum"])
@pytest.mark.parametrize("declaration", ["intrinsic {name}", "real(8),intrinsic::{name}"])
def test_unrelated_interface_intrinsic_cannot_hide_effectful_host_function(tmp_path, name, declaration):
    interface = ("interface\nsubroutine unrelated()\n" + declaration.format(name=name)
                 + "\nend subroutine\nend interface")
    source = SOURCE.replace("integer::j,k", "integer::j,k\n" + interface)
    source = source.replace("end module", f"""real(8) function {name}(v)
real(8),intent(in)::v(:)
later(1)=v(1)
{name}=v(1)
end function
end module""")
    _path, _analysis, _originals, _demanded, proof = setup(tmp_path, f"value={name}(values)", source=source)
    assert not proof.barriers, proof.public()["barriers"]
    assert f"work::{name}" in {call.callee for call in proof.calls}
    assert any(item.storage.resource == "work::later" and item.kind == "write" for item in proof.accesses)


def test_plain_host_intrinsic_keeps_original_input_reads(tmp_path):
    source = SOURCE.replace("implicit none\n", "implicit none\nintrinsic norm2\n")
    _path, _analysis, _originals, _demanded, proof = setup(tmp_path, "value=norm2(values)", source=source)
    assert not proof.barriers, proof.public()["barriers"]
    assert any(item.storage.resource == "work::input" and item.kind == "read" for item in proof.accesses)


def test_new_control_graph_keeps_registered_source_and_policy_freshness(tmp_path, monkeypatch):
    from compiler.frontend import resource_reachability

    path, analysis, originals, demanded, proof = setup(tmp_path, "do\nvalue=norm2(values)\nexit\nenddo")
    assert not proof.barriers
    with pytest.raises(CompilationError):
        copy(proof).validate(analysis, "work::owner", originals, demanded)
    with pytest.raises(CompilationError):
        proof.validate(SourceEffects([path]), "work::owner", originals, demanded)
    monkeypatch.setattr(resource_reachability, "VERSION", resource_reachability.VERSION + 1)
    with pytest.raises(CompilationError, match="budgets changed"):
        proof.validate(analysis, "work::owner", originals, demanded)
