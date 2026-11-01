"""A native child exit preserves descriptors only after complete source scanning."""
from copy import copy

import pytest

from compiler.frontend.descriptor_windows import _Check
from compiler.frontend.native_prefix import prove_native_prefix
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_joined_closures import fixture as joined_fixture
from compiler.tests.test_descriptor_joined_closures import prove as joined_prove
from compiler.tests.test_descriptor_window_proofs import ENTRY, fixture, prove


@pytest.mark.parametrize("statement", ["return", "if(size(a)==0) return", "if(size(a)==0) then\nreturn\nend if"])
def test_complete_callee_bare_returns_do_not_trim_following_source(tmp_path, statement):
    path, analysis, _graph, selection = fixture(tmp_path, child=statement + "\na(1)=a(1)+1.0_8")
    before = path.read_bytes()
    proof = prove(analysis, selection)
    public = proof.public()
    item, = public["native_return_descriptor_requirements"]
    assert item["procedure"] == "intervals::read_modify"
    assert item["original_statement"] == "RETURN"
    assert item["complete_original_callee_required"]
    assert item["all_following_original_nodes_scanned"]
    assert not item["execution_authorized"]
    assert not item["entry_definition_authority"]
    assert not item["payload_access_authorized"]
    assert not public["reachability_authority"]
    assert not public["entry_definition_authority"]
    assert proof.validate(analysis, ENTRY, selection) is proof
    assert path.read_bytes() == before


def test_normal_child_return_inside_calling_openmp_team_does_not_exit_the_team(tmp_path):
    _path, analysis, selection = joined_fixture(tmp_path,
        helper="if(index<1) return\ng(1,1)=x\ng(2,1)=x+1.0_8")
    record = joined_prove(analysis, selection).public()
    item, = record["native_return_descriptor_requirements"]
    assert item["procedure"].endswith("::gradient")
    assert record["native_joined_descriptor_requirements"]
    assert not record["execution_authorized"]


@pytest.mark.parametrize("statement", ["return", "if(flag) return"])
def test_selected_root_return_remains_a_control_boundary(tmp_path, statement):
    _path, analysis, _graph, selection = fixture(tmp_path, selected=statement)
    with pytest.raises(CompilationError, match="bare complete-callee exit"):
        prove(analysis, selection)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_alternate_return_does_not_gain_control_authority(tmp_path):
    _path, analysis, _graph, selection = fixture(tmp_path, child="return 1")
    with pytest.raises(CompilationError, match="bare complete-callee exit"):
        prove(analysis, selection)


@pytest.mark.parametrize("body", [
    "!$omp parallel\nreturn\n!$omp end parallel",
    "!$omp parallel do private(i)\ndo i=1,3\nif(i==2) return\na(i)=1.0_8\nend do\n!$omp end parallel do",
])
def test_return_lexically_exiting_an_original_openmp_region_stays_closed(tmp_path, body):
    _path, analysis, _graph, selection = fixture(tmp_path, child=body,
        child_decl="real(8),intent(inout)::a(:)\ninteger::i")
    with pytest.raises(CompilationError):
        prove(analysis, selection)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_native_prefix_directive_authority_cannot_bypass_original_team_return_restriction(tmp_path):
    _path, analysis, _graph, selection = fixture(tmp_path,
        child_decl="real(8),intent(inout)::a(:)\ninteger::i\nreal(8)::total",
        child="total=0\n!$omp parallel do private(i) reduction(+:total)\ndo i=1,3\n"
              "if(i==2) return\ntotal=total+a(i)\nend do\n!$omp end parallel do")
    prefix = prove_native_prefix(analysis, "intervals::read_modify")
    assert prefix.public()["synchronous_return"]
    with pytest.raises(CompilationError, match="outside original OpenMP"):
        prove(analysis, selection)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("tail", ["deallocate(a)", "allocate(a(3))", "a=[1.0_8,2.0_8]", "call absent(a)"])
def test_unreachable_lifetime_and_escape_effects_after_return_are_still_scanned(tmp_path, tail):
    _path, analysis, _graph, selection = fixture(tmp_path, child="return\n" + tail,
        child_decl="real(8),allocatable,intent(inout)::a(:)")
    with pytest.raises(CompilationError):
        prove(analysis, selection)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_local_allocatable_finalization_requires_separate_authority(tmp_path):
    _path, analysis, _graph, selection = fixture(tmp_path, child="return",
        child_decl="real(8),intent(inout)::a(:)\nreal(8),allocatable::local(:)")
    with pytest.raises(CompilationError, match="allocation finalization"):
        prove(analysis, selection)


def test_return_does_not_reset_shared_operation_budget(tmp_path):
    """Keep the original node while checking its counter and complete tail."""
    _path, analysis, _graph, _selection = fixture(tmp_path, child="return\na(1)=a(1)+1.0_8")
    check = _Check(analysis)
    check.operations = analysis.operation_limit
    check.active.append(ENTRY)
    check.procedure("intervals::read_modify")
    assert check.operations == analysis.operation_limit + 2
    assert "intervals::read_modify" in check.checked


def test_retained_return_proof_rejects_replaced_original_child_execution(tmp_path):
    _path, analysis, _graph, selection = fixture(tmp_path, child="return")
    proof = prove(analysis, selection)
    child = analysis.routines["intervals::read_modify"]
    child.execution = copy(child.execution)
    with pytest.raises(CompilationError):
        proof.validate(analysis, ENTRY, selection)
