"""Joined source-helper proofs supply descriptor evidence, never execution."""

from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend.descriptor_windows import _Check, prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "stage_owner::advance"
CHILD = "numerics::update"
ROOT = "fields::viscosity"


def fixture(tmp_path, *, helper="g(1,1)=x\ng(2,1)=x+1.0_8", extra="", tail="continue",
            opening="parallel do private(i,g,value) schedule(runtime)", ending="end parallel do",
            environment=False):
    environment_use = "use ieee_exceptions" if environment else ""
    environment_declaration = "logical::modes(size(ieee_all))" if environment else ""
    environment_before = """!$omp parallel
call ieee_get_halting_mode(ieee_all,modes)
call ieee_set_halting_mode(ieee_all,.false.)
!$omp end parallel""" if environment else ""
    environment_after = """!$omp parallel
call ieee_set_halting_mode(ieee_all,modes)
!$omp end parallel""" if environment else ""
    path = tmp_path / "renamed.f90"
    path.write_text(f"""module fields
real(8),allocatable::viscosity(:)
end module
module numerics
use fields,only:output=>viscosity
contains
subroutine update(a,n)
{environment_use}
real(8),intent(in)::a(-2:)
integer,intent(in)::n
integer::i
real(8)::g(2,2),value
{environment_declaration}
{environment_before}
!$omp {opening}
do i=1,n
call gradient(g,i)
call compose(value,g)
output(i)=value
end do
!$omp {ending}
{environment_after}
{tail}
contains
pure subroutine gradient(g,index)
real(8),intent(out)::g(2,2)
integer,intent(in)::index
real(8)::x
x=a(index)
{helper}
end subroutine
pure subroutine compose(value,g)
real(8),intent(out)::value
real(8),intent(in)::g(2,2)
value=dot_product(g(:,1),g(:,1))
end subroutine
{extra}
end subroutine
end module
module stage_owner
use numerics,only:update
use fields,only:viscosity
contains
subroutine advance(a,n)
real(8),allocatable,intent(in)::a(:)
integer,intent(in)::n
call unknown_prefix()
call update(a,n)
if(allocated(viscosity)) deallocate(viscosity)
end subroutine
end module
""")
    analysis = SourceEffects([path], summary_cache=tmp_path / "cache")
    graph = analysis.structure(ENTRY)
    call = next(node for node in analysis.routines[ENTRY].execution.content
                if type(node).__name__ == "Call_Stmt" and str(node.items[0]).lower() == "update")
    return path, analysis, (graph.node_id(call),)


def prove(analysis, selected):
    return prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",), module_roots=(ROOT,))


def test_private_group_registration_admits_original_pure_helper_closure_only(tmp_path):
    path, analysis, selected = fixture(tmp_path)
    original = path.read_bytes()
    before = analysis.summarize(CHILD)
    assert not before["complete"]
    registries = {name: dict(getattr(analysis, name)) for name in
                  ("_joined_completions", "_numerical_completions", "_native_environments", "_native_functions")}
    proof = prove(analysis, selected)
    assert proof.validate(analysis, ENTRY, selected) is proof
    public = proof.public()
    join, = public["native_joined_descriptor_requirements"]
    assert join["proof_role"] == "numerical_source_completion"
    assert join["whole_original_group_scanned"]
    assert join["retains_original_team_and_directives"]
    assert not join["execution_authorized"]
    assert not join["entry_definition_authority"]
    assert not join["gpu_legality_established"]
    assert not join["internal_cuts_authorized"]
    assert {CHILD, CHILD + "::gradient", CHILD + "::compose"} <= set(public["source_call_closure"])
    assert all(getattr(analysis, name) == registry for name, registry in registries.items())
    assert analysis.stable_module_allocatables == frozenset()
    assert analysis._allocation_authorizations == {}
    assert analysis.summarize(CHILD) == before
    assert path.read_bytes() == original


@pytest.mark.parametrize("helper", [
    "call unavailable(g)",
    "!$omp task\ng(1,1)=x\n!$omp end task",
    "call gradient(g,index)",
    "call nested(g,x)",
])
def test_unknown_recursive_or_hidden_team_helper_is_not_descriptor_authority(tmp_path, helper):
    extra = """pure subroutine nested(g,x)
real(8),intent(out)::g(2,2)
real(8),intent(in)::x
integer::j
!$omp parallel do private(j)
do j=1,2
g(j,1)=x
end do
!$omp end parallel do
end subroutine"""
    _path, analysis, selected = fixture(tmp_path, helper=helper, extra=extra)
    with pytest.raises(CompilationError, match="whole-closure|recursive|unknown|OpenMP"):
        prove(analysis, selected)
    assert not analysis._joined_completions
    assert not analysis._numerical_completions


@pytest.mark.parametrize("tail", ["call opaque(output)", "deallocate(output)", "output=1.0_8"])
def test_complete_original_child_rescan_cannot_hide_later_storage_effects(tmp_path, tail):
    _path, analysis, selected = fixture(tmp_path, tail=tail)
    with pytest.raises(CompilationError, match="unknown selected storage|whole allocatable"):
        prove(analysis, selected)


def test_private_join_tokens_cannot_bless_original_summary_or_foreign_analysis(tmp_path):
    _path, analysis, selected = fixture(tmp_path)
    checker = _Check(analysis)
    child = analysis.routines[CHILD]
    graph = analysis.structure(CHILD)
    checker.joined_groups(child, graph, (graph.root_id,))
    token, = checker.join_analysis._numerical_completions.values()
    assert not analysis._numerical_completions
    with pytest.raises(CompilationError, match="registered original source authority"):
        token.validate(analysis)
    foreign = copy(checker.join_analysis)
    foreign._numerical_completions = {token.identity: replace(token)}
    with pytest.raises(CompilationError, match="registered original source authority"):
        token.validate(foreign)
    proof = prove(analysis, selected)
    child.execution.content.pop()
    with pytest.raises(CompilationError, match="original source|authority"):
        proof.validate(analysis, ENTRY, selected)


@pytest.mark.parametrize(("depth", "accepted"), [(8, True), (9, False)])
def test_expression_limit_counts_calls_without_argument_container_wrappers(tmp_path, depth, accepted):
    expression = "x"
    for _ in range(depth):
        expression = "abs(" + expression + ")"
    _path, analysis, selected = fixture(tmp_path, helper="g(1,1)=" + expression)
    if accepted:
        assert prove(analysis, selected).public()["limits"]
    else:
        with pytest.raises(CompilationError, match="expression depth"):
            prove(analysis, selected)
    assert analysis.depth_limit == 8


def test_parentheses_and_keyword_labels_do_not_spend_expression_depth(tmp_path):
    expression = "real(a=sqrt(max(1.0_8-x*cos(1.0_8+x),0.0_8)),kind=8)"
    _path, analysis, selected = fixture(tmp_path, helper="g(1,1)=" + "(" * 12 + expression + ")" * 12)
    assert prove(analysis, selected).public()["schema_version"] == 4


def test_original_native_environment_calls_retain_all_thread_and_execution_requirements(tmp_path):
    _path, analysis, selected = fixture(tmp_path, environment=True)
    assert not analysis._native_environments
    assert not analysis._native_environment_states
    public = prove(analysis, selected).public()
    operations = public["native_environment_descriptor_requirements"]
    assert len(operations) == 3
    assert {item["intrinsic"].rsplit("::", 1)[-1] for item in operations} == {
        "ieee_get_halting_mode", "ieee_set_halting_mode"}
    assert all(item["requires_original_execution"] and item["requires_original_thread_participation"]
               and item["requires_completed_device_work"] and item["descriptor_evidence_only"]
               and not item["allocation_or_escape"] and not item["execution_authorized"]
               and not item["gpu_legality_established"] for item in operations)
    assert len(public["native_environment_state_requirements"]) == 1
    assert len(public["native_joined_descriptor_requirements"]) == 3
    assert not analysis._native_environments
    assert not analysis._native_environment_states


@pytest.mark.parametrize("change", [
    lambda text: text.replace("use ieee_exceptions", "use foreign_intrinsics"),
    lambda text: text.replace("ieee_set_halting_mode(ieee_all,.false.)", "ieee_set_halting_mode(ieee_all,output)"),
])
def test_spelling_or_managed_array_cannot_acquire_native_environment_authority(tmp_path, change):
    path, _analysis, _selected = fixture(tmp_path, environment=True)
    path.write_text(change(path.read_text()))
    analysis = SourceEffects([path])
    graph = analysis.structure(ENTRY)
    call = next(node for node in analysis.routines[ENTRY].execution.content
                if type(node).__name__ == "Call_Stmt" and str(node.items[0]).lower() == "update")
    with pytest.raises(CompilationError, match="whole-closure|unknown|environment|unproved"):
        prove(analysis, (graph.node_id(call),))
    assert not analysis._native_environments
