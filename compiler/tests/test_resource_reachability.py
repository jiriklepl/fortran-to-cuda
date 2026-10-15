"""Source storage demands precede complete native-effect materialization."""
from copy import copy

import pytest

from compiler.frontend.resource_reachability import (
    ResourceReachabilityBudgetExceeded,
    prove_resource_reachability,
)
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module data
implicit none
type profiles
real(8),allocatable::flux(:)
contains
procedure::unrelated
end type
type(profiles)::current
real(8),allocatable::viscosity(:)
contains
subroutine unrelated(self)
class(profiles)::self
end subroutine
end module
module work
use data
implicit none
contains
subroutine owner(old,star,n,enabled)
real(8),intent(in)::old(:)
real(8),intent(inout)::star(:)
integer,intent(in)::n
logical,intent(in)::enabled
call native(star,n,enabled)
end subroutine
subroutine native(a,n,enabled)
real(8),intent(inout)::a(:)
integer,intent(in)::n
logical,intent(in)::enabled
real(8)::value
if(enabled)then
value=sum(current%flux(1:n))
call add(a,value)
endif
end subroutine
subroutine add(a,value)
real(8),intent(inout)::a(:)
real(8),intent(in)::value
integer::i
!$omp parallel do private(i)
do i=1,size(a)
a(i)=a(i)+value
enddo
!$omp end parallel do
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, **kwargs):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path], **kwargs)
    routine = analysis.routines["work::owner"]
    selected = tuple(node for node in routine.execution.content if type(node).__name__ == "Call_Stmt")
    demanded = (routine.scope.bindings["old"], analysis.modules["data"].bindings["viscosity"])
    return path, analysis, selected, demanded


def test_native_profile_reachability_precedes_summarize_and_keeps_alias_obligations(tmp_path, monkeypatch):
    _path, analysis, selected, demanded = fixture(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("resource demand called the complete effect summarizer")
    monkeypatch.setattr(analysis, "summarize", forbidden)
    proof = prove_resource_reachability(analysis, "work::owner", selected, demanded)
    assert proof.validate(analysis, "work::owner", selected, demanded) is proof
    public = proof.public()
    assert public["selection_covered"], public["barriers"]
    assert all(not effect["may_access"] for effect in public["demanded_effects"])
    fields = [access for access in public["accesses"] if access["storage"]["component_path"] == ["flux"]]
    assert fields
    assert fields[0]["kind"] == "read"
    assert "allocatable" in fields[0]["storage"]["attributes"]
    assert {call["callee"] for call in public["calls"]} == {"work::native", "work::add"}
    obligations = public["storage_disjointness_obligations"]
    assert any(item["native_storage"]["resource"] == "argument::star" and item["demanded_resource"] == "argument::old"
               and "write" in item["actions"] for item in obligations)
    assert any(item["native_storage"]["resource"] == "data::current%flux" for item in obligations)
    assert all(not item["discharged"] for item in obligations)
    assert not public["execution_authorized"]
    assert not public["whole_procedure_effects_complete"]
    assert not public["alias_authorized"]
    assert not public["capture_authorized"]


def test_hidden_module_write_and_forwarded_managed_actual_are_visible(tmp_path):
    source = SOURCE.replace("call native(star,n,enabled)", "call native(old,n,enabled)").replace(
        "real(8),intent(in)::old(:)", "real(8),intent(inout)::old(:)").replace(
        "value=sum(current%flux(1:n))", "value=sum(current%flux(1:n))\nviscosity(1)=value")
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    public = prove_resource_reachability(analysis, "work::owner", selected, demanded).public()
    effects = {item["resource"]: item["may_access"] for item in public["demanded_effects"]}
    assert {"read", "write"} <= set(effects["argument::old"])
    assert "write" in effects["data::viscosity"]


@pytest.mark.parametrize(("replacement", "kind"), [
    ("call absent(a)", "call"),
    ("call callback(a)", "call"),
    ("allocate(viscosity(n))", "association"),
    ("deallocate(viscosity)", "association"),
    ("stop", "statement"),
])
def test_unknown_execution_or_lifetime_is_a_precise_barrier(tmp_path, replacement, kind):
    source = SOURCE.replace("call add(a,value)", replacement)
    if "callback" in replacement:
        source = source.replace("real(8)::value", "real(8)::value\nexternal::callback")
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    public = prove_resource_reachability(analysis, "work::owner", selected, demanded).public()
    assert not public["selection_covered"]
    assert any(item["kind"] == kind and replacement.split("(")[0].lower() in item["source"].lower()
               for item in public["barriers"])


def test_pointer_component_is_not_mistaken_for_owned_allocation(tmp_path):
    source = SOURCE.replace("real(8),allocatable::flux(:)", "real(8),pointer::flux(:)")
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    public = prove_resource_reachability(analysis, "work::owner", selected, demanded).public()
    assert not public["selection_covered"]
    assert any(item["kind"] == "association" for item in public["barriers"])


def test_foreign_copied_or_changed_demand_cannot_borrow_proof(tmp_path):
    _path, analysis, selected, demanded = fixture(tmp_path)
    proof = prove_resource_reachability(analysis, "work::owner", selected, demanded)
    with pytest.raises(CompilationError):
        copy(proof).validate(analysis, "work::owner", selected, demanded)
    with pytest.raises(CompilationError):
        proof.validate(analysis, "work::owner", selected, (copy(demanded[0]), demanded[1]))
    other = SourceEffects([_path])
    with pytest.raises(CompilationError):
        proof.validate(other, "work::owner", selected, demanded)
    with pytest.raises(CompilationError):
        prove_resource_reachability(analysis, "work::owner", (copy(selected[0]),), demanded)


@pytest.mark.parametrize("mutation", ["file", "binding", "import", "budget", "component"])
def test_original_source_and_authority_mutations_invalidate(tmp_path, mutation):
    path, analysis, selected, demanded = fixture(tmp_path)
    proof = prove_resource_reachability(analysis, "work::owner", selected, demanded)
    if mutation == "file":
        path.write_text(SOURCE + "\n! changed\n")
    elif mutation == "binding":
        demanded[0].attributes |= {"pointer"}
    elif mutation == "import":
        analysis.routines["work::native"].scope.imports["current"] = ("other", "current")
    elif mutation == "component":
        schema = analysis.modules["data"].component_types["profiles"]
        schema.members.clear()
    else:
        analysis.operation_limit += 1
    with pytest.raises(CompilationError):
        proof.validate(analysis, "work::owner", selected, demanded)


def test_scan_budget_exhaustion_is_not_an_incomplete_success(tmp_path):
    _path, analysis, selected, demanded = fixture(tmp_path, operations=2)
    with pytest.raises(ResourceReachabilityBudgetExceeded, match="budget"):
        prove_resource_reachability(analysis, "work::owner", selected, demanded)
    assert not getattr(analysis, "_resource_reachability_proofs", {})


def test_expression_actual_is_read_at_original_call_not_invented_storage(tmp_path):
    source = SOURCE.replace("call add(a,value)", "call add(a,-2.0_8*value)")
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    public = prove_resource_reachability(analysis, "work::owner", selected, demanded).public()
    assert public["selection_covered"], public["barriers"]
    assert not any(item["native_storage"]["resource"] == "argument::value"
                   for item in public["storage_disjointness_obligations"])


def test_defined_operator_and_foreign_storage_constructor_are_barriers(tmp_path):
    source = SOURCE.replace("value=sum(current%flux(1:n))", "value=1.0_8 .opaque. 2.0_8")
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    public = prove_resource_reachability(analysis, "work::owner", selected, demanded).public()
    assert not public["selection_covered"]
    assert any(item["kind"] == "expression" for item in public["barriers"])


def test_loop_iterator_write_is_part_of_scalar_demand(tmp_path):
    _path, analysis, _selected, _demanded = fixture(tmp_path)
    routine = analysis.routines["work::add"]
    selected = tuple(routine.execution.content)
    demanded = (routine.scope.bindings["i"],)
    public = prove_resource_reachability(analysis, routine.qualified, selected, demanded).public()
    assert "write" in public["demanded_effects"][0]["may_access"]


def test_mutation_during_scan_cannot_publish_registered_proof(tmp_path, monkeypatch):
    from compiler.frontend import resource_reachability
    _path, analysis, selected, demanded = fixture(tmp_path)
    original = resource_reachability._Scan.routine
    def mutate_after_scan(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if not self.active:
            demanded[0].attributes |= {"pointer"}
    monkeypatch.setattr(resource_reachability._Scan, "routine", mutate_after_scan)
    with pytest.raises(CompilationError):
        prove_resource_reachability(analysis, "work::owner", selected, demanded)
    assert not getattr(analysis, "_resource_reachability_proofs", {})


def test_sealed_session_retains_exact_original_binding_and_component_authority(tmp_path):
    _path, analysis, selected, demanded = fixture(tmp_path)
    analysis.seal_session()
    proof = prove_resource_reachability(analysis, "work::owner", selected, demanded)
    assert proof.validate(analysis, "work::owner", selected, demanded) is proof
    assert proof.public()["selection_covered"]
    with pytest.raises(CompilationError):
        demanded[0].attributes |= {"pointer"}


@pytest.mark.parametrize(("assignment", "reallocates"), [
    ("current%flux = 0.0_8", True),
    ("current%flux(1) = 0.0_8", False),
])
def test_allocatable_component_whole_assignment_and_element_have_distinct_barriers(tmp_path, assignment, reallocates):
    source = SOURCE.replace("value=sum(current%flux(1:n))", assignment + "\nvalue=1.0_8")
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    public = prove_resource_reachability(analysis, "work::owner", selected, demanded).public()
    assert any(item["kind"] == "allocation" for item in public["barriers"]) is reallocates
    assert public["selection_covered"] is not reallocates


def test_forwarded_procedure_dummy_is_explicit_callback_barrier(tmp_path):
    source = SOURCE.replace("call native(star,n,enabled)", "call native(star,n,enabled,add)").replace(
        "subroutine native(a,n,enabled)", "subroutine native(a,n,enabled,callback)").replace(
        "real(8)::value", "real(8)::value\nexternal::callback")
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    public = prove_resource_reachability(analysis, "work::owner", selected, demanded).public()
    assert not public["selection_covered"]
    assert any(item["kind"] == "callback" for item in public["barriers"])


def test_character_entry_length_function_cannot_hide_effects(tmp_path):
    source = SOURCE.replace("real(8)::value", "real(8)::value\ncharacter(len=length())::scratch").replace(
        "end module\nmodule work", "integer function length()\nviscosity(1)=0.0_8\nlength=2\nend function\nend module\nmodule work")
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    public = prove_resource_reachability(analysis, "work::owner", selected, demanded).public()
    assert not public["selection_covered"]
    assert any(item["kind"] == "specification" for item in public["barriers"])


def test_late_module_reference_does_not_skip_storage_association_declarations(tmp_path):
    source = SOURCE.replace("real(8),allocatable::viscosity(:)", "real(8)::viscosity(3)\ncommon /shared/ viscosity")
    source = source.replace("call add(a,value)", "viscosity(1)=value")
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    public = prove_resource_reachability(analysis, "work::owner", selected, demanded).public()
    assert not public["selection_covered"]
    assert any(item["kind"] == "storage_association" for item in public["barriers"])


def test_source_module_c_binding_is_not_ordinary_native_call_authority(tmp_path):
    source = SOURCE.replace("subroutine add(a,value)", "subroutine add(a,value) bind(C)")
    _path, analysis, selected, demanded = fixture(tmp_path, source)
    public = prove_resource_reachability(analysis, "work::owner", selected, demanded).public()
    assert not public["selection_covered"]
    assert any(item["kind"] == "external_call" for item in public["barriers"])
