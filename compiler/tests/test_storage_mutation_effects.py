"""Descriptor lifetime does not require numerical or readonly payload proof."""
from copy import copy
from dataclasses import replace

import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError


def analysis(tmp_path, body="a(:)=a(:)+calculate(a)", *, declarations="", helpers="", module_declarations="",
             argument="real(8),allocatable,intent(inout)::a(:)"):
    path = tmp_path / "storage.f90"
    path.write_text(f"""module storage
implicit none
{module_declarations}
contains
subroutine advance(a)
{argument}
{declarations}
{body}
end subroutine
{helpers}
end module
""")
    return path, SourceEffects([path])


def token(tmp_path, body="a(:)=a(:)+1", **kwargs):
    path, source = analysis(tmp_path, body, **kwargs)
    return path, source, prove_storage_mutation_effects(source, "storage::advance")


def test_nonpure_function_payload_writes_and_private_math_do_not_require_readonly_or_numerical_proofs(tmp_path):
    _path, source, proof = token(tmp_path, "a(:)=a(:)+calculate(a)", module_declarations="real(8)::counter",
        helpers="""real(8) function calculate(x)
real(8),intent(in)::x(:)
real(8)::scratch(2)
scratch=2
counter=counter+1
call change(scratch)
calculate=sum(x)+scratch(1)
end function
subroutine change(x)
real(8),intent(inout)::x(:)
x(:)=x(:)*3
end subroutine""")
    assert proof.descriptor_unchanged("argument::a")
    assert proof.descriptor_unchanged("storage::counter")
    public = proof.public()
    assert {part["procedure"] for part in public["closure"]} == {"storage::advance", "storage::calculate", "storage::change"}
    assert not public["mutations"]
    assert not source._native_functions
    for flag in ("payload_access_authorized", "entry_definition_authority", "capture_authorized",
                 "execution_authorized", "motion_authorized", "gpu_legality_established", "alias_authority"):
        assert public[flag] is False
    assert not proof.descriptor_unchanged("absent::unproved")


@pytest.mark.parametrize("body", ["a=1", "allocate(a(4))", "deallocate(a)"])
def test_explicit_and_implicit_replacement_are_descriptor_mutations(tmp_path, body):
    _path, _source, proof = token(tmp_path, body)
    assert not proof.descriptor_unchanged("argument::a")
    assert any(event["resource"] == "argument::a" for event in proof.public()["mutations"])


@pytest.mark.parametrize("body", ["a(1)=1", "a(:)=1", "a(1:size(a):2)=2"])
def test_numeric_element_and_section_writes_are_not_whole_reallocation_or_definition_credit(tmp_path, body):
    _path, _source, proof = token(tmp_path, body)
    assert proof.descriptor_unchanged("argument::a")
    assert proof.public()["definition_kills"] == []
    assert proof.public()["entry_definition_authority"] is False


def test_hidden_module_reallocation_is_retained_through_nonallocatable_dummy_and_renamed_import(tmp_path):
    path, source = analysis(tmp_path, "call hidden(a)", module_declarations="real(8),allocatable::global(:)",
        helpers="""subroutine hidden(x)
real(8),intent(inout)::x(:)
deallocate(global)
x(:)=1
end subroutine""")
    proof = prove_storage_mutation_effects(source, "storage::advance")
    # The top dummy could be associated with the hidden module allocation.
    # Source spelling alone supplies no caller disjointness authority.
    assert not proof.descriptor_unchanged("argument::a")
    assert not proof.descriptor_unchanged("storage::global")
    assert any(event["resource"] == "storage::global" for event in proof.public()["mutations"])
    # The same allocation passed through a nonallocatable dummy remains the
    # module root. Formal attributes cannot erase this second access route.
    path.write_text(path.read_text().replace("call hidden(a)", "call hidden(global)"))
    source = SourceEffects([path])
    proof = prove_storage_mutation_effects(source, "storage::advance")
    assert not proof.descriptor_unchanged("storage::global")


def test_generic_source_actual_projection_maps_mutation_and_both_move_alloc_slots(tmp_path):
    _path, _source, proof = token(tmp_path, "call exchange(a,other)",
        module_declarations="real(8),allocatable::other(:)", helpers="""subroutine exchange(left,right)
real(8),allocatable,intent(inout)::left(:),right(:)
call move_alloc(left,right)
end subroutine""")
    assert not proof.descriptor_unchanged("argument::a")
    assert not proof.descriptor_unchanged("storage::other")
    assert {event["resource"] for event in proof.public()["mutations"] if event["kind"] == "move_alloc"} == {
        "argument::a", "storage::other"}


def test_import_rename_keeps_hidden_module_storage_identity(tmp_path):
    path = tmp_path / "renamed.f90"
    path.write_text("""module memory
real(8),allocatable::global(:)
contains
subroutine hidden(x)
real(8),intent(inout)::x(:)
deallocate(global)
end subroutine
end module
module storage
use memory,only:alias=>global,hidden
contains
subroutine advance(a)
real(8),allocatable,intent(inout)::a(:)
call hidden(alias)
end subroutine
end module""")
    proof = prove_storage_mutation_effects(SourceEffects([path]), "storage::advance")
    assert not proof.descriptor_unchanged("memory::global")
    assert not proof.descriptor_unchanged("argument::a")
    assert "alias" not in {event["resource"] for event in proof.public()["mutations"]}


def test_multiple_formals_aliasing_same_allocation_do_not_hide_replacement(tmp_path):
    _path, _source, proof = token(tmp_path, "call replace(a,a)", helpers="""subroutine replace(x,y)
real(8),allocatable,intent(inout)::x(:),y(:)
deallocate(y)
end subroutine""")
    assert not proof.descriptor_unchanged("argument::a")
    assert {event["resource"] for event in proof.public()["mutations"]} == {"argument::a"}


def test_sibling_host_dummy_is_not_rebound_as_same_named_child_formal(tmp_path):
    path = tmp_path / "host.f90"
    path.write_text("""module host
contains
subroutine advance(a,b)
real(8),allocatable,intent(inout)::a(:),b(:)
call middle(a)
contains
subroutine middle(b)
real(8),intent(inout)::b(:)
b(1)=1
call mutation()
end subroutine
subroutine mutation()
deallocate(b)
end subroutine
end subroutine
end module""")
    proof = prove_storage_mutation_effects(SourceEffects([path]), "host::advance")
    assert not proof.descriptor_unchanged("argument::a")  # Caller disjointness is still independent.
    assert not proof.descriptor_unchanged("argument::b")
    assert {event["resource"] for event in proof.public()["mutations"]} == {"argument::b"}


@pytest.mark.parametrize("allocatable", [False, True])
def test_intent_out_entry_definition_kill_is_separate_from_allocation_kill(tmp_path, allocatable):
    attrs = ",allocatable" if allocatable else ""
    _path, _source, proof = token(tmp_path, "call reset(a)", helpers=f"""subroutine reset(x)
real(8){attrs},intent(out)::x(:)
end subroutine""")
    assert proof.descriptor_unchanged("argument::a") is not allocatable
    assert any(event["resource"] == "argument::a" for event in proof.public()["definition_kills"])
    assert proof.public()["entry_definition_authority"] is False


def test_all_branches_and_original_tail_after_return_are_scanned_without_evaluation(tmp_path):
    _path, _source, proof = token(tmp_path, """if(.false.)then
deallocate(a)
endif
return
allocate(a(3))""")
    assert not proof.descriptor_unchanged("argument::a")
    assert {event["kind"] for event in proof.public()["mutations"]} == {"allocate", "deallocate"}


@pytest.mark.parametrize("saved", [False, True])
def test_ordinary_and_saved_local_allocatable_lifecycles_are_distinct_and_do_not_mutate_other_roots(tmp_path, saved):
    attrs = ",save" if saved else ""
    _path, _source, proof = token(tmp_path, "allocate(work(2))\nwork(:)=1\na(1)=work(1)",
                                declarations=f"real(8),allocatable{attrs}::work(:)")
    assert proof.descriptor_unchanged("argument::a") is not saved
    implicit = [event for event in proof.public()["mutations"] if event["kind"].startswith("local allocation implicit")]
    assert bool(implicit) is not saved


def test_write_is_observable_but_does_not_authorize_live_execution_or_mutate_numeric_descriptors(tmp_path):
    _path, _source, proof = token(tmp_path, "write(*,*)a(1)\nprint *,size(a)")
    assert proof.descriptor_unchanged("argument::a")
    assert len(proof.public()["observables"]) == 2
    assert proof.public()["execution_authorized"] is False


@pytest.mark.parametrize("body", ["call unknown(a)", "a(1)=hidden(a(1))", "call unknown()",
    "write(*,*,asynchronous='yes')a", "read(*,*)a", "!$omp parallel\na(1)=1\n!$omp end parallel"])
def test_unknown_call_escape_async_and_unjoined_activity_cannot_be_stable(tmp_path, body):
    _path, _source, proof = token(tmp_path, body)
    assert not proof.descriptor_unchanged("argument::a")
    assert proof.public()["frontiers"]


@pytest.mark.parametrize("attributes", [",target", ",pointer", ",asynchronous", ",volatile"])
def test_source_declaration_alias_uncertainty_is_not_a_capture_fact(tmp_path, attributes):
    _path, _source, proof = token(tmp_path, "a(1)=1", argument=f"real(8){attributes},intent(inout)::a(:)")
    assert not proof.descriptor_unchanged("argument::a")
    assert proof.public()["frontiers"]


@pytest.mark.parametrize("declarations", ["real(8)::local\ndata local/1/", "real(8)::local\nsave local",
    "real(8)::local\ncommon /shared/local", "real(8)::local,other\nequivalence(local,other)",
    "type opaque\nreal(8),allocatable::component(:)\nend type\ntype(opaque)::local"])
def test_implicit_persistent_alias_and_derived_finalization_lifecycles_decline(tmp_path, declarations):
    _path, _source, proof = token(tmp_path, declarations=declarations)
    assert not proof.descriptor_unchanged("argument::a")


def test_function_in_automatic_bound_is_not_ignored_at_entry(tmp_path):
    _path, _source, proof = token(tmp_path, declarations="real(8)::scratch(bound())",
        helpers="""integer function bound()
bound=3
end function""")
    # The existing original-call issuer deliberately excludes specification
    # expression calls. They remain an explicit lifecycle frontier here.
    assert not proof.descriptor_unchanged("argument::a")
    assert proof.public()["frontiers"]


def test_recursive_closure_and_procedure_limit_do_not_publish_partial_projection(tmp_path):
    _path, source = analysis(tmp_path, "call recursive_child(a)", helpers="""recursive subroutine recursive_child(x)
real(8),intent(inout)::x(:)
call recursive_child(x)
end subroutine""")
    with pytest.raises(CompilationError, match="closure/recursion"):
        prove_storage_mutation_effects(source, "storage::advance")
    assert not getattr(source, "_storage_mutation_effects", {})


def test_uncalled_contains_body_does_not_mint_source_budget(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _p, _a, before = token(first)
    _p, _a, after = token(second, helpers="subroutine unused()\ninteger::i\n" + "i=1\n"*100 + "end subroutine")
    assert before.public()["budget"]["source"] == after.public()["budget"]["source"]
    assert len(after.public()["closure"]) == 1


def test_allocatable_function_result_is_not_claimed_deallocated_at_function_exit(tmp_path):
    _path, _source, proof = token(tmp_path, "a(:)=give()", helpers="""function give()result(value)
real(8),allocatable::value(:)
allocate(value(2))
value(:)=1
end function""")
    assert not proof.descriptor_unchanged("argument::a")
    assert any("result transfer" in event["kind"] for event in proof.public()["frontiers"])
    assert not any(event["kind"].startswith("local allocation implicit") for event in proof.public()["mutations"])


@pytest.mark.parametrize("fault", ["associations", "actuals", "arguments", "target"])
def test_cached_original_call_edges_remain_bound_to_parser_actuals_and_bindings(tmp_path, fault):
    _path, source, proof = token(tmp_path, "call change(a)", helpers="""subroutine change(x)
real(8),intent(inout)::x(:)
x(:)=1
end subroutine""")
    edge, = source._original_call_associations.values()
    if fault == "associations":
        association, = edge.associations
        altered = replace(association, actual=copy(association.actual))
        object.__setattr__(edge, "associations", (altered,))
    elif fault == "actuals":
        object.__setattr__(edge, "actuals", (copy(edge.actuals[0]),))
    elif fault == "arguments":
        argument, = edge.original_arguments
        object.__setattr__(edge, "original_arguments", (replace(argument, actual=copy(argument.actual)),))
    else:
        object.__setattr__(edge, "target", "other")
    with pytest.raises(CompilationError, match="consumed original call association"):
        proof.validate()


def test_defined_numeric_assignment_cannot_hide_module_reallocation(tmp_path):
    path = tmp_path / "defined.f90"
    path.write_text("""module storage
real(8),allocatable::global(:)
interface assignment(=)
module procedure assign
end interface
contains
subroutine advance(a)
real(8),allocatable,intent(inout)::a(:)
a(1)=.true.
end subroutine
subroutine assign(left,right)
real(8),intent(out)::left
logical,intent(in)::right
deallocate(global)
left=1
end subroutine
end module""")
    proof = prove_storage_mutation_effects(SourceEffects([path]), "storage::advance")
    assert not proof.descriptor_unchanged("argument::a")
    assert any("defined assignment" in event["kind"] for event in proof.public()["frontiers"])


@pytest.mark.parametrize("fault", ["copy", "registry", "source", "namespace", "body", "limits"])
def test_retained_projection_revalidates_source_resolution_registry_and_policy(tmp_path, fault):
    path, source, proof = token(tmp_path)
    if fault == "copy":
        proof = copy(proof)
    elif fault == "registry":
        source._storage_mutation_effects[proof.identity] = replace(proof)
    elif fault == "source":
        path.write_text(path.read_text()+"!changed\n")
    elif fault == "namespace":
        source.routines["storage::advance"].scope.imports["foreign"] = ("missing", "foreign")
    elif fault == "body":
        node = next(node for node in walk(source.routines["storage::advance"].execution) if type(node).__name__ == "Name" and str(node)=="a")
        node.string = "missing"
    else:
        source.operation_limit += 1
    with pytest.raises(CompilationError):
        proof.validate()
