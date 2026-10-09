"""Native effects stay conservative independently of numerical GPU lowering."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from compiler.frontend import analyze_source_effects, lower_file
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def records(report):
    return {item["procedure"]: item for item in report["procedures"]}


def test_array_utilities_imported_generics_and_hidden_resources(tmp_path):
    kinds = write(tmp_path, "precision.f90", "module precision\ninteger,parameter::wp=kind(1.d0)\nend module\n")
    utilities = write(tmp_path, "utilities.f90", """module utilities
use precision,only:wp
implicit none
interface fill
module procedure fill_real,fill_integer
end interface
contains
subroutine fill_real(a,b)
real(wp),intent(out)::a(:,:,:)
real(wp),intent(in)::b
integer::k
do k=1,size(a,3)
a(:,:,k)=b
enddo
end subroutine
subroutine fill_integer(a,b)
real(wp),intent(out)::a(:,:,:)
integer,intent(in)::b
integer::k
do k=1,size(a,3)
a(:,:,k)=b
enddo
end subroutine
subroutine scale(a,b)
real(wp),intent(inout)::a(:,:,:)
real(wp),intent(in)::b
integer::k
do k=1,size(a,3)
a(:,:,k)=a(:,:,k)*b
enddo
end subroutine
end module
""")
    application = write(tmp_path, "application.f90", """module application
use precision,only:wp
use utilities,only:initialize=>fill,scale
implicit none
real(wp)::coefficient
real(wp)::hidden(3,3,3)
contains
subroutine advance(output)
real(wp),intent(out)::output(-2:,-2:,-2:)
call initialize(output,0)
call scale(output,coefficient)
hidden=output(1:3,1:3,1:3)
end subroutine
end module
""")
    report = analyze_source_effects([application, utilities, kinds], "application::advance")
    assert report["complete"], records(report)
    summaries = records(report)
    root = summaries["application::advance"]
    calls = [op for op in root["operations"] if op["kind"] == "call"]
    assert [op["procedure"] for op in calls] == ["utilities::fill_integer", "utilities::scale"]
    assert calls[0]["resource_mapping"] == {"argument::a": "argument::output"}
    assert calls[1]["resource_mapping"]["argument::b"] == "application::coefficient"
    assert any(op.get("resource") == "application::hidden" and op["kind"] == "overwrite" for op in root["operations"])
    assert summaries["utilities::fill_integer"]["definition_changes"] == ["argument::a"]
    descriptor = [op for op in summaries["utilities::fill_integer"]["operations"] if op["kind"] == "descriptor_read"]
    assert descriptor
    assert not any(op["kind"] == "read" and op["resource"] == "argument::a" for op in summaries["utilities::fill_integer"]["operations"])
    with pytest.raises(CompilationError):
        lower_file(utilities, "fill_integer")


def test_cpu_read_mirror_and_conditional_write_effects_are_distinct(tmp_path):
    path = write(tmp_path, "inspect.f90", """module inspectors
contains
subroutine inspect(a,total,change)
real(8),intent(inout)::a(:)
real(8),intent(out)::total
logical,intent(in)::change
total=sum(a)
if(change) a(1)=total
end subroutine
end module
""")
    report = analyze_source_effects([path], "inspect")
    assert report["complete"]
    operations = records(report)["inspectors::inspect"]["operations"]
    assert any(op["kind"] == "read" and op["resource"] == "argument::a" and not op["guard"] for op in operations)
    write_op = next(op for op in operations if op["kind"] == "write")
    assert write_op["resource"] == "argument::a"
    assert write_op["guard"] == ("change",)
    assert not any(op["kind"] == "overwrite" and op["resource"] == "argument::a" for op in operations)


def test_scalar_element_actual_has_no_whole_storage_effect_proof(tmp_path):
    path = write(tmp_path, "element.f90", """module elements
real(8)::field(8)
contains
subroutine replace(value)
real(8),intent(inout)::value
value=value+1.d0
end subroutine
subroutine wrapper()
call replace(field(1))
end subroutine
subroutine step()
call wrapper()
end subroutine
end module
""")
    report = analyze_source_effects([path], "elements::step")
    assert not report["complete"]
    wrapper = records(report)["elements::wrapper"]
    assert not wrapper["complete"]
    assert not wrapper["cloneable"]
    boundary, = wrapper["operations"]
    assert boundary["kind"] == "boundary"
    assert boundary["reason"].endswith("field(1)")
    assert "in-place mapping and coherence" in boundary["reason"]


@pytest.mark.parametrize("statement", ["call unknown(a)", "allocate(a(3))", "a=>b", "print *, a"])
def test_unknown_effects_and_lifetime_close_the_scope(tmp_path, statement):
    path = write(tmp_path, "unknown.f90", f"""module unsupported
contains
subroutine advance(a,b)
real(8),pointer::a(:),b(:)
{statement}
end subroutine
end module
""")
    report = analyze_source_effects([path], "advance")
    assert not report["complete"]
    root = records(report)["unsupported::advance"]
    assert root["reasons"]
    assert not root["cloneable"]
    assert any(op["kind"] == "boundary" for op in root["operations"])


def test_saved_state_is_original_native_state_not_a_clone(tmp_path):
    path = write(tmp_path, "saved.f90", """module counters
contains
subroutine advance(a)
real(8),intent(inout)::a(:)
integer,save::calls=0
calls=calls+1
a=a+calls
end subroutine
end module
""")
    report = analyze_source_effects([path], "advance")
    root = records(report)["counters::advance"]
    assert root["complete"]
    assert not root["cloneable"]
    assert root["persistent_state"] == ["counters::advance::calls"]
    assert any(op.get("resource") == "counters::advance::calls" for op in root["operations"])


def test_recursion_and_closure_budgets_never_drop_effects(tmp_path):
    path = write(tmp_path, "chain.f90", """module chain
contains
subroutine advance(a)
real(8),intent(inout)::a(:)
call leaf(a)
end subroutine
subroutine leaf(a)
real(8),intent(inout)::a(:)
a=a+1
end subroutine
subroutine recursive_step(a)
real(8),intent(inout)::a(:)
call recursive_step(a)
end subroutine
end module
""")
    assert analyze_source_effects([path], "advance")["complete"]
    for limits in ({"depth": 1}, {"procedures": 1}, {"operations": 1}):
        assert not analyze_source_effects([path], "advance", **limits)["complete"]
    assert not analyze_source_effects([path], "recursive_step")["complete"]


def test_effect_budgets_are_per_proof_and_do_not_poison_later_candidates(tmp_path):
    path = write(tmp_path, "independent.f90", """module independent
contains
subroutine first(a)
real(8),intent(inout)::a(:)
a=a+1
end subroutine
subroutine second(a)
real(8),intent(inout)::a(:)
a=a+2
end subroutine
subroutine excessive(a)
real(8),intent(inout)::a(:)
a=a+1
a=a+2
a=a+3
end subroutine
subroutine joined(a)
real(8),intent(inout)::a(:)
call first(a)
call second(a)
end subroutine
end module
""")
    analysis = SourceEffects([path], operations=3)
    assert not analysis.summarize("independent::excessive")["complete"]
    for name in ["first", "second"]:
        report = analysis.report("independent::" + name)
        assert report["complete"]
        assert report["summarized_operations"] == 2
        assert set(records(report)) == {"independent::" + name}
        assert {op["kind"] for op in records(report)["independent::" + name]["operations"]} == {"read", "overwrite"}
    assert not analysis.summarize("independent::joined")["complete"]
    assert analysis.summarize("independent::first")["complete"]
    with pytest.raises(CompilationError, match="budget exhausted"):
        analysis.summarize_span(["independent::first", "independent::second"])
    assert analysis.summarize_span(["independent::first", "independent::first"])["operations"] == 2


def test_cached_child_cannot_bypass_deeper_call_or_procedure_budgets(tmp_path):
    path = write(tmp_path, "depths.f90", """module depths
contains
subroutine root(a)
real(8),intent(inout)::a(:)
call middle(a)
call longer(a)
end subroutine
subroutine longer(a)
real(8),intent(inout)::a(:)
call middle(a)
end subroutine
subroutine middle(a)
real(8),intent(inout)::a(:)
call leaf(a)
end subroutine
subroutine leaf(a)
real(8),intent(inout)::a(:)
a=a+1
end subroutine
end module
""")
    analysis = SourceEffects([path], depth=3)
    assert analysis.report("middle")["complete"]
    assert not analysis.report("root")["complete"]
    assert analysis.report("longer")["complete"]
    analysis = SourceEffects([path], procedures=2)
    assert analysis.report("middle")["complete"]
    assert not analysis.report("root")["complete"]
    assert analysis.report("leaf")["complete"]


def test_opaque_contract_has_explicit_effects_and_identity(tmp_path):
    path = write(tmp_path, "contract.f90", """module caller
use opaque_library,only:solve
contains
subroutine advance(rhs,phi,velocity)
real(8),intent(in)::rhs(:),velocity(:)
real(8),intent(inout)::phi(:)
call solve(phi,rhs)
end subroutine
end module
""")
    assert not analyze_source_effects([path], "advance")["complete"]
    contract = {"identity": "opaque-library-ABI1", "lifetime": "stable", "escapes": False, "ordering": "serial",
                "complete": True, "descriptor_changes": False,
                "effects": [{"kind": "read", "argument": 1, "section": "whole"},
                            {"kind": "write", "argument": 0, "section": "whole"}]}
    report = analyze_source_effects([path], "advance", contracts={"opaque_library::solve": contract})
    assert report["complete"]
    operation = records(report)["caller::advance"]["operations"][0]
    assert operation["kind"] == "native_contract"
    assert {op["resource"] for op in operation["effects"]} == {"argument::rhs", "argument::phi"}
    assert len(operation["contract_sha256"]) == 64
    with pytest.raises(CompilationError, match="invalid native effect contract"):
        analyze_source_effects([path], "advance", contracts={"opaque_library::solve": {**contract, "escapes": True}})


def test_import_shadows_same_module_procedure(tmp_path):
    path = write(tmp_path, "shadow.f90", """module imported
contains
subroutine work(a)
real(8),intent(inout)::a(:)
a=2
end subroutine
end module
module local
contains
subroutine work(a)
real(8),intent(inout)::a(:)
a=1
end subroutine
subroutine advance(a)
use imported,only:work
real(8),intent(inout)::a(:)
call work(a)
end subroutine
end module
""")
    report = analyze_source_effects([path], "local::advance")
    assert report["complete"]
    assert records(report)["local::advance"]["operations"][0]["procedure"] == "imported::work"


def test_effect_cli_uses_an_independent_compiler_checkout(tmp_path):
    compiler_root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "compiler-checkout"
    shutil.copytree(compiler_root, checkout / "compiler", ignore=shutil.ignore_patterns("__pycache__", ".*cache"))
    source = write(tmp_path, "original.f90", """module original
contains
subroutine utility(a)
real(8),intent(out)::a(:,:)
a=3
end subroutine
end module
""")
    output = tmp_path / "untouched"
    cache = tmp_path / "summary-cache"
    result = subprocess.run([sys.executable, "-m", "compiler", "--input", str(source), "--kernel", "utility",
                             "--analyze-effects", "--json", "--output-dir", str(output),
                             "--summary-cache", str(cache)], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(checkout)}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["supported"]
    assert report["effects"]["complete"]
    repeated = subprocess.run([sys.executable, "-m", "compiler", "--input", str(source), "--kernel", "utility",
                               "--analyze-effects", "--json", "--summary-cache", str(cache)], cwd=tmp_path,
                              env={**os.environ, "PYTHONPATH": str(checkout)}, capture_output=True, text=True, timeout=60)
    assert repeated.returncode == 0, repeated.stderr
    assert json.loads(repeated.stdout)["effects"]["summary_cache"]["disk_hits"] == 1
    assert not report["outputs"]
    assert not output.exists()



def test_loop_control_variables_have_native_write_effects_even_for_empty_loops(tmp_path):
    path = write(tmp_path, "counters.f90", """module counters
integer::module_counter
contains
subroutine advance(counter)
integer,intent(inout)::counter
do counter=4,1
continue
enddo
do module_counter=1,3
continue
enddo
end subroutine
end module
""")
    report = analyze_source_effects([path], "advance")
    assert report["complete"]
    operations = records(report)["counters::advance"]["operations"]
    writes = [op for op in operations if op["kind"] == "write"]
    assert {op["resource"] for op in writes} == {"argument::counter", "counters::module_counter"}
    assert all(not op["guard"] for op in writes)


def test_unknown_wildcard_exports_cannot_prove_intrinsic_effects(tmp_path):
    path = write(tmp_path, "intrinsics.f90", """module inspectors
use unavailable_exports
contains
subroutine inspect(a,total)
real(8),intent(in)::a(:)
real(8),intent(out)::total
total=sum(a)
end subroutine
end module
""")
    report = analyze_source_effects([path], "inspect")
    assert not report["complete"]
    assert any("unresolved function effects" in reason
               for reason in records(report)["inspectors::inspect"]["reasons"])



@pytest.mark.parametrize(("body","expected"), [
    ("a=3", True),
    ("a(:,:,:)=3", True),
    ("do k=1,size(a,3)\na(:,:,k)=3\nenddo", True),
    ("do k=1,size(a,3)-1\na(:,:,k)=3\nenddo", False),
    ("do k=1,size(a,3),2\na(:,:,k)=3\nenddo", False),
    ("do k=1,size(a,3)\nif(k>2) a(:,:,k)=3\nenddo", False),
    ("a(:,:,1)=3", False),
])
def test_must_write_coverage_does_not_confuse_faces_holes_or_strides(tmp_path,body,expected):
    path=write(tmp_path,"fill.f90",f"""module utilities
contains
subroutine fill(a)
real(8),intent(out)::a(:,:,:)
integer::k
{body}
end subroutine
end module
""")
    summary=records(analyze_source_effects([path],"fill"))["utilities::fill"]
    assert ("argument::a" in summary["guaranteed_whole_overwrites"]) == expected


def test_negative_bounds_need_full_logical_sweep_not_size_starting_at_one(tmp_path):
    for index,bounds in enumerate(["lbound(a,3),ubound(a,3)","1,size(a,3)"]):
        path=write(tmp_path,f"fill{index}.f90",f"""module utilities
contains
subroutine fill(a)
real(8),intent(out)::a(-2:,-2:,-2:)
integer::k
do k={bounds}
a(:,:,k)=3
enddo
end subroutine
end module
""")
        summary=records(analyze_source_effects([path],"fill"))["utilities::fill"]
        assert ("argument::a" in summary["guaranteed_whole_overwrites"]) == (index==0)


def test_unsupported_module_storage_is_not_hidden_by_a_clean_routine(tmp_path):
    path=write(tmp_path,"shared.f90","""module shared
real(8)::a(8)
common /storage/ a
contains
subroutine advance()
a=1
end subroutine
end module
""")
    report=analyze_source_effects([path],"advance")
    assert not report["complete"]
    assert any("specification effect unavailable" in reason
               for reason in records(report)["shared::advance"]["reasons"])


def test_private_implementation_is_resolved_only_through_public_generic(tmp_path):
    utilities=write(tmp_path,"utilities.f90","""module utilities
private
public :: fill
interface fill
module procedure fill_real
end interface
contains
subroutine fill_real(a)
real(8),intent(out)::a(:)
a=3
end subroutine
end module
""")
    application=write(tmp_path,"application.f90","""module application
use utilities
contains
subroutine advance(a)
real(8),intent(out)::a(:)
call fill(a)
end subroutine
subroutine illegal_access(a)
real(8),intent(out)::a(:)
call fill_real(a)
end subroutine
end module
""")
    report=analyze_source_effects([application,utilities],"advance")
    assert report["complete"]
    assert "utilities::fill_real" in records(report)
    assert not analyze_source_effects([application,utilities],"illegal_access")["complete"]


def test_private_module_storage_cannot_be_used_through_wildcard_import(tmp_path):
    storage=write(tmp_path,"storage.f90","""module storage
real(8),private :: hidden(8)
real(8),public :: visible(8)
end module
""")
    application=write(tmp_path,"application.f90","""module application
use storage
contains
subroutine advance()
visible=3
end subroutine
subroutine illegal_access()
hidden=3
end subroutine
end module
""")
    assert analyze_source_effects([application,storage],"advance")["complete"]
    assert not analyze_source_effects([application,storage],"illegal_access")["complete"]


def test_specification_payload_reads_require_native_coherence(tmp_path):
    path = write(tmp_path, "spec.f90", """module specifications
contains
subroutine consume(bounds,value)
integer,intent(in)::bounds(:)
real(8),intent(out)::value
real(8)::scratch(bounds(1))
value=1.d0
end subroutine
subroutine descriptor(a,value)
real(8),intent(in)::a(:)
real(8),intent(out)::value
real(8)::scratch(size(a))
value=2.d0
end subroutine
end module
""")
    analysis = SourceEffects([path])
    payload = analysis.summarize("specifications::consume")
    assert payload["complete"]
    assert any(op["kind"] == "read" and op["resource"] == "argument::bounds" for op in payload["operations"])
    descriptor = analysis.summarize("specifications::descriptor")
    assert descriptor["complete"]
    assert any(op["kind"] == "descriptor_read" and op["resource"] == "argument::a"
               for op in descriptor["operations"])
    assert not any(op["kind"] == "read" and op["resource"] == "argument::a" for op in descriptor["operations"])


def test_unknown_specification_function_is_a_source_boundary(tmp_path):
    path = write(tmp_path, "spec.f90", """module specifications
contains
subroutine consume(a,n,value)
real(8),intent(in)::a(:)
integer,intent(in)::n
real(8),intent(out)::value
real(8)::scratch(user_bound(n))
value=1.d0
end subroutine
end module
""")
    result = SourceEffects([path]).summarize("specifications::consume")
    assert not result["complete"]
    assert any("unknown function" in reason and "user_bound" in reason.lower() for reason in result["reasons"])


def test_opaque_contract_preserves_scalar_effect_rank(tmp_path):
    path = write(tmp_path, "scalar_contract.f90", """module caller
use opaque_library,only:adjust
contains
subroutine advance(n)
integer,intent(inout)::n
call adjust(n)
end subroutine
end module
""")
    contract = {"identity": "opaque-ABI1", "lifetime": "stable", "escapes": False, "ordering": "serial",
                "complete": True, "descriptor_changes": False,
                "effects": [{"kind": "write", "argument": 0, "section": "whole"}]}
    result = SourceEffects([path], contracts={"opaque_library::adjust": contract}).summarize("caller::advance")
    assert result["complete"]
    operation, = result["operations"]
    effect, = operation["effects"]
    assert effect["resource"] == "argument::n"
    assert effect["rank"] == 0


def allocation_effects(tmp_path, body="a(1)=scratch(1)+size(scratch)", *, declaration="real(8),allocatable::scratch(:)",
                       specification="", helpers="", operations=256):
    path = write(tmp_path, "allocation_effects.f90", f"""module allocations
{declaration}
contains
subroutine inspect(a)
real(8),intent(inout)::a(:)
{specification}
{body}
end subroutine
{helpers}
end module
""")
    return path, SourceEffects([path], operations=operations)


def test_module_allocatable_effects_require_explicit_lifetime_authority_by_default(tmp_path):
    _path, analysis = allocation_effects(tmp_path)
    report = analysis.report("allocations::inspect")
    assert not report["complete"]
    summary = records(report)["allocations::inspect"]
    assert summary["capture_lifetime_requirements"] == [{"resource": "allocations::scratch", "authorized": False}]
    assert report["capture_lifetime_authorizations"] == []
    assert analysis.stable_module_allocatables == frozenset()
    assert any("storage lifetime requires capture proof" in reason for reason in summary["reasons"])


@pytest.mark.parametrize("body", ["a(1)=scratch(1)+size(scratch)",
                                 "scratch(1)=a(1)", "scratch(:)=a", "a=scratch", "if(allocated(scratch)) a(1)=1"])
def test_source_bound_authorized_module_array_reads_descriptors_and_section_writes(tmp_path, body):
    path, analysis = allocation_effects(tmp_path, body)
    analysis.authorize_stable_module_allocatables(frozenset({"allocations::scratch"}))
    report = analysis.report("allocations::inspect")
    assert report["complete"], report
    summary = records(report)["allocations::inspect"]
    assert summary["capture_lifetime_requirements"] == [{"resource": "allocations::scratch", "authorized": True}]
    assert report["capture_lifetime_authorizations"] == [
        {"resource": "allocations::scratch", "source": str(path), "source_sha256": report["sources"][str(path)]}]
    assert analysis.stable_module_allocatables == frozenset({"allocations::scratch"})
    # Lifetime authority does not invent a fixed original lower bound.
    assert not summary["native_sections"]["available"]
    assert "lifetime requires capture proof" not in " ".join(summary["reasons"])


def test_authorized_target_module_array_keeps_capture_aliasing_as_a_separate_requirement(tmp_path):
    _path, analysis = allocation_effects(tmp_path, declaration="real(8),allocatable,target::scratch(:)")
    analysis.authorize_stable_module_allocatables({"allocations::scratch"})
    assert analysis.summarize("allocations::inspect")["complete"]


def test_authorization_of_one_root_does_not_waive_other_hidden_or_formal_storage(tmp_path):
    _path, analysis = allocation_effects(tmp_path, "a(1)=scratch(1)+other(1)",
                                        declaration="real(8),allocatable::scratch(:),other(:)")
    analysis.authorize_stable_module_allocatables({"allocations::scratch"})
    summary = analysis.summarize("allocations::inspect")
    assert not summary["complete"]
    assert summary["capture_lifetime_requirements"] == [
        {"resource": "allocations::other", "authorized": False}, {"resource": "allocations::scratch", "authorized": True}]
    assert summary["reasons"] == ["storage lifetime requires capture proof: allocations::other"]
    path, _analysis = allocation_effects(tmp_path)
    path.write_text(path.read_text().replace("real(8),intent(inout)::a(:)", "real(8),allocatable,intent(inout)::a(:)"))
    analysis = SourceEffects([path])
    analysis.authorize_stable_module_allocatables({"allocations::scratch"})
    summary = analysis.summarize("allocations::inspect")
    assert not summary["complete"]
    assert "storage lifetime requires capture proof: argument::a" in summary["reasons"]


def test_imported_alias_authorization_uses_defining_canonical_module_identity(tmp_path):
    storage = write(tmp_path, "storage.f90", "module storage\nreal(8),allocatable::buffer(:)\nend module\n")
    caller = write(tmp_path, "borrower.f90", """module borrower
use storage,only:remote=>buffer
contains
subroutine inspect(a)
real(8),intent(inout)::a(:)
a(1)=remote(1)
end subroutine
end module
""")
    analysis = SourceEffects([storage, caller])
    with pytest.raises(CompilationError, match="canonical module array"):
        analysis.authorize_stable_module_allocatables({"borrower::remote"})
    analysis.authorize_stable_module_allocatables({"storage::buffer"})
    summary = analysis.summarize("borrower::inspect")
    assert summary["complete"]
    assert summary["capture_lifetime_requirements"] == [{"resource": "storage::buffer", "authorized": True}]


@pytest.mark.parametrize("body", ["scratch=a", "scratch=0.d0", "scratch=scratch+1.d0"])
def test_authorized_whole_allocatable_assignment_still_rejects_implicit_allocation(tmp_path, body):
    _path, analysis = allocation_effects(tmp_path, body)
    analysis.authorize_stable_module_allocatables({"allocations::scratch"})
    summary = analysis.summarize("allocations::inspect")
    assert not summary["complete"]
    assert summary["guaranteed_whole_overwrites"] == []
    assert "whole allocatable assignment may change storage: allocations::scratch" in summary["reasons"]


@pytest.mark.parametrize("body", ["allocate(scratch(8))", "deallocate(scratch)",
                                 "call move_alloc(scratch,other)", "scratch=>other"])
def test_authority_does_not_hide_explicit_storage_or_association_changes(tmp_path, body):
    _path, analysis = allocation_effects(tmp_path, body, declaration="real(8),allocatable::scratch(:),other(:)")
    analysis.authorize_stable_module_allocatables({"allocations::scratch", "allocations::other"})
    summary = analysis.summarize("allocations::inspect")
    assert not summary["complete"]
    assert summary["reasons"]


@pytest.mark.parametrize("intent", ["in", "inout", "out"])
def test_unused_allocatable_callee_formals_keep_descriptor_semantics_visible(tmp_path, intent):
    helper = f"""subroutine unused(value)
real(8),allocatable,intent({intent})::value(:)
end subroutine
"""
    _path, analysis = allocation_effects(tmp_path, "call unused(scratch)", helpers=helper)
    analysis.authorize_stable_module_allocatables({"allocations::scratch"})
    summary = analysis.summarize("allocations::inspect")
    if intent == "in":
        assert summary["complete"]
        call = next(operation for operation in summary["operations"] if operation["kind"] == "call")
        assert call["resource_mappings"][0]["requirements"]["original_allocation_descriptor"]
        assert analysis.summarize("allocations::unused")["descriptor_requirements"]
    else:
        assert not summary["complete"]
        assert any("allocatable callee formals" in reason for reason in summary["reasons"])


@pytest.mark.parametrize("declaration", ["real(8),pointer::scratch(:)", "real(8),allocatable::scratch",
                                       "real(8)::scratch(8)", "real(8),allocatable,volatile::scratch(:)",
                                       "real(8),allocatable,asynchronous::scratch(:)", "integer(8),allocatable::scratch(:)"])
def test_lifetime_authority_rejects_unsupported_module_storage(tmp_path, declaration):
    _path, analysis = allocation_effects(tmp_path, declaration=declaration)
    with pytest.raises(CompilationError, match="canonical module array"):
        analysis.authorize_stable_module_allocatables({"allocations::scratch"})
    assert analysis.stable_module_allocatables == frozenset()


@pytest.mark.parametrize("root", ["argument::a", "allocations::inspect::local", "allocations::missing", "ALLOCATIONS::scratch"])
def test_authorization_cannot_cover_formal_local_missing_or_noncanonical_roots(tmp_path, root):
    _path, analysis = allocation_effects(tmp_path, specification="real(8),allocatable::local(:)")
    with pytest.raises(CompilationError, match="canonical module array"):
        analysis.authorize_stable_module_allocatables({root})


def test_module_named_argument_cannot_waive_a_formal_with_the_same_resource_spelling(tmp_path):
    path = write(tmp_path, "argument.f90", """module argument
real(8),allocatable::scratch(:)
contains
subroutine inspect(scratch)
real(8),allocatable::scratch(:)
scratch(1)=1
end subroutine
end module
""")
    analysis = SourceEffects([path])
    with pytest.raises(CompilationError, match="canonical module array"):
        analysis.authorize_stable_module_allocatables({"argument::scratch"})


@pytest.mark.parametrize("query", ["summary", "sections", "report"])
def test_lifetime_authorization_must_precede_every_proof_or_cached_query(tmp_path, query):
    _path, analysis = allocation_effects(tmp_path)
    if query == "summary":
        analysis.summarize("allocations::inspect")
    elif query == "sections":
        analysis.native_sections("allocations::inspect")
    else:
        analysis.report("allocations::inspect")
    with pytest.raises(CompilationError, match="precede source analysis"):
        analysis.authorize_stable_module_allocatables({"allocations::scratch"})
    assert analysis.stable_module_allocatables == frozenset()


def test_lifetime_authorization_is_bounded_and_commits_all_roots_atomically(tmp_path):
    _path, analysis = allocation_effects(tmp_path, declaration="real(8),allocatable::scratch(:),other(:)", operations=1)
    with pytest.raises(CompilationError, match="bounded root set"):
        analysis.authorize_stable_module_allocatables({"allocations::scratch", "allocations::other"})
    with pytest.raises(CompilationError, match="bounded root set"):
        analysis.authorize_stable_module_allocatables(["allocations::scratch"])
    with pytest.raises(CompilationError, match="canonical source roots"):
        analysis.authorize_stable_module_allocatables({False})
    assert analysis.stable_module_allocatables == frozenset()
    analysis.authorize_stable_module_allocatables({"allocations::scratch"})
    with pytest.raises(CompilationError, match="canonical module array"):
        analysis.authorize_stable_module_allocatables({"allocations::missing"})
    assert analysis.stable_module_allocatables == frozenset({"allocations::scratch"})


def test_source_mutation_prevents_lifetime_authority_before_any_proof(tmp_path):
    path, analysis = allocation_effects(tmp_path)
    path.write_text(path.read_text() + "! changed\n")
    with pytest.raises(CompilationError, match="source changed"):
        analysis.authorize_stable_module_allocatables({"allocations::scratch"})
    assert analysis.stable_module_allocatables == frozenset()


def test_lifetime_authority_does_not_waive_opaque_call_or_openmp_ownership(tmp_path):
    _path, analysis = allocation_effects(tmp_path, "call external(scratch)")
    analysis.authorize_stable_module_allocatables({"allocations::scratch"})
    assert not analysis.summarize("allocations::inspect")["complete"]
    path, _analysis = allocation_effects(tmp_path)
    path.write_text(path.read_text().replace("contains", "!$omp threadprivate(scratch)\ncontains", 1))
    analysis = SourceEffects([path])
    analysis.authorize_stable_module_allocatables({"allocations::scratch"})
    summary = analysis.summarize("allocations::inspect")
    assert summary["complete"]
    assert not summary["native_completion"]["available"]


@pytest.mark.parametrize(("lower", "upper", "expected"), [
    ("1", "ubound(scratch,1)", []),
    ("1", "size(scratch,1)", []),
    ("lbound(scratch,1)", "ubound(scratch,1)", ["allocations::scratch"]),
])
def test_allocatable_whole_sweep_requires_actual_lower_and_upper_origins(tmp_path, lower, upper, expected):
    body = f"do i={lower},{upper}\nscratch(i)=1.d0\nenddo"
    _path, analysis = allocation_effects(tmp_path, body, specification="integer::i")
    analysis.authorize_stable_module_allocatables({"allocations::scratch"})
    summary = analysis.summarize("allocations::inspect")
    assert summary["complete"]
    # A valid allocation field(0:n) preserves cell0 in the first loop. Using
    # the declaration's deferred lower bound would suppress a required mirror.
    assert summary["guaranteed_whole_overwrites"] == expected


def test_allocatable_explicit_full_section_is_whole_independently_of_runtime_origin(tmp_path):
    _path, analysis = allocation_effects(tmp_path, "scratch(:)=1.d0")
    analysis.authorize_stable_module_allocatables({"allocations::scratch"})
    summary = analysis.summarize("allocations::inspect")
    assert summary["complete"]
    assert summary["guaranteed_whole_overwrites"] == ["allocations::scratch"]
