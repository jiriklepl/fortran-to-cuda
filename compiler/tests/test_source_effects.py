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
    result = subprocess.run([sys.executable, "-m", "compiler", "--input", str(source), "--kernel", "utility",
                             "--analyze-effects", "--json", "--output-dir", str(output)], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(checkout)}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["supported"]
    assert report["effects"]["complete"]
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
