"""Public source scopes retain original native procedures and source guards."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FACT = {"storage":"stable","initialized":"whole","escapes":False,"allocation_changes":False}

PROGRAM = """module original
implicit none
contains
subroutine producer(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=2*a(i)+real(i,8)
enddo
end subroutine
subroutine transform(b)
real(8),intent(inout)::b(:)
b=3*b
end subroutine
subroutine consumer(a,b,out,n)
real(8),intent(in)::a(:),b(:)
real(8),intent(out)::out(:)
integer,intent(in)::n
integer::i
do i=1,n
out(i)=a(i)+b(i)
enddo
end subroutine
subroutine step(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:),out(:)
integer,intent(in)::n
call producer(a,b,n)
call transform(b)
call consumer(a,b,out,n)
end subroutine
end module
"""

DRIVER = """program caller
use original,only:step
implicit none
real(8),allocatable::a(:),b(:),output(:)
integer::n,shape,repeat,i
do shape=1,2
 n=8+8*shape
 allocate(a(-2:n-3),b(-2:n-3),output(-2:n-3))
 do repeat=1,2
  do i=-2,n-3
   a(i)=real(i+3,8)*0.25d0+repeat
  enddo
  b=-99
  output=-99
  call step(a,b,output,n)
  do i=-2,n-3
   if (b(i)/=6*a(i)+3*real(i+3,8)) error stop 'native intermediate disagreement'
   if (output(i)/=7*a(i)+3*real(i+3,8)) error stop 'complete field disagreement'
  enddo
 enddo
 deallocate(a,b,output)
enddo
print *, 'FIELDS_OK'
end program
"""


def run(command, *, cwd, env=None, timeout=120):
    result = subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
    assert result.returncode == 0, result.stdout+result.stderr
    return result


def generate(directory, source=PROGRAM, *, mode="sections", entry="step", facts=None, checkout=None, profile=None):
    directory.mkdir(parents=True,exist_ok=True)
    original = directory/"original.f90"
    original.write_text(source)
    captures = {"argument::a":FACT,
                "argument::b":{**FACT,"initialized":"none"},
                "argument::out":{**FACT,"initialized":"none"}}
    facts_file = directory/"captures.json"
    document={"schema_version":1,"participation":"serial","captures":captures} if facts is None else facts
    document={"sources":{str(original):sha256(original.read_bytes()).hexdigest()},**document}
    facts_file.write_text(json.dumps(document))
    output = directory/"output"
    response = run([sys.executable,"-m","compiler","--form-scopes","--scope-facts",str(facts_file),
                    "--input",str(original),"--kernel",entry,"--memory-model","scoped","--gpu-policy",mode,
                    *( ["--calibration-profile", str(profile)] if profile else []),
                    "--json","--output-dir",str(output)],cwd=checkout or ROOT,
                   env={**os.environ,"PYTHONPATH":str(checkout or ROOT)})
    report = json.loads(response.stdout)
    assert report["supported"]
    manifest = json.loads((output/"scope-manifest.json").read_text())
    assert manifest == report["scopes"]
    return original,output,manifest


def test_source_scope_has_public_artifacts_and_original_native_operations(tmp_path):
    original,output,manifest = generate(tmp_path)
    assert manifest["scope_count"] == 1
    scope, = manifest["scopes"]
    assert scope["calls"] == ["original::producer","original::transform","original::consumer"]
    assert scope["gpu_leaves"] == ["original::consumer","original::producer"]
    assert not scope["estimate_available"]
    replacement = output/manifest["sources"][str(original)]["replacement"]
    text = replacement.read_text()
    assert "subroutine transform(b)" in text
    assert "b=3*b" in text
    assert "fort_scope_host_begin" in text
    assert "fort_scope_host_end" in text
    assert manifest["runtime"]["link_once"]
    assert original.read_text() == PROGRAM
    assert {r["resource"] for r in scope["resources"]} == {"argument::a","argument::b","argument::out"}
    assert all((output/s["path"]).is_file() for s in manifest["build_sources"])


def test_scopes_preserve_outer_guard_and_native_unknown_boundary(tmp_path):
    source = PROGRAM.replace("call producer(a,b,n)\ncall transform(b)\ncall consumer(a,b,out,n)",
                             "call unavailable(a)\nif(n>0) then\ncall producer(a,b,n)\ncall transform(b)\ncall consumer(a,b,out,n)\nendif\ncall unavailable(out)")
    original,output,manifest = generate(tmp_path,source)
    assert manifest["scope_count"] == 1
    assert len(manifest["boundaries"]) == 2
    replacement = (output/manifest["sources"][str(original)]["replacement"]).read_text()
    assert "call unavailable(a)\nif(n>0) then\ncall fort_scope_owner_" in replacement
    assert "endif\ncall unavailable(out)" in replacement


def test_exhausted_earlier_branch_cannot_hide_later_scope_resources(tmp_path):
    large = ("subroutine excessive(a)\nreal(8),intent(inout)::a(:)\n"
             + "a=a+1\n"*140 + "end subroutine\n")
    source = PROGRAM.replace("end module", large + "end module").replace(
        "call producer(a,b,n)", "if(n<0) call excessive(b)\ncall producer(a,b,n)")
    _, _, manifest = generate(tmp_path, source)
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    scope, = manifest["scopes"]
    assert scope["calls"] == ["original::producer", "original::transform", "original::consumer"]
    assert {r["resource"] for r in scope["resources"]} == {"argument::a", "argument::b", "argument::out"}
    assert scope["effect_closure"]["operations"] <= 256
    assert not manifest["native_effects"]["complete"]


def test_missing_capture_proof_keeps_original_source(tmp_path):
    facts = {"schema_version":1,"participation":"serial","captures":{}}
    original,output,manifest = generate(tmp_path,facts=facts)
    assert manifest["scope_count"] == 0
    assert not manifest["automatic_scope_available"]
    assert "missing stable storage" in manifest["boundaries"][0]["reason"]
    assert not manifest["source_edits"]
    assert not (output/"entries").exists()
    assert original.read_text() == PROGRAM


@pytest.mark.parametrize("mode", ["sections", "auto"])
def test_allocatable_owning_roots_are_guarded_at_original_caller(tmp_path, mode):
    original, output, manifest = generate(tmp_path, ALLOCATABLE_PROGRAM, mode=mode)
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    scope, = manifest["scopes"]
    assert scope["allocation_preflight"]["resources"] == ["argument::a", "argument::b", "argument::out"]
    edit = next(edit for edit in manifest["source_edits"] if edit["first_line"] <= edit["last_line"])
    guarded = edit["replacement"]
    assert guarded.startswith("block\n")
    coordinator_import = next(line for line in guarded.splitlines()
                              if line.startswith("use fort_scoped_memory, only:"))
    assert "=> fort_scope_serial_caller" in coordinator_import
    assert "intrinsic :: allocated\n" in guarded
    assert guarded.index("() == 0) then") < guarded.index("allocated(a)") < guarded.index("call fort_scope_owner_")
    span = "".join(original.read_text().splitlines(keepends=True)[edit["first_line"]-1:edit["last_line"]])
    assert guarded.count(span) == 2
    assert all(inquiry not in guarded for inquiry in ("size(", "lbound(", "is_contiguous(", "c_loc(", "fort_scope_create("))
    text = (output / manifest["sources"][str(original)]["replacement"]).read_text()
    assert "if (allocated(a) .and. allocated(b) .and. allocated(out)) then\nblock\n" in text
    owner = text.split("subroutine fort_scope_owner_", 1)[1].split("end subroutine", 1)[0]
    # Owners preserve the original descriptors for read-only allocatable
    # native children. Payload registration still occurs after caller guards.
    assert owner.lower().count("allocatable, target, intent(") == 3
    assert "allocate(" not in owner.lower()
    assert "deallocate(" not in owner.lower()
    if mode == "sections":
        assert "if (all(fort_extents_0 > 0)) fort_host_pointer = c_loc(" in owner
        assert "fort_scope_close(fort_context)" in owner


def test_allocatable_owner_guard_forces_intrinsic_when_host_procedure_shadows_it(tmp_path):
    source = ALLOCATABLE_PROGRAM.replace("end module", """logical function allocated(value)
real(8),intent(in)::value(:)
allocated=.false.
end function
end module""")
    _, _, manifest = generate(tmp_path, source)
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    edit = next(edit for edit in manifest["source_edits"] if edit["first_line"] <= edit["last_line"])
    assert "block\n" in edit["replacement"]
    assert "intrinsic :: allocated\n" in edit["replacement"]


def test_allocatable_owner_guard_coordinator_cannot_hide_original_callee(tmp_path):
    _, _, first = generate(tmp_path / "first", ALLOCATABLE_PROGRAM)
    edit = next(edit for edit in first["source_edits"] if edit["first_line"] <= edit["last_line"])
    coordinator_import = next(line for line in edit["replacement"].splitlines()
                              if line.startswith("use fort_scoped_memory, only:"))
    alias = coordinator_import.split("only: ", 1)[1].split(" =>", 1)[0]
    # Procedure/call renaming preserves the owning entry and call-span positions.
    source = ALLOCATABLE_PROGRAM.replace("producer(", alias + "(")
    _, _, manifest = generate(tmp_path / "collision", source)
    assert not manifest["automatic_scope_available"]
    assert not manifest["source_edits"]
    assert any("allocation guard coordinator conflicts with an original call: " + alias in item["reason"]
               for item in manifest["boundaries"])


def test_explicit_module_allocatable_roots_use_caller_allocation_guard(tmp_path):
    before, step = ALLOCATABLE_PROGRAM.split("subroutine step(", 1)
    before = before.replace("implicit none\ncontains", "implicit none\nreal(8),allocatable::a(:),b(:),out(:)\ncontains")
    step = step.replace("a,b,out,n)", "n)", 1).replace("real(8),allocatable,intent(in)::a(:)\n", "")
    step = step.replace("real(8),allocatable,intent(inout)::b(:),out(:)\n", "")
    source = before + "subroutine step(" + step
    facts = {"schema_version":1,"participation":"serial","captures":{
        "original::a":FACT,"original::b":{**FACT,"initialized":"none"},
        "original::out":{**FACT,"initialized":"none"}}}
    _, _, manifest = generate(tmp_path, source, facts=facts)
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    assert manifest["scopes"][0]["allocation_preflight"]["resources"] == ["original::a", "original::b", "original::out"]


def test_allocatable_owner_capture_named_allocated_is_an_explained_boundary(tmp_path):
    before, step = ALLOCATABLE_PROGRAM.split("subroutine step(", 1)
    step = step.replace("step(a,b,out,n)", "step(allocated,b,out,n)").replace("::a(:)", "::allocated(:)")
    step = step.replace("allocated(a)", "allocated(allocated)").replace("producer(a,b,n)", "producer(allocated,b,n)")
    step = step.replace("consumer(a,b,out,n)", "consumer(allocated,b,out,n)")
    step = step.replace("if (allocated(allocated) .and. allocated(b) .and. allocated(out)) then", "if(n>0) then")
    source = before + "subroutine step(" + step.replace("a,b,out,n)", "allocated,b,out,n)", 1)
    facts = {"schema_version":1,"participation":"serial","captures":{
        "argument::allocated":FACT,"argument::b":{**FACT,"initialized":"none"},
        "argument::out":{**FACT,"initialized":"none"}}}
    _, _, manifest = generate(tmp_path, source, facts=facts)
    assert not manifest["automatic_scope_available"]
    assert not manifest["source_edits"]
    assert any("allocation guard intrinsic conflicts" in item["reason"] for item in manifest["boundaries"])


def test_allocatable_owning_root_still_requires_stable_lifetime_facts(tmp_path):
    facts = {"schema_version":1,"participation":"serial","captures":{
        "argument::a":{**FACT,"allocation_changes":True},
        "argument::b":{**FACT,"initialized":"none"},"argument::out":{**FACT,"initialized":"none"}}}
    _, _, manifest = generate(tmp_path, ALLOCATABLE_PROGRAM, facts=facts)
    assert not manifest["automatic_scope_available"]
    assert any("missing stable storage/definition facts: argument::a" in item["reason"]
               for item in manifest["boundaries"])


@pytest.mark.parametrize("intent", ["inout", "out"])
def test_allocatable_callee_formals_remain_original_native_boundaries(tmp_path, intent):
    source = ALLOCATABLE_PROGRAM.replace("real(8),intent(out)::b(:)",
                                         "real(8),allocatable,intent(" + intent + ")::b(:)", 1)
    source = source.replace("call producer(a,b,n)\ncall transform(b)\ncall consumer(a,b,out,n)",
                             "call producer(a,b,n)\ncall producer(a,b,n)")
    original, _, manifest = generate(tmp_path, source)
    assert not manifest["automatic_scope_available"]
    assert not manifest["source_edits"]
    assert original.read_text() == source
    assert any("allocatable callee formals require original descriptor and allocation semantics" in item["reason"]
               for item in manifest["boundaries"])


def test_unused_allocatable_out_formal_is_not_an_ordinary_borrowed_view(tmp_path):
    source = ALLOCATABLE_PROGRAM.replace("real(8),intent(out)::b(:)", "real(8),allocatable,intent(out)::b(:)", 1)
    source = source.replace("integer::i\ndo i=1,n\nb(i)=2*a(i)+real(i,8)\nenddo", "", 1)
    source = source.replace("call producer(a,b,n)\ncall transform(b)\ncall consumer(a,b,out,n)",
                             "call producer(a,b,n)\ncall producer(a,b,n)")
    _, _, manifest = generate(tmp_path, source)
    assert not manifest["automatic_scope_available"]
    assert not manifest["source_edits"]
    assert any("allocatable callee formals require original descriptor and allocation semantics" in item["reason"]
               for item in manifest["boundaries"])


@pytest.mark.parametrize("fact", [None, {**FACT, "allocation_changes": True}, FACT])
def test_hidden_allocatable_native_effect_requires_its_own_stable_capture(tmp_path, fact):
    source = PROGRAM.replace("implicit none\ncontains", "implicit none\nreal(8),allocatable::hidden(:)\ncontains")
    source = source.replace("b=3*b", "b=3*b+hidden")
    facts = {"schema_version":1,"participation":"serial","captures":{
        "argument::a":FACT,"argument::b":{**FACT,"initialized":"none"},
        "argument::out":{**FACT,"initialized":"none"}}}
    if fact is not None:
        facts["captures"]["original::hidden"] = fact
    _, output, manifest = generate(tmp_path, source, facts=facts)
    if fact == FACT:
        assert manifest["scope_count"] == 1, manifest["boundaries"]
        scope, = manifest["scopes"]
        assert scope["allocation_preflight"]["resources"] == ["original::hidden"]
        assert scope["gpu_leaves"] == ["original::consumer", "original::producer"]
        text = (output / next(iter(manifest["sources"].values()))["replacement"]).read_text()
        assert "allocated(hidden)" in text
        assert "b=3*b+hidden" in text.lower()
    else:
        assert not manifest["automatic_scope_available"]
        assert any("storage lifetime requires capture proof: original::hidden" in item["reason"]
                   for item in manifest["boundaries"])


@pytest.mark.parametrize("statement", ["hidden = 3*b", "allocate(hidden(8))", "deallocate(hidden)"])
def test_module_capture_assertion_cannot_override_source_allocation_effects(tmp_path, statement):
    source = PROGRAM.replace("implicit none\ncontains", "implicit none\nreal(8),allocatable::hidden(:)\ncontains")
    source = source.replace("b=3*b", statement + "\nb=3*b+hidden")
    facts = {"schema_version":1,"participation":"serial","captures":{
        "argument::a":FACT,"argument::b":{**FACT,"initialized":"none"},
        "argument::out":{**FACT,"initialized":"none"},"original::hidden":FACT}}
    _, _, manifest = generate(tmp_path, source, facts=facts)
    assert not manifest["automatic_scope_available"]
    assert manifest["boundaries"]


def test_hidden_module_origin_is_not_replaced_by_a_declared_numerical_origin(tmp_path):
    source = PROGRAM.replace("implicit none\ncontains", "implicit none\nreal(8),allocatable::hidden(:)\ncontains")
    source = source.replace("real(8),intent(out)::b(:)", "real(8),intent(inout)::b(:)")
    source = source.replace("real(8),intent(out)::b(:),out(:)", "real(8),intent(inout)::b(:),out(:)")
    source = source.replace("b(i)=2*a(i)+real(i,8)", "b(i)=2*a(i)+hidden(i-3)+real(i,8)")
    facts = {"schema_version":1,"participation":"serial","captures":{
        "argument::a":FACT,"argument::b":FACT,
        "argument::out":{**FACT,"initialized":"none"},"original::hidden":FACT}}
    _, _, manifest = generate(tmp_path, source, facts=facts)
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    assert manifest["scopes"][0]["gpu_leaves"] == ["original::consumer"]
    decision = next(item for item in manifest["numerical_decisions"] if item["procedure"] == "original::producer")
    assert not decision["supported"]
    assert "runtime lower bounds" in decision["reason"]


@pytest.mark.parametrize("shadowed", [False, True])
def test_allocatable_owner_and_shadowing_guards_compile_as_fortran(tmp_path, shadowed):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    source = ALLOCATABLE_PROGRAM
    if shadowed:
        source = source.replace("end module", """logical function allocated(value)
real(8),intent(in)::value(:)
allocated=.false.
end function
end module""")
    _, output, manifest = generate(tmp_path, source)
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] == role and item["language"] == "fortran":
                run([fortran,"-std=f2018","-fopenmp","-fcheck=all","-c",str(output/item["path"]),
                     "-o",str((output/item["path"]).with_suffix(".o"))],cwd=output)


def test_collective_scope_is_rejected_without_source_changes(tmp_path):
    directory=tmp_path/"collective"
    directory.mkdir()
    original=directory/"original.f90"
    original.write_text(PROGRAM)
    facts=directory/"facts.json"
    facts.write_text(json.dumps({"schema_version":1,"participation":"collective","captures":{}}))
    result=subprocess.run([sys.executable,"-m","compiler","--form-scopes","--scope-facts",str(facts),
                           "--input",str(original),"--kernel","step","--memory-model","scoped","--gpu-policy","sections",
                           "--json","--output-dir",str(directory/"output")],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode
    assert "serial" in json.loads(result.stdout)["reason"]
    assert not (directory/"output").exists()


def test_automatic_unknown_cost_selects_original_native_span(tmp_path):
    original,output,manifest = generate(tmp_path,mode="auto")
    assert manifest["scope_count"] == 1
    assert not manifest["automatic_estimate_available"]
    assert manifest["scopes"][0]["placement"].startswith("native;")
    replacement = (output/manifest["sources"][str(original)]["replacement"]).read_text()
    owner = replacement[replacement.index("subroutine fort_scope_owner_"):]
    assert "call producer(" in owner
    assert "call transform(" in owner
    assert "call consumer(" in owner
    assert "fort_status = fort_scope_create" not in owner


@pytest.mark.parametrize("mode", ["sections", "auto"])
@pytest.mark.parametrize("body", ["b(2:size(b)-1)=4.d0", "b=4.d0\nb=b+1.d0"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_native_out_partial_writes_and_later_reads_keep_original_source(tmp_path, mode, body, wrapped):
    source = (WRAPPER_PROGRAM if wrapped else PROGRAM).replace(
        "real(8),intent(inout)::b(:)\nb=3*b", "real(8),intent(out)::b(:)\n" + body)
    # Only the defined interior is consumed after the partial native OUT write.
    source = source.replace("do i=1,n\nout(i)=a(i)+b(i)", "do i=2,n-1\nout(i)=a(i)+b(i)")
    original, output, manifest = generate(tmp_path, source, mode=mode)
    assert manifest["scope_count"] == 0
    assert not manifest["source_edits"]
    assert not (output / "entries").exists()
    assert original.read_text() == source
    assert any("native INTENT(OUT) effects require original-position definition hooks: original::transform argument::b"
               in boundary["reason"] for boundary in manifest["boundaries"])


@pytest.mark.parametrize("mode", ["sections", "auto"])
def test_native_out_whole_write_without_reads_remains_supported(tmp_path, mode):
    source = PROGRAM.replace("real(8),intent(inout)::b(:)\nb=3*b", "real(8),intent(out)::b(:)\nb=4.d0")
    _, _, manifest = generate(tmp_path, source, mode=mode)
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    scope, = manifest["scopes"]
    assert scope["gpu_leaves"] == ["original::consumer", "original::producer"]
    assert scope["calls"] == ["original::producer", "original::transform", "original::consumer"]


@pytest.mark.parametrize("mode", ["sections", "auto"])
def test_numerical_out_partial_write_workers_remain_supported(tmp_path, mode):
    source = PARTIAL_PROGRAM.replace("intent(inout)::b", "intent(out)::b").replace(
        "intent(inout)::out", "intent(out)::out")
    _, _, manifest = generate(tmp_path, source, mode=mode)
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    assert manifest["scopes"][0]["gpu_leaves"] == ["original::consumer", "original::producer"]


@pytest.mark.parametrize("mode", ["sections", "auto"])
@pytest.mark.parametrize("order", [("whole_out", "partial_out"), ("partial_out", "whole_out")])
def test_native_nested_out_changes_cannot_reuse_an_earlier_overwrite_proof(tmp_path, mode, order):
    source = PROGRAM.replace("real(8),intent(inout)::b(:)\nb=3*b",
                             "real(8),intent(out)::b(:)\n" + "\n".join("call " + name + "(b)" for name in order))
    source = source.replace("do i=1,n\nout(i)=a(i)+b(i)", "do i=2,n-1\nout(i)=a(i)+b(i)")
    helpers = ("subroutine whole_out(b)\nreal(8),intent(out)::b(:)\nb=4.d0\nend subroutine\n"
               "subroutine partial_out(b)\nreal(8),intent(out)::b(:)\nb(2:size(b)-1)=5.d0\nend subroutine\n")
    source = source.replace("end module", helpers + "end module")
    original, _, manifest = generate(tmp_path, source, mode=mode)
    assert not manifest["source_edits"]
    assert not manifest["automatic_scope_available"]
    assert original.read_text() == source
    assert any("nested native definition changes require original-position hooks" in item["reason"]
               for item in manifest["boundaries"])


def test_writable_alias_is_a_boundary_including_inside_wrappers(tmp_path):
    source=PROGRAM.replace("call producer(a,b,n)","call producer(b,b,n)")
    _,_,manifest=generate(tmp_path,source)
    assert manifest["boundaries"][0]["reason"] == "writable source call arguments alias"
    assert not any("original::producer" in scope["gpu_leaves"] for scope in manifest["scopes"])


@pytest.mark.parametrize("actual", ["b(1)", "b(n)", "b(1)+1.d0"])
def test_indexed_actual_keeps_its_source_position_between_shared_scopes(tmp_path, actual):
    helper = ("subroutine inspect_scalar(value,total)\n"
              "real(8),intent(in)::value\nreal(8),intent(out)::total\n"
              "total=value\nend subroutine\n")
    if ":" in actual:
        helper = helper.replace("::value\n", "::value(:)\n").replace("total=value", "total=sum(value)")
    source = PROGRAM.replace("end module", helper + "end module")
    source = source.replace("subroutine step(a,b,out,n)\n", "subroutine step(a,b,out,n)\nreal(8)::total\n")
    source = source.replace("call consumer(a,b,out,n)",
                            f"call inspect_scalar({actual},total)\ncall consumer(a,b,out,n)\ncall transform(out)")
    original, output, manifest = generate(tmp_path, source)
    assert manifest["scope_count"] == 2
    boundary, = manifest["boundaries"]
    prefix = "array-element/section actual requires in-place mapping and coherence: "
    assert boundary["reason"].startswith(prefix)
    assert boundary["reason"][len(prefix):].lower().replace(" ", "") == actual
    assert [scope["calls"] for scope in manifest["scopes"]] == [
        ["original::producer", "original::transform"], ["original::consumer", "original::transform"]]
    replacement = (output / manifest["sources"][str(original)]["replacement"]).read_text()
    # The original indexed expression remains between completed owners, where
    # scope close has restored GPU writes before its ordinary native evaluation.
    step = replacement.split("subroutine step(a,b,out,n)", 1)[1].split("end subroutine", 1)[0]
    assert step.count("call fort_scope_owner_") == 2
    assert f"\ncall inspect_scalar({actual},total)\ncall fort_scope_owner_" in step


def test_rectangular_native_actual_stays_inside_shared_owner(tmp_path):
    helper = ("subroutine inspect_section(value,total)\n"
              "real(8),intent(in)::value(:)\nreal(8),intent(out)::total\n"
              "total=sum(value)\nend subroutine\n")
    source = PROGRAM.replace("end module", helper + "end module")
    source = source.replace("subroutine step(a,b,out,n)\n", "subroutine step(a,b,out,n)\nreal(8)::total\n")
    source = source.replace("call consumer(a,b,out,n)",
                            "call inspect_section(b(1:n),total)\ncall consumer(a,b,out,n)\ncall transform(out)")
    _, _, manifest = generate(tmp_path, source)
    assert manifest["scope_count"] == 1
    assert not manifest["boundaries"]
    scope, = manifest["scopes"]
    assert scope["calls"] == ["original::producer", "original::transform", "original::inspect_section",
                              "original::consumer", "original::transform"]
    call, = scope["borrowed_views"]["calls"]
    mapping, = call["mappings"]
    assert mapping["resource"] == "argument::b"
    assert mapping["section"]["source_access"].replace(" ", "") == "b(1:n)"
    assert mapping["view_abi_version"] == 1


@pytest.mark.parametrize("actual", ["3.d0", "factor"])
def test_whole_scalar_and_literal_actuals_remain_supported(tmp_path, actual):
    source = PROGRAM.replace("subroutine transform(b)\nreal(8),intent(inout)::b(:)\nb=3*b",
                             "subroutine transform(b,value)\nreal(8),intent(inout)::b(:)\n"
                             "real(8),intent(in)::value\nb=value*b")
    source = source.replace("subroutine step(a,b,out,n)\n", "subroutine step(a,b,out,n)\nreal(8)::factor\n")
    source = source.replace("call producer(a,b,n)", "factor=3.d0\ncall producer(a,b,n)")
    source = source.replace("call transform(b)", f"call transform(b,{actual})")
    _, _, manifest = generate(tmp_path, source)
    assert manifest["scope_count"] == 1
    assert not manifest["boundaries"]
    assert manifest["native_effects"]["complete"]


def test_capture_sections_and_budget_are_public_and_checked(tmp_path):
    facts={"schema_version":1,"participation":"serial","device_budget_bytes":400,
           "captures":{"argument::a":{**FACT,"initialized":"sections",
                                        "sections":[{"lower":[2],"upper":[6]}]},
                       "argument::b":{**FACT,"initialized":"none"},
                       "argument::out":{**FACT,"initialized":"none"}}}
    _,output,manifest=generate(tmp_path,facts=facts)
    assert manifest["device_budget_bytes"]==400
    assert manifest["scopes"][0]["resources"][0]["initialized_sections"]==facts["captures"]["argument::a"]["sections"]
    assert all(sha256((output/name).read_bytes()).hexdigest()==digest
               for name,digest in manifest["artifacts_sha256"].items())


MIRROR_PROGRAM=PROGRAM.replace("subroutine transform(b)\nreal(8),intent(inout)::b(:)\nb=3*b",
                               "subroutine transform(b,c,total)\nreal(8),intent(in)::b(:),c(:)\n"
                               "real(8),intent(out)::total\ntotal=sum(b)+sum(c)")
MIRROR_PROGRAM=MIRROR_PROGRAM.replace("call producer(a,b,n)\ncall transform(b)",
                                     "call producer(a,b,n)\ncall transform(b,b,total)")
MIRROR_PROGRAM=MIRROR_PROGRAM.replace("subroutine step(a,b,out,n)\n", "subroutine step(a,b,out,n)\nreal(8)::total\n")
MIRROR_DRIVER=DRIVER.replace("6*a(i)+3*real(i+3,8)","2*a(i)+real(i+3,8)")
MIRROR_DRIVER=MIRROR_DRIVER.replace("7*a(i)+3*real(i+3,8)","3*a(i)+real(i+3,8)")

ELEMENT_PROGRAM=PROGRAM.split("subroutine step", 1)[0].replace(
    "implicit none", "implicit none\nreal(8)::field(16)", 1) + """subroutine inspect_scalar(value,total)
real(8),intent(in)::value
real(8),intent(out)::total
total=value
end subroutine
subroutine step(a,out,total)
real(8),intent(in)::a(:)
real(8),intent(out)::out(:),total
call producer(a,field,16)
call transform(field)
call inspect_scalar(field(1),total)
call consumer(a,field,out,16)
call transform(out)
end subroutine
end module
"""
ELEMENT_DRIVER="""program caller
use original,only:step,field
real(8)::a(16),output(16),total
integer::repeat,i
do repeat=1,2
 do i=1,16
  a(i)=real(i,8)*0.25d0+repeat
 enddo
 field=-99
 output=-99
 total=-99
 call step(a,output,total)
 if (total/=6*a(1)+3) error stop 'scalar actual observed stale GPU field'
 do i=1,16
  if (field(i)/=6*a(i)+3*real(i,8)) error stop 'module field disagreement'
  if (output(i)/=21*a(i)+9*real(i,8)) error stop 'post-boundary field disagreement'
 enddo
enddo
print *, 'FIELDS_OK'
end program
"""

WRAPPER_PROGRAM=PROGRAM.replace("subroutine step(a,b,out,n)","subroutine wrapper(a,b,out,n)")
WRAPPER_PROGRAM=WRAPPER_PROGRAM.replace("end module", """subroutine step(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:),out(:)
integer,intent(in)::n
call wrapper(a,b,out,n)
call wrapper(a,b,out,n)
end subroutine
end module""")

CONTIGUOUS_WRAPPER_PROGRAM = WRAPPER_PROGRAM.replace("real(8),intent", "real(8),contiguous,intent")


def test_postproof_native_fallback_preserves_contiguous_views(tmp_path, monkeypatch):
    from compiler.tests.test_scoped_planning_sources import generate as planning_generate

    outputs, report = planning_generate(tmp_path, monkeypatch, CONTIGUOUS_WRAPPER_PROGRAM)
    assert report["automatic_estimate_available"]
    scope, = report["scopes"]
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    owner = text.split("subroutine " + scope["owner"] + "(", 1)[1].split("end subroutine", 1)[0]
    preproof = owner.split("_view =>", 1)[0]
    fallback = owner.split("fort_decision%gpu_units == 0) then", 1)[1].split("endif", 1)[0]
    arrays = [parameter["name"] for parameter in scope["parameters"] if parameter["resource"] != "argument::n"]
    for parameter in arrays:
        assert parameter in preproof
        assert parameter + "_view" in fallback
    assert "call wrapper(" in preproof
    assert "call wrapper(" in fallback


def compile_contiguous_native_fallback(directory, *, checkout, profile, source=None, shared_objects=None):
    """Small public-interface fixture; reusable CUDA objects need byte proof."""
    directory.mkdir(parents=True, exist_ok=True)
    original = source or directory / "original.f90"
    if source is None:
        original.write_text(CONTIGUOUS_WRAPPER_PROGRAM)
    captures = {"schema_version": 1, "participation": "serial",
                "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
                "captures": {"argument::a": FACT, "argument::b": {**FACT, "initialized": "none"},
                             "argument::out": {**FACT, "initialized": "none"}}}
    facts = directory / "captures.json"
    facts.write_text(json.dumps(captures))
    output = directory / "output"
    response = run([sys.executable, "-m", "compiler", "--input", str(original), "--kernel", "step",
                    "--form-scopes", "--scope-facts", str(facts), "--memory-model", "scoped", "--gpu-policy", "auto",
                    "--calibration-profile", str(profile), "--json", "--output-dir", str(output)], cwd=checkout,
                   env={**os.environ, "PYTHONPATH": str(checkout), "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "4"})
    manifest = json.loads(response.stdout)["scopes"]
    assert manifest["automatic_estimate_available"], manifest["boundaries"]
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    host = shutil.which("g++-14") or shutil.which("g++")
    nvcc = shutil.which("nvcc")
    assert fortran, "CUDA/Fortran fixture requires installed Fortran toolchain"
    assert host, "CUDA/Fortran fixture requires installed C++ toolchain"
    assert nvcc, "CUDA/Fortran fixture requires installed CUDA toolchain"
    objects, reusable = [], {}
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] != role:
                continue
            target = output / (item["path"].replace("/", "_") + ".o")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = manifest["artifacts_sha256"][item["path"]]
            old = (shared_objects or {}).get(item["path"]) if item["language"] == "cuda" else None
            if old:
                assert old["sha256"] == digest, "CUDA object reuse requires byte-identical source: " + item["path"]
                target = Path(old["object"])
            else:
                command = ([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp", "-I", str(output)]
                           if item["language"] == "cuda" else [fortran, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"])
                run([*command, "-c", str(output / item["path"]), "-o", str(target)], cwd=output)
            objects.append(str(target))
            if item["language"] == "cuda":
                reusable[item["path"]] = {"sha256": digest, "object": str(target)}
    driver = output / "caller.f90"
    driver.write_text(DRIVER)
    target = output / "caller"
    run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", str(driver), *objects,
         "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(target)], cwd=output)
    native = output / "native"
    run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", str(original), str(driver),
         "-o", str(native)], cwd=output)
    reference = run([str(native)], cwd=output, env={**os.environ, "OMP_NUM_THREADS": "4"})
    assert "FIELDS_OK" in reference.stdout
    assert "array temporary" not in reference.stderr.lower()
    return target, output, manifest, reusable


@pytest.fixture(scope="module")
def contiguous_native_fallback(tmp_path_factory):
    profile = os.environ.get("FORT_TEST_SCOPED_PROFILE")
    if not profile:
        pytest.skip("matching explicit offline scoped calibration profile required")
    if not all(shutil.which(tool) for tool in ("nvcc", "gfortran-15", "g++-14")):
        pytest.skip("CUDA/Fortran toolchain unavailable")
    return compile_contiguous_native_fallback(tmp_path_factory.mktemp("contiguous_native_fallback"),
                                              checkout=ROOT, profile=Path(profile).resolve())


@pytest.mark.native
def test_calibrated_all_native_wrappers_do_not_create_array_temporaries(contiguous_native_fallback):
    target, output, _manifest, _reusable = contiguous_native_fallback
    result = run([str(target)], cwd=output,
                 env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"})
    assert "FIELDS_OK" in result.stdout
    assert "array temporary" not in result.stderr.lower()
    decisions = [json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in result.stderr.splitlines()
                 if line.startswith("FORT_SCOPED evidence ")]
    decisions = [entry for entry in decisions if entry["event"] == "decision"]
    assert decisions
    assert all(entry["available"] == 1 for entry in decisions)
    assert all(entry["gpu_units"] == 0 for entry in decisions)
    assert "FORT_SCOPED launch" not in result.stderr
    assert "FORT_SCOPED upload" not in result.stderr
    assert "FORT_SCOPED download" not in result.stderr

_before_step, _step = PROGRAM.split("subroutine step(", 1)
ALLOCATABLE_PROGRAM = _before_step + "subroutine step(" + _step.replace(
    "real(8),intent(in)::a(:)", "real(8),allocatable,intent(in)::a(:)", 1).replace(
    "real(8),intent(out)::b(:),out(:)", "real(8),allocatable,intent(inout)::b(:),out(:)", 1).replace(
    "call producer(a,b,n)\ncall transform(b)\ncall consumer(a,b,out,n)",
    "if (allocated(a) .and. allocated(b) .and. allocated(out)) then\n"
    "call producer(a,b,n)\ncall transform(b)\ncall consumer(a,b,out,n)\nendif")

ALLOCATABLE_DRIVER = DRIVER.replace("do shape=1,2", "n=1\ncall step(a,b,output,n)\ndo shape=0,2").replace(
    "n=8+8*shape", "n=0\n if(shape>0) n=8+8*shape")

PARTIAL_PROGRAM="""module original
contains
subroutine producer(a,b)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer::i
do i=3,6
b(i)=2*a(i)
enddo
end subroutine
subroutine consumer(a,b,out)
real(8),intent(in)::a(:),b(:)
real(8),intent(inout)::out(:)
integer::i
do i=3,6
out(i)=a(i)+b(i)
enddo
end subroutine
subroutine step(a,b,out)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),out(:)
call producer(a,b)
call consumer(a,b,out)
end subroutine
end module
"""
PARTIAL_DRIVER="""program caller
use original,only:step
real(8)::a(-2:5),b(-2:5),output(-2:5)
a=4
b=-99
output=-99
call step(a,b,output)
if (any(b(0:3)/=8) .or. any(output(0:3)/=12)) error stop 'partial field disagreement'
if (any(b(-2:-1)/=-99) .or. any(b(4:5)/=-99)) error stop 'input halo overwritten'
if (any(output(-2:-1)/=-99) .or. any(output(4:5)/=-99)) error stop 'output halo overwritten'
print *, 'FIELDS_OK'
end program
"""

TEAM_DRIVER="""program caller
use original,only:step
real(8)::a(16),b(16),output(16)
!$omp parallel private(a,b,output)
a=4
call step(a,b,output,16)
if(any(b/=[(24.d0+3*i,i=1,16)])) error stop 'team intermediate disagreement'
if(any(output/=[(28.d0+3*i,i=1,16)])) error stop 'team output disagreement'
!$omp end parallel
print *, 'FIELDS_OK'
end program
"""

ALLOCATABLE_TEAM_DRIVER = TEAM_DRIVER.replace("real(8)::a(16),b(16),output(16)",
                                             "real(8),allocatable::a(:),b(:),output(:)").replace(
    "a=4\ncall step", "allocate(a(-2:13),b(-2:13),output(-2:13))\na=4\ncall step").replace(
    "!$omp end parallel", "deallocate(a,b,output)\n!$omp end parallel")


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    nvcc=shutil.which("nvcc")
    host=shutil.which("g++-14") or shutil.which("g++")
    fortran=shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA/Fortran toolchain unavailable")
    directory=tmp_path_factory.mktemp("automatic_source_scope")
    checkout=directory/"independent-compiler"
    shutil.copytree(ROOT/"compiler",checkout/"compiler",ignore=shutil.ignore_patterns("__pycache__",".*cache"))
    targets={}
    runtime_objects = {}
    partial_facts={"schema_version":1,"participation":"serial","captures":{
        "argument::a":{**FACT,"initialized":"sections","sections":[{"lower":[2],"upper":[6]}]},
        "argument::b":{**FACT,"initialized":"none"},"argument::out":{**FACT,"initialized":"none"}}}
    budget_facts={"schema_version":1,"participation":"serial","device_budget_bytes":400,
                  "captures":{"argument::a":FACT,"argument::b":{**FACT,"initialized":"none"},
                              "argument::out":{**FACT,"initialized":"none"}}}
    contiguous = PROGRAM.replace("real(8),intent", "real(8),contiguous,intent")
    element_facts={"schema_version":1,"participation":"serial","captures":{
        "argument::a":FACT,"original::field":FACT,"argument::out":{**FACT,"initialized":"none"}}}
    cases=[("sections","sections",PROGRAM,DRIVER,None), ("auto","auto",PROGRAM,DRIVER,None),
           ("allocated_sections","sections",ALLOCATABLE_PROGRAM,ALLOCATABLE_DRIVER,None),
           ("allocated_auto","auto",ALLOCATABLE_PROGRAM,ALLOCATABLE_DRIVER,None),
           ("contiguous","sections",contiguous,DRIVER,None),
           ("budget","sections",PROGRAM,DRIVER,budget_facts),
           ("mirror","sections",MIRROR_PROGRAM,MIRROR_DRIVER,None),
           ("wrapper","sections",WRAPPER_PROGRAM,DRIVER,None),
           ("partial","sections",PARTIAL_PROGRAM,PARTIAL_DRIVER,partial_facts),
           ("indexed","sections",ELEMENT_PROGRAM,ELEMENT_DRIVER,element_facts)]
    for label,mode,program,driver_source,facts in cases:
        case=directory/label
        checks = "-fcheck=all,array-temps" if label.startswith("allocated") else "-fcheck=all"
        profile = os.environ.get("FORT_TEST_SCOPED_PROFILE") if label == "allocated_auto" else None
        original,output,manifest=generate(case,program,mode=mode,facts=facts,checkout=checkout,profile=profile)
        assert manifest["scope_count"] == (2 if label == "indexed" else 1)
        objects=[]
        # Build common interface first, then generated interfaces, then the
        # original modules with approved replacements. No IR/CUDA introspection.
        sources=manifest["build_sources"]
        for item in [s for s in sources if s["role"]=="common_runtime"]:
            target=output/Path(item["path"]).with_suffix(".o").name
            runtime_key = (manifest["runtime"]["runtime_id"], item["path"])
            if item["language"] == "cuda" and runtime_key in runtime_objects:
                # Runtime identity binds all published headers and sources;
                # this suite uses identical CUDA compiler flags for every case.
                assert sha256((output/item["path"]).read_bytes()).hexdigest() == manifest["artifacts_sha256"][item["path"]]
                objects.append(runtime_objects[runtime_key])
                continue
            command=([nvcc,"-std=c++17","-ccbin",host,"-arch=sm_86","-Xcompiler=-fopenmp","-c"]
                     if item["language"]=="cuda" else [fortran,"-std=f2018","-c"])
            run([*command,str(output/item["path"]),"-o",str(target)],cwd=output)
            objects.append(str(target))
            if item["language"] == "cuda":
                runtime_objects[runtime_key] = str(target)
        for item in [s for s in sources if s["role"]=="shared_entry"]:
            target=(output/item["path"]).with_suffix(".o")
            if item["language"]=="cuda":
                command=[nvcc,"-std=c++17","-ccbin",host,"-arch=sm_86","-Xcompiler=-fopenmp","-I",str(output),"-c"]
            else:
                command=[fortran,"-std=f2018","-c"]
            run([*command,str(output/item["path"]),"-o",str(target)],cwd=output)
            objects.append(str(target))
        for item in [s for s in sources if s["role"]=="original_source"]:
            target=(output/item["path"]).with_suffix(".o")
            run([fortran,"-std=f2018","-fopenmp",checks,"-c",str(output/item["path"]),
                 "-o",str(target)],cwd=output)
            objects.append(str(target))
        driver=output/"caller.f90"
        driver.write_text(driver_source)
        target=output/"caller"
        run([fortran,"-std=f2018","-fopenmp",checks,str(driver),*objects,
             "-L/usr/local/cuda/lib64","-Wl,-rpath,/usr/local/cuda/lib64","-lcudart","-lstdc++",
             "-o",str(target)],cwd=output)
        # Check the original native program too, with all shape/call variations.
        native=output/"native"
        run([fortran,"-std=f2018","-fopenmp",checks,str(original),str(driver),"-o",str(native)],cwd=output)
        run([str(native)],cwd=output,env={**os.environ,"OMP_NUM_THREADS":"4"})
        targets[label]=(target,output)
        if label in {"sections", "allocated_sections"}:
            driver.write_text(ALLOCATABLE_TEAM_DRIVER if label == "allocated_sections" else TEAM_DRIVER)
            team=output/"team-caller"
            run([fortran,"-std=f2018","-fopenmp",checks,str(driver),*objects,
                 "-L/usr/local/cuda/lib64","-Wl,-rpath,/usr/local/cuda/lib64","-lcudart","-lstdc++",
                 "-o",str(team)],cwd=output)
            targets["allocated_team" if label == "allocated_sections" else "team"]=(team,output)
    return targets


@pytest.mark.native
def test_compiler_formed_scope_continues_without_cuda(compiled):
    for target,output in compiled.values():
        result=run([str(target)],cwd=output,
                   env={**os.environ,"CUDA_VISIBLE_DEVICES":"","FORT_RUNTIME_TRACE":"1","OMP_NUM_THREADS":"4"})
        assert "FIELDS_OK" in result.stdout
        assert "FORT_SCOPED upload" not in result.stderr
        assert "FORT_SCOPED launch" not in result.stderr


@pytest.mark.native
def test_existing_openmp_team_keeps_original_native_span_before_descriptor_access(compiled):
    for case in ("team", "allocated_team"):
        target,output=compiled[case]
        result=run([str(target)],cwd=output,
                   env={**os.environ,"FORT_RUNTIME_TRACE":"1","OMP_NUM_THREADS":"4","OMP_DYNAMIC":"FALSE"})
        assert "FIELDS_OK" in result.stdout
        assert "FORT_SCOPED" not in result.stderr


@pytest.mark.cuda
def test_allocatable_roots_preserve_empty_guards_reallocation_and_fields(compiled):
    for case in ("allocated_sections", "allocated_auto"):
        target,output=compiled[case]
        result=run([str(target)],cwd=output,
                   env={**os.environ,"FORT_RUNTIME_TRACE":"1","OMP_NUM_THREADS":"4"})
        assert "FIELDS_OK" in result.stdout
        assert "array temporary" not in result.stderr.lower()
        if case == "allocated_auto" and os.environ.get("FORT_TEST_SCOPED_PROFILE"):
            decisions = [line for line in result.stderr.splitlines() if line.startswith("FORT_SCOPED decision ")]
            assert decisions
            assert all("available=1" in line for line in decisions)
        if case == "allocated_sections":
            if "FORT_SCOPED launch" not in result.stderr:
                pytest.skip("CUDA device unavailable")
            lines = result.stderr.splitlines()
            assert sum(line.startswith("FORT_SCOPED launch ") for line in lines) == 8
            assert sum(line.startswith("FORT_SCOPED upload ") for line in lines) == 8
            assert sum(line.startswith("FORT_SCOPED download ") for line in lines) == 8


@pytest.mark.cuda
def test_compiler_formed_scope_shares_input_across_native_transform(compiled):
    target,output=compiled["sections"]
    result=run([str(target)],cwd=output,
               env={**os.environ,"FORT_RUNTIME_TRACE":"1","OMP_NUM_THREADS":"4"})
    if "FORT_SCOPED launch" not in result.stderr:
        pytest.skip("CUDA device unavailable")
    lines=result.stderr.splitlines()
    assert sum("FORT_SCOPED upload buffer=1 " in line for line in lines)==4
    assert sum("FORT_SCOPED upload buffer=2 " in line for line in lines)==4
    assert sum(line.startswith("FORT_SCOPED upload ") for line in lines)==8
    assert sum(line.startswith("FORT_SCOPED download ") for line in lines)==8
    assert sum(line.startswith("FORT_SCOPED launch ") for line in lines)==8
    assert "FIELDS_OK" in result.stdout


@pytest.mark.cuda
def test_scalar_element_boundary_observes_gpu_produced_module_field(compiled):
    target,output=compiled["indexed"]
    result=run([str(target)],cwd=output,
               env={**os.environ,"FORT_RUNTIME_TRACE":"1","OMP_NUM_THREADS":"4"})
    if "FORT_SCOPED launch" not in result.stderr:
        pytest.skip("CUDA device unavailable")
    assert sum(line.startswith("FORT_SCOPED launch ") for line in result.stderr.splitlines()) == 4
    assert "FIELDS_OK" in result.stdout


@pytest.mark.cuda
@pytest.mark.parametrize(("case","uploads","downloads","launches"),[
    ("mirror",4,8,8), ("wrapper",12,12,16), ("partial",1,2,2), ("budget",8,6,6),
    ("contiguous",8,8,8),
])
def test_source_scope_mirrors_clones_partial_fields_and_native_continuation(compiled,case,uploads,downloads,launches):
    target,output=compiled[case]
    result=run([str(target)],cwd=output,env={**os.environ,"FORT_RUNTIME_TRACE":"1","OMP_NUM_THREADS":"4"})
    if "FORT_SCOPED launch" not in result.stderr:
        pytest.skip("CUDA device unavailable")
    lines=result.stderr.splitlines()
    assert sum(line.startswith("FORT_SCOPED upload ") for line in lines)==uploads, result.stderr
    assert sum(line.startswith("FORT_SCOPED download ") for line in lines)==downloads, result.stderr
    assert sum(line.startswith("FORT_SCOPED launch ") for line in lines)==launches, result.stderr
    if case=="partial":
        assert all("bytes=32" in line for line in lines if " upload " in line or " download " in line)
    assert "FIELDS_OK" in result.stdout
