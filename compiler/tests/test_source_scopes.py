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


def generate(directory, source=PROGRAM, *, mode="sections", entry="step", facts=None, checkout=None):
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
                    "--json","--output-dir",str(output)],cwd=ROOT,
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


@pytest.mark.parametrize("actual", ["b(1)", "b(n)", "b(1)+1.d0", "b(1:n)"])
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
           ("contiguous","sections",contiguous,DRIVER,None),
           ("budget","sections",PROGRAM,DRIVER,budget_facts),
           ("mirror","sections",MIRROR_PROGRAM,MIRROR_DRIVER,None),
           ("wrapper","sections",WRAPPER_PROGRAM,DRIVER,None),
           ("partial","sections",PARTIAL_PROGRAM,PARTIAL_DRIVER,partial_facts),
           ("indexed","sections",ELEMENT_PROGRAM,ELEMENT_DRIVER,element_facts)]
    for label,mode,program,driver_source,facts in cases:
        case=directory/label
        original,output,manifest=generate(case,program,mode=mode,facts=facts,checkout=checkout)
        assert manifest["scope_count"] == (2 if label == "indexed" else 1)
        objects=[]
        # Build common interface first, then generated interfaces, then the
        # original modules with approved replacements. No IR/CUDA introspection.
        sources=manifest["build_sources"]
        for item in [s for s in sources if s["role"]=="common_runtime"]:
            target=output/Path(item["path"]).with_suffix(".o").name
            command=([nvcc,"-std=c++17","-ccbin",host,"-arch=sm_86","-Xcompiler=-fopenmp","-c"]
                     if item["language"]=="cuda" else [fortran,"-std=f2018","-c"])
            run([*command,str(output/item["path"]),"-o",str(target)],cwd=output)
            objects.append(str(target))
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
            run([fortran,"-std=f2018","-fopenmp","-fcheck=all","-c",str(output/item["path"]),
                 "-o",str(target)],cwd=output)
            objects.append(str(target))
        driver=output/"caller.f90"
        driver.write_text(driver_source)
        target=output/"caller"
        run([fortran,"-std=f2018","-fopenmp","-fcheck=all",str(driver),*objects,
             "-L/usr/local/cuda/lib64","-Wl,-rpath,/usr/local/cuda/lib64","-lcudart","-lstdc++",
             "-o",str(target)],cwd=output)
        # Check the original native program too, with all shape/call variations.
        native=output/"native"
        run([fortran,"-std=f2018","-fopenmp","-fcheck=all",str(original),str(driver),"-o",str(native)],cwd=output)
        run([str(native)],cwd=output,env={**os.environ,"OMP_NUM_THREADS":"4"})
        targets[label]=(target,output)
        if label=="sections":
            driver.write_text(TEAM_DRIVER)
            team=output/"team-caller"
            run([fortran,"-std=f2018","-fopenmp","-fcheck=all",str(driver),*objects,
                 "-L/usr/local/cuda/lib64","-Wl,-rpath,/usr/local/cuda/lib64","-lcudart","-lstdc++",
                 "-o",str(team)],cwd=output)
            targets["team"]=(team,output)
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
    target,output=compiled["team"]
    result=run([str(target)],cwd=output,
               env={**os.environ,"FORT_RUNTIME_TRACE":"1","OMP_NUM_THREADS":"4","OMP_DYNAMIC":"FALSE"})
    assert "FIELDS_OK" in result.stdout
    assert "FORT_SCOPED" not in result.stderr


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
