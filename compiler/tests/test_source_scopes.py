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


def test_writable_alias_is_a_boundary_including_inside_wrappers(tmp_path):
    source=PROGRAM.replace("call producer(a,b,n)","call producer(b,b,n)")
    _,_,manifest=generate(tmp_path,source)
    assert manifest["boundaries"][0]["reason"] == "writable source call arguments alias"
    assert not any("original::producer" in scope["gpu_leaves"] for scope in manifest["scopes"])


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
    cases=[("sections","sections",PROGRAM,DRIVER,None), ("auto","auto",PROGRAM,DRIVER,None),
           ("budget","sections",PROGRAM,DRIVER,budget_facts),
           ("mirror","sections",MIRROR_PROGRAM,MIRROR_DRIVER,None),
           ("wrapper","sections",WRAPPER_PROGRAM,DRIVER,None),
           ("partial","sections",PARTIAL_PROGRAM,PARTIAL_DRIVER,partial_facts)]
    for label,mode,program,driver_source,facts in cases:
        case=directory/label
        original,output,manifest=generate(case,program,mode=mode,facts=facts,checkout=checkout)
        assert manifest["scope_count"] == 1
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
@pytest.mark.parametrize(("case","uploads","downloads","launches"),[
    ("mirror",4,8,8), ("wrapper",12,12,16), ("partial",1,2,2), ("budget",8,6,6),
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
