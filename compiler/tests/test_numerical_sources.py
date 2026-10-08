"""Normalized source contracts preserve original ownership and coordinate events."""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from hashlib import sha256

import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.numerical import load_numerical_sources
from compiler.tests.test_source_scopes import FACT, ROOT, run

ORIGINAL = """module settings
implicit none
real(8) :: gain=2
real(8) :: weights(32)
end module
module relay
use settings
end module
module original
use relay
implicit none
contains
subroutine producer(a,b,n)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(-2:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i)
do i=-1,n-4
b(i)=gain*a(i)+weights(i+3)+real(i,8)
enddo
!$omp end parallel do
end subroutine
subroutine transform(b)
real(8),intent(inout)::b(:)
b=3*b
end subroutine
subroutine consumer(b,out,n)
real(8),intent(in)::b(:)
real(8),intent(inout)::out(:)
integer,intent(in)::n
integer::i
do i=2,n-1
out(i)=b(i)
enddo
end subroutine
subroutine step(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),out(:)
integer,intent(in)::n
call producer(a,b,n)
call transform(b)
call consumer(b,out,n)
end subroutine
end module
"""

NORMALIZED = """module extracted
implicit none
contains
subroutine producer(a,b,n,gain,weights,alb,blb,wlb)
real(8),intent(in)::a(:),weights(:),gain
real(8),intent(inout)::b(:)
integer,intent(in)::n,alb,blb,wlb
integer::i
do i=-1,n-4
b(i-blb+1)=gain*a(i-alb+1)+weights(i+3-wlb+1)+real(i,8)
enddo
end subroutine
end module
"""

DRIVER = """program caller
use original
implicit none
real(8),allocatable::a(:),b(:),output(:)
integer::n,i,shape,repeat
weights=[(real(i,8)*0.1d0,i=1,32)]
do shape=1,2
 n=8+shape*8
 allocate(a(-2:n-3),b(-2:n-3),output(-2:n-3))
 do repeat=1,2
  gain=2+repeat
  a=[(real(i,8)*0.25d0,i=-2,n-3)]
  b=-99
  output=-99
  call step(a,b,output,n)
  if (b(-2)/=-297 .or. b(n-3)/=-297) error stop 'intermediate halo'
  if (output(-2)/=-99 .or. output(n-3)/=-99) error stop 'output halo'
  do i=-1,n-4
   if (b(i)/=3*(gain*a(i)+weights(i+3)+real(i,8))) error stop 'source coordinates'
   if (output(i)/=b(i)) error stop 'consumer disagreement'
  enddo
 enddo
 deallocate(a,b,output)
enddo
print *, 'FIELDS_OK'
end program
"""


def package(directory, *, original=ORIGINAL, normalized=NORMALIZED):
    directory.mkdir(parents=True,exist_ok=True)
    source=directory/"original.f90"
    source.write_text(original)
    numerical=directory/"normalized.f90"
    numerical.write_text(normalized)
    hashes={str(source):sha256(source.read_bytes()).hexdigest()}
    parameters=[{"name":name,"resource":resource,**({"physical_origin":[0]} if rank else {})}
                for name,resource,rank in [("a","argument::a",1),("b","argument::b",1),
                                           ("n","argument::n",0),("gain","settings::gain",0),
                                           ("weights","settings::weights",1)]]
    parameters += [{"name":name,"resource":root,"lower_bound_dimension":1}
                   for name,root in [("alb","argument::a"),("blb","argument::b"),("wlb","settings::weights")]]
    document={"schema_version":1,"source_inputs":hashes,"entries":[{
        "procedure":"original::producer","source_sha256":hashes[str(source)],
        "path":str(numerical),"sha256":sha256(numerical.read_bytes()).hexdigest(),
        "entry":"extracted::producer","normalization":"whole_storage_rebased_v1",
        "participation":"serial_coordinator","capture_safe":True,"preserves_source_order":True,
        "parameters":parameters}]}
    return source,numerical,document


def generate(directory, *, original=ORIGINAL, normalized=NORMALIZED):
    source,_numerical,document=package(directory,original=original,normalized=normalized)
    package_path=directory/"numerical.json"
    package_path.write_text(json.dumps(document))
    facts={"schema_version":1,"participation":"serial","sources":document["source_inputs"],
           "captures":{root:FACT for root in ("argument::a","argument::b","argument::out","settings::weights")}}
    facts_path=directory/"facts.json"
    facts_path.write_text(json.dumps(facts))
    output=directory/"output"
    response=run([sys.executable,"-m","compiler","--input",str(source),"--kernel","original::step",
                  "--form-scopes","--scope-facts",str(facts_path),"--numerical-sources",str(package_path),
                  "--memory-model","scoped","--gpu-policy","sections","--json","--output-dir",str(output)],cwd=ROOT)
    manifest=json.loads(response.stdout)["scopes"]
    assert manifest["scope_count"]==1,manifest["boundaries"]
    return source,output,manifest


def test_package_rebases_original_bounds_and_resolves_reexported_roots(tmp_path):
    source,output,manifest=generate(tmp_path)
    assert manifest["scopes"][0]["gpu_leaves"]==["original::consumer","original::producer"]
    assert "settings::weights" in {r["resource"] for r in manifest["scopes"][0]["resources"]}
    text=(output/manifest["sources"][str(source)]["replacement"]).read_text()
    assert "lbound(a, 1)" in text
    assert "lbound(b, 1)" in text
    assert "lbound(weights, 1)" in text
    assert "!$OMP PARALLEL DO" not in text[text.index("subroutine fort_scope_clone_"):]
    assert "!$omp parallel do private(i)" in text # original fallback retained
    assert manifest["numerical_sources"][0]["procedure"]=="original::producer"


@pytest.mark.parametrize(("field","value","reason"),[
    ("sha256","stale","hash differs"), ("normalization","compact","coordinate facts"),
    ("capture_safe",False,"capture/coordinate facts"),
    ("participation","collective","capture/coordinate facts"),
    ("source_sha256","stale","source/capture"),
])
def test_package_rejects_stale_or_unsupported_contracts(tmp_path,field,value,reason):
    source,_normalized,document=package(tmp_path)
    document["entries"][0][field]=value
    with pytest.raises(CompilationError,match=reason):
        load_numerical_sources(document,SourceEffects([source]))


def test_package_rejects_wrong_array_mapping_and_missing_write_intent(tmp_path):
    source,_normalized,document=package(tmp_path)
    altered=copy.deepcopy(document)
    altered["entries"][0]["parameters"][1]["physical_origin"]=[1]
    with pytest.raises(CompilationError,match="zero-origin"):
        load_numerical_sources(altered,SourceEffects([source]))
    source,_normalized,document=package(tmp_path,normalized=NORMALIZED.replace("intent(inout)::b", "intent(in)::b"))
    with pytest.raises(CompilationError,match="written resource"):
        load_numerical_sources(document,SourceEffects([source]))


def test_original_output_definition_is_kept_outside_normalized_access_intents(tmp_path):
    source,output,manifest=generate(tmp_path,original=ORIGINAL.replace("intent(inout)::b(-2:)","intent(out)::b(-2:)"))
    text=(output/manifest["sources"][str(source)]["replacement"]).read_text()
    clone=text[text.index("subroutine fort_scope_clone_"):]
    assert clone.index("fort_scope_forget_definition") < clone.index("fort_status = fort_run(")


def test_module_and_formal_alias_is_an_explained_native_boundary(tmp_path):
    # A read-only alias is valid native Fortran, but separate GPU descriptors
    # would prepare the same handle twice. Reject before any transformed work.
    source=ORIGINAL.replace("call producer(a,b,n)","call producer(weights,b,n)")
    original,_numerical,document=package(tmp_path,original=source)
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.scopes.source import form_source_scopes

    facts={"schema_version":1,"participation":"serial","sources":document["source_inputs"],
           "captures":{root:FACT for root in ("argument::a","argument::b","argument::out","settings::weights")}}
    _outputs,manifest=form_source_scopes([original],"original::step",facts=facts,options=CompilerOptions(opt_level=1),
                                         config=OffloadConfig(policy="sections"),numerical_sources=document)
    assert any("merged entry access descriptors" in boundary["reason"] for boundary in manifest["boundaries"])


def test_effects_keep_do_header_after_leading_openmp_comment(tmp_path):
    source,_normalized,_document=package(tmp_path,original=ORIGINAL.replace("integer::i\n!$omp parallel", "integer::i\nb=0\n!$omp parallel"))
    summary=SourceEffects([source]).summarize("original::producer")
    assert summary["complete"],summary["reasons"]
    assert not summary["cloneable"]
    assert any(operation.get("resource")=="settings::weights" and operation["guard"]
               for operation in summary["operations"])


@pytest.fixture(scope="module")
def compiled_package(tmp_path_factory):
    nvcc=shutil.which("nvcc")
    host=shutil.which("g++-14") or shutil.which("g++")
    fortran=shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA/Fortran toolchain unavailable")
    source,output,manifest=generate(tmp_path_factory.mktemp("numerical_source_scope"))
    objects=[]
    for role in ("common_runtime","shared_entry","original_source"):
        for unit in manifest["build_sources"]:
            if unit["role"]!=role:
                continue
            target=(output/unit["path"]).with_suffix(".o")
            command=([nvcc,"-std=c++17","-ccbin",host,"-arch=sm_86","-Xcompiler=-fopenmp","-I",str(output)]
                     if unit["language"]=="cuda" else [fortran,"-std=f2018","-fopenmp","-fcheck=all"])
            run([*command,"-c",str(output/unit["path"]),"-o",str(target)],cwd=output)
            objects.append(str(target))
    driver=output/"driver.f90"
    driver.write_text(DRIVER)
    native=output/"native"
    run([fortran,"-fopenmp","-fcheck=all",str(source),str(driver),"-o",str(native)],cwd=output)
    run([str(native)],cwd=output,env={**os.environ,"OMP_NUM_THREADS":"4"})
    target=output/"caller"
    run([fortran,"-fopenmp","-fcheck=all",str(driver),*objects,"-L/usr/local/cuda/lib64",
         "-Wl,-rpath,/usr/local/cuda/lib64","-lcudart","-lstdc++","-o",str(target)],cwd=output)
    return target,output


@pytest.mark.native
def test_normalized_package_full_fields_without_cuda(compiled_package):
    target,output=compiled_package
    response=run([str(target)],cwd=output,env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"4"})
    assert "FIELDS_OK" in response.stdout


@pytest.mark.cuda
def test_normalized_package_full_fields_and_shared_module_input(compiled_package):
    target,output=compiled_package
    response=run([str(target)],cwd=output,env={**os.environ,"FORT_RUNTIME_TRACE":"1","OMP_NUM_THREADS":"4"})
    if "FORT_SCOPED launch" not in response.stderr:
        pytest.skip("CUDA device unavailable")
    assert "FIELDS_OK" in response.stdout
    assert response.stderr.count("FORT_SCOPED launch")==8
