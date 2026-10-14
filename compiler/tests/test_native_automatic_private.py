"""Native automatic PRIVATE arrays retain entry-time allocation descriptors."""

import os
import re
import shutil
from pathlib import Path

import pytest

from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError
from compiler.numerical_contract import cuda_compile_options
from compiler.scopes.native_atomic import summarize
from compiler.scopes.regions import extract_region
from compiler.scopes.segments import grouped_nodes
from compiler.tests.test_source_scopes import FACT, PROGRAM, generate, run


def source_text():
    source = PROGRAM.replace("implicit none\ncontains", "implicit none\ninteger::extent=0,visits=0\ncontains")
    source = source.replace("integer,intent(in)::n\ncall producer(a,b,n)",
                            "integer,intent(in)::n\ninteger::i\nreal(8)::scratch(-2:extent)\n"
                            "extent=extent+7\nvisits=visits+1\ncall producer(a,b,n)")
    return source.replace("call transform(b)\ncall consumer", """!$omp parallel do private(i,scratch) shared(a,b,n)
do i=1,n
scratch=a(i)
b(i)=b(i)+sum(scratch)+real(lbound(scratch,1)+ubound(scratch,1),8)
enddo
!$omp end parallel do
call consumer""")


def generated(tmp_path):
    return generate(tmp_path, source_text(), facts={"schema_version": 1, "participation": "serial",
        "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}})


def test_dynamic_private_uses_original_bounds_without_storage_capture(tmp_path):
    original, output, manifest = generated(tmp_path)
    scope, = manifest["scopes"]
    native, = [item for item in scope["native_operations"] if item["kind"] == "joined native OpenMP"]
    assert native["private_resources"] == ["original::step::i", "original::step::scratch"]
    assert native["sections"]["available"]
    assert set(native["resources"]) == {"argument::a", "argument::b"}
    assert all(item["resource"] != "original::step::scratch" for item in scope["resources"])
    private, = native["completion"]["original_private_arrays"]
    assert private["numerical_authority"] is False
    text = (output / manifest["sources"][str(original)]["replacement"]).read_text()
    owner = text[text.index("subroutine " + scope["owner"]):text.index("end subroutine " + scope["owner"])]
    assert ":: scratch(fort_private_bounds_" in owner
    assert "scratch(-2:extent)" not in owner
    compact = "".join(text.replace("&", "").split())
    assert "lbound(scratch,1,kind=" in compact
    assert "ubound(scratch,1,kind=" in compact
    assert "private(i,scratch)" in owner.lower()
    assert original.read_text() == source_text()


@pytest.mark.parametrize("declaration", ["real(8),save::scratch(3)", "real(8),allocatable::scratch(:)",
    "real(8),pointer::scratch(:)", "real(8),target::scratch(3)", "real(8)::scratch(3)=1"])
def test_private_storage_with_other_lifetime_or_association_is_rejected(tmp_path, declaration):
    source = source_text().replace("real(8)::scratch(-2:extent)", declaration)
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["original::step"]
    group, = [node for node in grouped_nodes(_children(routine.execution)) if isinstance(node, tuple)]
    with pytest.raises(CompilationError, match="PRIVATE"):
        analysis.joined_completion(routine.qualified, group)


def test_native_authority_does_not_allow_dynamic_private_numerical_lowering(tmp_path):
    path = tmp_path / "source.f90"
    path.write_text(source_text())
    analysis = SourceEffects([path])
    routine = analysis.routines["original::step"]
    group, = [node for node in grouped_nodes(_children(routine.execution)) if isinstance(node, tuple)]
    assert analysis.joined_completion(routine.qualified, group).public()["original_private_arrays"]
    with pytest.raises(CompilationError, match="constant explicit bounds"):
        extract_region(analysis, routine, group)


@pytest.mark.parametrize("name", ["CD4divU", "CD4divV", "CD4divW", "Filter3ord"])
def test_primary_source_bodies_retain_native_automatic_private_arrays(tmp_path, name):
    # Optional real-source probe; no application procedure names are recognized
    # by the compiler. The generic fixtures establish the reusable contract.
    filename = "filters.f90" if name == "Filter3ord" else "advection.f90"
    upstream = Path(os.environ.get("FORT_TEST_ELMM_SOURCE", str(
        Path(__file__).resolve().parents[2] / "elmm-pipeline/.cache/repos/elmm/src"))) / filename
    if not upstream.exists():
        pytest.skip("upstream source checkout unavailable")
    source = upstream.read_text()
    body = re.search(r"(?im)^  subroutine " + name + r"\(.*?^  end subroutine " + name + r"\s*$", source, re.S).group()
    path = tmp_path / "original.f90"
    path.write_text("module source_probe\nimplicit none\ninteger,parameter::knd=8\n"
        "integer::tilenx(8),tileny(8),tilenz(8),Unx,Uny,Unz,Vnx,Vny,Vnz,Wnx,Wny,Wnz\n"
        "real(8)::dxmin,dymin,dzmin\ninteger,allocatable::Utype(:,:,:),Vtype(:,:,:),Wtype(:,:,:)\n"
        "contains\n" + body + "\nend module\n")
    analysis = SourceEffects([path])
    analysis.authorize_stable_module_allocatables({"source_probe::" + name for name in ("utype", "vtype", "wtype")})
    routine = analysis.routines["source_probe::" + name.lower()]
    groups = []

    def collect(nodes):
        for node in grouped_nodes(nodes):
            if isinstance(node, tuple):
                groups.append(node)
            elif _kind(node) == "If_Construct":
                collect(_children(node))

    collect(_children(routine.execution))
    assert len(groups) == (3 if name == "Filter3ord" else 1)
    for group in groups:
        proof = analysis.joined_completion(routine.qualified, group)
        assert len(proof.public()["original_private_arrays"]) == (1 if name == "Filter3ord" else 2)
        assert all(not item["numerical_authority"] for item in proof.public()["original_private_arrays"])
        effects = summarize(analysis, routine.qualified, group, proof)
        assert effects["complete"], effects["reasons"]
        assert max(effects["native_atomic"]["unit_operation_counts"]) <= analysis.operation_limit
        arrays = {item["resource"] for item in effects["operations"]
                  if item.get("rank") and item["resource"] not in proof.private_roots}
        expected = ({"argument::u", "argument::utype"} if name == "Filter3ord" else
                    {"argument::u", "argument::v", "argument::w", "argument::" + name[-1].lower() + "2",
                     "source_probe::" + name[-1].lower() + "type"})
        assert arrays == expected


@pytest.mark.native
def test_generated_owner_preserves_entry_bounds_empty_storage_and_native_team(tmp_path):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not all((fc, nvcc, host)):
        pytest.skip("native Fortran and CUDA build toolchain required")
    original, output, manifest = generated(tmp_path)
    reference = output / "reference.f90"
    reference.write_text(original.read_text().replace("module original", "module reference"))
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] != role:
                continue
            path = output / item["path"]
            target = output / f"artifact-{len(objects)}.o"
            flags = ([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp",
                      *cuda_compile_options()] if item["language"] == "cuda" else
                     [fc, "-std=f2018", "-fopenmp", "-fcheck=all", "-ffp-contract=off"])
            run([*flags, "-c", str(path), "-o", str(target)], cwd=output)
            objects.append(str(target))
    driver = output / "driver.f90"
    driver.write_text("""program verify
use iso_fortran_env,only:int64
use original,only:step,extent,visits
use reference,only:native_step=>step,native_extent=>extent,native_visits=>visits
implicit none
real(8)::a(19),b(19),out(19),native_b(19),native_out(19)
integer::initial,i,pass
do pass=1,3
initial=merge(-3,merge(2,300,pass==2),pass==1)
extent=initial
native_extent=initial
visits=0
native_visits=0
a=[(real(i,8)/8,i=1,19)]
call native_step(a,native_b,native_out,19)
call step(a,b,out,19)
if(any(transfer(b,[0_int64],size(b))/=transfer(native_b,[0_int64],size(native_b)))) &
 error stop 'entry-time PRIVATE shape changed'
if(any(transfer(out,[0_int64],size(out))/=transfer(native_out,[0_int64],size(native_out)))) &
 error stop 'complete output differs'
if(visits/=1.or.native_visits/=1) error stop 'prefix replayed'
if(extent/=initial+7.or.native_extent/=extent) error stop 'native scalar update changed'
enddo
print *,'AUTOMATIC_PRIVATE_NATIVE_FIELDS_OK'
end program
""")
    target = output / "verify"
    run([fc, "-std=f2018", "-fopenmp", "-fcheck=all", "-ffp-contract=off", str(reference), str(driver),
         *objects, "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++",
         "-o", str(target)], cwd=output)
    for threads in (1, 4):
        result = run([str(target)], cwd=output, env={**os.environ, "OMP_NUM_THREADS": str(threads),
            "OMP_DYNAMIC": "FALSE", "CUDA_VISIBLE_DEVICES": "", "FORT_RUNTIME_TRACE": "1"})
        assert "AUTOMATIC_PRIVATE_NATIVE_FIELDS_OK" in result.stdout
        assert not re.search(r"FORT_SCOPED (launch|upload|download) ", result.stderr)
