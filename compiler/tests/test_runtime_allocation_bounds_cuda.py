"""Full fields use original hidden allocation bounds in serial and team entries."""

from __future__ import annotations

import json
import os
import shutil
import sys
from hashlib import sha256

import pytest

from compiler.tests.test_module_allocatable_scopes_runtime import ROOT, _run


def _loops(rank, statement):
    lines = (["do j=2,nj-1"] if rank == 2 else []) + ["do i=2,ni-1", statement, "enddo"]
    return "\n".join(lines + (["enddo"] if rank == 2 else []))


def _source(rank, team):
    shape = ",".join(":" for _ in range(rank))
    physical = "i,j" if rank == 2 else "i"
    logical = ",".join(f"lbound(field,{axis},kind=8)+{iterator}-1"
                       for axis, iterator in enumerate(("i", "j")[:rank], 1))
    first = ",".join(f"lbound(field,{axis},kind=8)+1" for axis in range(1, rank + 1))
    value = "real(lbound(field,1,kind=8)+i-1,8)+ &\nreal(size(field,1),8)"
    if rank == 2:
        value += "+ &\n100.d0*real(lbound(field,2,kind=8)+j-1,8)"
    value += f"+ &\nreal(ubound(field,{rank},kind=8),8)"
    produce = _loops(rank, f"field({logical})= &\n{value}")
    consume = _loops(rank, f"out({physical})=field({logical})+real({'i+j' if rank == 2 else 'i'},8)")
    if team:
        produce = "!$omp do\n" + produce + "\n!$omp end do"
        consume = "!$omp do\n" + consume + "\n!$omp end do"
    touch = (f"field({first})= &\nfield({first})+sum(field)+ &\nreal(lbound(field,1,kind=8),8)")
    condition = "ni>2" + (".and.nj>2" if rank == 2 else "")
    if team:
        touch = "!$omp do\ndo i=1,1\nif(" + condition + ") " + touch + "\nenddo\n!$omp end do"
    else:
        touch = "if(allocated(field)) then\nif(" + condition + ") " + touch + "\nendif"
    call_sequence = "call produce(ni,nj)\ncall touch(ni,nj)\ncall consume(out,ni,nj)"
    body = (call_sequence if team else
            "if(allocated(field)) then\n" + call_sequence + "\nelse\ncall touch(ni,nj)\nendif")
    qualified = (f"""subroutine qualified(out,ni,nj)
real(8),contiguous,intent(inout)::out({shape})
integer,intent(in)::ni,nj
if(.not.allocated(field)) return
!$omp parallel default(none) shared(out,ni,nj,field) num_threads(4)
call step(out,ni,nj)
!$omp end parallel
end subroutine
""" if team else "")
    return f"""module original
implicit none
real(8),allocatable::field({shape})
contains
subroutine produce(ni,nj)
integer,intent(in)::ni,nj
integer::i,j
{produce}
end subroutine
subroutine consume(out,ni,nj)
real(8),contiguous,intent(inout)::out({shape})
integer,intent(in)::ni,nj
integer::i,j
{consume}
end subroutine
subroutine touch(ni,nj)
integer,intent(in)::ni,nj
integer::i
{touch}
end subroutine
subroutine step(out,ni,nj)
real(8),contiguous,intent(inout)::out({shape})
integer,intent(in)::ni,nj
{body}
end subroutine
{qualified}end module
"""


def _normalized(rank):
    shape = ",".join(":" for _ in range(rank))
    indices = "i,j" if rank == 2 else "i"
    lowers = ",".join("lb" + str(axis) for axis in range(1, rank + 1))
    value = "real(lb1+i-1,8)+real(size(field,1),8)"
    if rank == 2:
        value += "+100.d0*real(lb2+j-1,8)"
    value += f"+real(lb{rank}+size(field,{rank})-1,8)"
    produce = _loops(rank, f"field({indices})={value}")
    consume = _loops(rank, f"out({indices})=field({indices})+real({'i+j' if rank == 2 else 'i'},8)")
    return f"""module extracted
implicit none
contains
subroutine produce(field,ni,nj,{lowers})
real(8),intent(inout)::field({shape})
integer,intent(in)::ni,nj,{lowers}
integer::i,j
{produce}
end subroutine
subroutine consume(field,out,ni,nj,{lowers})
real(8),intent(in)::field({shape})
real(8),intent(inout)::out({shape})
integer,intent(in)::ni,nj,{lowers}
integer::i,j
{consume}
end subroutine
end module
"""


def _driver(rank, team):
    shape = ",".join(":" for _ in range(rank))
    allocation = "lo1:lo1+ni-1" + (",lo2:lo2+nj-1" if rank == 2 else "")
    output_allocation = "-17:-17+ni-1" + (",-21:-21+nj-1" if rank == 2 else "")
    indices = "lo1+i-1" + (",lo2+j-1" if rank == 2 else "")
    output_indices = "-17+i-1" + (",-21+j-1" if rank == 2 else "")
    first = "lo1+1" + (",lo2+1" if rank == 2 else "")
    condition = "ni>2" + (".and.nj>2" if rank == 2 else "")
    value = "real(lbound(field,1,kind=8)+i-1,8)+ &\nreal(size(field,1),8)"
    if rank == 2:
        value += "+ &\n100.d0*real(lbound(field,2,kind=8)+j-1,8)"
    value += f"+ &\nreal(ubound(field,{rank},kind=8),8)"
    expected = _loops(rank, f"expected({indices})= &\n{value}")
    output = _loops(rank, f"expected_out({output_indices})=expected({indices})+real({'i+j' if rank == 2 else 'i'},8)")
    called = "qualified" if team else "step"
    allocated = (f"allocate(field({allocation}), &\nexpected({allocation}), &\nout({output_allocation}), &\n"
                 f"expected_out({output_allocation}))")
    if rank == 2:
        allocated = ("if(mode=='wide-empty-axis') then\n"
                     " allocate(field(1:0,-1_8:2147483646_8),expected(1:0,-1_8:2147483646_8), &\n"
                     "          out(1:0,1:0),expected_out(1:0,1:0))\n"
                     "else\n " + allocated + "\nendif")
    return f"""program verify
use original
use omp_lib,only:omp_set_dynamic
implicit none
real(8),allocatable::out({shape}),expected({shape}),expected_out({shape})
integer::ni,nj,i,j,shape_id,repeat
integer,parameter::isizes(4)=[0,5,9,13],jsizes(4)=[4,0,7,5]
integer(kind=8)::lo1,lo2
character(24)::mode
call omp_set_dynamic(.false.)
call get_command_argument(1,mode)
if(mode=='unallocated') then
 ni=3
 nj=4
 allocate(out({output_allocation}))
 out=-70.d0
 call {called}(out,ni,nj)
 if(allocated(field).or.any(out/=-70.d0)) error stop 'unallocated capture changed'
 print *, 'UNALLOCATED_OK'
 stop
endif
do shape_id=1,merge(4,1,mode=='fields')
 do repeat=1,2
  ni=isizes(shape_id)
  nj=jsizes(shape_id)
  lo1=-7-3*shape_id-repeat
  lo2=-11-2*shape_id-repeat
  if(mode/='fields') then
   ni=3
   nj=4
   if(mode=='wide-lower') lo1=-2147483649_8
   if(mode=='wide-positive') lo1=2147483648_8
   if(mode=='wide-upper') lo1=2147483646_8
   if(mode=='wide-empty-axis') then
    ni=0
    nj=0
   endif
  endif
  {allocated}
  field=-50.d0
  out=-70.d0
  expected=field
  expected_out=out
  {expected}
  if({condition}) then
   expected({first})=expected({first})+sum(expected)+real(lbound(field,1,kind=8),8)
  endif
  {output}
  call {called}(out,ni,nj)
  if(any(field/=expected)) error stop 'hidden numerical field or halo mismatch'
  if(any(out/=expected_out)) error stop 'consumer field or halo mismatch'
  print *, 'CALL_OK',ni,nj,repeat,lbound(field,1,kind=8),sum(field),sum(out)
  deallocate(field,expected,out,expected_out)
 enddo
enddo
print *, 'FIELDS_OK'
end program
"""


def _facts(source, team):
    stable = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(source): sha256(source.read_bytes()).hexdigest()},
             "captures": {root: dict(stable) for root in ("argument::out", "original::field")}}
    if team:
        lines = source.read_text().splitlines(keepends=True)
        first = next(index for index, line in enumerate(lines, 1) if line.startswith("call step"))
        facts["schema_version"] = 2
        facts["captures"] = {root: {**stable, "association": "shared_whole_storage", "descriptor_uniform": True}
                             for root in facts["captures"]}
        for name in ("ni", "nj"):
            facts["captures"]["argument::" + name] = {**stable, "association": "shared_immutable_control"}
        facts["participation"] = {"kind": "omp_full_team", "dispatch": "qualified_companion",
                                  "entry": "original::step", "expected_omp_level": 1, "host_threads": 4,
                                  "call_sites": [{"source": str(source), "caller": "original::qualified",
                                                  "first_line": first, "last_line": first,
                                                  "span_sha256": sha256(lines[first-1].encode()).hexdigest(),
                                                  "team_first_line": first-1, "team_last_line": first+1,
                                                  "uniform_guard": "unconditional"}]}
    return facts


def _package(source, normalized, rank):
    hashes = {str(source): sha256(source.read_bytes()).hexdigest()}
    entries = []
    for name in ("produce", "consume"):
        parameters = [{"name": "field", "resource": "original::field", "physical_origin": [0]*rank}]
        if name == "consume":
            parameters.append({"name": "out", "resource": "argument::out", "physical_origin": [0]*rank})
        parameters += [{"name": name, "resource": "argument::" + name} for name in ("ni", "nj")]
        parameters += [{"name": "lb" + str(axis), "resource": "original::field", "lower_bound_dimension": axis}
                       for axis in range(1, rank + 1)]
        entries.append({"procedure": "original::" + name, "source_sha256": hashes[str(source)],
                        "path": str(normalized), "sha256": sha256(normalized.read_bytes()).hexdigest(),
                        "entry": "extracted::" + name, "normalization": "whole_storage_rebased_v1",
                        "participation": "serial_coordinator", "capture_safe": True, "preserves_source_order": True,
                        "parameters": parameters})
    return {"schema_version": 1, "source_inputs": hashes, "entries": entries}


@pytest.fixture(scope="module")
def allocation_bound_binaries(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA, Fortran and OpenMP toolchains are required")
    directory = tmp_path_factory.mktemp("original-allocation-bounds")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    fortran_flags = [fortran, "-O0", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    cuda_flags = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp"]
    cache, targets, reuse = {}, {}, []
    for rank in (1, 2):
        for team in (False, True):
            label = ("team" if team else "serial") + "-" + str(rank)
            case = directory / label
            case.mkdir()
            source, normalized, driver = case / "original.f90", case / "normalized.f90", case / "driver.f90"
            source.write_text(_source(rank, team))
            normalized.write_text(_normalized(rank))
            driver.write_text(_driver(rank, team))
            native = case / "native"
            _run([*fortran_flags, str(source), str(driver), "-o", str(native)], case)
            facts, package = case / "captures.json", case / "numerical.json"
            facts.write_text(json.dumps(_facts(source, team), indent=2) + "\n")
            package.write_text(json.dumps(_package(source, normalized, rank), indent=2) + "\n")
            output = case / "generated"
            command = [sys.executable, "-m", "compiler", "--form-scopes", "--scope-facts", str(facts),
                       "--numerical-sources", str(package), "--input", str(source), "--kernel", "step",
                       "--memory-model", "scoped", "--gpu-policy", "sections", "--host-threads", "4",
                       "--json", "--output-dir", str(output)]
            if team:
                command.append("--gpu-collective")
            response = _run(command, checkout, env={**os.environ, "PYTHONPATH": str(checkout), "PYTHONHASHSEED": "0"})
            (case / "public.json").write_text(response.stdout)
            public = json.loads(response.stdout)
            assert public["supported"], public
            manifest = public["scopes"]
            assert manifest["scope_count"] == 1, manifest["boundaries"]
            scope, = manifest["scopes"]
            assert scope["gpu_leaves"] == ["original::consume", "original::produce"]
            assert scope["allocation_preflight"]["bounds_guard"]["resources"] == ["original::field"]
            objects = []
            for role in ("common_runtime", "shared_entry", "original_source"):
                for item in manifest["build_sources"]:
                    if item["role"] != role:
                        continue
                    artifact = item["path"]
                    target = output / (artifact.replace("/", "_") + ".o")
                    if item["language"] == "cuda":
                        headers = [name for name in manifest["artifacts_sha256"] if name.endswith((".h", ".hpp", ".cuh"))]
                        hashes = {name: sha256((output / name).read_bytes()).hexdigest() for name in sorted(headers)}
                        key = (role, manifest["artifacts_sha256"][artifact], tuple(hashes.items()), tuple(cuda_flags))
                        reused = key in cache
                        if reused:
                            target = cache[key]
                        else:
                            _run([*cuda_flags, "-I", str(output), "-c", str(output / artifact), "-o", str(target)], output)
                            cache[key] = target
                        reuse.append({"case": label, "source_sha256": key[1], "headers_sha256": hashes,
                                      "flags": cuda_flags, "reused": reused, "object": str(target)})
                    else:
                        _run([*fortran_flags, "-c", str(output / artifact), "-o", str(target)], output)
                    objects.append(str(target))
            target = output / "verify"
            _run([*fortran_flags, str(driver), *objects, "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64",
                  "-lcudart", "-lstdc++", "-o", str(target)], output)
            references = {}
            modes = ["fields", "unallocated", "wide-lower", "wide-positive", "wide-upper"]
            if rank == 2:
                modes.append("wide-empty-axis")
            for mode in modes:
                reference = _run([str(native), mode], case, timeout=30)
                (output / (mode + "-native.stdout")).write_text(reference.stdout)
                assert "array temporary" not in reference.stderr.lower()
                references[mode] = reference.stdout
            targets[label] = target, output, references
    (directory / "cuda-object-reuse.json").write_text(json.dumps(reuse, indent=2) + "\n")
    return targets


@pytest.mark.cuda
@pytest.mark.parametrize(("label", "mode"), [
    (label, mode) for label in ("serial-1", "team-1", "serial-2", "team-2")
    for mode in ("fields", "unallocated", "wide-lower", "wide-positive", "wide-upper")]
    + [(label, "wide-empty-axis") for label in ("serial-2", "team-2")])
def test_original_allocation_bounds_match_complete_native_fields(allocation_bound_binaries, label, mode):
    executable, output, references = allocation_bound_binaries[label]
    response = _run([str(executable), mode], output,
                    env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"},
                    timeout=30)
    (output / (mode + "-scoped.stdout")).write_text(response.stdout)
    (output / (mode + "-scoped.stderr")).write_text(response.stderr)
    assert response.stdout == references[mode]
    assert "array temporary" not in response.stderr.lower()
    if mode == "fields":
        if "FORT_SCOPED launch " not in response.stderr:
            pytest.skip("CUDA execution unavailable")
        assert response.stderr.count("FORT_SCOPED launch ") == (12 if label.endswith("1") else 8)
        assert response.stdout.count("CALL_OK") == 8
    else:
        assert "FORT_SCOPED" not in response.stderr
