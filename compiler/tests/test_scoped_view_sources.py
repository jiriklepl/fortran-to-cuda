"""Borrowed numerical companions preserve full-root pitches and source indices."""
from __future__ import annotations

import ctypes as c
import json
import os
import shutil
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.common.resources import read_common_header, read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_file
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.numerical import load_numerical_sources
from compiler.tests.test_module_allocatable_scopes_runtime import ROOT, _run
from compiler.tests.test_scoped_planning_runtime import Costs, Decision, costs
from compiler.tests.test_scoped_runtime import SIZE, TOKEN, Access, Layout, Scope, Stats, access
from compiler.tests.test_scoped_views import View, ViewLayout, view

ORIGINAL = """module original
contains
subroutine advance(a,b,out)
real(8),intent(in)::a(-4:,7:),b(-4:,7:)
real(8),intent(inout)::out(-4:,7:)
integer::i,j
do j=lbound(out,2),ubound(out,2)
do i=lbound(out,1),ubound(out,1)
out(i,j)=a(i,j)+2*b(i,j)+real(i+3*j+size(a)+lbound(a,1)+ubound(a,2),8)
enddo
enddo
end subroutine
end module
"""

NORMALIZED = """module numerical
contains
subroutine advance(a,b,out,la,lb)
real(8),intent(in)::a(:,:),b(:,:)
real(8),intent(inout)::out(:,:)
integer,intent(in)::la,lb
integer::i,j
do j=1,size(out,2)
do i=1,size(out,1)
out(i,j)=a(i,j)+2*b(i,j)+real(i+la-1+3*(j+lb-1)+size(a)+la+lbound(a,1)-1+lb+ubound(a,2)-1,8)
enddo
enddo
end subroutine
subroutine replace(out)
real(8),intent(out)::out(:,:)
integer::i,j
do j=1,size(out,2)
do i=1,size(out,1)
out(i,j)=real(i+17*j,8)
enddo
enddo
end subroutine
subroutine bump(out)
real(8),intent(inout)::out(:,:)
integer::i,j
do j=1,size(out,2)
do i=1,size(out,1)
out(i,j)=out(i,j)+1000
enddo
enddo
end subroutine
subroutine dual(left,right)
real(8),intent(inout)::left(:,:),right(:,:)
integer::i,j
do j=1,size(left,2)
do i=1,size(left,1)
left(i,j)=left(i,j)+real(10+i+3*j,8)
right(i,j)=2*right(i,j)+real(2*i-j,8)
enddo
enddo
end subroutine
subroutine fail_late(out,n)
real(8),intent(inout)::out(:,:)
integer,intent(in)::n
integer::i,j
do i=1,size(out,1)
out(i,1)=out(i,1)+100
enddo
do j=2,n
do i=1,size(out,1)
out(i,j)=out(i,j)+200
enddo
enddo
end subroutine
end module
"""

REFERENCE = """subroutine reference(a,b,out,n,m) bind(C)
use iso_c_binding
integer(c_int),value::n,m
real(c_double),intent(in)::a(-4:n-5,7:m+6),b(-4:n-5,7:m+6)
real(c_double),intent(inout)::out(-4:n-5,7:m+6)
integer::i,j
do j=lbound(out,2),ubound(out,2)
do i=lbound(out,1),ubound(out,1)
out(i,j)=a(i,j)+2*b(i,j)+real(i+3*j+size(a)+lbound(a,1)+ubound(a,2),c_double)
enddo
enddo
end subroutine
"""


def emit(directory, entry="advance", *, root_views=True, collective=False):
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "numerical.f90"
    source.write_text(NORMALIZED)
    function, plan = prepare_function(lower_file(source, entry), options=CompilerOptions(gpu_policy="sections"))
    runtime_sources, runtime = read_scoped_runtime()
    emitted = generate_scoped(function, plan, OffloadConfig("sections", host_threads=4, collective=collective),
                              "common_functions.cuh", runtime_id=runtime["runtime_id"], root_views=root_views)
    return emitted, runtime_sources


def normalization_package(directory):
    original, normalized = directory / "original.f90", directory / "normalized.f90"
    original.write_text(ORIGINAL)
    normalized.write_text(NORMALIZED)
    identity = {str(original): sha256(original.read_bytes()).hexdigest()}
    parameters = [{"name": name, "resource": "argument::" + name, "physical_origin": [0, 0]}
                  for name in ("a", "b", "out")]
    parameters += [{"name": name, "resource": "argument::a", "lower_bound_dimension": axis}
                   for name, axis in (("la", 1), ("lb", 2))]
    package = {"schema_version": 1, "source_inputs": identity, "entries": [{
        "procedure": "original::advance", "path": str(normalized), "entry": "numerical::advance",
        "sha256": sha256(normalized.read_bytes()).hexdigest(), "source_sha256": identity[str(original)],
        "normalization": "whole_storage_rebased_v1", "participation": "serial_coordinator",
        "capture_safe": True, "preserves_source_order": True, "parameters": parameters}]}
    loaded = load_numerical_sources(package, SourceEffects([original]))
    return package, loaded


def test_prepared_companion_retains_public_normalization_and_borrowed_view_contract(tmp_path):
    package, loaded = normalization_package(tmp_path)
    assert [parameter.lower_bound_dimension for parameter in loaded["original::advance"].parameters[-2:]] == [1, 2]
    emitted, sources = emit(tmp_path / "borrowed")
    ordinary, _ = emit(tmp_path / "ordinary", root_views=False)
    assert emitted.report["entry"] != ordinary.report["entry"]
    assert emitted.report["argument_order"] == ["context", "mode", "a", "b", "out", "la", "lb"]
    assert all(parameter["passing"] == "root_view_v1" for parameter in emitted.report["array_parameters"])
    assert emitted.report["borrowed_views"]["abi_version"] == 1
    assert "one-based numerical ABI" in emitted.report["borrowed_views"]["logical_bounds"]
    assert "view_entry.hpp" in sources
    assert "RootView<const double, 2>" in emitted.cuda
    assert "ViewAccessBatch<3>" in emitted.cuda
    assert "fort_scope_view_v1" in emitted.fortran
    assert package["entries"][0]["normalization"] == "whole_storage_rebased_v1"


def test_out_companion_uses_partial_definition_events_in_query_and_execution(tmp_path):
    emitted, _ = emit(tmp_path, "replace")
    assert emitted.report["definition_changes"] == ["out"]
    assert emitted.report["planning"]["query_available"]
    definitions = [line for line in emitted.cuda.splitlines() if "forget_view(fort_context," in line]
    assert len(definitions) == 2
    assert any(line.endswith(", false));") for line in definitions)
    assert any(line.endswith(", true));") for line in definitions)
    assert "fort_scope_forget_definition(" not in emitted.cuda


def test_borrowed_companion_keeps_original_full_root_entry_and_rejects_collective_generation(tmp_path):
    with pytest.raises(CompilationError, match="serial host coordinator"):
        emit(tmp_path, collective=True)


def emit_bundle(directory):
    directory.mkdir(parents=True, exist_ok=True)
    package, _ = normalization_package(directory)
    reports = {}
    for entry in ("advance", "replace", "bump", "dual", "fail_late"):
        emitted, runtime_sources = emit(directory / entry, entry)
        (directory / (entry + ".cu")).write_text(emitted.cuda)
        (directory / (entry + ".f90")).write_text(emitted.fortran)
        reports[entry] = emitted.report
    for name, text in runtime_sources.items():
        (directory / name).write_text(text)
    (directory / "common_functions.cuh").write_text(read_common_header())
    (directory / "normalization-package.json").write_text(json.dumps(package, indent=2) + "\n")
    (directory / "bundle.json").write_text(json.dumps(reports, indent=2) + "\n")


def compiler_inventory(directory):
    return {str(path.relative_to(directory)): sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts}


@pytest.fixture(scope="module")
def borrowed_library(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA, C++ and Fortran toolchains required")
    directory = tmp_path_factory.mktemp("borrowed_views")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    before = compiler_inventory(checkout / "compiler")
    (directory / "compiler-snapshot-before.json").write_text(json.dumps(before, indent=2) + "\n")
    generated = directory / "generated"
    _run([sys.executable, "-m", "compiler.tests.test_scoped_view_sources", "--generate", str(generated)], checkout,
         env={**os.environ, "PYTHONPATH": str(checkout), "PYTHONDONTWRITEBYTECODE": "1"})
    reports = json.loads((generated / "bundle.json").read_text())
    flags = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fPIC,-fopenmp"]
    objects = []
    for name in ("scoped_runtime", "advance", "replace", "bump", "dual", "fail_late"):
        target = generated / (name + ".o")
        _run([*flags, "-I", str(generated), "-c", str(generated / (name + ".cu")), "-o", str(target)], generated)
        objects.append(str(target))
    target = generated / "borrowed.so"
    _run([*flags, "-shared", *objects, "-o", str(target)], generated)
    # Compile every public Fortran descriptor interface as an independent caller.
    _run([fortran, "-std=f2018", "-c", "fort_scoped_memory.f90", "-o", "memory.o"], generated)
    for name in reports:
        _run([fortran, "-std=f2018", "-c", name + ".f90", "-o", name + "-interface.o"], generated)
    native = generated / "reference.f90"
    native.write_text(REFERENCE)
    _run([fortran, "-O2", "-std=f2018", "-fcheck=all", "-shared", "-fPIC", str(native), "-o", "native.so"], generated)
    reference = c.CDLL(str(generated / "native.so"))
    reference.reference.argtypes = [c.POINTER(c.c_double)] * 3 + [c.c_int, c.c_int]
    runtime = c.CDLL(str(target))
    runtime.fort_scope_error.restype = c.c_char_p
    signatures = {
        "create": [c.c_int, c.POINTER(TOKEN)],
        "register": [TOKEN, TOKEN, TOKEN, c.POINTER(Layout), c.c_int, c.POINTER(TOKEN)],
        "unregister": [TOKEN, TOKEN], "close": [TOKEN], "wait": [TOKEN], "abandon": [TOKEN],
        "device_begin": [TOKEN, TOKEN, c.POINTER(Access), c.POINTER(c.c_void_p)], "device_end": [TOKEN, TOKEN],
        "host_begin": [TOKEN, TOKEN, c.POINTER(Access)], "host_end": [TOKEN, TOKEN],
        "stats_get": [TOKEN, c.POINTER(Stats)],
        "view_get_v1": [TOKEN, c.POINTER(View), c.POINTER(ViewLayout)],
        "plan_reset_mode": [TOKEN, c.c_uint32], "plan_validate": [TOKEN],
        "plan_select": [TOKEN, c.POINTER(Costs), c.c_int, c.POINTER(Decision)],
    }
    for name, signature in signatures.items():
        getattr(runtime, "fort_scope_" + name).argtypes = signature
    for report in reports.values():
        arrays, scalars = len(report["array_parameters"]), len(report["scalar_parameters"])
        getattr(runtime, report["entry"]).argtypes = [TOKEN, c.c_int] + [c.POINTER(View)] * arrays + [c.POINTER(c.c_int)] * scalars
        getattr(runtime, report["planning"]["entry"]).argtypes = [TOKEN] + [c.POINTER(View)] * arrays + [
            c.POINTER(c.c_int)] * len(report["planning"]["scalar_inputs"])
    cudart = c.CDLL(str(Path(nvcc).resolve().parents[1] / "lib64/libcudart.so"))
    cudart.cudaDeviceSynchronize.argtypes = []
    cudart.cudaMemcpy.argtypes = [c.c_void_p, c.c_void_p, SIZE, c.c_int]
    runtime.cudart = cudart
    after = compiler_inventory(checkout / "compiler")
    (directory / "compiler-snapshot-after.json").write_text(json.dumps(after, indent=2) + "\n")
    assert before == after
    inventory = {str(path.relative_to(generated)): sha256(path.read_bytes()).hexdigest()
                 for path in generated.rglob("*") if path.is_file() and path.name != "artifact-inventory.json"}
    (generated / "artifact-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return runtime, reports, reference


def run(library, entry, scope, descriptors, *, mode=1, scalars=()):
    runtime, reports, _ = library
    values = [c.c_int(value) for value in scalars]
    return getattr(runtime, reports[entry]["entry"])(scope.handle, mode,
        *[c.byref(spec) for spec in descriptors], *[c.byref(value) for value in values])


def packed(values, root_shape, origin, shape):
    return (c.c_double * (shape[0] * shape[1]))(*[
        values[origin[0]+i+root_shape[0]*(origin[1]+j)] for j in range(shape[1]) for i in range(shape[0])])


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(0, 3), (3, 4), (7, 5)])
@pytest.mark.parametrize("mode", [0, 1])
def test_noncontiguous_views_original_bounds_inquiries_and_read_aliases_match_native(borrowed_library, shape, mode):
    runtime, _, reference = borrowed_library
    scope = Scope(runtime)
    n, m = shape
    root_shape = n+4, m+5
    size = root_shape[0] * root_shape[1]
    inputs = (c.c_double * size)(*[i*0.125-21 for i in range(size)])
    output = (c.c_double * size)(*[-91-i*0.25 for i in range(size)])
    expected = list(output)
    status, source = scope.register(inputs, root_shape, lower=[-13, 8])
    scope.check(status)
    status, target = scope.register(output, root_shape, lower=[-13, 8], identity=2)
    scope.check(status)
    origins = ([1, 1], [2, 2], [2, 1])
    descriptors = [view(handle, origin, shape) for handle, origin in zip((source, source, target), origins, strict=True)]
    a, b = (packed(inputs, root_shape, origin, shape) for origin in origins[:2])
    result = packed(output, root_shape, origins[2], shape)
    reference.reference(a, b, result, n, m)
    for j in range(m):
        for i in range(n):
            expected[origins[2][0]+i+root_shape[0]*(origins[2][1]+j)] = result[i+n*j]
    scope.check(run(borrowed_library, "advance", scope, descriptors, mode=mode, scalars=(-4, 7)))
    scope.cpu_begin(target, access(flags=1))
    scope.cpu_end(target)
    assert list(output) == expected
    stats = scope.stats()
    if mode == 1 and n*m:
        union = 2*n*m-(n-1)*(m-1)
        assert stats.allocations == 2
        assert stats.launches > 0
        assert stats.upload_bytes == union*8
        assert stats.download_bytes == n*m*8
    else:
        assert stats.allocations == stats.launches == stats.upload_bytes == stats.download_bytes == 0
    scope.close()
    assert list(output) == expected


@pytest.mark.cuda
def test_disjoint_writable_views_share_one_device_allocation(borrowed_library):
    scope = Scope(borrowed_library[0])
    shape, chunk = (6, 8), (3, 2)
    values = (c.c_double * 48)(*range(48))
    expected = list(values)
    status, buffer = scope.register(values, shape, lower=[-21, -7])
    scope.check(status)
    first, second = view(buffer, [1, 1], chunk), view(buffer, [1, 4], chunk)
    for j in range(2):
        for i in range(3):
            left, right = 1+i+6*(1+j), 1+i+6*(4+j)
            expected[left] += 10+(i+1)+3*(j+1)
            expected[right] = 2*expected[right]+2*(i+1)-(j+1)
    scope.check(run(borrowed_library, "dual", scope, [first, second]))
    stats = scope.stats()
    assert stats.allocations == 1
    assert stats.launches > 0
    assert stats.upload_bytes == 2*3*2*8
    scope.close()
    assert list(values) == expected


@pytest.mark.cuda
@pytest.mark.parametrize("planned", [False, True])
def test_partial_out_preserves_unrelated_dirty_device_sections(borrowed_library, planned):
    runtime, reports, _ = borrowed_library
    scope = Scope(runtime)
    values = (c.c_double * 48)(*range(48))
    expected = [value+1000 for value in values]
    status, buffer = scope.register(values, [6, 8], lower=[-21, -7])
    scope.check(status)
    full, partial = view(buffer, [0, 0], [6, 8]), view(buffer, [2, 3], [3, 2])
    scope.check(run(borrowed_library, "bump", scope, [full]))
    if planned:
        scope.check(runtime.fort_scope_wait(scope.handle))
        scope.check(runtime.fort_scope_plan_reset_mode(scope.handle, 1))
        scope.check(getattr(runtime, reports["replace"]["planning"]["entry"])(scope.handle, c.byref(partial)))
        scope.check(runtime.fort_scope_plan_validate(scope.handle))
        calibration, decision = costs(), Decision()
        calibration.cpu_flops = calibration.cpu_bandwidth = 1
        scope.check(runtime.fort_scope_plan_select(scope.handle, c.byref(calibration), 1, c.byref(decision)))
        assert decision.available
        assert decision.gpu_units > 0
    scope.check(run(borrowed_library, "replace", scope, [partial], mode=2 if planned else 1))
    for j in range(2):
        for i in range(3):
            expected[2+i+6*(3+j)] = (i+1)+17*(j+1)
    assert scope.stats().allocations == 1
    assert scope.stats().upload_bytes == 48*8
    scope.close()
    assert list(values) == expected


@pytest.mark.cuda
def test_view_generation_changes_after_reallocation_and_invalid_views_do_not_execute(borrowed_library):
    runtime = borrowed_library[0]
    scope = Scope(runtime)
    prior = None
    for generation, shape in enumerate(((3, 4), (0, 7), (7, 5)), 1):
        values = (c.c_double * (shape[0]*shape[1]))(*range(shape[0]*shape[1]))
        status, buffer = scope.register(values, shape, lower=[-generation, -11], generation=generation)
        scope.check(status)
        full = view(buffer, [0, 0], shape, generation=generation)
        if prior is not None:
            layout = ViewLayout()
            assert runtime.fort_scope_view_get_v1(scope.handle, c.byref(prior), c.byref(layout)) == 2
        before = scope.stats().launches
        invalid = view(buffer, [0, 0], shape, lower=[-4, 7], generation=generation)
        assert run(borrowed_library, "bump", scope, [invalid]) != 0
        assert scope.stats().launches == before
        scope.check(run(borrowed_library, "bump", scope, [full]))
        scope.check(runtime.fort_scope_wait(scope.handle))
        scope.check(runtime.fort_scope_unregister(scope.handle, buffer))
        assert list(values) == [i+1000 for i in range(shape[0]*shape[1])]
        prior = full
    scope.close()


@pytest.mark.cuda
def test_overlapping_writable_views_fail_before_numerical_execution(borrowed_library):
    scope = Scope(borrowed_library[0])
    values = (c.c_double * 48)(*range(48))
    status, buffer = scope.register(values, [6, 8])
    scope.check(status)
    first, second = view(buffer, [1, 1], [3, 3]), view(buffer, [2, 2], [3, 3])
    assert run(borrowed_library, "dual", scope, [first, second]) == 3
    assert scope.stats().allocations == scope.stats().launches == 0
    scope.close()
    assert list(values) == list(range(48))


@pytest.mark.cuda
def test_stale_view_is_recoverable_before_work_and_postwork_failure_poison_prevents_replay(borrowed_library):
    runtime, reports, _ = borrowed_library
    assert reports["fail_late"]["parallel_regions"] == 2
    scope = Scope(runtime)
    values = (c.c_double * 24)(*range(24))
    status, buffer = scope.register(values, [4, 6])
    scope.check(status)
    full = view(buffer, [0, 0], [4, 6])
    stale = view(buffer, [0, 0], [4, 6], generation=2)
    assert run(borrowed_library, "fail_late", scope, [stale], scalars=(7,)) == 2
    assert scope.stats().allocations == scope.stats().launches == 0
    assert list(values) == list(range(24))
    # Keep a diagnostic pointer to initialized device storage before execution.
    # After poisoning, inspect that storage directly without reopening access or
    # publishing it to the application's host array.
    pointer = c.c_void_p()
    initialize = access(flags=1)
    scope.check(runtime.fort_scope_device_begin(scope.handle, buffer, c.byref(initialize), c.byref(pointer)))
    scope.check(runtime.fort_scope_device_end(scope.handle, buffer))
    scope.check(runtime.fort_scope_wait(scope.handle))
    assert run(borrowed_library, "fail_late", scope, [full], scalars=(7,)) == 6
    observed = (c.c_double * 24)()
    assert runtime.cudart.cudaDeviceSynchronize() == 0
    assert runtime.cudart.cudaMemcpy(c.cast(observed, c.c_void_p), pointer, c.sizeof(observed), 2) == 0
    assert list(observed) == [i+100 if i<4 else i for i in range(24)]
    assert list(values) == list(range(24))
    assert run(borrowed_library, "fail_late", scope, [full], scalars=(6,)) == 6
    scope.check(runtime.fort_scope_abandon(scope.handle))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", type=Path, required=True)
    emit_bundle(parser.parse_args().generate)
