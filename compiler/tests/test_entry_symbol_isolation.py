"""Independently generated entries link without adapter symbol rewrites."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import replace

import pytest

from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import lower_file
from compiler.tests.test_language_cuda import CUDA_RUNTIME
from compiler.tests.test_native import cuda_device as cuda_device
from compiler.tests.test_sessions import _simulate_cuda


def _entries(directory):
    entries = []
    for name, expression in (("shift_field", "a(i)+5"), ("scale_field", "a(i)*3")):
        output = directory / name
        output.mkdir()
        source = output / "original.f90"
        source.write_text(
            f"module {name}_module\ncontains\n"
            f"subroutine {name}(a,n)\n"
            "integer,intent(inout)::a(:)\ninteger,intent(in)::n\ninteger::i\n"
            f"do i=2,n-1\na(i)={expression}\nenddo\n"
            "end subroutine\nend module\n"
        )
        function, plan = prepare_function(lower_file(source, name))
        entries.append((function, generate_sources(function, plan), output))
    return entries


def test_entry_specific_timing_exports_and_local_cuda_kernels(tmp_path):
    kernels = []
    for function, generated, _ in _entries(tmp_path):
        for phase in ("start", "finish"):
            symbol = f"cpp_{function.name}_{phase}_hot"
            assert f'extern "C" void {symbol}()' in generated.cpp
            assert f'extern "C" void {symbol}()' in generated.cuda
            assert f"bind(C, name='{symbol}')" in generated.fortran
            assert f"subroutine {phase}_hot()" in generated.fortran
            assert f"cpp_{phase}_hot" not in generated.cpp + generated.cuda + generated.fortran
        declarations = re.findall(r"^static __global__ void (\w+)\(", generated.cuda, re.MULTILINE)
        assert len(declarations) == 1
        kernels.append(declarations)
    # Both independent plans deliberately reuse the same region ID and ABI.
    # Internal linkage, rather than renaming the kernel, isolates their bodies.
    assert kernels[0] == kernels[1]


def test_long_entry_timing_bindings_fit_fortran_line_limits(tmp_path):
    function, _, _ = _entries(tmp_path)[0]
    function = replace(function, name="long_entry_" + "x" * 52)
    # Reuse a separately prepared plan; changing the entry name does not alter
    # its numerical statements or their symbols.
    _, plan = prepare_function(function)
    generated = generate_sources(function, plan)
    assert all(len(line) <= 132 for line in generated.fortran.splitlines())
    assert f"bind(C, name='cpp_{function.name}_finish_hot')" in generated.fortran


def _tool(name):
    executable = shutil.which(name)
    if executable is None:
        pytest.skip(f"Native capability unavailable: {name}")
    return executable


def _run(argv, directory):
    result = subprocess.run(
        [str(argument) for argument in argv],
        cwd=directory,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _link_and_execute(directory, backend):
    fortran = _tool("gfortran")
    compiler = _tool("nvcc" if backend == "cuda" else "g++")
    objects = []
    entries = _entries(directory)
    for _function, generated, output in entries:
        (output / "common_functions.cuh").write_text(read_common_header())
        interface = output / "interface.f90"
        interface.write_text(generated.fortran)
        bridge = output / "interface.o"
        _run([fortran, "-fcheck=all", "-J", directory, "-c", interface, "-o", bridge], directory)
        source = generated.cpp if backend == "cpp" else generated.cuda
        if backend == "simulated-cuda":
            (output / "cuda_runtime.h").write_text(CUDA_RUNTIME)
            source = _simulate_cuda(source)
        source_path = output / ("entry.cu" if backend == "cuda" else "entry.cpp")
        source_path.write_text(source)
        implementation = output / "entry.o"
        _run([compiler, "-std=c++17", "-O0", "-c", source_path, "-o", implementation], directory)
        objects.extend((bridge, implementation))
    shift, scale = (function.module for function, _, _ in entries)
    driver = directory / "driver.f90"
    driver.write_text(
        f"program main\nuse {shift}, only: shift_field, start_shift=>start_hot, finish_shift=>finish_hot\n"
        f"use {scale}, only: scale_field, start_scale=>start_hot, finish_scale=>finish_hot\n"
        "implicit none\ninteger::a(5),b(5)\n"
        "a=[-7,1,2,3,90]\nb=[-2,10,11,12,99]\n"
        "call start_shift()\ncall shift_field(a,5)\ncall finish_shift()\n"
        "if(any(a/=[-7,6,7,8,90])) error stop 1\n"
        "call start_scale()\ncall scale_field(a,5)\ncall scale_field(b,5)\ncall finish_scale()\n"
        "if(any(a/=[-7,18,21,24,90])) error stop 2\n"
        "if(any(b/=[-2,30,33,36,99])) error stop 3\n"
        "call shift_field(a,0)\nif(any(a/=[-7,18,21,24,90])) error stop 4\n"
        "print *, 'independent entries: complete arrays and halos correct'\nend program\n"
    )
    driver_object = directory / "driver.o"
    _run([fortran, "-fcheck=all", "-I", directory, "-c", driver, "-o", driver_object], directory)
    binary = directory / "run"
    libraries = ["-lgfortran"] if backend == "cuda" else ["-lstdc++"]
    linker = compiler if backend == "cuda" else fortran
    _run([linker, *objects, driver_object, *libraries, "-o", binary], directory)
    result = _run([binary], directory)
    assert "independent entries: complete arrays and halos correct" in result.stdout


@pytest.mark.native
@pytest.mark.parametrize("backend", ["cpp", "simulated-cuda"])
def test_two_generated_entries_link_and_execute_on_cpu(tmp_path, backend):
    _link_and_execute(tmp_path, backend)


@pytest.mark.native
@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_device")
def test_two_generated_entries_link_and_execute_on_cuda(tmp_path):
    _link_and_execute(tmp_path, "cuda")
