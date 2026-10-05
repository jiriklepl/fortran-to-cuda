"""Exercise persistent ownership, selective coherence, and workspace lifetimes."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SOURCE = """! kernels
module resident_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine resident(a, b, delta, factor, n)
    real(knd), contiguous, intent(inout) :: a(:), b(:)
    real(knd), intent(in) :: delta, factor
    integer, intent(in) :: n
    integer :: i
    do i = 1, n
      a(i) = a(i) + delta
      b(i) = b(i) * factor
    end do
  end subroutine resident
end module resident_module
"""

DRIVER = """program main
  use resident_module
  implicit none
  type(resident_workspace) :: work, alias, independent
  real(knd) :: a(4), b(4), short(3)
  character(len=16) :: mode
  call get_command_argument(1, mode)
  a = [1.0_knd, 2.0_knd, 3.0_knd, 4.0_knd]
  b = 2.0_knd
  call resident_create(work, a + 0.0_knd, b)
  call resident_create(independent, a, b)
  alias = work
  a = -999.0_knd
  b = -999.0_knd
  call resident_run(work, delta=2.0_knd, factor=3.0_knd, n=4)
  a = 10.0_knd
  call resident_update_device(work, a=a)
  call resident_run(alias, delta=1.0_knd, factor=2.0_knd, n=4)
  call resident_update_host(work, b=b)
  if (any(b /= 12.0_knd)) stop 11
  if (any(a /= 10.0_knd)) stop 12
  call resident_update_host(work, a=a)
  if (any(a /= 11.0_knd)) stop 13
  call resident_update_device(work)
  call resident_update_host(work)
  if (mode == 'shape') then
    call resident_update_device(work, a=short)
    stop 20
  end if
  if (mode == 'double_create') then
    call resident_create(work, a, b)
    stop 21
  end if
  call resident_destroy(work)
  call resident_destroy(work)
  if (mode == 'stale') then
    call resident_run(alias, delta=1.0_knd, factor=2.0_knd, n=4)
    stop 22
  end if
  call resident_update_host(independent, a=a, b=b)
  if (any(a /= [1.0_knd,2.0_knd,3.0_knd,4.0_knd])) stop 14
  if (any(b /= 2.0_knd)) stop 15
  call resident_destroy(independent)
  print *, 'resident ok'
end program main
"""


def _tool(name):
    result = shutil.which(name)
    if not result:
        pytest.skip(f"Native capability unavailable: {name}")
    return result


def _run(arguments, directory, *, check=True, env=None):
    result = subprocess.run(arguments, cwd=directory, text=True, capture_output=True, timeout=60, env=env)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def cpu_workspace(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cpu_workspace")
    source = directory / "source.f90"
    source.write_text(SOURCE)
    _run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(source),
            "--kernel",
            "resident",
            "--output-dir",
            str(directory),
        ],
        ROOT,
    )
    (directory / "driver.f90").write_text(DRIVER)
    _run([_tool("g++"), "-std=c++17", "-c", "generated_cpp_impl.cpp", "-o", "impl.o"], directory)
    _run([_tool("gfortran"), "generated_interface.f90", "driver.f90", "impl.o", "-lstdc++", "-o", "run"], directory)
    return directory


@pytest.mark.native
def test_cpu_workspace_owns_arrays_and_updates_selectively(cpu_workspace):
    result = _run(["./run"], cpu_workspace)
    assert "resident ok" in result.stdout


@pytest.mark.native
@pytest.mark.parametrize(
    ("mode", "message"),
    [("shape", "shape does not match"), ("stale", "stale workspace"), ("double_create", "already initialized")],
)
@pytest.mark.parametrize("workspace", ["cpu_workspace", "simulated_cuda_workspace"])
def test_workspace_invalid_uses_are_diagnosed(request, workspace, mode, message):
    directory = request.getfixturevalue(workspace)
    binary = "run_simulated_cuda" if workspace == "simulated_cuda_workspace" else "run"
    result = _run([f"./{binary}", mode], directory, check=False)
    assert result.returncode != 0
    assert message in result.stderr


def _simulate_cuda(source):
    source = source.replace("#include <cuda_runtime.h>", '#include "cuda_runtime.h"')
    expected_launches = source.count("<<<")
    source, launches = re.subn(r"(kernel_region_\d+_device)<<<([^>]+)>>>\(", r"fort_test_launch(\1, \2, ", source)
    assert launches == expected_launches
    return source


@pytest.fixture(scope="module")
def simulated_cuda_workspace(cpu_workspace, tmp_path_factory):
    from compiler.tests.test_language_cuda import CUDA_RUNTIME

    directory = tmp_path_factory.mktemp("simulated_cuda_workspace")
    for name in ("generated_interface.f90", "common_functions.cuh", "driver.f90"):
        shutil.copyfile(cpu_workspace / name, directory / name)
    (directory / "cuda_runtime.h").write_text(CUDA_RUNTIME)
    source = (cpu_workspace / "generated_code.cu").read_text()
    (directory / "simulated_cuda.cpp").write_text(_simulate_cuda(source))
    _run([_tool("g++"), "-std=c++17", "-c", "simulated_cuda.cpp", "-o", "simulated_cuda.o"], directory)
    _run(
        [
            _tool("gfortran"),
            "generated_interface.f90",
            "driver.f90",
            "simulated_cuda.o",
            "-lstdc++",
            "-o",
            "run_simulated_cuda",
        ],
        directory,
    )
    return directory


def _assert_cuda_session_result(result):
    assert result.returncode == 0, result.stdout + result.stderr
    assert "resident ok" in result.stdout
    assert result.stderr.count("FORT_RUNTIME alloc") == 4
    assert result.stderr.count("FORT_RUNTIME kernel") == 2
    assert result.stderr.count("FORT_RUNTIME upload") == 5
    assert result.stderr.count("FORT_RUNTIME download") == 4
    assert result.stderr.count("FORT_RUNTIME free") == 4


@pytest.mark.native
def test_simulated_cuda_sessions_preserve_values_and_transfer_counts(simulated_cuda_workspace):
    result = _run(
        ["./run_simulated_cuda"],
        simulated_cuda_workspace,
        env={**os.environ, "FORT_RUNTIME_TRACE": "1"},
    )
    _assert_cuda_session_result(result)


@pytest.mark.native
@pytest.mark.parametrize("backend", ["serial", "openmp", "cuda-simulated"])
def test_sessions_preserve_logical_arguments_and_host_conditionals(tmp_path, monkeypatch, backend):
    from compiler.driver.pipeline import prepare_function
    from compiler.emission import generate_sources
    from compiler.tests import test_language as language
    from compiler.tests.test_language_cuda import CUDA_RUNTIME

    function, driver = language.structured_case(tmp_path)
    optimized, plan = prepare_function(function)
    resident_driver = """program main
use language_case
real(knd)::a(4)
type(entry_workspace)::work
a=[-4.d0,1.d0,3.d0,9.d0]
call entry_create(work,a)
call entry_run(work,4,.true.)
call entry_update_host(work,a=a)
print *,a
call entry_run(work,4,.false.)
call entry_update_host(work,a=a)
print *,a
call entry_destroy(work)
end program
"""
    if backend == "cuda-simulated":
        (tmp_path / "cuda_runtime.h").write_text(CUDA_RUNTIME)

        def simulated_sources(function, plan):
            sources = generate_sources(function, plan)
            return replace(sources, cpp=_simulate_cuda(sources.cuda))

        monkeypatch.setattr(language, "generate_sources", simulated_sources)
    language.run_reference_and_cpp(
        tmp_path, optimized, plan, driver, openmp=backend == "openmp", generated_driver=resident_driver
    )


@pytest.mark.native
@pytest.mark.parametrize("output_first", [True, False])
def test_session_input_initialization_with_mixed_intents(tmp_path, output_first):
    parameters = "a, b, n" if output_first else "b, a, n"
    source = tmp_path / "source.f90"
    source.write_text(f"""! kernels
module mixed_module
  implicit none
contains
  ! kernel
  subroutine mixed({parameters})
    real, contiguous, intent(out) :: a(:)
    real, contiguous, intent(in) :: b(:)
    integer, intent(in) :: n
    integer :: i
    do i = 1, n
      a(i) = b(i) * 3.0
    end do
  end subroutine mixed
end module mixed_module
""")
    _run(
        [sys.executable, "-m", "compiler", "--input", str(source), "--kernel", "mixed", "--output-dir", str(tmp_path)],
        ROOT,
    )
    (tmp_path / "driver.f90").write_text("""program main
  use mixed_module
  implicit none
  type(mixed_workspace) :: work
  real :: a(3), b(3)
  b = [2., 3., 4.]
  a = -9.
  call mixed_create(work, a=a, b=b+0.)
  b = -1.
  call mixed_run(work, n=3)
  call mixed_update_host(work, a=a)
  if (any(a /= [6., 9., 12.])) stop 1
  call mixed_destroy(work)
end program main
""")
    _run([_tool("g++"), "-std=c++17", "-c", "generated_cpp_impl.cpp", "-o", "impl.o"], tmp_path)
    _run([_tool("gfortran"), "generated_interface.f90", "driver.f90", "impl.o", "-lstdc++", "-o", "run"], tmp_path)
    _run(["./run"], tmp_path)


@pytest.fixture(scope="module", params=[False, True], ids=["pageable", "pinned"])
def cuda_workspace(cpu_workspace, request):
    filename = "run_cuda_pinned" if request.param else "run_cuda"
    flags = ["-DUSE_PINNED_MEMORY"] if request.param else []
    _run([_tool("gfortran"), "-c", "generated_interface.f90", "driver.f90"], cpu_workspace)
    _run(
        [
            _tool("nvcc"),
            "-std=c++17",
            *flags,
            "generated_code.cu",
            "generated_interface.o",
            "driver.o",
            "-lgfortran",
            "-o",
            filename,
        ],
        cpu_workspace,
    )
    return cpu_workspace, filename


@pytest.mark.native
@pytest.mark.cuda
def test_cuda_workspace_compiles(cuda_workspace):
    directory, filename = cuda_workspace
    assert (directory / filename).is_file()


@pytest.mark.native
@pytest.mark.cuda
def test_cuda_workspace_executes(cuda_workspace):
    directory, filename = cuda_workspace
    # Running the same lifecycle driver also checks the native CUDA/Fortran ABI.
    result = _run([f"./{filename}"], directory, check=False, env={**os.environ, "FORT_RUNTIME_TRACE": "1"})
    unavailable = ("CUDA driver version is insufficient", "no CUDA-capable device", "initialization error")
    if result.returncode and any(message in result.stderr for message in unavailable):
        pytest.skip(result.stderr.strip())
    _assert_cuda_session_result(result)


@pytest.mark.native
def test_workspace_names_do_not_collide_with_bridge_dummy_names(tmp_path):
    source = tmp_path / "source.f90"
    source.write_text("""! kernels
module name_collision
contains
  ! kernel
  subroutine fort_v0_a(a_workspace)
    integer, intent(out) :: a_workspace(:)
    integer :: i
    do i = 1, size(a_workspace, 1)
      a_workspace(i) = i
    end do
  end subroutine fort_v0_a
end module name_collision
""")
    _run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(source),
            "--kernel",
            "fort_v0_a",
            "--output-dir",
            str(tmp_path),
        ],
        ROOT,
    )
    _run([_tool("gfortran"), "-c", "generated_interface.f90"], tmp_path)
    _run([_tool("g++"), "-std=c++17", "-c", "generated_cpp_impl.cpp"], tmp_path)
