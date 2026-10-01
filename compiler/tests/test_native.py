"""Execute generated code against Fortran; missing native capabilities alone may skip."""

import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
OUTPUT_FILES = (
    "generated_code.cu",
    "generated_cpp_impl.cpp",
    "generated_interface.f90",
    "common_functions.cuh",
)
LONG_PROCEDURE = "native_long_procedure_" + "x" * 41
LONG_ARRAY = "long_array_argument_" + "a" * 43
LONG_VALUE = "long_value_argument_" + "v" * 43


@dataclass(frozen=True)
class Case:
    name: str
    kernel: str
    source: Path
    module: str
    shape: tuple[int, int, int] = (5, 4, 3)
    halo: int = 0
    driver: Path | None = None
    keyword_arguments: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return math.prod(n + 2 * self.halo for n in self.shape)


CASES = [
    Case("fill", "fill_array", FIXTURES / "fill_array.f90", "fill_array_module"),
    Case("scale", "scale_array", FIXTURES / "scale_array.f90", "scale_array_module"),
    Case("fill_singleton", "fill_array", FIXTURES / "fill_array.f90", "fill_array_module", (1, 1, 1)),
    Case("scale_singleton", "scale_array", FIXTURES / "scale_array.f90", "scale_array_module", (1, 1, 1)),
    Case("fill_empty", "fill_array", FIXTURES / "fill_array.f90", "fill_array_module", (0, 4, 3)),
    Case("scale_empty", "scale_array", FIXTURES / "scale_array.f90", "scale_array_module", (5, 0, 3)),
    Case("halo", "native_halo", FIXTURES / "native_halo.f90", "native_halo_module", halo=1),
    Case("halo_zero_trip", "native_halo", FIXTURES / "native_halo.f90", "native_halo_module", (0, 4, 3), halo=1),
    Case("halo_empty_range", "native_halo", FIXTURES / "native_halo.f90", "native_halo_module", (-1, 4, 3), halo=1),
    Case("halo_singleton", "native_halo", FIXTURES / "native_halo.f90", "native_halo_module", (1, 1, 1), halo=1),
    Case("order", "native_order", FIXTURES / "native_order.f90", "native_order_module"),
    Case("literals", "native_literals", FIXTURES / "native_literals.f90", "native_literals_module"),
    Case("half_bound", "native_half_bound", FIXTURES / "native_half_bound.f90", "native_half_bound_module"),
    Case(
        "shadow",
        "native_shadow",
        FIXTURES / "native_shadow.f90",
        "native_shadow_module",
        keyword_arguments=("arr", "value", "size", "int", "real"),
    ),
    Case(
        "long_names",
        LONG_PROCEDURE,
        FIXTURES / "native_long_names.f90",
        "native_long_names_module",
        keyword_arguments=(LONG_ARRAY, LONG_VALUE, "nx", "ny", "nz"),
    ),
    *[
        Case(
            name,
            name,
            ROOT / "benchmarks" / name / "Fortran" / filename,
            "MomentumAdvection",
            driver=ROOT / "benchmarks" / name / "test_main.f90",
        )
        for name, filename in (("CDU", "cdu.f90"), ("CDV", "cvd.f90"), ("CDW", "cdw.f90"))
    ],
]


def _run(command: list[str | Path], cwd: Path, *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(arg) for arg in command],
        cwd=cwd,
        env={**os.environ, "OMP_NUM_THREADS": "2", "OMP_DYNAMIC": "FALSE"},
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    assert result.returncode == 0, f"Command: {command}\n{result.stdout}\n{result.stderr}"
    return result


def _tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        pytest.skip(f"Native capability unavailable: {name} is not installed")
    return executable


def _generate(case: Case, output: Path, *, verbose: bool = False) -> None:
    command = [
        sys.executable,
        "-m",
        "compiler",
        "--input",
        str(case.source),
        "--kernel",
        case.kernel,
        "--output-dir",
        str(output),
    ]
    if verbose:
        command.append("--verbose")
    _run(command, ROOT)


def _driver(case: Case, destination: Path) -> Path:
    if case.driver is not None:
        return case.driver
    nx, ny, nz = case.shape
    arguments = ["arr", "1.75_knd", "nx", "ny", "nz"]
    if case.keyword_arguments:
        arguments = [f"{name}={argument}" for name, argument in zip(case.keyword_arguments, arguments, strict=True)]
    call_arguments = ", &\n      ".join(arguments)
    # Inout cases print all cells, making untouched halo corruption observable.
    destination.write_text(
        f"""program native_test
  use {case.module}
  implicit none
  integer, parameter :: nx = {nx}, ny = {ny}, nz = {nz}, halo = {case.halo}
  real(knd) :: arr(nx + 2*halo, ny + 2*halo, nz + 2*halo)
  integer :: i, j, k
  do k = 1, size(arr, 3)
    do j = 1, size(arr, 2)
      do i = 1, size(arr, 1)
        arr(i,j,k) = 0.25_knd*i + 1.5_knd*j - 0.125_knd*k
      end do
    end do
  end do
  call {case.kernel}( &
      {call_arguments})
  do k = 1, size(arr, 3)
    do j = 1, size(arr, 2)
      do i = 1, size(arr, 1)
        write(*,'(g0.17)') arr(i,j,k)
      end do
    end do
  end do
end program native_test
"""
    )
    return destination


@dataclass(frozen=True)
class Generated:
    case: Case
    directory: Path
    driver: Path


@pytest.fixture(scope="module", params=CASES, ids=lambda case: case.name)
def generated(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Generated:
    case = request.param
    directory = tmp_path_factory.mktemp(f"native_{case.name}")
    output = directory / "generated"
    _generate(case, output)
    return Generated(case, output, _driver(case, directory / "driver.f90"))


def _fortran_objects(generated: Generated, directory: Path, *, reference: bool = False) -> list[Path]:
    compiler = _tool("gfortran")
    module_source = generated.case.source if reference else generated.directory / "generated_interface.f90"
    nx, ny, nz = generated.case.shape
    flags = ["-cpp", "-O0", "-fcheck=bounds", "-ffree-line-length-none", "-J", str(directory), "-I", str(directory)]
    objects = [directory / "module.o", directory / "driver.o"]
    _run([compiler, *flags, "-c", module_source, "-o", objects[0]], directory)
    _run(
        [
            compiler,
            *flags,
            f"-DVAR_NX={nx}",
            f"-DVAR_NY={ny}",
            f"-DVAR_NZ={nz}",
            "-c",
            generated.driver,
            "-o",
            objects[1],
        ],
        directory,
    )
    return objects


def _values(executable: Path, expected_count: int) -> list[float]:
    result = _run([executable], executable.parent)
    values = [float(token) for token in result.stdout.split()]
    assert len(values) == expected_count, result.stdout
    assert all(math.isfinite(value) for value in values), result.stdout
    return values


@pytest.fixture(scope="module")
def reference(generated: Generated) -> list[float]:
    directory = generated.directory.parent / "reference"
    directory.mkdir()
    objects = _fortran_objects(generated, directory, reference=True)
    executable = directory / "reference"
    _run([_tool("gfortran"), *objects, "-o", executable], directory)
    return _values(executable, generated.case.count)


def test_generation_is_deterministic(generated: Generated, tmp_path: Path) -> None:
    for verbose in (False, True):
        output = tmp_path / ("verbose" if verbose else "normal")
        _generate(generated.case, output, verbose=verbose)
        for filename in OUTPUT_FILES:
            assert (output / filename).read_bytes() == (generated.directory / filename).read_bytes(), filename

    if generated.case.name in ("CDU", "CDV", "CDW"):
        cuda = (generated.directory / "generated_code.cu").read_text()
        assert len(re.findall(r"\b__global__\s+void\b", cuda)) == 4


@pytest.mark.parametrize("existing_outputs", [False, True], ids=["new-directory", "existing-outputs"])
def test_rejected_program_preserves_outputs(tmp_path: Path, *, existing_outputs: bool) -> None:
    output = tmp_path / "output"
    expected = {filename: f"original {filename}\n" for filename in OUTPUT_FILES} if existing_outputs else {}
    if existing_outputs:
        output.mkdir()
        for filename, contents in expected.items():
            (output / filename).write_text(contents)
    source = FIXTURES / "native_recurrence.f90"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(source),
            "--kernel",
            "native_recurrence",
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0, result.stdout
    assert re.search(r"native_recurrence\.f90:\d+", result.stderr), result.stderr
    assert "arr" in result.stderr, result.stderr
    actual = {path.name: path.read_text() for path in output.iterdir()} if output.exists() else {}
    assert actual == expected


@pytest.mark.native
@pytest.mark.parametrize("openmp", [False, True], ids=["serial", "openmp"])
def test_cpp_matches_fortran(generated: Generated, reference: list[float], tmp_path: Path, *, openmp: bool) -> None:
    compiler = _tool("g++")
    objects = _fortran_objects(generated, tmp_path)
    cpp_object = tmp_path / "generated.o"
    flags = ["-fopenmp"] if openmp else []
    _run(
        [
            compiler,
            "-std=c++17",
            "-O0",
            "-ffp-contract=off",
            *flags,
            "-I",
            generated.directory,
            "-c",
            generated.directory / "generated_cpp_impl.cpp",
            "-o",
            cpp_object,
        ],
        tmp_path,
    )
    executable = tmp_path / "generated"
    _run([_tool("gfortran"), *objects, cpp_object, "-lstdc++", *flags, "-o", executable], tmp_path)
    assert _values(executable, generated.case.count) == pytest.approx(reference, rel=0, abs=1e-10)


@pytest.fixture(scope="module")
def cuda_object(generated: Generated) -> Path:
    compiler = _tool("nvcc")
    directory = generated.directory.parent / "cuda"
    directory.mkdir()
    destination = directory / "generated.o"
    _run(
        [
            compiler,
            "-std=c++17",
            "-O0",
            "--fmad=false",
            "-I",
            generated.directory,
            "-c",
            generated.directory / "generated_code.cu",
            "-o",
            destination,
        ],
        directory,
    )
    return destination


@pytest.mark.cuda
def test_cuda_compiles(cuda_object: Path) -> None:
    assert cuda_object.stat().st_size > 0


@pytest.fixture(scope="session")
def cuda_device(tmp_path_factory: pytest.TempPathFactory) -> None:
    compiler = _tool("nvcc")
    directory = tmp_path_factory.mktemp("cuda_capability")
    source = directory / "probe.cu"
    source.write_text(
        """#include <cuda_runtime.h>
#include <cstdio>
int main() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        std::fprintf(stderr, "CUDA device unavailable: %s; count=%d\\n", cudaGetErrorString(status), count);
        return 77;
    }
    status = cudaSetDevice(0);
    if (status == cudaSuccess) status = cudaFree(nullptr);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA context unavailable: %s\\n", cudaGetErrorString(status));
        return 77;
    }
    return 0;
}
"""
    )
    executable = directory / "probe"
    _run([compiler, source, "-o", executable], directory)
    result = subprocess.run([str(executable)], cwd=directory, capture_output=True, text=True, check=False, timeout=30)
    if result.returncode == 77:
        pytest.skip(result.stderr.strip())
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.native
@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_device")
def test_cuda_matches_fortran(generated: Generated, reference: list[float], cuda_object: Path, tmp_path: Path) -> None:
    objects = _fortran_objects(generated, tmp_path)
    executable = tmp_path / "generated"
    _run([_tool("nvcc"), *objects, cuda_object, "-lgfortran", "-o", executable], tmp_path)
    assert _values(executable, generated.case.count) == pytest.approx(reference, rel=0, abs=1e-10)
