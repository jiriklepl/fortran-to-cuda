"""Execute generated code against Fortran; missing native capabilities alone may skip."""

import math
import json
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
    "shared_entry.cu", "shared_interface.f90", "common_functions.cuh",
    "scoped_runtime.cu", "fort_scoped_memory.f90", "scoped-runtime.json",
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
    expected_count: int | None = None
    integer_output: bool = False

    @property
    def count(self) -> int:
        if self.expected_count is not None:
            return self.expected_count
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
    Case("imperfect", "native_imperfect", FIXTURES / "native_imperfect.f90", "native_imperfect_module"),
    Case("host_coherence", "native_host_coherence", FIXTURES / "native_host_coherence.f90", "native_host_module"),
    *[
        Case(
            name,
            "native_strides",
            FIXTURES / "native_strides.f90",
            "native_strides_module",
            bounds,
            driver=FIXTURES / "native_strides_driver.f90",
            expected_count=30,
        )
        for name, bounds in (
            ("stride_positive", (2, 8, 2)),
            ("stride_negative", (8, 2, -2)),
            ("stride_positive_singleton", (3, 3, 2)),
            ("stride_negative_singleton", (4, 4, -3)),
            ("stride_positive_empty", (8, 2, 2)),
            ("stride_negative_empty", (2, 8, -2)),
        )
    ],
    Case(
        "integer_rank5",
        "native_integer_rank5",
        FIXTURES / "native_integer_rank5.f90",
        "native_integer_module",
        driver=FIXTURES / "native_integer_rank5_driver.f90",
        expected_count=24,
        integer_output=True,
    ),
    Case(
        "nonaffine",
        "native_nonaffine",
        FIXTURES / "native_nonaffine.f90",
        "native_nonaffine_module",
        driver=FIXTURES / "native_nonaffine_driver.f90",
        expected_count=27,
    ),
    Case(
        "square",
        "native_square",
        FIXTURES / "native_square.f90",
        "native_square_module",
        driver=FIXTURES / "native_square_driver.f90",
        expected_count=16,
    ),
    Case(
        "device_bounds",
        "native_device_bounds",
        FIXTURES / "native_device_bounds.f90",
        "native_device_bounds_module",
        driver=FIXTURES / "native_device_bounds_driver.f90",
        expected_count=13,
    ),
    Case(
        "single_precision",
        "native_single_precision",
        FIXTURES / "native_single_precision.f90",
        "native_single_module",
        driver=FIXTURES / "native_single_precision_driver.f90",
        expected_count=9,
    ),
    *[
        Case(
            name,
            name,
            ROOT / "benchmarks" / "cases" / name / "Fortran" / filename,
            "MomentumAdvection",
            driver=ROOT / "benchmarks" / "cases" / name / "test_main.f90",
        )
        for name, filename in (("CDU", "cdu.f90"), ("CDV", "cvd.f90"), ("CDW", "cdw.f90"))
    ],
]


def _run(command: list[str | Path], cwd: Path, *, timeout: int = 120, threads: int = 2) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(arg) for arg in command],
        cwd=cwd,
        env={**os.environ, "OMP_NUM_THREADS": str(threads), "OMP_DYNAMIC": "FALSE"},
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


def _generate(case: Case, output: Path, *, verbose: bool = False, opt_level: int = 1) -> dict:
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
        "--opt-level",
        str(opt_level),
    ]
    if verbose:
        command.append("--verbose")
    command.append("--json")
    return json.loads(_run(command, ROOT).stdout)


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
    opt_level: int
    report: dict


@pytest.fixture(
    scope="module",
    params=[(case, opt_level) for case in CASES for opt_level in (0, 1)],
    ids=lambda item: f"{item[0].name}-opt{item[1]}",
)
def generated(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Generated:
    case, opt_level = request.param
    directory = tmp_path_factory.mktemp(f"native_{case.name}_opt{opt_level}")
    output = directory / "generated"
    report = _generate(case, output, opt_level=opt_level)
    return Generated(case, output, _driver(case, directory / "driver.f90"), opt_level, report)


def _shared_caller(generated: Generated, mode: int) -> str:
    """Test-only unmanaged caller: register, run, and publish through the public ABI.

    The numerical body comes exclusively from the public compiler artifacts.
    Keep the original argument names in the outer wrapper for keyword calls;
    canonical names in the inner wrapper prevent intrinsic shadowing.
    """
    from compiler.frontend import lower_file
    from compiler.emission.fortran.declarations import public_declaration
    from compiler.emission.fortran.formatting import _fortran_list

    function = lower_file(generated.case.source, generated.case.kernel)
    arrays = [s for s in function.parameters if s.rank]
    scalars = [s for s in function.parameters if not s.rank]
    names = [s.name for s in function.parameters]
    private = [s.cpp_name for s in function.parameters]
    lines = [f"module {generated.case.module}", "use iso_c_binding", "use fort_scoped_memory",
             f"use {generated.report['scoped']['fortran_module']}, only: shared_run=>run",
             "implicit none", "integer,parameter::knd=kind(1.d0)", "contains"]
    lines += _fortran_list(f"subroutine {generated.case.kernel}(", names, ")", 0)
    lines += [public_declaration(s, s.name) for s in function.parameters]
    lines += _fortran_list("call invoke_shared(", names, ")", 0)
    lines += ["end subroutine"]
    lines += _fortran_list("subroutine invoke_shared(", private, ")", 0)
    lines += [public_declaration(s, s.cpp_name).replace(", contiguous", ", target, contiguous")
              for s in function.parameters]
    lines += ["integer(c_int64_t)::context", "type(fort_scope_layout)::layout"]
    for i, s in enumerate(arrays):
        lines += [f"integer(c_int64_t)::handle{i}",
                  f"integer(c_size_t),target::extents{i}({s.rank})",
                  f"integer(c_int64_t),target::bounds{i}({s.rank})"]
    lines += ["call check(fort_scope_create(0_c_int,context))"]
    kinds = {"real": ("REAL64", 8), "real32": ("REAL32", 4), "integer": ("INTEGER32", 4)}
    for i, s in enumerate(arrays):
        kind, size = kinds[s.dtype.value]
        lines += [f"extents{i}=shape({s.cpp_name},kind=c_size_t)", f"bounds{i}=1_c_int64_t",
                  f"layout=fort_scope_layout({s.rank},FORT_SCOPE_{kind},{size}_c_size_t, &",
                  f"  c_loc({s.cpp_name}),c_loc(extents{i}),c_loc(bounds{i}),1_c_int64_t)",
                  f"call check(fort_scope_register(context,{i+1}_c_int64_t,1_c_int64_t,layout, &",
                  f"  {int(s.intent != 'out')}_c_int,handle{i}))"]
    args = ["context", f"{mode}_c_int", *[f"handle{i}" for i in range(len(arrays))],
            *[s.cpp_name for s in scalars]]
    lines += _fortran_list("call check(shared_run(", args, "))", 0)
    lines += ["call check(fort_scope_close(context))", "end subroutine",
              "subroutine check(status)", "integer(c_int),intent(in)::status",
              "if(status/=FORT_SCOPE_OK) then", "print *, 'shared runtime status',status",
              "error stop 1", "endif", "end subroutine", "end module"]
    return "\n".join(lines)+"\n"


def _fortran_objects(generated: Generated, directory: Path, *, reference: bool = False,
                     mode: int = 0) -> list[Path]:
    compiler = _tool("gfortran")
    nx, ny, nz = generated.case.shape
    flags = ["-cpp", "-O0", "-fcheck=bounds", "-ffree-line-length-none", "-J", str(directory), "-I", str(directory)]
    objects = []
    if reference:
        module_source = generated.case.source
    else:
        for name in ("fort_scoped_memory", "shared_interface"):
            obj = directory / (name + ".o")
            _run([compiler, *flags, "-c", generated.directory / (name + ".f90"), "-o", obj], directory)
            objects.append(obj)
        module_source = directory / "caller.f90"
        module_source.write_text(_shared_caller(generated, mode))
    module, driver = directory / "module.o", directory / "driver.o"
    _run([compiler, *flags, "-c", module_source, "-o", module], directory)
    _run([compiler, *flags, f"-DVAR_NX={nx}", f"-DVAR_NY={ny}", f"-DVAR_NZ={nz}",
          "-c", generated.driver, "-o", driver], directory)
    return [*objects, module, driver]


def _values(executable: Path, case: Case, *, threads: int = 2) -> list[float | int]:
    result = _run([executable], executable.parent, threads=threads)
    convert = int if case.integer_output else float
    values = [convert(token) for token in result.stdout.split()]
    assert len(values) == case.count, result.stdout
    assert all(math.isfinite(value) for value in values), result.stdout
    return values


@pytest.fixture(scope="module")
def reference(generated: Generated) -> list[float | int]:
    directory = generated.directory.parent / "reference"
    directory.mkdir()
    objects = _fortran_objects(generated, directory, reference=True)
    executable = directory / "reference"
    _run([_tool("gfortran"), *objects, "-o", executable], directory)
    return _values(executable, generated.case)


def test_generation_is_deterministic(generated: Generated, tmp_path: Path) -> None:
    for verbose in (False, True):
        output = tmp_path / ("verbose" if verbose else "normal")
        _generate(generated.case, output, verbose=verbose, opt_level=generated.opt_level)
        for filename in OUTPUT_FILES:
            assert (output / filename).read_bytes() == (generated.directory / filename).read_bytes(), filename

    if generated.case.name in ("CDU", "CDV", "CDW"):
        cuda = (generated.directory / "shared_entry.cu").read_text()
        assert len(re.findall(r"\b__global__\s+void\b", cuda)) == (4 if generated.opt_level == 0 else 1)


@pytest.mark.parametrize("existing_outputs", [False, True], ids=["new-directory", "existing-outputs"])
@pytest.mark.parametrize("kernel", ["native_recurrence", "native_indirect_conflict", "native_square_collision"])
def test_rejected_program_preserves_outputs(tmp_path: Path, kernel: str, *, existing_outputs: bool) -> None:
    output = tmp_path / "output"
    expected = {filename: f"original {filename}\n" for filename in OUTPUT_FILES} if existing_outputs else {}
    if existing_outputs:
        output.mkdir()
        for filename, contents in expected.items():
            (output / filename).write_text(contents)
    source = FIXTURES / f"{kernel}.f90"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(source),
            "--kernel",
            kernel,
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
    assert re.search(rf"{kernel}\.f90:\d+", result.stderr), result.stderr
    assert "arr" in result.stderr, result.stderr
    actual = {path.name: path.read_text() for path in output.iterdir()} if output.exists() else {}
    assert actual == expected


@pytest.mark.native
@pytest.mark.parametrize("threads", [1, 2], ids=["one-thread", "two-threads"])
def test_shared_cpu_matches_fortran(
    generated: Generated, reference: list[float | int], cuda_object: Path,
    shared_runtime_object: Path, tmp_path: Path, threads: int,
) -> None:
    objects = _fortran_objects(generated, tmp_path)
    executable = tmp_path / "generated"
    _run([_tool("nvcc"), *objects, cuda_object, shared_runtime_object,
          "-Xcompiler=-fopenmp", "-lgfortran", "-o", executable], tmp_path)
    actual = _values(executable, generated.case, threads=threads)
    if generated.case.integer_output:
        assert actual == reference
    else:
        assert actual == pytest.approx(reference, rel=0, abs=1e-10)


@pytest.fixture(scope="module")
def shared_runtime_object(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("shared_native_runtime")
    target = directory / "runtime.o"
    _run([_tool("nvcc"), "-std=c++17", "-O0", "-Xcompiler=-fopenmp",
          "-c", ROOT / "compiler/runtime/scoped_runtime.cu", "-o", target], directory)
    return target


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
            "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true",
            "-Xcompiler=-ffp-contract=off,-fopenmp",
            "-I",
            generated.directory,
            "-c",
            generated.directory / "shared_entry.cu",
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
def test_cuda_matches_fortran(
    generated: Generated, reference: list[float | int], cuda_object: Path,
    shared_runtime_object: Path, tmp_path: Path
) -> None:
    objects = _fortran_objects(generated, tmp_path, mode=1)
    executable = tmp_path / "generated"
    _run([_tool("nvcc"), *objects, cuda_object, shared_runtime_object,
          "-Xcompiler=-fopenmp", "-lgfortran", "-o", executable], tmp_path)
    actual = _values(executable, generated.case)
    if generated.case.integer_output:
        assert actual == reference
    else:
        assert actual == pytest.approx(reference, rel=0, abs=1e-10)
