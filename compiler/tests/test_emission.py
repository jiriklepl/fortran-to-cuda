"""Validate emitter options through the CLI and native compiler frontends."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILL = Path(__file__).parent / "fixtures" / "fill_array.f90"


@pytest.mark.native
@pytest.mark.parametrize("compiler", ["g++", pytest.param("nvcc", marks=pytest.mark.cuda)])
def test_custom_common_header_compiles(compiler: str, tmp_path: Path) -> None:
    executable = shutil.which(compiler)
    if executable is None:
        pytest.skip(f"Native capability unavailable: {compiler} is not installed")
    generation = subprocess.run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(FILL),
            "--kernel",
            "fill_array",
            "--output-dir",
            str(tmp_path),
            "--common-header",
            "custom_support.cuh",
            "--cpp-output",
            "implementation.cpp",
            "--cuda-output",
            "device.cu",
            "--fortran-output",
            "bridge.f90",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert generation.returncode == 0, generation.stdout + generation.stderr
    assert (tmp_path / "custom_support.cuh").is_file()
    assert not (tmp_path / "common_functions.cuh").exists()
    source = "implementation.cpp" if compiler == "g++" else "device.cu"
    compilation = subprocess.run(
        [executable, "-std=c++17", "-c", source, "-o", "generated.o"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr


def _entry_source(name: str) -> str:
    return f"""! kernels
module emission_names
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine {name}(arr, fort_internal_bridge, fort_internal_c_integer)
    real(knd), contiguous, intent(out) :: arr(:)
    integer, intent(in) :: fort_internal_bridge
    real(knd), intent(in) :: fort_internal_c_integer
    integer :: i
    do i = 1, fort_internal_bridge
      arr(i) = fort_internal_c_integer
    end do
  end subroutine {name}
end module emission_names
"""


def _generate_entry(name: str, tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    source = tmp_path / "input.f90"
    source.write_text(_entry_source(name))
    output = tmp_path / "generated"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(source),
            "--kernel",
            name,
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return result, output


@pytest.mark.native
@pytest.mark.parametrize("entry", ["c_int", "c_double", "c_float", "c_size_t"])
def test_iso_kind_entry_names_compile(entry: str, tmp_path: Path) -> None:
    executable = shutil.which("gfortran")
    if executable is None:
        pytest.skip("Native capability unavailable: gfortran is not installed")
    generation, output = _generate_entry(entry, tmp_path)
    assert generation.returncode == 0, generation.stdout + generation.stderr
    compilation = subprocess.run(
        [executable, "-c", "generated_interface.f90", "-o", "generated.o"],
        cwd=output,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr


@pytest.mark.parametrize("entry", ["start_hot", "finish_hot", "knd"])
def test_reserved_public_entry_names_are_diagnosed(entry: str, tmp_path: Path) -> None:
    generation, output = _generate_entry(entry, tmp_path)
    assert generation.returncode != 0
    assert "conflicts with the generated public interface" in generation.stderr
    assert "input.f90:" in generation.stderr
    assert not output.exists()


_TYPED_ARRAY_SOURCE = """! kernels
module typed_arrays
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine typed_fill(indices, values, factor)
    integer, intent(out) :: indices(:,:,:,:)
    real, intent(out) :: values(:,:,:,:)
    real, intent(in) :: factor
    integer :: i,j,k,l
    do l = 1, size(indices, 4)
      do k = 1, size(indices, 3)
        do j = 1, size(indices, 2)
          do i = 1, size(indices, 1)
            indices(i,j,k,l) = i + 10*j + 100*k + 1000*l
            values(i,j,k,l) = factor + 0.25*i
          end do
        end do
      end do
    end do
  end subroutine typed_fill
end module typed_arrays
"""


@pytest.mark.native
@pytest.mark.parametrize("compiler", ["g++", "gfortran", pytest.param("nvcc", marks=pytest.mark.cuda)])
def test_generic_array_abi_compiles(compiler: str, tmp_path: Path) -> None:
    executable = shutil.which(compiler)
    if executable is None:
        pytest.skip(f"Native capability unavailable: {compiler} is not installed")
    source = tmp_path / "typed_arrays.f90"
    source.write_text(_TYPED_ARRAY_SOURCE)
    output = tmp_path / "generated"
    generation = subprocess.run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(source),
            "--kernel",
            "typed_fill",
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert generation.returncode == 0, generation.stdout + generation.stderr
    if compiler == "gfortran":
        arguments = ["-c", "generated_interface.f90"]
    elif compiler == "nvcc":
        arguments = ["-std=c++17", "-DUSE_PINNED_MEMORY", "-c", "generated_code.cu"]
    else:
        arguments = ["-std=c++17", "-fopenmp", "-c", "generated_cpp_impl.cpp"]
    compilation = subprocess.run(
        [executable, *arguments, "-o", "generated.o"],
        cwd=output,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr


@pytest.mark.native
def test_maximum_fortran_rank_with_long_public_name_compiles(tmp_path: Path) -> None:
    executable = shutil.which("gfortran")
    if executable is None:
        pytest.skip("Native capability unavailable: gfortran is not installed")
    name = "array_" + "q" * 57
    shape = ",".join(":" for _ in range(15))
    indices = ",".join(["i", *("1" for _ in range(14))])
    source = tmp_path / "long_rank.f90"
    source.write_text(
        f"""! kernels
module long_rank
contains
  ! kernel
  subroutine rank_fifteen( &
      {name})
    integer, intent(out) :: &
      {name}({shape})
    integer :: i
    do i = 1, size({name}, 1)
      {name}({indices}) = i
    end do
  end subroutine rank_fifteen
end module long_rank
"""
    )
    output = tmp_path / "generated"
    generation = subprocess.run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(source),
            "--kernel",
            "rank_fifteen",
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert generation.returncode == 0, generation.stdout + generation.stderr
    compilation = subprocess.run(
        [executable, "-c", "generated_interface.f90", "-o", "generated.o"],
        cwd=output,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr
