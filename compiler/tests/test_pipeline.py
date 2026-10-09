"""Exercise option publication safety and checked schedules against source execution."""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.frontend import lower_file
from compiler.ir import ArrayAccess, Assignment, Binary, Literal, Loop, Reference, Size, Unary

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILES = ("generated_code.cu", "generated_cpp_impl.cpp", "generated_interface.f90", "common_functions.cuh")


@pytest.mark.parametrize("opt_level", [-1, 2])
def test_invalid_optimization_level_is_rejected(opt_level: int) -> None:
    from compiler.ir import CompilationError

    with pytest.raises(CompilationError, match="optimization level must be 0 or 1"):
        CompilerOptions(opt_level=opt_level)


def test_nondefault_scope_transfers_require_explicit_scoped_ownership():
    from compiler.emission import generate_sources
    from compiler.ir import CompilationError
    from compiler.offload.config import OffloadConfig

    with pytest.raises(CompilationError, match="require scoped memory"):
        CompilerOptions(gpu_policy="sections", scope_transfers="pinned")
    options = CompilerOptions(gpu_policy="sections", memory_model="scoped", scope_transfers="pinned")
    function, plan = prepare_function(lower_file(Path(__file__).parent / "fixtures" / "fill_array.f90", "fill_array"),
                                     options=options)
    with pytest.raises(CompilationError, match="require scoped memory"):
        generate_sources(function, plan, offload_config=OffloadConfig("sections", scope_transfers="pinned"))


def test_invalid_scope_transfer_combination_preserves_public_outputs(tmp_path):
    output = tmp_path / "generated"
    output.mkdir()
    expected = {name: f"original {name}" for name in OUTPUT_FILES}
    for name, text in expected.items():
        (output / name).write_text(text)
    result = subprocess.run([
        sys.executable, "-m", "compiler", "--input",
        str(Path(__file__).parent / "fixtures" / "fill_array.f90"), "--kernel", "fill_array",
        "--output-dir", str(output), "--gpu-policy", "sections", "--scope-transfers", "pinned", "--json",
    ], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    import json

    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert not report["supported"]
    assert report["outputs"] == []
    assert "require scoped memory" in report["reason"]
    assert {path.name: path.read_text() for path in output.iterdir()} == expected


@pytest.mark.parametrize("existing_outputs", [False, True], ids=["new-directory", "existing-outputs"])
@pytest.mark.parametrize("opt_level", ["-1", "2", "x"])
def test_invalid_optimization_options_preserve_outputs(
    tmp_path: Path, opt_level: str, *, existing_outputs: bool
) -> None:
    output = tmp_path / "generated"
    expected = {name: f"original {name}" for name in OUTPUT_FILES} if existing_outputs else {}
    if existing_outputs:
        output.mkdir()
        for name, contents in expected.items():
            (output / name).write_text(contents)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(Path(__file__).parent / "fixtures" / "fill_array.f90"),
            "--kernel",
            "fill_array",
            "--output-dir",
            str(output),
            f"--opt-level={opt_level}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "opt-level" in result.stderr.lower()
    actual = {path.name: path.read_text() for path in output.iterdir()} if output.exists() else {}
    assert actual == expected


@pytest.mark.parametrize("existing_outputs", [False, True], ids=["new-directory", "existing-outputs"])
@pytest.mark.parametrize("tiles", ["0", "-2", "1,,2", "x", "1,2,3,4", str(2**64)])
def test_invalid_tile_options_preserve_outputs(tmp_path: Path, tiles: str, *, existing_outputs: bool) -> None:
    output = tmp_path / "generated"
    expected = {name: f"original {name}" for name in OUTPUT_FILES} if existing_outputs else {}
    if existing_outputs:
        output.mkdir()
        for name, contents in expected.items():
            (output / name).write_text(contents)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(Path(__file__).parent / "fixtures" / "fill_array.f90"),
            "--kernel",
            "fill_array",
            "--output-dir",
            str(output),
            f"--tile-sizes={tiles}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "tile" in result.stderr.lower()
    actual = {path.name: path.read_text() for path in output.iterdir()} if output.exists() else {}
    assert actual == expected


def _evaluate(expression, scalars, arrays, extents):
    if isinstance(expression, Literal):
        return int(expression.value) if expression.dtype.value == "integer" else float(expression.value)
    if isinstance(expression, Reference):
        return scalars[expression.symbol.name]
    if isinstance(expression, Size):
        return extents[expression.symbol.name][expression.dimension - 1]
    if isinstance(expression, ArrayAccess):
        coordinates = tuple(_evaluate(index, scalars, arrays, extents) for index in expression.indices)
        return arrays[expression.symbol.name][coordinates]
    if isinstance(expression, Unary):
        value = _evaluate(expression.operand, scalars, arrays, extents)
        return value if expression.operator == "+" else -value
    if isinstance(expression, Binary):
        left = _evaluate(expression.left, scalars, arrays, extents)
        right = _evaluate(expression.right, scalars, arrays, extents)
        return {"+": lambda: left + right, "-": lambda: left - right, "*": lambda: left * right}[expression.operator]()
    raise AssertionError(type(expression))


def _interpret(block, scalars, arrays, extents):
    for statement in block.statements:
        if isinstance(statement, Assignment):
            value = _evaluate(statement.value, scalars, arrays, extents)
            if isinstance(statement.target, Reference):
                scalars[statement.target.symbol.name] = value
            else:
                coordinates = tuple(_evaluate(index, scalars, arrays, extents) for index in statement.target.indices)
                arrays[statement.target.symbol.name][coordinates] = value
        elif isinstance(statement, Loop):
            lower = _evaluate(statement.lower, scalars, arrays, extents)
            upper = _evaluate(statement.upper, scalars, arrays, extents)
            stride = (
                statement.step
                if isinstance(statement.step, int)
                else _evaluate(statement.step, scalars, arrays, extents)
            )
            for index in range(lower, upper + (1 if stride > 0 else -1), stride):
                scalars[statement.iterator.name] = index
                _interpret(statement.body, scalars, arrays, extents)
        else:
            raise AssertionError(type(statement))


def _ordered_points(region, scalars, arrays, extents):
    schedule = region.schedule
    ranges = []
    for loop in region.loops:
        lower = _evaluate(loop.lower, scalars, arrays, extents)
        upper = _evaluate(loop.upper, scalars, arrays, extents)
        stride = loop.step if isinstance(loop.step, int) else _evaluate(loop.step, scalars, arrays, extents)
        ranges.append(tuple(range(lower, upper + (1 if stride > 0 else -1), stride)))
    coordinates = list(itertools.product(*(range(len(axis)) for axis in ranges)))
    if schedule.tile_sizes:
        coordinates.sort(
            key=lambda point: (
                tuple(point[axis] // schedule.tile_sizes[axis] for axis in reversed(schedule.axis_order)),
                tuple(point[axis] % schedule.tile_sizes[axis] for axis in reversed(schedule.axis_order)),
            )
        )
    else:
        coordinates.sort(key=lambda point: tuple(point[axis] for axis in reversed(schedule.axis_order)))
    return [tuple(ranges[axis][coordinate] for axis, coordinate in enumerate(point)) for point in coordinates]


@pytest.mark.parametrize("stride", [1, -1, 2, -2])
@pytest.mark.parametrize("tiles", [(), (2, 3)])
def test_fusion_and_schedule_match_brute_force_source_execution(
    tmp_path: Path, stride: int, tiles: tuple[int, ...]
) -> None:
    source = tmp_path / "small_domain.f90"
    lower, upper = ("1", "n") if stride > 0 else ("n", "1")
    source.write_text(f"""! kernels
module small_domain
contains
  ! kernel
  subroutine entry(a,b,n)
    integer, intent(inout) :: a(:,:), b(:,:)
    integer, intent(in) :: n
    integer :: i,j,scale
    do i={lower},{upper},{stride}
      do j=1,n
        a(i,j)=a(i,j)+i-j
      end do
    end do
    scale=3
    do i={lower},{upper},{stride}
      do j=1,n
        b(i,j)=a(i,j)*scale+b(i,j)
      end do
    end do
  end subroutine
end module
""")
    function = lower_file(source, "entry")
    optimized, plan = prepare_function(function, options=CompilerOptions(tile_sizes=tiles))
    assert len(plan.regions) == 1
    for n in range(5):
        shape = {"a": (n, n), "b": (n, n)}
        initial = {
            name: {(i, j): seed + 10 * i + j for i in range(1, n + 1) for j in range(1, n + 1)}
            for name, seed in (("a", 7), ("b", -3))
        }
        expected = {name: dict(values) for name, values in initial.items()}
        _interpret(function.body, {"n": n}, expected, shape)
        for reverse in (False, True):
            actual = {name: dict(values) for name, values in initial.items()}
            scalars = {"n": n}
            for statement in optimized.body.statements:
                if isinstance(statement, Assignment):
                    scalars[statement.target.symbol.name] = _evaluate(statement.value, scalars, actual, shape)
            region = plan.regions[0]
            points = _ordered_points(region, scalars, actual, shape)
            if reverse:
                points.reverse()
            for coordinates in points:
                for loop, coordinate in zip(region.loops, coordinates, strict=True):
                    scalars[loop.iterator.name] = coordinate
                _interpret(region.body, scalars, actual, shape)
            assert actual == expected
