"""Exact bounded rectangle normalization preserves holes and physical cells."""
from __future__ import annotations

import itertools
import json
import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"

DRIVER = r'''
#include "scoped_regions.hpp"
#include <iostream>
using namespace fort_scoped::coherence;
Region read_region(size_t rank) {
    size_t count; std::cin >> count;
    Region result;
    for (size_t i=0; i<count; ++i) {
        Box box; box.lo.resize(rank); box.hi.resize(rank);
        for (auto &value : box.lo) std::cin >> value;
        for (auto &value : box.hi) std::cin >> value;
        result.push_back(std::move(box));
    }
    return result;
}
int main() {
    size_t operation, rank, checks;
    std::cin >> operation >> rank >> checks;
    auto first = read_region(rank), second = read_region(rank);
    Budget budget; budget.checks = checks;
    try {
        auto result = operation == 0 ? unite(std::move(first), second, budget)
                                    : difference(first, second, budget);
        auto repeated = result;
        coalesce(repeated);
        if (repeated.size() != result.size()) return 2;
        for (size_t i=0; i<result.size(); ++i)
            if (repeated[i].lo != result[i].lo || repeated[i].hi != result[i].hi) return 3;
        std::cout << "{\"fragmented\":false,\"checks\":" << budget.checks << ",\"boxes\":[";
        for (size_t i=0; i<result.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '[';
            for (size_t k=0; k<rank; ++k) { if (k) std::cout << ','; std::cout << result[i].lo[k]; }
            std::cout << ",";
            for (size_t k=0; k<rank; ++k) { if (k) std::cout << ','; std::cout << result[i].hi[k]; }
            std::cout << ']';
        }
        std::cout << "]}\n";
    } catch (const Fragmented &) {
        std::cout << "{\"fragmented\":true,\"checks\":" << budget.checks << "}\n";
    }
}
'''


@pytest.fixture(scope="module")
def regions(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    directory = tmp_path_factory.mktemp("scoped_regions")
    source, executable = directory / "driver.cpp", directory / "driver"
    source.write_text(DRIVER)
    completed = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(RUNTIME),
         str(source), "-o", str(executable)], capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr

    def run(first, second=(), *, difference=False, checks=0):
        rank = len((first or second)[0][0])
        values = [int(difference), rank, checks]
        for boxes in (first, second):
            values.append(len(boxes))
            for lo, hi in boxes:
                values.extend([*lo, *hi])
        completed = subprocess.run([str(executable)], input=" ".join(map(str, values)),
                                   capture_output=True, text=True, timeout=10)
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        if not result["fragmented"]:
            result["boxes"] = [(tuple(box[:rank]), tuple(box[rank:])) for box in result["boxes"]]
        return result

    return run


def cells(boxes):
    return {point for lo, hi in boxes for point in itertools.product(
        *(range(start, stop) for start, stop in zip(lo, hi, strict=True)))}


def assert_exact_disjoint(result, expected):
    assert not result["fragmented"]
    assert cells(result["boxes"]) == expected
    assert sum(len(cells([box])) for box in result["boxes"]) == len(expected)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_touching_faces_merge_on_each_axis(regions, axis):
    lower, middle, upper = [1, 1, 1], [4, 4, 4], [4, 4, 4]
    middle[axis], upper[axis] = 2, 5
    next_lower = lower.copy()
    next_lower[axis] = middle[axis]
    boxes = [(tuple(lower), tuple(middle)), (tuple(next_lower), tuple(upper))]
    result = regions([], boxes)
    assert_exact_disjoint(result, cells(boxes))
    assert result["boxes"] == [(tuple(lower), tuple(upper))]


@pytest.mark.parametrize("boxes", [
    [((0, 0), (2, 2)), ((3, 0), (5, 2))],  # A one-cell gap.
    [((0, 0), (2, 2)), ((2, 2), (4, 4))],  # A corner touches.
    [((0, 0), (2, 2)), ((2, 0), (4, 1))],  # Only part of a face touches.
])
def test_nonrectangular_neighbours_do_not_fill_missing_cells(regions, boxes):
    result = regions([], boxes)
    assert_exact_disjoint(result, cells(boxes))
    assert len(result["boxes"]) == 2


def test_overlapping_union_preserves_its_holes(regions):
    boxes = [((0, 0), (3, 2)), ((2, 1), (4, 3))]
    result = regions([], boxes)
    assert_exact_disjoint(result, cells(boxes))
    assert (3, 0) not in cells(result["boxes"])
    assert (0, 2) not in cells(result["boxes"])
    assert regions([], boxes) == result


def test_difference_preserves_an_interior_hole_and_coalesces_touching_pieces(regions):
    full, cut = [((0, 0), (4, 4))], [((1, 1), (3, 3))]
    result = regions(full, cut, difference=True)
    assert_exact_disjoint(result, cells(full) - cells(cut))
    assert len(result["boxes"]) == 4
    pieces = [((0, 0), (2, 2)), ((2, 0), (4, 2))]
    result = regions(pieces, difference=True)
    assert result["boxes"] == [((0, 0), (4, 2))]


def test_stencil_union_coalesces_ten_shifted_footprints_to_three(regions):
    shape = (5, 4, 3)
    shifts = [(0, 0, 0), (-1, 0, 0), (0, 1, 0), (-1, 1, 0), (0, -1, 0),
              (-1, -1, 0), (0, 0, 1), (-1, 0, 1), (0, 0, -1), (-1, 0, -1)]
    boxes = [(tuple(1 + offset for offset in shift),
              tuple(extent + 1 + offset for extent, offset in zip(shape, shift, strict=True)))
             for shift in shifts]
    result = regions([], boxes)
    assert_exact_disjoint(result, cells(boxes))
    assert len(result["boxes"]) == 3


def test_adjacent_extreme_endpoints_require_no_overflowing_arithmetic(regions):
    maximum = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1
    boxes = [((maximum - 4,), (maximum - 2,)), ((maximum - 2,), (maximum,))]
    assert regions([], boxes)["boxes"] == [((maximum - 4,), (maximum,))]


def test_rectangle_and_intersection_budgets_are_not_extended(regions):
    # Existing 33-box input fails before an otherwise possible normalization.
    boxes = [((index,), (index + 1,)) for index in range(32)]
    result = regions(boxes, [((32,), (33,))])
    assert result["fragmented"]
    result = regions([((0,), (2,))], [((1,), (2,))], difference=True, checks=1024)
    assert result == {"fragmented": True, "checks": 1025}
