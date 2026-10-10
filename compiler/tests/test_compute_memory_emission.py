"""Bounded emitted memory costs use unique working sets and repeated traffic."""

from copy import deepcopy
import math
import shutil
import subprocess

import pytest

from compiler.emission.cuda.compute_memory import COMPUTE_MEMORY_INCLUDES, generate_compute_memory
from compiler.offload.numerical_calibration import memory_compute_seconds


def model(coordinates=(100, 300, 1000), rates=(0.25, 0.75, 0.5)):
    return {"kind": "piecewise_bandwidth_v1", "working_set_range": [coordinates[0], coordinates[-1]],
            "knots": [{"working_set_bytes": size, "seconds_per_traffic_byte": rate}
                      for size, rate in zip(coordinates, rates, strict=True)]}


def run_cpp(tmp_path, helpers, body):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = "\n".join((*COMPUTE_MEMORY_INCLUDES, "#include <cassert>",
                         *helpers, "int main() {", body, "}"))
    (tmp_path / "check.cpp").write_text(source)
    built = subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            "check.cpp", "-o", "check"], cwd=tmp_path, text=True,
                           capture_output=True, timeout=30)
    assert built.returncode == 0, built.stderr
    executed = subprocess.run([str(tmp_path / "check")], text=True, capture_output=True, timeout=10)
    assert executed.returncode == 0, executed.stderr


def test_endpoints_interpolation_and_traffic_are_separate_from_working_set(tmp_path):
    run_cpp(tmp_path, generate_compute_memory("estimate", model()), """
double seconds = -1;
assert(estimate(200,100,seconds) && seconds == 50);
assert(estimate(200,200,seconds) && seconds == 100);
assert(estimate(200,300,seconds) && seconds == 150);
assert(estimate(200,650,seconds) && seconds == 125);
assert(estimate(200,1000,seconds) && seconds == 100);
assert(estimate(600,300,seconds) && seconds == 450);
assert(estimate(0,300,seconds) && seconds == 0);
""")


def test_outside_range_and_invalid_runtime_inputs_are_unavailable(tmp_path):
    run_cpp(tmp_path, generate_compute_memory("estimate", model()), """
double seconds = 7;
assert(!estimate(200,99,seconds) && seconds == 0);
seconds = 7; assert(!estimate(200,1001,seconds) && seconds == 0);
seconds = 7; assert(!estimate(0,0,seconds) && seconds == 0);
seconds = 7; assert(!estimate(-1,300,seconds) && seconds == 0);
seconds = 7; assert(!estimate(std::numeric_limits<double>::infinity(),300,seconds) && seconds == 0);
seconds = 7; assert(!estimate(std::numeric_limits<double>::quiet_NaN(),300,seconds) && seconds == 0);
""")


def test_data_dependent_overflow_underflow_and_large_coordinate_differences(tmp_path):
    high = (1 << 64) - 1
    helpers = (*generate_compute_memory("large_rate", model((10, 20), (2.0, 4.0))),
               *generate_compute_memory("small_rate", model((10, 20), (1e-300, 1e-300))),
               *generate_compute_memory("large_coordinates", model((high - 100, high), (0.25, 0.75))))
    run_cpp(tmp_path, helpers, f"""
double seconds = 9;
assert(!large_rate(std::numeric_limits<double>::max(),15,seconds) && seconds == 0);
assert(large_rate(100,15,seconds) && seconds == 300);
seconds = 9; assert(!small_rate(std::numeric_limits<double>::denorm_min(),15,seconds) && seconds == 0);
if (std::numeric_limits<std::size_t>::max() == {high}ULL) {{
    assert(large_coordinates(200,{high - 50}ULL,seconds) && seconds == 100);
    assert(large_coordinates(200,{high}ULL,seconds) && seconds == 150);
}}
""")


def test_malformed_models_emit_false_before_any_estimate(tmp_path):
    malformed = [None, [], {}, model((10,), (1,)), model(tuple(range(1, 10)), (1,) * 9),
                 model((10, 10), (1, 2)), model((20, 10), (1, 2)), model((0, 10), (1, 2)),
                 model((10, 1 << 64), (1, 2)), model((10, 20), (True, 1)),
                 model((10, 20), (math.inf, 1)), model((10, 20), (math.nan, 1)),
                 model((10, 20), (0, 1)), model((10, 20), (-1, 1)),
                 {"kind": "constant_bandwidth_v1", "seconds_per_traffic_byte": 0.5}]
    wrong_bounds = deepcopy(model())
    wrong_bounds["working_set_range"] = [100, 999]
    malformed.append(wrong_bounds)
    boolean_coordinate = deepcopy(model())
    boolean_coordinate["knots"][0]["working_set_bytes"] = True
    malformed.append(boolean_coordinate)
    bad_row = deepcopy(model())
    bad_row["knots"][1] = None
    malformed.append(bad_row)
    helpers, assertions = [], []
    for index, value in enumerate(malformed):
        helpers.extend(generate_compute_memory("invalid_" + str(index), value))
        assertions.append(f"seconds = 9; assert(!invalid_{index}(200,300,seconds) && seconds == 0);")
    run_cpp(tmp_path, helpers, "double seconds = 9;\n" + "\n".join(assertions))


@pytest.mark.parametrize("count", [2, 3, 8])
def test_bounded_model_matches_python_reader_formula(tmp_path, count):
    value = model(tuple(100 * (i + 1) for i in range(count)),
                  tuple(1e-10 * (i % 3 + 1) for i in range(count)))
    assertions = []
    for working_set in (100, 125, *(100 * i for i in range(2, count + 1))):
        for traffic in (0, 120, 10000):
            expected = memory_compute_seconds(value, traffic, working_set)
            assertions.append(f"assert(estimate({traffic},{working_set},seconds)); "
                              f"assert(std::abs(seconds-{expected.hex()}) <= 1e-14 * "
                              f"std::max(std::abs(seconds),std::abs({expected.hex()}))); ")
    run_cpp(tmp_path, ("#include <algorithm>", *generate_compute_memory("estimate", value)),
            "double seconds = 0;\n" + "\n".join(assertions))


@pytest.mark.parametrize("name", [None, "", "bad-name", "a::b", "name()", "3name"])
def test_helper_name_is_not_interpreted_as_source(name):
    with pytest.raises(ValueError, match="identifier"):
        generate_compute_memory(name, model())
