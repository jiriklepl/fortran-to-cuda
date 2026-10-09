"""Definition preflight rejects bounded metadata independently of placement."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"

DRIVER = r'''
#include "scoped_planning.hpp"
#include <iostream>
using namespace fort_scoped::planning;
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::string scenario = argv[1];
    Resource b; b.handle = 1; b.element_bytes = 8; b.bytes = 8*65; b.extents = {65};
    Region full{Box{{0}, {65}}}; b.initialized = full;
    Inputs in; in.resources = {b};
    Operation read{FORT_SCOPE_PLAN_WORKER, 1, {{1, {full, {}, {}}}}, 0, 0, false};
    in.operations = {read};
    if (scenario == "copy_state_irrelevant") {
        in.resources[0].host_current = {Box{{99}, {100}}};
        in.resources[0].device_current = full;
        in.device_budget = 0; in.pending = true;
        in.operations[0].flops = std::numeric_limits<double>::quiet_NaN();
        in.operations[0].memory_bytes = std::numeric_limits<double>::infinity();
    } else if (scenario == "unknown_kind") {
        in.operations[0].kind = std::numeric_limits<uint32_t>::max();
    } else if (scenario == "unknown_handle") {
        in.operations[0].bindings[0].buffer = 2;
    } else if (scenario == "duplicate_handle") {
        in.resources.push_back(b);
    } else if (scenario == "duplicate_binding") {
        in.operations[0].bindings.push_back(in.operations[0].bindings[0]);
    } else if (scenario == "unknown_unit") {
        in.operations[0].unit = 0;
    } else if (scenario == "outside_layout") {
        in.operations[0].bindings[0].effects.reads = {Box{{0}, {66}}};
    } else if (scenario == "wrong_rank") {
        in.operations[0].bindings[0].effects.reads = {Box{{0, 0}, {1, 1}}};
    } else if (scenario == "unknown_layout") {
        in.resources[0].extents.clear();
    } else if (scenario == "overflow_layout") {
        in.resources[0].extents = {std::numeric_limits<size_t>::max(), 2};
    } else if (scenario == "empty_layout") {
        in.resources[0].extents = {0, std::numeric_limits<size_t>::max()};
        in.resources[0].bytes = 0; in.resources[0].initialized.clear();
        in.operations[0].bindings[0].effects.reads.clear();
    } else if (scenario == "bad_overwrite") {
        in.operations[0].bindings[0].effects.overwrites = full;
    } else if (scenario == "resource_budget") {
        in.resources.assign(257, b);
    } else if (scenario == "operation_budget") {
        in.operations.assign(257, Operation{});
    } else if (scenario == "worker_budget") {
        in.operations.assign(65, read);
    } else if (scenario == "region_budget") {
        in.resources[0].initialized.assign(33, Box{{0}, {1}});
    } else if (scenario == "fragmented") {
        in.resources[0].initialized.clear();
        for (size_t i = 0; i < 32; ++i)
            in.resources[0].initialized.push_back(Box{{2*i+1}, {2*i+2}});
    } else if (scenario == "undefined_read") {
        in.resources[0].initialized.clear();
    } else if (scenario != "basic") return 3;
    auto result = validate_definitions(in);
    std::cout << "{\"status\":" << result.status << ",\"reason\":\"" << result.reason
              << "\",\"operation\":" << result.operation << ",\"buffer\":" << result.buffer << "}\n";
}
'''


@pytest.fixture(scope="module")
def definitions(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    directory = tmp_path_factory.mktemp("scoped_definitions")
    source, executable = directory / "driver.cpp", directory / "driver"
    source.write_text(DRIVER)
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(RUNTIME),
         str(source), "-o", str(executable)], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr

    def run(scenario):
        completed = subprocess.run([str(executable), scenario], capture_output=True, text=True, timeout=10)
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)

    return run


@pytest.mark.parametrize("scenario", ["basic", "copy_state_irrelevant", "empty_layout"])
def test_definitions_need_no_copy_locations_costs_or_allocation_budget(definitions, scenario):
    result = definitions(scenario)
    assert result["status"] == 0
    assert result["reason"] == "definition_plan_valid"
    assert result["buffer"] == 0


@pytest.mark.parametrize(("scenario", "reason"), [
    ("unknown_kind", "unknown_planning_operation"),
    ("unknown_handle", "unknown_resource_handle"),
    ("duplicate_handle", "duplicate_resource_handle"),
    ("duplicate_binding", "duplicate_operation_binding"),
    ("unknown_unit", "unknown_worker_work"),
    ("outside_layout", "invalid_resource_region"),
    ("wrong_rank", "invalid_resource_region"),
    ("unknown_layout", "invalid_resource_layout"),
    ("overflow_layout", "arithmetic_overflow"),
    ("bad_overwrite", "overwrite_exceeds_write_region"),
    ("resource_budget", "planning_record_budget_exceeded"),
    ("operation_budget", "planning_record_budget_exceeded"),
    ("worker_budget", "planning_worker_budget_exceeded"),
    ("region_budget", "region_budget_exceeded"),
    ("fragmented", "region_fragmentation_unavailable"),
])
def test_definition_metadata_fails_with_an_explained_boundary(definitions, scenario, reason):
    result = definitions(scenario)
    assert result["status"] == 5
    assert result["reason"] == reason


def test_undefined_requirement_identifies_the_operation_and_buffer(definitions):
    result = definitions("undefined_read")
    assert result == {"status": 7, "reason": "uninitialized_read", "operation": 0, "buffer": 1}
