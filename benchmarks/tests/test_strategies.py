"""Validate the comparison's data checks and independently measured evidence."""

import json
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from benchmarks.harness.strategies import (
    Strategies,
    _overlap_seconds,
    compare_dumps,
    comparison_gates,
    driver_source,
    parse_profile,
    parse_timing,
    parse_trace,
    parser,
    read_dump,
)


def write_dump(path, source, destination):
    with path.open("wb") as output:
        output.write(b"STRAT001" + struct.pack("=3q", *source.shape))
        output.write(source.ravel(order="F").astype(np.float64).tobytes())
        output.write(destination.ravel(order="F").astype(np.float64).tobytes())


def test_timing_parser_reports_complete_call_time_and_rejects_duplicates():
    result = parse_timing("call_seconds 1.25D-02\nchecksum -4.0\ncalls 5\n")
    assert result["seconds_per_call"] == 0.0025
    with pytest.raises(ValueError, match="duplicate"):
        parse_timing("call_seconds 1\nchecksum 0\ncalls 1\ncalls 2\n")
    with pytest.raises(ValueError, match="invalid"):
        parse_timing("call_seconds nan\nchecksum 0\ncalls 1\n")


def test_decision_units_are_not_combined_across_different_meanings():
    trace = parse_trace(
        "FORT_OFFLOAD entry=a mode=hybrid gpu_units=12 cpu_units=4 unit_kind=slabs\n"
        "FORT_OFFLOAD entry=b mode=auto gpu_units=2 cpu_units=1 unit_kind=regions\n"
        "FORT_OFFLOAD_INTERVAL entry=b begin=1 end=3 mode=gpu work=4000 launches=2 upload_bytes=1024 download_bytes=128 estimate_seconds=0.02 volume_valid=1\n"
        "FORT_RUNTIME upload bytes=1024\nFORT_RUNTIME kernel\nFORT_RUNTIME download bytes=128\n"
    )
    assert trace["unit_totals_by_kind"] == {
        "slabs": {"gpu_units": 12, "cpu_units": 4},
        "regions": {"gpu_units": 2, "cpu_units": 1},
    }
    assert trace["runtime"]["upload_bytes"] == 1024
    assert trace["runtime"]["kernel_count"] == 1
    assert trace["runtime"]["download_bytes"] == 128
    assert trace["intervals"][0]["begin"] == 1
    assert trace["intervals"][0]["end"] == 3
    assert trace["intervals"][0]["estimate_seconds"] == 0.02


def test_native_only_decision_is_valid_evidence_without_fake_gpu_work():
    trace = parse_trace(
        "FORT_OFFLOAD entry=filter mode=native cpu_units=3 gpu_units=0 unit_kind=regions"
    )
    assert trace["runtime"] == {}
    assert trace["unit_totals_by_kind"]["regions"]["gpu_units"] == 0


def test_dump_comparison_checks_halos_and_readonly_inputs(tmp_path):
    source = np.arange(60, dtype=float).reshape((3, 4, 5))
    destination = source + 0.125
    reference, actual = tmp_path / "reference.bin", tmp_path / "actual.bin"
    write_dump(reference, source, destination)
    write_dump(actual, source, destination)
    result = compare_dumps(reference, actual)
    assert result["passed"]
    assert result["fields"]["destination"]["values"] == 60
    changed = destination.copy()
    changed[0, 1, 2] += 1
    write_dump(actual, source, changed)
    with pytest.raises(ValueError, match="destination full-array"):
        compare_dumps(reference, actual)
    changed = source.copy()
    changed[1, 2, 3] += 1
    write_dump(actual, changed, destination)
    with pytest.raises(ValueError, match="source full-array"):
        compare_dumps(reference, actual)


def test_dump_rejects_truncation_and_nonfinite_values(tmp_path):
    source = np.ones((3, 3, 3))
    reference, actual = tmp_path / "reference.bin", tmp_path / "actual.bin"
    write_dump(reference, source, source)
    write_dump(actual, source, source)
    actual.write_bytes(actual.read_bytes()[:-8])
    with pytest.raises(ValueError, match="length"):
        read_dump(actual)
    source[1, 1, 1] = float("nan")
    write_dump(actual, source, source)
    with pytest.raises(ValueError, match="non-finite"):
        compare_dumps(reference, actual)


def test_overlap_uses_union_not_double_counted_range_pairs():
    assert _overlap_seconds([(0, 20), (0, 20)], [(5, 15), (10, 25)]) == 15 / 1e9


def test_profile_uses_actual_kernels_copies_and_cpu_compute_ranges(tmp_path):
    path = tmp_path / "activity.sqlite"
    with sqlite3.connect(path) as database:
        database.execute(
            'CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, "end" INTEGER, streamId INTEGER)'
        )
        database.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?)",
            [(10, 30, 7), (40, 60, 8)],
        )
        database.execute(
            'CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER,"end" INTEGER,streamId INTEGER,copyKind INTEGER,bytes INTEGER)'
        )
        database.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?)",
            [(5, 20, 8, 1, 80), (45, 55, 8, 2, 40)],
        )
        database.execute(
            'CREATE TABLE NVTX_EVENTS (start INTEGER,"end" INTEGER,text TEXT,textId INTEGER)'
        )
        database.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)",
            [
                (0, 50, "FORT hybrid CPU window", None),
                (0, 50, "FORT hybrid CPU window", None),
                (0, 100, "FORT hybrid GPU pump", None),
                (0, 5, "FORT hybrid pack", None),
                (50, 55, "FORT hybrid unpack", None),
                (0, 2, "FORT offload decision", None),
            ],
        )
    result = parse_profile(path)
    assert result["kernel_count"] == 2
    assert result["h2d_bytes"] == 80
    assert result["d2h_bytes"] == 40
    assert result["cpu_gpu_overlap_seconds"] == 30 / 1e9
    # Same-stream kernel/copy intersections are explicitly excluded.
    assert result["kernel_copy_overlap_seconds"] == 10 / 1e9
    assert result["cpu_copy_overlap_seconds"] == 20 / 1e9
    assert result["host_ranges"]["FORT hybrid pack"]["seconds_sum"] == 5 / 1e9
    assert result["host_ranges"]["FORT offload decision"]["seconds_sum"] == 2 / 1e9


def test_profile_resolves_registered_nvtx_strings_and_cpu_only_exports(tmp_path):
    path = tmp_path / "activity.sqlite"
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE StringIds (id INTEGER,value TEXT)")
        database.execute("INSERT INTO StringIds VALUES (7,'FORT hybrid CPU window')")
        database.execute(
            'CREATE TABLE NVTX_EVENTS (start INTEGER,"end" INTEGER,textId INTEGER)'
        )
        database.execute("INSERT INTO NVTX_EVENTS VALUES (0,10,7)")
    result = parse_profile(path)
    assert result["kernel_count"] == 0
    assert result["cpu_compute_ranges"] == 1
    assert result["cpu_gpu_overlap_seconds"] == 0


def test_driver_preserves_original_storage_and_uses_serial_caller():
    driver = driver_source("generated_module", "ripple_filter")
    assert "source(0:nx+1,-1:ny,2:nz+3)" in driver
    assert "source=source+0.03125d0" in driver
    assert "write(unit) source,destination" in driver
    assert "!$omp parallel" not in driver
    assert "call kernel(source,destination,nx,ny,nz)" in driver


def test_driver_honors_compiler_query_before_calling_original_native_fallback():
    driver = driver_source("ripple_fields", "ripple_filter", query="ripple_decision", native_module="ripple_native")
    assert "select_gpu => ripple_decision" in driver
    assert "use ripple_native, only: native_kernel => ripple_filter" in driver
    assert driver.count("if (select_gpu(source,destination,nx,ny,nz)) then") == 2
    assert driver.count("call native_kernel(source,destination,nx,ny,nz)") == 2


def test_public_cli_build_contract_and_fixed_budget(tmp_path):
    args = parser().parse_args(
        [
            "--calibration-profile",
            str(tmp_path / "offline.json"),
            "--output",
            str(tmp_path),
            "--build-only",
            "--architecture", "80",
        ]
    )
    comparison = Strategies.__new__(Strategies)
    comparison.args = args
    comparison.output = tmp_path
    comparison.profile = args.calibration_profile
    comparison.binaries = {}
    comparison.report = {"generation": []}
    commands = []

    def run(command, cwd, **kwargs):
        commands.append([str(value) for value in command])
        if "--gpu-policy" in command:
            return json.dumps({"supported": True, "outputs": []}), "", 0.0
        return "", "", 0.0

    comparison.run = run
    comparison.build("pointwise", "hybrid")
    assert any("-arch=sm_80" in command for command in commands)
    generate = commands[0]
    assert generate[generate.index("--gpu-policy") + 1] == "hybrid"
    assert generate[generate.index("--host-threads") + 1] == "4"
    assert "--gpu-collective" not in generate
    assert "--calibration-profile" in generate
    assert any("-Xcompiler=-fopenmp" in command for command in commands)
    assert all("--use_fast_math" not in command for command in commands)
    assert "use tidal_fields, only:" in (tmp_path / "pointwise/build/hybrid/driver.f90").read_text()


def test_defaults_are_one_warmup_three_samples_and_all_six_modes():
    args = parser().parse_args(["--calibration-profile", "offline.json"])
    assert args.rounds == 3
    assert args.warmup_rounds == 1
    assert len(args.modes) == 6
    assert args.host_threads == 4
    assert args.calibration_profile == Path("offline.json")


def test_gates_use_complete_wall_time_and_require_actual_hybrid_overlap():
    timings = [
        {
            "case": "compute", "grid": [32, 24, 16], "mode": mode,
            "process_seconds_median": seconds, "seconds_per_call_median": 0.01,
        }
        for mode, seconds in (("native", 10), ("sections", 8), ("auto", 11), ("hybrid", 7))
    ]
    profile = {
        "case": "compute", "grid": [32, 24, 16], "mode": "hybrid",
        "cpu_gpu_overlap_seconds": 0.1, "kernel_copy_overlap_seconds": 0.1,
        "trace": {"decisions": [{"cpu_units": 1, "gpu_units": 3}]},
    }
    gates = comparison_gates(timings, [profile])[0]
    assert not gates["auto"]["within_5_percent"]
    assert gates["hybrid"]["qualifies"]
    faster_current = {**timings[0], "mode": "current", "process_seconds_median": 6}
    current_gate = comparison_gates([*timings, faster_current], [profile])[0]["hybrid"]
    assert current_gate["best_native_or_synchronous_mode"] == "current"
    assert not current_gate["at_least_10_percent_faster"]
    assert not comparison_gates(timings, [])[0]["hybrid"]["qualifies"]
    profile["trace"]["decisions"][0]["cpu_units"] = 0
    assert not comparison_gates(timings, [profile])[0]["hybrid"]["qualifies"]


def test_borderline_extension_adds_samples_for_both_compared_modes():
    comparison = Strategies.__new__(Strategies)
    comparison.args = parser().parse_args(["--calibration-profile", "offline.json"])
    comparison.report = {"timings": [
        {"case": "compute", "grid": [1, 2, 3], "mode": mode, "process_seconds_median": seconds}
        for mode, seconds in (("native", 10), ("auto", 10.6), ("sections", 8), ("hybrid", 7.3))
    ]}
    calls = []
    comparison.time_mode = lambda case, mode, shape, **kw: calls.append((mode, kw["rounds"]))
    comparison.extend_borderline("compute", (1, 2, 3))
    assert calls == [(mode, 7) for mode in ("native", "current", "sections", "auto", "hybrid")]

@pytest.mark.parametrize('capability,target', [('8.6','86'), ('9.0','90'), ('10.0','100')])
def test_target_defaults_to_selected_calibration_not_this_machine(tmp_path, capability, target):
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'precision_bits':64, 'cpu_threads':4,
                                   'hardware':{'compute_capability':capability}}))
    args = parser().parse_args(['--calibration-profile',str(profile),'--compiler-root',str(tmp_path),
                               '--output',str(tmp_path / 'output')])
    comparison = Strategies(args)
    assert args.architecture == target
    assert comparison.report['configuration']['architecture'] == target
