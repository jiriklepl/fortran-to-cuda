"""Measured source-submit/scratch costs; no placement or numerical authority."""

from __future__ import annotations

import json
import math
import os
import statistics
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from compiler.emission.common.resources import read_scoped_runtime
from compiler.numerical_contract import cuda_compile_options, require_explicit_cuda_environment

from .tile_protocol_calibration import original_team_benchmark, source_scratch_benchmark

PROTOCOL_ID = 0x4654534F55520001
NATIVE_PROTOCOL_ID = 0x465453484F530001
NATIVE_RAW_NAMES = ("source_native_metadata_read_host", "source_native_metadata_readwrite_host",
                    "source_native_resident_read_host", "source_native_resident_readwrite_host")
SOURCE_COST_NAMES = ("source_submit_seconds", "scratch_cold_acquire_seconds", "scratch_reuse_acquire_seconds",
                     "scratch_idle_grow_seconds", "scratch_release_seconds")
RAW_NAMES = dict(zip(SOURCE_COST_NAMES, (
    "source_submit_host_enqueue_included", "scratch_cold_acquire_host", "scratch_reuse_acquire_host",
    "scratch_idle_grow_host", "scratch_release_host"), strict=True))
BINDINGS = (1, 2, 4, 6, 8)
LAUNCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FIT_BINDINGS, FIT_LAUNCHES = (1, 4, 8), (1, 4, 16, 64, 256)
TEAM_UNITS = (1, 2, 3, 4, 8)


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"source protocol {name} must be finite and positive")
    return float(value)


def _median(record):
    values = record.get("seconds")
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError("source protocol requires exactly five samples per row")
    return statistics.median(_positive(value, "sample") for value in values)


def _parse(text):
    try:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    except ValueError as error:
        raise ValueError("source protocol emitted non-JSON output") from error
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError("source protocol emitted no typed observations")
    return records


def _sizes(max_bytes):
    return sorted({size for size in (4096, 65536, 1048576, 4194304, 16777216, max_bytes) if size <= max_bytes})


def _schedule(record):
    """Retain raw API identity; compare only the explicitly known modifier bit.

    GNU Fortran's omp_get_schedule binding can expose the base kind while the
    C entry exposes omp_sched_monotonic. Dispatch must still match the full C
    identity; normalizing this observation does not erase that modifier guard.
    """
    kind, chunk = record.get("schedule_kind"), record.get("schedule_chunk")
    if (type(kind) is not int or not -(1 << 31) <= kind < 1 << 32
            or type(chunk) is not int or not 0 <= chunk < 1 << 31):
        raise ValueError("source protocol requires an exact OpenMP schedule identity")
    bits = kind & 0xFFFFFFFF
    base = bits & 0x7FFFFFFF
    if base not in (1, 2, 3, 4):
        raise ValueError("source protocol has an unknown OpenMP schedule kind or modifier")
    return {"raw_kind": kind, "base_kind": base, "monotonic_modifier": bool(bits & 0x80000000), "chunk": chunk}


def profile_with_source_measurements(profile, records, *, runtime_id, max_bytes, team_records=()):
    """Use a fixed fit grid and independent holdouts; never rerun until accepted.

    A holdout above the fixed maximum fit median retains all observations and
    marks this optional protocol unavailable. It does not expand the fit using
    the failed holdout or relax a runtime threshold.
    """
    from .profile import validate_profile

    validate_profile(profile, scoped_runtime_id=runtime_id)
    if type(max_bytes) is not int or not 8 * 1024**2 <= max_bytes <= 1024**3:
        raise ValueError("source protocol requires an explicit bounded byte range")
    identities = [row for row in records if row.get("kind") == "source_protocol_identity"]
    if len(identities) != 1:
        raise ValueError("source protocol requires one hardware identity")
    identity = identities[0]
    expected = {"version": 1, "gpu_uuid": profile["hardware"]["gpu_uuid"],
                "device": profile["hardware"]["device_ordinal"], "cpu_threads": profile["cpu_threads"],
                "precision": profile["precision_bits"],
                **{key: profile["toolchain"][key] for key in ("cuda_runtime_version", "driver_version")}}
    if any(identity.get(key) != value or type(identity.get(key)) is not type(value) for key, value in expected.items()):
        raise ValueError("source protocol hardware/toolchain/thread identity mismatch")
    schedule = _schedule(identity)
    rows, native_transactions = {}, {}
    sizes = _sizes(max_bytes)
    for row in records:
        if row is identity:
            continue
        if row.get("kind") == "source_protocol_observation" and row.get("name") in NATIVE_RAW_NAMES:
            if (any(type(row.get(key)) is not int for key in ("bytes", "bindings", "launches", "rounds"))
                    or row["bytes"] != 0 or row["bindings"] not in BINDINGS or row["launches"] != 0
                    or row["rounds"] != 128):
                raise ValueError("native source protocol has an out-of-grid measurement")
            key = row["name"], row["bindings"]
            if key in native_transactions:
                raise ValueError("native source protocol has a duplicate measurement")
            native_transactions[key] = (row, _median(row))
            continue
        if row.get("kind") != "source_protocol_observation" or row.get("name") not in RAW_NAMES.values():
            raise ValueError("source protocol has an unknown measurement")
        if any(type(row.get(key)) is not int for key in ("bytes", "bindings", "launches", "rounds")):
            raise ValueError("source protocol dimensions must be exact integers")
        name = row["name"]
        if name == RAW_NAMES["source_submit_seconds"]:
            valid = (row["bytes"] == 0 and row["bindings"] in BINDINGS
                     and row["launches"] in LAUNCHES and row["rounds"] == 32)
        else:
            valid = (row["bytes"] in (sizes[1:] if name == RAW_NAMES["scratch_idle_grow_seconds"] else sizes)
                     and row["bindings"] == row["launches"] == 0
                     and row["rounds"] == (128 if name in (RAW_NAMES["scratch_reuse_acquire_seconds"],
                                                            RAW_NAMES["scratch_release_seconds"]) else 8))
        if not valid:
            raise ValueError("source protocol has an out-of-grid measurement")
        key = (name, row["bytes"], row["bindings"], row["launches"])
        if key in rows:
            raise ValueError("source protocol has a duplicate measurement")
        rows[key] = (row, _median(row))
    expected_keys = {(RAW_NAMES["source_submit_seconds"], 0, bindings, launches)
                     for bindings in BINDINGS for launches in LAUNCHES}
    expected_keys |= {(raw, size, 0, 0) for name, raw in RAW_NAMES.items() if name != "source_submit_seconds"
                      for size in (sizes[1:] if name == "scratch_idle_grow_seconds" else sizes)}
    if set(rows) != expected_keys:
        raise ValueError("source protocol fixed measurement grid is incomplete")
    costs, fit = {}, {}
    for name, raw in RAW_NAMES.items():
        training = [(key, value) for key, value in rows.items() if key[0] == raw
                    and (name != "source_submit_seconds" or key[2] in FIT_BINDINGS and key[3] in FIT_LAUNCHES)]
        key, (_, median) = max(training, key=lambda item: item[1][1])
        costs[name] = median
        fit[name] = {"maximum_median_row": list(key), "fit_rows": len(training)}
    holdouts = [key for key, (_, median) in rows.items() if key[0] == RAW_NAMES["source_submit_seconds"]
                and not (key[2] in FIT_BINDINGS and key[3] in FIT_LAUNCHES)
                and median > costs["source_submit_seconds"]]
    native_rows = []
    seen = set()
    for row in team_records:
        if (row.get("kind") != "original_team_observation"
                or any(type(row.get(key)) is not int for key in (
                    "rank", "collapse", "units", "cpu_threads", "rounds", "schedule_kind", "schedule_chunk"))
                or not 1 <= row["collapse"] <= row["rank"] <= 3
                or row["units"] not in TEAM_UNITS or row["cpu_threads"] != profile["cpu_threads"]
                or row["rounds"] != 128):
            raise ValueError("original-team protocol identity or geometry mismatch")
        native_schedule = _schedule(row)
        if (native_schedule["base_kind"] != schedule["base_kind"] or native_schedule["chunk"] != schedule["chunk"]
                or native_schedule["monotonic_modifier"] and not schedule["monotonic_modifier"]):
            raise ValueError("original-team protocol actual schedule differs from the source protocol")
        key = (row["rank"], row["collapse"], row["units"])
        if key in seen:
            raise ValueError("original-team protocol has a duplicate measurement")
        seen.add(key)
        native_rows.append({"rank": row["rank"], "collapse": row["collapse"], "units": row["units"],
                            "seconds": _median(row), "schedule": native_schedule})
    if team_records and seen != {(rank, collapse, units) for rank in (1, 2, 3)
                                for collapse in range(1, rank + 1) for units in TEAM_UNITS}:
        raise ValueError("original-team protocol exact geometry grid is incomplete")
    result = deepcopy(profile)
    result["source_tasks"] = {
        "version": 1, "protocol_id": PROTOCOL_ID, "runtime_id": runtime_id,
        "precision_bits": profile["precision_bits"], "available": not holdouts,
        "max_bindings": max(BINDINGS), "max_launches": max(LAUNCHES), "max_scratch_bytes": max_bytes,
        "costs": costs, "fit": {"method": "maximum median on predetermined fit grid",
            "rows": fit, "failed_holdout_rows": [list(key) for key in holdouts],
            "coverage": "observed calibration grid only; no statistical confidence bound"},
        "identity": deepcopy(identity), "measurements": deepcopy(records),
        "original_teams": {"measurements": deepcopy(list(team_records)), "costs": native_rows,
            "schedule": {"c_identity": schedule,
                "comparison": "same base kind and chunk; only known monotonic modifier may be masked by Fortran API",
                "dispatch": "require the complete original C raw kind and chunk; no modifier erasure"},
            "scope": "exact rank/collapse/schedule/units; excludes body and PRIVATE allocation"},
        "cost_scope": "source submit includes per-binding access and ALL kernel enqueue; GPU compute excludes enqueue",
        "quote_endpoints": {
            "native": "publish and retire the complete context/domain, then original native team",
            "gpu": "local source transaction plus projected complete terminal retirement",
            "publication": "once per endpoint through the same live coherence simulator",
            "common_quote_overhead": "unmeasured; excluded from both alternatives",
            "whole_owner_estimate_available": False,
            "full_wall_prediction": False,
        },
        "application_profiled": False,
    }
    if native_transactions:
        if set(native_transactions) != {(name, count) for name in NATIVE_RAW_NAMES for count in BINDINGS}:
            raise ValueError("native source protocol fixed measurement grid is incomplete")
        training = [(key, value) for key, value in native_transactions.items() if key[1] in FIT_BINDINGS]
        key, (_, median) = max(training, key=lambda item: item[1][1])
        failed = [list(k) for k, (_, value) in native_transactions.items() if k[1] not in FIT_BINDINGS and value > median]
        result["source_tasks"]["native_transactions"] = {
            "version": 1, "protocol_id": NATIVE_PROTOCOL_ID, "available": not failed,
            "max_bindings": max(BINDINGS), "prepare_start_complete_seconds": median,
            "fit": {"method": "maximum median on predetermined fit grid", "fit_bindings": list(FIT_BINDINGS),
                    "maximum_median_row": list(key), "failed_holdout_rows": failed,
                    "scope": "complete typed prepare/start/complete management; excludes copies, waits and original body/team"},
            "measurements": [deepcopy(row) for row, _ in native_transactions.values()],
        }
    validate_source_costs(result, runtime_id)
    return result


def validate_source_costs(profile, runtime_id):
    """Read bounded optional evidence without discovering hardware or CUDA."""
    section = profile.get("source_tasks")
    if (not isinstance(runtime_id, str) or len(runtime_id) != 64
            or any(character not in "0123456789abcdef" for character in runtime_id)
            or not isinstance(section, dict) or type(section.get("version")) is not int or section["version"] != 1
            or section.get("protocol_id") != PROTOCOL_ID or type(section.get("available")) is not bool
            or section.get("runtime_id") != runtime_id or section.get("precision_bits") != profile["precision_bits"]
            or section.get("application_profiled") is not False):
        raise ValueError("source protocol identity or runtime mismatch")
    for name in ("max_bindings", "max_launches", "max_scratch_bytes"):
        if type(section.get(name)) is not int or section[name] <= 0:
            raise ValueError("source protocol requires positive integer bounds")
    if (section["max_bindings"] > max(BINDINGS) or section["max_launches"] > max(LAUNCHES)
            or section["max_scratch_bytes"] > 1024**3):
        raise ValueError("source protocol bounds exceed the measured protocol")
    identity = section.get("identity", {})
    expected = {"version": 1, "gpu_uuid": profile["hardware"]["gpu_uuid"],
                "device": profile["hardware"]["device_ordinal"], "cpu_threads": profile["cpu_threads"],
                "precision": profile["precision_bits"],
                **{key: profile["toolchain"][key] for key in ("cuda_runtime_version", "driver_version")}}
    if not isinstance(identity, dict) or any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("source protocol hardware identity differs from the hardware profile")
    _schedule(identity)
    if set(section.get("costs", {})) != set(SOURCE_COST_NAMES):
        raise ValueError("source protocol costs are incomplete")
    for name in SOURCE_COST_NAMES:
        _positive(section["costs"][name], name)
    if "native_transactions" in section:
        native = section["native_transactions"]
        if (not isinstance(native, dict) or type(native.get("version")) is not int or native["version"] != 1
                or native.get("protocol_id") != NATIVE_PROTOCOL_ID or type(native.get("available")) is not bool
                or type(native.get("max_bindings")) is not int or not 0 < native["max_bindings"] <= max(BINDINGS)):
            raise ValueError("native source protocol identity or bounds are invalid")
        _positive(native.get("prepare_start_complete_seconds"), "native prepare/start/complete")
    return section


def source_native_costs(profile, runtime_id):
    # The native protocol has independent observations and holdouts. A failed
    # GPU submit fit cannot erase valid CPU management evidence.
    section = validate_source_costs(profile, runtime_id).get("native_transactions")
    if section is None or not section["available"]:
        raise ValueError("native source protocol independent holdout coverage is unavailable")
    return {name: section[name] for name in ("version", "protocol_id", "max_bindings", "prepare_start_complete_seconds")}


def source_task_costs(profile, runtime_id):
    """Unavailable or failed independent evidence cannot enable a placement."""
    section = validate_source_costs(profile, runtime_id)
    if not section["available"]:
        raise ValueError("source protocol independent holdout coverage is unavailable")
    return section


def calibrate_source_protocol(profile, args, directory, nvcc, host, *, run, tool):
    """Run generic independent probes once, preserving exact generated inputs."""
    require_explicit_cuda_environment()
    sources, manifest = read_scoped_runtime()
    runtime_id = manifest["runtime_id"]
    root = directory / "source-tasks"
    root.mkdir(exist_ok=True)
    for name, content in sources.items():
        (root / name).write_text(content)
    path, binary = root / "source-protocol.cu", root / "source-protocol"
    structural = getattr(args, "source_submit_model", False)
    path.write_text(source_scratch_benchmark(precision=args.precision, structural_submit=structural))
    build = [nvcc, "-O3", *cuda_compile_options(), "-std=c++17", "-arch=" + args.arch, "-ccbin", host,
             "-Xcompiler=-fopenmp", "-I" + str(root), str(path), str(root / "scoped_runtime.cu"), "-o", str(binary)]
    run(build, root, root / "build.log", timeout=180)
    environment = dict(os.environ)
    for name in list(environment):
        if name in {"FORT_RUNTIME_TRACE", "FORT_PHASE_TIMING", "CUDA_LAUNCH_BLOCKING"} or name.startswith("FORT_SCOPE_TEST_"):
            environment.pop(name)
    command = [str(binary), str(args.threads), str(args.max_mib), str(args.device)]
    records = _parse(run(command, root, root / "measurements.jsonl", timeout=240, env=environment))
    fc = tool(args.fortran, ("gfortran-15", "gfortran-14", "gfortran"))
    flags = list(args.fortran_flag) or ["-std=f2018", "-O3", "-fopenmp"]
    if "-fopenmp" not in flags:
        raise ValueError("source original-team calibration requires -fopenmp")
    compiler_version = run([fc, "--version"], root, root / "fortran-version.log", timeout=30)
    team_records, commands, original_sources = [], [], {}
    for rank in (1, 2, 3):
        for collapse in range(1, rank + 1):
            source, target = root / f"team-{rank}-{collapse}.f90", root / f"team-{rank}-{collapse}"
            source.write_text(original_team_benchmark(units=TEAM_UNITS, rank=rank, collapse=collapse))
            compile_command = [fc, *flags, str(source), "-o", str(target)]
            run(compile_command, root, root / f"team-{rank}-{collapse}-build.log", timeout=60)
            command_team = [str(target), str(args.threads), "128"]
            team_records += _parse(run(command_team, root, root / f"team-{rank}-{collapse}.jsonl",
                                       timeout=60, env=environment))
            commands += [compile_command, command_team]
            original_sources[source.name] = sha256(source.read_bytes()).hexdigest()
    legacy = [row for row in records if row.get("name") != RAW_NAMES["source_submit_seconds"]
              or (row["bindings"] in BINDINGS and row["launches"] in LAUNCHES)]
    result = profile_with_source_measurements(profile, legacy, runtime_id=runtime_id,
        max_bytes=args.max_mib * 1024**2, team_records=team_records)
    if structural:
        from .source_submit_model import fit_model

        result["source_submit_model"] = fit_model(records, result, runtime_id)
        # The old constant protocol did not declare this launch-argument
        # envelope. Keep legacy consumers unavailable rather than relabel it.
        result["source_tasks"]["available"] = False
        result["source_tasks"]["availability_exclusion"] = "launch ABI envelope requires the separate structural submit model"
    result["source_tasks"]["calibration"] = {
        "build_command": build, "run_command": command, "original_team_commands": commands,
        "runtime": manifest, "source_sha256": sha256(path.read_bytes()).hexdigest(),
        "generator_sha256": sha256(Path(__file__).with_name("tile_protocol_calibration.py").read_bytes()).hexdigest(),
        "original_team_source_sha256": original_sources, "fortran": {"executable": fc,
            "version": compiler_version, "flags": flags},
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "artifacts": str(root), "application_profiled": False,
    }
    return result
