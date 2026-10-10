"""Optional frozen-model validation for original runtime-static OpenMP DOs.

Run separately from numerical calibration. The original profile is preserved;
this producer adds raw paired observations without changing any coefficient.
Use the same native Fortran flags and CPU placement as the original profile.
"""
from __future__ import annotations

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path

from .calibrate import CalibrationError, _run, _tool
from .collective_calibration import normalize_fortran_options
from .numerical_calibration import (
    COMPUTE_CLASSES,
    COMPUTE_HOLDOUT_SIZES,
    NumericalCalibrationError,
    compute_generator_identity,
    native_fixture_source,
    validate_numerical_profile,
)
from .schedule_calibration import (
    _execution_identity,
    profile_from_schedule_measurements,
    runtime_fixture_source,
    schedule_prediction_seconds,
)


def measurement_environment(environment):
    result = {name: value for name, value in environment.items()
              if name not in {"FORT_RUNTIME_TRACE", "FORT_PHASE_TIMING", "CUDA_LAUNCH_BLOCKING",
                              "OMP_PLACES", "GOMP_CPU_AFFINITY"}
              and not name.startswith("FORT_SCOPE_TEST_")}
    result.update(OMP_DYNAMIC="FALSE", OMP_PROC_BIND="false", OMP_SCHEDULE="static")
    return result


def parse_schedule_measurements(text):
    """Keep failed agreement evidence; reject unrecognized or ambiguous output."""
    records, identities = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, TypeError) as error:
            raise NumericalCalibrationError("schedule fixture emitted non-JSON output") from error
        if not isinstance(row, dict):
            raise NumericalCalibrationError("schedule fixture emitted a non-object")
        if row.get("kind") == "schedule_identity_v1":
            identities.append(row)
        elif row.get("kind") == "schedule_cost_v1":
            records.append(row)
        else:
            raise NumericalCalibrationError("schedule fixture emitted an unknown observation")
    if len(identities) != 1:
        raise NumericalCalibrationError("exactly one schedule execution identity is required")
    identity = {key: value for key, value in identities[0].items() if key != "kind"}
    for mode in ("static", "runtime"):
        try:
            compiler = identity["fortran"][mode]
            compiler["semantic_options"] = normalize_fortran_options(compiler["compiler_options"])
        except (KeyError, TypeError) as error:
            raise NumericalCalibrationError("schedule fixture lacks native Fortran identity") from error
    return records, identity


def calibrate_schedule(profile, args, *, run=_run):
    section = profile.get("numerical")
    if not isinstance(section, dict) or section.get("schema_version") != 2:
        raise NumericalCalibrationError("schedule validation requires native numerical v2")
    validate_numerical_profile(section, profile)
    if section.get("generator_id") != compute_generator_identity():
        raise NumericalCalibrationError("schedule validation numerical generator mismatch")
    # Do not spend minutes validating a protocol with no complete base model.
    # This preflight does not trim observations or change any applicability.
    accepted = [name for name, modes in section["workload_validation"].items()
                if modes["native_fork_join"]["status"] == "accepted"]
    if not accepted:
        raise NumericalCalibrationError("no independently validated native workload class is available")
    schedule_prediction_seconds(profile, COMPUTE_CLASSES[accepted[0]][0], COMPUTE_HOLDOUT_SIZES[-1])
    precision, threads = profile["precision_bits"], profile["cpu_threads"]
    affinity = section["identity"]["cpu_affinity"]
    flags = list(args.fortran_flag or [])
    if not flags or "-fopenmp" not in flags or any("fast-math" in flag or flag == "-Ofast" for flag in flags):
        raise NumericalCalibrationError("supply original native Fortran flags including -fopenmp without fast-math")
    fortran = _tool(args.fortran, ("gfortran",))
    host = _tool(args.host_cxx, ("g++",))
    target = Path(args.build_dir).resolve()
    target.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_suffix(".cpp")
    objects, commands = [], []
    for mode, contents in (("static", native_fixture_source(precision)),
                           ("runtime", runtime_fixture_source(precision))):
        prepared, obj = target / (mode + ".f90"), target / (mode + ".o")
        prepared.write_text(contents)
        command = [fortran, *flags, "-c", str(prepared), "-o", str(obj)]
        commands.append(command)
        run(command, target, target / (mode + "-build.log"), timeout=180)
        objects.append(obj)
    binary = target / "schedule-validation"
    command = [host, "-O3", "-std=c++17", "-fopenmp", "-DCALIBRATION_PRECISION=" + str(precision),
               str(source), *(str(obj) for obj in objects), "-lgfortran", "-o", str(binary)]
    commands.append(command)
    run(command, target, target / "driver-build.log", timeout=180)
    environment = measurement_environment(os.environ)
    environment["OMP_NUM_THREADS"] = str(threads)
    invocation = ["taskset", "-c", ",".join(map(str, affinity)), str(binary), str(threads)]
    identity_text = run([*invocation, "--identity"], target, target / "identity.jsonl", timeout=30, env=environment)
    _, identity = parse_schedule_measurements(identity_text)
    _execution_identity(profile, identity)
    text = run(invocation, target, target / "measurements.jsonl", timeout=1800, env=environment)
    records, identity = parse_schedule_measurements(text)
    result = profile_from_schedule_measurements(profile, records, identity, calibration={
        "build_commands": commands, "run_command": invocation, "artifacts": str(target),
        "sampling": "seven predetermined interleaved static/runtime batches >=200ms per family/size; no retries",
        "validation": "runtime medians against exact existing emitted totals; no coefficient fitting",
        "protocol_source_sha256": sha256(source.read_bytes()).hexdigest()})
    (target / "evidence.json").write_text(json.dumps(result["source_schedule_validation"], indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--fortran")
    parser.add_argument("--host-cxx")
    parser.add_argument("--fortran-flag", action="append", default=[])
    args = parser.parse_args(argv)
    if args.profile.resolve() == args.output.resolve() or args.output.exists():
        parser.error("output must be a new path; preserve the original profile")
    try:
        profile = json.loads(args.profile.read_text())
        result = calibrate_schedule(profile, args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, CalibrationError) as error:
        parser.exit(1, "schedule validation: " + str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
