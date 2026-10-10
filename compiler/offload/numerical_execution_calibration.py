"""Build, authenticate and execute one frozen CPU numerical protocol.

Every attempt uses a new directory. Worker objects are shared by the timed and
untimed proof executables; rejected and interrupted observations stay on disk.
No historical numerical model or GPU estimate is substituted into this route.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from hashlib import sha256
from pathlib import Path

from compiler.numerical_contract import HOST_OPTIONS, require_explicit_cuda_environment
from compiler.offload.calibrate import CalibrationError, _run, _tool, cpu_identity
from compiler.offload.collective_calibration import normalize_fortran_options
from compiler.offload.cpu_protocol_calibration import TEAM_PROOF_SOURCE
from compiler.offload.numerical_calibration import NumericalCalibrationError, require_numerical_profile_contract
from compiler.offload.numerical_execution import PROOF_BACKEND, validate_execution_preflight
from compiler.offload.numerical_execution_workloads import (
    BACKEND_ID,
    CPU_BACKENDS,
    PROTOCOL_ID,
    STARTUP_SIZES,
    execution_generator_identity,
    execution_recipes,
    execution_registry,
    native_identity_source,
    registry_identity,
    registry_source,
)
from compiler.offload.schedule_calibrate import measurement_environment

HOST_FLAGS = ("-O3", "-std=c++17", "-fopenmp", *HOST_OPTIONS)


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def parse_execution_output(text):
    allowed = {
        "numerical_execution_identity_v1",
        "numerical_execution_startup_v1",
        "numerical_execution_team_proof_v1",
        "numerical_execution_cost_v1",
        "numerical_execution_smoke_v1",
    }
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise NumericalCalibrationError("numerical execution output must be JSONL") from error
        if not isinstance(row, dict) or row.get("kind") not in allowed:
            raise NumericalCalibrationError("unknown numerical execution record")
        rows.append(row)
    identities = [row for row in rows if row["kind"] == "numerical_execution_identity_v1"]
    if len(identities) != 1:
        raise NumericalCalibrationError("one numerical execution identity is required")
    return identities[0], [row for row in rows if row["kind"] != "numerical_execution_identity_v1"]


def _checked_identity(raw, *, precision, affinity, environment, generator, registry):
    required = {
        "protocol_id": PROTOCOL_ID,
        "backend_id": BACKEND_ID,
        "generator_id": generator,
        "registry_id": registry,
        "precision_bits": precision,
        "cpu_threads": 4,
        "cpu_affinity": affinity,
        "actual_team_threads": 4,
        "omp_dynamic": False,
        "omp_proc_bind": "false",
        "omp_wait_policy": environment.get("OMP_WAIT_POLICY"),
        "gomp_spincount": environment.get("GOMP_SPINCOUNT"),
    }
    if any(
        raw.get(k) != v or (type(v) in {bool, int} and type(raw.get(k)) is not type(v)) for k, v in required.items()
    ):
        raise NumericalCalibrationError("numerical execution source/backend/environment identity mismatch")
    if type(raw.get("thread_limit")) is not int or raw["thread_limit"] < 4:
        raise NumericalCalibrationError("numerical execution thread limit is below four")
    for value in (raw.get("fortran", {}).get("compiler_version"), raw.get("fortran", {}).get("compiler_options")):
        if not isinstance(value, str) or not value:
            raise NumericalCalibrationError("numerical execution original Fortran identity missing")
    return raw


def reconcile_raw_batches(raw_records, aggregate, registry):
    """Require exact source/order/value agreement, never repair conflicting raw facts."""
    recipes = {r["name"]: r for r in registry}
    indexed = {}
    for row in aggregate:
        key = row.get("kind"), row.get("recipe"), row.get("backend"), row.get("items")
        if key in indexed:
            raise NumericalCalibrationError("duplicate aggregate numerical execution cell")
        recipe = recipes.get(row.get("recipe"))
        if recipe is None or row.get("recipe_id") != recipe["recipe_id"] or row.get("family") != recipe["family"]:
            raise NumericalCalibrationError("aggregate numerical execution recipe identity mismatch")
        indexed[key] = row
    expected = []
    for batch in range(7):
        for items in STARTUP_SIZES:
            for turn in range(2):
                expected.append(
                    (batch, "numerical_execution_startup_v1", "memory", CPU_BACKENDS[1 + (batch + turn) % 2], items)
                )
        for recipe in registry:
            for items in recipe["sizes"]:
                for turn in range(3):
                    expected.append(
                        (batch, "numerical_execution_cost_v1", recipe["name"], CPU_BACKENDS[(batch + turn) % 3], items)
                    )
    if len(raw_records) != len(expected):
        raise NumericalCalibrationError("incomplete numerical execution raw batches")
    seen = set()
    for order, (raw, wanted) in enumerate(zip(raw_records, expected, strict=True)):
        batch, kind, name, backend, items = wanted
        raw_kind = (
            "numerical_execution_startup_batch_v1" if kind.endswith("startup_v1") else "numerical_execution_batch_v1"
        )
        recipe = recipes[name]
        key = (kind, name, backend, items)
        row = indexed.get(key)
        if row is None or raw.get("kind") != raw_kind:
            raise NumericalCalibrationError("numerical execution raw registry/order mismatch")
        role = "fit" if items in recipe["fit_sizes"] else "holdout"
        facts = {
            "recipe": name,
            "recipe_id": recipe["recipe_id"],
            "family": recipe["family"],
            "backend": backend,
            "items": items,
            "role": role,
            "batch": batch,
            "global_order": order,
            "agreement_passed": True,
            "traffic_bytes": items * recipe["precision_bits"] // 8 * 3,
            "working_set_bytes": items * recipe["precision_bits"] // 8 * 3,
        }
        if any(
            raw.get(k) != v or (type(v) in {bool, int} and type(raw.get(k)) is not type(v)) for k, v in facts.items()
        ):
            raise NumericalCalibrationError("numerical execution raw source/role/order conflict")
        samples = row.get("samples")
        if not isinstance(samples, list) or len(samples) != 7:
            raise NumericalCalibrationError("seven aggregate samples required")
        sample = samples[batch]
        for key in ("batch", "global_order", "repetitions"):
            if (
                type(raw.get(key)) is not int
                or type(sample.get(key)) is not int
                or raw[key] < (1 if key == "repetitions" else 0)
            ):
                raise NumericalCalibrationError("numerical execution raw integer field invalid")
        for key in ("elapsed_seconds", "wall_seconds"):
            for value in (raw.get(key), sample.get(key)):
                if (
                    type(value) not in {int, float}
                    or not math.isfinite(value)
                    or value <= 0
                    or key == "wall_seconds"
                    and value < 0.2
                ):
                    raise NumericalCalibrationError("numerical execution raw duration invalid")
        for key in ("batch", "global_order", "repetitions", "elapsed_seconds", "wall_seconds"):
            if sample.get(key) != raw.get(key):
                raise NumericalCalibrationError("numerical execution raw/aggregate duration conflict")
        if row.get("role") != role or row.get("agreement_passed") is not True:
            raise NumericalCalibrationError("aggregate numerical execution agreement or role mismatch")
        seen.add((kind, name, backend, items))
    if seen != set(indexed):
        raise NumericalCalibrationError("unplanned aggregate numerical execution cells")


def calibrate_numerical_execution(profile, args, directory, nvcc, host, *, run, tool):
    """CPU-only first: compile/prove both identities, then sample once, without retries."""
    del nvcc  # The new section grants no GPU coefficient or execution claim.
    require_explicit_cuda_environment()
    require_numerical_profile_contract(profile)
    precision = getattr(args, "precision", profile.get("precision_bits"))
    threads = getattr(args, "threads", profile.get("cpu_threads"))
    if type(threads) is not int or threads != 4 or profile.get("cpu_threads") != 4:
        raise NumericalCalibrationError("numerical execution v1 requires the four-thread series")
    if precision not in (32, 64) or precision != profile.get("precision_bits"):
        raise NumericalCalibrationError("numerical execution precision mismatch")
    current_cpu = cpu_identity()
    if profile.get("hardware", {}).get("cpu_name") != current_cpu:
        raise NumericalCalibrationError("numerical execution CPU identity differs from the base profile")
    flags = list(getattr(args, "fortran_flag", None) or [])
    if (
        not flags
        or "-fopenmp" not in flags
        or any(
            not isinstance(f, str) or not f or f in {"-ffast-math", "-Ofast", "-flto"} or f.startswith("-flto=")
            for f in flags
        )
    ):
        raise NumericalCalibrationError(
            "supply exact original native Fortran flags including -fopenmp, without LTO/fast-math"
        )
    if "numerical_execution" in profile:
        raise NumericalCalibrationError(
            "preserve the existing numerical execution attempt; use an unmodified base profile"
        )
    allowed = sorted(os.sched_getaffinity(0))
    supplied = getattr(args, "cpu_affinity", None)
    try:
        affinity = sorted(set(map(int, supplied.split(",")))) if supplied else allowed[:4]
    except (AttributeError, ValueError) as error:
        raise NumericalCalibrationError("CPU affinity must be four comma-separated CPU indices") from error
    if len(affinity) != 4 or not set(affinity) <= set(allowed):
        raise NumericalCalibrationError("numerical execution CPU affinity unavailable")
    environment = measurement_environment(os.environ)
    environment["OMP_NUM_THREADS"] = "4"
    if environment.get("OMP_THREAD_LIMIT") is not None:
        try:
            limit = int(environment["OMP_THREAD_LIMIT"])
        except ValueError as error:
            raise NumericalCalibrationError("invalid OpenMP thread limit") from error
        if limit < 4:
            raise NumericalCalibrationError("OpenMP thread limit is below four")
    for key in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT"):
        value = environment.get(key)
        if value is not None and (len(value) > 128 or any(ord(c) < 32 or ord(c) > 126 for c in value)):
            raise NumericalCalibrationError("unsupported OpenMP wait/spin identity")
    fortran = tool(getattr(args, "fortran", None), ("gfortran-15", "gfortran-14", "gfortran"))
    target = (Path(directory) / "numerical-execution").resolve()
    target.mkdir(parents=True, exist_ok=False)
    host_version = run([host, "--version"], target, target / "host-version.log", timeout=30).strip()
    if profile.get("toolchain", {}).get("host_cxx_version") != host_version:
        raise NumericalCalibrationError("numerical execution host compiler differs from the base profile")
    generator = execution_generator_identity()
    registry = execution_registry(precision)
    _write(
        target / "registry.json",
        {
            "protocol_id": PROTOCOL_ID,
            "backend_id": BACKEND_ID,
            "generator_id": generator,
            "registry_id": registry_identity(precision),
            "recipes": registry,
            "native_flags": flags,
            "generated_flags": list(HOST_FLAGS),
            "cpu_affinity": affinity,
        },
    )
    recipes = execution_recipes(precision)
    (target / "numerical_execution_recipes.hpp").write_text(registry_source(recipes, generator))
    native_identity = target / "native-identity.f90"
    native_identity.write_text(native_identity_source())
    proof_source = target / "team-proof.cpp"
    proof_source.write_text(TEAM_PROOF_SOURCE)
    objects = []
    commands = []
    used = set()
    runtime = Path(__file__).parents[1] / "runtime"

    def invoke(command, log, timeout=180, env=None):
        commands.append(command)
        _write(
            target / f"invocation-{len(commands):03}.json",
            {"argv": command, "log": str(log), "timed": log.name == "measurements.jsonl"},
        )
        return run(command, target, log, timeout=timeout, env=env)

    for recipe in recipes:
        for name, text in (*recipe.native_sources, *recipe.cpp_sources):
            if Path(name).name != name or name in used:
                raise NumericalCalibrationError("execution source basenames must be unique")
            used.add(name)
            path = target / name
            path.write_text(text)
            obj = target / (name + ".o")
            compiler, options = (fortran, flags) if path.suffix == ".f90" else (host, list(HOST_FLAGS))
            invoke(
                [compiler, *options, "-I", str(runtime), "-c", str(path), "-o", str(obj)],
                target / (name + ".build.log"),
            )
            objects.append(obj)
    identity_object = target / "native-identity.o"
    invoke([fortran, *flags, "-c", str(native_identity), "-o", str(identity_object)], target / "identity-build.log")
    objects.append(identity_object)
    driver = target / "driver.o"
    source = Path(__file__).with_name("numerical_execution_driver.cpp")
    invoke([host, *HOST_FLAGS, "-I", str(target), "-c", str(source), "-o", str(driver)], target / "driver-build.log")
    objects.append(driver)
    binary = target / "numerical-execution"
    proof_binary = target / "numerical-execution-proof"
    invoke([host, "-fopenmp", *map(str, objects), "-lgfortran", "-o", str(binary)], target / "link.log")
    invoke(
        [
            host,
            *HOST_FLAGS,
            *map(str, objects),
            str(proof_source),
            "-lgfortran",
            "-Wl,--wrap=GOMP_parallel",
            "-o",
            str(proof_binary),
        ],
        target / "proof-link.log",
    )
    timed_objects = {p.name: _sha(p) for p in objects}
    prefix = ["taskset", "-c", ",".join(map(str, affinity))]
    invocation = [*prefix, str(binary), "4"]
    original_identity, unexpected = parse_execution_output(
        invoke([*invocation, "--identity"], target / "identity.jsonl", timeout=30, env=environment)
    )
    if unexpected:
        raise NumericalCalibrationError("identity preflight unexpectedly executed measurements")
    _checked_identity(
        original_identity,
        precision=precision,
        affinity=affinity,
        environment=environment,
        generator=generator,
        registry=registry_identity(precision),
    )
    identity = {
        **original_identity,
        "fortran": {
            **original_identity["fortran"],
            "semantic_options": normalize_fortran_options(original_identity["fortran"]["compiler_options"]),
        },
        "host": {"compiler_version": host_version, "semantic_options": list(HOST_FLAGS)},
        "cpu_name": current_cpu,
        "proof_backend": PROOF_BACKEND,
        "timed_objects": timed_objects,
    }

    def phase(command, log, timeout):
        if any(_sha(p) != timed_objects[p.name] for p in objects):
            raise NumericalCalibrationError("timed numerical execution objects changed")
        observed, rows = parse_execution_output(invoke(command, log, timeout=timeout, env=environment))
        if any(_sha(p) != timed_objects[p.name] for p in objects) or execution_generator_identity() != generator:
            raise NumericalCalibrationError("numerical execution sources or timed objects changed")
        if observed != original_identity:
            raise NumericalCalibrationError("numerical execution identity changed across phases")
        return rows

    proofs = phase([*prefix, str(proof_binary), "4", "--proof"], target / "team-proof.jsonl", 30)
    if len(proofs) != 6 or {(p.get("backend"), p.get("items")) for p in proofs} != {
        (b, n) for b in CPU_BACKENDS[1:] for n in STARTUP_SIZES
    }:
        raise NumericalCalibrationError("complete same-object numerical execution team proofs required")
    for p in proofs:
        if (
            p.get("kind") != "numerical_execution_team_proof_v1"
            or p.get("agreement_passed") is not True
            or p.get("parallel_entries") != 1
            or p.get("wrong_team_visits") != 0
            or p.get("thread_visits") != [1] * 4
        ):
            raise NumericalCalibrationError("numerical execution did not retain one exact original team")
        p.update(proof_backend=PROOF_BACKEND, timed_objects=timed_objects)
    smokes = phase([*invocation, "--smoke"], target / "smoke.jsonl", 180)
    expected_smokes = {(r["name"], b, n) for r in registry for b in CPU_BACKENDS for n in (0, 1, 8, 17)}
    if (
        len(smokes) != len(expected_smokes)
        or {(r.get("recipe"), r.get("backend"), r.get("items")) for r in smokes} != expected_smokes
        or any(r.get("kind") != "numerical_execution_smoke_v1" or r.get("agreement_passed") is not True for r in smokes)
    ):
        raise NumericalCalibrationError("complete numerical execution smoke required before sampling")
    validate_execution_preflight(profile, identity, proofs)
    preflight = {
        "identity": identity,
        "proofs": proofs,
        "smoke_cases": len(smokes),
        "sampling_started": False,
        "gpu_measured": False,
        "commands": commands.copy(),
        "timed_binary_sha256": _sha(binary),
        "proof_binary_sha256": _sha(proof_binary),
        "reader_source_sha256": _sha(Path(__file__).with_name("numerical_execution.py")),
    }
    _write(target / "preflight-receipt.json", preflight)
    if getattr(args, "preflight_only", False):
        return profile.copy()
    records = phase(invocation, target / "measurements.jsonl", 1800)
    raw = target / "numerical-execution-raw-samples.jsonl"
    try:
        raw_rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        raise NumericalCalibrationError("numerical execution raw batches unavailable") from error
    reconcile_raw_batches(raw_rows, records, registry)
    calibration = {
        "commands": commands,
        "artifacts": str(target),
        "native_flags": flags,
        "generated_flags": list(HOST_FLAGS),
        "reader_source_sha256": _sha(Path(__file__).with_name("numerical_execution.py")),
        "registry_artifact": {"path": str(target / "registry.json"), "sha256": _sha(target / "registry.json")},
        "timed_objects": timed_objects,
        "timed_binary_sha256": _sha(binary),
        "proof_binary_sha256": _sha(proof_binary),
        "raw_batch_artifact": {"path": str(raw), "sha256": _sha(raw)},
        "source_artifacts": {p.name: _sha(p) for p in target.iterdir() if p.suffix in {".f90", ".cpp", ".hpp"}},
        "sampling": "seven complete global rounds; >=200ms every fixed cell; allocation/preparation/checks outside clocks; no retries",
        "startup_coverage": "warmed original fork/join only; native serial S=0; no cold-runtime or numerical-environment guard claim",
        "application_profiled": False,
        "gpu_measured": False,
    }
    _write(target / "calibration-receipt.json", calibration)
    from compiler.offload.numerical_execution import profile_from_execution_measurements

    result = profile_from_execution_measurements(profile, [identity, *proofs, *records], calibration=calibration)
    _write(target / "evidence.json", result["numerical_execution"])
    return result


def main(argv=None):
    """Run the CPU-only optional protocol against a preserved hardware profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--fortran")
    parser.add_argument("--host-cxx")
    parser.add_argument("--fortran-flag", action="append", default=[])
    parser.add_argument("--cpu-affinity", help="exact four comma-separated CPU indices")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="compile and prove without sampling; output retains the base profile unchanged",
    )
    args = parser.parse_args(argv)
    if args.output.exists() or args.profile.resolve() == args.output.resolve() or args.build_dir.exists():
        parser.error("output and build directory must be new; preserve every attempt and the base profile")
    try:
        from compiler.offload.profile import load_profile

        profile_sha = _sha(args.profile)
        profile = load_profile(args.profile)
        host = _tool(args.host_cxx, ("g++", "c++"))
        result = calibrate_numerical_execution(profile, args, args.build_dir, None, host, run=_run, tool=_tool)
        if _sha(args.profile) != profile_sha:
            raise NumericalCalibrationError("base profile changed during numerical execution")
        target = args.build_dir / "numerical-execution"
        _write(
            target / "base-profile-receipt.json",
            {
                "path": str(args.profile.resolve()),
                "sha256": profile_sha,
                "preflight_only": args.preflight_only,
                "gpu_base_remeasured": False,
            },
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write(args.output, result)
    except (OSError, ValueError, CalibrationError) as error:
        parser.exit(1, "numerical execution: " + str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
