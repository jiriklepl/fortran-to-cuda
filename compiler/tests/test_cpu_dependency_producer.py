"""Producer identities are checked before sampling; no real calibration here."""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from compiler.ir import ScalarType
from compiler.offload.compute_dependencies import ComputeOperation
from compiler.offload.cpu_dependency_calibration import (
    _require_literal_three,
    calibrate_dependency,
    dependency_generator_identity,
    profile_from_dependency_measurements,
)
from compiler.offload.cpu_dependency_workloads import dependency_recipes
from compiler.offload.cpu_protocol_calibration import profile_from_cpu_protocol_measurements
from compiler.offload.numerical_calibration import NumericalCalibrationError
from compiler.tests.test_cpu_dependency_calibration import observations


class FakeDependencyRun:
    """Write fake build products and return already-predeclared raw samples."""

    def __init__(self, profile, rows, *, mutation=None):
        self.profile = profile
        self.rows = rows
        self.mutation = mutation
        self.calls = []
        self.measurements = 0

    def identity(self):
        protocol = self.profile["cpu_execution_protocol"]["execution_identity"]
        identity = {
            "kind": "cpu_dependency_identity_v1",
            "cpu_threads": self.profile["cpu_threads"],
            "precision_bits": self.profile["precision_bits"],
            "cpu_affinity": deepcopy(protocol["cpu_affinity"]),
            "omp_dynamic": False,
            "omp_proc_bind": "false",
            "registry_generator_id": dependency_generator_identity(),
            "fortran": {key: protocol["fortran"][key] for key in ("compiler_version", "compiler_options")},
        }
        for key in ("actual_team_threads", "thread_limit", "omp_wait_policy", "gomp_spincount"):
            if key in protocol:
                identity[key] = protocol[key]
        if self.mutation == "threads":
            identity["cpu_threads"] += 1
        elif self.mutation == "precision":
            identity["precision_bits"] = 32
        elif self.mutation == "placement":
            identity["cpu_affinity"] = [999]
        elif self.mutation == "registry":
            identity["registry_generator_id"] = "e" * 64
        elif self.mutation == "fortran":
            identity["fortran"]["compiler_options"] = "-O0 -fopenmp"
        elif self.mutation == "dynamic":
            identity["omp_dynamic"] = True
        elif self.mutation == "binding":
            identity["omp_proc_bind"] = "close"
        elif self.mutation == "team":
            identity["actual_team_threads"] = 1
        elif self.mutation == "thread_limit":
            identity["thread_limit"] = 1
        elif self.mutation == "wait_policy":
            identity["omp_wait_policy"] = "ACTIVE" if protocol.get("omp_wait_policy") != "ACTIVE" else "PASSIVE"
        elif self.mutation == "spin_count":
            identity["gomp_spincount"] = "1" if protocol.get("gomp_spincount") != "1" else "2"
        return identity

    def __call__(self, command, cwd, log, *, timeout, env=None):
        self.calls.append((tuple(command), env))
        if "-o" in command:
            output = Path(command[command.index("-o") + 1])
            output.write_bytes(sha256("\0".join(command).encode()).digest())
        if "--version" in command:
            value = self.profile["cpu_execution_protocol"]["execution_identity"]["host"]["compiler_version"]
            output = "incompatible C++ toolchain" if self.mutation == "host" else value
        elif command[0] == "taskset":
            output = json.dumps(self.identity()) + "\n"
            if "--identity" not in command:
                self.measurements += 1
                output += "\n".join(json.dumps(row) for row in self.rows) + "\n"
                raw_rows = [
                    {**{key: value for key, value in row.items() if key not in {"kind", "samples"}},
                     **sample, "kind": "cpu_dependency_batch_v1"}
                    for row in self.rows for sample in row["samples"]
                ]
                (Path(cwd) / "cpu-dependency-raw-samples.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in raw_rows) + "\n"
                )
        else:
            output = ""
        Path(log).write_text(output)
        return output


def setup_producer(monkeypatch, tmp_path, *, mutation=None):
    profile, rows, _ = observations()
    affinity = profile["cpu_execution_protocol"]["execution_identity"]["cpu_affinity"]
    monkeypatch.setattr("compiler.offload.cpu_dependency_calibration.os.sched_getaffinity", lambda _: set(affinity))
    monkeypatch.setattr(
        "compiler.offload.cpu_dependency_calibration._tool", lambda chosen, defaults: chosen or defaults[0]
    )
    args = SimpleNamespace(
        fortran="gfortran", host_cxx="g++", build_dir=tmp_path / "attempt", fortran_flag=["-O3", "-fopenmp"]
    )
    return profile, args, FakeDependencyRun(profile, rows, mutation=mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        "threads",
        "precision",
        "placement",
        "registry",
        "fortran",
        "host",
        "dynamic",
        "binding",
        "team",
        "thread_limit",
        "wait_policy",
        "spin_count",
    ],
)
def test_backend_source_and_placement_mismatch_reject_before_any_timed_batch(monkeypatch, tmp_path, mutation):
    profile, args, run = setup_producer(monkeypatch, tmp_path, mutation=mutation)
    reason = "actual team/thread limit unavailable" if mutation in {"team", "thread_limit"} else "identity mismatch"
    with pytest.raises(NumericalCalibrationError, match=reason):
        calibrate_dependency(profile, args, run=run)
    assert run.measurements == 0
    assert any("--identity" in command for command, _ in run.calls)
    assert not (args.build_dir / "measurements.jsonl").exists()


def test_fake_producer_seals_actual_built_objects_and_original_semantic_flags(monkeypatch, tmp_path):
    profile, args, run = setup_producer(monkeypatch, tmp_path)
    result = calibrate_dependency(profile, args, run=run)
    assert run.measurements == 1
    section = result["cpu_dependency"]
    assert section["generator_id"] == dependency_generator_identity()
    assert section["measurements"] == run.rows
    assert result["numerical"] == profile["numerical"]
    assert section["calibration"]["application_profiled"] is False
    assert section["calibration"]["favorable_retry"] is False
    raw_receipt = section["calibration"]["raw_batch_artifact"]
    raw_path = Path(raw_receipt["path"])
    assert raw_path == args.build_dir / "cpu-dependency-raw-samples.jsonl"
    assert sha256(raw_path.read_bytes()).hexdigest() == raw_receipt["sha256"]
    assert len(raw_path.read_text().splitlines()) == 7 * len(run.rows)
    actual_objects = section["execution_identity"]["timed_objects"]
    assert len(actual_objects) == 2 + sum(
        len(recipe.fortran_sources) + len(recipe.cpp_sources)
        for recipe in dependency_recipes(profile["precision_bits"])
    )
    assert all(
        sha256((args.build_dir / name).read_bytes()).hexdigest() == value for name, value in actual_objects.items()
    )
    commands = json.loads((args.build_dir / "commands.json").read_text())
    assert any(command[0] == "gfortran" and "-O3" in command and "-fopenmp" in command for command in commands)
    for recipe in dependency_recipes(profile["precision_bits"]):
        if recipe.helper_form == "separate":
            paths = [command[command.index("-c") + 1] for command in commands if "-c" in command]
            for sources in (recipe.fortran_sources,):
                assert paths.index(str(args.build_dir / sources[0][0])) < paths.index(
                    str(args.build_dir / sources[1][0])
                )
    timed = [env for command, env in run.calls if command[0] == "taskset"]
    assert all(
        env["OMP_NUM_THREADS"] == "4" and env["OMP_DYNAMIC"] == "FALSE" and env["OMP_PROC_BIND"] == "false"
        for env in timed
    )
    assert all("GOMP_CPU_AFFINITY" not in env and "OMP_PLACES" not in env for env in timed)


@pytest.mark.parametrize(
    "flags", [[], ["-O3"], ["-O3", "-fopenmp", "-ffast-math"], ["-Ofast", "-fopenmp"], ["-O3", "-fopenmp", "-flto"]]
)
def test_unsupported_semantic_flags_never_build_or_sample(monkeypatch, tmp_path, flags):
    profile, args, run = setup_producer(monkeypatch, tmp_path)
    args.fortran_flag = flags
    with pytest.raises(NumericalCalibrationError, match="Fortran flags"):
        calibrate_dependency(profile, args, run=run)
    assert run.calls == []
    assert not args.build_dir.exists()


def test_unavailable_cpu_placement_never_builds_or_samples(monkeypatch, tmp_path):
    profile, args, run = setup_producer(monkeypatch, tmp_path)
    monkeypatch.setattr("compiler.offload.cpu_dependency_calibration.os.sched_getaffinity", lambda _: {999})
    with pytest.raises(NumericalCalibrationError, match="placement"):
        calibrate_dependency(profile, args, run=run)
    assert run.calls == []


def test_producer_never_overwrites_a_prior_attempt(monkeypatch, tmp_path):
    profile, args, run = setup_producer(monkeypatch, tmp_path)
    args.build_dir.mkdir()
    receipt = args.build_dir / "prior-rejection.json"
    receipt.write_text('{"rejected":true}\n')
    with pytest.raises(FileExistsError):
        calibrate_dependency(profile, args, run=run)
    assert run.calls == []
    assert receipt.read_text() == '{"rejected":true}\n'


def test_rejected_fresh_memory_prevents_any_build_or_sampling(monkeypatch, tmp_path):
    profile, args, run = setup_producer(monkeypatch, tmp_path)
    protocol = profile["cpu_execution_protocol"]
    rows = deepcopy(protocol["measurements"])
    for row in rows:
        if row["kind"] == "cpu_memory_cost_v1" and row["role"] == "holdout":
            for sample in row["samples"]:
                sample["elapsed_seconds"] *= 3
                sample["wall_seconds"] *= 3
    rejected = profile_from_cpu_protocol_measurements(profile, rows,
        protocol["execution_identity"], protocol["team_proofs"])
    assert rejected["numerical"] == profile["numerical"]
    assert all(backend["memory"]["status"] == "rejected"
               for backend in rejected["cpu_execution_protocol"]["backends"].values())
    with pytest.raises(NumericalCalibrationError, match="no accepted fresh execution/memory protocol"):
        calibrate_dependency(rejected, args, run=run)
    assert run.calls == []
    assert not args.build_dir.exists()


@pytest.mark.parametrize("mutation", ["calibration", "recipe", "threads", "precision", "cpu_index"])
def test_malformed_optional_profile_fields_fail_as_unavailable_evidence(mutation):
    profile, rows, identity = observations()
    calibration = None
    if mutation == "calibration":
        calibration = []
    elif mutation == "recipe":
        rows[0]["recipe"] = []
    elif mutation == "threads":
        identity["cpu_threads"] = float(identity["cpu_threads"])
    elif mutation == "precision":
        identity["precision_bits"] = float(identity["precision_bits"])
    else:
        identity["cpu_affinity"][0] = float(identity["cpu_affinity"][0])
    with pytest.raises(NumericalCalibrationError):
        profile_from_dependency_measurements(profile, rows, identity, calibration=calibration)


@pytest.mark.parametrize("operands", [1, (None, 1), (None, ("real",)), (None, ("real", 3))])
def test_malformed_literal_provenance_does_not_escape_estimate_rejection(operands):
    operation = ComputeOperation("divide_constant", ScalarType.REAL, ((), ()), False, False, literal_operands=operands)
    with pytest.raises(NumericalCalibrationError):
        _require_literal_three(operation)
