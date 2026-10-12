"""Nonzero control preflight and immutable-object preservation, without CUDA."""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from compiler.offload import calibrate
from compiler.offload import numerical_gpu_execution as gpu
from compiler.offload import numerical_gpu_execution_costs as costs
from compiler.offload import numerical_gpu_launch_calibration as producer
from compiler.tests.test_numerical_gpu_execution_costs import evidence, launch_observations


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    saved = evidence()
    section = saved[gpu.SECTION]
    base = {key: value for key, value in saved.items() if key != gpu.SECTION}
    records, calibration = deepcopy(section["measurements"]), deepcopy(section["calibration"])
    original = tmp_path / "retained"
    original.mkdir()
    for name in ("workers.cu", "gpu_recipes.hpp", "driver.cu", "workers.o", "driver.o", "numerical-gpu-execution"):
        (original / name).write_bytes((name + " retained production artifact").encode())
    tools = {}
    for name in ("nvcc", "host"):
        path = tmp_path / name
        path.write_bytes((name + " exact compiler binary").encode())
        tools[name] = str(path)
    identity = records[0]
    identity["build_paths"] = {**identity["build_paths"], **tools, "artifacts": str(original)}
    identity["build_commands"] = gpu.gpu_build_commands(identity["build_paths"], identity["architecture"])
    for collection in ("emitted_sources", "objects"):
        identity[collection] = {name: sha256((original / name).read_bytes()).hexdigest() for name in identity[collection]}
    identity["binary_sha256"] = sha256((original / "numerical-gpu-execution").read_bytes()).hexdigest()
    calibration["artifacts"] = str(original)
    calibration["raw_samples"]["path"] = str(original / "gpu-raw-samples.jsonl")
    calibration["run_commands"] = [["taskset", "-c", "4", str(original / "numerical-gpu-execution"), "0", mode]
                                   for mode in ("--identity", "--preflight", "--measure")]
    profile = gpu.profile_from_gpu_execution_measurements(base, records, calibration)
    input_path = tmp_path / "parent.json"
    input_path.write_text(json.dumps(profile))
    args = SimpleNamespace(build_dir=tmp_path / "new-launch", nvcc=tools["nvcc"], cuda_host_cxx=tools["host"],
                           profile_reference=producer._reference(input_path))
    control, _ = launch_observations(profile)
    raw_keys = ("kind", "protocol_id", "backend_id", "generator_id", "source_generator_id", "registry_id",
                "precision_bits", "device_ordinal", "gpu_name", "gpu_uuid", "compute_capability",
                "cuda_runtime_version", "driver_version", "sampling_cpu_affinity")
    raw_identity = {key: identity[key] for key in raw_keys}
    calls, change = [], {}
    monkeypatch.setattr(calibrate, "cpu_identity", lambda: profile["hardware"]["cpu_name"])
    monkeypatch.setattr(producer.os, "sched_getaffinity", lambda _: set(range(12)))
    for key in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(key, raising=False)

    def run(argv, directory, log, *, timeout, env=None):
        calls.append(list(argv))
        if argv[-1] == "--version":
            value = profile["toolchain"]["nvcc_version" if argv[0] == tools["nvcc"] else "host_cxx_version"]
        elif "-o" in argv:
            Path(argv[argv.index("-o") + 1]).write_bytes(b"new control object or executable")
            value = ""
        else:
            mode = argv[-1]
            assert argv[:3] == ["taskset", "-c", "4"]
            assert env["FORT_PHASE_TIMING"] == "0"
            assert "CUDA_LAUNCH_BLOCKING" not in env
            emitted = [deepcopy(raw_identity)]
            if mode != "--identity":
                emitted.extend(deepcopy(control[1:5]))
            if mode == "--measure":
                emitted.extend(deepcopy(control[5:]))
                (directory / "launch-raw-samples.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in emitted[5:]))
            if change.get("mode") == mode:
                change["callback"](emitted, directory)
            value = "".join(json.dumps(row) + "\n" for row in emitted)
        log.write_text(value)
        return value

    return SimpleNamespace(profile=profile, base=deepcopy(profile), original=original, tools=tools,
                           args=args, run=run, tool=lambda path, _: path, calls=calls, change=change)


def collect(attempt):
    return producer.calibrate_gpu_launch(attempt.profile, attempt.args, run=attempt.run, tool=attempt.tool)


def test_producer_links_exact_retained_workers_and_preserves_every_parent_section(attempt):
    result = collect(attempt)
    costs.validate_gpu_execution_costs(result)
    section = result[costs.SECTION]
    assert attempt.profile == attempt.base
    assert {k: v for k, v in result.items() if k != costs.SECTION} == attempt.base
    builds = [argv for argv in attempt.calls if "-o" in argv]
    assert len(builds) == 2
    assert builds == section["identity"]["build_commands"]
    assert "workers.cu" not in " ".join(arg for argv in builds for arg in argv)
    assert (attempt.args.build_dir / "workers.o").read_bytes() == (attempt.original / "workers.o").read_bytes()
    assert [argv[-1] for argv in attempt.calls[-3:]] == ["--identity", "--preflight", "--measure"]
    assert json.loads((attempt.args.build_dir / "preflight-receipt.json").read_text())["sampling_started"] is False
    assert len((attempt.args.build_dir / "launch-raw-samples.jsonl").read_text().splitlines()) == 21
    assert section["launch_enqueue_included"] is True


@pytest.mark.parametrize("defect", ["agreement", "missing", "launches", "identity"])
def test_failed_preflight_never_begins_sampling(attempt, defect):
    def corrupt(rows, directory):
        if defect == "missing":
            rows.pop()
        elif defect == "agreement":
            rows[-1]["agreement_passed"] = False
        elif defect == "launches":
            rows[1]["launches"] = 1
        else:
            rows[0]["gpu_uuid"] = "another device"

    attempt.change.update(mode="--preflight", callback=corrupt)
    with pytest.raises(gpu.NumericalGpuExecutionError):
        collect(attempt)
    assert "--measure" not in [argv[-1] for argv in attempt.calls]
    assert not (attempt.args.build_dir / "launch-raw-samples.jsonl").exists()
    assert (attempt.args.build_dir / "failure.json").is_file()
    assert attempt.profile == attempt.base


@pytest.mark.parametrize("artifact", ["workers.o", "driver.cu", "tool"])
def test_parent_or_tool_mutation_during_preflight_rejects_before_measure(attempt, artifact):
    def corrupt(rows, directory):
        path = Path(attempt.tools["nvcc"]) if artifact == "tool" else attempt.original / artifact
        path.write_bytes(b"changed after actual identity")

    attempt.change.update(mode="--preflight", callback=corrupt)
    with pytest.raises(gpu.NumericalGpuExecutionError, match="changed/differs"):
        collect(attempt)
    assert "--measure" not in [argv[-1] for argv in attempt.calls]


def test_interrupted_sampling_retains_raw_and_never_retries(attempt):
    def interrupt(rows, directory):
        (directory / "launch-raw-samples.jsonl").write_text(json.dumps(rows[5]) + "\n")
        raise KeyboardInterrupt

    attempt.change.update(mode="--measure", callback=interrupt)
    with pytest.raises(KeyboardInterrupt):
        collect(attempt)
    failure = json.loads((attempt.args.build_dir / "failure.json").read_text())
    assert failure["error_type"] == "KeyboardInterrupt"
    assert [argv[-1] for argv in attempt.calls].count("--measure") == 1
    assert len((attempt.args.build_dir / "launch-raw-samples.jsonl").read_text().splitlines()) == 1
    assert attempt.profile == attempt.base


@pytest.mark.parametrize("text", ["", "null", "[]", "not json", '{"kind":"unknown"}'])
def test_parser_rejects_invalid_raw_output_with_controlled_error(text):
    with pytest.raises(gpu.NumericalGpuExecutionError):
        producer._parse(text)
