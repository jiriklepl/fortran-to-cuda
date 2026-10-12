"""Host-only production rendering and fail-closed GPU evidence orchestration."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from compiler.emission.cuda.kernels import generate_kernel
from compiler.offload import calibrate
from compiler.offload import numerical_gpu_execution as gpu
from compiler.offload import numerical_gpu_execution_calibration as producer
from compiler.offload.numerical_execution_workloads import NAMES, execution_recipes
from compiler.tests.test_numerical_gpu_execution import observations


@pytest.mark.parametrize("precision", [32, 64])
def test_all_seventeen_gpu_workers_use_exact_production_kernels(precision, tmp_path):
    recipes = execution_recipes(precision)
    sources = producer.gpu_sources(recipes, "a" * 64, "b" * 64, "c" * 64)
    workers, header = sources["workers.cu"], sources["gpu_recipes.hpp"]
    for name, text in sources.items():
        (tmp_path / name).write_text(text)
    assert len(recipes) == 17
    for recipe in recipes:
        name = "execution_gpu_" + recipe.name
        assert "namespace " + name + "_detail {\n" + "\n".join(generate_kernel(recipe.region)) + "\n}" in workers
        assert 'extern "C" int ' + name + "(" in workers
        assert "namespace " + name + " {" not in workers
        assert "using namespace " + name + "_detail;" in workers
        assert recipe.metadata["recipe_id"] in header
        assert name + "_training_reference" in header
        assert name + "_holdout_reference" in header
    assert workers.count("++*launches;") == 17
    assert "if(items<0||!launches)" in workers
    assert "native_serial" not in header
    assert ("using execution_real=double;" if precision == 64 else "using execution_real=float;") in header


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    profile, records, _ = observations()
    base = deepcopy(profile)
    identity = records[0]
    raw_keys = ("kind", "protocol_id", "backend_id", "generator_id", "source_generator_id", "registry_id",
                "precision_bits", "device_ordinal", "gpu_name", "gpu_uuid", "compute_capability",
                "cuda_runtime_version", "driver_version", "sampling_cpu_affinity")
    raw_identity = {key: identity[key] for key in raw_keys}
    target = tmp_path / "gpu-evidence"
    args = SimpleNamespace(build_dir=target, nvcc="/tools/nvcc", cuda_host_cxx="/tools/g++", arch="sm_86",
                           device=0, cpu_affinity="4", preflight_only=False)
    monkeypatch.setattr(calibrate, "cpu_identity", lambda: profile["hardware"]["cpu_name"])
    monkeypatch.setattr(producer.os, "sched_getaffinity", lambda _: set(range(12)))
    monkeypatch.setattr(producer, "current_gpu_identities", lambda _: (
        identity["source_generator_id"], identity["registry_id"], identity["generator_id"], identity["renderer_sources"]))
    monkeypatch.setattr(producer, "gpu_sources", lambda *args: {"workers.cu": "// production-render test elsewhere\n",
                                                             "gpu_recipes.hpp": "// frozen registry\n"})
    for key in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(key, raising=False)
    calls = []
    change = {}

    def run(argv, directory, log, *, timeout, env=None):
        calls.append(list(argv))
        if argv[-1] == "--version":
            key = "nvcc_version" if argv[0] == args.nvcc else "host_cxx_version"
            value = profile["toolchain"][key]
        elif "-o" in argv:
            Path(argv[argv.index("-o") + 1]).write_bytes(b"independent compiled object/binary")
            value = ""
        else:
            mode = argv[-1]
            assert argv[:3] == ["taskset", "-c", ",".join(map(str, raw_identity["sampling_cpu_affinity"]))]
            assert env["FORT_PHASE_TIMING"] == "0"
            assert not env.get("CUDA_LAUNCH_BLOCKING")
            emitted = [deepcopy(raw_identity)]
            if mode != "--identity":
                emitted.extend(deepcopy(records[1:1 + gpu.PREFLIGHT_RECORDS]))
            if mode == "--measure":
                batches = records[1 + gpu.PREFLIGHT_RECORDS:]
                raw = "".join(json.dumps(row) + "\n" for row in batches)
                (directory / "gpu-raw-samples.jsonl").write_text(raw)
                emitted.extend(deepcopy(batches))
            if change.get("mode") == mode:
                change["callback"](emitted, directory)
            value = "".join(json.dumps(row) + "\n" for row in emitted)
        log.write_text(value)
        return value

    return SimpleNamespace(profile=profile, base=base, records=records, identity=raw_identity, args=args,
                           target=target, run=run, tool=lambda path, _: path, calls=calls, change=change)


def collect(attempt):
    return producer.calibrate_gpu_execution(attempt.profile, attempt.args, run=attempt.run, tool=attempt.tool)


def test_producer_retains_exact_build_raw_and_base_without_auto(attempt):
    result = collect(attempt)
    gpu.validate_gpu_execution_profile(result)
    assert attempt.profile == attempt.base
    assert {key: value for key, value in result.items() if key != gpu.SECTION} == attempt.base
    assert [argv[-1] for argv in attempt.calls[-3:]] == ["--identity", "--preflight", "--measure"]
    section = result[gpu.SECTION]
    assert section["identity"]["build_commands"] == attempt.calls[2:5]
    assert section["calibration"]["raw_samples"]["sha256"] == producer._sha(attempt.target / "gpu-raw-samples.jsonl")
    assert section["identity"]["sampling_cpu_affinity"] == [4]
    assert section["identity"]["cpu_name"] == attempt.base["hardware"]["cpu_name"]
    assert json.loads((attempt.target / "preflight-receipt.json").read_text())["sampling_started"] is False
    assert (attempt.target / "evidence.json").is_file()
    assert section["automatic_placement_authorized"] is False


@pytest.mark.parametrize("preflight_only", [False, True])
@pytest.mark.parametrize("defect", ["agreement", "recipe", "launches", "missing"])
def test_bad_preflight_never_starts_timing(attempt, preflight_only, defect):
    attempt.args.preflight_only = preflight_only

    def corrupt(rows, directory):
        if defect == "missing":
            rows.pop()
        elif defect == "agreement":
            rows[1]["agreement_passed"] = False
        elif defect == "recipe":
            rows[1]["recipe_id"] = "0" * 64
        else:
            rows[1]["launches"] = 1

    attempt.change.update(mode="--preflight", callback=corrupt)
    with pytest.raises(gpu.NumericalGpuExecutionError):
        collect(attempt)
    assert "--measure" not in [argv[-1] for argv in attempt.calls]
    assert not (attempt.target / "gpu-raw-samples.jsonl").exists()
    assert (attempt.target / "failure.json").is_file()
    assert attempt.profile == attempt.base


@pytest.mark.parametrize("preflight_only", [False, True])
def test_producer_preserves_entire_cpu_section(attempt, preflight_only):
    attempt.args.preflight_only = preflight_only
    attempt.profile["numerical_execution"] = {"registry": list(gpu_registry(attempt.profile)),
        "identity": {"cpu_affinity": [4, 5, 6, 7]}, "retained_cpu_evidence": {"raw": [1, 2, 3]}}
    original = deepcopy(attempt.profile)
    attempt.args.cpu_affinity = None
    attempt.identity["sampling_cpu_affinity"] = [4, 5, 6, 7]
    result = collect(attempt)
    assert original == attempt.profile
    assert {key: value for key, value in result.items() if key != gpu.SECTION} == original
    assert result["numerical_execution"] == original["numerical_execution"]
    assert (gpu.SECTION in result) is not preflight_only
    assert (attempt.target / "gpu-raw-samples.jsonl").exists() is not preflight_only
    assert ("--measure" in [argv[-1] for argv in attempt.calls]) is not preflight_only


def gpu_registry(profile):
    from compiler.offload.numerical_execution_workloads import execution_registry

    return execution_registry(profile["precision_bits"])


@pytest.mark.parametrize("path", ["workers.cu", "driver.o", "numerical-gpu-execution"])
def test_artifact_mutation_during_preflight_rejects_before_measure(attempt, path):
    def corrupt(rows, directory):
        (directory / path).write_text("replaced after identity")

    attempt.change.update(mode="--preflight", callback=corrupt)
    with pytest.raises(gpu.NumericalGpuExecutionError, match="changed during execution"):
        collect(attempt)
    assert "--measure" not in [argv[-1] for argv in attempt.calls]


def test_interruption_preserves_raw_attempt_and_does_not_retry(attempt):
    def interrupt(rows, directory):
        (directory / "gpu-raw-samples.jsonl").write_text(json.dumps(rows[-1]) + "\n")
        raise KeyboardInterrupt

    attempt.change.update(mode="--measure", callback=interrupt)
    with pytest.raises(KeyboardInterrupt):
        collect(attempt)
    failure = json.loads((attempt.target / "failure.json").read_text())
    assert failure["error_type"] == "KeyboardInterrupt"
    assert failure["automatic_placement_authorized"] is False
    assert len((attempt.target / "gpu-raw-samples.jsonl").read_text().splitlines()) == 1
    assert [argv[-1] for argv in attempt.calls].count("--measure") == 1
    assert attempt.profile == attempt.base


@pytest.mark.parametrize("placement", [None, "", "99", "4,4", "x", "5,4"])
def test_unavailable_or_missing_cpu_placement_fails_before_compilation(attempt, placement):
    attempt.args.cpu_affinity = placement
    with pytest.raises(gpu.NumericalGpuExecutionError, match="CPU"):
        collect(attempt)
    assert not attempt.calls


@pytest.mark.parametrize("text", ["", "not json\n", "null\n", "[]\n", '{"kind":"unknown"}\n'])
def test_output_parser_has_controlled_errors(text):
    with pytest.raises(gpu.NumericalGpuExecutionError):
        producer._parse(text)


def test_driver_protocol_keeps_zero_launch_and_preflight_before_sampling():
    source = Path(producer.__file__).with_name("numerical_gpu_execution_driver.cu").read_text()
    assert source.index('if(mode=="--preflight")return 0;') < source.index('std::fopen("gpu-raw-samples.jsonl","wx")')
    assert "data.agrees()&&launches==std::uint64_t(n>0)" in source
    assert "end==argv[1]" in source
    assert tuple(NAMES) == tuple(recipe["name"] for recipe in gpu_registry({"precision_bits": 64}))
