"""Actual CUDA correctness for graph AUTO with synthetic ranking costs.

The profile binds real hardware/tool identities but deliberately prices CPU
work very slowly. This is a policy and coherence regression, not a timing or
speedup measurement. No numerical/source-family timing record is supplied.
"""

from __future__ import annotations

import json
import os
import shutil
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.numerical_contract import require_numerical_build_contract
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_graph_source_owner_cuda import DRIVER, MAPPED_SOURCE
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run

SOURCE = MAPPED_SOURCE.replace("y(i)=y(i)+3.0_8",
    "if(y(i)>0.0_8) y(i)=sqrt(y(i)*y(i))+3.0_8")


@pytest.fixture(scope="module")
def graph_auto_binaries(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA, C++ and Fortran toolchains required")
    directory = tmp_path_factory.mktemp("source_graph_auto")
    profile = actual_profile(directory, nvcc, host)
    # A conspicuous cold debt makes a first-call warm choice observable. This
    # debt is a synthetic prediction and is never substituted for elapsed time.
    profile["scoped"]["costs"]["cold_driver_startup_seconds"] = 20.0
    for direction in ("h2d", "d2h"):
        profile["rates"][direction + "_pageable"].update(
            latency_seconds=1e-7, bandwidth_bytes_per_second=1e12)
    assert not any(name.startswith("numerical_") and name != "numerical_contract" for name in profile)
    architecture = profile["hardware"]["compute_capability"].replace(".", "")
    original = directory / "original.f90"
    original.write_text(SOURCE)
    builder = ScopeBuilder([original], "graph_fixture::advance",
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
               "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "x")}},
        options=CompilerOptions(opt_level=0), config=OffloadConfig("auto", profile))
    outputs, manifest = builder.run()
    owner, = manifest["scopes"]
    assert owner["gpu_leaves"]
    assert owner["counted_controls"]
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for item in manifest["numerical_decisions"]:
        if item["procedure"] in owner["gpu_leaves"]:
            assert item["graph_auto_costs"]["heuristic"]
            assert item["automatic_estimate_available"], item
    for name, text in outputs.items():
        artifact = directory / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    flags = [fortran, "-O2", "-cpp", "-std=f2018", "-fopenmp",
             "-fcheck=all,array-temps", "-ffp-contract=off"]
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] != role:
                continue
            artifact = directory / item["path"]
            target = artifact.with_suffix(".o")
            if item["language"] == "cuda":
                contract = item["numerical_contract"]
                require_numerical_build_contract(contract)
                compiler = [nvcc, "-O2", "-std=c++17", "-ccbin", host,
                    "-arch=sm_" + architecture, "-Xcompiler=-fopenmp",
                    *contract["required_cuda_options"],
                    *("-Xcompiler=" + option for option in contract["required_host_options"])]
            else:
                compiler = flags
            run([*compiler, "-I", str(directory), "-c", str(artifact), "-o", str(target)], cwd=directory)
            objects.append(str(target))
    driver = directory / "driver.f90"
    driver.write_text(DRIVER)
    native, candidate = directory / "native", directory / "candidate"
    run([*flags, str(original), str(driver), "-o", str(native)], cwd=directory)
    run([*flags, str(driver), *objects, "-L/usr/local/cuda/lib64",
         "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(candidate)], cwd=directory)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (directory / "synthetic-policy-profile.json").write_text(json.dumps(profile, indent=2))
    return directory, native, candidate


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_graph_auto_generic_rankings_preserve_every_field_halo_and_native_fallback(graph_auto_binaries, disabled):
    directory, native, candidate = graph_auto_binaries
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        env["CUDA_VISIBLE_DEVICES"] = ""
    run([str(native)], cwd=directory, env=env)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(candidate)], cwd=directory, env=env)
    assert (directory / "fields.bin").read_bytes() == expected
    assert "array temporary" not in result.stderr.lower()
    (directory / ("disabled.stderr" if disabled else "enabled.stderr")).write_text(result.stderr)
    events = [json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in result.stderr.splitlines()
              if line.startswith("FORT_SCOPED evidence ")]
    decisions = [event for event in events if event["event"] == "decision"]
    assert decisions
    assert all(event["ranking_policy"] == "warm_graph_heuristic" for event in decisions)
    if disabled:
        assert all(event["gpu_units"] == 0 for event in decisions)
        assert "FORT_SCOPED upload " not in result.stderr
        assert "FORT_SCOPED download " not in result.stderr
        assert "FORT_SCOPED graph_submit " not in result.stderr
        assert "FORT_SCOPED launch " not in result.stderr
    else:
        selected = [event for event in decisions if event["gpu_units"]]
        assert selected
        cold = [event for event in selected if event["cold_driver_startup_debt_seconds"]]
        assert cold
        assert cold[0]["cold_driver_startup_debt_seconds"] == 20.0
        assert cold[0]["estimated_seconds"] > cold[0]["native_seconds"]
        assert cold[0]["warm_gate_seconds"] <= (cold[0]["warm_gate_counterfactual_seconds"]
            - cold[0]["warm_gate_required_saving_seconds"])
        assert cold[0]["terminal_debt_adjusted_ranking_seconds"] >= 0
        assert cold[0]["startup_debt_is_prediction"] is True
        assert any("FORT_SCOPED " + action + " " in result.stderr for action in ("launch", "graph_submit"))
