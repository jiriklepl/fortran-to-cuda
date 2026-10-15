"""Complete-process evidence never prices a warm call as a cold application."""

import json
import os
import shutil
import subprocess
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import ModuleType

import pytest

from compiler.offload.exact_window import digest
from compiler.offload.selection_cost import PROTOCOL, process_evidence_decision
from compiler.tests.test_exact_window import observations


@pytest.fixture
def build_validation(monkeypatch):
    import sys

    module = ModuleType("compiler.offload.application_build")
    checked = []

    def validate(value, *, contract, mode):
        if value != {"contract_identity": contract["identity"], "variant": mode, "checked_fixture": True}:
            raise ValueError("executed build attestation differs")
        checked.append(mode)
        return value

    module.validate_application_build_observation = validate
    module.require_application_build_contract = lambda requested, **identity: {
        "identity": digest(identity), "checked_fixture": True, **identity}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return checked


def process_fixture(*, resident=.1, native=1.0):
    bundle = {"schema_version": 1, "entry": "fixture::advance", "groups": ["s"]}
    bundle["identity"] = digest(bundle)
    contract = {"protocol": PROTOCOL, "key": [{"resource": "n", "property": "integer_value"}],
                "threads": 4, "selection_identity": "a" * 64, "selection_bundle": bundle,
                "build_contract": {"identity": "b" * 64}, "expected_launches": 1}
    campaign = observations(contract, resident=resident, native=native)
    campaign["protocol"] = PROTOCOL
    workload = {"argv": ["application", "case"], "environment": {"EXAMPLE": None},
                "files": {"case.conf": "c" * 64}, "completion": {"steps": 109, "end": 1.0}}
    workload["identity"] = digest(workload)
    campaign["workload"] = workload
    contract["workloads"] = [workload]
    campaign["selection_bundle"] = bundle
    for row in campaign["observations"]:
        row["process"] = {"returncode": 0, "workload_identity": workload["identity"],
                          **{name: workload[name] for name in ("argv", "environment", "files", "completion")},
                          "selection_bundle_identity": bundle["identity"],
                          "wall_time_includes": ["startup", "first_invocation", "all_invocations", "output", "teardown"],
                          "seconds": row["seconds"]}
        row["build"] = {"contract_identity": contract["build_contract"]["identity"],
                        "variant": row["mode"], "checked_fixture": True}
        row["invocations"] = [{"ordinal": ordinal, "selection": contract["selection_identity"],
                               "key": [value], "seconds": .01, "outcome": row["mode"],
                               "launches": int(row["mode"] == "resident")}
                              for ordinal, value in enumerate((96, 128, 96))]
    return contract, {"protocol": PROTOCOL, "campaigns": [campaign]}


def test_complete_process_pairs_bind_both_builds_and_all_observed_keys(build_validation):
    contract, evidence = process_fixture()
    result = process_evidence_decision(evidence, contract)
    assert result["selection"] == "resident", result
    assert [entry["key"] for entry in result["accepted_keys"]] == [[96], [128]]
    assert len(build_validation) == 18
    assert build_validation.count("native") == 9
    assert result["campaigns"][0]["invocations"] == 3
    assert "no future invocation-count guarantee" in result["applicability"]


@pytest.mark.parametrize("mutation", [
    "startup", "completion", "sequence", "missing_native", "fake_gpu", "mixed_gpu",
    "build", "bundle", "observed_bundle", "nan_time", "bool_launches", "missing_first",
])
def test_incomplete_or_incompatible_process_evidence_cannot_authorize_gpu(build_validation, mutation):
    contract, evidence = process_fixture()
    campaign = evidence["campaigns"][0]
    row = next(item for item in campaign["observations"] if item["mode"] == "resident")
    if mutation == "startup":
        row["process"]["wall_time_includes"].remove("first_invocation")
    elif mutation == "completion":
        row["process"]["completion"] = {"steps": 2, "end": .01}
    elif mutation == "sequence":
        row["invocations"][1]["key"] = [129]
    elif mutation == "missing_native":
        campaign["observations"].pop(0)
    elif mutation == "fake_gpu":
        for observation in campaign["observations"]:
            if observation["mode"] == "resident":
                for invocation in observation["invocations"]:
                    invocation.update(outcome="native", launches=0)
    elif mutation == "mixed_gpu":
        row["invocations"][0]["launches"] = 2
    elif mutation == "build":
        row["build"]["contract_identity"] = "d" * 64
    elif mutation == "bundle":
        campaign["selection_bundle"] = {"identity": "d" * 64}
    elif mutation == "observed_bundle":
        row["process"]["selection_bundle_identity"] = "d" * 64
    elif mutation == "nan_time":
        row["invocations"][0]["seconds"] = float("nan")
    elif mutation == "bool_launches":
        row["invocations"][0]["launches"] = True
    elif mutation == "missing_first":
        row["invocations"].pop(0)
    result = process_evidence_decision(evidence, contract)
    assert result["selection"] == "native", result
    assert not result["available"], result


def test_expensive_complete_process_is_native_even_if_warm_invocations_are_fast(build_validation):
    contract, evidence = process_fixture(resident=2)
    result = process_evidence_decision(evidence, contract)
    assert result["available"]
    assert result["selection"] == "native"
    assert result["accepted_keys"] == []


def test_one_unprofitable_workload_vetoes_shared_key_from_profitable_workload(build_validation):
    contract, evidence = process_fixture()
    _, expensive = process_fixture(resident=2)
    evidence["campaigns"].extend(deepcopy(expensive["campaigns"]))
    assert process_evidence_decision(evidence, contract)["selection"] == "native"


def test_multiple_admitted_selections_cannot_borrow_aggregate_profit_for_a_subset(build_validation):
    contract, evidence = process_fixture()
    contract["selection_bundle"]["groups"].append("another")
    result = process_evidence_decision(evidence, contract)
    assert not result["available"]
    assert result["selection"] == "native"
    assert "one admitted selection" in result["reason"]


def test_partial_resident_outcomes_cannot_authorize_an_unmeasured_subset(build_validation):
    contract, evidence = process_fixture()
    for row in evidence["campaigns"][0]["observations"]:
        if row["mode"] == "resident":
            row["invocations"][1].update(outcome="native", launches=0)
    result = process_evidence_decision(evidence, contract)
    assert result["selection"] == "native", result
    assert not result["available"]


@pytest.mark.parametrize("variant", ["native", "resident"])
def test_checked_tiled_group_capture_keeps_reached_prefix_and_complete_key(tmp_path, build_validation, variant):
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.tests.test_closed_selection import builder

    selected = builder(tmp_path / "original.f90", options=CompilerOptions(gpu_policy="auto"),
                       config=OffloadConfig(policy="auto", scope_execution="bounded",
                                            exact_window={"coverage": "original_selection", "calibration": variant,
                                                          "workloads": process_fixture()[0]["workloads"],
                                                          "build_contract": {"checked_fixture": True}}))
    outputs, report = selected.run()
    assert report["scope_count"] == 2
    source = outputs[report["sources"][str(tmp_path / "original.f90")]["replacement"]]
    assert "tx=tiles(1)\nty=tiles(2)\ntz=tiles(3)\ncoefficient=2.0_8\nif(order==4) then\nblock" in source
    for scope in report["scopes"]:
        record = scope["exact_window"]
        assert record["capture_variant"] == variant
        assert record["contract"]["protocol"] == PROTOCOL
        scalar = {item["resource"] for item in record["contract"]["key"] if item["property"] == "integer_value"}
        assert scalar == {"selected_groups::" + name for name in ("nx", "ny", "nz")} | {
            "selected_groups::advance::" + name for name in ("tx", "ty", "tz")}
        assert len([item for item in record["contract"]["key"] if item["property"] == "size"]) == 6
    if variant == "native":
        assert "TARGET :: a" not in source
        assert "!$omp parallel do private(i,j,k,bi,bj,bk)" in source
        assert not any("cudaGetDevice(" in text for path, text in outputs.items()
                       if path.startswith("exact/") and path.endswith(".cu"))
    else:
        assert source.index("fort_scope_plan_validate") < source.index("fort_selection_key(1)")
        assert "fort_team_run = .true." in source


@pytest.fixture(scope="module")
def workload_guard(tmp_path_factory):
    from compiler.offload.selection_workload import workload_source

    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("selection-workload")
    payload = (b"exact configuration\n" * 1000) + b"end"
    workload = {"argv": ["application", "--case"], "environment": {"SELECTION_TEST_ENV": "present"},
                "files": {"case.conf": sha256(payload).hexdigest()}}
    source = workload_source([{"key": [96], "workload": workload}]) + r'''
#include <iostream>
int main(){int64_t key[]={std::getenv("SELECTION_TEST_KEY")?128:96};
 std::cout<<fort_selection_workload_matches(key)<<"\n"<<selection_file_hash("case.conf")<<"\n";}
'''
    path = directory / "guard.cpp"
    path.write_text(source)
    subprocess.run([compiler, "-std=c++17", "-O2", str(path), "-o", str(directory / "guard")], check=True,
                   capture_output=True, text=True)
    return directory / "guard", payload


def test_runtime_workload_guard_observes_current_arguments_environment_and_full_bytes(tmp_path, workload_guard):
    binary, payload = workload_guard
    configuration = tmp_path / "case.conf"
    configuration.write_bytes(payload)
    environment = {**os.environ, "SELECTION_TEST_ENV": "present"}
    environment.pop("SELECTION_TEST_KEY", None)

    def run(*arguments, env=None):
        return subprocess.run([str(binary), *arguments], cwd=tmp_path, env=environment if env is None else env,
                              capture_output=True, text=True, check=True).stdout.splitlines()

    assert run("--case") == ["1", sha256(payload).hexdigest()]
    assert run("--other")[0] == "0"
    assert run("--case", env={**environment, "SELECTION_TEST_ENV": "changed"})[0] == "0"
    assert run("--case", env={**environment, "SELECTION_TEST_KEY": "128"})[0] == "0"
    stat = configuration.stat()
    configuration.write_bytes(payload[:-1] + b"X")
    os.utime(configuration, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert run("--case")[0] == "0"  # Content is rehashed; mtime is not authority.
    configuration.unlink()
    assert run("--case")[0] == "0"
    original = tmp_path / "original.conf"
    original.write_bytes(payload)
    configuration.symlink_to(original)
    assert run("--case")[0] == "0"


@pytest.mark.parametrize("payload", [b"", b"abc", bytes(range(256)) * 257], ids=["empty", "abc", "multiple-blocks"])
def test_generated_digest_matches_independent_sha256_across_padding_boundaries(tmp_path, workload_guard, payload):
    binary, _ = workload_guard
    Path(tmp_path / "case.conf").write_bytes(payload)
    result = subprocess.run([str(binary), "--case"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines()[1] == sha256(payload).hexdigest()


@pytest.mark.parametrize("variant", ["native", "resident"])
def test_generated_selection_observers_and_original_team_wrappers_compile(tmp_path, build_validation, variant):
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.tests.test_closed_selection import builder

    nvcc = shutil.which("nvcc")
    fortran = shutil.which("gfortran-14") or shutil.which("gfortran")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not all((nvcc, fortran, host)):
        pytest.skip("CUDA/C++/Fortran compile toolchains unavailable")
    selected = builder(tmp_path / "original.f90", options=CompilerOptions(gpu_policy="auto"),
                       config=OffloadConfig(policy="auto", scope_execution="bounded", exact_window={
                           "coverage": "original_selection", "calibration": variant,
                           "workloads": process_fixture()[0]["workloads"], "build_contract": {"checked_fixture": True}}))
    outputs, report = selected.run()
    generated = tmp_path / "generated"
    generated.mkdir()
    for relative, contents in outputs.items():
        path = generated / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    for scope in report["scopes"]:
        identity = scope["exact_window"]["contract_sha256"]
        relative = "exact/fort_selection_" + identity[:12] + ".cu"
        macros = [] if variant == "native" else ["-DFORT_EXACT_WINDOW_BUILD_" + identity + "=1"]
        # Each issued translation unit carries its own non-toolchain headers;
        # application recipes do not add an artifact-root include directory.
        command = [nvcc, "-std=c++17", "-O0", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp", *macros, "-c",
                   str(generated / relative), "-o", str(generated / (identity[:12] + ".o"))]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
    originals = [item["path"] for item in report["build_sources"] if item["language"] == "fortran"]
    ordered = sorted(originals, key=lambda path: (0 if path == "fort_scoped_team_observer.f90" else
                                                1 if path == "fort_scoped_memory.f90" else
                                                2 if path.startswith("entries/") else
                                                3 if path.startswith("regions/") else 4, path))
    for index, relative in enumerate(ordered):
        command = [fortran, "-O0", "-fopenmp", "-ffp-contract=off", "-I" + str(generated),
                   "-J" + str(generated), "-c", str(generated / relative), "-o", str(generated / f"fortran-{index}.o")]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


def test_native_and_resident_capture_bind_the_same_checked_semantics(tmp_path, build_validation):
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.tests.test_closed_selection import builder

    contracts = []
    for variant in ("native", "resident"):
        selected = builder(tmp_path / "original.f90", options=CompilerOptions(gpu_policy="auto"),
                           config=OffloadConfig(policy="auto", scope_execution="bounded", exact_window={
                               "coverage": "original_selection", "calibration": variant,
                               "workloads": process_fixture()[0]["workloads"], "build_contract": {"checked_fixture": True}}))
        _, report = selected.run()
        contracts.append([scope["exact_window"]["contract"] for scope in report["scopes"]])
    assert contracts[0] == contracts[1]
    assert json.loads(json.dumps(contracts[0])) == contracts[0]


def test_original_private_descriptor_costs_are_not_erased_by_lowering(tmp_path, build_validation):
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.tests.test_closed_selection import SOURCE, builder

    source = SOURCE.replace("subroutine advance(a,b)", "subroutine advance(a,b,n)\ninteger,intent(in)::n")
    source = source.replace("real(8):: coefficient", "real(8):: coefficient\nreal(8)::scratch(-2:n)")
    source = source.replace("private(i,j,k,bi,bj,bk)", "private(i,j,k,bi,bj,bk,scratch)")
    selected = builder(tmp_path / "original.f90", source, options=CompilerOptions(gpu_policy="auto"),
                       config=OffloadConfig(policy="auto", scope_execution="bounded", exact_window={
                           "coverage": "original_selection", "calibration": "native",
                           "workloads": process_fixture()[0]["workloads"], "build_contract": {"checked_fixture": True}}))
    outputs, report = selected.run()
    for scope in report["scopes"]:
        key = scope["exact_window"]["contract"]["key"]
        assert [item["property"] for item in key if item["resource"].endswith("::scratch")] == ["size", "lbound"]
    rewritten = outputs[report["sources"][str(tmp_path / "original.f90")]["replacement"]]
    assert "size(scratch,1,kind=c_int64_t)" in rewritten
    assert "lbound(scratch,1,kind=c_int64_t)" in rewritten


def test_inlined_helper_payload_work_does_not_disappear_from_the_key_proof(tmp_path, build_validation):
    from compiler.driver.options import CompilerOptions
    from compiler.ir import CompilationError
    from compiler.offload.config import OffloadConfig
    from compiler.tests.test_closed_selection import SOURCE, builder

    source = SOURCE.replace("coefficient*a(i,j,k)", "cell(a(i,j,k),coefficient)").replace("end module", """pure real(8) function cell(x,c) result(y)
real(8),intent(in)::x,c
y=c*x
if(x>0.0_8) y=y+1.0_8
end function
end module""")
    selected = builder(tmp_path / "original.f90", source, options=CompilerOptions(gpu_policy="auto"),
                       config=OffloadConfig(policy="auto", scope_execution="bounded", exact_window={
                           "coverage": "original_selection", "calibration": "resident",
                           "workloads": process_fixture()[0]["workloads"], "build_contract": {"checked_fixture": True}}))
    with pytest.raises(CompilationError, match="workload control depends on payload"):
        selected.run()
