"""Complete-process evidence never prices a warm call as a cold application."""

import json
import os
import shutil
import subprocess
from contextlib import contextmanager
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

    class CheckedBuilds(list):
        pass

    checked = CheckedBuilds()
    checked.numerical = []
    verified = set()

    @contextmanager
    def session():
        verified.clear()
        yield

    def validate(value, *, contract, mode, selection_contract_identity=None):
        if value != {"contract_identity": contract["identity"], "variant": mode, "checked_fixture": True}:
            raise ValueError("executed build attestation differs")
        if mode not in verified:
            verified.add(mode)
            checked.append(mode)
        return value

    module.application_build_validation = session
    module.validate_application_build_observation = validate
    module.require_application_build_contract = lambda requested, **identity: {
        "identity": digest(identity), "checked_fixture": True, **identity}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    # These tests exercise cost/identity orchestration. Raw-file numerical and
    # relink authentication have independent tests against their real helpers.
    numerical = ModuleType("compiler.offload.selection_validation")

    def normalize(requested, *, build_contract):
        if requested != {"checked_policy_fixture": True}:
            raise ValueError("numerical policy differs")
        return {**requested, "identity": digest(requested)}

    def validate_numerics(campaign, **arguments):
        if campaign.get("numerical_validation") != {"path": "checked-comparison.json", "sha256": "e" * 64}:
            raise ValueError("numerical payload verification failed")
        checked.numerical.append(arguments)
        return {"receipt_sha256": "e" * 64, "policy_identity": arguments["contract"]["validation_policy"]["identity"]}

    numerical.normalize_policy = normalize
    numerical.validate_campaign_numerics = validate_numerics
    monkeypatch.setitem(sys.modules, numerical.__name__, numerical)
    return checked


def process_fixture(*, resident=.1, native=1.0):
    bundle = {"schema_version": 1, "entry": "fixture::advance", "groups": ["s"]}
    bundle["identity"] = digest(bundle)
    contract = {"protocol": PROTOCOL, "key": [{"resource": "n", "property": "integer_value"}],
                "threads": 4, "selection_identity": "a" * 64, "selection_bundle": bundle,
                "build_contract": {"identity": "b" * 64},
                "outcome_contract": {"successful_completion": True, "fallbacks": 0, "requires_process_gpu_launch": True},
                "validation_policy": {"checked_policy_fixture": True, "identity": digest({"checked_policy_fixture": True})}}
    campaign = observations(contract, resident=resident, native=native)
    campaign["protocol"] = PROTOCOL
    workload = {"argv": ["application", "case"], "environment": {"EXAMPLE": None},
                "files": {"case.conf": "c" * 64}, "completion": {"steps": 109, "end": 1.0}}
    workload["identity"] = digest(workload)
    campaign["workload"] = workload
    contract["workloads"] = [workload]
    campaign["selection_bundle"] = bundle
    campaign["numerical_validation"] = {"path": "checked-comparison.json", "sha256": "e" * 64}
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
                               "launches": int(row["mode"] == "resident"), "complete": True, "fallbacks": 0}
                              for ordinal, value in enumerate((96, 128, 96))]
    return contract, {"protocol": PROTOCOL, "campaigns": [campaign]}


def test_complete_process_pairs_bind_both_builds_and_all_observed_keys(build_validation):
    contract, evidence = process_fixture()
    result = process_evidence_decision(evidence, contract)
    assert result["selection"] == "resident", result
    assert [entry["key"] for entry in result["accepted_keys"]] == [[96], [128]]
    assert len(build_validation) == 2
    assert build_validation.count("native") == 1
    assert len(build_validation.numerical) == 1
    assert build_validation.numerical[0]["key_sequence"] == [(96,), (128,), (96,)]
    assert result["campaigns"][0]["invocations"] == 3
    assert "no future invocation-count guarantee" in result["applicability"]


def test_process_numerical_authority_is_separate_from_legacy_flags_and_hashes(build_validation):
    contract, evidence = process_fixture()
    for index, row in enumerate(evidence["campaigns"][0]["observations"]):
        row["agreement"] = False
        row["fields_sha256"] = digest(index)
    result = process_evidence_decision(evidence, contract)
    assert result["available"], result
    assert result["selection"] == "resident", result
    assert len(build_validation.numerical) == 1
    evidence["campaigns"][0].pop("numerical_validation")
    for row in evidence["campaigns"][0]["observations"]:
        row["agreement"], row["fields_sha256"] = True, "f" * 64
    result = process_evidence_decision(evidence, contract)
    assert not result["available"], result
    assert result["selection"] == "native", result
    assert "numerical payload verification failed" in result["reason"]


def test_build_session_final_recheck_precedes_placement_authority(build_validation, monkeypatch):
    import sys

    @contextmanager
    def changed_build():
        yield
        raise ValueError("build input changed before decision completed")

    monkeypatch.setattr(sys.modules["compiler.offload.application_build"], "application_build_validation", changed_build)
    contract, evidence = process_fixture()
    result = process_evidence_decision(evidence, contract)
    assert not result["available"], result
    assert result["selection"] == "native", result
    assert "build input changed" in result["reason"]


@pytest.mark.parametrize("mutation", [
    "startup", "completion", "sequence", "missing_native", "fake_gpu", "mixed_gpu",
    "build", "bundle", "observed_bundle", "nan_time", "bool_launches", "missing_first",
    "unfinished_interval", "missing_completion", "bool_fallbacks", "no_gpu_work",
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
        row["invocations"][0].update(launches=2, fallbacks=1)
    elif mutation == "unfinished_interval":
        row["invocations"][0]["complete"] = False
    elif mutation == "missing_completion":
        row["invocations"][0].pop("complete")
    elif mutation == "bool_fallbacks":
        row["invocations"][0]["fallbacks"] = False
    elif mutation == "no_gpu_work":
        for invocation in row["invocations"]:
            invocation["launches"] = 0
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


def test_empirical_interval_accepts_conditional_work_without_fixed_launch_prediction(build_validation):
    contract, evidence = process_fixture()
    contract["completion_manifest"] = {"native_interruptions": ["original correction"],
                                       "conditional_gpu_operations": ["producer", "diffusion", "accumulation"]}
    for row in evidence["campaigns"][0]["observations"]:
        if row["mode"] == "resident":
            for invocation, launches in zip(row["invocations"], (3, 0, 7), strict=True):
                invocation["launches"] = launches
    decision = process_evidence_decision(evidence, contract)
    assert decision["selection"] == "resident", decision
    assert decision["available"]


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
                                                          "build_contract": {"checked_fixture": True},
                                                          "validation_policy": {"checked_policy_fixture": True}}))
    outputs, report = selected.run()
    assert report["scope_count"] == 2
    source = outputs[report["sources"][str(tmp_path / "original.f90")]["replacement"]]
    assert "tx=tiles(1)\nty=tiles(2)\ntz=tiles(3)\ncoefficient=2.0_8\nif(order==4) then\nblock" in source
    for scope in report["scopes"]:
        record = scope["exact_window"]
        assert record["capture_variant"] == variant
        assert record["contract"]["protocol"] == PROTOCOL
        scalar = {item["resource"] for item in record["contract"]["key"] if item["property"] == "integer_value"}
        assert {"selected_groups::advance::" + name for name in ("tx", "ty", "tz")} <= scalar
        assert "entry_projection_identity" in record["contract"]
        assert "internal native work is measured, not predicted" in record["contract"]["applicability"]
        assert len([item for item in record["contract"]["key"] if item["property"] == "size"]) == 6
    if variant == "native":
        assert "TARGET :: a" not in source
        assert "!$omp parallel do private(i,j,k,bi,bj,bk)" in source
        assert not any("cudaGetDevice(" in text for path, text in outputs.items()
                       if path.startswith("exact/") and path.endswith(".cu"))
    else:
        assert source.index("fort_interval_key(1)") < source.index("fort_scope_plan_validate")
        assert source.index("fort_scope_plan_validate") < source.index("_choose() == 0")
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


def test_interval_observer_latches_root_key_and_records_later_fallback(tmp_path, monkeypatch):
    """Exercise emitted observer state with a fake selector, not GPU evidence."""
    from compiler.offload.selection_cost import _observer_source

    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    contract, _evidence = process_fixture()
    (tmp_path / "case.conf").write_text("fixed input\n")
    contract["workloads"][0]["files"]["case.conf"] = sha256(b"fixed input\n").hexdigest()

    def selector(name, _record, *, observer):
        return r'''
#include <omp.h>
#include <sched.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include "scoped_runtime.h"
namespace generated_kernels { namespace offload {
static std::string cpu_identity(){return "fixture CPU";}
}}
static std::string quoted(const std::string &s){return "\""+s+"\"";}
static int selector_calls=0;
static void OBSERVER(const int64_t*,const std::string&);
extern "C" int SELECT(const int64_t *key,const char*,const char*) {
 ++selector_calls;if(key[0]!=96||omp_get_level()!=1||omp_get_num_threads()!=4)return 0;
 OBSERVER(key,"{}");return 1;
}
extern "C" int fort_scope_stats_get(fort_scope_t,fort_scope_stats *stats) {
 *stats=fort_scope_stats{};stats->launches=3;return 0;
}
'''.replace("OBSERVER", observer).replace("SELECT", name)

    monkeypatch.setattr("compiler.offload.exact_window.selector_source", selector)
    record = {"contract": contract, "capture": True, "capture_variant": "resident", "expressions": ["n"]}
    source = _observer_source("observed", record) + r'''
int main() {
 int64_t key[]={96};observed_begin();observed_key(key,"compiler","flags");key[0]=999;
 int selected=0;
 #pragma omp parallel num_threads(4) shared(selected)
 {
  #pragma omp master
  selected=observed_choose();
 }
 if(selected!=1||observed_choose()!=1||selector_calls!=1)return 10;
 observed_close(1);observed_fallback();
 #pragma omp parallel num_threads(4) shared(selected)
 {
  #pragma omp master
  selected=observed_choose();
 }
 if(selected!=0||selector_calls!=1)return 11;
 observed_end();return 0;
}
'''
    path = tmp_path / "observer.cpp"
    path.write_text(source)
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    subprocess.run([compiler, "-std=c++17", "-fopenmp", "-I" + str(runtime), str(path),
                    "-o", str(tmp_path / "observer")], check=True, capture_output=True, text=True)
    environment = {key: value for key, value in os.environ.items() if key != "EXAMPLE"}
    result = subprocess.run([str(tmp_path / "observer"), "case"], cwd=tmp_path, env=environment,
                            check=True, capture_output=True, text=True)
    row = json.loads(result.stderr.removeprefix("FORT_SELECTION_OBSERVATION "))
    assert row["key"] == [96]
    assert row["launches"] == 3
    assert row["complete"] is True
    assert row["fallbacks"] == 1
    assert row["outcome"] == "fallback"


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
                           "workloads": process_fixture()[0]["workloads"], "build_contract": {"checked_fixture": True},
                           "validation_policy": {"checked_policy_fixture": True}}))
    outputs, report = selected.run()
    generated = tmp_path / "generated"
    generated.mkdir()
    for relative, contents in outputs.items():
        path = generated / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    for scope in report["scopes"]:
        assert scope["exact_window"]["contract"]["protocol"] == PROTOCOL
        identity = scope["exact_window"]["contract_sha256"]
        relative = "exact/fort_selection_" + identity[:12] + ".cu"
        assert relative in outputs
        assert "exact/fort_exact_window_" + identity[:12] + ".cu" not in outputs
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
                               "workloads": process_fixture()[0]["workloads"], "build_contract": {"checked_fixture": True},
                               "validation_policy": {"checked_policy_fixture": True}}))
        _, report = selected.run()
        contracts.append([scope["exact_window"]["contract"] for scope in report["scopes"]])
    assert contracts[0] == contracts[1]
    assert json.loads(json.dumps(contracts[0])) == contracts[0]


def test_cross_procedure_interval_uses_one_native_and_resident_cost_contract(tmp_path, build_validation):
    from compiler.offload.config import OffloadConfig
    from compiler.tests.test_interval_draft import generate

    contracts = []
    for variant in ("native", "resident"):
        config = OffloadConfig(policy="auto", scope_execution="reached", exact_window={
            "coverage": "checked_interval", "calibration": variant,
            "workloads": process_fixture()[0]["workloads"], "build_contract": {"checked_fixture": True},
            "validation_policy": {"checked_policy_fixture": True}})
        path, _builder, outputs, report = generate(tmp_path, config=config)
        assert report["scope_count"] == 1, report.get("lexical_owner_attempt")
        scope, = report["scopes"]
        plan = scope["checked_execution_plan"]
        assert len(plan["procedures"]) == 3
        assert plan["numerical_invocations"] == 3
        assert not scope["boundaries"]
        record = scope["exact_window"]
        contract = record["contract"]
        contracts.append(contract)
        bundle = contract["selection_bundle"]
        assert {group["selection_identity"] for group in bundle["groups"]} == {contract["selection_identity"]}
        assert contract["selection_identity"] == digest(contract["selection"])
        assert record["capture_variant"] == variant
        assert len(contract["completion_manifest"]["operations"]) == 3
        assert {item["resource"] for item in contract["key"]} == {
            "argument::n", *("interval_fields::" + name for name in ("old", "viscosity", "star", "out"))}
        source = outputs[report["sources"][str(path)]["replacement"]]
        assert source.count("visits=visits+1") == 1
        assert "v(0)=v(0)+a(0)" in source
        if variant == "native":
            assert source.count("v(0)=v(0)+a(0)") == 1
            assert "call producer(old,viscosity,n)" in source
            assert "call momentum(old,viscosity,star,out,n,correcting)" in source
            assert not any(name.startswith("entries/") for name in outputs)
        else:
            assert source.index("fort_interval_key(1)") < source.index("_choose() == 0")
            assert source.count("_choose() == 0") == 3
            assert "_fallback()" in source
            assert "_end()" in source
    assert contracts[0] == contracts[1]


def test_original_private_work_remains_in_checked_empirical_alternative(tmp_path, build_validation):
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.tests.test_closed_selection import SOURCE, builder

    source = SOURCE.replace("subroutine advance(a,b)", "subroutine advance(a,b,n)\ninteger,intent(in)::n")
    source = source.replace("real(8):: coefficient", "real(8):: coefficient\nreal(8)::scratch(-2:n)")
    source = source.replace("private(i,j,k,bi,bj,bk)", "private(i,j,k,bi,bj,bk,scratch)")
    selected = builder(tmp_path / "original.f90", source, options=CompilerOptions(gpu_policy="auto"),
                       config=OffloadConfig(policy="auto", scope_execution="bounded", exact_window={
                           "coverage": "original_selection", "calibration": "native",
                           "workloads": process_fixture()[0]["workloads"], "build_contract": {"checked_fixture": True},
                           "validation_policy": {"checked_policy_fixture": True}}))
    outputs, report = selected.run()
    for scope in report["scopes"]:
        contract = scope["exact_window"]["contract"]
        assert any(item["resource"].endswith("::scratch") for team in contract["original_participation"]
                   for item in team["original_private_arrays"])
        assert contract["entry_projection_identity"]
    rewritten = outputs[report["sources"][str(tmp_path / "original.f90")]["replacement"]]
    assert "private(i,j,k,bi,bj,bk,scratch)" in rewritten


def test_internal_payload_branch_is_measured_without_entry_re_evaluation(tmp_path, build_validation):
    from compiler.driver.options import CompilerOptions
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
                           "workloads": process_fixture()[0]["workloads"], "build_contract": {"checked_fixture": True},
                           "validation_policy": {"checked_policy_fixture": True}}))
    outputs, report = selected.run()
    assert report["scope_count"] == 2
    for scope in report["scopes"]:
        contract = scope["exact_window"]["contract"]
        assert "internal native work is measured, not predicted" in contract["applicability"]
        assert not any(item["property"] == "payload" for item in contract["key"])
    source = outputs[report["sources"][str(tmp_path / "original.f90")]["replacement"]]
    assert "cell(" in source  # Original guarded native alternative remains present.
    assert not any("cell(" in text for path, text in outputs.items() if path.startswith("exact/"))
