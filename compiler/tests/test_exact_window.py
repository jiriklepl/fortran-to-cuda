"""Complete-window evidence never substitutes for legality or nearby shapes."""

import copy

import pytest

from compiler.offload.collective_calibration import normalize_fortran_options
from compiler.offload.exact_window import ENVIRONMENT, PROTOCOL, evidence_decision
from compiler.offload.exact_window_calibration import emit, fixture_source


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    source = tmp_path_factory.mktemp("exact-window") / "original.f90"
    source.write_text(fixture_source())
    outputs, report, record = emit(source, {"calibration": True})
    return source, outputs, report, record


def observations(contract, *, resident=0.1, native=1.0):
    identity = {
        "cpu_name": "fixture CPU",
        "gpu_uuid": "fixture",
        "gpu_name": "fixture",
        "cuda_runtime": 13000,
        "cuda_driver": 13000,
        "host_compiler": "fixture",
        "cuda_compiler": "13.0.0",
        "fortran_version": "fixture",
        "fortran_options": "-O3 -fopenmp -ffp-contract=off",
        "affinity": [0, 1, 2, 3],
        "threads": 4,
        "dynamic": 0,
        "proc_bind": 0,
        "environment": {name: None for name in ENVIRONMENT},
    }
    key = [98 if item["property"] == "size" else 0 if item["property"] == "lbound" else 96 for item in contract["key"]]
    cpu = {
        "cpu_name": identity["cpu_name"],
        "fortran_version": identity["fortran_version"],
        "fortran_options": normalize_fortran_options(identity["fortran_options"]),
        "team": [4, 0, 0],
        "affinity": identity["affinity"],
        "environment": identity["environment"],
    }
    rows = []
    for index in range(9):
        for turn in range(2):
            mode = ("native", "resident")[(index + turn) % 2]
            rows.append(
                {
                    "round": index,
                    "mode": mode,
                    "role": "fit" if index < 7 else "holdout",
                    "identity": identity,
                    "key": key,
                    "agreement": True,
                    "observed_cpu_identity": cpu,
                    "observed_key": key,
                    "fields_sha256": "a" * 64,
                    "seconds": native if mode == "native" else resident,
                }
            )
    return {"protocol": PROTOCOL, "contract": contract, "key": key, "identity": identity, "observations": rows}


def test_exact_complete_window_has_real_workers_and_current_host_native_reader(prepared):
    from hashlib import sha256

    _source, outputs, report, record = prepared
    assert len(record["contract"]["workers"]) == 3
    assert all(
        set(worker) == {"shared_entry.cu", "shared_interface.f90"} for worker in record["contract"]["workers"].values()
    )
    assert record["contract"]["common_functions_sha256"]
    assert (
        sha256(outputs["exact/common_functions.cuh"].encode()).hexdigest()
        == record["contract"]["common_functions_sha256"]
    )
    assert not report["scopes"][0]["boundaries"]
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    assert text.count("audit=sum(a(:,1,1))") == 1
    assert "fort_mode = 1" in text
    assert "fort_choose(" not in text
    assert text.index("fort_exact_key(1)") < text.index("fort_scope_create(")
    assert record["selection"] == "calibration"
    assert not record["available"]


def test_only_independently_profitable_complete_observations_select_resident(prepared):
    contract = prepared[-1]["contract"]
    accepted = evidence_decision(observations(contract), contract)
    assert accepted["available"]
    assert accepted["selection"] == "resident"
    expensive = evidence_decision(observations(contract, resident=2.0), contract)
    assert expensive["available"]
    assert expensive["selection"] == "native"
    assert not evidence_decision(None, contract)["available"]


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "header",
        "rows",
        "order",
        "agreement",
        "time",
        "control",
        "placement",
        "native_cpu",
        "native_key",
        "fields",
    ],
)
def test_invalid_or_stale_observations_cannot_enable_a_window(prepared, change):
    contract = prepared[-1]["contract"]
    evidence = copy.deepcopy(observations(contract))
    if change == "source":
        evidence["contract"]["source_graph"] = "changed"
    elif change == "header":
        evidence["contract"]["common_functions_sha256"] = "changed"
    elif change == "rows":
        evidence["observations"].pop()
    elif change == "order":
        evidence["observations"][0]["mode"] = "resident"
    elif change == "agreement":
        evidence["observations"][0]["agreement"] = False
    elif change == "time":
        evidence["observations"][0]["seconds"] = float("nan")
    elif change == "control":
        evidence["key"] = [1]
    elif change == "placement":
        evidence["identity"]["dynamic"] = 1
    elif change == "native_cpu":
        evidence["observations"][0]["observed_cpu_identity"] = {}
    elif change == "native_key":
        evidence["observations"][0]["observed_key"] = [1]
    elif change == "fields":
        evidence["observations"][0]["fields_sha256"] = "b" * 64
    result = evidence_decision(evidence, contract)
    assert not result["available"]
    assert result["selection"] == "native"


def test_measured_auto_uses_one_guard_and_preserves_native_without_estimates(prepared):
    source, _outputs, _report, record = prepared
    evidence = observations(record["contract"])
    outputs, report, auto = emit(source, {"evidence": evidence})
    assert auto["contract"] == record["contract"]
    assert auto["selection"] == "resident"
    assert auto["available"]
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    assert "fort_choose(" not in text
    assert "fort_exact_window_" in text
    selector = next(value for name, value in outputs.items() if name.startswith("exact/"))
    assert "FORT_EXACT_CALIBRATION_MODE" not in selector
    assert "key[24]!=96LL" in selector
    assert "fort_scope_team_fortran_compatible_v1" in selector
    assert "#if !defined(FORT_EXACT_WINDOW_BUILD_" in selector
    assert report["scopes"][0]["estimate_available"]
    outputs, _report, missing = emit(source, {})
    assert missing["selection"] == "native"
    assert not missing["available"]
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    assert "= .false." in text
    assert not any(name.startswith("exact/") for name in outputs)


@pytest.mark.parametrize(
    "replacement",
    [
        ("do k=1,n", "do k=1,int(a(0,0,0))"),
        ("do k=1,n", "v=a(0,0,0)\ndo k=1,int(v)"),
        ("d(i,j,k)=b(i-1,j,k)", "d(i,j,k)=b(int(a(i,j,k)),j,k)"),
        ("audit=sum(a(:,1,1))", "audit=sum(a(:int(a(0,0,0)),0,0))"),
        ("audit=sum(a(:,1,1))", "audit=sqrt(a(0,0,0))"),
        ("audit=sum(a(:,1,1))", "audit=sum(a(:,0,0),mask=a(:,0,0)>0)"),
        ("audit=sum(a(:,1,1))", "if(n>4) audit=sum(a(:,1,1))"),
        ("do k=1,n", "do k=1,j"),
        ("audit=sum(a(:,1,1))", "audit=a(i,0,0)"),
    ],
)
def test_uncovered_payload_control_or_primitive_is_not_admitted(tmp_path, replacement):
    from hashlib import sha256

    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.scopes.source import ScopeBuilder
    from compiler.tests.test_source_scopes import FACT

    source = tmp_path / "declined.f90"
    source.write_text(fixture_source().replace(*replacement, 1))
    facts = {
        "schema_version": 1,
        "participation": "serial",
        "sources": {str(source): sha256(source.read_bytes()).hexdigest()},
        "captures": {"argument::" + name: FACT for name in ("a", "b", "d", "out")},
    }
    outputs, report = ScopeBuilder(
        [source],
        "exact_fixture::advance",
        facts=facts,
        options=CompilerOptions(),
        config=OffloadConfig(policy="auto", scope_execution="reached", exact_window={}),
    ).run()
    assert not any("exact_window" in scope for scope in report["scopes"])
    assert not any(name.startswith("exact/") for name in outputs)
    # A request is not admission: every alternative remains ordinary AUTO or
    # statically native, never a selector-free forced resident scope.
    for scope in report["scopes"]:
        assert scope.get("mode") in {None, "auto"}
        if scope.get("automatic_preflight"):
            assert scope["automatic_preflight"]["selection"] == "native"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fortran_options", None),
        ("cpu_name", None),
        ("affinity", None),
        ("environment", []),
        ("environment", {"OMP_NUM_THREADS": 4}),
    ],
)
def test_malformed_execution_identity_fails_closed(prepared, field, value):
    contract = prepared[-1]["contract"]
    evidence = copy.deepcopy(observations(contract))
    evidence["identity"][field] = value
    decision = evidence_decision(evidence, contract)
    assert decision["selection"] == "native"
    assert not decision["available"]


@pytest.mark.parametrize(("automatic_seconds", "model_miss"), [(0.1, False), (2.0, True)])
def test_actual_production_validation_reports_model_miss_without_revising_decision(
    prepared, tmp_path, monkeypatch, automatic_seconds, model_miss
):
    from hashlib import sha256
    from types import SimpleNamespace

    from compiler.offload import exact_window_calibration as producer

    evidence = observations(prepared[-1]["contract"])
    payload = b"complete float64 fields checked independently"
    for row in evidence["observations"]:
        row["fields_sha256"] = sha256(payload).hexdigest()
    monkeypatch.setattr(producer, "cpu_identity", lambda: "fixture CPU")
    monkeypatch.setattr(producer.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})

    def measured(command, *, cwd, environment):
        assert "FORT_RUNTIME_TRACE" not in environment
        assert "FORT_EXACT_CALIBRATION_MODE" not in environment
        (cwd / "fields.bin").write_bytes(payload)
        seconds = automatic_seconds if command[0] == "automatic" else 1.0
        return SimpleNamespace(
            stdout=f"WINDOW_SECONDS {seconds}\nFORTRAN_VERSION fixture\n"
            "FORTRAN_OPTIONS -O3 -fopenmp -ffp-contract=off\nCPU_TEAM 4 0 0\n"
            "NATIVE_KEY " + " ".join(map(str, evidence["key"])) + "\n",
            stderr="",
        )

    monkeypatch.setattr(producer, "_run", measured)
    decision = {"selection": "resident"}
    result = producer.validate_production("automatic", "native", tmp_path, {}, 96, evidence, decision)
    assert result["model_miss"] is model_miss
    assert result["automatic_seconds"] == automatic_seconds
    assert len(result["observations"]) == 18
    assert (tmp_path / "production-observations.json").is_file()
    assert decision == {"selection": "resident"}


@pytest.mark.native
def test_emitted_identity_quoting_round_trips_special_characters(prepared, tmp_path):
    import json
    import shutil
    import subprocess

    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler unavailable")
    selector = next(value for name, value in prepared[1].items() if name.startswith("exact/") and name.endswith(".cu"))
    quoted = selector[selector.index("static std::string quoted(") : selector.index('\nextern "C"')]
    original = 'quote" path\\line\x01\t\n\r'
    source = tmp_path / "quote.cpp"
    source.write_text(
        "#include <string>\n#include <sstream>\n#include <iomanip>\n#include <iostream>\n"
        + quoted
        + "\nint main(){const char value[]={"
        + ",".join(str(ord(character)) for character in original)
        + "};std::cout<<quoted(std::string(value,sizeof(value)));}\n"
    )
    binary = tmp_path / "quote"
    subprocess.run(
        [cxx, "-std=c++17", "-Werror=multichar", str(source), "-o", str(binary)], check=True, capture_output=True
    )
    actual = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout
    assert json.loads(actual) == original
