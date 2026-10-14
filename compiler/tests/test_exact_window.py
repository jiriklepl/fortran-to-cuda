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
        "thread_limit": 2147483647,
        "max_active_levels": 1,
        "environment": {name: None for name in ENVIRONMENT},
    }
    key = [98 if item["property"] == "size" else 0 if item["property"] == "lbound" else 96 for item in contract["key"]]
    cpu = {
        "cpu_name": identity["cpu_name"],
        "fortran_version": identity["fortran_version"],
        "fortran_options": normalize_fortran_options(identity["fortran_options"]),
        "team": [4, 0, 0, 2147483647, 1],
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
    assert record["contract"]["checked_execution_identity"] == report["scopes"][0]["checked_execution_plan"]["execution_identity"]
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


@pytest.mark.parametrize("original_teams", [False, True], ids=["serial", "original-teams"])
def test_independent_capture_and_production_share_portable_execution_identity(tmp_path, original_teams):
    source = tmp_path / "original.f90"
    source.write_text(fixture_source(original_teams=original_teams))
    _outputs, capture_report, capture = emit(source, {"calibration": True})
    _outputs, repeat_report, repeated = emit(source, {"calibration": True})
    execution = capture_report["scopes"][0]["checked_execution_plan"]["execution_identity"]
    assert repeat_report["scopes"][0]["checked_execution_plan"]["execution_identity"] == execution
    assert repeated["contract"] == capture["contract"]
    # Synthetic profitable rows test the contract handoff, never performance.
    _outputs, production_report, production = emit(source, {"evidence": observations(capture["contract"])})
    assert production_report["scopes"][0]["checked_execution_plan"]["execution_identity"] == execution
    assert production["contract"]["checked_execution_identity"] == execution
    assert production["selection"] == "resident"


def test_complete_window_fingerprint_is_portable_across_process_hash_seeds(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    for mode in ("serial", "teams"):
        (tmp_path / (mode + ".f90")).write_text(fixture_source(original_teams=mode == "teams"))
    program = """
import json
from pathlib import Path
import sys
from compiler.offload.exact_window_calibration import emit
root = Path(sys.argv[1])
records = {}
for mode in ('serial', 'teams'):
    _, report, exact = emit(root / (mode + '.f90'), {'calibration': True})
    plan = report['scopes'][0]['checked_execution_plan']
    records[mode] = {'execution_identity': plan['execution_identity'],
                     'contract_sha256': exact['contract_sha256']}
print(json.dumps(records, sort_keys=True))
"""
    records = []
    for seed in (1, 91237):
        result = subprocess.run([sys.executable, "-c", program, str(tmp_path)],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            check=True, capture_output=True, text=True)
        records.append(json.loads(result.stdout))
    assert records[0] == records[1]


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "execution",
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
    elif change == "execution":
        evidence["contract"]["checked_execution_identity"] = "0" * 64
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


@pytest.mark.parametrize("combined", [True, False])
def test_complete_original_teams_have_independent_source_bound_calibration(tmp_path, combined):
    source = tmp_path / "original.f90"
    text = fixture_source(original_teams=True)
    if not combined:
        text = text.replace(
            "!$omp parallel do private(i,j,k,v) schedule(static)",
            "!$omp parallel private(i,j,k,v)\n!$omp do schedule(static)",
        )
        text = text.replace(
            "!$omp parallel do private(i,j,k) schedule(static)",
            "!$omp parallel private(i,j,k)\n!$omp do schedule(static)",
        )
        text = text.replace("!$omp end parallel do", "!$omp end do\n!$omp end parallel")
    source.write_text(text)
    _outputs, _report, calibrated = emit(source, {"calibration": True})
    participation = calibrated["contract"]["original_participation"]
    assert participation["kind"] == "original_joined_teams"
    assert len(participation["teams"]) == 3
    assert all(team["proof_identity"] and team["selected_original_nodes"] for team in participation["teams"])
    evidence = observations(calibrated["contract"])
    outputs, _report, automatic = emit(source, {"evidence": evidence})
    assert automatic["selection"] == "resident"
    generated = next(value for name, value in outputs.items() if name.startswith("sources/"))
    assert "fort_mode = 1" in generated
    assert "fort_choose(" not in generated
    assert source.read_text() == text
    # Serial evidence cannot authorize a team, even for identical descriptors.
    evidence = copy.deepcopy(evidence)
    evidence["contract"]["original_participation"] = {"kind": "serial", "teams": []}
    assert evidence_decision(evidence, calibrated["contract"])["selection"] == "native"


@pytest.mark.parametrize(("field", "value"), [("thread_limit", 2), ("max_active_levels", 0)])
def test_insufficient_original_team_runtime_limits_do_not_authorize_placement(prepared, field, value):
    contract = prepared[-1]["contract"]
    evidence = observations(contract)
    evidence["identity"][field] = value
    decision = evidence_decision(evidence, contract)
    assert not decision["available"]
    assert decision["selection"] == "native"


def test_pre_team_protocol_observations_are_stale(prepared):
    contract = prepared[-1]["contract"]
    evidence = observations(contract)
    evidence["protocol"] = "complete-original-window-cold-v1"
    assert not evidence_decision(evidence, contract)["available"]


def test_native_automatic_private_descriptors_key_whole_window_cost_once(tmp_path):
    source = tmp_path / "original.f90"
    text = fixture_source(original_teams=True).replace(
        "implicit none\ncontains", "implicit none\ninteger::extent=3\ncontains"
    ).replace("real(8)::v", "real(8)::v\nreal(8)::scratch(-2:extent),unused(-1:extent)")
    interruption = """!$omp parallel do private(i,scratch,unused) schedule(static)
do i=1,n
scratch=a(i,1,1)
b(i,1,1)=b(i,1,1)+sum(scratch)
enddo
!$omp end parallel do
"""
    text = text.replace("audit=sum(a(:,1,1))", interruption * 2 + "audit=sum(a(:,1,1))")
    source.write_text(text)
    outputs, report, calibrated = emit(source, {"calibration": True})
    contract = calibrated["contract"]
    private_key = [item for item in contract["key"] if item["resource"].startswith("exact_fixture::advance::")]
    assert private_key == [
        {"resource": "exact_fixture::advance::" + name, "property": inquiry, "axis": 1}
        for name in ("scratch", "unused")
        for inquiry in ("size", "lbound")
    ]
    generated = next(value for name, value in outputs.items() if name.startswith("sources/"))
    for name in ("scratch", "unused"):
        for inquiry in ("size", "lbound"):
            assert f"{inquiry}({name},1,kind=c_int64_t)" in generated
    assert "int(extent,c_int64_t)" not in generated
    assert all(not item["resource"].endswith(("::scratch", "::unused"))
               for item in report["scopes"][0]["resource_bindings"]["resources"])

    evidence = observations(contract)
    evidence["key"][-4:] = [6, -2, 5, -1]
    outputs, _report, automatic = emit(source, {"evidence": evidence})
    assert automatic["selection"] == "resident"
    selector = next(value for name, value in outputs.items() if name.endswith(".cu") and name.startswith("exact/"))
    for index, value in enumerate((6, -2, 5, -1), len(contract["key"]) - 4):
        assert f"key[{index}]!={value}LL" in selector
    # A changed automatic extent cannot borrow observations for the old key.
    incompatible = copy.deepcopy(evidence)
    incompatible["observations"][0]["observed_key"] = [*incompatible["key"][:-4], 7, -2, 6, -1]
    assert not evidence_decision(incompatible, contract)["available"]
    assert source.read_text() == text


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
            "FORTRAN_OPTIONS -O3 -fopenmp -ffp-contract=off\nCPU_TEAM 4 0 0 2147483647 1\n"
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
