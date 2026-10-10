"""Independent startup proof cannot bless failed numerical classes."""
import json
import math
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from compiler.offload.cpu_protocol_calibration import (
    CPU_BACKENDS,
    GENERATED_CONTROL_HEADER,
    HOST_FLAGS,
    MEASURED_BACKENDS,
    MEMORY_ACCESS_CLASS,
    STARTUP_SIZES,
    TEAM_PROOF_SOURCE,
    calibrate_cpu_protocol,
    cpu_protocol_costs,
    cpu_protocol_identity,
    generated_control_identity,
    generated_cpu_control_source,
    parse_cpu_protocol_measurements,
    profile_from_cpu_protocol_measurements,
    validate_cpu_protocol,
)
from compiler.offload.numerical_calibration import (
    NumericalCalibrationError,
    memory_compute_seconds,
    native_fixture_source,
)
from compiler.offload.schedule_calibrate import measurement_environment
from compiler.tests.test_numerical_calibration_v2 import calibrated_v2, observations_v2


def samples(seconds):
    repetitions = math.ceil(.21 / seconds)
    return [{"batch": i, "repetitions": repetitions, "elapsed_seconds": repetitions * seconds,
             "wall_seconds": repetitions * seconds} for i in range(7)]


def evidence(base=None):
    base = base or calibrated_v2()
    identity = {"cpu_threads": base["cpu_threads"], "precision_bits": base["precision_bits"],
                "cpu_affinity": deepcopy(base["numerical"]["identity"]["cpu_affinity"]),
                "actual_team_threads": base["cpu_threads"], "thread_limit": (1 << 31) - 1,
                "omp_wait_policy": None, "gomp_spincount": None,
                "omp_dynamic": False, "omp_proc_bind": "false", "proof_backend": "GNU GOMP_parallel wrap v1",
                "fortran": deepcopy(base["numerical"]["identity"]["fortran"]),
                "host": {"compiler_version": base["toolchain"]["host_cxx_version"], "semantic_options": list(HOST_FLAGS)},
                "timed_objects": {"native": "a" * 64, "driver": "b" * 64}}
    records, proofs = [], []
    for backend in MEASURED_BACKENDS:
        for n in STARTUP_SIZES:
            records.append({"kind": "cpu_startup_cost_v1", "backend": backend, "items": n,
                            "agreement_passed": True, "samples": samples(2e-6)})
            proofs.append({"kind": "cpu_team_proof_v1", "backend": backend, "items": n,
                           "agreement_passed": True, "parallel_entries": 1, "wrong_team_visits": 0,
                           "thread_visits": [1] * base["cpu_threads"]})
    for row in base["numerical"]["measurements"]:
        if row.get("family") == "memory_v2" and row.get("backend") in CPU_BACKENDS:
            records.append({key: deepcopy(value) for key, value in row.items()
                            if key not in {"family", "kind"}} | {"kind": "cpu_memory_cost_v1"})
    return records, identity, proofs


def corrected(base=None):
    base = base or calibrated_v2()
    records, identity, proofs = evidence(base)
    return profile_from_cpu_protocol_measurements(base, records, identity, proofs)


def test_startup_is_independently_measured_and_numerical_v2_is_immutable():
    original = calibrated_v2()
    before = deepcopy(original)
    result = corrected(original)
    assert original == before
    assert result["numerical"] == before["numerical"]
    assert result["cpu_execution_protocol"]["calibration"]["coefficients_refitted"] is False
    assert result["cpu_execution_protocol"]["generated_control_sha256"] == generated_control_identity(64)
    for backend in CPU_BACKENDS:
        costs = cpu_protocol_costs(result, backend, access_class=MEMORY_ACCESS_CLASS)
        assert costs["fixed_seconds"] == pytest.approx(0 if backend == "native_serial" else 2e-6)
        assert result["cpu_execution_protocol"]["backends"][backend]["memory"]["status"] == "accepted"
        assert all(row["status"] == "accepted" for row in
                   result["cpu_execution_protocol"]["backends"][backend]["workload_validation"].values())


@pytest.mark.parametrize("precision", [32, 64])
def test_generated_control_uses_actual_worker_renderer_and_cyclic_team_protocol(precision, monkeypatch):
    from compiler.emission.cuda.offload import _cpu_worker
    from compiler.offload import cpu_protocol_calibration as protocol

    calls = []

    def render(unit, signature, name):
        calls.append((unit, signature, name))
        return _cpu_worker(unit, signature, name)

    monkeypatch.setattr(protocol, "_cpu_worker", render)
    generated_cpu_control_source.cache_clear()
    header = generated_cpu_control_source(precision)
    unit, signature, name = calls[0]
    assert len(calls) == 1
    assert "\n".join(_cpu_worker(unit, signature, name)) in header
    assert "for (std::size_t flat = tid; flat < total; flat += team)" in header
    assert "#pragma omp parallel num_threads(threads)" in header
    assert "omp_get_thread_num(), omp_get_num_threads()" in header
    assert "#pragma omp parallel for" not in header
    assert f"CALIBRATION_PRECISION == {precision}" in header


def test_protocol_identity_covers_worker_renderer_dependencies(monkeypatch):
    from compiler.offload import cpu_protocol_calibration as protocol

    before = cpu_protocol_identity()
    original = protocol.worker_renderer_identities()
    monkeypatch.setattr(protocol, "worker_renderer_identities", lambda: {**original, "emission/common/loops.py": "a" * 64})
    assert cpu_protocol_identity() != before


@pytest.mark.parametrize("precision", [True, 16, 128, "64"])
def test_control_renderer_rejects_unknown_precision(precision):
    with pytest.raises(NumericalCalibrationError, match="precision"):
        generated_cpu_control_source(precision)


@pytest.fixture(scope="module", params=(32, 64))
def actual_protocol_controls(request, tmp_path_factory):
    fortran, cxx = shutil.which("gfortran"), shutil.which("g++")
    if not fortran or not cxx or not shutil.which("taskset") or not hasattr(os, "sched_getaffinity"):
        pytest.skip("original native Fortran/GNU OpenMP fixture backend unavailable")
    target = tmp_path_factory.mktemp("cpu-protocol-control-" + str(request.param))
    original = target / "original.f90"
    original.write_text(native_fixture_source(request.param))
    (target / GENERATED_CONTROL_HEADER).write_text(generated_cpu_control_source(request.param))
    proof_source = target / "proof.cpp"
    proof_source.write_text(TEAM_PROOF_SOURCE)
    native_object, driver_object = target / "native.o", target / "driver.o"
    driver = Path(__file__).parents[1] / "offload" / "cpu_protocol_calibration.cpp"
    plain, proof = target / "plain", target / "proof"
    commands = [
        [fortran, "-cpp", "-O3", "-fopenmp", "-fbacktrace", "-g", *(["-DDPREC"] if request.param == 64 else []),
         "-c", str(original), "-o", str(native_object)],
        [cxx, *HOST_FLAGS, "-I", str(target), "-DCALIBRATION_PRECISION=" + str(request.param),
         "-c", str(driver), "-o", str(driver_object)],
        [cxx, "-fopenmp", str(native_object), str(driver_object), "-lgfortran", "-o", str(plain)],
        [cxx, *HOST_FLAGS, str(native_object), str(driver_object), str(proof_source), "-lgfortran",
         "-Wl,--wrap=GOMP_parallel", "-o", str(proof)],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, cwd=target)
        assert result.returncode == 0, result.stderr
    affinity = sorted(os.sched_getaffinity(0))[:4]
    environment = measurement_environment(os.environ)
    environment.update(OMP_NUM_THREADS=str(len(affinity)), OMP_THREAD_LIMIT="2147483647",
                       OMP_WAIT_POLICY="PASSIVE", GOMP_SPINCOUNT="0")
    prefix = ["taskset", "-c", ",".join(map(str, affinity))]
    return request.param, target, plain, proof, prefix, environment


@pytest.mark.native
def test_actual_original_native_and_production_worker_empty_controls_agree(actual_protocol_controls):
    precision, target, plain, proof, prefix, environment = actual_protocol_controls
    rows = {}
    for mode, binary, option in (("identity", plain, "--identity"), ("proof", proof, "--proof"), ("smoke", plain, "--smoke")):
        completed = subprocess.run([*prefix, str(binary), environment["OMP_NUM_THREADS"], option],
            capture_output=True, text=True, timeout=20, cwd=target, env=environment)
        assert completed.returncode == 0, completed.stderr
        rows[mode] = [json.loads(line) for line in completed.stdout.splitlines()]
        assert not any("samples" in row for row in rows[mode])
        identity = rows[mode][0]
        assert identity["precision_bits"] == precision
        assert identity["actual_team_threads"] == int(environment["OMP_NUM_THREADS"])
        assert identity["omp_wait_policy"] == "PASSIVE"
        assert identity["gomp_spincount"] == "0"
    assert rows["identity"] == [rows["proof"][0]] == [rows["smoke"][0]]
    for mode in ("proof", "smoke"):
        controls = rows[mode][1:]
        assert len(controls) == 6
        assert {row["items"] for row in controls} == {0, 1, 8}
        assert all(row["agreement_passed"] is True for row in controls)
        if mode == "proof":
            assert all(row["parallel_entries"] == 1 and row["wrong_team_visits"] == 0 and
                       row["thread_visits"] == [1] * int(environment["OMP_NUM_THREADS"]) for row in controls)
    assert not (target / "cpu-protocol-raw-samples.jsonl").exists()


@pytest.mark.native
def test_actual_control_preserves_production_cyclic_assignment(actual_protocol_controls):
    precision, target, _, _, prefix, environment = actual_protocol_controls
    source = target / "cyclic-assignment.cpp"
    source.write_text(f"""#define CALIBRATION_PRECISION {precision}
using real={'double' if precision == 64 else 'float'};
#include "{GENERATED_CONTROL_HEADER}"
int main() {{
constexpr int n=17;
real a[n],b[n],output[n];
for(int i=0;i<n;++i){{a[i]=real(i)*real(.125);b[i]=real(.5);output[i]=real(-99);}}
fort_cpu_protocol_generated::pointwise_worker(a,n,b,n,output,n,n,2,4);
for(int i=0;i<n;++i){{
 const real expected=i%4==2?a[i]+real(.25)*b[i]:real(-99);
 if(output[i]!=expected)return 3;
}}
fort_cpu_protocol_generated::pointwise_worker(nullptr,0,nullptr,0,nullptr,0,0,2,4);
generated_cpu(a,b,output,n,4);
for(int i=0;i<n;++i)if(output[i]!=a[i]+real(.25)*b[i])return 4;
}}
""")
    binary = target / "cyclic-assignment"
    built = subprocess.run([shutil.which("g++"), *HOST_FLAGS, "-I", str(target), str(source), "-o", str(binary)],
                           capture_output=True, text=True, timeout=30, cwd=target)
    assert built.returncode == 0, built.stderr
    checked = subprocess.run([*prefix, str(binary)], capture_output=True, text=True, timeout=10,
                             cwd=target, env=environment)
    assert checked.returncode == 0, checked.stderr


def test_regression_intercept_is_not_execution_startup_or_a_memory_prerequisite():
    records = observations_v2()
    for row in records:
        if row.get("backend") in MEASURED_BACKENDS and row.get("family") == "arithmetic_v2":
            for sample in row["samples"]:
                sample["elapsed_seconds"] += sample["repetitions"] * 30e-6
                sample["wall_seconds"] = sample["elapsed_seconds"]
    base = calibrated_v2(records)
    for backend in MEASURED_BACKENDS:
        assert base["numerical"]["families"]["memory_v2"][backend]["status"] == "rejected"
    result = corrected(base)
    assert result["numerical"] == base["numerical"]
    for backend in MEASURED_BACKENDS:
        assert cpu_protocol_costs(result, backend, access_class=MEMORY_ACCESS_CLASS)["fixed_seconds"] == pytest.approx(2e-6)


def test_fixed_term_is_charged_once_when_traffic_exceeds_working_set():
    costs = cpu_protocol_costs(corrected(), "native_fork_join", access_class=MEMORY_ACCESS_CLASS)
    working_set = 65536 * 24
    single = costs["fixed_seconds"] + memory_compute_seconds(costs["memory_cost_model"], working_set, working_set)
    double = costs["fixed_seconds"] + memory_compute_seconds(costs["memory_cost_model"], 2 * working_set, working_set)
    assert double == pytest.approx(2 * single - costs["fixed_seconds"])


@pytest.mark.parametrize("mutation", ["no_team", "twice", "wrong_size", "sentinel", "missing", "slow_control"])
def test_failed_protocol_or_control_rejects_only_affected_backend(mutation):
    base = calibrated_v2()
    records, identity, proofs = evidence(base)
    if mutation == "no_team":
        proofs[0]["thread_visits"] = [0] * 4
    elif mutation == "twice":
        proofs[0]["parallel_entries"] = 2
    elif mutation == "wrong_size":
        proofs[0]["wrong_team_visits"] = 4
    elif mutation == "sentinel":
        proofs[0]["agreement_passed"] = False
    elif mutation == "missing":
        proofs.pop(0)
    else:
        records[1]["samples"] = samples(4e-6)
    result = profile_from_cpu_protocol_measurements(base, records, identity, proofs)
    assert result["cpu_execution_protocol"]["team_proofs"] == proofs
    with pytest.raises(NumericalCalibrationError, match="rejected"):
        cpu_protocol_costs(result, "native_fork_join", access_class=MEMORY_ACCESS_CLASS)
    cpu_protocol_costs(result, "generated_cpu", access_class=MEMORY_ACCESS_CLASS)
    cpu_protocol_costs(result, "native_serial", access_class=MEMORY_ACCESS_CLASS)


def test_memory_only_acceptance_does_not_depend_on_failed_expression_holdouts():
    records = observations_v2()
    for row in records:
        if row.get("family") == "ordinary_mix_v2":
            for sample in row["samples"]:
                sample["elapsed_seconds"] *= 3
                sample["wall_seconds"] *= 3
    result = corrected(calibrated_v2(records))
    for backend in CPU_BACKENDS:
        cpu_protocol_costs(result, backend, access_class=MEMORY_ACCESS_CLASS)
        classes = result["cpu_execution_protocol"]["backends"][backend]["legacy_compute_diagnostic"]["workload_validation"]
        assert classes["memory_only_v1"]["status"] == "accepted"
        assert classes["ordinary_expression_v2"]["status"] == "rejected"
        assert classes["scalar_expression_v2"]["status"] == "accepted"


def test_historical_memory_diagnostic_cannot_authorize_fresh_runtime_identity():
    base = calibrated_v2()
    records, identity, proofs = evidence(base)
    records = [row for row in records if row["kind"] == "cpu_startup_cost_v1"]
    result = profile_from_cpu_protocol_measurements(base, records, identity, proofs)
    assert result["numerical"] == base["numerical"]
    for backend in CPU_BACKENDS:
        details = result["cpu_execution_protocol"]["backends"][backend]
        assert details["startup"]["status"] == "accepted"
        assert details["legacy_memory_diagnostic"]["status"] == "accepted"
        assert details["memory"]["reason_codes"] == ["missing_or_incorrect_memory_observation"]
        with pytest.raises(NumericalCalibrationError, match="rejected"):
            cpu_protocol_costs(result, backend, access_class=MEMORY_ACCESS_CLASS)


def test_fresh_memory_is_independent_of_historical_failed_memory_observations():
    base = calibrated_v2()
    records, identity, proofs = evidence(base)
    observations = observations_v2()
    for row in observations:
        if row.get("family") == "memory_v2" and row["backend"] in CPU_BACKENDS and row["items"] == 524288:
            row["samples"] = samples(1.0)
    failed_base = calibrated_v2(observations)
    result = profile_from_cpu_protocol_measurements(failed_base, records, identity, proofs)
    for backend in CPU_BACKENDS:
        details = result["cpu_execution_protocol"]["backends"][backend]
        assert details["legacy_memory_diagnostic"]["status"] == "rejected"
        assert details["memory"]["status"] == "accepted"
        cpu_protocol_costs(result, backend, access_class=MEMORY_ACCESS_CLASS)


@pytest.mark.parametrize("mutation", ["missing", "agreement", "slow_holdout"])
def test_fresh_memory_failure_keeps_only_affected_role_unavailable(mutation):
    base = calibrated_v2()
    records, identity, proofs = evidence(base)
    row = next(row for row in records if row["kind"] == "cpu_memory_cost_v1" and
               row["backend"] == "native_serial" and row["items"] == 524288)
    if mutation == "missing":
        records.remove(row)
    elif mutation == "agreement":
        row["agreement_passed"] = False
    else:
        row["samples"] = samples(1.0)
    result = profile_from_cpu_protocol_measurements(base, records, identity, proofs)
    assert result["cpu_execution_protocol"]["backends"]["native_serial"]["memory"]["status"] == "rejected"
    with pytest.raises(NumericalCalibrationError, match="rejected"):
        cpu_protocol_costs(result, "native_serial", access_class=MEMORY_ACCESS_CLASS)
    for backend in MEASURED_BACKENDS:
        cpu_protocol_costs(result, backend, access_class=MEMORY_ACCESS_CLASS)


@pytest.mark.parametrize("mutation", ["traffic", "working_set", "float_bytes", "role", "items", "backend", "duplicate"])
def test_malformed_fresh_memory_records_are_not_reinterpreted(mutation):
    base = calibrated_v2()
    records, identity, proofs = evidence(base)
    row = next(row for row in records if row["kind"] == "cpu_memory_cost_v1")
    if mutation == "traffic":
        row["traffic_bytes"] += 1
    elif mutation == "working_set":
        row["working_set_bytes"] *= 2
    elif mutation == "float_bytes":
        row["traffic_bytes"] = float(row["traffic_bytes"])
    elif mutation == "role":
        row["role"] = "holdout" if row["role"] == "fit" else "fit"
    elif mutation == "items":
        row["items"] = True
    elif mutation == "backend":
        row["backend"] = "gpu"
    else:
        records.append(deepcopy(row))
    with pytest.raises(NumericalCalibrationError):
        profile_from_cpu_protocol_measurements(base, records, identity, proofs)


@pytest.mark.parametrize(("key", "value"), [
    ("actual_team_threads", 1), ("actual_team_threads", True), ("actual_team_threads", None),
    ("thread_limit", 1), ("thread_limit", True), ("thread_limit", 1 << 31),
    ("omp_wait_policy", "x" * 129), ("gomp_spincount", "1\n2"), ("gomp_spincount", 100),
])
def test_original_team_and_bounded_runtime_identity_are_required(key, value):
    base = calibrated_v2()
    records, identity, proofs = evidence(base)
    identity[key] = value
    with pytest.raises(NumericalCalibrationError, match="OpenMP"):
        profile_from_cpu_protocol_measurements(base, records, identity, proofs)


@pytest.mark.parametrize("key", ["omp_wait_policy", "gomp_spincount"])
def test_absent_runtime_environment_identity_is_distinct_from_unset(key):
    base = calibrated_v2()
    records, identity, proofs = evidence(base)
    del identity[key]
    with pytest.raises(NumericalCalibrationError, match="OpenMP"):
        profile_from_cpu_protocol_measurements(base, records, identity, proofs)


def test_memory_and_coefficient_failures_are_not_overridden_by_good_startup():
    records = observations_v2()
    for row in records:
        if (row.get("backend") == "native_fork_join" and row.get("items") == 524288 and
                row.get("family") in {"memory_v2", "primitive_cos_v2"}):
            for sample in row["samples"]:
                sample["elapsed_seconds"] *= 3
                sample["wall_seconds"] *= 3
    result = corrected(calibrated_v2(records))
    backend = result["cpu_execution_protocol"]["backends"]["native_fork_join"]
    assert backend["startup"]["status"] == "accepted"
    assert backend["memory"]["status"] == "rejected"
    assert backend["legacy_compute_diagnostic"]["coefficients"]["primitive_cos_v2"]["status"] == "rejected"
    with pytest.raises(NumericalCalibrationError, match="rejected"):
        cpu_protocol_costs(result, "native_fork_join", access_class=MEMORY_ACCESS_CLASS)


@pytest.mark.parametrize("mutation", ["schema", "source", "native_source", "raw", "acceptance", "objects", "flags", "affinity", "duration", "duplicate"])
def test_saved_evidence_is_authenticated_and_reconstructed(mutation):
    result = corrected()
    saved = result["cpu_execution_protocol"]
    if mutation == "schema":
        saved["schema_version"] = True
    elif mutation == "source":
        saved["protocol_source_sha256"] = "f" * 64
    elif mutation == "native_source":
        saved["native_source_sha256"] = "f" * 64
    elif mutation == "raw":
        saved["numerical_observations_sha256"] = "f" * 64
    elif mutation == "acceptance":
        saved["backends"]["native_fork_join"]["startup"]["fixed_seconds"] = 0
    elif mutation == "objects":
        saved["execution_identity"]["timed_objects"]["native"] = "bad"
    elif mutation == "flags":
        saved["execution_identity"]["fortran"]["semantic_options"] = "-Ofast"
    elif mutation == "affinity":
        saved["execution_identity"]["cpu_affinity"] = [1, 2, 3, 4]
    elif mutation == "duration":
        saved["measurements"][0]["samples"][0]["wall_seconds"] = .199
    else:
        saved["measurements"].append(deepcopy(saved["measurements"][0]))
    with pytest.raises(NumericalCalibrationError):
        validate_cpu_protocol(result)


def test_unrepresentable_negative_memory_residual_retains_rejection():
    base = calibrated_v2()
    records, identity, proofs = evidence(base)
    for row in records:
        if row["backend"] == "native_fork_join":
            row["samples"] = samples(1.0)
    result = profile_from_cpu_protocol_measurements(base, records, identity, proofs)
    backend = result["cpu_execution_protocol"]["backends"]["native_fork_join"]
    assert backend["startup"]["status"] == "accepted"
    assert backend["memory"]["reason_codes"] == ["measured_startup_exceeds_memory_measurement"]


def test_producer_links_exact_objects_in_separate_proof_binary_and_scrubs_affinity(tmp_path, monkeypatch):
    base = calibrated_v2()
    records, identity, proofs = evidence(base)
    emitted_identity = {"kind": "cpu_protocol_identity_v1", **{
        key: value for key, value in identity.items() if key not in {"host", "timed_objects"}}}
    calls = []
    monkeypatch.setattr("os.sched_getaffinity", lambda _: {2, 3, 4, 5, 6})
    monkeypatch.setenv("OMP_PLACES", "cores")
    monkeypatch.setenv("GOMP_CPU_AFFINITY", "8-11")
    monkeypatch.setenv("FORT_PHASE_TIMING", "1")
    monkeypatch.setenv("OMP_WAIT_POLICY", "PASSIVE")
    monkeypatch.setenv("GOMP_SPINCOUNT", "10000")
    identity.update(omp_wait_policy="PASSIVE", gomp_spincount="10000")
    emitted_identity.update(omp_wait_policy="PASSIVE", gomp_spincount="10000")
    monkeypatch.setattr("compiler.offload.cpu_protocol_calibration._tool", lambda value, _: value)

    def run(command, directory, log, *, timeout, env=None):
        calls.append((command, env))
        if "-c" in command and "-o" in command:
            Path = type(tmp_path)
            Path(command[command.index("-o") + 1]).write_bytes(b"exact compiled object")
        if "--version" in command:
            return base["toolchain"]["host_cxx_version"] + "\n"
        if env is None:
            return ""
        assert env["OMP_PROC_BIND"] == "false"
        assert env["OMP_WAIT_POLICY"] == "PASSIVE"
        assert env["GOMP_SPINCOUNT"] == "10000"
        assert not {"OMP_PLACES", "GOMP_CPU_AFFINITY", "FORT_PHASE_TIMING"} & env.keys()
        rows = [] if "--identity" in command else proofs if "--proof" in command else records
        return "\n".join(json.dumps(row) for row in [emitted_identity, *rows])

    args = SimpleNamespace(fortran="/gfortran", host_cxx="/g++", fortran_flag=["-O3", "-fopenmp"],
                           build_dir=tmp_path / "new-build")
    result = calibrate_cpu_protocol(base, args, run=run)
    assert (args.build_dir / GENERATED_CONTROL_HEADER).read_text() == generated_cpu_control_source(64)
    driver_compile = next(command for command, _ in calls if "-c" in command and any("cpu_protocol_calibration.cpp" in item for item in command))
    assert driver_compile[driver_compile.index("-I") + 1] == str(args.build_dir)
    cpu_protocol_costs(result, "generated_cpu", access_class=MEMORY_ACCESS_CLASS)
    links = [command for command, _ in calls if "-lgfortran" in command]
    assert len(links) == 2
    shared_objects = {token for token in links[0] if token.endswith(".o")}
    assert shared_objects == {token for token in links[1] if token.endswith(".o")}
    assert not any("--wrap" in token for token in links[0])
    assert "-Wl,--wrap=GOMP_parallel" in links[1]
    runs = [command for command, env in calls if env is not None]
    assert runs[0][-1] == "--identity"
    assert runs[1][-1] == "--proof"
    assert runs[0][:3] == ["taskset", "-c", "2,3,4,5"]
    assert result["numerical"] == base["numerical"]


@pytest.mark.parametrize("text", ["not JSON", "[]", '{"kind":"unknown"}', ""])
def test_parser_rejects_unknown_or_missing_identity(text):
    with pytest.raises(NumericalCalibrationError):
        parse_cpu_protocol_measurements(text)


@pytest.mark.parametrize("field", ["fortran", "host"])
def test_malformed_backend_identity_fails_closed(field):
    base = calibrated_v2()
    records, identity, proofs = evidence(base)
    identity[field] = []
    with pytest.raises(NumericalCalibrationError, match="identity|backend"):
        profile_from_cpu_protocol_measurements(base, records, identity, proofs)


@pytest.mark.parametrize("access_class", [None, "stencil_v1", "read_modify_write_v1", "zero_fill_v1", "copy_v1"])
def test_memory_table_requires_its_exact_independently_validated_access_class(access_class):
    with pytest.raises(NumericalCalibrationError, match="access class"):
        cpu_protocol_costs(corrected(), "native_fork_join", access_class=access_class)
