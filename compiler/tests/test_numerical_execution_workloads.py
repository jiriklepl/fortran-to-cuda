"""Source/dispatch and fixed-protocol controls, without timed observations."""

from __future__ import annotations

import copy
import json
import math
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from compiler.ir import IntrinsicCall
from compiler.ir.nodes import walk_expr
from compiler.numerical_contract import numerical_build_contract
from compiler.offload import numerical_execution_calibration as producer
from compiler.offload import numerical_execution_workloads as workloads
from compiler.offload.analysis import _compute_operation_counts
from compiler.offload.cpu_dependency_workloads import _round
from compiler.offload.numerical_calibration import NumericalCalibrationError


def test_registry_is_metadata_only_fixed_and_canonical(monkeypatch):
    monkeypatch.setattr(workloads, "lower_source", lambda *a, **k: pytest.fail("metadata parsed numerical source"))
    registry = workloads.execution_registry(64)
    assert tuple(row["name"] for row in registry) == workloads.NAMES
    assert len(registry) == 17
    assert sum(len(row["sizes"]) for row in registry) == 97
    assert {r["family"] for r in registry if r["role"] == "coefficient"} == {
        "arithmetic",
        "sqrt",
        "acos",
        "cos",
        "divide",
    }
    assert {r["name"] for r in registry if r["holdout_kind"] == "separate_helper"} == {
        "helper_scalar_mix",
        "helper_private_mix",
    }
    assert {r["family"] for r in registry if r["holdout_kind"] == "domain"} == {"sqrt", "acos", "cos"}
    for row in registry:
        expected = copy.deepcopy(row)
        expected.pop("recipe_id")
        assert (
            row["recipe_id"]
            == sha256(json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        )
        assert set(row["fit_sizes"]) <= set(row["sizes"])
        if row["role"] == "holdout":
            assert row["fit_sizes"] == []
    registry[0]["arithmetic"] = -1
    assert workloads.execution_registry(64)[0]["arithmetic"] == 520


@pytest.mark.parametrize("precision", [32, 64])
def test_actual_prepared_worker_counts_math_private_closures_and_finite_references(precision):
    recipes = workloads.execution_recipes(precision)
    assert len(recipes) == 17
    for recipe in recipes:
        count, divisions, reason = _compute_operation_counts(recipe.region.body)
        assert reason is None
        assert count == recipe.metadata["arithmetic"]
        assert divisions == recipe.metadata["intrinsics"].get("divide", 0)
        intrinsic_counts = {
            name: sum(
                isinstance(node, IntrinsicCall) and node.name.lower() == name
                for statement in recipe.region.body.statements
                for node in walk_expr(statement.value)
            )
            for name in ("sqrt", "acos", "cos")
        }
        assert {k: v for k, v in intrinsic_counts.items() if v} == {
            k: v for k, v in recipe.metadata["intrinsics"].items() if k != "divide"
        }
        cpp = recipe.cpp_sources[0][1]
        assert "omp_get_thread_num(), omp_get_num_threads()" in cpp
        assert "#pragma omp parallel num_threads(threads)" in cpp
        assert "#pragma omp parallel for" not in cpp
        assert "for (std::size_t flat = tid; flat < total; flat += team)" in cpp
        assert all(
            math.isfinite(value) for split in ("training", "holdout") for value in recipe.reference_period(split)
        )
        native = "\n".join(text for _, text in recipe.native_sources)
        assert "schedule(static) default(none)" in native
        assert "num_threads(threads)" in native
        assert "select case" not in native.lower()
        assert "input_value=input" in native
        assert "other_value=other" in native
    private = next(r for r in recipes if r.name == "private_mix")
    assert private.metadata["private_features"] == {
        "private_array_groups": 4,
        "private_array_elements": 64,
        "max_private_array_elements": 16,
        "max_private_array_rank": 2,
    }
    helper = next(r for r in recipes if r.name == "helper_private_mix")
    assert len(helper.native_sources) == 2
    assert helper.native_sources[0][0].endswith("_helper.f90")
    assert "use execution_helper_helper_private_mix,only:" in helper.native_sources[1][1]
    assert len(helper.cpp_sources) == 1


def test_real32_math_reference_receives_typed_lattice():
    recipe = next(r for r in workloads.execution_recipes(32) if r.name == "domain_cos")
    a, b = recipe.lattices("holdout")
    assert a[0] == _round(-math.pi, 32)
    assert a[0] != -math.pi
    expected = _round(_round(math.cos(a[0]), 32) + _round(b[0] * _round(0.001, 32), 32), 32)
    assert recipe.reference_period("holdout")[0] == expected


@pytest.mark.parametrize("precision", [32, 64])
def test_registry_header_exports_typed_original_and_generated_abi(precision):
    text = workloads.registry_source(workloads.execution_recipes(precision), "a" * 64)
    assert "using execution_real=" + ("double" if precision == 64 else "float") + ";" in text
    assert "execution_worker=void(*)(int,int,const execution_real*,const execution_real*,execution_real*)" in text
    assert text.count('extern "C" void execution_') == 51
    assert workloads.registry_identity(precision) in text
    assert "execution_helper_scalar_mix_serial" in text
    assert "execution_domain_acos_holdout_reference" in text


def _records():
    registry = workloads.execution_registry(64)
    aggregate = {}
    raw = []
    order = 0
    for batch in range(7):
        cells = [
            ("numerical_execution_startup_v1", registry[1], n, workloads.CPU_BACKENDS[1 + (batch + turn) % 2])
            for n in (0, 1, 8)
            for turn in range(2)
        ]
        cells += [
            ("numerical_execution_cost_v1", r, n, workloads.CPU_BACKENDS[(batch + turn) % 3])
            for r in registry
            for n in r["sizes"]
            for turn in range(3)
        ]
        for kind, recipe, n, backend in cells:
            key = kind, recipe["name"], backend, n
            role = "fit" if n in recipe["fit_sizes"] else "holdout"
            row = aggregate.setdefault(
                key,
                {
                    "kind": kind,
                    "recipe": recipe["name"],
                    "recipe_id": recipe["recipe_id"],
                    "family": recipe["family"],
                    "backend": backend,
                    "items": n,
                    "role": role,
                    "agreement_passed": True,
                    "samples": [None] * 7,
                },
            )
            sample = {
                "batch": batch,
                "repetitions": 17,
                "elapsed_seconds": 0.201,
                "wall_seconds": 0.201,
                "global_order": order,
            }
            row["samples"][batch] = dict(sample)
            raw.append(
                {
                    **{k: v for k, v in row.items() if k != "samples"},
                    **sample,
                    "kind": "numerical_execution_startup_batch_v1"
                    if kind.endswith("startup_v1")
                    else "numerical_execution_batch_v1",
                    "traffic_bytes": n * 24,
                    "working_set_bytes": n * 24,
                }
            )
            order += 1
    return registry, list(aggregate.values()), raw


def test_raw_registry_has_seven_global_rounds_and_exact_reconstruction():
    registry, aggregate, raw = _records()
    assert len(raw) == 2079
    assert len(aggregate) == 297
    assert raw[0]["batch"] == 0
    assert raw[296]["batch"] == 0
    assert raw[297]["batch"] == 1
    producer.reconcile_raw_batches(raw, aggregate, registry)


@pytest.mark.parametrize(
    "mutation", ["missing", "order", "family", "recipe_id", "duration", "agreement", "bool_order", "bool_repetitions"]
)
def test_raw_conflicts_and_interruption_are_never_repaired(mutation):
    registry, aggregate, raw = _records()
    if mutation == "missing":
        raw.pop()
    elif mutation == "order":
        raw[0], raw[1] = raw[1], raw[0]
    elif mutation == "family":
        raw[0]["family"] = "arithmetic"
    elif mutation == "recipe_id":
        raw[0]["recipe_id"] = "0" * 64
    elif mutation == "duration":
        raw[0]["elapsed_seconds"] += 0.001
    elif mutation == "agreement":
        raw[0]["agreement_passed"] = False
    elif mutation == "bool_order":
        raw[1]["global_order"] = True
    else:
        raw[0]["repetitions"] = True
        aggregate[0]["samples"][0]["repetitions"] = 1
    with pytest.raises(NumericalCalibrationError):
        producer.reconcile_raw_batches(raw, aggregate, registry)


def _profile():
    return {
        "precision_bits": 64,
        "cpu_threads": 4,
        "hardware": {"cpu_name": "fixture CPU"},
        "toolchain": {"host_cxx_version": "fixture C++"},
        "numerical_contract": numerical_build_contract(),
    }


def _args(**kwargs):
    return SimpleNamespace(
        precision=64,
        threads=4,
        fortran_flag=["-cpp", "-O3", "-fopenmp", "-DDPREC", "-fbacktrace", "-g"],
        cpu_affinity="4,5,6,7",
        fortran="gfortran",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threads", 2),
        ("precision", 32),
        ("fortran_flag", []),
        ("fortran_flag", ["-O3", "-fopenmp", "-ffast-math"]),
        ("fortran_flag", ["-O3", "-fopenmp", "-flto=2"]),
    ],
)
def test_invalid_native_contract_rejected_before_build(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(producer, "cpu_identity", lambda: "fixture CPU")
    args = _args()
    setattr(args, field, value)
    with pytest.raises(NumericalCalibrationError):
        producer.calibrate_numerical_execution(
            _profile(),
            args,
            tmp_path,
            "unused",
            "g++",
            run=lambda *a, **k: pytest.fail("invalid contract ran a tool"),
            tool=lambda *a: pytest.fail("invalid contract selected a tool"),
        )
    assert not list(tmp_path.iterdir())


def test_cpu_mismatch_rejected_before_build(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "cpu_identity", lambda: "different CPU")
    with pytest.raises(NumericalCalibrationError, match="CPU identity"):
        producer.calibrate_numerical_execution(
            _profile(),
            _args(),
            tmp_path,
            "unused",
            "g++",
            run=lambda *a, **k: pytest.fail("CPU mismatch ran a tool"),
            tool=lambda *a: pytest.fail("CPU mismatch selected a tool"),
        )
    assert not list(tmp_path.iterdir())


def test_host_mismatch_and_disabled_fast_math_fail_before_numerical_compile(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "cpu_identity", lambda: "fixture CPU")
    monkeypatch.setattr(producer.os, "sched_getaffinity", lambda pid: {4, 5, 6, 7})
    args = _args()
    args.fortran_flag += ["-fno-fast-math", "-fno-lto"]
    calls = []

    def run(argv, *a, **k):
        calls.append(argv)
        assert argv == ["g++", "--version"]
        return "different C++"

    with pytest.raises(NumericalCalibrationError, match="host compiler"):
        producer.calibrate_numerical_execution(
            _profile(), args, tmp_path, "unused", "g++", run=run, tool=lambda *a: "gfortran"
        )
    assert calls == [["g++", "--version"]]
    assert not list(tmp_path.rglob("*.o"))


def test_driver_declares_global_protocol_and_no_cuda_execution():
    source = Path(workloads.__file__).with_name("numerical_execution_driver.cpp").read_text()
    assert 'std::fopen("numerical-execution-raw-samples.jsonl","wx")' in source
    assert "std::fflush(file)" in source
    assert (
        source.index("if(smoke)return 0;")
        < source.index("std::fopen(")
        < source.index("for(int batch=0;batch!=7;++batch)")
    )
    assert source.index("data.reset();data.execute(backend,threads)") < source.index("std::fopen(")
    assert "while(result.elapsed<.2)" in source
    assert "cuda" not in source.lower()


@pytest.mark.parametrize("reject_preflight", [False, True])
def test_same_object_preflight_finishes_before_any_sampling(tmp_path, monkeypatch, reject_preflight):
    monkeypatch.setattr(producer, "cpu_identity", lambda: "fixture CPU")
    monkeypatch.setattr(producer.os, "sched_getaffinity", lambda pid: {4, 5, 6, 7})
    monkeypatch.setattr(producer, "execution_generator_identity", lambda: "a" * 64)
    monkeypatch.setattr(
        producer,
        "execution_recipes",
        lambda precision: [
            SimpleNamespace(
                native_sources=(("fixture.f90", "! native source"),),
                cpp_sources=(("fixture.cpp", "// generated source"),),
            )
        ],
    )
    monkeypatch.setattr(producer, "registry_source", lambda *a: "// frozen registry")
    registry = workloads.execution_registry(64)
    identity = {
        "kind": "numerical_execution_identity_v1",
        "protocol_id": workloads.PROTOCOL_ID,
        "backend_id": workloads.BACKEND_ID,
        "generator_id": "a" * 64,
        "registry_id": workloads.registry_identity(64),
        "precision_bits": 64,
        "cpu_threads": 4,
        "cpu_affinity": [4, 5, 6, 7],
        "actual_team_threads": 4,
        "thread_limit": 100,
        "omp_dynamic": False,
        "omp_proc_bind": "false",
        "omp_wait_policy": None,
        "gomp_spincount": None,
        "fortran": {"compiler_version": "fixture Fortran", "compiler_options": "-O3 -fopenmp"},
    }
    calls = []
    validated = []
    for key in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT"):
        monkeypatch.delenv(key, raising=False)

    def run(argv, directory, log, **kwargs):
        calls.append(argv)
        if "--version" in argv:
            return "fixture C++"
        if "-o" in argv:
            Path(argv[argv.index("-o") + 1]).write_bytes(repr(argv).encode())
            return ""
        if "--identity" in argv:
            rows = []
        elif "--proof" in argv:
            rows = [
                {
                    "kind": "numerical_execution_team_proof_v1",
                    "backend": backend,
                    "items": n,
                    "recipe": "memory",
                    "recipe_id": registry[1]["recipe_id"],
                    "family": "memory",
                    "role": "holdout",
                    "traffic_bytes": 24 * n,
                    "working_set_bytes": 24 * n,
                    "agreement_passed": True,
                    "parallel_entries": 1,
                    "wrong_team_visits": 0,
                    "thread_visits": [1] * 4,
                }
                for n in (0, 1, 8)
                for backend in workloads.CPU_BACKENDS[1:]
            ]
        elif "--smoke" in argv:
            rows = [
                {
                    "kind": "numerical_execution_smoke_v1",
                    "recipe": row["name"],
                    "backend": backend,
                    "items": n,
                    "agreement_passed": True,
                }
                for row in registry
                for backend in workloads.CPU_BACKENDS
                for n in (0, 1, 8, 17)
            ]
        else:
            pytest.fail("timed invocation happened before or after untimed preflight")
        text = "\n".join(json.dumps(row) for row in [identity, *rows]) + "\n"
        log.write_text(text)
        return text

    def validate(profile, observed, proofs):
        assert "--smoke" in calls[-1]
        assert observed["timed_objects"]
        assert len(proofs) == 6
        assert all(proof["timed_objects"] == observed["timed_objects"] for proof in proofs)
        assert observed["proof_backend"] == producer.PROOF_BACKEND
        validated.append(True)
        if reject_preflight:
            raise NumericalCalibrationError("deliberate exact-native preflight mismatch")

    monkeypatch.setattr(producer, "validate_execution_preflight", validate)
    args = _args()
    args.preflight_only = True
    if reject_preflight:
        with pytest.raises(NumericalCalibrationError, match="deliberate"):
            producer.calibrate_numerical_execution(
                _profile(), args, tmp_path, None, "g++", run=run, tool=lambda *a: "gfortran"
            )
    else:
        result = producer.calibrate_numerical_execution(
            _profile(), args, tmp_path, None, "g++", run=run, tool=lambda *a: "gfortran"
        )
        assert result == _profile()
        receipt = json.loads((tmp_path / "numerical-execution/preflight-receipt.json").read_text())
        assert receipt["sampling_started"] is False
        assert receipt["smoke_cases"] == 204
    assert validated == [True]
    assert not list(tmp_path.rglob("*raw-samples*"))


def test_standalone_cli_binds_base_hash_and_preserves_preflight_profile(tmp_path, monkeypatch):
    profile_path = tmp_path / "base.json"
    profile_path.write_text(json.dumps(_profile()))
    output = tmp_path / "output.json"
    build = tmp_path / "build"
    from compiler.offload import profile as profiles

    monkeypatch.setattr(profiles, "load_profile", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(producer, "_tool", lambda *a: "fixture-c++")

    def calibrate(profile, args, directory, nvcc, host, **kwargs):
        assert args.preflight_only is True
        assert args.cpu_affinity == "4,5,6,7"
        assert host == "fixture-c++"
        (directory / "numerical-execution").mkdir(parents=True)
        return profile.copy()

    monkeypatch.setattr(producer, "calibrate_numerical_execution", calibrate)
    assert (
        producer.main(
            [
                "--profile",
                str(profile_path),
                "--output",
                str(output),
                "--build-dir",
                str(build),
                "--cpu-affinity",
                "4,5,6,7",
                "--preflight-only",
            ]
        )
        == 0
    )
    assert json.loads(output.read_text()) == _profile()
    receipt = json.loads((build / "numerical-execution/base-profile-receipt.json").read_text())
    assert receipt["sha256"] == sha256(profile_path.read_bytes()).hexdigest()
    assert receipt["gpu_base_remeasured"] is False
    with pytest.raises(SystemExit):
        producer.main(["--profile", str(profile_path), "--output", str(output), "--build-dir", str(build)])
