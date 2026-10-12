"""Attachment-local cost reuse retains strict input and backend authority.

The observations are the existing synthetic CPU/GPU fixtures. No numerical
worker, CUDA compilation, calibration, or application execution is performed.
"""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from compiler.offload import cost_projection
from compiler.offload import cpu_dependency_calibration as cpu_calibration
from compiler.offload import numerical_gpu_execution as gpu_execution
from compiler.offload import numerical_gpu_execution_costs as gpu_costs
from compiler.offload.cost_projection import _SourceCostProjection
from compiler.offload.cpu_dependency_calibration import ACCESS_CLASS, DOMAINS, cpu_dependency_costs
from compiler.offload.dependency_source import dependency_source_model
from compiler.offload.numerical_calibration import NumericalCalibrationError
from compiler.offload.numerical_gpu_execution_costs import gpu_execution_costs
from compiler.offload.source_compute import apply_source_compute_costs
from compiler.tests.test_current_source_compute import bundle as bundle
from compiler.tests.test_current_source_compute import prepared_source


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    _, _, prepared, applicability = prepared_source(tmp_path_factory.mktemp("projection-source"))
    assert applicability is not None
    return prepared.offload, applicability


@pytest.fixture
def profile(bundle):
    return deepcopy(bundle)


def cpu_keywords(unit):
    return {"workload_class": unit.workload_class, "workload_features": unit.workload_features,
            "primitive_domains": DOMAINS, "access_class": ACCESS_CLASS, "compiled_form": "direct_loop"}


def gpu_arguments(unit):
    counts = dict(unit.intrinsic_work_per_iteration)
    if unit.compute_runtime_divisions_per_iteration:
        counts["divide"] = unit.compute_runtime_divisions_per_iteration
    return counts, {"workload_class": unit.workload_class,
                    "workload_features": unit.workload_features, "access_class": ACCESS_CLASS}


def cpu_query(projection, unit, backend="native_serial"):
    return projection.cpu_costs(unit.compute_dependencies, backend, **cpu_keywords(unit))


def gpu_query(projection, unit):
    counts, keywords = gpu_arguments(unit)
    return projection.gpu_costs(counts, **keywords)


def test_projection_matches_fresh_public_queries_and_owns_returned_values(profile, source):
    analysis, _ = source
    unit, = analysis.units
    before = deepcopy(profile)
    expected_cpu = {backend: cpu_dependency_costs(profile, unit.compute_dependencies, backend,
                    **cpu_keywords(unit)) for backend in ("native_serial", "generated_cpu")}
    counts, keywords = gpu_arguments(unit)
    expected_gpu = gpu_execution_costs(profile, counts, **keywords)
    projection = _SourceCostProjection(profile)
    for backend, expected in expected_cpu.items():
        actual = cpu_query(projection, unit, backend)
        assert actual == expected
        actual["memory_cost_model"].clear()
        assert cpu_query(projection, unit, backend) == expected
    actual_gpu = gpu_query(projection, unit)
    assert actual_gpu == expected_gpu
    actual_gpu["memory_cost_model"].clear()
    actual_gpu["cpu_affinity"].append(999)
    assert gpu_query(projection, unit) == expected_gpu
    projection.finish()
    assert profile == before


def test_eligibility_rejection_does_not_poison_valid_projection(profile, source):
    analysis, _ = source
    unit, = analysis.units
    projection = _SourceCostProjection(profile)
    rejected = {**cpu_keywords(unit), "compiled_form": "unproved"}
    with pytest.raises(NumericalCalibrationError):
        projection.cpu_costs(unit.compute_dependencies, "native_serial", **rejected)
    assert cpu_query(projection, unit)["compiled_form"] == "direct_loop"
    counts, keywords = gpu_arguments(unit)
    with pytest.raises(NumericalCalibrationError):
        projection.gpu_costs(counts, **{**keywords, "access_class": "unproved_stencil"})
    assert gpu_query(projection, unit)["launch_enqueue_included"] is True
    projection.finish()


def test_unused_projection_does_not_examine_invalid_profile_and_expires():
    projection = _SourceCostProjection({"unexamined": object()})
    projection.finish()
    with pytest.raises(NumericalCalibrationError):
        projection.finish()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_successful_projection_expires_for_every_query(profile, source, backend):
    analysis, _ = source
    unit, = analysis.units
    projection = _SourceCostProjection(profile)
    cpu_query(projection, unit)
    projection.finish()
    with pytest.raises(NumericalCalibrationError):
        cpu_query(projection, unit) if backend == "cpu" else gpu_query(projection, unit)


def change_profile(profile, defect):
    if defect == "raw":
        profile["cpu_dependency"]["measurements"][0]["samples"][0]["elapsed_seconds"] *= 2
    elif defect == "status":
        family = profile["cpu_dependency"]["backends"]["native_serial"]["families"]["ordinary"]
        family["status"] = "rejected" if family["status"] == "accepted" else "accepted"
    elif defect == "placement":
        profile["numerical_execution"]["identity"]["cpu_affinity"].append(999)
    elif defect == "contract":
        profile["numerical_contract"]["identity"] = "f" * 64
    elif defect == "tuple":
        row = profile["cpu_dependency"]["measurements"][0]
        row["samples"] = tuple(row["samples"])
    else:
        raise AssertionError(defect)


@pytest.mark.parametrize("defect", ["raw", "status", "placement", "contract", "tuple"])
def test_caller_mutation_rejects_finish_and_expires_projection(profile, source, defect):
    analysis, _ = source
    unit, = analysis.units
    projection = _SourceCostProjection(profile)
    assert cpu_query(projection, unit)["seconds_per_item"] > 0
    change_profile(profile, defect)
    with pytest.raises(NumericalCalibrationError):
        projection.finish()
    with pytest.raises(NumericalCalibrationError):
        cpu_query(projection, unit)


@pytest.mark.parametrize("defect", ["tuple", "key", "nonfinite", "cycle"])
def test_snapshot_rejects_non_json_types_without_normalizing_them(profile, source, defect):
    analysis, _ = source
    unit, = analysis.units
    if defect == "tuple":
        change_profile(profile, defect)
        with pytest.raises(NumericalCalibrationError):
            cpu_dependency_costs(profile, unit.compute_dependencies, "native_serial", **cpu_keywords(unit))
    elif defect == "key":
        profile["unrelated"] = {1: "must not become a string key"}
    elif defect == "nonfinite":
        profile["unrelated"] = float("nan")
    else:
        profile["unrelated"] = profile
    projection = _SourceCostProjection(profile)
    with pytest.raises(NumericalCalibrationError):
        cpu_query(projection, unit)


@pytest.mark.parametrize("precision", [None, True, 16])
def test_malformed_precision_has_a_controlled_gpu_error(profile, source, precision):
    analysis, _ = source
    unit, = analysis.units
    if precision is None:
        del profile["precision_bits"]
    else:
        profile["precision_bits"] = precision
    projection = _SourceCostProjection(profile)
    with pytest.raises(NumericalCalibrationError):
        gpu_query(projection, unit)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_current_backend_change_rejects_finish(profile, source, monkeypatch, backend):
    analysis, _ = source
    unit, = analysis.units
    projection = _SourceCostProjection(profile)
    if backend == "cpu":
        cpu_query(projection, unit)
        monkeypatch.setattr(cost_projection, "_cpu_identity", lambda: ("changed-reader-or-backend",))
    else:
        gpu_query(projection, unit)
        monkeypatch.setattr(cost_projection, "_gpu_identity", lambda _: ("changed-renderer",))
    with pytest.raises(NumericalCalibrationError):
        projection.finish()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_identity_witness_does_not_alias_mutable_provider_values(profile, source, monkeypatch, backend):
    analysis, _ = source
    unit, = analysis.units
    witness = ({"nested": {"source": "first"}},)
    if backend == "cpu":
        monkeypatch.setattr(cost_projection, "_cpu_identity", lambda: witness)
    else:
        monkeypatch.setattr(cost_projection, "_gpu_identity", lambda _: witness)
    projection = _SourceCostProjection(profile)
    cpu_query(projection, unit) if backend == "cpu" else gpu_query(projection, unit)
    witness[0]["nested"]["source"] = "changed"
    with pytest.raises(NumericalCalibrationError, match="identity changed"):
        projection.finish()
    with pytest.raises(NumericalCalibrationError, match="expired"):
        cpu_query(projection, unit)


def test_independent_attachment_revalidates_changed_raw_profile(profile, source):
    analysis, _ = source
    unit, = analysis.units
    projection = _SourceCostProjection(profile)
    cpu_query(projection, unit)
    projection.finish()
    change_profile(profile, "raw")
    independent = _SourceCostProjection(profile)
    with pytest.raises(NumericalCalibrationError):
        cpu_query(independent, unit)


def test_serial_multi_unit_attachment_matches_independent_models(profile, source, monkeypatch, record_property):
    analysis, applicability = source
    unit, = analysis.units
    expected = dependency_source_model(unit, profile, "serial", source_applicability=applicability)
    multi = replace(analysis, units=(unit, replace(unit, index=unit.index + 1)))
    before = deepcopy(profile)
    reconstructions = {"cpu_dependency": 0, "gpu_parent": 0, "gpu_costs": 0}

    def trace(module, name, key):
        original = getattr(module, name)

        def observed(*args, **kwargs):
            reconstructions[key] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, observed)

    trace(cpu_calibration, "validate_dependency_profile", "cpu_dependency")
    trace(gpu_execution, "_build", "gpu_parent")
    trace(gpu_costs, "_build_from_parent", "gpu_costs")
    result = apply_source_compute_costs(multi, profile, "serial", source_applicability=applicability)
    assert len(result.units) == 2
    assert all(value.compute_model == expected for value in result.units)
    assert all(value.region is unit.region and value.footprints == unit.footprints for value in result.units)
    assert profile == before
    assert reconstructions == {"cpu_dependency": 1, "gpu_parent": 1, "gpu_costs": 1}
    record_property("calibration_reconstructions", json.dumps(reconstructions, sort_keys=True))


def test_later_unit_profile_mutation_discards_earlier_attached_models(profile, source):
    analysis, applicability = source
    unit, = analysis.units

    class MutatingFeatures:
        changed = False

        def to_dict(self):
            if not self.changed:
                self.changed = True
                change_profile(profile, "raw")
            return unit.workload_features.to_dict()

    features = MutatingFeatures()
    second = replace(unit, index=unit.index + 1, workload_features=features)
    result = apply_source_compute_costs(replace(analysis, units=(unit, second)), profile, "serial",
        source_applicability=applicability)
    assert features.changed
    assert len(result.units) == 2
    assert all(value.compute_model is None and value.work_per_iteration is None for value in result.units)
    assert all(value.work_estimate_reason for value in result.units)
    assert all(value.region is unit.region and value.footprints == unit.footprints for value in result.units)


@pytest.mark.parametrize("gate", ["form", "primitive"])
def test_source_gates_decline_before_invalid_evidence_is_demanded(tmp_path, profile, source, gate):
    profile["cpu_dependency"]["measurements"] = object()
    if gate == "form":
        analysis, _ = source
        applicability = None
        reason = "proved original source compiled form"
    else:
        _, _, prepared, applicability = prepared_source(tmp_path,
            body="x=a(i)\ny=b(i)\nout(i)=sqrt(x*x+y*y)")
        analysis = prepared.offload
        reason = "primitive interval proof"
    result = apply_source_compute_costs(analysis, profile, "serial", source_applicability=applicability)
    output, = result.units
    assert output.compute_model is None
    assert reason in output.work_estimate_reason
