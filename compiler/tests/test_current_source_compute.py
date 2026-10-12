"""Current source costs require original form and both independent backends."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.numerical_preparation import prepare_original_region
from compiler.frontend.source_effects import SourceEffects
from compiler.offload import numerical_gpu_execution as gpu
from compiler.offload import numerical_gpu_execution_costs as gpu_costs
from compiler.offload.config import OffloadConfig
from compiler.offload.cpu_dependency_calibration import profile_from_dependency_measurements
from compiler.offload.numerical_execution import _hash
from compiler.offload.source_applicability import source_compute_applicability
from compiler.offload.source_compute import apply_source_compute_costs
from compiler.tests.test_cpu_dependency_current import current_observations
from compiler.tests.test_dependency_source import BODY
from compiler.tests.test_numerical_execution import scale
from compiler.tests.test_numerical_gpu_execution_costs import evidence, launch_observations


@pytest.fixture(scope="module")
def bundle():
    # Reconstruct synthetic raw evidence through the actual strict readers.
    # No saved acceptance flag or historical current identity is patched.
    with pytest.MonkeyPatch.context() as patch:
        base, rows, identity = current_observations.__wrapped__(patch)
        profile = profile_from_dependency_measurements(base, rows, identity)
        parent = evidence()
        raw = deepcopy(parent[gpu.SECTION]["measurements"])
        calibration = deepcopy(parent[gpu.SECTION]["calibration"])
        cpu_name = profile["hardware"]["cpu_name"]
        host_version = profile["toolchain"]["host_cxx_version"]
        profile["hardware"] = {**parent["hardware"], "cpu_name": cpu_name}
        profile["toolchain"] = {**parent["toolchain"], "host_cxx_version": host_version}
        affinity = identity["cpu_affinity"]
        raw[0].update(cpu_name=cpu_name, host_cxx_version=host_version, sampling_cpu_affinity=affinity)
        calibration.update(base_profile_identity=_hash(profile), compatibility_identity=gpu.compatibility_identity(profile),
                           sampling_cpu_affinity=affinity)
        for command in calibration["run_commands"]:
            command[2] = ",".join(map(str, affinity))
        profile = gpu.profile_from_gpu_execution_measurements(profile, raw, calibration)
        controls, calibration = launch_observations(profile)
        for command in calibration["run_commands"]:
            command[2] = ",".join(map(str, affinity))
        profile = gpu_costs.profile_from_gpu_launch_measurements(profile, controls, calibration)
        parent_identity = profile[gpu.SECTION]["identity"]
        patch.setattr(gpu, "current_gpu_identities", lambda _: (
            parent_identity["source_generator_id"], parent_identity["registry_id"],
            parent_identity["generator_id"], parent_identity["renderer_sources"]))
        yield profile


def prepared_source(directory, *, body=BODY, participation="serial", helper="", nested=False):
    opening, closing = "", ""
    if participation == "fork_join":
        opening, closing = "!$omp parallel do private(i,x,y) schedule(static)\n", "!$omp end parallel do\n"
    dimensions = ":,:" if nested else ":"
    loop = "do j=1,n\ndo i=1,n\n" if nested else "do i=1,n\n"
    ending = "enddo\nenddo\n" if nested else "enddo\n"
    source = f"""module original_cost_form
implicit none
contains
{helper}
subroutine evaluate(a,b,out,n)
real(8),intent(in)::a({dimensions}),b({dimensions})
real(8),intent(out)::out({dimensions})
integer,intent(in)::n
integer::i,j
real(8)::x,y
{opening}{loop}{body}
{ending}{closing}end subroutine
end module
"""
    path = directory / "original.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    entry = "original_cost_form::evaluate"
    selection = tuple(analysis.routines[entry].execution.content)
    prepared = prepare_original_region(analysis, entry, selection,
        options=CompilerOptions(gpu_policy="sections", memory_model="scoped"),
        config=OffloadConfig(policy="sections"), environment_root=entry)
    applicability = source_compute_applicability(analysis, prepared, native_participation=participation)
    return path, analysis, prepared, applicability


def model(prepared, profile, applicability, participation="serial"):
    result, = apply_source_compute_costs(prepared.offload, profile, participation,
        source_applicability=applicability).units
    return result


@pytest.mark.parametrize("participation", ["serial", "fork_join"])
def test_current_complete_bundle_precedes_cpu_only_placeholder(tmp_path, bundle, participation):
    _, _, prepared, applicability = prepared_source(tmp_path, participation=participation)
    before = deepcopy(bundle)
    assert applicability is not None
    assert bundle["numerical_execution"]["workload_validation"]["ordinary_expression_v2"]["native_serial"]["status"] == "rejected"
    result = model(prepared, bundle, applicability, participation)
    costs = result.compute_model
    assert costs is not None, result.work_estimate_reason
    assert costs["cost_model"] == "source_work_span_cpu_gpu_v1"
    assert costs["source_applicability"] == applicability.public()
    assert costs["native_fortran"]["compiled_form"] == "direct_loop"
    assert costs["generated_cpu"]["compiled_form"] == "direct_loop"
    assert costs["gpu"]["launch_enqueue_included"] is True
    assert costs["gpu"]["fixed_seconds"] > 0
    assert costs["cpu_affinity"] == bundle["numerical_execution"]["identity"]["cpu_affinity"]
    assert costs["fortran"] == bundle["numerical_execution"]["identity"]["fortran"]
    assert costs["item_range"] == [65536, 1048576]
    assert result.region is prepared.offload.units[0].region
    assert result.footprints == prepared.offload.units[0].footprints
    assert bundle == before


@pytest.mark.parametrize("missing", ["cpu_dependency", "numerical_execution_gpu_costs", "numerical_execution_gpu"])
def test_partial_bundle_never_borrows_legacy_gpu(tmp_path, bundle, missing):
    _, _, prepared, applicability = prepared_source(tmp_path)
    profile = deepcopy(bundle)
    del profile[missing]
    result = model(prepared, profile, applicability)
    assert result.compute_model is None
    assert result.work_per_iteration is None


@pytest.mark.parametrize("defect", ["missing", "form", "identity", "participation", "access"])
def test_original_form_handoff_is_required(tmp_path, bundle, defect):
    _, _, prepared, applicability = prepared_source(tmp_path)
    if defect == "missing":
        applicability = None
    elif defect == "form":
        applicability = replace(applicability, native_compiled_form="value_function_same_unit")
    elif defect == "identity":
        applicability = replace(applicability, preparation_identity="invalid")
    elif defect == "participation":
        applicability = replace(applicability, native_participation="fork_join")
    else:
        applicability = replace(applicability, access_class="stencil_v1")
    result = model(prepared, bundle, applicability)
    assert result.compute_model is None
    assert "proved original source compiled form" in result.work_estimate_reason


def test_stale_source_and_helper_forms_do_not_get_direct_loop_witness(tmp_path):
    path, analysis, prepared, applicability = prepared_source(tmp_path)
    with pytest.raises(FrozenInstanceError):
        applicability.native_compiled_form = "value_function_same_unit"
    path.write_text(path.read_text().replace("out(i)=", "out(i)=1d0+"))
    assert source_compute_applicability(analysis, prepared, native_participation="serial") is None
    assert source_compute_applicability(analysis, None, native_participation="serial") is None
    helper = "pure function leaf(a,b) result(value)\nreal(8),intent(in)::a,b\nreal(8)::value\nvalue=a*b+a+b\nend function\n"
    _, analysis, prepared, applicability = prepared_source(tmp_path,
        body="out(i)=leaf(a(i),b(i))", helper=helper)
    assert prepared.extraction.numerical_helpers
    assert applicability is None


@pytest.mark.parametrize(("body", "nested", "reason"), [
    ("x=a(i)\ny=b(i)\nout(i)=sqrt(x*x+y*y)", False, "primitive interval proof"),
    ("x=a(i)+a(i+1)\ny=b(i)\nout(i)=x+y", False, "memory access class"),
    (BODY.replace("(i)", "(i,j)"), True, "dense traversal"),
])
def test_pointwise_domain_and_traversal_restrictions_remain(tmp_path, bundle, body, nested, reason):
    _, _, prepared, applicability = prepared_source(tmp_path, body=body, nested=nested)
    result = model(prepared, bundle, applicability)
    assert result.compute_model is None
    assert reason in result.work_estimate_reason


@pytest.mark.parametrize("backend", ["native_serial", "generated_cpu", "gpu"])
def test_rejected_independent_backend_cannot_supply_complete_model(tmp_path, bundle, backend):
    _, _, prepared, applicability = prepared_source(tmp_path)
    profile = deepcopy(bundle)
    if backend == "gpu":
        previous = profile.pop(gpu_costs.SECTION)
        rows = deepcopy(previous["measurements"])
        for row in rows[5:]:
            if row["items"] == 17:
                row["resident_event_seconds"] *= .5
        profile = gpu_costs.profile_from_gpu_launch_measurements(profile, rows, previous["calibration"])
    else:
        rows = deepcopy(profile["cpu_dependency"]["measurements"])
        for row in rows:
            if row["recipe"] == "dep_ordinary_w2_inline_p64" and row["backend"] == backend:
                scale(row, 3)
        profile = profile_from_dependency_measurements(profile, rows, profile["cpu_dependency"]["execution_identity"])
    result = model(prepared, profile, applicability)
    assert result.compute_model is None
    assert result.work_per_iteration is None


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_current_backend_identity_is_required_for_composed_costs(tmp_path, bundle, monkeypatch, backend):
    _, _, prepared, applicability = prepared_source(tmp_path)
    profile = deepcopy(bundle)
    if backend == "cpu":
        profile["cpu_dependency"]["generator_id"] = "f" * 64
    else:
        identity = profile[gpu.SECTION]["identity"]
        monkeypatch.setattr(gpu, "current_gpu_identities", lambda _: (
            "f" * 64, identity["registry_id"], identity["generator_id"], identity["renderer_sources"]))
    result = model(prepared, profile, applicability)
    assert result.compute_model is None
    assert result.work_per_iteration is None


def test_independently_valid_gpu_placement_must_match_cpu_cost_placement(tmp_path, bundle):
    _, _, prepared, applicability = prepared_source(tmp_path)
    profile = deepcopy(bundle)
    previous = profile.pop(gpu_costs.SECTION)
    parent = profile.pop(gpu.SECTION)
    rows = deepcopy(parent["measurements"])
    calibration = deepcopy(parent["calibration"])
    rows[0]["sampling_cpu_affinity"] = [4]
    calibration["sampling_cpu_affinity"] = [4]
    for command in calibration["run_commands"]:
        command[2] = "4"
    profile = gpu.profile_from_gpu_execution_measurements(profile, rows, calibration)
    controls, calibration = launch_observations(profile)
    profile = gpu_costs.profile_from_gpu_launch_measurements(profile, controls, calibration)
    assert profile[gpu_costs.SECTION]["startup"] == previous["startup"]
    result = model(prepared, profile, applicability)
    assert result.compute_model is None
    assert "sampling placement differs" in result.work_estimate_reason
