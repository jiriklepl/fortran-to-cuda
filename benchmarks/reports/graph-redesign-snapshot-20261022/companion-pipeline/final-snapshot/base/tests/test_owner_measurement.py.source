"""Diagnostic call clocks never supply pricing or placement authority."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from elmm_pipeline.owner_measurement import validate_owner_measurements


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seal(record):
    record["identity"] = digest({key: value for key, value in record.items() if key not in {"identity", "variant"}})


def fixture(tmp_path, variant="native"):
    source = tmp_path / "caller.f90"
    source.write_text("subroutine caller\ncall step()\nend subroutine\n")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = {name: digest(name) for name in (
        "execution_identity", "source_inputs_identity", "configuration_identity", "capture_facts_identity")}
    call = {"caller": "caller", "source": str(source), "source_sha256": sha,
            "first_line": 2, "last_line": 2, "call_sha256": hashlib.sha256(b"call step()\n").hexdigest()}
    call["identity"] = digest(call)
    runtime = {"schema_version": 1, "runtime_id": "synthetic runtime"}
    record = {"schema_version": 1, "protocol": "complete-lexical-call-clock-v1", "variant": variant,
              **plan, "runtime_identity": digest(runtime), "compiler_identity": digest("compiler"),
              "call_sites": [call], "placement_authority": False, "automatic_estimate_available": False,
              "calibration_evidence": False,
              "interval": "before original call argument evaluation through return and finalization",
              "clock": "system_clock:int64", "observation_prefix": "FORT_SCOPE_MEASUREMENT ",
              "limitations": ["No runtime workload key; observations are not automatic placement evidence.",
                  "Resident variant may execute guarded native fallbacks; verify runtime traces separately.",
                  "Only listed direct call sites in registered subroutines of the authenticated supplied source graph are observed."]}
    seal(record)
    return {"scope_measurement": variant, "scope_execution": "reached", "scope_count": 1,
            "automatic_estimate_available": False, "scopes": [{"checked_execution_plan": plan,
                "owner_measurement": record}], "source_inputs": {str(source): sha},
            "sources": {str(source): {"sha256": sha, "replacement": "sources/caller.f90"}},
            "build_sources": [{"path": "sources/caller.f90", "language": "fortran", "role": "original_source"}],
            "runtime": runtime if variant == "resident" else None}


@pytest.mark.parametrize("variant", ["native", "resident"])
def test_authenticated_diagnostic_record_has_no_placement_authority(tmp_path, variant):
    manifest = fixture(tmp_path, variant)
    assert validate_owner_measurements(manifest) == variant
    changed = deepcopy(manifest)
    changed["scope_measurement"] = changed["scopes"][0]["owner_measurement"]["variant"] = (
        "resident" if variant == "native" else "native")
    assert changed["scopes"][0]["owner_measurement"]["identity"] == manifest["scopes"][0]["owner_measurement"]["identity"]


def test_ordinary_package_requires_no_measurement_contract():
    assert validate_owner_measurements({"scopes": []}) is None


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("protocol", "unknown"), ("variant", "resident"),
    ("placement_authority", True), ("calibration_evidence", True),
    ("automatic_estimate_available", True), ("execution_identity", "a" * 64),
    ("source_inputs_identity", "a" * 64), ("configuration_identity", "a" * 64),
    ("capture_facts_identity", "a" * 64), ("compiler_identity", "invalid"),
    ("clock", "cpu_time"), ("observation_prefix", "ordinary timing"), ("limitations", []),
    ("call_sites", []), ("extra", 1),
])
def test_measurement_contract_mismatch_rejects_even_after_rehash(tmp_path, field, value):
    manifest = fixture(tmp_path)
    record = manifest["scopes"][0]["owner_measurement"]
    record[field] = value
    seal(record)
    with pytest.raises(ValueError, match="Compiler owner measurement"):
        validate_owner_measurements(manifest)


@pytest.mark.parametrize("mutation", ["identity", "top_variant", "scope_count", "missing_record", "automatic",
                                       "native_runtime", "native_cuda", "native_helper", "native_extra_source"])
def test_native_exception_cannot_relax_ordinary_runtime_requirements(tmp_path, mutation):
    manifest = fixture(tmp_path)
    if mutation == "identity":
        manifest["scopes"][0]["owner_measurement"]["identity"] = "a" * 64
    elif mutation == "top_variant":
        del manifest["scope_measurement"]
    elif mutation == "scope_count":
        manifest["scope_count"] = 0
    elif mutation == "missing_record":
        del manifest["scopes"][0]["owner_measurement"]
    elif mutation == "automatic":
        manifest["automatic_estimate_available"] = True
    elif mutation == "native_runtime":
        manifest["runtime"] = {"runtime": "resident"}
    elif mutation == "native_cuda":
        manifest["build_sources"][0]["language"] = "cuda"
    elif mutation == "native_helper":
        manifest["build_sources"][0]["role"] = "common_runtime"
    else:
        manifest["sources"][str(tmp_path / "unrelated.f90")] = {"replacement": "sources/unrelated.f90"}
    with pytest.raises(ValueError, match="Compiler owner measurement"):
        validate_owner_measurements(manifest)


@pytest.mark.parametrize("mutation", ["caller_source", "source_digest", "call_digest", "line", "duplicate", "file_changed"])
def test_call_sites_are_bound_to_exact_original_source(tmp_path, mutation):
    manifest = fixture(tmp_path)
    record = manifest["scopes"][0]["owner_measurement"]
    call = record["call_sites"][0]
    if mutation == "caller_source":
        call["source"] = str(tmp_path / "different.f90")
    elif mutation == "source_digest":
        call["source_sha256"] = "a" * 64
    elif mutation == "call_digest":
        call["call_sha256"] = "a" * 64
    elif mutation == "line":
        call["first_line"] = call["last_line"] = 1
    elif mutation == "duplicate":
        record["call_sites"].append(deepcopy(call))
    else:
        Path(call["source"]).write_text("subroutine caller\ncall other()\nend subroutine\n")
    call["identity"] = digest({key: value for key, value in call.items() if key != "identity"})
    seal(record)
    with pytest.raises(ValueError, match="Compiler owner measurement"):
        validate_owner_measurements(manifest)


def test_resident_runtime_identity_is_authenticated(tmp_path):
    manifest = fixture(tmp_path, "resident")
    manifest["runtime"]["runtime_id"] = "changed"
    with pytest.raises(ValueError, match="runtime identity"):
        validate_owner_measurements(manifest)


def test_public_compiler_measurement_variants_apply_and_compose(tmp_path):
    from elmm_pipeline.local_adapter import compose_scope_packages, transform_scopes

    original = Path(os.environ.get("ELMM_TEST_COMPILER_ROOT", Path(__file__).resolve().parents[2]))
    if not (original / "compiler/__main__.py").is_file():
        pytest.skip("Set ELMM_TEST_COMPILER_ROOT to an explicit compiler checkout")
    compiler = tmp_path / "compiler-checkout"
    shutil.copytree(original / "compiler", compiler / "compiler",
                    ignore=shutil.ignore_patterns("__pycache__", ".*cache"))
    source = tmp_path / "application.f90"
    source.write_text("""module measurement_application
contains
subroutine step(a,b,n)
integer,intent(in)::n
real(8),intent(in)::a(n)
real(8),intent(out)::b(n)
integer::i
!$omp parallel do
do i=1,n
b(i)=2*a(i)
enddo
!$omp end parallel do
end subroutine
subroutine invoke(a,b,n)
integer,intent(in)::n
real(8),intent(in)::a(n)
real(8),intent(out)::b(n)
call step(a,b,n)
end subroutine
end module
""")
    stable = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial", "captures": {
        "argument::a": stable, "argument::b": {**stable, "initialized": "none"}}}
    records = []
    for variant in ("native", "resident"):
        metadata = transform_scopes(source, compiler, Path(sys.executable), tmp_path / variant,
            entry="measurement_application::step", capture_facts=facts,
            gpu_options={"gpu_policy": "sections", "scope_execution": "reached", "scope_measurement": variant})
        manifest = metadata["scope_manifest"]
        record = manifest["scopes"][0]["owner_measurement"]
        records.append(record)
        assert metadata["measurement_only"] is True and metadata["scope_measurement"] == variant
        assert metadata["command"][metadata["command"].index("--scope-measurement") + 1] == variant
        assert record["calibration_evidence"] is record["placement_authority"] is False
        target = tmp_path / (variant + "-copy.f90")
        target.write_bytes(source.read_bytes())
        composition = compose_scope_packages([metadata], {str(source): target})
        assert composition["measurement_only"] is True
        assert "FORT_SCOPE_MEASUREMENT" in target.read_text()
        if variant == "native":
            assert manifest["runtime"] is None
            assert metadata["gpu_sources"] == metadata["extra_sources"] == []
            assert "fort_scope_owner_" not in target.read_text()
        else:
            assert manifest["runtime"] is not None and metadata["gpu_sources"]
    assert records[0]["identity"] == records[1]["identity"]

    from elmm_pipeline.adaptation import adapt_project

    application = tmp_path / "application"
    application.mkdir()
    (application / source.name).write_bytes(source.read_bytes())
    with patch("elmm_pipeline.gpu_toolchain.compile_gpu_sources", side_effect=AssertionError("native has no CUDA")):
        adapter = adapt_project("local", application, Path(sys.executable), tmp_path / "native-adaptation",
            compiler_root=compiler, target="gpu", memory_model="scoped",
            gpu_options={"gpu_policy": "sections", "scope_execution": "reached", "scope_measurement": "native"},
            scope_options={"entry": "measurement_application::step", "source": source.name, "capture_facts": facts})
    assert adapter["measurement_only"] and adapter["scope_measurement"] == "native"
    assert adapter["native_success"] is True and adapter["optimized_regions"] == 0
    assert adapter["gpu_sources"] == adapter["extra_sources"] == []
