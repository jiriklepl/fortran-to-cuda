"""Source variants are reusable, bounded and transactional."""

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.scopes.variants import VariantRegistry
from compiler.tests.test_source_scopes import FACT, PROGRAM, WRAPPER_PROGRAM


def register(registry, procedure="library::apply", *, interface="root_handles_v1"):
    return registry.register(procedure, interface=interface, role="call_worker", name="generated_worker",
                             summary_identity="source-proof", requirements=("source effects",),
                             shared_artifacts=("workers/interface.f90",))


def builder(tmp_path, source=PROGRAM):
    path = tmp_path / "original.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": FACT, "argument::b": {**FACT, "initialized": "none"},
                          "argument::out": {**FACT, "initialized": "none"}}}
    return ScopeBuilder([path], "original::step", facts=facts, options=CompilerOptions(),
                        config=OffloadConfig(policy="sections"))


def test_mode_bearing_variant_reuses_the_same_worker_and_artifacts():
    registry = VariantRegistry()
    first = register(registry)
    assert register(registry) is first
    public = registry.public()
    assert public["generated_count"] == 1
    worker, = public["procedures"][0]["variants"]
    assert worker["shared_artifacts"] == ["workers/interface.f90"]
    assert worker["placement"] == "runtime mode; no shape or CPU/GPU partition specialization"
    worker["requirements"].append("public mutation")
    assert registry.public()["procedures"][0]["variants"][0]["requirements"] == ["source effects"]


def test_fifth_distinct_wrapper_for_one_procedure_is_an_explicit_boundary():
    registry = VariantRegistry()
    for index in range(4):
        register(registry, interface="proved_interface_" + str(index))
    with pytest.raises(CompilationError, match="variant budget exhausted for library::apply: limit 4"):
        register(registry, interface="proved_interface_4")
    assert registry.public()["generated_count"] == 4


def test_compilation_variant_budget_is_independent_of_per_procedure_budget():
    registry = VariantRegistry()
    for index in range(128):
        register(registry, "library::procedure_" + str(index))
    with pytest.raises(CompilationError, match="variant compilation budget exhausted: limit 128"):
        register(registry, "library::additional")
    assert registry.public()["generated_count"] == 128


def test_rejected_candidate_restores_budget_and_preserves_earlier_variants():
    registry = VariantRegistry(per_procedure=1, total=2)
    retained = register(registry, "library::retained")
    before = registry.checkpoint()
    register(registry, "library::temporary")
    registry.restore(before)
    assert register(registry, "library::retained") is retained
    register(registry, "library::accepted")
    assert {item["procedure"] for item in registry.public()["procedures"]} == {
        "library::retained", "library::accepted"}


def test_conflicting_artifacts_cannot_reuse_an_old_variant_identity():
    registry = VariantRegistry()
    register(registry)
    with pytest.raises(CompilationError, match="variant identity changed"):
        registry.register("library::apply", interface="root_handles_v1", role="call_worker", name="different",
                          summary_identity="source-proof")


def test_repeated_wrapper_calls_share_mode_workers_and_numerical_artifacts(tmp_path):
    compiler = builder(tmp_path, WRAPPER_PROGRAM)
    outputs, report = compiler.run()
    assert report["scope_count"] == 1, report["boundaries"]
    variants = report["implementation_variants"]
    assert variants["limits"] == {"per_procedure": 4, "compilation": 128}
    procedures = {item["procedure"]: item["variants"] for item in variants["procedures"]}
    assert len(procedures["original::wrapper"]) == 1
    for procedure in ("original::producer", "original::consumer"):
        assert {item["role"] for item in procedures[procedure]} == {"call_worker", "numerical_entry"}
        numerical = next(item for item in procedures[procedure] if item["role"] == "numerical_entry")
        assert all(artifact in outputs for artifact in numerical["shared_artifacts"])
    owner, = procedures["original::step"]
    assert report["scopes"][0]["owner_variant"] == owner["identity"]
    assert all(item["native_entry"] == item["procedure"] for item in variants["procedures"])


def test_generation_limit_keeps_native_source_without_dangling_artifacts(tmp_path):
    compiler = builder(tmp_path)
    compiler.variants = VariantRegistry(per_procedure=1)
    outputs, report = compiler.run()
    assert report["scope_count"] == 0
    assert any("variant budget exhausted" in item["reason"] for item in report["boundaries"])
    assert report["implementation_variants"]["generated_count"] == 0
    assert report["source_edits"] == report["build_sources"] == []
    assert set(outputs) == {"scope-manifest.json"}


def test_borrowed_view_companion_reuses_its_worker_and_publishes_runtime_dependencies(tmp_path):
    compiler = builder(tmp_path)
    public, directory = compiler.entry_artifacts("original::producer", views=True)
    companion = compiler.view_generated["original::producer"]
    assert all(parameter["passing"] == "root_view_v2" for parameter in public["array_parameters"])
    assert compiler.entry_artifacts("original::producer", views=True) == (public, directory)
    assert compiler.view_generated["original::producer"] is companion
    assert all(directory + "/" + name in compiler.outputs for name in (
        "shared_entry.cu", "shared_interface.f90", "scoped_entry.hpp", "scoped_regions.hpp", "view_entry.hpp"))
    default, full_directory = compiler.entry_artifacts("original::producer")
    assert full_directory != directory
    assert all(parameter.get("passing") not in {"root_view_v1", "root_view_v2"}
               for parameter in default["array_parameters"])
    assert compiler.variants.public()["generated_count"] == 2


@pytest.mark.parametrize(("per_procedure", "total"), [(0, 128), (4, 0), (True, 128), (4, 1.5)])
def test_variant_limits_require_positive_integer_budgets(per_procedure, total):
    with pytest.raises(ValueError, match="positive integers"):
        VariantRegistry(per_procedure=per_procedure, total=total)
