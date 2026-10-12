"""Original cost metadata survives emission and shared artifact dispatch."""

from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.emission import generate_sources
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_current_source_compute import bundle as current_bundle
from compiler.tests.test_current_source_compute import prepared_source
from compiler.tests.test_dependency_source import BODY
from compiler.tests.test_source_scopes import FACT


@pytest.fixture(scope="module")
def bundle():
    yield from current_bundle.__wrapped__()


def test_current_costs_reach_public_queries_from_original_preparation(tmp_path, bundle):
    _, _, prepared, applicability = prepared_source(tmp_path)
    generated = generate_sources(prepared.function, prepared.plan,
        offload_config=OffloadConfig(policy="auto", profile=bundle, native_participation="serial",
                                    source_applicability=applicability), memory_model="scoped")
    public = generated.scoped
    assert public["compute_estimates"]["source_applicability"] == applicability.public()
    unit, = public["planning"]["units"]
    assert unit["compute_model"]["cost_model"] == "source_work_span_cpu_gpu_v2"
    assert unit["compute_model"]["gpu"]["launch_enqueue_included"]
    assert public["planning"]["available"]
    cuda = next(text for path, text in generated.artifacts.items()
                if path.endswith(".cu") and "record_compute(" in text)
    assert "FORT_SCOPE_COMPUTE_GPU_LAUNCH_INCLUDED" in cuda


def test_lowered_ir_alone_does_not_supply_original_source_cost_form(tmp_path, bundle):
    _, _, prepared, _ = prepared_source(tmp_path)
    generated = generate_sources(prepared.function, prepared.plan,
        offload_config=OffloadConfig(policy="auto", profile=bundle, native_participation="serial"),
        memory_model="scoped")
    assert generated.scoped["compute_estimates"]["source_applicability"] is None
    unit, = generated.scoped["planning"]["units"]
    assert unit["compute_model"] is None
    assert "proved original source compiled form" in unit["work_estimate_reason"]
    assert not generated.scoped["automatic_estimate_available"]


def test_form_metadata_cannot_bypass_unpriced_environment_protocol(tmp_path, bundle):
    _, _, prepared, applicability = prepared_source(tmp_path)
    generated = generate_sources(replace(prepared.function, requires_numerical_environment=True), prepared.plan,
        offload_config=OffloadConfig(policy="auto", profile=bundle, native_participation="serial",
                                    source_applicability=applicability), memory_model="scoped")
    assert not generated.scoped["automatic_estimate_available"]
    assert generated.scoped["automatic_reason"] == "numerical_environment_protocol_calibration_unavailable"


def inline_builder(tmp_path, bundle):
    loop = "do i=1,n\n" + BODY + "\nenddo\n"
    source = """module original_dispatch_costs
implicit none
contains
subroutine evaluate(a,b,out,n)
real(8),intent(in)::a(:),b(:)
real(8),intent(out)::out(:)
integer,intent(in)::n
integer::i
real(8)::x,y
""" + loop + loop + "end subroutine\nend module\n"
    path = tmp_path / "original.f90"
    path.write_text(source)
    builder = ScopeBuilder([path], "original_dispatch_costs::evaluate", options=CompilerOptions(),
        config=OffloadConfig(policy="auto", profile=bundle), facts={
            "schema_version": 1, "participation": "serial",
            "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
            "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}})
    builder.inline.prepare(tuple(builder.entry.execution.content))
    assert len(builder.inline.regions) == 2, builder.boundaries
    return path, builder


def test_identical_regions_share_worker_but_keep_each_original_cost_provenance(tmp_path, bundle):
    _, builder = inline_builder(tmp_path, bundle)
    inline = builder.inline
    first, second = inline.regions
    assert inline.entries[first] == inline.entries[second]
    assert inline.generated[first] is inline.generated[second]
    a, b = inline.public()["regions"]
    assert a["source_compute_applicability"]["preparation_identity"] != b["source_compute_applicability"]["preparation_identity"]
    assert a["shared_compute_applicability"] == b["shared_compute_applicability"]
    assert a["shared_compute_applicability"] == a["source_compute_applicability"]
    for region in (a, b):
        assert region["source_compute_applicability"]["extraction_source_identity"] == region["source_identity"]
    unit, = inline.generated[first].scoped["planning"]["units"]
    assert unit["compute_model"] is not None, unit["work_estimate_reason"]


def test_optional_cost_metadata_does_not_replace_hard_source_validation(tmp_path, bundle):
    path, builder = inline_builder(tmp_path, bundle)
    path.write_text(path.read_text().replace("out(i)=", "out(i)=1d0+"))
    with pytest.raises(CompilationError):
        builder.inline.original_selection(next(iter(builder.inline.nodes.values())))


def test_equal_arithmetic_with_different_cost_contracts_does_not_reuse_query(tmp_path, bundle, monkeypatch):
    from compiler.offload import source_applicability

    classify = source_applicability.source_compute_applicability
    issued = []

    def first_only(*args, **kwargs):
        value = classify(*args, **kwargs)
        issued.append(value)
        return value if len(issued) == 1 else None

    monkeypatch.setattr(source_applicability, "source_compute_applicability", first_only)
    _, builder = inline_builder(tmp_path, bundle)
    first, second = builder.inline.regions
    assert all(value is not None for value in issued)
    assert builder.inline.entries[first] != builder.inline.entries[second]
    unit, = builder.inline.generated[first].scoped["planning"]["units"]
    assert unit["compute_model"] is not None, unit["work_estimate_reason"]
    assert builder.inline.generated[second].scoped["planning"]["units"][0]["compute_model"] is None


def test_existing_uncalibrated_generation_does_not_revalidate_for_optional_costs(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("no current cost bundle consumes this classification")

    monkeypatch.setattr("compiler.offload.source_applicability.source_compute_applicability", unexpected)
    _, builder = inline_builder(tmp_path, None)
    first, second = builder.inline.regions
    assert builder.inline.entries[first] == builder.inline.entries[second]
    assert all(region["source_compute_applicability"] is None for region in builder.inline.public()["regions"])


def test_config_rejects_untyped_or_mismatched_cost_metadata(tmp_path):
    _, _, _, applicability = prepared_source(tmp_path)
    with pytest.raises(ValueError, match="original-region metadata"):
        OffloadConfig(native_participation="serial", source_applicability={})
    with pytest.raises(ValueError, match="participation differs"):
        OffloadConfig(native_participation="fork_join", source_applicability=applicability)
