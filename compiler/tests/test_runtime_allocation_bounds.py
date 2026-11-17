"""Normalized numerical entries use guarded original allocation descriptors."""

from __future__ import annotations

import pytest

from compiler.offload.config import OffloadConfig
from compiler.driver.options import CompilerOptions
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.numerical import load_numerical_sources
from compiler.scopes.source import form_source_scopes
from compiler.tests.test_numerical_sources import NORMALIZED, ORIGINAL, package
from compiler.tests.test_source_scopes import FACT


def _package(directory, *, rank=1, descriptor_only=False, shadowed=None):
    original = ORIGINAL.replace("real(8) :: weights(32)", "real(8),allocatable :: weights(:)")
    normalized = NORMALIZED
    if shadowed:
        original = original.replace("integer::i\n!$omp", "integer::i," + shadowed + "\n!$omp", 1)
        original = original.replace("!$omp parallel do private(i)\n", "").replace("!$omp end parallel do\n", "")
    if rank == 2:
        original = original.replace("weights(:)", "weights(:,:)").replace("weights(i+3)", "weights(i+3,2)")
        normalized = normalized.replace("weights(:)", "weights(:,:)")
        normalized = normalized.replace("alb,blb,wlb", "alb,blb,wlb,wlb2")
        normalized = normalized.replace("weights(i+3-wlb+1)", "weights(i+3-wlb+1,2-wlb2+1)")
    if descriptor_only:
        original = original.replace("weights(i+3)", "real(lbound(weights,1),8)")
        normalized = normalized.replace(
            "subroutine producer(a,b,n,gain,weights,alb,blb,wlb)",
            "subroutine producer(a,b,n,gain,alb,blb,wlb)").replace("a(:),weights(:),gain", "a(:),gain")
        normalized = normalized.replace("weights(i+3-wlb+1)", "real(wlb,8)")
    source, numerical, document = package(directory, original=original, normalized=normalized)
    parameters = document["entries"][0]["parameters"]
    if rank == 2:
        next(p for p in parameters if p["name"] == "weights")["physical_origin"] = [0, 0]
        parameters.append({"name": "wlb2", "resource": "settings::weights", "lower_bound_dimension": 2})
    if descriptor_only:
        document["entries"][0]["parameters"] = [p for p in parameters if p["name"] != "weights"]
    return source, numerical, document


def _analysis(source):
    analysis = SourceEffects([source])
    analysis.authorize_stable_module_allocatables({"settings::weights"})
    return analysis


def _generate(directory, *, rank=1, descriptor_only=False, shadowed=None):
    source, _, document = _package(directory, rank=rank, descriptor_only=descriptor_only, shadowed=shadowed)
    facts = {"schema_version": 1, "participation": "serial", "sources": document["source_inputs"],
             "captures": {root: dict(FACT) for root in
                          ("argument::a", "argument::b", "argument::out", "settings::weights")}}
    outputs, report = form_source_scopes([source], "original::step", facts=facts,
                                         options=CompilerOptions(opt_level=1),
                                         config=OffloadConfig(policy="sections"), numerical_sources=document)
    assert report["scope_count"] == 1, report["boundaries"]
    return outputs[report["sources"][str(source)]["replacement"]], report


@pytest.mark.parametrize("rank", [1, 2])
def test_package_keeps_actual_origins_separate_from_declared_origins(tmp_path, rank):
    source, _, document = _package(tmp_path, rank=rank)
    entries = load_numerical_sources(document, _analysis(source))
    entry = entries["original::producer"]
    assert entry.runtime_origins == {"settings::weights"}
    runtime = [p for p in entry.parameters if p.runtime_lower_bound]
    assert [p.lower_bound_dimension for p in runtime] == list(range(1, rank + 1))
    assert all(not p.runtime_lower_bound for p in entry.parameters if p.resource.startswith("argument::"))


@pytest.mark.parametrize("rank", [1, 2])
def test_dynamic_package_requires_every_origin_axis(tmp_path, rank):
    source, _, document = _package(tmp_path, rank=rank)
    next(p for p in document["entries"][0]["parameters"] if p["name"] == "wlb")["resource"] = "argument::a"
    with pytest.raises(CompilationError, match="every original runtime lower-bound dimension"):
        load_numerical_sources(document, _analysis(source))


def test_dynamic_package_rejects_duplicate_origin_axis(tmp_path):
    source, _, document = _package(tmp_path, rank=2)
    next(p for p in document["entries"][0]["parameters"] if p["name"] == "wlb2")["lower_bound_dimension"] = 1
    with pytest.raises(CompilationError, match="unique lower-bound dimension"):
        load_numerical_sources(document, _analysis(source))
