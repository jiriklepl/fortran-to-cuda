"""Original cost metadata survives emission and shared artifact dispatch."""

from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.offload.config import OffloadConfig
from compiler.driver.options import CompilerOptions
from compiler.emission import generate_sources
from compiler.ir import CompilationError
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_current_source_compute import bundle as current_bundle
from compiler.tests.test_current_source_compute import prepared_source
from compiler.tests.test_dependency_source import BODY
from compiler.tests.test_source_scopes import FACT


@pytest.fixture(scope="module")
def bundle():
    yield from current_bundle.__wrapped__()


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


def test_optional_cost_metadata_does_not_replace_hard_source_validation(tmp_path, bundle):
    path, builder = inline_builder(tmp_path, bundle)
    facade = next(iter(builder.inline.nodes.values()))
    selection = builder.inline.original_selection(facade)
    path.write_text(path.read_text().replace("out(i)=", "out(i)=1d0+"))
    assert builder.inline.original_selection(facade) == selection
    with pytest.raises(CompilationError, match="source changed"):
        builder.finish()
    assert not builder.outputs
    assert not builder.edits


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
