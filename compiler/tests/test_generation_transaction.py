"""Rejected source candidates cannot retain generation or proof-cache state."""

from copy import copy, deepcopy
from hashlib import sha256
from types import MethodType, SimpleNamespace

import pytest

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.emission import GeneratedSources
from compiler.ir import CompilationError
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_module_optional_scalar_companions import ENTRY, optional_sources
from compiler.tests.test_reached_window_owner import window_fixture
from compiler.tests.test_source_scopes import FACT


def builder_fixture(directory, *, cache=None):
    directory.mkdir()
    sources = optional_sources(calls=("call renamed(escape=escape,n=n,option=option,x=b)",))
    paths = []
    for name, source in sources.items():
        path = directory / name
        path.write_text(source)
        paths.append(path)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {"argument::" + name: deepcopy(FACT) for name in ("a", "b", "out")}}
    return ScopeBuilder(paths, ENTRY, facts=facts, options=CompilerOptions(),
                        config=OffloadConfig(policy="sections", scope_execution="reached"),
                        summary_cache=cache)


def assert_original_attributes(owner, original):
    assert vars(owner).keys() == original.keys()
    for name, value in original.items():
        assert getattr(owner, name) is value, name


def test_real_companion_attempt_then_failure_leaves_all_parent_state_unchanged(tmp_path):
    parent = builder_fixture(tmp_path / "source")
    original, analysis = dict(vars(parent)), dict(vars(parent.analysis))
    registry = dict(vars(parent.inline))
    variants, provenance = parent.variants.public(), parent.runtime_provenance.public()

    def rejected_attempt():
        with GenerationTransaction(parent) as transaction:
            candidate = transaction.builder
            assert candidate.analysis is parent.analysis
            assert candidate.entry is parent.entry
            _outputs, report = candidate.run()
            owner, = report["scopes"]
            child, = owner["module_coordinators"]
            assert child["procedure"] == "shifted_leaf::adjust"
            assert candidate.region_owners[child["procedure"]].generated
            assert candidate.outputs
            assert candidate.edits
            assert candidate.resolved_calls
            assert candidate.runtime_provenance.records
            assert candidate.variants.checkpoint()
            assert not parent.outputs
            assert not parent.edits
            assert not parent.scopes
            assert not parent.resolved_calls
            assert not parent.coordinators
            raise CompilationError("reject after original companion")

    with pytest.raises(CompilationError, match="reject after original companion"):
        rejected_attempt()

    assert_original_attributes(parent, original)
    assert_original_attributes(parent.analysis, analysis)
    assert_original_attributes(parent.inline, registry)
    assert parent.variants.public() == variants
    assert parent.runtime_provenance.public() == provenance
    assert parent.region_owners == {parent.entry.qualified: parent.inline}
    assert not parent.scopes
    assert not parent.boundaries
    assert not parent.coordinator_reasons


def test_nested_payloads_and_all_inline_registry_containers_are_detached(tmp_path):
    parent = builder_fixture(tmp_path / "source")
    prepared = parent.inline.prepare(tuple(parent.entry.execution.content))
    assert prepared
    assert parent.inline.regions
    parent.outputs["retained"] = "original"
    parent.edits[parent.entry.scope.path] = [(1, 0, "original")]
    parent.scopes.append({"nested": {"units": [1]}})
    parent.boundaries.append({"nested": ["original"]})
    parent.resolved_calls["retained"] = {"actuals": ["original"]}
    parent.generated["retained"] = GeneratedSources("", "", "", scoped={"nested": [1]})
    parent.analysis.summaries["retained"] = {"nested": [1]}
    fact = deepcopy(parent.facts)
    region = next(iter(parent.inline.regions))
    completion = deepcopy(parent.inline.regions[region].completion)
    registries = {name: value for name, value in vars(parent.inline).items()
                  if isinstance(value, (dict, list, set))}

    with GenerationTransaction(parent) as transaction:
        candidate = transaction.builder
        assert candidate.inline.builder is candidate
        assert candidate.region_owners[parent.entry.qualified] is candidate.inline
        for name, original in registries.items():
            assert getattr(candidate.inline, name) is not original, name
        candidate.inline.regions[region].completion["candidate_only"] = [1]
        assert candidate.inline.regions[region].bindings[0] is parent.inline.regions[region].bindings[0]
        assert candidate.inline.nodes[region] is parent.inline.nodes[region]
        candidate.facts["captures"]["argument::a"]["initialized"] = "none"
        candidate.outputs["retained"] = "candidate"
        candidate.edits[parent.entry.scope.path].append((2, 1, "candidate"))
        candidate.scopes[0]["nested"]["units"].append(2)
        candidate.boundaries[0]["nested"].append("candidate")
        candidate.resolved_calls["retained"]["actuals"].append("candidate")
        candidate.generated["retained"].scoped["nested"].append(2)
        candidate.analysis.summaries["retained"]["nested"].append(2)
        candidate.analysis._stable_module_allocatables = frozenset({"candidate::only"})
        candidate.analysis._candidate_registry = {"temporary": True}

    assert parent.facts == fact
    assert parent.outputs == {"retained": "original"}
    assert parent.edits[parent.entry.scope.path] == [(1, 0, "original")]
    assert parent.scopes == [{"nested": {"units": [1]}}]
    assert parent.boundaries == [{"nested": ["original"]}]
    assert parent.resolved_calls == {"retained": {"actuals": ["original"]}}
    assert parent.generated["retained"].scoped == {"nested": [1]}
    assert parent.inline.regions[region].completion == completion
    assert parent.analysis.summaries["retained"] == {"nested": [1]}
    assert parent.analysis.stable_module_allocatables == frozenset()
    assert not hasattr(parent.analysis, "_candidate_registry")


def test_bound_capture_closure_uses_only_the_forked_coordinator(tmp_path):
    parent = builder_fixture(tmp_path / "source")
    holder = SimpleNamespace(builder=copy(parent), root=parent, state={"visits": []})

    def capture(proxy, binding):
        holder.state["visits"].append(binding.root)
        return holder.root.facts["captures"][binding.root]

    holder.builder.capture = MethodType(capture, holder.builder)
    parent.coordinators["retained"] = holder
    binding = parent.entry.scope.bindings["a"]
    with GenerationTransaction(parent) as transaction:
        child = transaction.builder.coordinators["retained"]
        assert child is not holder
        assert child.root is transaction.builder
        assert child.builder.analysis is parent.analysis
        child.builder.capture(binding)
        assert child.state["visits"] == [binding.root]
        assert holder.state["visits"] == []
        transaction.adopt()
    adopted = parent.coordinators["retained"]
    assert adopted.root is parent
    adopted.builder.capture(binding)
    assert adopted.state["visits"] == [binding.root, binding.root]
    assert holder.state["visits"] == []


def test_registered_original_tokens_and_syntax_keep_exact_identity(tmp_path):
    _path, parent, proof, originals = window_fixture(tmp_path / "source")
    original_cache = parent.analysis._reached_window_proofs
    with GenerationTransaction(parent) as transaction:
        candidate = transaction.builder
        assert candidate.analysis is parent.analysis
        assert candidate.entry.execution is parent.entry.execution
        assert candidate.entry.scope.bindings["a"] is parent.entry.scope.bindings["a"]
        assert candidate.analysis._reached_window_proofs is not original_cache
        assert candidate.analysis._reached_window_proofs[proof.identity] is proof
        assert proof.validate(candidate.analysis, candidate.entry.qualified, originals) is proof
    assert parent.analysis._reached_window_proofs is original_cache
    assert proof.validate(parent.analysis, parent.entry.qualified, originals) is proof


def test_explicit_adoption_accepts_complete_real_companion_state(tmp_path):
    parent = builder_fixture(tmp_path / "source")
    original_facts, analysis = parent.facts, parent.analysis
    with GenerationTransaction(parent) as transaction:
        _outputs, report = transaction.builder.run()
        transaction.adopt()
    assert parent.analysis is analysis
    assert parent.facts == original_facts
    assert parent.inline.builder is parent
    assert parent.region_owners[parent.entry.qualified] is parent.inline
    child = parent.region_owners["shifted_leaf::adjust"]
    assert child.builder.analysis is analysis
    assert child.builder.inline is child
    assert parent.scopes == report["scopes"]
    assert parent.outputs
    assert parent.edits
    assert parent.resolved_calls
    assert parent.variants.checkpoint()
    assert parent.runtime_provenance.records
    assert not hasattr(parent, "_generation_transaction_marker")


def test_window_builder_projection_can_be_adopted_without_copying_its_proof(tmp_path):
    _path, parent, proof, originals = window_fixture(tmp_path / "source")
    with GenerationTransaction(parent) as transaction:
        owner = ReachedWindowOwner(transaction.builder, proof, originals)
        assert owner.run()
        transaction.adopt(owner.builder)
    assert parent.validate_reached_window() is proof
    assert parent.inline.builder is parent
    assert parent.region_owners[parent.entry.qualified] is parent.inline
    _outputs, report = parent.finish()
    assert report["scopes"][0]["reached_window"]["proof_identity"] == proof.identity


@pytest.mark.parametrize("adopt", [False, True])
def test_speculation_never_writes_persistent_summary_cache(tmp_path, adopt):
    cache_path = tmp_path / "summary-cache"
    parent = builder_fixture(tmp_path / "source", cache=cache_path)
    cache, stats = parent.analysis._summary_cache, parent.analysis._summary_cache.stats
    with GenerationTransaction(parent) as transaction:
        transaction.builder.analysis.structure(parent.entry.qualified)
        assert transaction.builder.analysis._summary_cache.directory is None
        assert not cache_path.exists()
        if adopt:
            transaction.adopt()
    assert parent.analysis._summary_cache is cache
    assert not cache_path.exists()
    if adopt:
        assert cache.stats["stores"] > stats["stores"]
    else:
        assert cache.stats == stats


def test_exception_after_adoption_still_rejects(tmp_path):
    parent = builder_fixture(tmp_path / "source")
    analysis = dict(vars(parent.analysis))
    def rejected_attempt():
        with GenerationTransaction(parent) as transaction:
            transaction.builder.outputs["candidate"] = "only"
            transaction.adopt()
            raise RuntimeError("late failure")

    with pytest.raises(RuntimeError, match="late failure"):
        rejected_attempt()
    assert not parent.outputs
    assert_original_attributes(parent.analysis, analysis)


def test_foreign_projection_and_source_fact_changes_cannot_commit(tmp_path):
    parent = builder_fixture(tmp_path / "source")
    with GenerationTransaction(parent) as transaction:
        with pytest.raises(CompilationError, match="same original"):
            transaction.adopt(copy(parent))
        child = copy(transaction.builder)
        child.entry = parent.analysis.routines["shifted_leaf::adjust"]
        with pytest.raises(CompilationError, match="same original"):
            transaction.adopt(child)
    fact = deepcopy(parent.facts)
    def rejected_attempt():
        with GenerationTransaction(parent) as transaction:
            transaction.builder.facts["captures"]["argument::a"]["initialized"] = "none"
            transaction.adopt()

    with pytest.raises(CompilationError, match="independently admitted"):
        rejected_attempt()
    assert parent.facts == fact


def test_transactions_are_single_use_and_same_analysis_cannot_overlap(tmp_path):
    parent = builder_fixture(tmp_path / "source")
    transaction = GenerationTransaction(parent)
    with transaction:
        with pytest.raises(CompilationError, match="overlap"), GenerationTransaction(parent):
            pytest.fail("overlapping transaction entered")
        transaction.builder.outputs["candidate"] = "only"
    assert not parent.outputs
    with pytest.raises(CompilationError, match="single-use"), transaction:
        pytest.fail("closed transaction entered")
    with pytest.raises(CompilationError, match="active transaction"):
        transaction.adopt()
