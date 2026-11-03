"""Actual hierarchical emission matches the independent full-validation path."""
import json
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.frontend.immutable_graph import original_type
from compiler.ir import CompilationError
from compiler.scopes import storage_admission
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.hierarchical_regions import HierarchicalRegionOwner
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.storage_epoch import StorageEpochProof
from compiler.tests.test_empty_member_edit_boundaries import empty_source
from compiler.tests.test_hierarchical_child_regions import fixture, owner_at
from compiler.tests.test_inherited_lexical_namespace import scalar_element_source, source_fixture


def original_source(kind):
    if kind == "readonly":
        return source_fixture(native_call=False)
    if kind == "native_write":
        return source_fixture(native_call=True)
    if kind == "scalar_element":
        return scalar_element_source()
    if kind == "empty_six":
        return empty_source(arity=6, nested=True)
    if kind == "atomic_varying":
        from compiler.tests.test_same_family_section_composition_cuda import source

        return source()
    raise AssertionError(kind)


def emit(directory, parent, source, kind, monkeypatch, enabled, *, fault=None):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    builder = parent
    path = directory / "source.f90"
    facts = deepcopy(builder.facts)
    original_scope = storage_admission.emission_admission
    original_full, original_unit = StorageEpochProof.validate, LexicalOwner.emit_unit
    spans, units, span = [], [], None
    changed = False

    @contextmanager
    def actual_scope(plan):
        nonlocal span
        assert span is None
        assert original_type(plan._owner) is HierarchicalRegionOwner
        assert getattr(plan._owner, "position_epoch", None) is None
        span = {"plan": plan, "full_checks": 0}
        spans.append(span)
        try:
            with original_scope(plan) if enabled else nullcontext():
                yield
        finally:
            span = None

    def full(self, plan):
        if span is not None and plan is span["plan"]:
            span["full_checks"] += 1
        return original_full(self, plan)

    def unit(self, current, index):
        nonlocal changed
        admission = storage_admission.active_admission()
        assert span is not None
        assert (admission is not None) is enabled
        if enabled:
            assert admission._plan is self.owner.execution_plan is span["plan"]
        if fault is not None and not changed:
            member, task = next((member, task) for member in self.owner.execution_plan.members
                                for task in member.source_device_tasks)
            if fault == "nodes":
                object.__setattr__(task, "nodes", tuple([*task.nodes]))
            elif fault == "proof":
                object.__setattr__(task, "proof", object())
            elif fault == "placement":
                object.__setattr__(task, "placement", object())
            elif fault == "registry":
                member.builder.analysis._atomic_worksharing_teams.pop(task.proof.identity)
            else:
                raise AssertionError(fault)
            changed = True
        units.append((self.routine.qualified, index))
        return original_unit(self, current, index)

    with monkeypatch.context() as patch:
        patch.setattr(storage_admission, "emission_admission", actual_scope)
        patch.setattr(StorageEpochProof, "validate", full)
        patch.setattr(LexicalOwner, "emit_unit", unit)
        owner = owner_at(builder)
        nodes = builder.inline.prepare(owner.closed_selection)
        builder.generated.update(builder.inline.generated)
        builder.numerical_ir.update(builder.inline.ir)
        builder.numerical_reasons.update({name: None for name in builder.inline.regions})
        assert owner.run(nodes)
        outputs, report = builder.finish(materialize_native_effects=False)
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    assert len(spans) == 1
    assert len(units) >= 2
    assert storage_admission.active_admission() is None
    assert {name for name, _index in units} >= {"hierarchical::producer", "hierarchical::consumer"}
    package = directory / ("enabled-package" if enabled else "disabled-package")
    for name, content in outputs.items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    manifest = json.loads(outputs["scope-manifest.json"])
    validate_checked_execution_plans(manifest, capture_facts=builder.facts)
    _scope_package_build({"artifact_root": str(package), "scope_manifest": manifest})
    assert builder.finish()
    builder.analysis.inputs.verify_live()
    assert path.read_text() == source
    assert builder.facts == facts
    return path, builder, outputs, manifest, units, spans[0]["full_checks"], facts


@pytest.mark.parametrize("kind", ["readonly", "native_write", "scalar_element", "empty_six", "atomic_varying"])
def test_real_hierarchical_emission_keeps_identical_sources_and_manifest(tmp_path, monkeypatch, kind):
    from compiler.frontend import source_effects
    from compiler.frontend.summary_cache import SummaryCache

    source = original_source(kind)
    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    _path, parent = fixture(tmp_path, source)
    if kind == "atomic_varying":
        parent.config = replace(parent.config, host_threads=4)
    # Registered original-source queries are monotonic across rejected
    # generation branches. Materialize them through one ordinary full run so
    # both comparisons start from the same parsed authority and cache state.
    for _warm in range(2):
        with GenerationTransaction(parent) as branch:
            emit(tmp_path, branch.builder, source, kind, monkeypatch, enabled=False)
    with GenerationTransaction(parent) as branch:
        first = emit(tmp_path, branch.builder, source, kind, monkeypatch, enabled=False)
    with GenerationTransaction(parent) as branch:
        second = emit(tmp_path, branch.builder, source, kind, monkeypatch, enabled=True)
    assert first[0] == second[0]
    assert first[6] == second[6]
    assert first[4] == second[4]
    assert first[2] == second[2]
    assert first[5] > second[5] == 2
    first[1].analysis.inputs.verify_live()
    second[1].analysis.inputs.verify_live()
    assert sha256(first[0].read_bytes()).hexdigest() == first[6]["sources"][str(first[0])]
    if kind == "atomic_varying":
        epoch = second[3]["scopes"][0]["checked_execution_plan"]["storage_epoch"]
        tasks = [task for member in epoch["members"] for task in member.get("source_device_tasks", [])]
        assert any(task["kind"] == "original_atomic_conditional_joined_team" for task in tasks)


@pytest.mark.parametrize("fault", ["nodes", "proof", "placement", "registry"])
def test_actual_atomic_emission_rejects_changed_task_before_storage_membership(tmp_path, monkeypatch, fault):
    source = original_source("atomic_varying")
    _path, parent = fixture(tmp_path, source)
    parent.config = replace(parent.config, host_threads=4)
    with pytest.raises(CompilationError), GenerationTransaction(parent) as branch:
        emit(tmp_path, branch.builder, source, "atomic_varying", monkeypatch, enabled=True, fault=fault)
    assert storage_admission.active_admission() is None
    parent.analysis.inputs.verify_live()
    assert not parent.outputs
    assert not parent.scopes
