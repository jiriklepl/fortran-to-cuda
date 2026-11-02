"""Ordinary sealed invocation admission retains storage ownership, not task credit."""

import json

import pytest

from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.storage_admission import (
    StorageAdmission,
    _supported,
    active_admission,
    admit_emitted_storage,
    emission_admission,
)
from compiler.scopes.storage_epoch import StorageEpochProof
from compiler.tests.test_atomic_source_call_bindings import SOURCE, atomic_builder
from compiler.tests.test_checked_execution_plan import scanned
from compiler.tests.test_graph_epoch_capture_batch import issued
from compiler.tests.test_storage_emission_admission import query
from compiler.tests.test_tile_private import SOURCE as TILE_SOURCE


@pytest.fixture
def final_plan(tmp_path):
    path, owner, _epoch, _bindings = issued(tmp_path)
    assert owner.run(owner.closed_selection)
    owner.builder._graph_storage_epoch = owner.position_epoch
    return path, owner, owner.execution_plan


def test_ordinary_invocation_activates_and_keeps_identical_record(tmp_path, monkeypatch):
    with scanned(tmp_path, config=OffloadConfig(policy="sections", scope_execution="graph")) as (owner, plan):
        assert getattr(owner, "position_epoch", None) is None
        assert _supported(plan)
        proof = plan.storage_epoch()
        before = proof._record
        before_public = proof.public()
        member, bindings = query(plan)
        expected = proof.admits_many(plan, member, bindings)
        original, calls = StorageEpochProof.validate, []

        def full(self, current):
            calls.append(current)
            return original(self, current)

        monkeypatch.setattr(StorageEpochProof, "validate", full)
        for _ in range(30):
            assert proof.admits_many(plan, member, bindings) == expected
        assert calls == [plan] * 30
        calls.clear()
        with emission_admission(plan) as admission:
            assert active_admission() is admission
            for _ in range(30):
                assert admit_emitted_storage(proof, plan, member, bindings) == expected
            assert calls == [plan]
        assert calls == [plan, plan]
        assert proof._record == before
        assert proof.public() == before_public
        assert active_admission() is None


def test_last_live_source_callback_cannot_remove_registry_before_return(final_plan, monkeypatch):
    _path, owner, plan = final_plan
    member, bindings = query(plan)
    analysis = owner.builder.analysis
    inputs = analysis.inputs
    original = type(inputs).verify_live
    calls = []

    def changed(self):
        result = original(self)
        if self is inputs:
            calls.append(self)
            if len(calls) == 3:
                analysis._descriptor_window_proofs.pop(owner.position_epoch._descriptor.identity)
        return result

    def inspect():
        with StorageAdmission(plan) as admission:
            monkeypatch.setattr(type(inputs), "verify_live", changed)
            admission.admits_many(plan.storage_epoch(), plan, member, bindings)
            pytest.fail("the last publication callback must reject before canonical membership returns")

    with pytest.raises(CompilationError, match="retained"):
        inspect()
    assert len(calls) == 3
    assert active_admission() is None


@pytest.mark.parametrize("tile", [False, True])
def test_actual_atomic_and_tile_companions_enter_ordinary_invocation_scope(tmp_path, monkeypatch, tile):
    text = SOURCE.replace(
        "call relay(a,p,n,flag)\ncall relay(b,q,n,flag)",
        """do i=1,n
a(i)=2*p(i)
b(i)=3*q(i)
enddo
call relay(a,p,n,flag)
call relay(b,q,n,flag)
do i=1,n
a(i)=a(i)+p(i)
enddo""",
    )
    text = text.replace("logical,intent(in)::flag\ncall relay", "logical,intent(in)::flag\ninteger::i\ncall relay")
    text = text.replace("logical,intent(in)::flag\ndo i=1,n", "logical,intent(in)::flag\ninteger::i\ndo i=1,n")
    if tile:
        routine = TILE_SOURCE.split("contains\n", 1)[1].rsplit("end module", 1)[0]
        text = text.replace(
            "call relay(b,q,n,flag)\ndo i=1,n", "call relay(b,q,n,flag)\ncall advance(p,a,n,4)\ndo i=1,n"
        )
        text = text.rsplit("end module", 1)[0] + routine + "end module\n"
    events, owners = [], []
    original_enter, original_exit = StorageAdmission.__enter__, StorageAdmission.__exit__
    original_full = StorageEpochProof.validate

    def enter(self):
        owners.append(self._owner)
        assert getattr(self._owner, "position_epoch", None) is None
        events.append("enter")
        return original_enter(self)

    def leave(self, *args):
        try:
            return original_exit(self, *args)
        finally:
            events.append("exit")

    def full(self, plan):
        if events and events[-1] != "exit":
            events.append("full")
        return original_full(self, plan)

    monkeypatch.setattr(StorageAdmission, "__enter__", enter)
    monkeypatch.setattr(StorageAdmission, "__exit__", leave)
    monkeypatch.setattr(StorageEpochProof, "validate", full)
    builder = atomic_builder(tmp_path, source=text)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report["graph_owner_attempt"]
    assert events == ["enter", "full", "full", "exit"]
    assert len(owners) == 1
    assert sum(len(member.units) for member in owners[0].execution_plan.members) >= 2
    tasks = report["scopes"][0]["source_device_tasks"]
    assert any(item["kind"] == "original_atomic_conditional_joined_team" for item in tasks)
    if tile:
        assert len(tasks) == 2
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    validate_checked_execution_plans(json.loads(outputs["scope-manifest.json"]), capture_facts=builder.facts)

    from compiler.scopes import source_device_tasks
    from compiler.scopes.atomic_joined_team import AtomicJoinedTeamPatch

    plan = owners[0].execution_plan
    proof = plan.storage_epoch()
    member, bindings = query(plan)
    before = proof._record
    counts = {"atomic_public": 0, "tile_provider": 0}
    atomic_public, tile_provider = AtomicJoinedTeamPatch.public, source_device_tasks._provider

    def atomic(self, current):
        counts["atomic_public"] += 1
        return atomic_public(self, current)

    def provider(*args, **kwargs):
        counts["tile_provider"] += 1
        return tile_provider(*args, **kwargs)

    monkeypatch.setattr(AtomicJoinedTeamPatch, "public", atomic)
    monkeypatch.setattr(source_device_tasks, "_provider", provider)
    for _ in range(5):
        proof.admits_many(plan, member, bindings)
    direct = dict(counts)
    counts.update(atomic_public=0, tile_provider=0)
    with emission_admission(plan):
        for _ in range(5):
            admit_emitted_storage(proof, plan, member, bindings)
    assert direct["atomic_public"] == 5
    assert counts["atomic_public"] == 2
    if tile:
        assert direct["tile_provider"] == 5
        assert counts["tile_provider"] == 2
    assert proof._record == before

    # The aliased legacy owner marker is an exact source authority too.
    member = next(member for member in plan.members if member.source_device_tasks)
    atomic_task = next(task for task in member.source_device_tasks if isinstance(task, AtomicJoinedTeamPatch))
    analysis = atomic_task.proof._analysis

    def inspect():
        with StorageAdmission(plan) as admission:
            analysis._atomic_worksharing_owner = object()
            admission.validate(proof, plan)

    with pytest.raises(CompilationError, match="retained"):
        inspect()


@pytest.mark.parametrize("fault", ["host_metadata", "host_only_reads", "native", "tree", "guarded"])
def test_ordinary_storage_role_topology_cannot_change_before_membership(tmp_path, fault):
    with scanned(tmp_path, config=OffloadConfig(policy="sections", scope_execution="graph")) as (owner, plan):
        member, bindings = query(plan)
        unit = next(unit for unit in owner.units if unit.scope.native)

        def inspect():
            with StorageAdmission(plan) as admission:
                if fault == "host_metadata":
                    object.__setattr__(unit, "host_metadata", dict(unit.host_metadata or {}))
                elif fault == "host_only_reads":
                    operation = unit.scope.native[0]
                    object.__setattr__(operation, "host_only_reads", dict(operation.host_only_reads))
                elif fault == "native":
                    object.__setattr__(unit.scope, "native", tuple(unit.scope.native) + (unit.scope.native[0],))
                elif fault == "tree":
                    object.__setattr__(unit.scope, "tree", list(unit.scope.tree))
                else:
                    object.__setattr__(unit.scope, "guarded", dict(unit.scope.guarded))
                admission.admits_many(plan.storage_epoch(), plan, member, bindings)
                pytest.fail("changed storage roles must not return canonical membership")

        with pytest.raises(CompilationError):
            inspect()


@pytest.mark.parametrize("fault", ["registry", "context", "arena", "handle"])
def test_final_source_callback_rechecks_dependency_context_and_generation(final_plan, monkeypatch, fault):
    _path, owner, plan = final_plan
    member, bindings = query(plan)
    inputs = owner.builder.analysis.inputs
    original = type(inputs).verify_live
    calls = []

    def inspect():
        with StorageAdmission(plan) as admission:
            dependency = next(item for item in admission._dependencies if item.fact is not None)

            def changed(self):
                result = original(self)
                if self is inputs:
                    calls.append(self)
                    if len(calls) == 3:
                        analysis, fact = dependency.owner, dependency.fact
                        arena = analysis._source_derivation_arena
                        if fault == "registry":
                            dependency.registry.pop(dependency.key)
                        elif fault == "context":
                            object.__setattr__(analysis, "_source_authority_context", None)
                        elif fault == "arena":
                            object.__setattr__(analysis, "_source_derivation_arena", arena.fork())
                        else:
                            key = (fact.derived.family.name, id(analysis), fact.query)
                            while key not in arena._handles:
                                arena = arena._parent
                            arena._handles.pop(key)
                return result

            monkeypatch.setattr(type(inputs), "verify_live", changed)
            admission.admits_many(plan.storage_epoch(), plan, member, bindings)
            pytest.fail("changed dependency ownership must reject before admission returns")

    with pytest.raises(CompilationError, match="retained"):
        inspect()
    assert len(calls) == 3
