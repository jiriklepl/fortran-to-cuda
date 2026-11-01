"""Runtime decisions remain explicit without inventing physical cost credits."""

from dataclasses import is_dataclass, replace

import pytest

from compiler.analysis.task_dependencies import TaskGraphError, build_task_graph, structured_task
from compiler.frontend.immutable_graph import GraphFreezer
from compiler.ir.task_graph import (
    BranchRegion,
    DeferredPlacement,
    ExecutionTarget,
    LoopRegion,
    ResourceAccess,
    Task,
    TaskEffects,
)
from compiler.scheduling.task_graph import (
    ScheduledTask,
    TaskCost,
    TaskSchedule,
    TransferEstimate,
    schedule_tasks,
    verify_schedule,
)

CPU, GPU = ExecutionTarget.CPU, ExecutionTarget.GPU
BOTH = frozenset({CPU, GPU})
POLICY = DeferredPlacement("live_source_quote_v2", "a" * 64)


def dynamic(name="choose"):
    return Task(name, TaskEffects(reads=(ResourceAccess("a"),), may_writes=(ResourceAccess("b"),), ordered=True),
                BOTH, deferred_placement=POLICY)


def static(name, *, targets=BOTH):
    return Task(name, TaskEffects(reads=(ResourceAccess("b"),)), frozenset(targets))


@pytest.mark.parametrize(("policy", "identity"), [("", "a" * 64), ("bad policy", "a" * 64),
                                                 ("valid", "A" * 64), ("valid", "a" * 63)])
def test_policy_binding_requires_exact_named_identity(policy, identity):
    with pytest.raises(ValueError, match="Deferred placement"):
        DeferredPlacement(policy, identity)


@pytest.mark.parametrize("change", [{"implementations": frozenset({CPU})},
                                    {"effects": TaskEffects()}, {"deferred_placement": object()}])
def test_dynamic_choice_requires_both_targets_and_original_order(change):
    with pytest.raises(ValueError, match="Deferred placement"):
        replace(dynamic(), **change)


def test_reached_policy_is_explicit_and_has_no_cost_or_copy_estimate():
    graph = build_task_graph((dynamic(), static("later"), static("last")))
    schedule = schedule_tasks(graph, {name: TaskCost(100, 0) for name in graph.task_ids},
                              initial_residency=("a",), group_search_budget=100)
    first, *tail = schedule.placements
    assert first.target is None
    assert first.deferred_placement == POLICY
    assert all(item.target == CPU for item in tail)
    assert all(item.estimated_cost is None and not item.transfers and not item.evictions
               for item in schedule.placements)
    assert schedule.task_ids == graph.task_ids
    assert schedule.estimated_cost is None
    assert graph.public()["tasks"][0]["implementations"] == ["cpu", "gpu"]
    assert graph.public()["tasks"][0]["deferred_placement"] == POLICY.public()
    assert schedule.public()["placements"][0]["target"] is None
    assert schedule.public()["placements"][0]["deferred_placement"] == POLICY.public()
    verify_schedule(graph, schedule)


@pytest.mark.parametrize("forgery", [
    {"target": GPU}, {"deferred_placement": None}, {"estimated_cost": 0},
    {"deferred_placement": DeferredPlacement("live_source_quote_v2", "b" * 64)},
    {"transfers": (TransferEstimate("a", GPU, 8),)}, {"evictions": ("a",)},
])
def test_static_or_mismatched_policy_cannot_replace_the_checked_choice(forgery):
    graph = build_task_graph((dynamic(),))
    item, = schedule_tasks(graph).placements
    with pytest.raises(TaskGraphError, match="Runtime placement"):
        verify_schedule(graph, TaskSchedule((replace(item, **forgery),)))


@pytest.mark.parametrize("forgery", [{"estimated_cost": 1},
                                    {"transfers": (TransferEstimate("b", CPU, 8),)},
                                    {"evictions": ("b",)}])
def test_later_work_cannot_claim_residency_or_cost_from_an_unresolved_choice(forgery):
    graph = build_task_graph((dynamic(), static("after")))
    first, second = schedule_tasks(graph).placements
    with pytest.raises(TaskGraphError, match="residency cost credit"):
        verify_schedule(graph, TaskSchedule((first, replace(second, **forgery))))


def test_missing_estimate_requires_actual_runtime_uncertainty():
    graph = build_task_graph((static("ordinary"),))
    with pytest.raises(TaskGraphError, match="Unavailable estimate"):
        verify_schedule(graph, TaskSchedule((ScheduledTask("ordinary", CPU, None),)))


def test_runtime_policy_cannot_also_be_forced_to_a_static_target():
    graph = build_task_graph((dynamic(),))
    with pytest.raises(TaskGraphError, match="static required target"):
        schedule_tasks(graph, required_targets={"choose": GPU})


def test_unknown_residency_preserves_required_gpu_work_but_not_impossible_memory():
    graph = build_task_graph((dynamic(), static("gpu", targets={GPU})))
    schedule = schedule_tasks(graph, resource_bytes={"b": 8}, memory_budget_bytes=8)
    assert schedule.placements[1].target == GPU
    assert schedule.placements[1].estimated_cost is None
    with pytest.raises(TaskGraphError, match="working set"):
        schedule_tasks(graph, resource_bytes={"b": 8}, memory_budget_bytes=7)


def test_independent_work_cannot_cross_the_reached_runtime_choice():
    graph = build_task_graph((static("before"), dynamic(), static("after")))
    schedule = schedule_tasks(graph, {"before": TaskCost(10, 1), "after": TaskCost(0, 0)})
    assert schedule.task_ids == graph.task_ids
    with pytest.raises(TaskGraphError, match="dependency"):
        verify_schedule(graph, TaskSchedule(tuple(reversed(schedule.placements))))


def test_unresolved_tail_cannot_be_reordered_even_when_independent():
    graph = build_task_graph((dynamic(), static("first"), static("second")))
    first, second, third = schedule_tasks(graph).placements
    with pytest.raises(TaskGraphError, match="remaining original source order"):
        verify_schedule(graph, TaskSchedule((first, third, second)))


@pytest.mark.parametrize("kind", ["branch", "loop"])
def test_structured_choice_propagates_unknown_estimate_without_changing_control(kind):
    body = build_task_graph((dynamic(), static("inside")))
    if kind == "branch":
        structure = BranchRegion("original_predicate", body, build_task_graph((static("other"),)))
    else:
        structure = LoopRegion("i", 1, 0, 1, body)
    graph = build_task_graph((structured_task("control", structure), static("after")))
    schedule = schedule_tasks(graph)
    assert schedule.task_ids == graph.task_ids
    assert schedule.estimated_cost is None
    assert schedule.placements[0].children[0].placements[0].target is None
    assert schedule.placements[1].target == CPU
    verify_schedule(graph, schedule)


def test_unknown_outer_state_reaches_nested_control_without_fake_costs():
    body = build_task_graph((static("body"),))
    loop = structured_task("loop", LoopRegion("i", 1, 9, 1, body))
    graph = build_task_graph((dynamic(), loop))
    schedule = schedule_tasks(graph)
    assert schedule.placements[1].children[0].placements[0].estimated_cost is None
    verify_schedule(graph, schedule)


def test_unknown_choice_breaks_gpu_prefix_search_and_never_truncates_work():
    tasks = [Task("producer", TaskEffects(may_writes=(ResourceAccess("a"),)), BOTH)]
    tasks.extend(dynamic("choice_" + str(index)) for index in range(150))
    tasks.append(static("consumer", targets={GPU}))
    graph = build_task_graph(tasks)
    schedule = schedule_tasks(graph, {"producer": TaskCost(100, 0)}, group_search_budget=1)
    assert schedule.task_ids == graph.task_ids
    assert len([group for group in schedule.groups if group.target is None]) == 150
    assert schedule.placements[0].target == GPU
    assert schedule.placements[-1].target == GPU
    verify_schedule(graph, schedule)


def test_owned_frozen_graph_and_policy_retain_semantic_verification():
    graph = build_task_graph((dynamic(), static("after")))
    frozen = GraphFreezer(own_if=is_dataclass).freeze(graph)
    schedule = schedule_tasks(frozen)
    verify_schedule(frozen, schedule)
    assert schedule.placements[0].deferred_placement.public() == POLICY.public()
