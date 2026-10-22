"""Dependency legality, mixed placement, and persistent-resource scheduling."""

from dataclasses import FrozenInstanceError, is_dataclass, replace

import pytest

from compiler.analysis.task_dependencies import (
    TaskGraphError,
    build_task_graph,
    execution_plan_graph,
    structured_task,
    verify_task_graph,
)
from compiler.frontend.immutable_graph import GraphFreezer
from compiler.ir import (
    Assignment,
    Binary,
    Block,
    ExecutionPlan,
    FunctionIR,
    HostBlock,
    Literal,
    Reference,
    ScalarType,
    SourceLocation,
    Symbol,
)
from compiler.ir.task_graph import (
    BranchRegion,
    DependencyKind,
    ExecutionTarget,
    LoopRegion,
    MemoryRegion,
    ResourceAccess,
    Task,
    TaskEffects,
    TaskGraph,
)
from compiler.scheduling.task_graph import (
    ScheduledTask,
    TaskCost,
    TaskSchedule,
    schedule_tasks,
    verify_schedule,
)

CPU, GPU = ExecutionTarget.CPU, ExecutionTarget.GPU
BOTH = frozenset({CPU, GPU})


def access(resource, lower=None, upper=None):
    return ResourceAccess(resource, MemoryRegion() if lower is None else MemoryRegion(((lower, upper),)))


def work(task_id, *, reads=(), writes=(), overwrite=(), targets=BOTH, **effects):
    return Task(task_id, TaskEffects(tuple(reads), tuple(writes), tuple(overwrite), **effects), targets)


def edge_set(graph):
    return {(edge.before, edge.after, edge.kind, edge.resource) for edge in graph.dependencies}


def test_regions_normalize_alias_views_and_do_not_conflict_across_disjoint_bytes():
    region = MemoryRegion(((16, 24), (0, 8), (8, 16), (4, 4)))
    assert region.intervals == ((0, 24),)
    assert region.nbytes == 24
    assert not region.overlaps(MemoryRegion(((24, 40),)))
    assert region.contains(MemoryRegion(((8, 16),)))
    assert not MemoryRegion(()).overlaps(MemoryRegion())
    graph = build_task_graph(
        (
            work("left", writes=(access("a", 0, 8),)),
            work("right", writes=(access("a", 8, 16),)),
            work("alias", reads=(access("a", 4, 12),)),
        )
    )
    assert edge_set(graph) == {("left", "alias", DependencyKind.RAW, "a"), ("right", "alias", DependencyKind.RAW, "a")}


def test_immutable_effects_copy_callers_containers_and_require_honest_overwrite():
    writes = [access("a")]
    effects = TaskEffects(may_writes=writes, scalar_reads={"n"})
    writes.clear()
    assert effects.may_writes == (access("a"),)
    with pytest.raises(FrozenInstanceError):
        effects.opaque = True
    with pytest.raises(ValueError, match="covered"):
        TaskEffects(may_writes=(access("a", 0, 8),), must_overwrites=(access("a"),))
    with pytest.raises(ValueError, match="nonnegative"):
        MemoryRegion(((-1, 2),))


def test_array_scalar_control_lifetime_and_observable_dependencies_are_independent():
    first = work(
        "first",
        reads=(access("a"),),
        writes=(access("b"),),
        scalar_reads={"n"},
        scalar_writes={"m"},
        observable_tokens={"io"},
    )
    second = replace(
        work(
            "second",
            reads=(access("b"),),
            writes=(access("a"), access("b")),
            scalar_reads={"m"},
            scalar_writes={"n"},
            observable_tokens={"io"},
        ),
        predecessors={"first"},
    )
    retire = work("retire", lifetime_resources={"a"})
    edges = edge_set(build_task_graph((first, second, retire)))
    assert {
        ("first", "second", DependencyKind.RAW, "b"),
        ("first", "second", DependencyKind.WAR, "a"),
        ("first", "second", DependencyKind.WAW, "b"),
        ("first", "second", DependencyKind.RAW, "scalar:m"),
        ("first", "second", DependencyKind.WAR, "scalar:n"),
        ("first", "second", DependencyKind.OBSERVABLE, "io"),
        ("first", "second", DependencyKind.CONTROL, ""),
        ("first", "retire", DependencyKind.LIFETIME, "a"),
        ("second", "retire", DependencyKind.LIFETIME, "a"),
    } <= edges


def mixed_chain(host_cost):
    producer = work("producer", writes=(access("a"),), overwrite=(access("a"),))
    host = work("host", reads=(access("x"),), writes=(access("x"),), targets={CPU})
    consumer = work("consumer", reads=(access("a"),), writes=(access("b"),), overwrite=(access("b"),))
    graph = build_task_graph((producer, host, consumer))
    schedule = schedule_tasks(
        graph,
        {"producer": TaskCost(100, 0.1), "host": TaskCost(host_cost, 100), "consumer": TaskCost(100, 0.1)},
        required_targets={"producer": GPU, "consumer": GPU},
    )
    return graph, schedule


@pytest.mark.parametrize(
    ("host_cost", "expected"), [(5, ("producer", "consumer", "host")), (0.01, ("host", "producer", "consumer"))]
)
def test_independent_native_work_can_sink_or_hoist_without_moving_resident_array(host_cost, expected):
    graph, schedule = mixed_chain(host_cost)
    assert schedule.task_ids == expected
    assert all(not placement.transfers for placement in schedule.placements)
    assert next(group.task_ids for group in schedule.groups if group.target == GPU) == ("producer", "consumer")
    verify_schedule(graph, schedule)


def test_a_real_host_consumer_retains_its_required_round_trip():
    graph = build_task_graph(
        (
            work("producer", writes=(access("a"),), overwrite=(access("a"),)),
            work("correct", reads=(access("a", 0, 8),), writes=(access("a", 0, 8),), targets={CPU}),
            work("consumer", reads=(access("a"),)),
        )
    )
    schedule = schedule_tasks(graph, required_targets={"producer": GPU, "consumer": GPU})
    assert schedule.task_ids == graph.task_ids
    assert [
        (placement.task_id, [(copy.resource, copy.target) for copy in placement.transfers])
        for placement in schedule.placements
    ] == [("producer", []), ("correct", [("a", CPU)]), ("consumer", [("a", GPU)])]


def test_repeated_consumers_share_upload_and_native_writes_only_invalidate_touched_storage():
    graph = build_task_graph(
        (
            work("read1", reads=(access("a"),)),
            work("host", writes=(access("x"),), targets={CPU}),
            work("read2", reads=(access("a"),)),
        )
    )
    schedule = schedule_tasks(graph, required_targets={"read1": GPU, "read2": GPU}, preserve_source_order=True)
    assert schedule.task_ids == graph.task_ids
    assert len(schedule.placements[0].transfers) == 1
    assert schedule.placements[-1].transfers == ()


def test_partial_write_preserves_incoming_value_but_proven_complete_overwrite_does_not_upload():
    partial = work("partial", writes=(access("a", 0, 8),), overwrite=(access("a", 0, 8),))
    full = work("full", writes=(access("b"),), overwrite=(access("b"),))
    schedule = schedule_tasks(
        build_task_graph((partial, full)), required_targets={"partial": GPU, "full": GPU}, preserve_source_order=True
    )
    assert [copy.resource for copy in schedule.placements[0].transfers] == ["a"]
    assert schedule.placements[1].transfers == ()


def test_device_memory_budget_evicts_and_publishes_a_dirty_last_copy():
    graph = build_task_graph(
        (
            work("a", writes=(access("a"),), overwrite=(access("a"),)),
            work("b", writes=(access("b"),), overwrite=(access("b"),)),
        )
    )
    schedule = schedule_tasks(
        graph,
        required_targets={"a": GPU, "b": GPU},
        resource_bytes={"a": 64, "b": 64},
        memory_budget_bytes=64,
        preserve_source_order=True,
    )
    assert schedule.placements[1].evictions == ("a",)
    assert [(copy.resource, copy.target, copy.reason) for copy in schedule.placements[1].transfers] == [
        ("a", CPU, "eviction")
    ]
    with pytest.raises(TaskGraphError, match="memory budget"):
        schedule_tasks(graph, required_targets={"a": GPU}, memory_budget_bytes=32, resource_bytes={"a": 64, "b": 64})


def test_unknown_native_call_is_an_ordering_and_publication_barrier():
    graph = build_task_graph(
        (
            work("producer", writes=(access("a"),), overwrite=(access("a"),)),
            work("unknown", targets={CPU}, opaque=True),
            work("consumer", reads=(access("a"),)),
        )
    )
    schedule = schedule_tasks(graph, required_targets={"producer": GPU, "consumer": GPU})
    assert schedule.task_ids == graph.task_ids
    assert [(copy.resource, copy.target) for copy in schedule.placements[1].transfers] == [("a", CPU)]
    assert [(copy.resource, copy.target) for copy in schedule.placements[2].transfers] == [("a", GPU)]


def test_known_ordered_native_work_preserves_unrelated_device_versions():
    graph = build_task_graph(
        (
            work("producer", writes=(access("a"),), overwrite=(access("a"),)),
            work("ordered", targets={CPU}, ordered=True),
            work("consumer", reads=(access("a"),)),
        )
    )
    schedule = schedule_tasks(graph, required_targets={"producer": GPU, "consumer": GPU})
    assert schedule.task_ids == graph.task_ids
    assert all(not placement.transfers for placement in schedule.placements)


def test_verifiers_reject_missing_edges_illegal_order_target_and_incomplete_coverage():
    graph = build_task_graph((work("write", writes=(access("a"),)), work("read", reads=(access("a"),), targets={CPU})))
    with pytest.raises(TaskGraphError, match="Missing required RAW"):
        verify_task_graph(replace(graph, dependencies=()))
    with pytest.raises(TaskGraphError, match="violates RAW"):
        verify_schedule(graph, TaskSchedule((ScheduledTask("read", CPU), ScheduledTask("write", CPU))))
    with pytest.raises(TaskGraphError, match="exactly once"):
        verify_schedule(graph, TaskSchedule((ScheduledTask("write", CPU),)))
    with pytest.raises(TaskGraphError, match="implementation"):
        verify_schedule(graph, TaskSchedule((ScheduledTask("write", GPU), ScheduledTask("read", GPU))))
    with pytest.raises(TaskGraphError, match="predecessor"):
        build_task_graph((replace(graph.tasks[0], predecessors={"absent"}),))


def test_structured_branches_keep_guards_and_do_not_promote_one_arm_to_overwrite():
    then_graph = build_task_graph((work("then", writes=(access("a"),), overwrite=(access("a"),)),))
    else_graph = build_task_graph(())
    branch = structured_task(
        "branch",
        BranchRegion("positive", then_graph, else_graph),
        header_effects=TaskEffects(scalar_reads={"condition"}),
    )
    graph = build_task_graph((branch, work("read", reads=(access("a"),))))
    assert branch.effects.must_overwrites == ()
    assert branch.structure.predicate == "positive"
    schedule = schedule_tasks(graph, required_targets={"then": GPU, "read": GPU})
    assert len(schedule.placements[0].children) == 2
    assert schedule.placements[0].children[0].task_ids == ("then",)
    assert schedule.placements[0].children[1].task_ids == ()
    assert schedule.placements[1].transfers  # A is not guaranteed device-current on the untaken arm.
    with pytest.raises(TaskGraphError, match="original control region"):
        verify_schedule(graph, TaskSchedule((ScheduledTask("then", GPU), ScheduledTask("read", GPU))))
    with pytest.raises(TaskGraphError, match="omits child effects"):
        build_task_graph((replace(branch, effects=TaskEffects()),))


def test_counted_loop_remains_structural_including_zero_trip_and_iterator_effects():
    body = build_task_graph((work("body", writes=(access("a"),), overwrite=(access("a"),)),))
    loop = structured_task("loop", LoopRegion("i", 5, 1, 1, body), header_effects=TaskEffects(scalar_writes={"i"}))
    graph = build_task_graph((loop, work("after", reads=(access("a"),), scalar_reads={"i"})))
    schedule = schedule_tasks(graph, required_targets={"body": GPU, "after": GPU})
    assert loop.structure.lower == 5
    assert loop.structure.upper == 1
    assert loop.effects.must_overwrites == ()
    assert schedule.placements[0].children[0].task_ids == ("body",)
    assert schedule.placements[1].transfers  # Zero trips retain the original host version.
    with pytest.raises(TaskGraphError, match="overwrite proof"):
        build_task_graph((replace(loop, effects=replace(loop.effects, must_overwrites=(access("a"),))),))


def test_unbounded_sparse_metadata_gap_is_preserved_and_has_no_spurious_dependencies():
    tasks = [work("producer", writes=(access("a"),))]
    tasks.extend(work(f"metadata:{index}", targets={CPU}, scalar_writes={f"s:{index}"}) for index in range(1000))
    tasks.append(work("consumer", reads=(access("a"),)))
    graph = build_task_graph(tasks)
    assert len(graph.tasks) == 1002
    assert edge_set(graph) == {("producer", "consumer", DependencyKind.RAW, "a")}
    schedule = schedule_tasks(graph, required_targets={"producer": GPU, "consumer": GPU}, preserve_source_order=True)
    assert len(schedule.placements) == 1002
    assert schedule.placements[-1].transfers == ()


def test_existing_plan_adapter_moves_only_total_metadata_and_does_not_trust_intent():
    location = SourceLocation("test.f90", 1)
    n = Symbol(1, "n", ScalarType.INTEGER)
    r = Symbol(2, "r", ScalarType.REAL)
    array = Symbol(3, "a", ScalarType.REAL, rank=1, parameter=True, intent="out")
    statements = (
        Assignment(Reference(n), Literal("4", ScalarType.INTEGER), location),
        Assignment(Reference(r), Literal("1.0", ScalarType.REAL), location),
        Assignment(Reference(n), Binary("/", Reference(n), Literal("2", ScalarType.INTEGER)), location),
    )
    function = FunctionIR("example", "", (array,), (n, r, array), Block(statements), "")
    plan = ExecutionPlan(tuple(HostBlock((assignment,)) for assignment in statements))
    graph = execution_plan_graph(function, plan)
    assert not graph.tasks[0].effects.ordered
    assert graph.tasks[1].effects.ordered
    assert graph.tasks[2].effects.ordered
    assert all(not task.effects.opaque for task in graph.tasks)
    assert all(not task.effects.must_overwrites for task in graph.tasks)
    assert graph.tasks[0].payload is plan.steps[0]


def test_handmade_graph_cannot_hide_missing_dependencies_behind_an_original_order_schedule():
    tasks = work("writer", writes=(access("a"),)), work("reader", reads=(access("a"),))
    with pytest.raises(TaskGraphError, match="Missing required RAW"):
        schedule_tasks(TaskGraph(tasks))


def test_execution_plan_sealed_graph_and_schedule_remain_independently_verifiable():
    graph = build_task_graph((work("write", writes=(access("a"),)), work("read", reads=(access("a"),))))
    schedule = schedule_tasks(graph)
    freezer = GraphFreezer(own_if=is_dataclass)
    freezer.freeze((graph, schedule))
    verify_task_graph(graph)
    verify_schedule(graph, schedule)
