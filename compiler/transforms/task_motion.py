"""Rebuild numerical source IR from a checked dependency schedule.

The graph chooses order above numerical regions. Numerical independence and
fusion remain the responsibility of the existing fresh dependence proofs.
Structured children retain their original execution conditions.
"""

from __future__ import annotations

from dataclasses import replace

from compiler.analysis import build_execution_plan
from compiler.analysis.task_dependencies import execution_plan_graph
from compiler.ir import Block, ConditionalRegion, HostBlock, If, ParallelRegion, SequentialRegion
from compiler.ir.task_graph import BranchRegion, ExecutionTarget
from compiler.scheduling.task_graph import schedule_tasks, verify_schedule


def _scheduled_body(graph, schedule):
    by_id = {task.id: task for task in graph.tasks}
    statements = []
    for placement in schedule.placements:
        task = by_id[placement.task_id]
        step = task.payload
        if isinstance(step, HostBlock):
            statements.extend(step.assignments)
        elif isinstance(step, SequentialRegion):
            statements.extend(step.body.statements)
        elif isinstance(step, ParallelRegion):
            statements.append(step.loops[0])
        elif isinstance(step, ConditionalRegion):
            if not isinstance(task.structure, BranchRegion) or len(placement.children) != 2:
                raise ValueError("A conditional schedule requires both original child graphs")
            statements.append(
                If(
                    step.condition,
                    _scheduled_body(task.structure.then_graph, placement.children[0]),
                    _scheduled_body(task.structure.else_graph, placement.children[1]),
                    step.location,
                )
            )
        else:
            raise TypeError(f"Unknown scheduled numerical payload: {type(step).__name__}")
    return Block(tuple(statements))


def order_function_tasks(function, *, options):
    """Group ready device operations, preserving checked host/control effects."""
    original = build_execution_plan(function, options=options)
    graph = execution_plan_graph(function, original)

    def targets(current):
        required = {}
        for task in current.tasks:
            if isinstance(task.payload, ParallelRegion):
                required[task.id] = ExecutionTarget.GPU
            if isinstance(task.structure, BranchRegion):
                required.update(targets(task.structure.then_graph))
                required.update(targets(task.structure.else_graph))
        return required

    schedule = schedule_tasks(graph, required_targets=targets(graph), target_switch_cost=1.0)
    verify_schedule(graph, schedule)
    candidate = replace(function, body=_scheduled_body(graph, schedule))
    # Recheck availability, loop-carried dependencies and scalar lifetimes after
    # motion. Graph checks cannot replace numerical legality checks.
    checked = build_execution_plan(candidate, options=options)
    changed = schedule.task_ids != graph.task_ids
    report = (
        "task graph: dependency-verified order; "
        f"{len(graph.tasks)} tasks, {len(graph.dependencies)} dependencies, "
        f"{len(schedule.groups)} execution groups; "
        + ("independent work regrouped" if changed else "original order retained")
    )
    return candidate, replace(checked, reports=(*checked.reports, report))
