"""Explicit orchestration of validation, transforms, schedules, and emission."""

from compiler.addressing import plan_addressing
from compiler.driver.options import CompilerOptions
from compiler.ir import ExecutionPlan, FunctionIR
from compiler.scheduling import schedule_plan
from compiler.transforms import optimize_function


def prepare_function(
    function: FunctionIR, *, options: CompilerOptions | None = None
) -> tuple[FunctionIR, ExecutionPlan]:
    options = options or CompilerOptions()
    function, plan = optimize_function(function, options=options)
    plan = schedule_plan(function, plan, options=options)
    return function, plan_addressing(plan, options=options)
