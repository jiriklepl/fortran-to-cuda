"""Explicit orchestration of validation, transforms, schedules, and emission."""

from dataclasses import replace

from compiler.addressing import plan_addressing
from compiler.driver.options import CompilerOptions
from compiler.ir import ExecutionPlan, FunctionIR
from compiler.scheduling import schedule_plan
from compiler.transforms import optimize_function


def prepare_function(
    function: FunctionIR, *, options: CompilerOptions | None = None
) -> tuple[FunctionIR, ExecutionPlan]:
    options = options or CompilerOptions()
    # Runtime partitions need the original legal unit boundaries. The default
    # compiler path retains its existing checked fusion and scalar motion.
    preparation = replace(options, opt_level=0) if options.gpu_policy != "always" else options
    function, plan = optimize_function(function, options=preparation)
    plan = schedule_plan(function, plan, options=options)
    return function, plan_addressing(plan, options=options)
