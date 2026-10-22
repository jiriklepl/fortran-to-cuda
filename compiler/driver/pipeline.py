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
    graph_report = ()
    if options.scope_execution == "graph" and options.opt_level:
        from compiler.transforms.task_motion import order_function_tasks

        function, ordered = order_function_tasks(function, options=options)
        graph_report = ordered.reports
    preparation = (replace(options, opt_level=0)
                   if options.gpu_policy != "always" and options.scope_execution != "graph" else options)
    function, plan = optimize_function(function, options=preparation)
    if graph_report:
        plan = replace(plan, reports=(*graph_report, *plan.reports))
    plan = schedule_plan(function, plan, options=options)
    return function, plan_addressing(plan, options=options)
