"""Original Fortran counterfactuals require their own offline evidence."""

import re
from dataclasses import replace

from compiler.offload.numerical_calibration import NumericalCalibrationError, numerical_compute_model


def fork_join_schedule_compatible(nodes):
    """Require the original scheduling contract priced by offline evidence.

    The native fork/join fixture uses static worksharing. SCHEDULE(runtime)
    needs the separate validated runtime-schedule contract and original-caller
    guard, so it does not satisfy this fixed-schedule predicate. This affects
    estimates only: a joined-completion proof can retain the original directive
    for forced execution.

    Inspect only the supplied original region, so an unrelated sibling cannot
    invalidate an independently selected numerical region.
    """
    return fork_join_participation(nodes) == "fork_join"


def fork_join_participation(nodes):
    """Distinguish fixed static worksharing from a checked runtime schedule.

    Only the original selected nodes are inspected. Numerical completion and
    an independent optional runtime-schedule validation are still required.
    Explicit chunks and other schedules do not use the contiguous static cost.
    """
    from fparser.two.utils import walk

    from compiler.scopes.segments import directive

    runtime = False
    for node in walk(nodes):
        text = directive(node)
        if text is None:
            continue
        # These clauses change the team/startup protocol independently of
        # static worksharing. Their original expressions remain untouched;
        # the fixed-budget fixture does not price the resulting execution.
        if re.search(r"\b(?:num_threads|proc_bind|if)\s*\(", text, re.IGNORECASE):
            return "unknown"
        for match in re.finditer(r"\bschedule\s*\(([^)]*)\)", text):
            schedule = match.group(1).strip().lower()
            if schedule == "runtime":
                runtime = True
            elif schedule != "static":
                return "unknown"
    return "fork_join_runtime" if runtime else "fork_join"


def native_participation(analysis, procedure):
    """Classify original completion, never the lowered parallel IR.

    A source guard may surround a whole joined team. Its condition remains at
    the original execution point; only a reached numerical unit incurs team
    startup. Mixed serial/team bodies retain an unavailable counterfactual.
    """
    from compiler.frontend.source_effects import _children, _kind
    from compiler.ir import CompilationError
    from compiler.scopes.segments import directive, grouped_nodes

    summary = analysis.summarize(procedure)["native_completion"]
    if summary["available"] and not summary["has_openmp_in_closure"]:
        return "serial"
    routine = analysis.routines[procedure]
    requires_runtime = False

    def joined(nodes):
        nonlocal requires_runtime
        nodes = tuple(node for node in nodes if _kind(node) != "Comment" or directive(node) is not None)
        if not nodes:
            return True
        participation = fork_join_participation(nodes)
        if participation == "unknown":
            return False
        requires_runtime |= participation == "fork_join_runtime"
        groups = grouped_nodes(nodes)
        if len(groups) != 1:
            return False
        item = groups[0]
        if not isinstance(item, tuple) and _kind(item) == "If_Construct":
            branch = []
            for node in item.content:
                if _kind(node) in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                    if branch and not joined(branch):
                        return False
                    branch = []
                else:
                    branch.append(node)
            return not branch or joined(branch)
        try:
            analysis.joined_completion(procedure, item if isinstance(item, tuple) else (item,))
        except CompilationError:
            return False
        return True

    if summary["has_openmp_in_closure"] and joined(_children(routine.execution)):
        return "fork_join_runtime" if requires_runtime else "fork_join"
    return "unknown"


def apply_source_compute_costs(analysis, profile, native_participation, *, array_views=False):
    """Attach total-compute coefficients without changing numerical legality.

    Runtime dimensions must still satisfy each model's item range. Original
    Fortran identity and CPU placement are checked at the original caller and
    query respectively; a source model cannot authorize a different backend.
    """
    units = []
    for unit in analysis.units:
        if (unit.compute_arithmetic_operations_per_iteration is None
                or unit.compute_runtime_divisions_per_iteration is None or unit.work_is_upper_bound):
            units.append(replace(unit, work_per_iteration=None, compute_model=None,
                                 work_estimate_reason=unit.compute_arithmetic_estimate_reason or unit.work_estimate_reason))
            continue
        try:
            if native_participation in {"fork_join", "fork_join_runtime"} and len(analysis.units) != 1:
                raise NumericalCalibrationError("shared original fork/join startup requires a complete multi-loop cost contract")
            if profile is None:
                raise NumericalCalibrationError("native Fortran compute calibration is missing")
            if array_views and "cpu_dependency" in profile:
                raise NumericalCalibrationError("borrowed-view address compute calibration unavailable")
            primitives = dict(unit.intrinsic_work_per_iteration)
            if unit.compute_runtime_divisions_per_iteration:
                primitives["divide"] = unit.compute_runtime_divisions_per_iteration
            if "cpu_dependency" in profile:
                from compiler.offload.dependency_source import dependency_source_model
                model = dependency_source_model(unit, profile, native_participation)
            else:
                model = numerical_compute_model(profile, tuple(sorted(primitives.items())),
                    workload_class=unit.workload_class, workload_features=unit.workload_features,
                    native_participation="fork_join" if native_participation == "fork_join_runtime" else native_participation)
            if (any(model[role].get("memory_cost_model", {}).get("kind") == "piecewise_bandwidth_v1"
                    for role in ("native_fortran", "generated_cpu", "gpu"))
                    and any(not footprint.exact or footprint.full_upload or footprint.full_write
                            for footprint in unit.footprints)):
                raise NumericalCalibrationError("piecewise memory costs require exact physical working-set sections")
            if native_participation == "fork_join_runtime":
                from compiler.offload.schedule_calibration import runtime_schedule_requirement
                model.update(runtime_schedule_requirement(profile, model, primitives))
                model["native_participation"] = native_participation
        except NumericalCalibrationError as error:
            units.append(replace(unit, work_per_iteration=None, compute_model=None,
                                 work_estimate_reason=str(error)))
        else:
            units.append(replace(unit, work_per_iteration=unit.arithmetic_work_per_iteration,
                                 compute_model=model, work_estimate_reason=None))
    return replace(analysis, units=tuple(units))
