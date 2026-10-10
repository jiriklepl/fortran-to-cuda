"""Attach independently validated dependency costs to original source units."""

from copy import deepcopy

from compiler.ir import ArrayAccess, Assignment, Reference
from compiler.ir.nodes import walk_expr
from compiler.offload.cpu_dependency_calibration import ACCESS_CLASS, DOMAINS, cpu_dependency_costs
from compiler.offload.numerical_calibration import (
    COMPUTE_BACKEND_ID,
    COMPUTE_PRIVATE_LIMITS,
    NumericalCalibrationError,
    _compute_components,
    compute_generator_identity,
    validate_numerical_profile,
)


def _require_observable_assignments(region):
    """Do not price private work the native optimizer may discard.

    The admitted memory class has one output write and a straight-line body.
    Backward scalar liveness also catches unused input-load copies, which do
    not have arithmetic nodes in the occurrence DAG. Keep all original work;
    this check makes its cost unavailable rather than optimizing it away.
    """
    live = set()
    for statement in reversed(region.body.statements):
        if not isinstance(statement, Assignment):
            raise NumericalCalibrationError("source work observability unavailable")
        target = statement.target
        expressions = [statement.value]
        if isinstance(target, Reference):
            if target.symbol not in live:
                raise NumericalCalibrationError("unconsumed private source work has no cost contract")
            live.remove(target.symbol)
        elif isinstance(target, ArrayAccess):
            expressions.extend(target.indices)
        else:
            raise NumericalCalibrationError("source output observability unavailable")
        for expression in expressions:
            live.update(node.symbol for node in walk_expr(expression) if isinstance(node, Reference))


def dependency_source_model(unit, profile, native_participation):
    """Keep legality, source semantics and cost applicability separate.

    The first memory contract admits one-dimensional contiguous pointwise
    entries only. Borrowed views require a separate addressing-cost contract.
    Stencil, accumulation, uniform numerics, existing teams and unproved
    primitive domains remain unavailable, rather than borrowing a nearby
    calibration class.
    """
    if native_participation not in {"serial", "fork_join"}:
        raise NumericalCalibrationError("dependency costs require original serial or fixed fork/join participation")
    requirement = unit.memory_access_requirement
    if (requirement is None or not requirement.available or requirement.class_id != ACCESS_CLASS or
            len(unit.region.loops) != 1):
        raise NumericalCalibrationError("dependency source memory access class or dense traversal unavailable")
    if any(not footprint.exact or footprint.full_upload or footprint.full_write for footprint in unit.footprints):
        raise NumericalCalibrationError("dependency costs require exact physical sections")
    graph = unit.compute_dependencies
    if graph is None or not graph.available:
        raise NumericalCalibrationError("dependency source graph unavailable")
    _require_observable_assignments(unit.region)
    if any(operation.family in {"sqrt", "acos", "cos", "divide_dynamic"} for operation in graph.operations):
        raise NumericalCalibrationError("source-backed primitive interval proof unavailable")
    domains = {"divide_constant": DOMAINS["divide_constant"]}
    cpu = {}
    for role, backend in (("native_fortran", "native_" + native_participation), ("generated_cpu", "generated_cpu")):
        costs = cpu_dependency_costs(profile, graph, backend, workload_class=unit.workload_class,
            workload_features=unit.workload_features, primitive_domains=domains, access_class=ACCESS_CLASS)
        cpu[role] = {**costs, "compute_seconds_per_item": costs["seconds_per_item"],
            "arithmetic_seconds_per_operation": 0.0, "intrinsic_seconds_per_item": 0.0,
            "memory_seconds_per_byte": None, "fixed_cost_coverage": "measured_execution_protocol_once_outside_max"}

    section = profile.get("numerical", {})
    validate_numerical_profile(section, profile)
    if (section.get("schema_version") != 2 or section["backend_id"] != COMPUTE_BACKEND_ID or
            section["generator_id"] != compute_generator_identity()):
        raise NumericalCalibrationError("dependency source requires compatible independently validated GPU evidence")
    features = unit.workload_features.to_dict() if hasattr(unit.workload_features, "to_dict") else unit.workload_features
    if unit.workload_class == "fixed_private_array_v2" and any(
            type(features.get(name)) is not int or not 0 < features[name] <= limit
            for name, limit in COMPUTE_PRIVATE_LIMITS.items()):
        raise NumericalCalibrationError("GPU private workload outside calibrated applicability")
    primitives = dict(unit.intrinsic_work_per_iteration)
    if unit.compute_runtime_divisions_per_iteration:
        primitives["divide"] = unit.compute_runtime_divisions_per_iteration
    required = ("arithmetic_v2", "memory_v2", *("primitive_" + name + "_v2" for name in primitives))
    for family in required:
        if section["families"][family]["gpu"]["status"] != "accepted":
            raise NumericalCalibrationError("GPU dependency-source family rejected: " + family)
    workload = "ordinary_expression_v2" if unit.workload_class == "scalar_expression_v2" and\
        not any(name != "divide" for name in primitives) else unit.workload_class
    if section["workload_validation"][workload]["gpu"]["status"] != "accepted":
        raise NumericalCalibrationError("GPU dependency-source expression class rejected")
    components = _compute_components(section["families"], "gpu", profile["precision_bits"], tuple(primitives))
    gpu = {**components, "intrinsic_seconds_per_item": sum(count * components["intrinsic_seconds_per_operation"][name]
        for name, count in primitives.items()), "fixed_seconds": 0.0, "fixed_cost_coverage": "none", "backend_identity": "gpu"}
    return {**cpu, "gpu": gpu, "schema_version": 1, "cost_model": "source_work_span_cpu_v1",
        "item_range": cpu["native_fortran"]["item_range"], "native_participation": native_participation,
        "workload_class": unit.workload_class, "cpu_affinity": deepcopy(section["identity"]["cpu_affinity"]),
        "fortran": deepcopy(section["identity"]["fortran"]), "backend_id": "source-work-span-cpu-v1",
        "generator_id": cpu["native_fortran"]["generator_id"], "gpu_generator_id": section["generator_id"],
        "dependency_identity": graph.identity, "memory_access_class": ACCESS_CLASS, "require_unit_stride": True,
        "cpu_protocol_environment_contract": "wait/spin environment fixed from process launch; live mismatch rejects",
        "cpu_protocol_environment": {name: profile["cpu_dependency"]["execution_identity"][name]
                                     for name in ("omp_wait_policy", "gomp_spincount")}}
