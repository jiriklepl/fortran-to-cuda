"""Internal memory plans have explicit caller transfers and checked lifetimes."""

from dataclasses import replace

import pytest

from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    ConditionalRegion,
    ExecutionPlan,
    HostBlock,
    Literal,
    Loop,
    ParallelRegion,
    Reference,
    RegionReport,
    ScalarType,
    SequentialRegion,
    Size,
    SourceLocation,
    Symbol,
)
from compiler.memory import MemoryOperation, MemoryPlan, format_memory, plan_memory, validate_memory

LOCATION = SourceLocation("memory.f90", 12)
INPUT = Symbol(0, "source", ScalarType.REAL, 1, "in", True)
OUTPUT = Symbol(1, "output", ScalarType.REAL, 1, "out", True)
UPDATED = Symbol(2, "updated", ScalarType.REAL, 1, "inout", True)
SCALAR = Symbol(3, "n", ScalarType.INTEGER, intent="in", parameter=True)
FOREIGN = Symbol(4, "foreign", ScalarType.REAL, 1, "in", True)
LOCAL = Symbol(5, "local", ScalarType.REAL, 1)
ITERATOR = Symbol(6, "i", ScalarType.INTEGER)
PARAMETERS = (INPUT, OUTPUT, UPDATED, SCALAR)
ARRAYS = PARAMETERS[:3]
ONE = Literal("1", ScalarType.INTEGER)
ASSIGNMENT = Assignment(ArrayAccess(UPDATED, (ONE,)), ArrayAccess(INPUT, (ONE,)), LOCATION)
HOST = HostBlock((ASSIGNMENT,), (INPUT,), (UPDATED,))
EMPTY = ExecutionPlan(())
REPORT = RegionReport("", "", "", "", "", "", "")


def base_plan():
    return plan_memory(EMPTY, PARAMETERS)


def test_lifecycle_distinguishes_caller_transfers_from_coherence_ensures():
    memory = plan_memory(ExecutionPlan((HOST,)), PARAMETERS)
    assert memory.create == (
        MemoryOperation("acquire", ARRAYS),
        MemoryOperation("upload", (INPUT, UPDATED)),
    )
    assert memory.run == (
        MemoryOperation("host", (INPUT, UPDATED)),
        MemoryOperation("execute", step=HOST),
        MemoryOperation("host_write", (UPDATED,)),
    )
    assert memory.retrieve == (MemoryOperation("sync"), MemoryOperation("download", (OUTPUT, UPDATED)))
    assert memory.destroy == (MemoryOperation("sync"), MemoryOperation("release", ARRAYS))
    validate_memory(memory, PARAMETERS)


@pytest.mark.parametrize("parameters", [(), (SCALAR,)])
def test_scalar_only_lifecycle_keeps_empty_slots_and_synchronization(parameters):
    memory = plan_memory(EMPTY, parameters)
    validate_memory(memory, parameters)
    assert memory.create == (MemoryOperation("acquire"), MemoryOperation("upload"))
    assert memory.retrieve == (MemoryOperation("sync"), MemoryOperation("download"))
    assert memory.destroy == (MemoryOperation("sync"), MemoryOperation("release"))


def test_sync_can_be_inserted_in_every_phase_without_symbols():
    memory = base_plan()
    memory = MemoryPlan(
        *(getattr(memory, phase) + (MemoryOperation("sync"),) for phase in ("create", "run", "retrieve", "destroy"))
    )
    validate_memory(memory, PARAMETERS)
    assert format_memory(memory).count("sync") == 6


@pytest.mark.parametrize("policy", [None, "dedicated", "pooled"])
@pytest.mark.parametrize("parameters", [PARAMETERS, (SCALAR,)])
def test_allocation_policy_is_explicit_and_does_not_change_transfer_or_execution_plans(policy, parameters):
    memory = plan_memory(EMPTY, parameters, acquisition_policy=policy)
    original = plan_memory(EMPTY, parameters)
    assert memory.create[0].acquisition_policy == policy
    assert memory.create[1:] == original.create[1:]
    assert (memory.run, memory.retrieve, memory.destroy) == (original.run, original.retrieve, original.destroy)
    assert f"acquire [{policy or 'dedicated'}]" in format_memory(memory)
    validate_memory(memory, parameters)


@pytest.mark.parametrize("policy", ["cache", "", 1, [], object()])
def test_unknown_allocation_policy_is_rejected_including_scalar_only_plans(policy):
    with pytest.raises(ValueError, match="Unknown memory acquisition policy"):
        plan_memory(EMPTY, (SCALAR,), acquisition_policy=policy)


@pytest.mark.parametrize("phase", ["create", "run", "retrieve", "destroy"])
def test_allocation_policy_is_only_valid_on_acquisitions(phase):
    memory = base_plan()
    operations = (*getattr(memory, phase), MemoryOperation("sync", acquisition_policy="dedicated"))
    with pytest.raises(ValueError, match="cannot contain an acquisition policy"):
        validate_memory(replace(memory, **{phase: operations}), PARAMETERS)


def test_allocation_policy_is_keyword_only_and_preserves_existing_operation_arguments():
    with pytest.raises(TypeError):
        MemoryOperation("acquire", ARRAYS, None, (), (), "pooled")


@pytest.mark.parametrize("policy", [None, "dedicated", "pooled"])
def test_session_validation_requires_dedicated_allocations(policy):
    memory = plan_memory(EMPTY, PARAMETERS, acquisition_policy=policy)
    if policy == "pooled":
        with pytest.raises(ValueError, match="Explicit sessions require dedicated allocations"):
            validate_memory(memory, PARAMETERS, allow_pooled=False)
    else:
        validate_memory(memory, PARAMETERS, allow_pooled=False)


@pytest.mark.parametrize(
    ("phase", "operations", "message"),
    [
        ("run", (MemoryOperation("placeholder"),), "Unknown memory operation"),
        ("run", (object(),), "Expected a MemoryOperation"),
        ("run", (MemoryOperation("host", [INPUT]),), "tuple of symbols"),
        ("run", (MemoryOperation("host", ("source",)),), "tuple of symbols"),
        ("run", (MemoryOperation("host", (FOREIGN,)),), "unknown or nonarray"),
        ("run", (MemoryOperation("host", (SCALAR,)),), "unknown or nonarray"),
        ("run", (MemoryOperation("host", (LOCAL,)),), "unknown or nonarray"),
        ("run", (MemoryOperation("host", (INPUT, INPUT)),), "Duplicate symbols"),
        ("run", (MemoryOperation("device"),), "requires array symbols"),
        ("run", (MemoryOperation("sync", (INPUT,)),), "cannot contain symbols"),
        ("run", (MemoryOperation("sync", step=HOST),), "cannot contain an execution step"),
        ("run", (MemoryOperation("sync", then_ops=(MemoryOperation("sync"),)),), "cannot contain branch"),
        ("run", (MemoryOperation("sync", else_ops=[]),), "branch operations must be tuples"),
        ("run", (MemoryOperation("acquire", ARRAYS),), "invalid in run"),
        ("run", (MemoryOperation("release", ARRAYS),), "invalid in run"),
        ("run", (MemoryOperation("upload", (INPUT,)),), "invalid in run"),
        ("run", (MemoryOperation("download", (OUTPUT,)),), "invalid in run"),
        ("create", (), "acquire every array"),
        ("create", (MemoryOperation("device", (INPUT,)),), "invalid in create"),
        ("create", (MemoryOperation("acquire", ARRAYS),), "upload every input"),
        ("create", (MemoryOperation("upload", (INPUT,)),), "before acquisition"),
        ("create", (MemoryOperation("acquire", (FOREIGN,)),), "unknown or nonarray"),
        ("create", (MemoryOperation("acquire", ARRAYS), MemoryOperation("acquire", (INPUT,))), "Duplicate array"),
        (
            "create",
            (MemoryOperation("acquire", ARRAYS), MemoryOperation("upload", (OUTPUT,))),
            "conflicts with array intent",
        ),
        ("retrieve", (MemoryOperation("download", (OUTPUT, UPDATED)),), "synchronization before download"),
        ("retrieve", (MemoryOperation("sync"), MemoryOperation("download", (OUTPUT,))), "download every output"),
        ("retrieve", (MemoryOperation("sync"), MemoryOperation("download", (INPUT,))), "conflicts with array intent"),
        ("destroy", (MemoryOperation("release", ARRAYS),), "synchronization before release"),
        ("destroy", (MemoryOperation("sync"), MemoryOperation("release", (OUTPUT,))), "release every array"),
        (
            "destroy",
            (MemoryOperation("sync"), MemoryOperation("release", ARRAYS), MemoryOperation("release", (INPUT,))),
            "after release",
        ),
    ],
)
def test_invalid_memory_payloads_and_lifecycles_are_internal_errors(phase, operations, message):
    with pytest.raises(ValueError, match=message):
        validate_memory(replace(base_plan(), **{phase: operations}), PARAMETERS)


@pytest.mark.parametrize(
    "operation",
    [
        MemoryOperation("execute"),
        MemoryOperation("execute", step=object()),
        MemoryOperation("execute", step=ConditionalRegion(ONE, EMPTY, EMPTY, LOCATION)),
        MemoryOperation("branch"),
        MemoryOperation("branch", step=HOST),
    ],
)
def test_execution_operations_require_matching_typed_steps(operation):
    with pytest.raises(TypeError, match="Invalid execution step"):
        validate_memory(replace(base_plan(), run=(operation,)), PARAMETERS)


def test_unknown_array_in_executable_ir_cannot_hide_behind_empty_effect_metadata():
    assignment = Assignment(ArrayAccess(UPDATED, (ONE,)), Size(FOREIGN, 1), LOCATION)
    block = Block((assignment,))
    steps = (
        HostBlock(block.statements),
        SequentialRegion(0, block, "fallback"),
        ParallelRegion(0, (), block.statements, (), (), REPORT, block),
        ConditionalRegion(ONE, ExecutionPlan((HostBlock(block.statements),)), EMPTY, LOCATION),
    )
    for step in steps:
        operation = MemoryOperation("branch" if isinstance(step, ConditionalRegion) else "execute", step=step)
        with pytest.raises(ValueError, match="unknown array parameter"):
            validate_memory(replace(base_plan(), run=(operation,)), PARAMETERS)


def test_unknown_array_in_parallel_bounds_is_checked_without_captured_metadata():
    loop = Loop(ITERATOR, ONE, Size(FOREIGN, 1), Block(()), LOCATION)
    step = ParallelRegion(0, (loop,), (), (), (), REPORT, Block(()))
    with pytest.raises(ValueError, match="unknown array parameter"):
        plan_memory(ExecutionPlan((step,)), PARAMETERS)


def test_conditional_operations_preserve_ownership_and_path_specific_effects():
    loop = Loop(ITERATOR, ONE, Reference(SCALAR), Block((ASSIGNMENT,)), LOCATION)
    parallel = ParallelRegion(0, (loop,), (ASSIGNMENT,), (), (), REPORT, loop.body, (INPUT,), (UPDATED,))
    condition = Binary(">", ArrayAccess(UPDATED, (ONE,)), ONE)
    branch = ConditionalRegion(condition, ExecutionPlan((parallel,)), ExecutionPlan((HOST,)), LOCATION)
    memory = plan_memory(ExecutionPlan((branch,)), PARAMETERS)
    predicate, operation = memory.run
    assert predicate == MemoryOperation("host", (UPDATED,))
    assert operation.kind == "branch"
    assert [op.kind for op in operation.then_ops] == ["device", "execute", "device_write"]
    assert [op.kind for op in operation.else_ops] == ["host", "execute", "host_write"]
    operation = replace(operation, then_ops=(MemoryOperation("sync"), *operation.then_ops))
    validate_memory(replace(memory, run=(predicate, operation)), PARAMETERS)
    for field in ("then_ops", "else_ops"):
        malformed = replace(operation, **{field: (MemoryOperation("release", (INPUT,)),)})
        with pytest.raises(ValueError, match="invalid in run"):
            validate_memory(replace(memory, run=(predicate, malformed)), PARAMETERS)


def test_plan_and_phase_payloads_are_validated():
    with pytest.raises(ValueError, match="Expected a MemoryPlan"):
        validate_memory(None, PARAMETERS)
    with pytest.raises(ValueError, match="operations must be a tuple"):
        validate_memory(replace(base_plan(), run=[]), PARAMETERS)
    with pytest.raises(ValueError, match="function array parameters"):
        validate_memory(base_plan(), (LOCAL,))
