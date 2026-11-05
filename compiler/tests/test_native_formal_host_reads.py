"""Caller-owned allocations can reserve host-only READ ranges across children."""
from copy import copy

import pytest

from compiler.ir import CompilationError
from compiler.tests.test_native_call_graph import generate, mesh_source


def direct_source(intent="inout"):
    return (mesh_source().replace("\nreal(8),allocatable::mesh(:)", "")
            .replace("subroutine advance()",
                     "subroutine advance(mesh)\nreal(8),allocatable,intent(" + intent + ")::mesh(:)"))


def source():
    text = direct_source()
    call = "call update(points,a,mesh)"
    start = text.index(call)
    end = text.index("end subroutine", start)
    consumer = text[start + len(call):end]
    text = text[:start] + "call child(mesh)\n" + text[end:]
    child = ("subroutine child(coords)\nreal(8),intent(in)::coords(-1:)\ninteger::i,j,k\n"
             "call update(points,a,coords)\n" + consumer + "end subroutine\n")
    return text.replace("end module", child + "end module")


def test_caller_allocation_read_reservation_keeps_native_and_gpu_child(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    _path, builder, _outputs, report = generate(tmp_path, source())
    owner, = report["scopes"]
    assert not owner["boundaries"], owner["boundaries"]
    plan = owner["checked_execution_plan"]
    child, = [member for member in plan["procedures"] if member["procedure"] == "native_graph::child"]
    assert not child["boundaries"], child["boundaries"]
    operation, = [op for unit in child["units"] for op in unit["operations"]["native_operations"]
                   if op["kind"] == "complete original native CALL"]
    access, = operation["native_opaque_accesses"]["accesses"]
    assert access["resource"] == "argument::coords"
    assert access["mode"] == "R"
    assert "argument::mesh" not in builder.facts["captures"]
    resource, = [row for row in plan["storage_epoch"]["resources"] if row["resource"] == "argument::mesh"]
    assert resource["storage"] == "opaque host reservation"
    assert resource["device_replica_authorized"] is resource["initialization_authorized"] is False
    validate_checked_execution_plans(report, capture_facts=builder.facts)
    assert builder.finish()


@pytest.mark.parametrize("intent", ["in", "inout"])
def test_original_allocatable_formal_uses_read_effects_without_definition_credit(tmp_path, intent):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    _path, builder, _outputs, report = generate(tmp_path, direct_source(intent))
    owner, = report["scopes"]
    assert not owner["boundaries"], owner["boundaries"]
    operation, = [op for unit in owner["planning_segments"] for op in unit["operations"]["native_operations"]
                   if op["kind"] == "complete original native CALL"]
    access, = operation["native_opaque_accesses"]["accesses"]
    assert access["resource"] == "argument::mesh"
    assert access["permit"]["version"] == 2
    assert access["permit"]["managed_definitions"] is False
    assert operation["resource_effects"]["argument::mesh"] == ["read"]
    assert "argument::mesh" not in operation["managed_resources"]
    assert "argument::mesh" not in builder.facts["captures"]
    validate_checked_execution_plans(report, capture_facts=builder.facts)
    assert builder.finish()


@pytest.mark.parametrize("attribute", ["pointer", "target", "optional", "volatile", "asynchronous"])
def test_uncertain_caller_allocation_remains_native(tmp_path, attribute):
    text = source()
    before = "allocatable,intent(inout)::mesh"
    after = ("pointer" if attribute == "pointer" else "allocatable," + attribute) + ",intent(inout)::mesh"
    _path, _builder, _outputs, report = generate(tmp_path, text.replace(before, after))
    assert not any(op.get("native_opaque_accesses") for scope in report["scopes"]
                   for member in scope["checked_execution_plan"]["procedures"] for unit in member["units"]
                   for op in unit["operations"]["native_operations"])


@pytest.mark.parametrize("action", ["coords(-1)=7", "deallocate(mesh)"])
def test_readonly_reservation_never_authorizes_a_write_or_descriptor_change(tmp_path, action):
    text = direct_source().replace("real(8),intent(in)::coords(-1:)", "real(8),intent(inout)::coords(-1:)")
    text = text.replace("visits=visits+1", "visits=visits+1\n" + action)
    if action == "deallocate(mesh)":
        # The original callee cannot see the owner's dummy. Pass the allocatable
        # descriptor explicitly; no missing name is used to force rejection.
        text = text.replace("real(8),intent(inout)::coords(-1:)", "real(8),allocatable,intent(inout)::coords(:)")
        text = text.replace("deallocate(mesh)", "deallocate(coords)")
    _path, _builder, _outputs, report = generate(tmp_path, text)
    assert not any(op.get("native_opaque_accesses") for scope in report["scopes"]
                   for member in scope["checked_execution_plan"]["procedures"] for unit in member["units"]
                   for op in unit["operations"]["native_operations"])


@pytest.fixture
def forwarded_read(tmp_path, monkeypatch):
    from compiler.scopes.module_owner import ModuleOwner

    original = ModuleOwner.__init__
    children = []

    def remember(self, parent, routine, *args, **kwargs):
        result = original(self, parent, routine, *args, **kwargs)
        if routine.qualified == "native_graph::child":
            children.append(self)
        return result

    monkeypatch.setattr(ModuleOwner, "__init__", remember)
    generate(tmp_path, source())
    child, = children
    operation, = [op for unit in child.units for op in unit.scope.native if op.native_opaque_accesses]
    access, = operation.native_opaque_accesses.values()
    assert access.validate(operation) is access
    return child, operation, access


@pytest.mark.parametrize("timing", ["before", "last_callback"])
@pytest.mark.parametrize("fault", ["call", "canonical", "parent", "member", "actual"])
def test_forwarded_read_keeps_original_caller_association(forwarded_read, monkeypatch, timing, fault):
    import inspect

    child, operation, access = forwarded_read
    permit = access.permit

    def mutate():
        if fault == "call":
            object.__setattr__(child, "original_source_call", copy(child.original_source_call))
        elif fault == "canonical":
            object.__setattr__(child, "canonical_bindings", {access.binding.root: copy(permit._actual)})
        elif fault == "parent":
            object.__setattr__(child, "parent", copy(child.parent))
            object.__setattr__(child.parent, "builder", copy(child.parent.builder))
            child.parent.builder.entry = copy(child.parent.routine)
        elif fault == "member":
            child.builder._native_host_owner = copy(child)
        else:
            child.builder._native_host_array_actuals[access.binding.root] = copy(permit._actual)

    if timing == "before":
        mutate()
    else:
        inputs = permit._analysis.inputs
        original = type(inputs).verify_live
        reached = []

        def mutate_last(self):
            result = original(self)
            if self is inputs and inspect.currentframe().f_back.f_code is type(permit)._validate.__code__:
                reached.append(True)
                mutate()
            return result

        monkeypatch.setattr(type(inputs), "verify_live", mutate_last)
    with pytest.raises(CompilationError):
        access.validate(operation)
    if timing == "last_callback":
        assert reached == [True]
