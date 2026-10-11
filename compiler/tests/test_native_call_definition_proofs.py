"""Original call coordinates retain bounds, OUT kills and conditional coverage."""

import copy
from dataclasses import replace

import pytest

from compiler.frontend.call_bindings import SourceDependency
from compiler.frontend.native_definitions import (
    NativeCallCoordinatesUnsupported,
    invalidate_rectangles,
    prove_native_call_coordinates,
    prove_native_definitions,
    rectangle_dependencies,
)
from compiler.frontend.native_sections import NativeAxis, NativeBound, NativeIteration, NativeRectangle
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

CALLER = "coordinate_owner::step"
CHILD = "coordinate_owner::paint"


def fixture(tmp_path, *, kind=8, caller_lower="-7", formal_lower="-2", attributes="",
            body="dst(-1:2)=4", call="call paint(field,spare,n)", extra="", formal_extra=""):
    path = tmp_path / "coordinates.f90"
    path.write_text(f"""module coordinate_owner
implicit none
integer::saved_limit=5
real({kind})::hidden(-2:8)
{extra}
contains
subroutine paint(dst,unused,count)
real({kind}),intent(out)::dst({formal_lower}:),unused(:)
integer,intent(in)::count
{formal_extra}
{body}
end subroutine
subroutine step(field,spare,n)
real({kind}){attributes},intent(inout)::field({caller_lower}:),spare(:)
integer,intent(in)::n
{call}
end subroutine
end module
""")
    analysis = SourceEffects([path])
    caller = analysis.routines[CALLER]
    original = caller.execution.content[0]
    return path, analysis, original


def whole(rank=1):
    return (NativeRectangle(tuple(NativeAxis() for _ in range(rank)), "independently admitted whole"),)


def interval(lower, upper):
    return (NativeRectangle((NativeAxis(NativeBound("literal", lower), NativeBound("literal", upper)),),
                            "independently admitted rectangle"),)


def component_fixture(tmp_path, *, dtype="real(8)", intent="in", selector="options%delta"):
    path = tmp_path / "component.f90"
    path.write_text(f"""module coordinate_owner
implicit none
type configuration
{dtype}::delta
real(8),allocatable::unrelated(:)
end type
type(configuration)::options
contains
subroutine paint(dst,scale)
real(8),intent(inout)::dst(-2:)
{dtype},intent({intent})::scale
dst(-1)=4.0_8
end subroutine
subroutine step(field)
real(8),allocatable,intent(inout)::field(:)
call paint(field,{selector})
end subroutine
end module
""")
    analysis = SourceEffects([path])
    return analysis, analysis.routines[CALLER].execution.content[0]


@pytest.mark.parametrize("dtype", ["real(4)", "real(8)", "logical"])
def test_readonly_fixed_scalar_component_keeps_original_call_evaluation(tmp_path, dtype):
    analysis, original = component_fixture(tmp_path, dtype=dtype)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    component, = proof.public()["original_scalar_components"]
    assert component["source_selector"] == "options % delta"
    assert proof.scalar_mapping["argument::scale"] == "coordinate_owner::options%delta"
    assert proof.validate(analysis, CALLER, original) is proof
    assert not proof.public()["bounds_evaluated"]


@pytest.mark.parametrize(("dtype", "intent"), [("integer", "in"), ("real(8)", "inout")])
def test_integer_coordinate_or_mutable_component_stays_native(tmp_path, dtype, intent):
    analysis, original = component_fixture(tmp_path, dtype=dtype, intent=intent)
    with pytest.raises(NativeCallCoordinatesUnsupported, match="readonly noninteger scalar"):
        prove_native_call_coordinates(analysis, CALLER, original)


def test_component_schema_metadata_mutation_invalidates_call_token(tmp_path):
    analysis, original = component_fixture(tmp_path)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    schema = analysis.modules["coordinate_owner"].component_types["configuration"]
    schema.members["delta"] = schema.members["unrelated"]
    with pytest.raises(CompilationError, match="schema changed|type|rank|lifetime is uncertain"):
        proof.validate(analysis, CALLER, original)


def test_fixed_component_path_preserves_existing_depth_limit(tmp_path):
    path = tmp_path / "deep_component.f90"
    types = ["type t8\nreal(8)::delta\nend type"]
    types += [f"type t{index}\ntype(t{index + 1})::field\nend type" for index in reversed(range(8))]
    selector = "options" + "%field" * 8 + "%delta"
    path.write_text("module coordinate_owner\n" + "\n".join(types) + f"""
type(t0)::options
contains
subroutine paint(dst,scale)
real(8),intent(inout)::dst(-2:)
real(8),intent(in)::scale
dst(-1)=scale
end subroutine
subroutine step(field)
real(8),allocatable,intent(inout)::field(:)
call paint(field,{selector})
end subroutine
end module
""")
    analysis = SourceEffects([path])
    original = analysis.routines[CALLER].execution.content[0]
    with pytest.raises(CompilationError, match="budget|depth"):
        prove_native_call_coordinates(analysis, CALLER, original)


def test_component_schema_authentication_charges_sibling_members(tmp_path):
    analysis, original = component_fixture(tmp_path)
    path = tmp_path / "component.f90"
    fields = "\n".join(f"real(8)::sibling{index}" for index in range(256))
    path.write_text(path.read_text().replace("real(8),allocatable::unrelated(:)", fields))
    analysis = SourceEffects([path])
    original = analysis.routines[CALLER].execution.content[0]
    with pytest.raises(CompilationError, match="aggregate budget"):
        prove_native_call_coordinates(analysis, CALLER, original)


@pytest.mark.parametrize("kind", [4, 8])
def test_negative_original_bounds_map_partial_child_return_without_whole_promotion(tmp_path, kind):
    path, analysis, original = fixture(tmp_path, kind=kind)
    original_bytes = path.read_bytes()
    coordinates = prove_native_call_coordinates(analysis, CALLER, original)
    child = analysis.routines[CHILD]
    definition = prove_native_definitions(analysis, CHILD, tuple(child.execution.content), include_entry=True)
    child_result = definition.transfer()
    mapped = coordinates.child_to_caller(child_result.definitions)
    box, = mapped["argument::field"]
    assert (box.axes[0].lower.value, box.axes[0].upper.value) == (-6, -3)
    assert box.axes[0].lower.kind == box.axes[0].upper.kind == "literal"
    assert coordinates.caller_to_child(mapped)["argument::dst"][0].axes == child_result.definitions["argument::dst"][0].axes
    assert coordinates.entry_definition_changes == ("argument::field", "argument::spare")
    assert path.read_bytes() == original_bytes
    public = coordinates.public()
    assert not public["bounds_evaluated"]
    assert not public["entry_definition_authority"]
    assert not public["execution_authorized"]
    assert public["mapping_does_not_apply_entry_kills"]


def test_whole_markers_preserve_association_and_unused_out_is_never_lost(tmp_path):
    _path, analysis, original = fixture(tmp_path)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    entry = {"argument::field": whole(), "argument::spare": whole()}
    child_entry = proof.caller_to_child(entry)
    assert child_entry == {"argument::dst": whole(), "argument::unused": whole()}
    # The caller/planner applies original callee entry events before any leaf.
    # Even the unused formal must kill prior independently admitted coverage.
    state = dict(entry)
    for root in proof.entry_definition_changes:
        state[root] = ()
    assert proof.caller_to_child(state) == {"argument::dst": (), "argument::unused": ()}
    state.update(proof.child_to_caller({"argument::dst": interval(-1, 2), "argument::unused": ()}))
    assert state["argument::spare"] == ()
    assert state["argument::field"] != whole()


def test_nested_whole_associations_translate_each_origin_and_retain_unused_out(tmp_path):
    path, _analysis, _original = fixture(tmp_path)
    text = path.read_text().replace("call paint(field,spare,n)", "call relay(field,spare,n)")
    text = text.replace("subroutine step(field,spare,n)", """subroutine relay(tile,discard,n)
real(8),intent(inout)::tile(-3:),discard(:)
integer,intent(in)::n
call paint(tile,discard,n)
end subroutine
subroutine step(field,spare,n)""")
    path.write_text(text)
    analysis = SourceEffects([path])
    first = analysis.routines[CALLER].execution.content[0]
    second = analysis.routines["coordinate_owner::relay"].execution.content[0]
    inner = prove_native_call_coordinates(analysis, "coordinate_owner::relay", second)
    outer = prove_native_call_coordinates(analysis, CALLER, first)
    intermediate = inner.child_to_caller({"argument::dst": interval(-1, 2), "argument::unused": ()})
    assert inner.entry_definition_changes == ("argument::tile", "argument::discard")
    mapped = outer.child_to_caller(intermediate)
    box, = mapped["argument::field"]
    assert (box.axes[0].lower.value, box.axes[0].upper.value) == (-6, -3)
    assert mapped["argument::spare"] == ()
    assert outer.entry_definition_changes == ()  # Relay is INOUT; nested OUT remains an ordered child event.


@pytest.mark.parametrize(("otherwise", "defined"), [("", False), ("else\ndst(-1:2)=8", True)])
def test_child_branch_meet_remains_conditional_before_coordinate_mapping(tmp_path, otherwise, defined):
    _path, analysis, original = fixture(tmp_path, body="if(count>0)then\ndst(-1:2)=4\n" + otherwise + "\nendif")
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    child = analysis.routines[CHILD]
    transfer = prove_native_definitions(analysis, CHILD, tuple(child.execution.content), include_entry=True).transfer()
    assert bool(proof.child_to_caller(transfer.definitions)["argument::field"]) is defined


def test_geometry_preserves_empty_loop_guards_and_does_not_shift_trip_count(tmp_path):
    _path, analysis, original = fixture(tmp_path)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    guard = NativeIteration(NativeBound("literal", 4), NativeBound("literal", 1), 1)
    box = replace(interval(-1, 2)[0], iterations=(guard,))
    mapped, = proof.child_to_caller({"argument::dst": (box,)})["argument::field"]
    assert mapped.iterations == (guard,)
    assert (mapped.axes[0].lower.value, mapped.axes[0].upper.value) == (-6, -3)


def test_scalar_formal_mapping_retains_canonical_invalidation_dependencies(tmp_path):
    _path, analysis, original = fixture(tmp_path)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    binding = analysis.routines[CHILD].scope.bindings["count"]
    bound = NativeBound("scalar", resource=binding.root,
                        dependencies=(SourceDependency(binding.root, "scalar_read", binding),))
    box = NativeRectangle((NativeAxis(bound, bound, True),), "original child point")
    mapped, = proof.child_to_caller({"argument::dst": (box,)})["argument::field"]
    assert rectangle_dependencies(mapped) == ("argument::n",)
    assert invalidate_rectangles((mapped,), ("argument::n",)) == ()
    assert proof.scalar_mapping == {"argument::count": "argument::n"}


def test_integer_literal_actual_is_a_value_not_a_forged_binding(tmp_path):
    _path, analysis, original = fixture(tmp_path, call="call paint(field,spare,-2)")
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    value = NativeBound("scalar", resource="argument::count")
    box = NativeRectangle((NativeAxis(value, value, True),), "original child point")
    mapped, = proof.child_to_caller({"argument::dst": (box,)})["argument::field"]
    assert mapped.axes[0].lower.value == -7
    assert proof.scalar_mapping == {}
    assert analysis.routines[CALLER].scope.bindings.keys() == {"field", "spare", "n"}


def test_runtime_original_allocation_lower_is_deferred_not_assumed_one(tmp_path):
    _path, analysis, original = fixture(tmp_path, caller_lower="", attributes=",allocatable")
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    box, = proof.child_to_caller({"argument::dst": interval(-1, 2)})["argument::field"]
    assert box.axes[0].lower.kind == "binary"
    origin, offset = box.axes[0].lower.children
    assert (origin.kind, origin.resource, origin.dimension) == ("lbound", "argument::field", 1)
    assert offset.value == 1
    assert origin.dependencies[0].kind == "descriptor_read"
    assert not proof.public()["allocation_freshness_authority"]


def test_same_spelling_dummies_never_reinterpret_caller_descriptor_as_child(tmp_path):
    path, _analysis, _original = fixture(tmp_path, caller_lower="", attributes=",allocatable")
    path.write_text(path.read_text().replace("dst", "field"))
    analysis = SourceEffects([path])
    original = analysis.routines[CALLER].execution.content[0]
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    caller_binding = analysis.routines[CALLER].scope.bindings["field"]
    child_binding = analysis.routines[CHILD].scope.bindings["field"]
    assert caller_binding is not child_binding
    box, = proof.child_to_caller({"argument::field": interval(-1, 2)})["argument::field"]
    origin = box.axes[0].lower.children[0]
    assert origin.resource == "argument::field"
    assert origin.dependencies[0].binding is caller_binding
    assert proof.caller_to_child({"argument::field": whole()}) == {"argument::field": whole()}
    for boxes in (interval(-1, 2), (box,)):
        with pytest.raises(NativeCallCoordinatesUnsupported, match="caller descriptor origin snapshot"):
            proof.caller_to_child({"argument::field": boxes})


@pytest.mark.parametrize("inquiry", ["lbound", "ubound"])
def test_nontrivial_empty_sensitive_inquiries_are_not_replaced_by_declared_lower(tmp_path, inquiry):
    _path, analysis, original = fixture(tmp_path)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    bound = NativeBound(inquiry, resource="argument::dst", dimension=1)
    box = NativeRectangle((NativeAxis(bound, bound, True),), "original inquiry-dependent point")
    with pytest.raises(NativeCallCoordinatesUnsupported, match="empty-sensitive"):
        proof.child_to_caller({"argument::dst": (box,)})
    loop = replace(interval(-1, 2)[0], iterations=(NativeIteration(bound, NativeBound("literal", 0)),))
    with pytest.raises(NativeCallCoordinatesUnsupported, match="empty-sensitive"):
        proof.child_to_caller({"argument::dst": (loop,)})


def test_equal_lower_inquiries_and_size_preserve_zero_extent_semantics(tmp_path):
    _path, analysis, original = fixture(tmp_path, formal_lower="-7")
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    for name in ("lbound", "ubound", "size"):
        value = NativeBound(name, resource="argument::dst", dimension=1)
        box = NativeRectangle((NativeAxis(value, value, True),), "original descriptor inquiry")
        mapped, = proof.child_to_caller({"argument::dst": (box,)})["argument::field"]
        # Zero offset should preserve the original inquiry, including LB=1/UB=0 for an empty dimension.
        assert mapped.axes[0].lower.kind == name
        assert mapped.axes[0].lower.resource == "argument::field"


def test_hidden_original_module_storage_keeps_its_coordinate_system(tmp_path):
    _path, analysis, original = fixture(tmp_path)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    boxes = interval(-2, 1)
    assert proof.caller_to_child({"coordinate_owner::hidden": boxes}) == {"coordinate_owner::hidden": boxes}
    assert proof.child_to_caller({"coordinate_owner::hidden": boxes}) == {"coordinate_owner::hidden": boxes}
    with pytest.raises(NativeCallCoordinatesUnsupported, match="unmapped original resource"):
        proof.child_to_caller({"argument::not_associated": boxes})


def test_genuine_parameter_array_stays_original_and_unmanaged(tmp_path):
    path, _analysis, _original = fixture(tmp_path)
    text = path.read_text().replace("real(8)::hidden(-2:8)", "real(8),parameter::weights(3)=[1,2,3]")
    text = text.replace("paint(dst,unused,count)", "paint(dst,unused,count,coefficients)")
    text = text.replace("integer,intent(in)::count", "integer,intent(in)::count\nreal(8),intent(in)::coefficients(3)")
    text = text.replace("call paint(field,spare,n)", "call paint(field,spare,n,weights)")
    path.write_text(text)
    analysis = SourceEffects([path])
    original = analysis.routines[CALLER].execution.content[0]
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    assert proof.formal_roots == ("argument::dst", "argument::unused")
    assert proof.resource_mapping == {"argument::dst": "argument::field", "argument::unused": "argument::spare"}
    association, = proof.public()["unmanaged_immutable_associations"]
    assert association["resource"] == "coordinate_owner::weights"
    assert analysis.modules["coordinate_owner"].bindings["weights"].attributes == frozenset({"parameter"})


def test_mapping_never_scans_or_authorizes_unknown_child_body(tmp_path, monkeypatch):
    _path, analysis, original = fixture(tmp_path, body="call unknown_child(dst)")
    analysis.structure(CALLER)
    original_structure = analysis.structure

    def local_only(procedure):
        assert procedure == CALLER
        return original_structure(procedure)

    def no_summary(*_args, **_kwargs):
        raise AssertionError("eager child effects/descriptor analysis")

    monkeypatch.setattr(analysis, "structure", local_only)
    monkeypatch.setattr(analysis, "summarize", no_summary)
    monkeypatch.setattr(analysis, "descriptor_stability", no_summary)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    assert proof.child_to_caller({"argument::dst": ()}) == {"argument::field": ()}
    assert proof.public()["requires_registered_child_effect_and_completion_proofs"]
    assert not proof.public()["execution_authorized"]


@pytest.mark.parametrize("change", ["binding", "declaring_scope", "imports", "source"])
def test_source_resolution_mutation_rejects_registered_mapping(tmp_path, change):
    path, analysis, original = fixture(tmp_path)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    caller = analysis.routines[CALLER]
    if change == "binding":
        caller.scope.bindings["field"].kind = 4
    elif change == "declaring_scope":
        caller.scope.bindings["field"].declaring_scope = analysis.routines[CHILD].scope
    elif change == "imports":
        caller.scope.imports["paint"] = ("foreign", "paint")
    else:
        path.write_text(path.read_text() + "\n! mutation\n")
    with pytest.raises(CompilationError):
        proof.validate(analysis, CALLER, original)


def test_foreign_copied_and_unregistered_tokens_never_acquire_authority(tmp_path):
    _path, analysis, original = fixture(tmp_path)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    for changed in (copy.copy(proof), replace(proof, identity="0" * 64)):
        with pytest.raises(CompilationError, match="registered exact original"):
            changed.validate(analysis, CALLER, original)
    with pytest.raises(CompilationError, match="registered exact original"):
        proof.validate(copy.copy(analysis), CALLER, original)
    with pytest.raises(CompilationError, match="registered exact original"):
        proof.validate(analysis, CALLER, copy.copy(original))


@pytest.mark.parametrize("call", ["call paint(field(:),spare,n)", "call paint(field,field,n)",
                                  "call paint(field,spare,n+1)"])
def test_unsupported_association_closes_before_original_call(tmp_path, call):
    _path, analysis, original = fixture(tmp_path, call=call)
    with pytest.raises(NativeCallCoordinatesUnsupported):
        prove_native_call_coordinates(analysis, CALLER, original)
    assert not getattr(analysis, "_native_definition_proofs", {})


def test_integer_coordinate_overflow_declines_without_wrapping(tmp_path):
    _path, analysis, original = fixture(tmp_path, caller_lower="2147483646")
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    with pytest.raises(NativeCallCoordinatesUnsupported, match="exceeds INTEGER"):
        proof.child_to_caller({"argument::dst": interval(3, 3)})


def test_aggregate_work_failure_is_hard_and_never_a_support_close(tmp_path):
    _path, analysis, original = fixture(tmp_path)
    count = 0

    def consume(amount):
        nonlocal count
        count += amount
        if count > 1:
            raise CompilationError("parent mapped work budget exhausted")

    with pytest.raises(CompilationError, match="parent mapped work") as caught:
        prove_native_call_coordinates(analysis, CALLER, original, consume=consume)
    assert not isinstance(caught.value, NativeCallCoordinatesUnsupported)
    assert not getattr(analysis, "_native_definition_proofs", {})
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    with pytest.raises(CompilationError, match="parent mapped work") as caught:
        proof.child_to_caller({"argument::dst": interval(-1, 2)}, consume=consume)
    assert not isinstance(caught.value, NativeCallCoordinatesUnsupported)


def test_bounded_mapping_counts_transformed_expression_nodes(tmp_path):
    _path, analysis, original = fixture(tmp_path)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    boxes = tuple(interval(index, index)[0] for index in range(32))
    with pytest.raises(CompilationError, match="aggregate budget") as caught:
        proof.child_to_caller({"argument::dst": boxes, "argument::unused": boxes})
    assert not isinstance(caught.value, NativeCallCoordinatesUnsupported)


def test_cache_hit_validation_and_each_mapping_charge_shared_work(tmp_path):
    _path, analysis, original = fixture(tmp_path)
    charges = []
    proof = prove_native_call_coordinates(analysis, CALLER, original, consume=charges.append)
    initial = sum(charges)
    assert initial > 0
    charges.clear()
    assert prove_native_call_coordinates(analysis, CALLER, original, consume=charges.append) is proof
    # A cache hit authenticates current original source again, rather than
    # bypassing the parent's aggregate traversal budget.
    assert sum(charges) == initial
    for mapping, state in ((proof.caller_to_child, {"argument::field": interval(-6, -3)}),
                           (proof.child_to_caller, {"argument::dst": interval(-1, 2)})):
        charges.clear()
        mapping(state, consume=charges.append)
        assert sum(charges) > initial


def wide_coordinate_fixture(tmp_path, *, lower="-2", count=13, component_siblings=0):
    """Several original formals share each DIMENSION expression object."""
    names = [f"tile{index}" for index in range(count)]
    actuals = [f"field{index}" for index in range(count)]
    declarations = []
    for start in range(0, count - 1, 4):
        group = names[start:min(start + 4, count - 1)]
        declarations.append(f"real(8),dimension({lower}:,{lower}:,{lower}:),intent(inout)::" + ",".join(group))
    declarations.append(f"real(8),dimension({lower}:,{lower}:,{lower}:,1:),intent(inout)::" + names[-1])
    path = tmp_path / "wide_coordinates.f90"
    path.write_text("""module coordinate_owner
implicit none
integer,parameter::origin=-2
real(8),parameter::weights(2)=[1.0_8,2.0_8],rates(2)=[3.0_8,4.0_8]
type configuration
real(8)::delta
real(8),allocatable::unrelated(:)
end type
type(configuration)::options
contains
subroutine paint(""" + ",".join(names) + """,left,right,n,scale)
""" + "\n".join(declarations) + """
real(8),intent(in)::left(:),right(:),scale
integer,intent(in)::n
end subroutine
subroutine step(""" + ",".join(actuals) + """,n)
real(8),allocatable,intent(inout)::""" + ",".join(name + "(:,:,:)" for name in actuals[:-1]) + "\n" + f"real(8),allocatable,intent(inout)::{actuals[-1]}(:,:,:,:)\n" + """integer,intent(in)::n
call paint(""" + ",".join(actuals) + """,weights,rates,n,options%delta)
end subroutine
end module
""")
    if component_siblings:
        siblings = "\n".join(f"real(8)::setting{index}" for index in range(component_siblings))
        path.write_text(path.read_text().replace("real(8),allocatable::unrelated(:)",
                                                "real(8),allocatable::unrelated(:)\n" + siblings))
    analysis = SourceEffects([path])
    return path, analysis, analysis.routines[CALLER].execution.content[0]


def test_wide_shared_dimension_call_reuses_only_authenticated_constant_nodes(tmp_path, monkeypatch):
    from compiler.frontend import native_call_coordinates

    _path, analysis, original = wide_coordinate_fixture(tmp_path)
    original_bound = native_call_coordinates._bound
    calls = []

    def counted_bound(analysis, scope, node, binding, dimension):
        calls.append((id(scope), id(node)))
        return original_bound(analysis, scope, node, binding, dimension)

    monkeypatch.setattr(native_call_coordinates, "_bound", counted_bound)
    charges = []
    proof = prove_native_call_coordinates(analysis, CALLER, original, consume=charges.append)
    public = proof.public()
    assert analysis.operation_limit == 256
    assert len(public["coordinate_associations"]) == 13
    assert len(public["unmanaged_immutable_associations"]) == 2
    assert public["original_scalar_components"][0]["source_selector"] == "options % delta"
    assert public["source_work"] == sum(charges) <= analysis.operation_limit
    # Distinct same-text declaration nodes still resolve independently. Each
    # shared original bound is converted once within a single derivation.
    assert len(calls) == len(set(calls))
    assert 1 < len(calls) < 13 * 3
    for association in public["coordinate_associations"]:
        origins = association["formal_lower_origins"]
        assert [axis["value"] for axis in origins[:3]] == [-2, -2, -2]
        assert all(axis["kind"] == "lbound" for axis in association["actual_lower_origins"])
    charges.clear()
    calls.clear()
    assert prove_native_call_coordinates(analysis, CALLER, original, consume=charges.append) is proof
    assert sum(charges) == public["source_work"]
    assert len(calls) == len(set(calls))
    assert proof.validate(analysis, CALLER, original) is proof


@pytest.mark.parametrize("change", ["source", "bound", "declaring_scope", "parameter"])
def test_shared_constant_bound_memo_is_fresh_and_never_hides_authority_changes(tmp_path, change):
    path, analysis, original = wide_coordinate_fixture(tmp_path, lower="origin")
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    child = analysis.routines[CHILD]
    if change == "source":
        path.write_text(path.read_text() + "\n! changed original source\n")
    elif change == "bound":
        child.scope.bindings["tile1"].lower_bound_nodes = analysis.routines[CALLER].scope.bindings["n"].lower_bound_nodes
    elif change == "declaring_scope":
        child.scope.bindings["tile1"].declaring_scope = analysis.routines[CALLER].scope
    else:
        analysis.modules["coordinate_owner"].bindings["origin"].kind = 8
    with pytest.raises(CompilationError):
        proof.validate(analysis, CALLER, original)


def test_shared_constant_bound_overflow_never_populates_a_token(tmp_path):
    _path, analysis, original = wide_coordinate_fixture(tmp_path, lower="2147483647+1")
    with pytest.raises(CompilationError, match="INTEGER|range|overflow"):
        prove_native_call_coordinates(analysis, CALLER, original)
    assert not getattr(analysis, "_native_definition_proofs", {})


def test_shared_constant_bound_reuse_still_charges_parent_aggregate_work(tmp_path):
    _path, analysis, original = wide_coordinate_fixture(tmp_path)
    charges = []
    proof = prove_native_call_coordinates(analysis, CALLER, original, consume=charges.append)
    expected = sum(charges)
    charges.clear()

    def bounded(amount):
        charges.append(amount)
        if sum(charges) >= expected:
            raise CompilationError("parent mapped coordinate aggregate budget exhausted")

    with pytest.raises(CompilationError, match="parent mapped coordinate aggregate") as caught:
        prove_native_call_coordinates(analysis, CALLER, original, consume=bounded)
    assert not isinstance(caught.value, NativeCallCoordinatesUnsupported)
    assert sum(charges) == expected
    assert getattr(analysis, "_native_definition_proofs", {})[proof.identity] is proof


def wildcard_component_fixture(tmp_path):
    path, _analysis, _original = wide_coordinate_fixture(tmp_path)
    object_path = tmp_path / "object_authority.f90"
    object_path.write_text("""module field_owner
implicit none
type configuration
real(8)::delta
real(8),allocatable::unrelated(:)
end type
type(configuration)::settings
end module
module object_exports
use field_owner,only:options=>settings
end module
""")
    namespaces = []
    for index in range(48):
        namespace = tmp_path / f"namespace_{index}.f90"
        name = "delta" if index == 0 else f"unrelated{index}"
        namespace.write_text(f"module ambient{index}\nreal(8)::{name}\nend module\n")
        namespaces.append(namespace)
    original_types = """type configuration
real(8)::delta
real(8),allocatable::unrelated(:)
end type
type(configuration)::options
"""
    uses = "use object_exports,only:options\n" + "".join(f"use ambient{index}\n" for index in range(48))
    path.write_text(path.read_text().replace("module coordinate_owner\n", "module coordinate_owner\n" + uses)
                    .replace(original_types, ""))
    analysis = SourceEffects([path, object_path, *namespaces])
    return path, object_path, analysis, analysis.routines[CALLER].execution.content[0]


def test_imported_component_member_never_fans_out_through_ambient_wildcards(tmp_path, monkeypatch):
    from compiler.frontend import native_call_coordinates, source_functions

    _path, _objects, analysis, original = wildcard_component_fixture(tmp_path)
    original_validate = native_call_coordinates._validate_resolution
    lexical_names = []

    def validate(analysis, scope, nodes, **options):
        nodes = tuple(nodes)
        lexical_names.extend(str(node).lower() for node in nodes if type(node).__name__ == "Name")
        lexical_names.extend(options.get("names", ()))
        return original_validate(analysis, scope, nodes, **options)

    monkeypatch.setattr(native_call_coordinates, "_validate_resolution", validate)
    monkeypatch.setattr(source_functions, "_validate_resolution", validate)
    charges = []
    proof = prove_native_call_coordinates(analysis, CALLER, original, consume=charges.append)
    assert analysis.operation_limit == 256
    assert proof.public()["source_work"] == sum(charges) <= 256
    assert len(proof.public()["coordinate_associations"]) == 13
    component, = proof.public()["original_scalar_components"]
    assert component["object_resource"] == "field_owner::settings"
    assert component["component_path"] == ["delta"]
    # A real scalar named delta exists in an ambient wildcard module. It has
    # no lexical role in options%delta and supplies no field authority.
    assert "options" in lexical_names
    assert "delta" not in lexical_names
    assert proof.validate(analysis, CALLER, original) is proof


def test_original_member_occurrence_keeps_syntax_charge_and_subscript_names(tmp_path):
    from fparser.two.Fortran2003 import Call_Stmt

    from compiler.frontend.native_call_coordinates import _call_resolution_nodes
    from compiler.frontend.source_functions import _walk

    fixture(tmp_path)  # Initialize the supported parser before direct syntax construction.
    original = Call_Stmt("call paint(options%delta(delta),field)")
    charges = []
    nodes = tuple(_call_resolution_nodes(original, tick=charges.append))
    assert charges == ["syntax"]
    assert len(nodes) + len(charges) == len(tuple(_walk(original)))
    assert [str(node).lower() for node in nodes if type(node).__name__ == "Name"] == [
        "paint", "options", "delta", "field"]
    # The retained delta is the original index occurrence, not the field's
    # identical spelling; no original node is replaced or granted authority.
    member, index = [node for node in _walk(original) if type(node).__name__ == "Name"
                     and str(node).lower() == "delta"]
    assert all(node is not member for node in nodes)
    assert any(node is index for node in nodes)


@pytest.mark.parametrize("change", ["source", "member", "root_binding", "rename"])
def test_role_aware_field_resolution_retains_exact_import_and_type_authority(tmp_path, change):
    _path, objects, analysis, original = wildcard_component_fixture(tmp_path)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    owner = analysis.modules["field_owner"]
    if change == "source":
        objects.write_text(objects.read_text() + "\n! original type source changed\n")
    elif change == "member":
        schema = owner.component_types["configuration"]
        schema.members["delta"] = schema.members["unrelated"]
    elif change == "root_binding":
        owner.bindings["settings"].declaring_scope = analysis.routines[CHILD].scope
    else:
        analysis.modules["object_exports"].imports["options"] = ("ambient0", "delta")
    with pytest.raises(CompilationError):
        proof.validate(analysis, CALLER, original)


@pytest.mark.parametrize("to_child", [False, True])
@pytest.mark.parametrize("defined", [False, True])
def test_wide_mapping_has_separate_bounded_validation_and_transfer_phases(tmp_path, to_child, defined):
    _path, analysis, original = wide_coordinate_fixture(tmp_path, lower="-2+0+0", component_siblings=17)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    associations = proof.public()["coordinate_associations"]
    incoming = {item["actual_resource" if to_child else "formal_resource"]: whole(item["rank"]) if defined else ()
                for item in associations}
    expected = {item["formal_resource" if to_child else "actual_resource"]: whole(item["rank"]) if defined else ()
                for item in associations}
    source_work = proof.public()["source_work"]
    assert len(associations) == 13
    assert analysis.operation_limit == 256
    assert source_work == analysis.operation_limit
    # This source query is legal on its own. Even the none-coverage transfer
    # would exceed its remaining budget if the two phases shared a counter.
    assert source_work + len(associations) > analysis.operation_limit
    charges = []
    mapping = proof.caller_to_child if to_child else proof.child_to_caller
    assert mapping(incoming, consume=charges.append) == expected
    transfer_work = sum(charges) - source_work
    assert 0 < transfer_work <= analysis.operation_limit
    assert sum(charges) > analysis.operation_limit


@pytest.mark.parametrize("phase", ["validation", "transfer"])
def test_parent_coordinate_budget_is_shared_across_both_phases(tmp_path, phase):
    _path, analysis, original = fixture(tmp_path)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    source_work = proof.public()["source_work"]
    maximum = source_work - 1 if phase == "validation" else source_work + 1
    charges = []

    def consume(amount):
        charges.append(amount)
        if sum(charges) > maximum:
            raise CompilationError("parent aggregate coordinate budget exhausted")

    with pytest.raises(CompilationError, match="parent aggregate coordinate budget") as caught:
        proof.child_to_caller({"argument::dst": whole()}, consume=consume)
    assert not isinstance(caught.value, NativeCallCoordinatesUnsupported)
    assert sum(charges) == maximum + 1


def test_local_source_validation_budget_exhaustion_remains_hard_before_mapping(tmp_path):
    _path, analysis, original = fixture(tmp_path)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    analysis.operation_limit = proof.public()["source_work"] - 1
    with pytest.raises(CompilationError, match="budget exhausted") as caught:
        proof.child_to_caller({"argument::dst": whole()})
    assert not isinstance(caught.value, NativeCallCoordinatesUnsupported)


def test_local_rectangle_transfer_budget_exhaustion_does_not_reset_parent_work(tmp_path):
    _path, analysis, original = fixture(tmp_path)
    proof = prove_native_call_coordinates(analysis, CALLER, original)
    boxes = tuple(interval(index, index)[0] for index in range(32))
    charges = []
    with pytest.raises(CompilationError, match="aggregate budget exhausted") as caught:
        proof.child_to_caller({"argument::dst": boxes, "argument::unused": boxes}, consume=charges.append)
    assert not isinstance(caught.value, NativeCallCoordinatesUnsupported)
    assert sum(charges) == proof.public()["source_work"] + analysis.operation_limit
