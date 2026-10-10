"""Original private storage survives lowering without changing numerical proofs."""

import json
from dataclasses import fields, is_dataclass, replace

import pytest

from compiler.analysis import build_execution_plan
from compiler.driver.options import CompilerOptions
from compiler.frontend import lower_file
from compiler.ir import PrivateArrayOrigin, ScalarType, Symbol
from compiler.offload import analyze_offload
from compiler.offload.analysis import _workload_features
from compiler.scopes.region_dispatch import InlineRegions


def analyze(tmp_path, declarations, body, *, helpers="", kind=8):
    path = tmp_path / "generic_storage.f90"
    path.write_text(f"""module generic_storage
implicit none
contains
subroutine evaluate(a,b,n)
real({kind}),intent(in)::a(:)
real({kind}),intent(out)::b(:)
integer,intent(in)::n
integer::i,j,k
{declarations}
{body}
{helpers}
end subroutine
end module
""")
    function = lower_file(path, "evaluate")
    plan = build_execution_plan(function, options=CompilerOptions(opt_level=0))
    analysis = analyze_offload(function, plan)
    assert analysis.available, analysis.reason
    return function, plan, analysis


@pytest.mark.parametrize(("kind", "dtype"), [(4, ScalarType.REAL32), (8, ScalarType.REAL)])
def test_original_bounds_survive_helper_rebasing_and_lexical_capture(tmp_path, kind, dtype):
    function, _, analysis = analyze(tmp_path,
        f"real({kind})::tensor(-2:0,4:5),unused(16)",
        f"""do i=1,n
do j=4,5
do k=-2,0
tensor(k,j)=a(i)+real(k+j,{kind})
enddo
enddo
call shifted(tensor)
b(i)=dot_product(tensor(:,4),tensor(:,5))
enddo""",
        helpers=f"""contains
pure subroutine shifted(g)
real({kind}),intent(inout)::g(0:2,-1:0)
g(0,-1)=g(0,-1)+a(i)
end subroutine""", kind=kind)
    elements = [symbol for symbol in function.symbols
                if symbol.name.startswith("tensor_")]
    assert len(elements) == 6
    assert all(symbol.dtype is dtype for symbol in elements)
    assert {symbol.private_array_origin.bounds for symbol in elements} == {((-2, 0), (4, 5))}
    assert {symbol.private_array_origin.extents for symbol in elements} == {(3, 2)}
    assert [symbol.private_array_origin.element_offset for symbol in elements] == list(range(6))
    assert len({symbol.private_array_origin.group_id for symbol in elements}) == 1
    unit, = analysis.units
    assert unit.workload_class == "fixed_private_array_v2"
    feature = unit.workload_features.to_dict()
    assert feature["schema_version"] == 2
    assert feature["classification_complete"]
    assert feature["private_array_groups"] == 1
    assert feature["private_array_elements"] == feature["referenced_private_array_elements"] == 6
    assert feature["max_private_array_rank"] == 2
    group, = feature["private_arrays"]
    assert group["bounds"] == [[-2, 0], [4, 5]]
    assert group["extents"] == [3, 2]
    assert group["referenced_element_offsets"] == list(range(6))
    assert group["dtype"] == dtype.value
    public, = json.loads(json.dumps(analysis.to_dict()))["units"]
    assert public["workload_class"] == unit.workload_class
    assert public["workload_features"] == feature


def test_each_unit_counts_only_its_used_groups_with_full_declared_capacity(tmp_path):
    _, _, analysis = analyze(tmp_path, "real(8)::small(-1:1),pair(2),unused(16)", """
do i=1,n
small(-1)=a(i)
b(i)=small(-1)*small(-1)
enddo
do i=1,n
pair(1)=a(i)+1.0_8
pair(2)=a(i)-1.0_8
b(i)=pair(1)+pair(2)
enddo
do i=1,n
b(i)=a(i)*2.0_8
enddo
""")
    first, second, last = analysis.units
    assert first.workload_features.private_array_elements == 3
    assert first.workload_features.referenced_private_array_elements == 1
    assert first.workload_features.private_arrays[0].element_offsets == (0,)
    assert second.workload_features.private_array_elements == 2
    assert second.workload_features.referenced_private_array_elements == 2
    assert first.workload_features.private_arrays[0].group_id != second.workload_features.private_arrays[0].group_id
    assert last.workload_class == "scalar_expression_v2"
    assert last.workload_features.private_array_groups == last.workload_features.private_array_elements == 0
    assert last.workload_features.max_private_array_elements == last.workload_features.max_private_array_rank == 0
    assert all(unit.workload_features.private_array_elements < 16 for unit in analysis.units)


def test_repeated_helper_activations_have_distinct_original_storage(tmp_path):
    _, _, analysis = analyze(tmp_path, "", """do i=1,n
b(i)=measure(a(i))+measure(a(i)+1.0_8)
enddo""", helpers="""contains
pure function measure(x) result(value)
real(8),intent(in)::x
real(8)::value,private_vector(-1:0)
private_vector(-1)=x
private_vector(0)=x*x
value=dot_product(private_vector,private_vector)
end function""")
    unit, = analysis.units
    feature = unit.workload_features
    assert feature.private_array_groups == 2
    assert feature.private_array_elements == feature.referenced_private_array_elements == 4
    assert feature.max_private_array_elements == 2
    assert len({group.group_id for group in feature.private_arrays}) == 2
    assert {group.bounds for group in feature.private_arrays} == {((-1, 0),)}


def without_origins(value):
    if isinstance(value, Symbol):
        return replace(value, private_array_origin=None)
    if isinstance(value, tuple):
        return tuple(without_origins(child) for child in value)
    if is_dataclass(value):
        return replace(value, **{field.name: without_origins(getattr(value, field.name)) for field in fields(value)})
    return value


def test_cost_provenance_preserves_numerical_identity_proofs_and_work(tmp_path):
    function, plan, analysis = analyze(tmp_path, "real(8)::local(-1:0)", """do i=1,n
local(-1)=a(i)
local(0)=a(i)*a(i)
b(i)=local(-1)+local(0)
enddo""")
    stripped = without_origins(function)
    assert stripped == function
    assert hash(stripped.symbols[-1]) == hash(function.symbols[-1])
    stripped_plan = build_execution_plan(stripped, options=CompilerOptions(opt_level=0))
    assert stripped_plan == plan
    old = analyze_offload(stripped, stripped_plan)
    for before, after in zip(old.units, analysis.units, strict=True):
        assert before.footprints == after.footprints
        assert before.work_per_iteration == after.work_per_iteration
        assert before.work_is_upper_bound == after.work_is_upper_bound
        assert before.intrinsic_work_per_iteration == after.intrinsic_work_per_iteration
        assert before.arithmetic_work_per_iteration == after.arithmetic_work_per_iteration
    assert old.available == analysis.available
    assert old.chunk == analysis.chunk
    # Shared artifact canonicalization retains typed provenance in its identity.
    canonical = InlineRegions.artifact_ir(function, str(tmp_path), "generic-proof")
    assert [symbol.private_array_origin for symbol in canonical.symbols] == [
        symbol.private_array_origin for symbol in function.symbols]
    assert repr(function) != repr(stripped)


def test_unrelated_array_declarations_and_scalar_names_do_not_classify_work(tmp_path):
    _, _, analysis = analyze(tmp_path, "real(8)::matrix_0,unrelated(16)", """do i=1,n
matrix_0=a(i)*a(i)
b(i)=matrix_0
enddo""")
    unit, = analysis.units
    assert unit.workload_class == "scalar_expression_v2"
    assert unit.workload_features.to_dict()["private_arrays"] == []


def test_classification_never_applies_calibration_shape_limits_to_gpu_legality(tmp_path):
    _, _, analysis = analyze(tmp_path, "real(8)::buffer(256)", """do i=1,n
buffer(256)=a(i)
b(i)=buffer(256)
enddo""")
    unit, = analysis.units
    assert unit.workload_class == "fixed_private_array_v2"
    assert unit.workload_features.max_private_array_elements == 256
    assert unit.workload_features.referenced_private_array_elements == 1


@pytest.mark.parametrize("origin", [
    PrivateArrayOrigin(0, ((1, 2),), 0, schema_version=2),
    PrivateArrayOrigin(0, ((1, 2),), 0, schema_version=True),
    PrivateArrayOrigin(0, ((1, 2),), -1),
    PrivateArrayOrigin(0, ((1, 257),), 0),
    PrivateArrayOrigin(0, (), 0),
    PrivateArrayOrigin(True, ((1, 2),), 0),
])
def test_malformed_cost_provenance_does_not_invent_scalar_classification(origin):
    features = _workload_features((Symbol(0, "scalar", ScalarType.REAL, private_array_origin=origin),))
    assert not features.classification_complete
    assert features.reason


@pytest.mark.parametrize("second", [
    PrivateArrayOrigin(0, ((1, 3),), 1),
    PrivateArrayOrigin(0, ((1, 2),), 0),
])
def test_conflicting_group_or_element_origins_are_unclassified(second):
    first = Symbol(0, "first", ScalarType.REAL, private_array_origin=PrivateArrayOrigin(0, ((1, 2),), 0))
    other = Symbol(1, "other", ScalarType.REAL, private_array_origin=second)
    features = _workload_features((first, other))
    assert not features.classification_complete
    assert "conflicting" in features.reason
