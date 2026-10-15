"""Original native associations do not grant numerical or storage-map authority."""
from copy import copy
from dataclasses import FrozenInstanceError

import pytest
from fparser.two.utils import walk

from compiler.frontend.call_bindings import (
    resolve_original_call,
    resolve_original_function,
    resolve_source_call,
    resolve_source_function,
)
from compiler.frontend.resource_reachability import prove_resource_reachability
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module work
implicit none
interface add
module procedure add2,add_integer
end interface
real(8)::old(-1:3,-1:3,-1:3),dz(4)
contains
subroutine owner(a,n,dt,flux)
real(8),intent(inout)::a(:,:,:)
integer,intent(in)::n
real(8),intent(in)::dt,flux
call add(a(:,:,n),-dt*flux/dz(n))
end subroutine
subroutine add2(a,value)
real(8),intent(inout)::a(:,:)
real(8),intent(in)::value
a(1,1)=a(1,1)+value
end subroutine
subroutine add_integer(a,value)
real(8),intent(inout)::a(:,:)
integer,intent(in)::value
a(1,1)=real(value,8)
end subroutine
end module
"""


def setup(tmp_path, source=SOURCE, *, seal=False):
    path = tmp_path / "calls.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if seal:
        analysis.seal_session()
    routine = analysis.routines["work::owner"]
    node = next(node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt")
    return path, analysis, routine, node


def test_indexed_scalar_expression_and_rank_reduction_keep_original_evaluation(tmp_path, monkeypatch):
    _path, analysis, owner, node = setup(tmp_path)
    monkeypatch.setattr(analysis, "summarize", lambda *args, **kwargs: pytest.fail("complete effects requested"))
    result = resolve_original_call(analysis, owner.scope, node)
    assert result.procedure == "work::add2"
    assert result.actuals[0] is node.items[1].items[0]
    assert result.associations[0].actual_type.rank == 2
    assert result.associations[0].binding is owner.scope.bindings["a"]
    assert result.associations[1].actual_type.rank == 0
    assert result.associations[1].binding is None
    assert result.validate(analysis, owner.scope, node) is result
    assert not any(result.public()[name] for name in (
        "effect_authority", "capture_authority", "alias_authority", "completion_authority", "execution_authority"))
    with pytest.raises(CompilationError, match="array-element/section actual requires in-place mapping"):
        resolve_source_call(analysis, owner.scope, node)
    proof = prove_resource_reachability(analysis, owner.qualified, (node,), (analysis.modules["work"].bindings["dz"],))
    assert proof.public()["schema_version"] == 2
    assert proof.public()["selection_covered"], proof.public()["barriers"]
    assert proof.public()["demanded_effects"][0]["may_access"] == ["read"]


OBJECT_SOURCE = """module work
implicit none
type record
real(8),allocatable::flux(:)
contains
procedure::unused
end type
type(record)::records(4)
real(8)::old(4)
contains
subroutine unused(self)
class(record)::self
end subroutine
subroutine owner()
call native(records)
end subroutine
subroutine native(points)
type(record),intent(inout),target::points(:)
points(1)%flux(1)=old(1)
end subroutine
end module
"""


@pytest.mark.parametrize("seal", [False, True])
def test_derived_array_matches_declaring_type_without_sibling_layout_capture(tmp_path, seal):
    _path, analysis, owner, node = setup(tmp_path, OBJECT_SOURCE, seal=seal)
    result = resolve_original_call(analysis, owner.scope, node)
    typed = result.associations[0].actual_type
    assert typed.rank == 1
    assert typed.schema is analysis.modules["work"].component_types["record"]
    assert result.associations[0].formal_binding is analysis.routines["work::native"].scope.bindings["points"]
    with pytest.raises(CompilationError, match="fixed nonoptional scalar original storage"):
        resolve_source_call(analysis, owner.scope, node)
    proof = prove_resource_reachability(analysis, owner.qualified, (node,), (analysis.modules["work"].bindings["old"],))
    assert proof.public()["schema_version"] == 2
    assert proof.public()["selection_covered"], proof.public()["barriers"]
    assert proof.public()["demanded_effects"][0]["may_access"] == ["read"]
    assert any(item.native.components and item.native.components[-1].name == "flux" for item in proof.obligations)


def test_allocatable_component_actual_type_is_not_storage_mapping(tmp_path):
    source = OBJECT_SOURCE.replace("call native(records)", "call native(records(1)%flux)").replace(
        "type(record),intent(inout),target::points(:)\npoints(1)%flux(1)=old(1)",
        "real(8),intent(inout)::points(:)\npoints(1)=old(1)")
    _path, analysis, owner, node = setup(tmp_path, source)
    result = resolve_original_call(analysis, owner.scope, node)
    assert result.associations[0].actual_type.rank == 1
    assert result.associations[0].actual_type.kind == 8
    assert result.associations[0].binding is analysis.modules["work"].bindings["records"]
    with pytest.raises(CompilationError):
        resolve_source_call(analysis, owner.scope, node)


@pytest.mark.parametrize(("actual", "reason"), [
    ("a(:,:,n),1.0_4", "type, kind or rank mismatch"),
    ("a(:,n,n),1.0_8", "type, kind or rank mismatch"),
    ("a(:,:,n),unknown(n)", "separate source expression authority"),
    ("a(:,:,n),-dt .custom. flux", "separate source expression authority"),
])
def test_unknown_or_incompatible_original_actual_declines(tmp_path, actual, reason):
    source = SOURCE.replace("a(:,:,n),-dt*flux/dz(n)", actual).replace("call add(", "call add2(")
    _path, analysis, owner, node = setup(tmp_path, source)
    with pytest.raises(CompilationError, match=reason):
        resolve_original_call(analysis, owner.scope, node)


def test_same_spelling_different_declaring_type_rejects(tmp_path):
    source = """module other
 type record
 real(8)::x
 end type
end module
""" + OBJECT_SOURCE.replace("subroutine native(points)", "subroutine native(points)\nuse other,only:record")
    _path, analysis, owner, node = setup(tmp_path, source)
    with pytest.raises(CompilationError, match="canonical declaring type or rank mismatch"):
        resolve_original_call(analysis, owner.scope, node)


def test_mixed_generic_cannot_choose_ambiguous_same_type(tmp_path):
    source = SOURCE.replace("integer,intent(in)::value", "real(8),intent(in)::value")
    _path, analysis, owner, node = setup(tmp_path, source)
    with pytest.raises(CompilationError, match="unresolved or ambiguous"):
        resolve_original_call(analysis, owner.scope, node)


def test_keyword_order_preserved(tmp_path):
    source = SOURCE.replace("a(:,:,n),-dt*flux/dz(n)", "value=-dt*flux/dz(n),a=a(:,:,n)")
    _path, analysis, owner, node = setup(tmp_path, source)
    result = resolve_original_call(analysis, owner.scope, node)
    assert result.formals == ("a", "value")
    assert [item.formal for item in result.original_arguments] == ["value", "a"]
    assert [item.keyword for item in result.original_arguments] == ["value", "a"]


@pytest.mark.parametrize("mutation", ["copied_token", "foreign_analysis", "cloned_call", "source", "binding", "schema", "module_role", "budget"])
def test_original_association_requires_exact_issued_and_unchanged_authority(tmp_path, mutation):
    path, analysis, owner, node = setup(tmp_path, OBJECT_SOURCE)
    result = resolve_original_call(analysis, owner.scope, node)
    target_analysis, target_scope, target_node = analysis, owner.scope, node
    if mutation == "copied_token":
        result = copy(result)
    elif mutation == "foreign_analysis":
        target_analysis = SourceEffects([path])
    elif mutation == "cloned_call":
        target_node = copy(node)
    elif mutation == "source":
        path.write_text(OBJECT_SOURCE + "!changed\n")
    elif mutation == "binding":
        analysis.modules["work"].bindings["records"].rank = 2
    elif mutation == "schema":
        analysis.modules["work"].component_types["record"].members.clear()
    elif mutation == "module_role":
        analysis.modules["work"] = copy(analysis.modules["work"])
    else:
        analysis.operation_limit += 1
    with pytest.raises(CompilationError):
        result.validate(target_analysis, target_scope, target_node)
    with pytest.raises(FrozenInstanceError):
        result.procedure = "other"


def test_source_function_association_does_not_grant_result_or_effect_authority(tmp_path):
    source = SOURCE.replace("call add(a(:,:,n),-dt*flux/dz(n))", "a(1,1,1)=value(dz(n))").replace(
        "subroutine add2(a,value)", "real(8) function value(x)\nreal(8),intent(in)::x\nvalue=x\nend function\nsubroutine add2(a,value)")
    path = tmp_path / "calls.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    owner = analysis.routines["work::owner"]
    node = next(node for node in walk(owner.execution) if type(node).__name__ in {"Part_Ref", "Function_Reference"}
                and str(node.items[0]).lower() == "value")
    result = resolve_original_function(analysis, owner.scope, node)
    assert result.procedure == "work::value"
    assert result.associations[0].actual_type.rank == 0
    with pytest.raises(CompilationError, match="array-element actual requires scalar payload coherence"):
        resolve_source_function(analysis, owner.scope, node)


def test_out_entry_kill_is_visible_through_nested_call_even_without_assignment(tmp_path):
    source = """module work
real(8)::old(4)
contains
subroutine owner()
call wrapper(old)
end subroutine
subroutine wrapper(a)
real(8),intent(inout)::a(:)
call native(a)
end subroutine
subroutine native(a)
real(8),intent(out)::a(:)
end subroutine
end module
"""
    _path, analysis, owner, node = setup(tmp_path, source)
    proof = prove_resource_reachability(analysis, owner.qualified, (node,), (analysis.modules["work"].bindings["old"],))
    assert "undefine" in proof.public()["demanded_effects"][0]["may_access"]
    assert not proof.public()["definition_authorized"]


def test_writable_expression_actual_remains_rejected(tmp_path):
    source = SOURCE.replace("real(8),intent(in)::value", "real(8),intent(out)::value").replace("call add(", "call add2(")
    _path, analysis, owner, node = setup(tmp_path, source)
    with pytest.raises(CompilationError, match="writable or descriptor actual requires original storage"):
        resolve_original_call(analysis, owner.scope, node)


def test_equal_text_call_clone_cannot_issue_original_association(tmp_path):
    _path, analysis, owner, node = setup(tmp_path)
    with pytest.raises(CompilationError, match="exact original statement"):
        resolve_original_call(analysis, owner.scope, copy(node))


def test_retained_namespace_roles_cannot_be_mutated(tmp_path):
    _path, analysis, owner, node = setup(tmp_path)
    result = resolve_original_call(analysis, owner.scope, node)
    with pytest.raises(TypeError):
        result._roles["foreign"] = owner.scope
    assert result.validate(analysis, owner.scope, node) is result
