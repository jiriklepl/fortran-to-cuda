"""An original fixed-table element stays one scalar association through calls."""

from copy import copy, deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.frontend.call_bindings import resolve_source_call
from compiler.frontend.native_call_coordinates import prove_native_call_coordinates
from compiler.frontend.source_effects import SourceEffects, _children
from compiler.ir import CompilationError
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

ENTRY = "tide_owner::advance"
ROOTS = ("argument::a", "argument::b")
TABLE_ROOT = "tide_values::weights"

TABLE_SOURCE = """module tide_values
implicit none
real(8)::weights(-3:1)=[0.5_8,1.5_8,2.5_8,3.5_8,4.5_8]
integer::selector=-3
end module
"""

NUMERICAL_SOURCE = """module tide_numerics
use tide_values,only:selector
implicit none
integer::bridge_visits=0,leaf_visits=0
contains
subroutine bridge(x,y,n,ratio)
real(8),intent(in)::x(-2:)
real(8),intent(inout)::y(-2:)
integer,intent(in)::n
real(8),intent(in)::ratio
bridge_visits=bridge_visits+1
selector=1
call leaf(x,y,n,ratio)
end subroutine
subroutine leaf(x,y,n,scale)
real(8),intent(in)::x(-2:)
real(8),intent(inout)::y(-2:)
integer,intent(in)::n
real(8),intent(in)::scale
integer::i
leaf_visits=leaf_visits+1
do i=-2,n-3
y(i)=scale*x(i)+real(i,8)
enddo
end subroutine
end module
"""

OWNER_SOURCE = """module tide_owner
use tide_values,only:weights,selector
use tide_numerics,only:bridge
implicit none
contains
subroutine advance(a,b,n,next_selector,enabled)
real(8),intent(inout)::a(-2:),b(-2:)
integer,intent(in)::n,next_selector
logical,intent(in)::enabled
integer::i
selector=next_selector
do i=-2,n-3
b(i)=a(i)+real(i+4,8)
enddo
if(enabled) then
call bridge(b,a,n,weights(selector))
endif
do i=-2,n-3
b(i)=a(i)+2.0_8
enddo
end subroutine
end module
"""


def scalar_element_sources(directory, *, precision=8, table_source=TABLE_SOURCE,
                           numerical_source=NUMERICAL_SOURCE, owner_source=OWNER_SOURCE):
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(directory / name for name in ("values.f90", "numerics.f90", "owner.f90"))
    for path, source in zip(paths, (table_source, numerical_source, owner_source), strict=True):
        path.write_text(source.replace("real(8)", f"real({precision})")
                        .replace("_8", f"_{precision}").replace(",8)", f",{precision})"))
    return paths


def scalar_element_call(directory, **sources):
    paths = scalar_element_sources(directory, **sources)
    analysis = SourceEffects(paths)
    caller = analysis.routines[ENTRY]
    branch = next(node for node in _children(caller.execution) if type(node).__name__ == "If_Construct")
    original, = (node for node in branch.content if type(node).__name__ == "Call_Stmt")
    resolved = resolve_source_call(analysis, caller.scope, original)
    return paths, analysis, caller, original, resolved


@pytest.mark.parametrize("precision", [4, 8])
def test_element_is_not_an_array_or_renamed_scalar_resource(tmp_path, precision):
    _paths, analysis, caller, original, resolved = scalar_element_call(tmp_path, precision=precision)
    mapping, = (item for item in resolved.mappings if item.formal == "ratio")
    element = mapping.scalar_element
    assert element is not None
    assert element.actual is mapping.actual
    assert element.binding is analysis.modules["tide_values"].bindings["weights"]
    assert element.binding.rank == 1
    assert element.binding.kind == precision
    assert element.index.resource == "tide_values::selector"
    assert element.bounds == (-3, 1)
    assert mapping.resource is None
    assert mapping.public()["storage"] == "scalar_element"
    assert "argument::ratio" not in resolved.resource_mapping
    assert TABLE_ROOT not in resolved.resource_mapping.values()

    coordinates = prove_native_call_coordinates(analysis, caller.qualified, original)
    assert coordinates.scalar_elements == (element,)
    assert coordinates.resource_mapping == {"argument::x": "argument::b", "argument::y": "argument::a"}
    assert "argument::ratio" not in coordinates.scalar_mapping
    assert "argument::ratio" not in coordinates.public()["scalar_bounds"]
    assert coordinates.validate(analysis, caller.qualified, original) is coordinates


@pytest.mark.parametrize("change", ["none", "hidden_write", "writable_forward"])
def test_scalar_witness_follows_numerical_helpers_without_mapping_unrelated_elements(tmp_path, change):
    helper = """pure subroutine cell_formula(g,s,value,factor,i)
real(8),intent(out)::g(2),s
real(8),intent(in)::value,factor
integer,intent(in)::i
g(1)=value
g(2)=real(i,8)
s=factor*g(1)+g(2)
end subroutine
"""
    source = NUMERICAL_SOURCE.replace("integer::i\nleaf_visits", "integer::i\nreal(8)::g(2),s\nleaf_visits")
    source = source.replace("y(i)=scale*x(i)+real(i,8)", "call cell_formula(g,s,x(i),scale,i)\ny(i)=s")
    if change == "hidden_write":
        helper = helper.replace("pure subroutine", "subroutine").replace(
            "real(8),intent(out)", "use tide_values,only:weights\nreal(8),intent(out)", 1)
        helper = helper.replace("g(1)=value", "weights(-3)=9.0_8\ng(1)=value")
    elif change == "writable_forward":
        helper = helper.replace("real(8),intent(in)::value,factor",
                                "real(8),intent(in)::value\nreal(8),intent(inout)::factor")
    source = source.replace("end module\n", helper + "end module\n")
    _paths, analysis, caller, original, _resolved = scalar_element_call(tmp_path, numerical_source=source)
    if change != "none":
        with pytest.raises(CompilationError, match="backing storage|unchanged scalar INTENT"):
            prove_native_call_coordinates(analysis, caller.qualified, original)
        return

    coordinates = prove_native_call_coordinates(analysis, caller.qualified, original)
    closure = coordinates.public()["scalar_element_closure"]
    helper_record, = (item for item in closure["procedures"] if item["procedure"] == "tide_numerics::cell_formula")
    assert helper_record["associated_formals"] == ["argument::factor"]
    assert closure["effect_authority"] is False
    assert closure["definition_authority"] is False
    assert closure["completion_authority"] is False
    assert coordinates.validate(analysis, caller.qualified, original) is coordinates


@pytest.mark.parametrize("change", ["write", "write_restore", "aliased_out", "unknown_call", "writable_formal"])
def test_child_closure_cannot_change_or_escape_the_backing_table(tmp_path, change):
    source = NUMERICAL_SOURCE
    if change in {"write", "write_restore", "aliased_out"}:
        source = source.replace("use tide_values,only:selector", "use tide_values,only:selector,weights")
    if change == "write":
        source = source.replace("selector=1", "weights(-3)=9.0_8\nselector=1")
    elif change == "write_restore":
        source = source.replace("selector=1", "weights(-3)=weights(-3)+1.0_8\nweights(-3)=weights(-3)-1.0_8\nselector=1")
    elif change == "aliased_out":
        source = source.replace("selector=1", "call replace_values(weights)\nselector=1")
        source = source.replace("subroutine leaf(x,y,n,scale)",
            "subroutine replace_values(values)\nreal(8),intent(out)::values(:)\nvalues=7.0_8\n"
            "end subroutine\nsubroutine leaf(x,y,n,scale)")
    elif change == "unknown_call":
        source = source.replace("selector=1", "call opaque(ratio)\nselector=1")
    else:
        source = source.replace("real(8),intent(in)::ratio", "real(8),intent(inout)::ratio")
    def resolve_and_prove():
        _paths, analysis, caller, original, _resolved = scalar_element_call(tmp_path, numerical_source=source)
        prove_native_call_coordinates(analysis, caller.qualified, original)
    with pytest.raises(CompilationError):
        resolve_and_prove()


@pytest.mark.parametrize("attribute", ["target", "pointer"])
def test_association_rejects_storage_with_other_possible_views(tmp_path, attribute):
    if attribute == "target":
        source = TABLE_SOURCE.replace("real(8)::weights", "real(8),target::weights")
    else:
        source = TABLE_SOURCE.replace("real(8)::weights(-3:1)", "real(8),target::storage(-3:1)")
        source = source.replace("integer::selector", "real(8),pointer::weights(:)=>storage\ninteger::selector")
    def resolve_and_prove():
        _paths, analysis, caller, original, _resolved = scalar_element_call(tmp_path, table_source=source)
        prove_native_call_coordinates(analysis, caller.qualified, original)
    with pytest.raises(CompilationError):
        resolve_and_prove()


def test_original_storage_association_cannot_hide_a_backing_table_write(tmp_path):
    table_source = TABLE_SOURCE.replace("integer::selector=-3",
        "real(8)::other(-3:1)\nequivalence(weights,other)\ninteger::selector=-3")
    numerical_source = NUMERICAL_SOURCE.replace("use tide_values,only:selector",
        "use tide_values,only:selector,other").replace("selector=1", "other(-3)=9.0_8\nselector=1")

    def resolve_and_prove():
        _paths, analysis, caller, original, _resolved = scalar_element_call(
            tmp_path, table_source=table_source, numerical_source=numerical_source)
        prove_native_call_coordinates(analysis, caller.qualified, original)

    with pytest.raises(CompilationError):
        resolve_and_prove()


@pytest.mark.parametrize("mutation", ["source", "backing_binding", "formal_binding", "actual_ast"])
def test_registered_element_association_rejects_stale_original_authority(tmp_path, mutation):
    paths, analysis, caller, original, _resolved = scalar_element_call(tmp_path)
    proof = prove_native_call_coordinates(analysis, caller.qualified, original)
    if mutation == "source":
        paths[0].write_text(TABLE_SOURCE + "\n! changed source\n")
    elif mutation == "backing_binding":
        scope = analysis.modules["tide_values"]
        scope.bindings["weights"] = copy(scope.bindings["weights"])
    elif mutation == "formal_binding":
        scope = analysis.routines["tide_numerics::bridge"].scope
        scope.bindings["ratio"] = copy(scope.bindings["ratio"])
    else:
        arguments = copy(original.items[1])
        arguments.items = (*arguments.items[:-1], copy(arguments.items[-1]))
        original.items = (original.items[0], arguments)
    with pytest.raises(CompilationError):
        proof.validate(analysis, caller.qualified, original)


def test_copied_and_foreign_tokens_do_not_grant_scalar_association(tmp_path):
    _paths, analysis, caller, original, _resolved = scalar_element_call(tmp_path / "first")
    proof = prove_native_call_coordinates(analysis, caller.qualified, original)
    _paths, other, _caller, _original, _resolved = scalar_element_call(tmp_path / "other")
    with pytest.raises(CompilationError, match="registered"):
        copy(proof).validate(analysis, caller.qualified, original)
    with pytest.raises(CompilationError, match="registered"):
        proof.validate(other, caller.qualified, original)


def test_scalar_association_keeps_bounded_failures_hard(tmp_path):
    _paths, analysis, caller, original, _resolved = scalar_element_call(tmp_path)
    proof = prove_native_call_coordinates(analysis, caller.qualified, original)
    analysis.operation_limit = 1
    with pytest.raises(CompilationError, match="budget|bounded"):
        proof.validate(analysis, caller.qualified, original)
