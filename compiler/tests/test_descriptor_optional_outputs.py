"""Original optional output presence preserves storage, never payload facts."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module optional_outputs
contains
subroutine leaf(a,result,vector)
real(8),intent(inout)::a(:)
real(8),optional,intent(out)::result,vector(3)
if(present(result))result=sum(a)
if(present(vector))then
vector=1.0_8
endif
a(1)=a(1)+1
end subroutine
subroutine wrapper(a,result)
real(8),intent(inout)::a(:)
real(8),optional,intent(out)::result
call leaf(a,result)
end subroutine
subroutine advance(a)
real(8),allocatable,intent(inout)::a(:)
real(8)::value
call wrapper(a,value)
deallocate(a)
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE):
    path = tmp_path / "optional.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    entry = "optional_outputs::advance"
    routine = analysis.routines[entry]
    selection = (analysis.structure(entry).node_id(routine.execution.content[0]),)
    return path, analysis, entry, selection


def prove(data):
    _path, analysis, entry, selection = data
    return prove_descriptor_window(analysis, entry, selection, ("argument::a",))


@pytest.mark.parametrize("actual", ["value", None])
def test_original_scalar_outputs_and_omitted_fixed_vector_keep_exact_presence(tmp_path, actual):
    source = SOURCE.replace("call wrapper(a,value)", "call wrapper(a" + ("," + actual if actual else "") + ")")
    data = fixture(tmp_path, source)
    path, analysis, entry, selected = data
    before = path.read_bytes()
    public = prove(data).public()
    scalar = public["native_optional_scalar_requirements"]
    assert {item["presence"] for item in scalar} == {"forwarded_optional", "supplied" if actual else "omitted"}
    vector, = public["native_optional_array_requirements"]
    assert vector["formal"] == "vector"
    assert vector["presence"] == "omitted"
    assert vector["actual"] is None
    assert vector["actual_descriptor"] is None
    assert vector["fixed_formal_bounds"] == [1, 3]
    for item in [public, *scalar, vector]:
        assert item["entry_definition_authority"] is False
        assert item["execution_authorized"] is False
    for item in [*scalar, vector]:
        assert item["payload_access_authorized"] is False
        assert item["value_evaluation_authorized"] is False
        assert item["original_positive_present_required"] is True
    assert set(public["source_call_closure"]) == {"optional_outputs::wrapper", "optional_outputs::leaf"}
    assert path.read_bytes() == before
    assert prove(data).validate(analysis, entry, selected)


@pytest.mark.parametrize("dtype", ["real(4)", "real(8)", "integer(4)", "integer(8)"])
def test_ordinary_fixed_numeric_kinds_are_descriptor_only(tmp_path, dtype):
    data = fixture(tmp_path, SOURCE.replace("real(8)", dtype).replace("1.0_8", "1"))
    assert prove(data).public()["native_optional_array_requirements"][0]["formal_descriptor"]["rank"] == 1


@pytest.mark.parametrize("attrs", ["pointer", "allocatable", "target", "volatile", "asynchronous", "value"])
def test_optional_output_unsafe_alias_or_dynamic_attributes_stay_native(tmp_path, attrs):
    data = fixture(tmp_path, SOURCE.replace("optional,intent(out)", "optional," + attrs + ",intent(out)"))
    with pytest.raises(CompilationError):
        prove(data)


@pytest.mark.parametrize("shape", [":", "*", "2,2", "extent"])
def test_omitted_array_requires_one_exact_fixed_explicit_vector(tmp_path, shape):
    source = SOURCE.replace("::result,vector(3)", "::result,vector(" + shape + ")")
    source = source.replace("subroutine leaf(a,result,vector)", "subroutine leaf(a,result,vector)\ninteger::extent")
    with pytest.raises(CompilationError):
        prove(fixture(tmp_path, source))


@pytest.mark.parametrize(("shape", "bounds"), [("-1:1", [-1, 1]), ("0:-1", [0, -1])])
def test_original_explicit_bounds_are_recorded_without_reached_descriptor_reads(tmp_path, shape, bounds):
    data = fixture(tmp_path, SOURCE.replace("vector(3)", "vector(" + shape + ")"))
    assert prove(data).public()["native_optional_array_requirements"][0]["fixed_formal_bounds"] == bounds


@pytest.mark.parametrize("body", [
    "vector=1", "if(.not.present(vector))vector=1", "if(present(vector).and.sum(vector)>0)vector=1",
    "if(present(vector))then\ncontinue\nelse\nvector=1\nendif",
    "if(present(vector))then\ncontinue\nelse if(.true.)then\nvector=1\nendif",
])
def test_absent_array_cannot_be_evaluated_outside_positive_original_present(tmp_path, body):
    source = SOURCE.replace("if(present(vector))then\nvector=1.0_8\nendif", body)
    with pytest.raises(CompilationError, match="presence|optional|storage"):
        prove(fixture(tmp_path, source))


def test_matching_presence_on_another_formal_cannot_guard_vector(tmp_path):
    source = SOURCE.replace("if(present(vector))then", "if(present(result))then")
    with pytest.raises(CompilationError):
        prove(fixture(tmp_path, source))


def test_single_line_positive_present_and_nested_original_guards_remain_native(tmp_path):
    source = SOURCE.replace("if(present(vector))then\nvector=1.0_8\nendif",
                            "if(present(vector))then\nif(size(a)>0)vector=1.0_8\nendif")
    assert prove(fixture(tmp_path, source)).public()["native_optional_array_requirements"][0]["presence"] == "omitted"


@pytest.mark.parametrize("actual", ["scratch", "scratch(:)"])
def test_initial_slice_does_not_admit_supplied_optional_vector_mapping(tmp_path, actual):
    source = SOURCE.replace("real(8)::value", "real(8)::value,scratch(3)")
    source = source.replace("call wrapper(a,value)", "call leaf(a,value," + actual + ")")
    with pytest.raises(CompilationError):
        prove(fixture(tmp_path, source))


@pytest.mark.parametrize("tail", ["call unknown(vector)", "deallocate(a)", "allocate(scratch(3))"])
def test_every_original_callee_branch_and_late_storage_effect_is_scanned(tmp_path, tail):
    source = SOURCE.replace("real(8),intent(inout)::a(:)", "real(8),allocatable,intent(inout)::a(:)")
    source = source.replace("if(present(result))result=sum(a)", "real(8),allocatable::scratch(:)\nif(present(result))result=sum(a)")
    source = source.replace("vector=1.0_8", "vector=1.0_8\n" + tail)
    with pytest.raises(CompilationError):
        prove(fixture(tmp_path, source))


@pytest.mark.parametrize("attribute", ["target", "pointer", "allocatable", "parameter"])
def test_writable_scalar_actual_requires_original_independent_storage(tmp_path, attribute):
    source = SOURCE.replace("real(8)::value", "real(8)," + attribute + "::value" + ("=1" if attribute == "parameter" else ""))
    with pytest.raises(CompilationError):
        prove(fixture(tmp_path, source))


@pytest.mark.parametrize("declaration", ["common /shared/ value", "equivalence(value,alias)", "namelist /shared/ value"])
def test_original_declaration_aliases_cannot_become_independent_output_storage(tmp_path, declaration):
    source = SOURCE.replace("real(8)::value", "real(8)::value,alias\n" + declaration)
    with pytest.raises(CompilationError):
        prove(fixture(tmp_path, source))


def test_two_writable_optional_formals_cannot_borrow_the_same_actual(tmp_path):
    source = SOURCE.replace("subroutine leaf(a,result,vector)", "subroutine leaf(a,result,vector,other)")
    source = source.replace("::result,vector(3)", "::result,vector(3),other")
    source = source.replace("a(1)=a(1)+1", "if(present(other))other=2\na(1)=a(1)+1")
    source = source.replace("call wrapper(a,value)", "call leaf(a,value,other=value)")
    with pytest.raises(CompilationError):
        prove(fixture(tmp_path, source))


def test_replaced_original_binding_or_forged_parent_cannot_supply_presence(tmp_path):
    data = fixture(tmp_path)
    proof = prove(data)
    _path, analysis, entry, selected = data
    routine = analysis.routines["optional_outputs::leaf"]
    routine.scope.bindings["vector"] = copy(routine.scope.bindings["vector"])
    with pytest.raises(CompilationError):
        proof.validate(analysis, entry, selected)


def test_parent_pointer_without_original_child_membership_is_not_a_guard(tmp_path):
    from compiler.frontend.descriptor_optional import original_positive_presence
    from compiler.frontend.descriptor_windows import _Check

    _path, analysis, _entry, _selected = fixture(tmp_path)
    routine = analysis.routines["optional_outputs::leaf"]
    assignment = next(node for node in walk(routine.execution)
                      if type(node).__name__ == "Assignment_Stmt" and str(node).startswith("vector"))
    node = copy(assignment.items[0])
    node.parent = assignment
    assert not original_positive_presence(_Check(analysis), routine, node, routine.scope.bindings["vector"])
