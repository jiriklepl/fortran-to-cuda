"""Descriptor evidence for selected fields never becomes whole-record layout."""
from copy import copy

import pytest

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "metadata_steps::advance"


def fixture(tmp_path, *, field="real(8)::weight", root="allocatable", action="a(marks(i)%index)=marks(i)%weight",
            module_extra=""):
    path = tmp_path / "metadata.f90"
    path.write_text(f"""module metadata_types
type record
integer::index
{field}
real(8),allocatable::unrelated(:)
end type
type(record),{root}::marks(:)
{module_extra}
end module
module metadata_steps
use metadata_types
contains
subroutine update(a)
real(8),intent(inout)::a(:)
integer::i,n
n=size(marks)
do i=1,n
{action}
enddo
end subroutine
subroutine advance(a)
real(8),allocatable,intent(inout)::a(:)
real(8),allocatable::temporary(:)
call update(a)
call move_alloc(a,temporary)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    routine = analysis.routines[ENTRY]
    call = routine.execution.content[0]
    selection = (analysis.structure(ENTRY).node_id(call),)
    return path, analysis, selection


def prove(analysis, selection):
    return prove_descriptor_window(analysis, ENTRY, selection, ("argument::a",))


def test_original_descriptor_and_fixed_selected_fields_ignore_unaccessed_dynamic_siblings(tmp_path):
    path, analysis, selection = fixture(tmp_path)
    before = path.read_bytes()
    proof = prove(analysis, selection)
    public = proof.public()
    item, = public["native_metadata_descriptor_requirements"]
    assert item["original_binding"]["resource"] == "metadata_types::marks"
    assert item["declaring_type"] == {"module": "metadata_types", "name": "record"}
    assert set(item["selected_fields"]) == {"index", "weight"}
    assert item["original_reached_selector_required"]
    for name in ("whole_object_layout_authorized", "whole_object_copy_authorized", "capture_authorized",
                 "entry_definition_authority", "payload_access_authorized", "execution_authorized"):
        assert not item[name]
    assert not public["entry_definition_authority"]
    assert not public["fixed_source_object_requirements"]
    assert path.read_bytes() == before
    assert not getattr(analysis, "_native_metadata", False)


def test_scalar_field_write_is_not_a_root_descriptor_or_definition_grant(tmp_path):
    _path, analysis, selection = fixture(tmp_path, action="marks(i)%weight=a(marks(i)%index)")
    public = prove(analysis, selection).public()
    assert public["resources"][0]["stable"]
    assert not public["resources"][0]["entry_definition_authority"]
    assert not public["native_metadata_descriptor_requirements"][0]["payload_access_authorized"]


@pytest.mark.parametrize("field", ["real(8),pointer::weight", "real(8),allocatable::weight",
                                     "real(8)::weight(3)"])
def test_dynamic_pointer_and_array_fields_need_separate_storage_authority(tmp_path, field):
    _path, analysis, selection = fixture(tmp_path, field=field)
    with pytest.raises(CompilationError, match="association|lifetime|scalar numeric"):
        prove(analysis, selection)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("root", ["pointer", "allocatable,target", "allocatable,volatile"])
def test_array_object_pointer_and_alias_uncertainty_stays_a_boundary(tmp_path, root):
    _path, analysis, selection = fixture(tmp_path, root=root)
    with pytest.raises(CompilationError, match="metadata requires|metadata object|independent original"):
        prove(analysis, selection)


@pytest.mark.parametrize("action", ["marks(i)=marks(i)", "marks(i)%unrelated=1.0_8", "call absent(marks)",
                                      "deallocate(marks)"])
def test_whole_objects_dynamic_components_unknown_calls_and_retirement_do_not_gain_authority(tmp_path, action):
    _path, analysis, selection = fixture(tmp_path, action=action)
    with pytest.raises(CompilationError):
        prove(analysis, selection)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_source_and_selected_component_declaration_are_revalidated(tmp_path):
    path, analysis, selection = fixture(tmp_path)
    proof = prove(analysis, selection)
    public = proof.public()
    public["native_metadata_descriptor_requirements"][0]["selected_fields"].clear()
    assert proof.public()["native_metadata_descriptor_requirements"][0]["selected_fields"]
    with pytest.raises(CompilationError, match="registered original interval"):
        copy(proof).validate(analysis, ENTRY, selection)
    path.write_text(path.read_text().replace("real(8)::weight", "real(8),pointer::weight"))
    with pytest.raises(CompilationError, match="source changed"):
        proof.validate(analysis, ENTRY, selection)


@pytest.mark.parametrize("declaration", ["common /unproved/ marks", "namelist /unproved/ marks",
                                          "bind(C)::marks"])
def test_declaration_alias_and_external_linkage_are_not_descriptor_authority(tmp_path, declaration):
    _path, analysis, selection = fixture(tmp_path, module_extra=declaration)
    with pytest.raises(CompilationError, match="declaration association"):
        prove(analysis, selection)
