"""Native rectangles retain original references and exact overwrite sections."""

from __future__ import annotations

import json

import pytest

from compiler.frontend.source_effects import SourceEffects


def analyze(tmp_path, body, *, declaration="real(8),intent(inout)::a(-2:,:)", extra=""):
    path = tmp_path / "native.f90"
    path.write_text("module original\nimplicit none\n" + extra + "\ncontains\nsubroutine touch(a,b,n)\n"
                    + declaration + "\nreal(8),intent(in)::b(:,:)\ninteger,intent(in)::n\ninteger::i\n"
                    + body + "\nend subroutine\nend module\n")
    source = SourceEffects([path])
    return source, source.native_sections("original::touch")


def test_public_native_sections_keep_negative_dummy_bounds_and_original_rectangles(tmp_path):
    analysis, sections = analyze(tmp_path, "a(-1:1,:)=b(2:4,:)")
    assert sections.available, sections.reason
    output, input_ = sections.resources
    assert output.resource == "argument::a"
    assert output.lower_bounds == (-2, 1)
    assert output.reads == ()
    assert output.writes == output.overwrites
    assert type(output.writes[0].node).__name__ == "Part_Ref"
    assert input_.resource == "argument::b"
    assert input_.lower_bounds == (1, 1)
    assert input_.writes == ()
    public = analysis.report("original::touch")["procedures"][0]["native_sections"]
    assert public == sections.public()
    assert public["resources"][0]["logical_lower_bounds"] == [-2, 1]
    assert public["resources"][0]["writes"][0]["axes"][0]["lower"] == {"kind": "literal", "value": -1}
    assert json.loads(json.dumps(public)) == public
    assert analysis.native_sections("original::touch") is sections


def test_opposite_native_faces_stay_separate_and_read_modify_write_requires_reads(tmp_path):
    _, sections = analyze(tmp_path, "a(-2,:)=a(-2,:)+1\na(2,:)=7")
    assert sections.available, sections.reason
    resource, = sections.resources
    assert len(resource.reads) == 1
    assert len(resource.writes) == 2
    assert resource.writes == resource.overwrites
    assert [rectangle.axes[0].lower.value for rectangle in resource.writes] == [-2, 2]
    assert all(rectangle.axes[0].point for rectangle in resource.writes)


def test_full_arrays_and_full_axes_are_rectangles_without_payload_descriptor_reads(tmp_path):
    _, sections = analyze(tmp_path, "a=3*b\na(:,:)=4")
    assert sections.available, sections.reason
    output, input_ = sections.resources
    assert len(output.writes) == 1
    assert type(output.writes[0].node).__name__ == "Name"
    assert all(axis.lower is None and axis.upper is None for axis in output.writes[0].axes)
    assert len(input_.reads) == 1


def test_same_array_literal_dimension_inquiries_are_typed_bounds(tmp_path):
    _, sections = analyze(tmp_path, "a(lbound(a,1):ubound(a,1),:)=5")
    assert sections.available, sections.reason
    resource, = sections.resources
    axis = resource.writes[0].axes[0]
    assert axis.lower.public() == {"kind": "lbound", "dimension": 1}
    assert axis.upper.public() == {"kind": "ubound", "dimension": 1}
    assert resource.reads == ()


@pytest.mark.parametrize(("body", "reason"), [
    ("a(n,:)=7", "default INTEGER literals"),
    ("a(1:n,:)=7", "default INTEGER literals"),
    ("a(-2:2:2,:)=7", "unit stride"),
    ("a([1,2],:)=7", "default INTEGER literals"),
    ("a(-2:size(a,1),:)=7", "SIZE bounds require declared lower bound one"),
    ("a(lbound(b,1):ubound(b,1),:)=7", "own array and dimension"),
    ("a(lbound(a,2):ubound(a,2),:)=7", "own array and dimension"),
    ("a(2147483648,:)=7", "literal is out of range"),
    ("a(1:size(a,1)-1,:)=7", "default INTEGER literals"),
    ("if(n>0) a(-2,:)=7", "straight-line assignment-only"),
    ("do i=1,n\na(i,:)=7\nenddo", "straight-line assignment-only"),
    ("call other(a)", "straight-line assignment-only"),
])
def test_unknown_bounds_and_control_keep_conservative_effects(tmp_path, body, reason):
    _, sections = analyze(tmp_path, body)
    assert not sections.available
    assert reason in sections.reason
    assert sections.resources == ()


def test_native_section_union_has_a_bounded_rectangle_budget(tmp_path):
    body = "\n".join(f"a({index},:)=1" for index in range(33))
    _, sections = analyze(tmp_path, body)
    assert not sections.available
    assert "rectangle budget exceeded" in sections.reason


def test_native_out_reads_and_dynamic_lower_bounds_keep_original_position_boundaries(tmp_path):
    _, sections = analyze(tmp_path, "a=4\na=a+1", declaration="real(8),intent(out)::a(-2:,:)")
    assert not sections.available
    assert "INTENT(OUT) reads" in sections.reason
    _, dynamic = analyze(tmp_path, "a(:,:)=7", declaration="real(8),intent(inout)::a(n:,:)")
    assert not dynamic.available
    assert "default INTEGER literals" in dynamic.reason


@pytest.mark.parametrize("body", [
    "i=size(b(int(a(-2,1)):,:))",
    "i=size(dim=int(a(-2,1)),array=b)",
    "i=kind(a(-2,1))",
])
def test_descriptor_expression_and_keyword_arguments_cannot_hide_payload_reads(tmp_path, body):
    _, sections = analyze(tmp_path, body)
    assert not sections.available
    assert "positional whole-variable arguments" in sections.reason
    assert sections.resources == ()


def test_descriptor_dimensions_retain_payload_reads_and_unknown_storage_is_rejected(tmp_path):
    _, sections = analyze(tmp_path, "i=size(b,int(a(-2,1)))")
    assert sections.available, sections.reason
    resource, = sections.resources
    assert resource.resource == "argument::a"
    assert len(resource.reads) == 1
    assert all(axis.point for axis in resource.reads[0].axes)
    _, unknown = analyze(tmp_path, "i=size(missing)")
    assert not unknown.available
    assert "storage is unresolved" in unknown.reason


def test_native_specification_payload_bounds_are_not_omitted_from_refinement(tmp_path):
    _, sections = analyze(tmp_path, "real(8)::scratch(int(a(-2,1)))\ni=size(scratch)")
    assert not sections.available
    assert "default INTEGER literals" in sections.reason
    assert sections.resources == ()


def test_explicit_shape_dummy_does_not_claim_the_whole_larger_actual_storage(tmp_path):
    _, sections = analyze(tmp_path, "a=7", declaration="real(8),intent(inout)::a(8,3)")
    assert not sections.available
    assert "assumed-shape array dummies" in sections.reason
    assert sections.resources == ()


@pytest.mark.parametrize("directive", ["task", "target nowait", "target"])
def test_native_directives_require_completion_proof_even_before_first_assignment(tmp_path, directive):
    end = "target" if directive.startswith("target") else directive
    _, sections = analyze(tmp_path, f"!$omp {directive}\na=7\n!$omp end {end}")
    assert not sections.available
    assert "OpenMP participation and completion" in sections.reason
    assert sections.resources == ()


def test_hidden_module_arrays_keep_their_own_declared_negative_bounds(tmp_path):
    _, sections = analyze(tmp_path, "hidden(-1:1,:)=b(1:3,:)", extra="real(8)::hidden(-2:2,3)")
    assert sections.available, sections.reason
    hidden = next(resource for resource in sections.resources if resource.resource == "original::hidden")
    assert hidden.lower_bounds == (-2, 1)


@pytest.mark.parametrize("imported", [False, True])
def test_module_threadprivate_storage_cannot_claim_shared_native_sections(tmp_path, imported):
    extra = "real(8)::hidden(-2:2,3)\n!$omp threadprivate(hidden)"
    if not imported:
        _, sections = analyze(tmp_path, "hidden(-1:1,:)=b(1:3,:)", extra=extra)
    else:
        path = tmp_path / "native.f90"
        path.write_text("module fields\n" + extra + "\nend module\nmodule original\n"
                        "use fields,only:hidden\ncontains\nsubroutine touch(b)\n"
                        "real(8),intent(in)::b(:,:)\nhidden(-1:1,:)=b(1:3,:)\n"
                        "end subroutine\nend module\n")
        sections = SourceEffects([path]).native_sections("original::touch")
    assert not sections.available
    assert "module OpenMP ownership" in sections.reason
    assert sections.resources == ()


@pytest.mark.parametrize("imported", [False, True])
def test_descriptor_inquiry_shadowing_is_an_explained_boundary(tmp_path, imported):
    function = """integer function size(value,dim)
real(8),intent(in)::value(:,:)
integer,intent(in)::dim
size=3
end function
"""
    if imported:
        prefix = "module helpers\ncontains\n" + function + "end module\nmodule original\nuse helpers,only:size\n"
        suffix = ""
    else:
        prefix, suffix = "module original\n", function
    source = tmp_path / "shadow.f90"
    source.write_text(prefix + "implicit none\ncontains\nsubroutine touch(a)\nreal(8),intent(inout)::a(:,:)\n"
                      "a(1:size(a,1),:)=7\nend subroutine\n" + suffix + "end module\n")
    analysis = SourceEffects([source])
    sections = analysis.native_sections("original::touch")
    assert not sections.available
    assert "checked same-array" in sections.reason
    assert not analysis.summarize("original::touch")["complete"]
