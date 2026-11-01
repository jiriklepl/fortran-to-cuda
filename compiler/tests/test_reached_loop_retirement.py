"""An empty original native DO does not retire its caller's coherent domain."""

import json
from copy import deepcopy
from dataclasses import fields, replace

import pytest

from compiler.ir import CompilationError
from compiler.scopes.lexical import LexicalBoundaryPatch
from compiler.tests.test_hierarchical_child_regions import SOURCE, fixture


def source(header="do i=1,size(points)", *, body=None, suffix="", extra=""):
    body = body or "right(points(i)%xi+1)=sample(points(i),right)"
    declarations = """type point_record
integer::xi
real(8),allocatable::unrelated(:)
end type
type(point_record),allocatable::points(:)
integer::last_i,header_calls=0
"""
    function = """real(8) function sample(point,values)
type(point_record),intent(in)::point
real(8),intent(inout)::values(:)
sample=values(point%xi)+2.0_8
end function
"""
    return (
        SOURCE.replace("implicit none\n", "implicit none\n" + declarations, 1)
        .replace("observed=0.0_8\ndo i=1,n\nobserved=observed+right(lbound(right,1)+i-1)\nenddo\n", "")
        .replace("if(change)call relocate(left,right)", header + "\n" + body + "\nenddo\nlast_i=i\n" + suffix)
        .replace("end module", function + extra + "end module")
    )


def emit(tmp_path, text=None):
    path, builder = fixture(tmp_path, source() if text is None else text)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt", report.get("boundaries"))
    outputs = dict(outputs)
    owner = builder._conditional_return_draft.validate().owner
    (middle,) = [member for member in owner.members.values() if member.routine.qualified == "hierarchical::middle"]
    patches = [
        patch
        for _first, _last, patch in middle.patches
        if isinstance(patch, LexicalBoundaryPatch) and "sample(points(i),right)" in patch.original_source
    ]
    (patch,) = patches
    return path, builder, middle, patch, outputs, report


def with_entry(patch, entry):
    # Validate a prospective patch before ownership is sealed. A published
    # patch itself already rejects replacement through the immutable graph.
    values = {field.name: getattr(patch, field.name) for field in fields(LexicalBoundaryPatch)}
    return LexicalBoundaryPatch(**{**values, "counted_entry": entry})


@pytest.mark.parametrize("header", ["do i=1,size(points)", "do i=1,n", "do i=size(points),1,-1", "do i=2,1"])
def test_original_header_body_and_final_iterator_survive_first_entry_retirement(tmp_path, header):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    path, builder, middle, patch, outputs, report = emit(tmp_path, source(header))
    assert patch.counted_entry is not None
    entry = patch.counted_entry.authority
    assert entry["empty_loop_preserves_domain"] is True
    assert entry["payload_access_authorized"] is entry["motion_authorized"] is False
    rendered = patch.render(middle)
    assert rendered.count(header + "\n") == 1
    assert rendered.index(header + "\n") < rendered.index("fort_scope_resident_close_v1(")
    assert rendered.count(entry["flag"] + " = .true.") == 1
    assert rendered.count(entry["flag"] + " = .false.") == 1
    assert "right(points(i)%xi+1)=sample(points(i),right)\nenddo\nend block\n" in rendered
    assert patch.validate(middle, require_source=True) is patch
    validate_checked_execution_plans(report)
    validate_checked_execution_plans(json.loads(outputs["scope-manifest.json"]))
    assert "last_i=i" in outputs[report["sources"][str(path)]["replacement"]]
    assert builder.finish()


@pytest.mark.parametrize(
    ("header", "extra"),
    [
        (
            "do i=1,header_value()",
            """integer function header_value()
header_calls=header_calls+1
header_value=size(points)
end function
""",
        ),
        ("do i=1,int(right(2))", ""),
        ("do i=1,max(0,n)", ""),
    ],
)
def test_unsafe_or_unproved_header_keeps_original_eager_retirement(tmp_path, header, extra):
    _path, _builder, middle, patch, _outputs, _report = emit(tmp_path, source(header, extra=extra))
    assert patch.counted_entry is None
    rendered = patch.render(middle)
    assert rendered.index("fort_scope_resident_close_v1(") < rendered.index(header)
    assert rendered.count(header) == 1


@pytest.mark.parametrize("mutation", ["summary", "context", "flag", "source_span", "body", "header", "empty"])
def test_retained_entry_record_cannot_replace_original_control_authority(tmp_path, mutation):
    _path, _builder, middle, patch, _outputs, _report = emit(tmp_path)
    entry = patch.counted_entry
    record = deepcopy(entry.authority)
    if mutation == "summary":
        record["header_summary_identity"] = "0" * 64
    elif mutation == "context":
        record["context"] = "fort_wrong_context"
    elif mutation == "flag":
        record["flag"] = "fort_wrong_flag"
    elif mutation == "source_span":
        record["source_span"][1] += 1
    elif mutation == "body":
        record["body_source_sha256"] = "0" * 64
    elif mutation == "header":
        record["header_source_sha256"] = "0" * 64
    else:
        record["empty_loop_preserves_domain"] = False
    changed = with_entry(patch, replace(entry, authority=record))
    with pytest.raises(CompilationError, match="exact original header and body"):
        changed.validate(middle, require_source=True)


def test_foreign_body_occurrences_cannot_replace_retained_original_nodes(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    second = tmp_path / "second"
    second.mkdir()
    _path, _builder, middle, patch, _outputs, _report = emit(first)
    _path2, _builder2, _middle2, foreign, _outputs2, _report2 = emit(second)
    changed = with_entry(patch, replace(patch.counted_entry, nodes=foreign.counted_entry.nodes))
    with pytest.raises(CompilationError, match="exact original header and body"):
        changed.validate(middle, require_source=True)


def test_published_entry_is_already_immutable(tmp_path):
    _path, _builder, _middle, patch, _outputs, _report = emit(tmp_path)
    with pytest.raises(CompilationError, match="immutable"):
        patch.counted_entry = None


def test_named_cycle_and_exit_remain_inside_unchanged_original_body(tmp_path):
    body = "if(i==1)cycle visiting\nright(points(i)%xi+1)=sample(points(i),right)\nif(change)exit visiting"
    text = source("visiting: do i=1,size(points)", body=body).replace("if(change)exit visiting\nenddo", "if(change)exit visiting\nenddo visiting")
    _path, _builder, middle, patch, _outputs, _report = emit(tmp_path, text)
    assert patch.counted_entry is not None
    rendered = patch.render(middle)
    assert body + "\nenddo visiting\n" in rendered
    assert rendered.index("fort_scope_resident_close_v1(") < rendered.index("if(i==1)cycle visiting")


def test_shared_header_body_source_line_keeps_conservative_original_path(tmp_path):
    text = source().replace("do i=1,size(points)\nright(points(i)%xi+1)=sample(points(i),right)\nenddo",
                            "do i=1,size(points);right(points(i)%xi+1)=sample(points(i),right);enddo")
    _path, builder = fixture(tmp_path, text)
    outputs, report = builder.run()
    # This source is already outside the checked hierarchical edit contract.
    # The original pipeline may still compile independent producer work.
    assert not hasattr(builder, "_conditional_return_draft")
    assert "original_counted_body_retirement" not in json.dumps(report)
    assert any("do i=1,size(points);right(points(i)%xi+1)=sample(points(i),right);enddo" in value
               for name, value in outputs.items() if name.endswith(".f90"))


def test_changed_original_header_rejects_publication(tmp_path):
    path, builder, _middle, _patch, _outputs, _report = emit(tmp_path)
    path.write_text(path.read_text().replace("do i=1,size(points)", "do i=2,size(points)"))
    with pytest.raises(CompilationError):
        builder.finish()


@pytest.mark.parametrize("part", ["header", "body", "once", "context"])
def test_strict_transport_checks_actual_generated_first_entry_cut(tmp_path, part):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    _path, _builder, _middle, patch, outputs, _report = emit(tmp_path)
    manifest = json.loads(outputs["scope-manifest.json"])
    entry = patch.counted_entry.authority
    (edit,) = [row for row in manifest["source_edits"] if [row["first_line"], row["last_line"]] == entry["source_span"]]
    if part == "header":
        edit["replacement"] = edit["replacement"].replace("do i=1,size(points)", "do i=2,size(points)")
    elif part == "body":
        edit["replacement"] = edit["replacement"].replace("right(points(i)%xi+1)=", "right(points(i)%xi+2)=")
    elif part == "once":
        edit["replacement"] = edit["replacement"].replace(entry["flag"] + " = .false.", entry["flag"] + " = .true.")
    else:
        edit["replacement"] = edit["replacement"].replace(
            "fort_scope_resident_close_v1(" + entry["context"] + ")", "fort_scope_resident_close_v1(fort_wrong_context)"
        )
    with pytest.raises(ValueError, match="reached loop retirement"):
        validate_checked_execution_plans(manifest)


def test_empty_loop_cut_cannot_authorize_an_unconditional_closed_suffix(tmp_path):
    from compiler.scopes.lexical import LexicalClosedSuffix

    _path, _builder, middle, patch, _outputs, _report = emit(tmp_path)
    assert patch.counted_entry is not None
    suffix = LexicalClosedSuffix((), (), patch, "")
    with pytest.raises(CompilationError, match="unconditional original close"):
        suffix.validate(middle)
