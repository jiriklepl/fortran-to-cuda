"""Source rectangular actuals borrow one canonical full-layout allocation."""

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

PROGRAM = """module windows
implicit none
contains
subroutine produce(a,b)
real(8),intent(in)::a(:,:)
real(8),intent(out)::b(:,:)
integer::i,j
do j=1,size(b,2)
do i=1,size(b,1)
b(i,j)=2*a(i,j)+real(i+8*j,8)
enddo
enddo
end subroutine
subroutine inspect(b)
real(8),intent(inout)::b(:,:)
b=3*b
end subroutine
subroutine consume(a,b,out)
real(8),intent(in)::a(:,:),b(:,:)
real(8),intent(out)::out(:,:)
integer::i,j
do j=1,size(out,2)
do i=1,size(out,1)
out(i,j)=a(i,j)+b(i,j)
enddo
enddo
end subroutine
subroutine step(a,b,out,edge)
real(8),intent(in)::a(-2:,-3:)
real(8),intent(inout)::b(-2:,-3:),out(-2:,-3:)
integer,intent(in)::edge
call produce(b=b(-1:ubound(b,1)-1,-2:ubound(b,2)-1),a=a(-1:ubound(a,1)-1,-2:ubound(a,2)-1))
call inspect(b)
call consume(a(-1:ubound(a,1)-edge,-2:ubound(a,2)-1),b(-1:ubound(b,1)-edge,-2:ubound(b,2)-1), &
             out(-1:ubound(out,1)-edge,-2:ubound(out,2)-1))
end subroutine
end module
"""


def build(tmp_path, source=PROGRAM, *, policy="sections"):
    path = tmp_path / "windows.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}}
    compiler = ScopeBuilder([path], "windows::step", facts=facts, options=CompilerOptions(),
                            config=OffloadConfig(policy=policy))
    outputs, report = compiler.run()
    return compiler, outputs, report


@pytest.mark.parametrize("policy", ["sections", "auto"])
def test_direct_rectangles_use_shared_root_views_and_preflight_before_work(tmp_path, policy):
    _, outputs, report = build(tmp_path, policy=policy)
    assert report["scope_count"] == 1, report["boundaries"]
    owner, = report["scopes"]
    views = owner["borrowed_views"]
    assert [item["procedure"] for item in views["calls"]] == ["windows::produce", "windows::consume"]
    assert all(mapping["logical_lower_bounds"] == [1, 1] for call in views["calls"] for mapping in call["mappings"])
    assert len(owner["resources"]) == 3
    assert all(source["path"] in outputs for source in report["build_sources"])
    assert all('#include "common_functions.cuh"' in text for path, text in outputs.items()
               if path.endswith("_views/shared_entry.cu"))
    generated = outputs[next(path for path in outputs if path.startswith("sources/"))]
    if policy == "sections":
        assert generated.index("fort_scope_view_get_v2") < generated.index("fort_status = fort_view_plan_")
        assert "fort_scope_forget_definition(fort_context" not in generated
    assert all(name + "/view_entry.hpp" in outputs for name in {
        path.rsplit("/", 1)[0] for path in outputs if path.endswith("_views/shared_entry.cu")})


def test_view_entry_is_reused_for_multiple_rectangles_without_shape_variants(tmp_path):
    source = PROGRAM.replace("call inspect(b)", "call inspect(b)\ncall produce(a(-1:1,-2:2),b(-1:1,-2:2))")
    _, _, report = build(tmp_path, source)
    assert report["scope_count"] == 1, report["boundaries"]
    procedure = next(item for item in report["implementation_variants"]["procedures"] if item["procedure"] == "windows::produce")
    assert len(procedure["variants"]) == 1
    assert procedure["variants"][0]["interface"] == "root_view_v2"


def test_integer8_section_control_remains_an_explicit_native_boundary(tmp_path):
    source = PROGRAM.replace("integer,intent(in)::edge", "integer(8),intent(in)::edge")
    _, _, report = build(tmp_path, source)
    assert any("checked default INTEGER controls" in item["reason"] for item in report["boundaries"])


def test_native_rectangle_uses_projected_accesses_without_whole_root_coherence(tmp_path):
    source = PROGRAM.replace("call inspect(b)", "call inspect(b(-1:1,-2:2))")
    _, outputs, report = build(tmp_path, source)
    assert report["scope_count"] == 1, report["boundaries"]
    generated = outputs[next(path for path in outputs if path.startswith("sources/"))]
    assert "fort_scope_host_begin" in generated
    assert "_physical_origin" in generated
    if "fort_scope_view_query_" in generated:
        assert "type(fort_scope_plan_binding), target :: fort_bindings(" in generated
    assert "FORT_SCOPE_READ_ALL" not in generated


def test_mutating_bound_control_cannot_hoist_a_future_view(tmp_path):
    source = PROGRAM.replace("integer,intent(in)::edge", "integer,intent(inout)::edge").replace(
        "call inspect(b)", "call inspect(b)\nedge=2")
    _, outputs, report = build(tmp_path, source)
    assert any("per-segment descriptor preflight" in item["reason"] for item in report["boundaries"])
    # A safe earlier span can still be owned; the mutable view is not replayed.
    assert all(scope["last_line"] < PROGRAM.splitlines().index("call inspect(b)") + 4
               for scope in report["scopes"])
    assert all(source["path"] in outputs for source in report["build_sources"])


def nested_program():
    wrappers = """subroutine inner(a,b,out,edge)
real(8),intent(in)::a(:,:)
real(8),intent(inout)::b(:,:),out(:,:)
integer,intent(in)::edge
call produce(a(2:size(a,1)-edge,2:size(a,2)-1),b(2:size(b,1)-edge,2:size(b,2)-1))
call consume(a(2:size(a,1)-edge,2:size(a,2)-1),b(2:size(b,1)-edge,2:size(b,2)-1), &
             out(2:size(out,1)-edge,2:size(out,2)-1))
end subroutine
subroutine outer(a,b,out,edge)
real(8),intent(in)::a(:,:)
real(8),intent(inout)::b(:,:),out(:,:)
integer,intent(in)::edge
call inner(out=out(2:size(out,1)-1,2:size(out,2)-1),edge=edge, &
           b=b(2:size(b,1)-1,2:size(b,2)-1),a=a(2:size(a,1)-1,2:size(a,2)-1))
end subroutine
"""
    prefix, entry = PROGRAM.split("subroutine step(a,b,out,edge)")
    entry = entry[:entry.index("call produce")] + """call outer(a,b,out,edge)
call outer(a,b,out,edge)
end subroutine
end module
"""
    return prefix + wrappers + "subroutine step(a,b,out,edge)" + entry


def test_nested_call_only_wrappers_compose_and_reuse_borrowed_views(tmp_path):
    _, outputs, report = build(tmp_path, nested_program())
    assert report["scope_count"] == 1, report["boundaries"]
    scope, = report["scopes"]
    assert len(scope["resources"]) == 3
    variants = {item["procedure"]: item["variants"] for item in report["implementation_variants"]["procedures"]}
    for name in ("windows::inner", "windows::outer"):
        worker, = variants[name]
        assert worker["interface"] == "source_root_view_v2"
    for name in ("windows::produce", "windows::consume"):
        numerical, = variants[name]
        assert numerical["interface"] == "root_view_v2"
    generated = outputs[next(path for path in outputs if path.startswith("sources/"))]
    assert "_parent_origin" in generated
    assert "fort_scope_forget_definition(" not in generated


def test_nested_rectangular_native_middle_preserves_original_position(tmp_path):
    source = nested_program().replace("call consume(a(2:size(a,1)-edge", "call inspect(b(2:size(b,1)-edge,2:size(b,2)-1))\ncall consume(a(2:size(a,1)-edge")
    _, outputs, report = build(tmp_path, source)
    assert report["scope_count"] == 1, report["boundaries"]
    generated = outputs[next(path for path in outputs if path.startswith("sources/"))]
    assert "fort_scope_host_begin" in generated
    assert "_physical_origin" in generated
    if "fort_scope_view_query_" in generated:
        assert "type(fort_scope_plan_binding), target :: fort_bindings(" in generated
    assert "native rectangular hooks" not in report["scopes"][0]["borrowed_views"]["boundaries"]


def test_native_partial_out_discards_only_its_actual_view(tmp_path):
    source = PROGRAM.replace("intent(inout)::b(:,:)\nb=3*b", "intent(out)::b(:,:)\nb=5.d0").replace(
        "call inspect(b)", "call inspect(b(-1:1,-2:2))")
    _, outputs, report = build(tmp_path, source)
    assert report["scope_count"] == 1, report["boundaries"]
    generated = outputs[next(path for path in outputs if path.startswith("sources/"))]
    assert "fort_scope_plan_forget_sections_v1" in generated
    assert "fort_scope_forget_sections_v1" in generated
    assert "fort_scope_forget_definition(" not in generated
