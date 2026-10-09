"""Source-proven boundary actuals compose rank changes through shared roots."""
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT


SOURCE = """module renamed_planes
contains
subroutine fill(a,b)
real(8),intent(in)::a(:,:)
real(8),intent(out)::b(:,:)
integer::i,j
do j=1,size(b,2)
do i=1,size(b,1)
b(i,j)=2*a(i,j)+real(i+7*j,8)
enddo
enddo
end subroutine
subroutine native_middle(b)
real(8),intent(inout)::b(-11:,4:)
b=3*b
end subroutine
subroutine accumulate(b)
real(8),intent(inout)::b(:,:)
integer::i,j
do j=1,size(b,2)
do i=1,size(b,1)
b(i,j)=b(i,j)+17.d0
enddo
enddo
end subroutine
subroutine relay(a,b,k)
real(8),intent(in)::a(:,:,:)
real(8),intent(inout)::b(:,:,:)
integer,intent(in)::k
call fill(a(2:size(a,1)-1,k,2:size(a,3)-1),b(2:size(b,1)-1,k,2:size(b,3)-1))
call native_middle(b(2:size(b,1)-1,k,2:size(b,3)-1))
call accumulate(b(2:size(b,1)-1,k,2:size(b,3)-1))
end subroutine
subroutine step(a,b,k)
real(8),intent(in)::a(-3:,7:,-5:)
real(8),intent(inout)::b(-3:,7:,-5:)
integer,intent(in)::k
call fill(a(-2:ubound(a,1)-1,k,-4:ubound(a,3)-1),b(-2:ubound(b,1)-1,k,-4:ubound(b,3)-1))
call native_middle(b(-2:ubound(b,1)-1,k,-4:ubound(b,3)-1))
call accumulate(b(-2:ubound(b,1)-1,k,-4:ubound(b,3)-1))
end subroutine
end module
"""


def build(tmp_path, source=SOURCE):
    path = tmp_path / "planes.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: FACT for name in ("a", "b")}}
    return ScopeBuilder([path], "renamed_planes::step", facts=facts, options=CompilerOptions(),
                        config=OffloadConfig("sections")).run()


def test_boundary_planes_use_retained_axes_and_exact_native_hooks(tmp_path):
    outputs, report = build(tmp_path)
    assert report["scope_count"] == 1, report["boundaries"]
    owner, = report["scopes"]
    assert len(owner["resources"]) == 2
    assert owner["borrowed_views"]["abi_version"] == 2
    records = [mapping for call in owner["borrowed_views"]["calls"] for mapping in call["mappings"]]
    assert all(item["actual_rank"] == 3 and item["logical_rank"] == 2 for item in records)
    assert all(item["section"]["retained_axes"] == [0, 2] for item in records)
    generated = outputs[next(path for path in outputs if path.startswith("sources/"))]
    assert "fort_scope_view_get_v2" in generated and "fort_scope_host_begin" in generated
    assert "fort_scope_forget_definition(" not in generated
    entries = [text for path, text in outputs.items() if path.endswith("_views/shared_entry.cu")]
    assert entries and all(".rank != 2" in text for text in entries)
    assert all(".root.rank != 2" not in text for text in entries)


def test_rank_reduction_composes_inside_a_reused_call_only_wrapper(tmp_path):
    prefix, suffix = SOURCE.split("subroutine step(a,b,k)")
    suffix = (suffix[:suffix.index("call fill")] +
              "integer::logical_plane\nlogical_plane=k-6\n"
              "call relay(a,b,logical_plane)\ncall relay(a,b,logical_plane)\nend subroutine\nend module\n")
    outputs, report = build(tmp_path, prefix + "subroutine step(a,b,k)" + suffix)
    assert report["scope_count"] == 1, report["boundaries"]
    variants = {item["procedure"]: item["variants"] for item in report["implementation_variants"]["procedures"]}
    assert len(variants["renamed_planes::relay"]) == 1
    assert variants["renamed_planes::relay"][0]["interface"] == "source_root_view_v2"
    generated = outputs[next(path for path in outputs if path.startswith("sources/"))]
    assert "_parent_axes" in generated and "_parent_origin" in generated


@pytest.mark.parametrize("selector", ["(:,[7,8],:)", "(::2,7,:)", "(:,7,::-1)"])
def test_vector_or_strided_actuals_remain_explicit_boundaries(tmp_path, selector):
    source = SOURCE.replace("a(-2:ubound(a,1)-1,k,-4:ubound(a,3)-1)", "a" + selector)
    _outputs, report = build(tmp_path, source)
    assert any("affine scalar" in boundary["reason"] or "unit stride" in boundary["reason"]
               for boundary in report["boundaries"])
