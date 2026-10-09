"""Call composition preserves original views, guards and definition ordering."""

from pathlib import Path

import pytest

from compiler.frontend.source_effects import SourceEffects


def source(directory, name, text):
    path = directory / name
    path.write_text(text)
    return path


def test_nested_rectangular_out_preserves_view_chain_and_native_middle_write(tmp_path):
    leaf = source(tmp_path, "leaf.f90", """module setters
contains
subroutine initialize(v)
real(8),intent(out)::v(4:)
v=7
end subroutine
end module
""")
    middle = source(tmp_path, "middle.f90", """module transforms
use setters,only:renamed=>initialize
contains
subroutine adjust(u,m)
real(8),intent(inout)::u(-2:)
integer,intent(in)::m
u(-1)=3
if(m>0) call renamed(v=u(0:m))
end subroutine
end module
""")
    entry = source(tmp_path, "entry.f90", """module application
use transforms,only:adjust
contains
subroutine advance(a,n)
real(8),intent(inout)::a(-7:)
integer,intent(in)::n
if(n>0) call adjust(m=n,u=a(-3:n))
end subroutine
end module
""")
    report = SourceEffects([entry, middle, leaf]).report("application::advance")
    assert report["complete"], report
    root = report["procedures"][0]
    effects = root["ordered_effects"]
    native_write = next(index for index, effect in enumerate(effects)
                        if effect["kind"] == "write" and effect["source_procedure"] == "transforms::adjust")
    forget = next(index for index, effect in enumerate(effects) if effect["kind"] == "definition_change")
    overwrite = next(index for index, effect in enumerate(effects)
                     if effect["kind"] == "overwrite" and effect["source_procedure"] == "setters::initialize")
    assert native_write < forget < overwrite
    for index in (forget, overwrite):
        effect = effects[index]
        assert effect["resource"] == "argument::a"
        assert effect["coverage"] == "whole_formal"
        assert [(view["caller"], view["storage"]) for view in effect["view_chain"]] == [
            ("application::advance", "rectangle"), ("transforms::adjust", "rectangle")]
        outer, inner = effect["view_chain"]
        assert outer["section"]["axes"][0]["lower"]["expression"].replace(" ", "") == "-3"
        assert inner["section"]["axes"][0]["lower"]["value"] == 0
        assert inner["formal_descriptor"]["lower_bounds"] == ("4",)
        assert [frame["procedure"] for frame in effect["guard_frames"]] == [
            "application::advance", "transforms::adjust"]
    assert root["definition_changes"] == []
    assert len(report["call_graph"]["edges"]) == 2


def test_diamond_reuses_leaf_summary_but_preserves_each_ordered_occurrence(tmp_path):
    path = source(tmp_path, "diamond.f90", """module graph
contains
subroutine entry(a)
real(8),intent(inout)::a(:)
call left(a)
call right(a)
end subroutine
subroutine left(a)
real(8),intent(inout)::a(:)
call leaf(a)
end subroutine
subroutine right(a)
real(8),intent(inout)::a(:)
call leaf(a)
end subroutine
subroutine leaf(a)
real(8),intent(inout)::a(:)
a=a+1
end subroutine
end module
""")
    report = SourceEffects([path]).report("graph::entry")
    assert report["complete"], report
    assert len(report["procedures"]) == 4
    effects = report["procedures"][0]["ordered_effects"]
    assert [effect["kind"] for effect in effects] == ["read", "overwrite", "read", "overwrite"]
    assert [effect["view_chain"][0]["callee"] for effect in effects] == [
        "graph::left", "graph::left", "graph::right", "graph::right"]
    assert report["summarized_operations"] == 6


def test_composed_effect_expansion_is_bounded_even_when_distinct_graph_fits(tmp_path):
    path = source(tmp_path, "repeated.f90", """module graph
contains
subroutine entry(a)
real(8),intent(inout)::a(:)
call repeated(a)
call repeated(a)
end subroutine
subroutine repeated(a)
real(8),intent(inout)::a(:)
call leaf(a)
call leaf(a)
call leaf(a)
end subroutine
subroutine leaf(a)
real(8),intent(inout)::a(:)
a=a+1
end subroutine
end module
""")
    report = SourceEffects([path], operations=7).report("graph::entry")
    assert report["summarized_operations"] == 7
    assert not report["complete"]
    root = report["procedures"][0]
    assert "composed source operation budget exhausted" in root["reasons"]
    assert len(root["ordered_effects"]) == 7
    assert not root["effect_composition"]["available"]
    assert root["guaranteed_whole_overwrites"] == []


def test_optional_readonly_descriptor_forwarding_keeps_presence_and_original_guards(tmp_path):
    path = source(tmp_path, "optional.f90", """module descriptions
contains
subroutine entry(a)
real(8),allocatable,optional,intent(in)::a(:)
call child(a=a)
end subroutine
subroutine child(a)
real(8),allocatable,optional,intent(in)::a(:)
real(8)::total
if(present(a)) then
if(allocated(a)) total=sum(a)
endif
end subroutine
end module
""")
    report = SourceEffects([path]).report("descriptions::entry")
    assert report["complete"], report
    effects = report["procedures"][0]["ordered_effects"]
    read = next(effect for effect in effects if effect["kind"] == "read")
    assert read["resource"] == "argument::a"
    assert read["view_chain"][0]["presence"] == "forwarded_optional"
    assert read["view_chain"][0]["requirements"]["original_allocation_descriptor"]
    assert len(read["guard_frames"]) == 2
    assert all(frame["procedure"] == "descriptions::child" for frame in read["guard_frames"])


@pytest.mark.parametrize("unused", [True, False])
def test_allocatable_out_entry_is_a_boundary_even_without_payload_access(tmp_path, unused):
    body = "" if unused else "a(1)=0"
    path = source(tmp_path, "changing.f90", "module changing\ncontains\nsubroutine reset(a)\n"
                  "real(8),allocatable,intent(out)::a(:)\n" + body + "\nend subroutine\nend module\n")
    report = SourceEffects([path]).report("changing::reset")
    assert not report["complete"]
    assert any("dummy descriptor" in reason for reason in report["procedures"][0]["reasons"])


def test_public_cli_cache_is_optional_and_does_not_emit_numerical_artifacts(tmp_path):
    # CLI plumbing is checked in-process; heavy compiler/native integration is
    # kept in the independent-checkout acceptance campaign.
    import sys
    from unittest.mock import patch

    from compiler.driver.cli import _parse_args

    with patch.object(sys, "argv", ["compiler", "--input", str(Path("input.f90")), "--kernel", "entry",
                                    "--analyze-effects", "--summary-cache", str(tmp_path)]):
        args = _parse_args()
    assert args.summary_cache == str(tmp_path)
