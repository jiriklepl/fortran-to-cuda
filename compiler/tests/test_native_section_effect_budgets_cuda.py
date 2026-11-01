"""Separate native read/write lists preserve untouched resident GPU values."""
from __future__ import annotations

import json

import pytest

from compiler.tests import test_native_section_scope_runtime as base


@pytest.fixture(scope="module")
def distinct_native_effects(tmp_path_factory):
    program = base.PROGRAM.replace(
        "module original\nimplicit none\ncontains",
        "module original\nimplicit none\nlogical::touch=.false.\ncontains",
        1,
    )
    old = """subroutine native_faces(b)
real(8),contiguous,intent(inout)::b(-2:)
b(lbound(b,1))=b(lbound(b,1))+7.d0+real(lbound(b,1),8)
b(ubound(b,1))=b(ubound(b,1))+11.d0+real(ubound(b,1),8)
end subroutine"""
    assignments = "\n".join(
        f"b({i - 2})=b({i + 62})+7.d0+real(lbound(b,1),8)" for i in range(20)
    )
    replacement = (
        "subroutine native_faces(b)\nreal(8),contiguous,intent(inout)::b(-2:)\n"
        "if(touch) then\n" + assignments + "\nendif\nend subroutine"
    )
    assert program.count(old) == 1
    program = program.replace(old, replacement)
    driver = base.DRIVER.replace(" n=4+8*shape", " n=128+32*shape")
    driver = driver.replace("  select case(which)", "  touch=repeat==1\n  select case(which)")
    expected = """     if(i==1) expected_b=expected_b+5.d0
     if(i==n) expected_b=expected_b+real(n+8,8)"""
    assert driver.count(expected) == 1
    driver = driver.replace(expected, "     if(touch.and.i<=20) expected_b=2*a(actual+64)+real(i+64,8)+5.d0")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "PROGRAM", program)
        patch.setattr(base, "DRIVER", driver)
        patch.setattr(base, "CASES", (("faces", "step_faces", 4),))
        return base.compiled_native_sections.__wrapped__(tmp_path_factory)


@pytest.mark.cuda
def test_twenty_read_and_twenty_write_sections_preserve_interiors(distinct_native_effects):
    lines, launches = base._successful(distinct_native_effects, "faces")
    output = distinct_native_effects["faces"][1]
    manifest = json.loads((output / "scope-manifest.json").read_text())
    effect = next(row for row in manifest["native_effects"]["procedures"]
                  if row["procedure"] == "original::native_faces")
    physical = effect["native_sections"]
    assert physical["available"]
    resource, = physical["resources"]
    assert len(resource["reads"]) == len(resource["writes"]) == 20
    assert resource["overwrites"] == []
    # A possible write requires preserving its old value on the false arm.
    # The two disjoint 20-element physical unions fit the unchanged runtime
    # fragmentation limit, despite having 40 distinct source rectangles.
    # Both widths and branch outcomes compare every output/intermediate/halo.
    for index in range(4):
        between = lines[launches[2 * index] + 1:launches[2 * index + 1]]
        assert base._trace_bytes(between, "download") == 320
        assert base._trace_bytes(between, "upload") == 160
