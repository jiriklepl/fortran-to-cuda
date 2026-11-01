"""Conditional native faces preserve dirty GPU interiors across original calls."""
from __future__ import annotations

import json

import pytest

from compiler.tests import test_native_section_scope_runtime as base


@pytest.fixture(scope="module")
def conditional_native_faces(tmp_path_factory):
    program = base.PROGRAM.replace(
        "module original\nimplicit none\ncontains",
        "module original\nimplicit none\ninteger::upper=0\nlogical::touch=.false.\ncontains",
        1,
    )
    old = """subroutine native_faces(b)
real(8),contiguous,intent(inout)::b(-2:)
b(lbound(b,1))=b(lbound(b,1))+7.d0+real(lbound(b,1),8)
b(ubound(b,1))=b(ubound(b,1))+11.d0+real(ubound(b,1),8)
end subroutine"""
    new = """subroutine native_faces(b)
real(8),contiguous,intent(inout)::b(-2:)
integer::last
last=upper
if(touch) then
b(lbound(b,1))=b(lbound(b,1))+7.d0+real(lbound(b,1),8)
b(last)=b(last)+11.d0+real(last,8)
endif
end subroutine"""
    assert program.count(old) == 1
    program = program.replace(old, new)
    driver = base.DRIVER.replace("  select case(which)", "  upper=n-3\n  touch=repeat==1\n  select case(which)")
    driver = driver.replace("if(i==1) expected_b", "if(touch.and.i==1) expected_b")
    driver = driver.replace("if(i==n) expected_b", "if(touch.and.i==n) expected_b")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "PROGRAM", program)
        patch.setattr(base, "DRIVER", driver)
        patch.setattr(base, "CASES", (("faces", "step_faces", 4),))
        return base.compiled_native_sections.__wrapped__(tmp_path_factory)


@pytest.mark.cuda
def test_native_conditional_faces_keep_untouched_values_and_device_interiors(conditional_native_faces):
    lines, launches = base._successful(conditional_native_faces, "faces")
    output = conditional_native_faces["faces"][1]
    manifest = json.loads((output / "scope-manifest.json").read_text())
    effect = next(row for row in manifest["native_effects"]["procedures"]
                  if row["procedure"] == "original::native_faces")
    physical = effect["native_sections"]
    assert physical["available"]
    resource, = physical["resources"]
    assert resource["overwrites"] == []
    # Two widths and true/false branches compare every intermediate/output/halo
    # value with the original native calculation in both traced and plain runs.
    # Even the false arm preserves its possible write faces; interiors stay GPU
    # current, so physical traffic between the producer and consumer is bounded.
    for index in range(4):
        between = lines[launches[2 * index] + 1:launches[2 * index + 1]]
        assert base._trace_bytes(between, "download") == 16
        assert base._trace_bytes(between, "upload") == 16
