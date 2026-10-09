"""Reached source children share ownership without cloning persistent state."""

import json
import re

import pytest

from compiler.tests.test_scoped_planning_entries import calibration
from compiler.tests.test_source_scopes import FACT, PROGRAM, generate


def variants(manifest):
    return [variant for procedure in manifest["implementation_variants"]["procedures"]
            for variant in procedure["variants"]]


def source():
    return PROGRAM.replace("subroutine step(a,b,out,n)", """subroutine child(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),out(:)
integer,intent(inout)::n
logical::choose
call producer(a,b,n)
choose=n>2
if(choose) then
b(1)=b(1)+5.d0
else
call transform(b)
endif
n=n-1
call consumer(a,b,out,n)
end subroutine
subroutine step(a,b,out,n)""").replace(
        "integer,intent(in)::n\ncall producer(a,b,n)\ncall transform(b)\ncall consumer(a,b,out,n)",
        "integer,intent(inout)::n\ncall child(a,b,out,n)")


def test_mixed_child_borrows_context_and_plans_at_original_points(tmp_path):
    text = source()
    original, output, manifest = generate(tmp_path, text, facts={
        "schema_version": 1, "participation": "serial",
        "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}})
    assert original.read_text() == text
    scope, = manifest["scopes"]
    child, = manifest["borrowed_source_coordinators"]
    assert child["procedure"] == "original::child"
    assert child["role"] == "borrowed_reached_coordinator"
    assert scope["ownership"]["close_count"] == 1
    delegated, = scope["structured_tree"]["nodes"]
    assert delegated["kind"] == "reached_source_coordinator"
    worker = next(item["name"] for item in variants(manifest)
                  if item["procedure"] == "original::child" and item["role"] == "call_worker")
    generated = (output / manifest["sources"][str(original)]["replacement"]).read_text()
    body = generated[generated.index("subroutine " + worker):generated.index("end subroutine " + worker)]
    assert "fort_scope_create(" not in body and "fort_scope_close(" not in body
    assert "FORT_SCOPE_PLAN_CONTINUE" in body
    assert body.index("choose =") < body.index("fort_branch =")
    assert "if (fort_mode == 0_c_int) then" in body


def test_repeated_mixed_child_reuses_one_mode_bearing_worker(tmp_path):
    text = source().replace("call child(a,b,out,n)\nend subroutine", "call child(a,b,out,n)\ncall child(a,b,out,n)\nend subroutine")
    _original, _output, manifest = generate(tmp_path, text, facts={
        "schema_version": 1, "participation": "serial",
        "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}})
    child, = manifest["borrowed_source_coordinators"]
    assert child["procedure"] == "original::child"
    assert len([item for item in variants(manifest)
                if item["procedure"] == "original::child" and item["role"] == "call_worker"]) == 1


def test_calibrated_auto_child_imports_its_own_public_selector(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(calibration()))
    original, output, manifest = generate(tmp_path, source(), mode="auto", profile=profile, facts={
        "schema_version": 1, "participation": "serial",
        "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}})
    assert manifest["scopes"][0]["estimate_available"]
    worker = next(item["name"] for item in variants(manifest)
                  if item["procedure"] == "original::child" and item["role"] == "call_worker")
    generated = (output / manifest["sources"][str(original)]["replacement"]).read_text()
    body = generated[generated.index("subroutine " + worker):generated.index("end subroutine " + worker)]
    alias = re.search(r"use \w+, only: (fort_choose_\w+) => \w+", body).group(1)
    assert f"fort_status = {alias}(fort_context, fort_decision)" in body
    assert "fort_status = fort_choose(" not in body


def test_child_original_dummy_bounds_are_captured_before_mutable_controls(tmp_path):
    text = source().replace("real(8),intent(inout)::b(:),out(:)",
                            "real(8),intent(inout)::b(-n:),out(:)").replace(
        "n=n-1\ncall consumer", "n=n-1\nb(-n)=b(-n)+7.d0\ncall consumer")
    original, output, manifest = generate(tmp_path, text, facts={
        "schema_version": 1, "participation": "serial",
        "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}})
    worker = next(item["name"] for item in variants(manifest)
                  if item["procedure"] == "original::child" and item["role"] == "call_worker")
    generated = (output / manifest["sources"][str(original)]["replacement"]).read_text()
    body = generated[generated.index("subroutine " + worker):generated.index("end subroutine " + worker)]
    assert body.index("fort_original_lower_1 = lbound(b, kind=c_int64_t)") < body.index("n = n - 1")
    assert "int(fort_original_lower_1(1), c_int64_t)" in body
    assert "original_dummy_lower_bounds" in str(manifest["borrowed_source_coordinators"])


def test_saved_child_storage_is_not_duplicated(tmp_path):
    text = source().replace("logical::choose", "logical::choose\ninteger,save::calls=0\ncalls=calls+1")
    original, output, manifest = generate(tmp_path, text, facts={
        "schema_version": 1, "participation": "serial",
        "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}})
    assert original.read_text() == text
    assert not manifest["borrowed_source_coordinators"]
    assert not manifest["source_edits"]
    assert any("unsaved storage" in item["reason"] for item in manifest["source_coordinator_boundaries"]
               if item["procedure"] == "original::child")


def test_duplicate_readonly_actuals_keep_original_native_entry(tmp_path):
    child = """subroutine child(a,c,out,n)
real(8),intent(in)::a(:),c(:)
real(8),intent(out)::out(:)
integer,intent(in)::n
logical::choose
choose=sum(a)>sum(c)
call consumer(a,c,out,n)
end subroutine
"""
    text = PROGRAM.replace("subroutine step(a,b,out,n)", child + "subroutine step(a,b,out,n)").replace(
        "call producer(a,b,n)\ncall transform(b)\ncall consumer(a,b,out,n)\nend subroutine",
        "call child(a,a,out,n)\nend subroutine")
    original, _output, manifest = generate(tmp_path, text)
    assert original.read_text() == text
    assert not manifest["source_edits"]
    assert any("merged reached native effects" in item["reason"] for item in manifest["boundaries"])


@pytest.mark.parametrize("target", [False, True])
def test_original_hidden_component_writes_require_target_alias_proof(tmp_path, target):
    declarations = """type storage
real(8)::values(20)
end type
type(storage)""" + (",target" if target else "") + "::state\n"
    touch = """subroutine touch(a)
real(8),intent(in)::a(:)
state%values(1)=a(1)
end subroutine
"""
    text = source().replace("implicit none\ncontains", "implicit none\n" + declarations + "contains").replace(
        "subroutine child(a,b,out,n)", touch + "subroutine child(a,b,out,n)").replace(
        "choose=n>2\nif(choose)", "choose=n>2\ncall touch(a)\nif(choose)")
    original, _output, manifest = generate(tmp_path, text, facts={
        "schema_version": 1, "participation": "serial",
        "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT,
                     "original::state%values": FACT}})
    assert original.read_text() == text
    if target:
        assert manifest["scope_count"] == 1
        assert "original::state%values" in manifest["scopes"][0]["ownership"]["retained_resources"]
    else:
        assert not manifest["source_edits"]
        assert any("original TARGET capture" in item["reason"] for item in manifest["boundaries"])
