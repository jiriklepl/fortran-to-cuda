"""Native query and execution hooks describe the same original sections."""

from __future__ import annotations

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import form_source_scopes
from compiler.tests.test_source_scopes import FACT, PROGRAM

PARTIAL = PROGRAM.replace("do i=1,n", "do i=3,6").replace("intent(out)", "intent(inout)")


def generate(tmp_path, *, body="b(3:6)=7.d0", intent="inout", initialized="whole", wrapped=False):
    source = PARTIAL.replace("subroutine transform(b)\nreal(8),intent(inout)::b(:)\nb=3*b",
                             f"subroutine transform(b)\nreal(8),intent({intent})::b(:)\n{body}")
    if wrapped:
        source = source.replace("subroutine step(a,b,out,n)", "subroutine wrapper(a,b,out,n)")
        source = source.replace("end module", """subroutine step(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),out(:)
integer,intent(in)::n
call wrapper(a,b,out,n)
call wrapper(a,b,out,n)
end subroutine
end module""")
    path = tmp_path / "original.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": FACT, "argument::b": {**FACT, "initialized": initialized},
                          "argument::out": {**FACT, "initialized": initialized}}}
    outputs, report = form_source_scopes([path], "step", facts=facts,
                                         options=CompilerOptions(fallback="host", gpu_policy="sections", memory_model="scoped"),
                                         config=OffloadConfig("sections"))
    text = next((value for name, value in outputs.items() if name.startswith("sources/")), source)
    return source, text, report


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("intent", ["inout", "out"])
def test_partial_native_overwrite_has_matching_query_and_execution_sections(tmp_path, intent, wrapped):
    _source, text, report = generate(tmp_path, intent=intent, wrapped=wrapped, initialized="none")
    scope, = report["scopes"]
    assert scope["definition_preflight"]["query_available"]
    assert report["limits"]["native_rectangles_per_resource"] == 32
    assert "whole-resource fallback" in report["limits"]["physical_native_sections"]
    assert "3_c_int64_t - " in text
    assert "6_c_int64_t - " in text
    assert "%write_count" in text
    assert "%overwrite_count" in text
    assert "%access = fort_native_" in text
    assert "fort_scope_host_begin(fort_context," in text
    assert "shared native access preparation failed" in text
    assert "FORT_SCOPE_WRITE_ALL" not in text
    assert "FORT_SCOPE_READ_ALL" not in text
    if intent == "out":
        assert "FORT_SCOPE_PLAN_FORGET" in text
        assert "fort_scope_forget_definition(" in text


def test_direct_owner_query_failure_exits_to_cleanup_and_original_span(tmp_path):
    _source, text, report = generate(tmp_path)
    owner = text.split("subroutine " + report["scopes"][0]["owner"] + "(", 1)[1].split("end subroutine", 1)[0]
    recording = owner.split("fort_record_", 1)[1].split("fort_scope_plan_validate(", 1)[0]
    assert "FORT_SCOPE_BOUNDARY" in recording
    assert "exit fort_record_" in recording
    assert not any(line.strip() == "return" for line in recording.splitlines())
    fallback = owner.split("fort_scope_plan_validate(", 1)[1].split("call fort_scope_clone_", 1)[0]
    assert fallback.index("fort_scope_close(") < fallback.index("call producer(")
    assert "call transform(" in fallback
    assert "call consumer(" in fallback


def test_native_rmw_and_unknown_ranges_keep_distinct_safe_effects(tmp_path):
    _source, text, _report = generate(tmp_path, body="b(3:6)=b(3:6)+3.d0", initialized="none")
    assert "%read_count" in text
    assert "%write_count" in text
    assert "%overwrite_count" in text
    _source, text, report = generate(tmp_path, body="b(2:size(b)-1)=7.d0", initialized="whole")
    assert report["scope_count"] == 1
    assert "FORT_SCOPE_WRITE_ALL" in text
    assert "fort_scope_plan_validate(" in text
    assert "fort_native_" not in text


def test_partial_native_out_with_unproved_bounds_remains_an_original_boundary(tmp_path):
    _source, text, report = generate(tmp_path, body="b(2:size(b)-1)=7.d0", intent="out")
    assert all("original::transform" not in scope["calls"] for scope in report["scopes"])
    assert "call transform(b)" in text
    assert any("original-position definition hooks" in boundary["reason"] for boundary in report["boundaries"])


@pytest.mark.parametrize("wrapped", [False, True])
def test_native_openmp_helper_is_an_original_boundary_without_participation_proof(tmp_path, wrapped):
    _source, text, report = generate(tmp_path, body="!$omp task\nb(3:6)=7.d0\n!$omp end task", wrapped=wrapped)
    assert report["scope_count"] == 0
    assert "!$omp task" in text
    assert "call transform(b)" in text
    assert any("native OpenMP participation and completion" in boundary["reason"]
               for boundary in report["boundaries"])


def test_native_openmp_hidden_callee_is_an_original_boundary(tmp_path):
    path = tmp_path / "original.f90"
    source = PARTIAL.replace("b=3*b", "call finish(b)").replace("end module", """subroutine finish(b)
real(8),intent(inout)::b(:)
!$omp task
b=3*b
!$omp end task
end subroutine
end module""")
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}}
    _outputs, report = form_source_scopes([path], "step", facts=facts,
                                         options=CompilerOptions(fallback="host", gpu_policy="sections", memory_model="scoped"),
                                         config=OffloadConfig("sections"))
    assert any("native OpenMP participation and completion" in boundary["reason"]
               for boundary in report["boundaries"])
    assert all("original::finish" not in scope["gpu_leaves"] for scope in report["scopes"])


@pytest.mark.parametrize("directive", ["do", "parallel do"])
def test_serial_native_worksharing_with_proved_completion_keeps_conservative_hooks(tmp_path, directive):
    _source, text, report = generate(tmp_path, body=f"integer::i\n!$omp {directive}\n"
                                    "do i=1,size(b)\nb(i)=3*b(i)\nenddo\n"
                                    f"!$omp end {directive}")
    scope, = report["scopes"]
    assert "original::transform" in scope["calls"]
    assert "original::transform" not in scope["gpu_leaves"]
    assert "FORT_SCOPE_READ_ALL" in text
    assert "FORT_SCOPE_WRITE_ALL" in text
    assert "call transform(" in text
    assert f"!$omp {directive}" in text
