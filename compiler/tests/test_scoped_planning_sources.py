"""Complete-source planning proves controls immutable before querying them."""
from __future__ import annotations

import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.emission.common.resources import read_scoped_runtime
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import form_source_scopes
from compiler.tests.test_offload_profile import scoped_profile
from compiler.tests.test_source_scopes import ALLOCATABLE_PROGRAM, FACT, PROGRAM, WRAPPER_PROGRAM


def generate(tmp_path, monkeypatch, source=PROGRAM, *, captures=None):
    # Synthetic costs exercise source proof/generation only, never performance.
    monkeypatch.setattr("compiler.emission.cuda.scoped.compiler_identity", lambda profile: {})
    path = tmp_path / "original.f90"
    path.write_text(source)
    profile = scoped_profile()
    profile["scoped"]["runtime_id"] = read_scoped_runtime()[1]["runtime_id"]
    # profile_expression also verifies compiler banner identities.
    profile["toolchain"]["nvcc_version"] = "Cuda compilation tools, release 13.4, V13.4.88"
    profile["toolchain"]["host_cxx_version"] = "g++ (GCC) 14.4.0"
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": captures or {"argument::a": FACT, "argument::b": {**FACT, "initialized": "none"},
                                       "argument::out": {**FACT, "initialized": "none"}}}
    outputs, report = form_source_scopes([path], "step", facts=facts,
                                         options=CompilerOptions(fallback="host", gpu_policy="auto", memory_model="scoped"),
                                         config=OffloadConfig("auto", profile))
    return outputs, report


def test_complete_scope_uses_public_queries_and_protected_native_fallback(tmp_path, monkeypatch):
    outputs, report = generate(tmp_path, monkeypatch)
    assert report["automatic_estimate_available"]
    scope, = report["scopes"]
    assert scope["estimate_available"]
    assert [resource["registration_identity"] for resource in scope["resources"]] == [1, 2, 3]
    assert {resource["allocation_generation"] for resource in scope["resources"]} == {1}
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    assert "fort_scope_plan_reset(fort_context)" in text
    assert "fort_status = fort_choose(fort_context, fort_decision)" in text
    assert "fort_decision%gpu_units == 0" in text
    assert "FORT_SCOPE_PLAN_NATIVE" in text
    assert any("FORT_SCOPE_PLAN_FORGET" in value for name, value in outputs.items() if name.endswith(".cu"))
    query = text.split("subroutine fort_scope_query_", 1)[1].split("end subroutine", 1)[0]
    assert "INTENT(IN)" in query
    assert "INTENT(OUT)" not in query
    assert "fort_scope_forget_definition" not in query
    assert "fort_status = fort_plan(" in query


def test_allocatable_owning_roots_keep_calibrated_queries_behind_caller_guard(tmp_path, monkeypatch):
    outputs, report = generate(tmp_path, monkeypatch, ALLOCATABLE_PROGRAM)
    assert report["automatic_estimate_available"], report["boundaries"]
    scope, = report["scopes"]
    assert scope["estimate_available"]
    assert scope["allocation_preflight"]["position"] == "original caller before owner association"
    edit = next(edit for edit in report["source_edits"] if edit["first_line"] <= edit["last_line"])
    assert edit["replacement"].index("allocated(a)") < edit["replacement"].index("call fort_scope_owner_")
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    owner = text.split("subroutine fort_scope_owner_", 1)[1].split("end subroutine", 1)[0]
    assert owner.index("if (fort_scope_serial_caller() == 0)") < owner.index("is_contiguous(")
    assert owner.index("_view => ") < owner.index("fort_scope_plan_reset(fort_context)")
    assert "fort_status = fort_choose(fort_context, fort_decision)" in owner


def test_query_wrappers_preserve_source_definition_positions(tmp_path, monkeypatch):
    outputs, report = generate(tmp_path, monkeypatch, WRAPPER_PROGRAM)
    assert report["automatic_estimate_available"], report["boundaries"]
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    queries = text.split("subroutine fort_scope_query_")[1:]
    assert len(queries) >= 3
    assert any("FORT_SCOPE_PLAN_FORGET" in query.split("end subroutine", 1)[0] for query in queries)
    assert all("fort_scope_host_begin" not in query.split("end subroutine", 1)[0] for query in queries)


def test_changed_control_scalar_keeps_original_whole_span(tmp_path, monkeypatch):
    adjust = "subroutine adjust(n)\ninteger,intent(inout)::n\nn=n-1\nend subroutine\n"
    source = PROGRAM.replace("end module", adjust + "end module")
    # Replace only owning entry's intent; numerical leaves remain read-only.
    before, step = source.split("subroutine step(", 1)
    source = before + "subroutine step(" + step.replace("integer,intent(in)::n", "integer,intent(inout)::n", 1)
    source = source.replace("call transform(b)", "call adjust(n)")
    outputs, report = generate(tmp_path, monkeypatch, source)
    scope, = report["scopes"]
    assert not scope["estimate_available"]
    assert "scalar inputs change" in scope["planning_reason"]
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    owner = text.split("subroutine fort_scope_owner_", 1)[1].split("end subroutine", 1)[0]
    assert "fort_scope_create" not in owner
    assert "call adjust(" in owner


@pytest.mark.parametrize("initialized", ["whole", "none"])
def test_changed_integer_payload_cannot_be_queried_early(tmp_path, monkeypatch, initialized):
    source = PROGRAM.replace("producer(a,b,n)", "producer(a,b,limits)").replace(
        "consumer(a,b,out,n)", "consumer(a,b,out,limits)")
    source = source.replace("integer,intent(in)::n\ninteger::i", "integer,intent(in)::limits(:)\ninteger::i,n\nn=limits(1)")
    source = source.replace("subroutine transform(b)\nreal(8),intent(inout)::b(:)\nb=3*b",
                            "subroutine transform(b,limits)\nreal(8),intent(inout)::b(:)\n"
                            "integer,intent(inout)::limits(:)\nb=3*b\nlimits=limits-1")
    source = source.replace("step(a,b,out,n)", "step(a,b,out,limits)").replace(
        "integer,intent(in)::n\ncall producer", "integer,intent(inout)::limits(:)\ncall producer")
    source = source.replace("call transform(b)", "call transform(b,limits)")
    captures = {"argument::a": FACT, "argument::b": {**FACT, "initialized": "none"},
                "argument::out": {**FACT, "initialized": "none"},
                "argument::limits": {**FACT, "initialized": initialized}}
    _, report = generate(tmp_path, monkeypatch, source, captures=captures)
    scope, = report["scopes"]
    assert not scope["estimate_available"]
    assert "payload arrays change" in scope["planning_reason"]


def test_generated_planning_owner_and_queries_compile_as_fortran(tmp_path, monkeypatch):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    outputs, report = generate(tmp_path, monkeypatch)
    for name, value in outputs.items():
        target = tmp_path / "build" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    build = tmp_path / "build"
    sources = report["build_sources"]
    ordered = [item for role in ("common_runtime", "shared_entry", "original_source")
               for item in sources if item["role"] == role and item["language"] == "fortran"]
    for item in ordered:
        target = build / Path(item["path"]).with_suffix(".o")
        result = subprocess.run([fortran, "-std=f2018", "-fopenmp", "-c", str(build / item["path"]),
                                 "-o", str(target)], cwd=build, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr


def test_query_does_not_duplicate_automatic_specification_expression(tmp_path, monkeypatch):
    source = WRAPPER_PROGRAM.replace("subroutine wrapper(a,b,out,n)\n",
                                     "subroutine wrapper(a,b,out,n)\nreal(8)::scratch(size(a))\n")
    outputs, report = generate(tmp_path, monkeypatch, source)
    scope, = report["scopes"]
    assert not scope["estimate_available"]
    assert "dynamic specification bound" in scope["planning_reason"]
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    owner = text.split("subroutine fort_scope_owner_", 1)[1].split("end subroutine", 1)[0]
    assert "fort_scope_create" not in owner
    assert "fort_scope_query_" not in text


def test_explicit_dummy_shape_cannot_silently_use_larger_capture_extent(tmp_path, monkeypatch):
    source = PROGRAM.replace("real(8),intent(in)::a(:)\nreal(8),intent(out)::b(:)",
                             "real(8),intent(in)::a(16)\nreal(8),intent(out)::b(16)", 1)
    _, report = generate(tmp_path, monkeypatch, source)
    assert any("whole-storage shape mapping" in item["reason"] for item in report["boundaries"])
    assert all("original::producer" not in scope["gpu_leaves"] for scope in report["scopes"])


def test_calibrated_auto_keeps_native_out_partial_definition_outside_scopes(tmp_path, monkeypatch):
    source = PROGRAM.replace("real(8),intent(inout)::b(:)\nb=3*b",
                             "real(8),intent(out)::b(:)\nb(2:size(b)-1)=4.d0")
    source = source.replace("do i=1,n\nout(i)=a(i)+b(i)", "do i=2,n-1\nout(i)=a(i)+b(i)")
    outputs, report = generate(tmp_path, monkeypatch, source)
    assert not report["automatic_scope_available"]
    assert not report["automatic_estimate_available"]
    assert not report["source_edits"]
    assert not any(name.startswith("sources/") or name.startswith("entries/") for name in outputs)
    assert any("native INTENT(OUT) effects require original-position definition hooks" in item["reason"]
               for item in report["boundaries"])
