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
from compiler.tests.test_numerical_sources import NORMALIZED, ORIGINAL, package
from compiler.tests.test_offload_profile import scoped_profile
from compiler.tests.test_source_scopes import ALLOCATABLE_PROGRAM, FACT, PROGRAM, WRAPPER_PROGRAM


def generate(tmp_path, monkeypatch, source=PROGRAM, *, captures=None, numerical_sources=None):
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
                                         config=OffloadConfig("auto", profile), numerical_sources=numerical_sources)
    return outputs, report


def packaged_control(tmp_path, monkeypatch, *, name="query_start", declaration=None, dependency=False,
                     guarded=False, mutable_consumer=False):
    declaration = declaration or f"integer,parameter::{name}=-1"
    original = ORIGINAL.replace("integer::i\n!$omp", "integer::i\n" + declaration + "\n!$omp", 1)
    original = original.replace("do i=-1,n-4", f"do i={name},n-4", 1)
    if dependency:
        original = original.replace("real(8) :: gain=2", "integer,parameter::origin=3\nreal(8) :: gain=2")
        original = original.replace("subroutine producer(a,b,n)\n", "subroutine producer(a,b,n)\nuse settings,only:seed=>origin\n")
    normalized = NORMALIZED.replace("weights,alb,blb,wlb)", f"weights,alb,blb,wlb,{name})")
    normalized = normalized.replace("integer,intent(in)::n,alb,blb,wlb", f"integer,intent(in)::n,alb,blb,wlb,{name}")
    normalized = normalized.replace("do i=-1,n-4", f"do i={name},n-4")
    if guarded:
        span = "call producer(a,b,n)\ncall transform(b)\ncall consumer(b,out,n)"
        original = original.replace(span, "if(n>0) then\n" + span + "\nendif")
        original = original.replace("!$omp parallel do private(i)\ndo i=" + name + ",n-4",
                                    f"if({name}<0) then\n!$omp parallel do private(i)\ndo i={name},n-4", 1)
        original = original.replace("enddo\n!$omp end parallel do", "enddo\n!$omp end parallel do\nendif", 1)
        normalized = normalized.replace("do i=" + name + ",n-4", f"if({name}<0) then\ndo i={name},n-4", 1)
        normalized = normalized.replace("enddo\nend subroutine", "enddo\nendif\nend subroutine", 1)
    if mutable_consumer:
        original = original.replace("integer::i\ndo i=2,n-1", f"integer::i\ninteger::{name}\n{name}=2\ndo i={name},n-1")
        consumer = f"""subroutine consumer(b,out,n,{name})
real(8),intent(in)::b(:)
real(8),intent(inout)::out(:)
integer,intent(in)::n,{name}
integer::i
do i={name},n-1
out(i)=b(i)
enddo
end subroutine
"""
        normalized = normalized.replace("end module", consumer + "end module")
    _source, _numerical, document = package(tmp_path, original=original, normalized=normalized)
    document["entries"][0]["parameters"].append({"name": name, "resource": "original::producer::" + name})
    if mutable_consumer:
        document["entries"].append({**document["entries"][0], "procedure": "original::consumer",
                                   "entry": "extracted::consumer", "parameters": [
                                       {"name": "b", "resource": "argument::b", "physical_origin": [0]},
                                       {"name": "out", "resource": "argument::out", "physical_origin": [0]},
                                       {"name": "n", "resource": "argument::n"},
                                       {"name": name, "resource": "original::consumer::" + name}]})
    captures = {root: FACT for root in ("argument::a", "argument::b", "argument::out", "settings::weights")}
    return generate(tmp_path, monkeypatch, original, captures=captures, numerical_sources=document)


@pytest.mark.parametrize(("name", "declaration", "dependency"), [
    ("query_start", "integer,parameter::query_start=-1", False),
    ("lane_lower", "integer,parameter::lane_lower=seed-4", True),
    ("slice_begin", "integer,parameter::base=3,slice_begin=base-4", False),
])
def test_normalized_local_integer_constants_remain_in_leaf_queries(tmp_path, monkeypatch, name, declaration, dependency):
    outputs, report = packaged_control(tmp_path, monkeypatch, name=name, declaration=declaration, dependency=dependency)
    assert report["automatic_estimate_available"], report["boundaries"]
    scope, = report["scopes"]
    assert scope["estimate_available"], scope["planning_reason"]
    assert all(parameter["resource"] != "original::producer::" + name for parameter in scope["parameters"])
    text = next(value for path, value in outputs.items() if path.startswith("sources/"))
    queries = [value.split("end subroutine", 1)[0] for value in text.split("subroutine fort_scope_query_")[1:]]
    query = next(value for value in queries if "fort_status = fort_plan(" in value and name in value)
    clone = next(value.split("end subroutine", 1)[0] for value in text.split("subroutine fort_scope_clone_")[1:]
                 if "fort_status = fort_run(" in value and name in value)
    for operation, value in (("fort_plan", query), ("fort_run", clone)):
        assert "PARAMETER" in value
        assert name in value
        call = value.split("fort_status = " + operation + "(", 1)[1]
        assert name in call
        if dependency:
            assert "seed => origin" in value


@pytest.mark.parametrize(("declaration", "reason"), [
    ("integer,parameter::query_start=2147483648", "default INTEGER literal is out of range"),
    ("integer,parameter::query_start=2147483647+1", "default INTEGER constant expression overflows"),
    ("integer,parameter::query_start=missing", "unresolved INTEGER kind parameter missing"),
    ("integer,parameter::base=query_start,query_start=base", "cyclic kind parameter query_start"),
    ("integer,parameter::query_start=abs(-1)", "unsupported INTEGER kind constant ABS(- 1)"),
    ("integer(kind=4),parameter::query_start=-1", "unresolved INTEGER kind parameter query_start"),
])
def test_unproved_normalized_local_constants_keep_whole_native_span(tmp_path, monkeypatch, declaration, reason):
    outputs, report = packaged_control(tmp_path, monkeypatch, declaration=declaration)
    scope, = report["scopes"]
    assert not scope["estimate_available"]
    assert reason in scope["planning_reason"]
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    owner = text.split("subroutine fort_scope_owner_", 1)[1].split("end subroutine", 1)[0]
    assert "fort_scope_create" not in owner
    assert "fort_scope_query_" not in text
    assert "call producer(" in owner
    assert "call consumer(" in owner


def test_same_name_mutable_leaf_resource_is_not_discharged_as_another_leaf_constant(tmp_path, monkeypatch):
    outputs, report = packaged_control(tmp_path, monkeypatch, mutable_consumer=True)
    scope, = report["scopes"]
    assert not scope["estimate_available"]
    assert "hidden resource is unavailable at owning scope: original::consumer::query_start" in scope["planning_reason"]
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    assert "fort_scope_query_" not in text


def test_local_constant_proof_keeps_outer_and_numerical_guards(tmp_path, monkeypatch):
    outputs, report = packaged_control(tmp_path, monkeypatch, guarded=True)
    assert report["automatic_estimate_available"], report["boundaries"]
    text = next(value for name, value in outputs.items() if name.startswith("sources/"))
    assert "if(n>0) then\ncall fort_scope_owner_" in text
    assert "if(query_start<0) then" in text
    assert any("query_start" in value and "if (" in value for name, value in outputs.items() if name.endswith(".cu"))


@pytest.mark.parametrize("guarded", [False, True])
def test_normalized_local_parameter_queries_compile_as_fortran(tmp_path, monkeypatch, guarded):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    outputs, report = packaged_control(tmp_path, monkeypatch, name="lane_lower",
                                       declaration="integer,parameter::lane_lower=seed-4", dependency=True, guarded=guarded)
    assert report["automatic_estimate_available"]
    build = tmp_path / "build"
    for name, value in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] == role and item["language"] == "fortran":
                path = build / item["path"]
                result = subprocess.run([fortran, "-std=f2018", "-fopenmp", "-c", str(path),
                                         "-o", str(path.with_suffix(".o"))], cwd=build,
                                        capture_output=True, text=True, timeout=60)
                assert result.returncode == 0, result.stdout + result.stderr


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
