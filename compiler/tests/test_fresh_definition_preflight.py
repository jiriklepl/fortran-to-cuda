"""A fresh inevitable coarse read rejection precedes context allocation."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from compiler.scopes.registration import build_registration, fresh_undefined_read_guard
from compiler.tests.test_graph_source_owner import FACT, SOURCE, emit

MARKER = "! Fresh owner required-whole read has no initial definitions."
SOURCE_READ = SOURCE.replace(
    "implicit none", "implicit none\ninteger,protected::table(8)=[1,2,3,4,5,6,7,8]"
).replace("integer::i", "integer::i,m\nexternal opaque").replace(
    "do i=2,n\nx(i)=x(i-1)+1\nenddo", "m=table(table(n))\ncall opaque(a,n)\nm=table(table(n))"
).replace("a(i)=b(i)+1.0_8", "a(i)=b(i)+m")


def guard(fact=None, *, required=("argument::a",), facts=None):
    return fresh_undefined_read_guard(
        facts or {"argument::a": fact or FACT}, required, {"argument::a": "a"},
        context="context", query_plan="query", entry_guard="epoch",
        on_error=("enabled=.false.", "epoch=.false.", "query=.false.", "exit reached"))


@pytest.mark.parametrize(("required", "facts"), [
    ((), {"argument::a": FACT}),
    (("argument::a",), {"argument::a": {**FACT, "requires_registered_handle": True}}),
    (("argument::a",), {"argument::a": FACT,
                         "argument::b": {**FACT, "requires_registered_handle": True}}),
])
def test_missing_requirement_and_external_ownership_keep_runtime_path(required, facts):
    assert not guard(required=required, facts=facts)


def test_guard_uses_the_registration_epoch_policy_without_handle_value_assumptions():
    text = "\n".join(guard())
    assert "if (context == 0) then\nif (.not. query) then\nif (.not. epoch) then" in text
    assert "all(shape(a, kind=c_size_t) > 0_c_size_t)" in text
    assert "FORT_SCOPE_UNINITIALIZED" in text
    assert "handle" not in text
    registration = build_registration(FACT, 1, "handle", "fort_test", "1_c_int64_t", "layout",
                                      entry_guard="epoch")
    rendered = "\n".join(registration.prepare)
    assert "if (epoch) then" in rendered
    assert "1_c_int, handle)\nelse" in rendered
    assert "0_c_int, handle)" in rendered
    assert "if (.true.) then" in "\n".join(guard({**FACT, "initialized": "none"}))
    assert "if (.not. epoch) then" in "\n".join(guard({**FACT, "initialized": "sections", "sections": []}))


@pytest.mark.native
@pytest.mark.parametrize("coverage", ["whole", "none", "sections"])
def test_executed_guard_preserves_empty_live_query_and_pristine_paths(tmp_path, coverage):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    if fc is None:
        pytest.skip("Fortran compiler unavailable")
    statements = "\n".join(guard({**FACT, "initialized": coverage, "sections": []}))
    source = """program check_guard
use iso_c_binding
implicit none
integer(c_int),parameter::FORT_SCOPE_UNINITIALIZED=7
integer(c_int)::fort_status
integer(c_int64_t)::context,stale_handle
integer::size_case,live_case,query_case,pristine_case,rejected,created,original_runs,later_runs
real(8),allocatable::a(:)
logical::query,epoch,enabled,expected
! Nonzero local slot from an earlier closed owning context must not stop the proof.
stale_handle=1234567_c_int64_t
original_runs=0
later_runs=0
do size_case=0,2,2
 allocate(a(-3:size_case-4))
 do live_case=0,1
 do query_case=0,1
 do pristine_case=0,1
  context=int(live_case,c_int64_t)
  query=query_case==1
  epoch=pristine_case==1
  enabled=.true.
  fort_status=0
  created=0
  rejected=0
  expected=context==0.and..not.query.and.size_case>0.and.(""" + (
        ".true." if coverage == "none" else ".not.epoch") + """)
  reached:block
""" + statements + """
   ! The original registration/preflight path is retained whenever the proof cannot reject.
   created=created+1
  end block reached
  if(.not.enabled) then
   rejected=1
   original_runs=original_runs+1
   if(fort_status/=FORT_SCOPE_UNINITIALIZED) error stop 'wrong prework status'
   if(epoch.or.query) error stop 'fallback did not retire entry flags'
  endif
  if((rejected==1).neqv.expected) error stop 'incorrect fresh rejection'
  if(created/=1-rejected) error stop 'created doomed context or skipped valid path'
  if(stale_handle/=1234567_c_int64_t) error stop 'changed cached slot'
  ! A later independently reached unit can rearm after this one original execution.
  enabled=.true.
  if(enabled) later_runs=later_runs+1
 enddo
 enddo
 enddo
 deallocate(a)
enddo
if(later_runs/=16) error stop 'lost later unit'
if(original_runs/=""" + ("2" if coverage == "none" else "1") + """) error stop 'original source count'
print *, 'GUARD_FIELDS_OK'
end program
"""
    path = tmp_path / "guard.f90"
    path.write_text(source)
    binary = tmp_path / "guard"
    result = subprocess.run([fc, "-std=f2018", "-fcheck=all", str(path), "-o", str(binary)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GUARD_FIELDS_OK" in result.stdout


@pytest.mark.parametrize("mapping", [False, True], ids=["owning", "module-companion"])
def test_emitted_guard_follows_original_descriptor_guards_and_uses_existing_close(tmp_path, mapping):
    source = SOURCE_READ
    if mapping:
        source = source.replace("subroutine advance(a,b,x,n)", "subroutine helper(a,b,x,n)")
        source = source.replace("end module", """subroutine advance(a,b,x,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(inout)::x(:)
integer,intent(in)::n
call helper(a,b,x,n)
end subroutine
end module""")
    path, outputs, report = emit(tmp_path, source, captures={"graph_fixture::table": FACT})
    owner, = report["scopes"]
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.count(MARKER) >= 2
    if mapping:
        assert owner["module_coordinators"]
        assert "fort_child_context_" in text
    for block in text.split(MARKER)[1:]:
        guard_text, registration = block.split("fort_status = fort_scope_create", 1)
        assert "fort_status = FORT_SCOPE_UNINITIALIZED" in guard_text
        assert "fort_scope_close(" in guard_text
        assert "= .false." in guard_text
        assert "exit fort_reached_" in guard_text
        assert "fort_scope_register(" in registration
    # This first scalar read executes under pristine facts; an opaque cut then
    # leaves the exact same checked slot stale for the second scalar read.
    assert text.count("m=table(table(n))") == (4 if mapping else 2)
    assert "fort_scope_buffer_valid_v1" in text
    # The new inquiry is after original activation and descriptor checks.
    prefix = text[:text.index(MARKER)]
    assert "fort_can = is_contiguous(table)" in prefix
    assert "size(table, 1, kind=8)" in prefix


@pytest.mark.native
def test_full_generated_fortran_with_new_guard_compiles(tmp_path):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    if fc is None:
        pytest.skip("Fortran compiler unavailable")
    _path, outputs, report = emit(tmp_path / "inputs", SOURCE_READ,
                                  captures={"graph_fixture::table": FACT})
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, item in enumerate(report["build_sources"]):
            if item["language"] != "fortran" or item["role"] != role:
                continue
            result = subprocess.run([fc, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-c",
                                     str(build / item["path"]), "-J", str(build), "-I", str(build),
                                     "-o", str(build / f"source-{index}.o")],
                                    cwd=build, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
