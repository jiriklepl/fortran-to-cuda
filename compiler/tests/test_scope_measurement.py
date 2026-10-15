"""Caller observations execute one admitted invocation without pricing it."""

import json
import os
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.offload.scope_measurement import PREFIX, PROTOCOL
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_lexical_source_owner import SOURCE
from compiler.tests.test_source_scopes import FACT


def source(*, teams=False, call="call step(a,b,out,n,escape)"):
    text = SOURCE.replace("call opaque(b,n)", "b(-2)=b(-2)+sum(a)")
    if teams:
        text = text.replace("do i=-2,n-3", "!$omp parallel do private(i) schedule(static)\ndo i=-2,n-3")
        text = text.replace("enddo\n", "enddo\n!$omp end parallel do\n")
    return text.replace("end module", """subroutine caller(a,b,out,n,escape)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(-2:),out(-2:)
integer,intent(in)::n
logical,intent(in)::escape
""" + call + "\nend subroutine\nend module")


def builder(path, variant, text=None):
    if text is not None:
        path.write_text(text)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}}
    return ScopeBuilder([path], "local_owner::step", facts=facts, options=CompilerOptions(),
                        config=OffloadConfig(policy="sections", scope_execution="reached", scope_measurement=variant))


@pytest.mark.parametrize("teams", [False, True], ids=["serial", "original-team"])
def test_variants_share_identity_but_native_keeps_original_callee(tmp_path, teams):
    path = tmp_path / "original.f90"
    original = source(teams=teams)
    results = {variant: builder(path, variant, original).run() for variant in ("native", "resident")}
    records = {}
    for variant, (outputs, report) in results.items():
        scope, = report["scopes"]
        record = records[variant] = scope["owner_measurement"]
        assert record["protocol"] == PROTOCOL
        assert record["variant"] == variant
        assert not record["placement_authority"]
        assert not record["calibration_evidence"]
        assert not record["automatic_estimate_available"]
        assert not report["automatic_estimate_available"]
        assert record["execution_identity"] == scope["checked_execution_plan"]["execution_identity"]
        site, = record["call_sites"]
        assert site["caller"] == "local_owner::caller"
        lines = original.splitlines(keepends=True)[site["first_line"]-1:site["last_line"]]
        assert sha256("".join(lines).encode()).hexdigest() == site["call_sha256"]
        transformed = outputs[report["sources"][str(path)]["replacement"]]
        assert transformed.count("call step(a,b,out,n,escape)") == 1
        before, after = transformed.split("call step(a,b,out,n,escape)")
        assert before.rfind("call system_clock(") > before.rfind("subroutine caller(")
        assert after.index("call system_clock(") < after.index(PREFIX)
        if variant == "native":
            assert transformed.startswith(original[:original.index("subroutine caller(")])
            assert "TARGET" not in transformed
            assert "fort_scope_" not in transformed
            assert report["runtime"] is None
            assert all(row["role"] == "original_source" for row in report["build_sources"])
        else:
            assert "TARGET" in transformed
            assert "fort_scope_close(" in transformed
            assert report["runtime"] is not None
    assert records["native"]["identity"] == records["resident"]["identity"]
    assert records["native"]["execution_identity"] == records["resident"]["execution_identity"]


@pytest.mark.parametrize("call", ["if(escape) call step(a,b,out,n,escape)",
                                  "100 call step(a,b,out,n,escape)",
                                  "n=1;call step(a,b,out,n,escape)"])
def test_nonstandalone_or_labelled_call_is_rejected_without_artifacts(tmp_path, call):
    selected = builder(tmp_path / "original.f90", "native", source(call=call))
    with pytest.raises(CompilationError, match="measurement"):
        selected.run()
    assert not selected.edits
    assert not selected.scopes


@pytest.mark.parametrize("variant", ["native", "resident"])
def test_boundary_and_missing_callers_are_not_unobserved_successes(tmp_path, variant):
    for index, text in enumerate((source().replace("b(-2)=b(-2)+sum(a)", "return"), SOURCE)):
        selected = builder(tmp_path / f"original{index}.f90", variant, text)
        with pytest.raises(CompilationError, match="measurement|complete lexical"):
            selected.run()
        assert not selected.edits
        assert not selected.scopes


@pytest.mark.parametrize("context", ["pure", "elemental", "parallel", "concurrent", "directive"])
def test_unsupported_caller_context_is_rejected(tmp_path, context):
    text = source()
    call = "call step(a,b,out,n,escape)"
    if context in {"pure", "elemental"}:
        text = text.replace("subroutine caller(", context + " subroutine caller(")
    else:
        before, after = {"parallel": ("!$omp parallel\n", "\n!$omp end parallel"),
                         "concurrent": ("do concurrent(j=1:1)\n", "\nenddo"),
                         "directive": ("!$omp single\n", "\n!$omp end single")}[context]
        text = text.replace(call, before + call + after)
    selected = builder(tmp_path / "original.f90", "native", text)
    with pytest.raises(CompilationError, match="measurement"):
        selected.run()
    assert not selected.edits
    assert not selected.scopes


def test_call_after_completed_original_team_is_observed(tmp_path):
    text = source(call="!$omp parallel\n!$omp end parallel\ncall step(a,b,out,n,escape)")
    _outputs, report = builder(tmp_path / "original.f90", "native", text).run()
    assert len(report["scopes"][0]["owner_measurement"]["call_sites"]) == 1


def test_caller_authentication_is_rechecked_before_publication(tmp_path, monkeypatch):
    path = tmp_path / "original.f90"
    selected = builder(path, "native", source())
    original_finish = ScopeBuilder.finish

    def change(candidate, *args, **kwargs):
        path.write_text(path.read_text() + "\n! changed before publication\n")
        return original_finish(candidate, *args, **kwargs)

    monkeypatch.setattr(ScopeBuilder, "finish", change)
    with pytest.raises(CompilationError, match="changed"):
        selected.run()
    assert not selected.edits
    assert not selected.scopes


def test_native_original_team_executes_once_and_emits_complete_clock_rows(tmp_path):
    compiler = shutil.which("gfortran-14") or shutil.which("gfortran")
    if compiler is None:
        pytest.skip("Fortran compiler required")
    path = tmp_path / "original.f90"
    outputs, report = builder(path, "native", source(teams=True)).run()
    transformed = tmp_path / "measured.f90"
    transformed.write_text(outputs[report["sources"][str(path)]["replacement"]])
    driver = tmp_path / "driver.f90"
    driver.write_text("""program check
use local_owner,only:caller,visits
implicit none
real(8)::a(-2:5),b(-2:5),out(-2:5)
integer::i
do i=-2,5
a(i)=real(i,8)
enddo
b=-3
out=-7
call caller(a,b,out,8,.true.)
if(visits/=1) error stop 'replayed first invocation'
if(any(out(-1:5)/=4*a(-1:5)).or.out(-2)/=4) error stop 'first fields differ'
call caller(a,b,out,8,.false.)
if(visits/=2) error stop 'replayed second invocation'
if(any(out/=4*a)) error stop 'second fields differ'
end program
""")
    binary = tmp_path / "check"
    subprocess.run([compiler, "-O2", "-fopenmp", "-ffree-line-length-none", str(transformed), str(driver),
                    "-o", str(binary)], cwd=tmp_path, check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], cwd=tmp_path,
                            env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"},
                            check=True, capture_output=True, text=True)
    rows = [json.loads(line.removeprefix(PREFIX)) for line in result.stderr.splitlines() if line.startswith(PREFIX)]
    assert len(rows) == 2
    for row in rows:
        assert row["status"] == "complete"
        assert row["variant"] == "native"
        assert row["seconds"] >= 0
        assert row["clock_rate"] > 0
        assert row["seconds"] == pytest.approx((row["clock_finish"]-row["clock_start"])/row["clock_rate"])
        assert row["identity"] == report["scopes"][0]["owner_measurement"]["identity"]
