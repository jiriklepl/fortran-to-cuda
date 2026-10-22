"""Cost-selected CPU work keeps the checked graph's coherence authority."""

from __future__ import annotations

import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_scoped_planning_entries import calibration

PRODUCER = """do i=1,n
v=a(i)
do k=1,32
v=sqrt(v*v)+1.0_8
enddo
b(i)=v
enddo
"""
COPY = """do i=1,n
c(i)=b(i)
counter(i)=counter(i)+1
enddo
"""
CONSUMER = """do i=1,n
v=a(i)+b(i)
do k=1,32
v=sqrt(v*v)+1.0_8
enddo
d(i)=v
enddo
"""
DECLARATIONS = """real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:),d(:)
integer,intent(inout)::counter(:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i,k
real(8)::v
"""


def source_case(style):
    if style == "combined":
        team = "".join("!$omp parallel do private(i,k,v) default(none) shared(a,b,c,d,counter,n,flag)\n"
            + body + "!$omp end parallel do\n"
            for body in (PRODUCER, COPY, CONSUMER))
    else:
        team = "!$omp parallel private(i,k,v) shared(a,b,c,d,counter,n,flag) default(none)\n"
        team += "!$omp do\n" + PRODUCER + "!$omp end do\n"
        team += "if(flag) then\n!$omp do\n" + COPY + "!$omp end do\nendif\n"
        team += "!$omp do\n" + CONSUMER + "!$omp end do\n!$omp end parallel\n"
    return "module native_choice\nimplicit none\ncontains\nsubroutine advance(a,b,c,d,counter,n,flag)\n" + (
        DECLARATIONS + team + "end subroutine\nend module\n")


def emit(directory, style, *, profile=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "source.f90"
    path.write_text(source_case(style))
    builder = ScopeBuilder([path], "native_choice::advance",
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
               "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "c", "d", "counter")}},
        options=CompilerOptions(opt_level=0), config=OffloadConfig("auto", calibration() if profile is None else profile))
    outputs, report = builder.run()
    replacement = outputs[report["sources"][str(path)]["replacement"]]
    return outputs, report, replacement


def test_owning_combined_team_uses_independent_original_native_effect_authority(tmp_path):
    _, report, text = emit(tmp_path, "combined")
    owner, = report["scopes"]
    assert not owner["boundaries"]
    teams = [item["operations"]["combined_numerical_team"] for item in owner["planning_segments"]]
    assert len(teams) == 3
    assert all(team["native_effects_authority"] for team in teams)
    assert all(team["original_cpu_alternative"]["available"] for team in teams)
    assert all("original Fortran body once" in team["cpu_choice"] for team in teams)
    assert all(team["original_cpu_alternative"]["native_completion"]["available"] for team in teams)
    assert all(team["original_cpu_alternative"]["sections"]["available"] for team in teams)
    assert "fort_scope_plan_cancel_native_v1" not in text
    assert "fort_decision%gpu_units == 0" in text
    assert "fort_team_run = .not. fort_team_native" in text
    assert "fort_team_native_begin" in text
    assert "if (fort_team_native) then" in text
    assert "=> run_team" in text


def test_optional_native_decline_retains_generated_numerical_worker(tmp_path, monkeypatch):
    from compiler.ir import CompilationError
    from compiler.scopes import segments

    original = segments.fragment

    def decline(builder, nodes, **keywords):
        if keywords.get("retain_proof_analysis"):
            raise CompilationError("test unsupported independent original native effects")
        return original(builder, nodes, **keywords)

    monkeypatch.setattr(segments, "fragment", decline)
    _, report, text = emit(tmp_path, "combined")
    teams = [item["operations"]["combined_numerical_team"] for item in report["scopes"][0]["planning_segments"]]
    assert len(teams) == 3
    assert all(not team["original_cpu_alternative"]["available"] for team in teams)
    assert all("test unsupported" in team["original_cpu_alternative"]["reason"] for team in teams)
    assert "fort_team_native_begin" not in text
    assert "=> run_team" in text


def test_native_handoff_reuses_query_geometry_without_recording_or_reselecting(tmp_path):
    outputs, _, text = emit(tmp_path, "combined")
    entries = [source for name, source in outputs.items() if name.endswith("shared_entry.cu")]
    assert entries
    for entry in entries:
        if "_native_begin_v1(" not in entry:
            continue
        helper = entry.split("_native_begin_v1(", 1)[1].split('extern "C" int', 1)[0]
        assert "fort_native_expected.add(" in helper
        assert "fort_native_expected.begin(" in helper
        assert "fort_access.record(" not in helper
        assert "fort_scope_plan_reset" not in helper
        assert "fort_scope_plan_select" not in helper
        assert "fort_scope_plan_cancel" not in helper
    assert "if (fort_team_query_plan) then\nfort_team_try_native = .true." in text


def test_joined_team_cancels_only_cost_native_query_before_independent_native_hooks(tmp_path):
    _, report, text = emit(tmp_path, "joined")
    owner, = report["scopes"]
    assert not owner["boundaries"]
    joined, = [item["operations"] for item in owner["planning_segments"] if item["operations"].get("joined_team")]
    assert joined["native_operations"]
    assert "fort_scope_plan_cancel_native_v1" in text
    first_cancel = text.index("fort_status = fort_scope_plan_cancel_native_v1")
    assert "fort_decision%gpu_units == 0" in text[:first_cancel]
    after_cancel = text[first_cancel:]
    assert after_cancel.index("fort_scope_plan_reset_mode") < after_cancel.index("fort_scope_access_begin_v1")
    assert "counter(i)=counter(i)+1" in text


def test_default_none_shares_variables_on_continued_generated_declarations():
    from compiler.scopes.joined_team import declared_names

    assert declared_names(["integer(c_int64_t) :: origin(1), lower(1), &",
                           "  & upper(1)", "real(8) :: values(2,3)"]) == ("origin", "lower", "upper", "values")


@pytest.mark.native
@pytest.mark.parametrize("style", ["combined", "joined"])
def test_cost_native_team_control_and_cancel_interface_compile(tmp_path, style):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if fortran is None:
        pytest.skip("Fortran compiler required")
    outputs, report, _ = emit(tmp_path, style)
    for name, text in outputs.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] != role or item["language"] != "fortran":
                continue
            result = subprocess.run([fortran, "-O2", "-std=f2018", "-fopenmp", "-c",
                str(tmp_path / item["path"]), "-o", str(tmp_path / (item["path"].replace("/", "_") + ".o"))],
                cwd=tmp_path, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stdout + result.stderr
