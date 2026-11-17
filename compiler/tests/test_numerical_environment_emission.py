"""Erased real reductions retain preconditions at every public GPU entry."""

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from compiler.driver.pipeline import prepare_function
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.batch import attempt_helper, window_worker
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig
from compiler.tests.test_scoped_planning_entries import calibration


def prepared(tmp_path, *, required=True, structured=False):
    path = tmp_path / "ordinary.f90"
    source = """module numerical
contains
subroutine advance(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=a(i)+1.d0
enddo
end subroutine
end module
"""
    if structured:
        source = source.replace("do i=1,n", "if(n>0) then\ndo i=1,n").replace("enddo", "enddo\nendif")
    path.write_text(source)
    return prepare_function(replace(lower_file(path, "advance"), requires_numerical_environment=required))


def public_body(source, name, result):
    return source.split(f'extern "C" {result} {name}(', 1)[1]


def test_default_is_compatible_and_preparation_retains_requirement(tmp_path):
    function, _ = prepared(tmp_path, required=False)
    assert not function.requires_numerical_environment
    function, _ = prepared(tmp_path)
    assert function.requires_numerical_environment


@pytest.mark.parametrize("collective", [False, True])
def test_scoped_public_entries_reject_before_effects_and_report_requirement(tmp_path, collective):
    function, plan = prepared(tmp_path)
    scoped = generate_scoped(function, plan, OffloadConfig(policy="sections", collective=collective), "common.cuh")
    assert scoped.report["numerical_environment"]["required"]
    run = public_body(scoped.cuda, scoped.report["entry"], "int")
    assert run.index("unsupported numerical environment") < run.index("fort_scope_layout_get(")
    query = public_body(scoped.cuda, scoped.report["planning"]["entry"], "int")
    assert query.index("unsupported planning numerical environment") < query.index("fort_scope_layout_get(")
    if collective:
        team = public_body(scoped.cuda, scoped.report["team"]["entry"], "int")
        assert team.index("shared->environment_supported = false") < team.index("unsupported team numerical environment")
        assert team.index("unsupported team numerical environment") < team.index("fort_scope_layout_get(")


def test_public_batch_and_window_cannot_bypass_environment_check(tmp_path):
    function, plan = prepared(tmp_path)
    window = "\n".join(window_worker(function, plan, "window"))
    attempt = "\n".join(attempt_helper(function, (), "attempt", "prepare", "callback", None, None, None, 4, 64))
    for source in (window, attempt):
        assert source.index("numerical_environment_supported()") < source.index("return FORT_SCOPE_ARGUMENT")


@pytest.mark.parametrize("collective", [False, True])
def test_graph_flagged_heuristic_retains_unpriced_provenance_and_prework_guards(tmp_path, collective):
    from compiler.offload.config import OffloadConfig

    function, plan = prepared(tmp_path)
    scoped = generate_scoped(function, plan,
        OffloadConfig(policy="auto", profile=calibration(), collective=collective), "common.cuh",
        runtime_id=read_scoped_runtime()[1]["runtime_id"])
    report = scoped.report
    assert report["automatic_estimate_available"]
    assert report["numerical_environment"]["required"]
    provenance = report["graph_auto_costs"]
    assert provenance["heuristic"]
    assert provenance["cost_authorizes_execution"] is False
    unit, = provenance["units"].values()
    assert unit["kind"] == "hardware_roofline_heuristic"
    assert unit["static_work_proxy"] == 1
    assert unit["original_fortran_family_attestation"] is False
    assert unit["speedup_guaranteed"] is False
    assert provenance["unpriced_team_overhead"] is collective

    def body(name):
        # Limit each assertion to its actual public function, so later entries
        # cannot accidentally supply the guard or operation being checked.
        tail = public_body(scoped.cuda, name, "int")
        opening = tail.index("{")
        depth = 1
        for position in range(opening + 1, len(tail)):
            depth += (tail[position] == "{") - (tail[position] == "}")
            if depth == 0:
                return tail[opening + 1:position]
        raise AssertionError("public function body is not closed")

    planning = report["planning"]
    run, query, choose = (body(name) for name in
                          (report["entry"], planning["entry"], planning["selector"]))
    guard = "if (!fort_runtime::numerical_environment_supported())"
    assert run.index(guard) < run.index("fort_scope_layout_get(")
    assert query.index(guard) < query.index("fort_scope_layout_get(")
    assert choose.index(guard) < choose.index("costs.valid = 1;")
    record, = [line for line in query.splitlines() if "fort_access.record(FORT_SCOPE_PLAN_WORKER," in line]
    assert ", unit.units[0].flops, unit.units[0].memory_bytes, true," in record
    assert "const bool fort_gpu_requested = fort_mode == 1;" in run
    if collective:
        team = body(report["team"]["entry"])
        state_name = report["fortran_module"] + "_team_state"
        state = scoped.cuda.split("struct " + state_name + " {", 1)[1].split("};", 1)[0]
        assert "bool environment_supported = true;" in state
        assert team.index(state_name + " local_state{};") < team.index("#pragma omp single copyprivate(shared)")
        assert team.index("#pragma omp single copyprivate(shared)") < team.index("shared = &local_state;")
        assert team.index("shared = &local_state;") < team.index(guard)
        assert team.index(guard) < team.index("shared->environment_supported = false;")
        assert team.index("shared->environment_supported = false;") < team.index("unsupported team numerical environment")
        assert team.index("unsupported team numerical environment") < team.index("fort_scope_layout_get(")
        assert "#pragma omp atomic write" in team
        vote_done = team.index("#pragma omp barrier", team.index("shared->environment_supported = false;"))
        check = team.index("if (!shared->environment_supported)")
        assert vote_done < team.index("#pragma omp master", vote_done) < check


def compile_probe(tmp_path, source):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("requires a native C++ compiler")
    path = tmp_path / "probe.cpp"
    path.write_text(source)
    binary = tmp_path / "probe"
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    compiled = subprocess.run([compiler, "-std=c++17", "-O2", "-fopenmp", "-I", str(runtime), str(path), "-o", str(binary)],
                              cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    completed = subprocess.run([str(binary)], cwd=tmp_path, text=True, capture_output=True, timeout=30)
    if completed.returncode == 77:
        pytest.skip("runtime trap-mask inquiry requires glibc")
    assert completed.returncode == 0, completed.stdout + completed.stderr
