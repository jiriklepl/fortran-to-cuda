"""Erased real reductions retain preconditions at every public GPU entry."""

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.common.sessions import _preserve_numerical_environment, session_names
from compiler.emission.cuda.batch import attempt_helper, window_worker
from compiler.emission.cuda.generator import _numerical_environment_guard
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


def test_direct_and_each_session_run_check_before_work_and_environment_masking(tmp_path):
    function, plan = prepared(tmp_path)
    source = generate_sources(function, plan).cuda
    direct = public_body(source, "cpp_advance", "void")
    assert direct.index("numerical_environment_supported()") < direct.index("HostFloatingEnvironment")
    assert direct.index("HostFloatingEnvironment") < direct.index("fort_internal_state{")
    names = session_names(function)
    run = public_body(source, "cpp_" + names.run, "void")
    assert run.index("numerical_environment_supported()") < run.index("HostFloatingEnvironment") < run.index(".get(")
    # Creation, updates and teardown preserve first-use/runtime setup effects,
    # while session runs recheck the current thread rather than cached creation.
    for action, result in ((names.create, "std::int64_t"), (names.destroy, "void"),
                           (names.trim_cache, "void"), (names.update_device + "_0", "void")):
        assert "HostFloatingEnvironment" in public_body(source, "cpp_" + action, result).split('extern "C"', 1)[0]


@pytest.mark.parametrize("policy", ["sections", "auto", "chunked", "hybrid"])
def test_ordinary_policy_query_declines_but_direct_execution_never_uses_cpp_fallback(tmp_path, policy):
    function, plan = prepared(tmp_path)
    sources = generate_sources(function, plan, offload_config=OffloadConfig(policy=policy))
    query = public_body(sources.cuda, "cpp_" + sources.offload["native_fallback_query"], "int")
    assert "if (!fort_runtime::numerical_environment_supported()) { return 0; }" in query
    direct = public_body(sources.cuda, "cpp_advance", "void")
    assert direct.index("storage::fail(\"unsupported numerical environment") < direct.index("HostFloatingEnvironment")
    assert sources.offload["numerical_environment"]["required"]


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


def test_unflagged_entries_keep_existing_emission(tmp_path):
    function, plan = prepared(tmp_path, required=False)
    sources = generate_sources(function, plan)
    assert "numerical_environment_supported()" not in sources.cuda
    assert "HostFloatingEnvironment" not in sources.cuda


@pytest.mark.parametrize("structured", [False, True])
def test_flagged_ordinary_auto_rejects_unpriced_protocol_in_query_and_actual_chooser(tmp_path, structured):
    function, plan = prepared(tmp_path, structured=structured)
    sources = generate_sources(function, plan, offload_config=OffloadConfig(policy="auto", profile=calibration()))
    assert not sources.offload["estimate_available"]
    assert sources.offload["estimate_reason"] == "numerical_environment_protocol_calibration_unavailable"
    query = public_body(sources.cuda, "cpp_" + sources.offload["native_fallback_query"], "int")
    direct = public_body(sources.cuda, "cpp_advance", "void")
    for body in (query, direct):
        assert ("if (!(false))" in body if not structured else "offload::context_valid(4, false) && false" in body)


@pytest.mark.parametrize("policy", ["auto", "sections"])
def test_flagged_scoped_preserves_effect_queries_and_forced_gpu_but_declines_costs(tmp_path, policy):
    function, plan = prepared(tmp_path)
    scoped = generate_scoped(function, plan, OffloadConfig(policy=policy, profile=calibration()), "common.cuh",
                             runtime_id=read_scoped_runtime()[1]["runtime_id"])
    assert not scoped.report["automatic_estimate_available"]
    assert scoped.report["automatic_reason"] == "numerical_environment_protocol_calibration_unavailable"
    assert scoped.report["planning"]["query_available"]
    assert all(item["gpu_available"] for item in scoped.report["region_execution"])
    query = public_body(scoped.cuda, scoped.report["planning"]["entry"], "int")
    assert "FORT_SCOPE_PLAN_WORKER" in query
    assert ", 0.0, 0.0, false," in query
    choose = public_body(scoped.cuda, scoped.report["planning"]["selector"], "int")
    assert "costs.valid = 0;" in choose
    run = public_body(scoped.cuda, scoped.report["entry"], "int")
    assert "const bool fort_gpu_requested = fort_mode == 1;" in run


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


@pytest.mark.native
def test_guard_rejects_rounding_and_traps_before_work_and_preserves_existing_flags(tmp_path):
    function = SimpleNamespace(requires_numerical_environment=True)
    guard = "\n".join(_numerical_environment_guard(function))
    preserve = "\n".join(_preserve_numerical_environment())
    compile_probe(tmp_path, r'''
#include "floating_environment.hpp"
#include <cassert>
#include <stdexcept>
namespace storage { [[noreturn]] void fail(const char *) { throw std::runtime_error("rejected"); } }
static int work = 0;
void run() {
''' + guard + "\n" + preserve + r'''
    ++work;
    std::feraiseexcept(FE_INVALID);
    std::fesetround(FE_DOWNWARD);
}
int main() {
#if defined(__GLIBC__)
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_INEXACT);
    std::fesetround(FE_UPWARD);
    try { run(); assert(false); } catch(const std::runtime_error &) {}
    assert(work == 0 && std::fegetround() == FE_UPWARD);
    std::fesetround(FE_TONEAREST);
    feenableexcept(FE_INVALID);
    try { run(); assert(false); } catch(const std::runtime_error &) {}
    assert(work == 0 && fegetexcept() == FE_INVALID);
    fedisableexcept(FE_ALL_EXCEPT);
    run();
    assert(work == 1 && std::fegetround() == FE_TONEAREST);
    assert(std::fetestexcept(FE_ALL_EXCEPT) == FE_INEXACT && fegetexcept() == 0);
#else
    return 77;
#endif
}
''')


@pytest.mark.native
def test_collective_query_rejects_one_unsupported_participant_uniformly(tmp_path):
    guard = "\n".join(_numerical_environment_guard(SimpleNamespace(requires_numerical_environment=True),
                                                  collective=True, query=True))
    compile_probe(tmp_path, r'''
#include "floating_environment.hpp"
#include <cassert>
#include <omp.h>
int query() {
''' + guard + r'''
    return 1;
}
int main() {
#if defined(__GLIBC__)
    int rejected=0, accepted=0;
    #pragma omp parallel num_threads(4) reduction(+:rejected,accepted)
    {
        std::fesetround(omp_get_thread_num() == 1 ? FE_DOWNWARD : FE_TONEAREST);
        rejected += query() == 0;
        std::fesetround(FE_TONEAREST);
        accepted += query() == 1;
    }
    assert(rejected == 4 && accepted == 4);
#else
    return 77;
#endif
}
''')
