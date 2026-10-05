"""Allocation reuse is an explicit ordinary-call policy with a portable trim API."""

from dataclasses import replace

import pytest

from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.emission.c.sessions import append_cpu_sessions
from compiler.emission.common.abi import abi_arguments
from compiler.emission.common.memory import render_memory
from compiler.emission.common.sessions import lifecycle_name, session_names
from compiler.emission.cuda.generator import generate_cuda
from compiler.memory import MemoryOperation, plan_memory
from compiler.tests import test_call_ownership as calls
from compiler.tests.test_language import lower
from compiler.tests.test_memory_runtime import _run, _tool


def prepared(tmp_path):
    return prepare_function(lower(tmp_path, "do i=1,n\na(i)=a(i)+1\nenddo"))


def helper(source, name):
    return source.split(f"static void {name}(", 1)[1].split(") {\n", 1)[1].split("\n}\n", 1)[0]


def emit(function, plan, **kwargs):
    return generate_cuda(function, plan, abi_arguments(function.parameters), "common_functions.cuh", **kwargs)


def test_default_cuda_uses_pooled_ordinary_acquisition_and_dedicated_sessions(tmp_path):
    function, plan = prepared(tmp_path)
    generated = generate_sources(function, plan)
    assert emit(function, plan) == generated.cuda
    session_create = helper(generated.cuda, lifecycle_name(function, "create"))
    ordinary_create = helper(generated.cuda, lifecycle_name(function, "ordinary_create"))
    assert "AllocationPolicy::dedicated" in session_create
    assert "AllocationPolicy::pooled" in ordinary_create
    assert session_create.replace("dedicated", "pooled") == ordinary_create
    ordinary = generated.cuda.split('extern "C" void cpp_entry(', 1)[1]
    assert f"{lifecycle_name(function, 'ordinary_create')}(" in ordinary
    for phase in ("run", "retrieve", "destroy"):
        assert f"{lifecycle_name(function, phase)}(" in ordinary
        assert lifecycle_name(function, "ordinary_" + phase) not in generated.cuda
    assert "AllocationPolicy::pooled" not in generated.cpp
    assert "fort_internal_token" not in ordinary
    assert "_registry" not in ordinary


def with_extra_syncs(memory):
    return replace(
        memory,
        **{
            phase: (MemoryOperation("sync"), *getattr(memory, phase))
            for phase in ("create", "run", "retrieve", "destroy")
        },
    )


def test_legacy_explicit_memory_controls_every_phase_of_both_interfaces(tmp_path):
    function, plan = prepared(tmp_path)
    memory = with_extra_syncs(plan_memory(plan, function.parameters))
    source = emit(function, plan, memory=memory)
    assert "AllocationPolicy::pooled" not in source
    assert "_ordinary_" not in source
    for phase, syncs in (("create", 1), ("run", 1), ("retrieve", 2), ("destroy", 2)):
        body = helper(source, lifecycle_name(function, phase))
        assert body.count("storage::synchronize();") == syncs


@pytest.mark.parametrize("explicit_session", [False, True])
def test_ordinary_override_renders_all_its_phases_without_changing_sessions(tmp_path, explicit_session):
    function, plan = prepared(tmp_path)
    session = plan_memory(plan, function.parameters, acquisition_policy="dedicated")
    ordinary = with_extra_syncs(plan_memory(plan, function.parameters, acquisition_policy="pooled"))
    kwargs = {"memory": session} if explicit_session else {}
    source = emit(function, plan, ordinary_memory=ordinary, **kwargs)
    wrapper = source.split('extern "C" void cpp_entry(', 1)[1]
    for phase, syncs in (("create", 0), ("run", 0), ("retrieve", 1), ("destroy", 1)):
        assert helper(source, lifecycle_name(function, phase)).count("storage::synchronize();") == syncs
        name = lifecycle_name(function, "ordinary_" + phase)
        assert helper(source, name).count("storage::synchronize();") == syncs + 1
        assert name + "(" in wrapper


@pytest.mark.parametrize("policy", [None, "dedicated"])
def test_identical_explicit_plans_share_every_helper(tmp_path, policy):
    function, plan = prepared(tmp_path)
    memory = plan_memory(plan, function.parameters, acquisition_policy=policy)
    source = emit(function, plan, memory=memory, ordinary_memory=replace(memory))
    assert "_ordinary_" not in source
    assert "AllocationPolicy::pooled" not in source


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_pooled_session_plans_are_rejected(backend, tmp_path):
    function, plan = prepared(tmp_path)
    memory = plan_memory(plan, function.parameters, acquisition_policy="pooled")
    render = emit if backend == "cuda" else append_cpu_sessions
    with pytest.raises(ValueError, match="Explicit sessions require dedicated allocations"):
        render(function, plan, memory=memory)


def test_invalid_ordinary_plan_is_rejected_before_emission(tmp_path):
    function, plan = prepared(tmp_path)
    memory = plan_memory(plan, function.parameters)
    with pytest.raises(ValueError, match="Memory destruction"):
        emit(function, plan, ordinary_memory=replace(memory, destroy=()))


def test_completed_release_follows_planned_sync_and_precedes_slot_reset(tmp_path):
    function, plan = prepared(tmp_path)
    source = emit(function, plan)
    body = helper(source, lifecycle_name(function, "destroy"))
    assert body.index("storage::synchronize();") < body.index(".release_completed();") < body.index(".reset();")
    assert "update_host" not in body


@pytest.mark.parametrize("device", [False, True])
def test_renderer_refuses_unknown_or_misplaced_acquisition_policy(device):
    for operation in (
        MemoryOperation("acquire", acquisition_policy="unknown"),
        MemoryOperation("sync", acquisition_policy="pooled"),
    ):
        with pytest.raises(ValueError, match="acquisition policy"):
            render_memory((operation,), device=device)


@pytest.mark.parametrize("entry", ["entry", "long_entry_" + "x" * 52])
def test_trim_name_respects_collisions_limits_and_existing_names(tmp_path, entry):
    function, _ = prepared(tmp_path)
    function = replace(function, name=entry)
    original = session_names(function)
    parameter = replace(function.parameters[-1], name=original.trim_cache)
    collided = session_names(replace(function, parameters=(*function.parameters[:-1], parameter)))
    assert collided.trim_cache != original.trim_cache
    assert len(collided.trim_cache) <= 63
    for field in ("workspace", "create", "run", "update_device", "update_host", "destroy"):
        assert getattr(collided, field) == getattr(original, field)
    assert len(set(collided.__dict__.values())) == len(collided.__dict__)


def test_trim_api_is_shared_by_fortran_and_both_backends(tmp_path):
    function, plan = prepared(tmp_path)
    generated = generate_sources(function, plan)
    name = session_names(function).trim_cache
    assert f"public :: {name}" in generated.fortran
    assert f"subroutine {name}()" in generated.fortran
    assert f"bind(C, name='cpp_{name}')" in generated.fortran
    for source, profiled in ((generated.cuda, True), (generated.cpp, False)):
        body = source.split(f'extern "C" void cpp_{name}(', 1)[1].split("\n}\n", 1)[0]
        assert "storage::trim_cache();" in body
        assert ("ProfiledCallGuard" in body) is profiled


@pytest.mark.native
@pytest.mark.parametrize("override", [False, True], ids=["legacy-plan", "ordinary-override"])
def test_custom_planned_syncs_execute_in_every_ordinary_phase(tmp_path, monkeypatch, override):
    def sources(function, plan):
        generated = generate_sources(function, plan)
        memory = with_extra_syncs(plan_memory(plan, function.parameters))
        kwargs = {"ordinary_memory" if override else "memory": memory}
        return replace(generated, cuda=emit(function, plan, **kwargs))

    monkeypatch.setattr(calls, "generate_sources", sources)
    source = """! kernels
module pool_case
contains
! kernel
subroutine entry(a,n)
integer,intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=a(i)+1
enddo
end subroutine
end module
"""
    calls.compile_simulated(
        tmp_path,
        source,
        """#include <cassert>
int main() {
    int a[] = {4, 7};
    generated_kernels::cpp_entry(a, 2, 2);
    assert(a[0] == 5 && a[1] == 8);
    assert(fort_test_syncs == 6);
    assert(fort_test_uploads == 1 && fort_test_downloads == 1);
    assert(fort_test_allocations == 1 && fort_test_frees == 1);
}
""",
    )


@pytest.mark.native
def test_fortran_trim_links_with_cpu_and_preserves_owned_sessions(tmp_path):
    function, plan = prepared(tmp_path)
    generated = generate_sources(function, plan)
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "impl.cpp").write_text(generated.cpp)
    (tmp_path / "interface.f90").write_text(generated.fortran)
    (tmp_path / "driver.f90").write_text("""program main
use language_case
implicit none
type(entry_workspace)::work
real(knd)::a(2)
call entry_trim_cache()
a=[4.d0,7.d0]
call entry(a,2)
call entry_create(work,a)
call entry_trim_cache()
call entry_run(work,2)
call entry_update_host(work,a=a)
if(any(a/=[6.d0,9.d0])) stop 1
call entry_destroy(work)
call entry_trim_cache()
call entry_trim_cache()
end program
""")
    _run([_tool("g++"), "-std=c++17", "-c", "impl.cpp", "-o", "impl.o"], tmp_path)
    _run([_tool("gfortran"), "interface.f90", "driver.f90", "impl.o", "-lstdc++", "-o", "run"], tmp_path)
    _run(["./run"], tmp_path)
