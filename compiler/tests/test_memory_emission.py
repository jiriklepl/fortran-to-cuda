"""Check that lifecycle and execution decisions really come from memory plans."""

from dataclasses import replace

import pytest

from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.emission.c.generator import generate_cpp
from compiler.emission.c.sessions import append_cpu_sessions
from compiler.emission.common.abi import abi_arguments
from compiler.emission.common.memory import render_memory
from compiler.emission.cuda.generator import generate_cuda
from compiler.ir import HostBlock, Literal, ScalarType
from compiler.memory import MemoryOperation, plan_memory
from compiler.tests import test_call_ownership as calls
from compiler.tests.test_language import lower
from compiler.tests.test_memory_runtime import _run, _tool


@pytest.mark.native
def test_inserted_sync_is_executed_by_generated_cuda(tmp_path, monkeypatch):
    def sources(function, plan):
        generated = generate_sources(function, plan)
        memory = plan_memory(plan, function.parameters)
        memory = replace(memory, run=(MemoryOperation("sync"), *memory.run))
        cuda = generate_cuda(function, plan, abi_arguments(function.parameters), "common_functions.cuh", memory=memory)
        assert cuda != generated.cuda
        return replace(generated, cuda=cuda)

    monkeypatch.setattr(calls, "generate_sources", sources)
    source = """! kernels
module memory_case
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
    driver = """#include <cassert>
int main() {
    int a[] = {1, 2};
    generated_kernels::cpp_entry(a, 2, 2);
    assert(a[0] == 2 && a[1] == 3);
    assert(fort_test_syncs == 3);
    assert(fort_test_allocations == 1 && fort_test_frees == 1);
}
"""
    calls.compile_simulated(tmp_path, source, driver)


@pytest.mark.native
def test_cpu_session_executes_memory_plan_instead_of_ordinary_entry(tmp_path):
    function, plan = prepare_function(lower(tmp_path, "a(1)=1"))
    memory = plan_memory(plan, function.parameters)
    operations = []
    for operation in memory.run:
        if operation.kind == "execute":
            step = operation.step
            assert isinstance(step, HostBlock)
            assignment = replace(step.assignments[0], value=Literal("42", ScalarType.INTEGER))
            operation = replace(operation, step=replace(step, assignments=(assignment,)))
        operations.append(operation)
    memory = replace(memory, run=tuple(operations))
    abi = abi_arguments(function.parameters)
    source = generate_cpp(function, plan, abi, "common_functions.cuh")
    source += append_cpu_sessions(function, plan, memory=memory)
    source += """#include <cassert>
int main() {
    double a[] = {0};
    generated_kernels::cpp_entry(a, 1, 1);
    assert(a[0] == 1);
    const auto work = generated_kernels::cpp_entry_create(a, 1);
    generated_kernels::cpp_entry_run(work, 1);
    generated_kernels::cpp_entry_update_host_0(work, a, 1);
    assert(a[0] == 42);
    generated_kernels::cpp_entry_destroy(work);
}
"""
    (tmp_path / "test.cpp").write_text(source)
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    _run([_tool("g++"), "-std=c++17", "test.cpp", "-o", "run"], tmp_path)
    _run(["./run"], tmp_path)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("damage", ["missing_create", "missing_destroy", "unknown_operation"])
def test_emitters_reject_invalid_lifecycles_before_emitting(tmp_path, backend, damage):
    function, plan = prepare_function(lower(tmp_path, "a(1)=1"))
    memory = plan_memory(plan, function.parameters)
    if damage == "missing_create":
        memory = replace(memory, create=())
    elif damage == "missing_destroy":
        memory = replace(memory, destroy=())
    else:
        memory = replace(memory, run=(MemoryOperation("unimplemented"), *memory.run))

    def emit():
        if backend == "cpu":
            append_cpu_sessions(function, plan, memory=memory)
        else:
            generate_cuda(function, plan, abi_arguments(function.parameters), "common_functions.cuh", memory=memory)

    with pytest.raises(ValueError, match="Memory creation|Memory destruction|Unknown memory operation"):
        emit()


def test_memory_renderer_rejects_unknown_operation_instead_of_skipping_it():
    with pytest.raises(ValueError, match="Unsupported memory operation"):
        render_memory((MemoryOperation("unimplemented"),), device=True)


@pytest.mark.native
def test_destroy_renders_sync_then_release_without_retrieval(tmp_path):
    source = """! kernels
module memory_case
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
    driver = """#include <cassert>
int main() {
    int a[] = {1, 2};
    const auto work = generated_kernels::cpp_entry_create(a, 2);
    assert(fort_test_allocations == 1 && fort_test_uploads == 1);
    generated_kernels::cpp_entry_run(work, 2);
    assert(fort_test_syncs == 0 && fort_test_downloads == 0);
    generated_kernels::cpp_entry_destroy(work);
    assert(fort_test_syncs == 1 && fort_test_frees == 1);
    assert(fort_test_downloads == 0);
    assert(a[0] == 1 && a[1] == 2);
    generated_kernels::cpp_entry_destroy(0);
    assert(fort_test_syncs == 1 && fort_test_frees == 1);
}
"""
    calls.compile_simulated(tmp_path, source, driver)
