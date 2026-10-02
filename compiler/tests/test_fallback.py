"""Host fallback is restricted to semantically valid, unproved parallel regions."""

import re
from dataclasses import replace

import pytest

from compiler.analysis import ParallelizationError, build_execution_plan
from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError, ParallelRegion, SequentialRegion
from compiler.tests.test_language import compile_cuda_sources, lower, run_reference_and_cpp
from compiler.tests.test_native import cuda_device as cuda_device

HOST = CompilerOptions(fallback="host")


def test_conflict_fallback_has_structured_reason_and_neighboring_parallel_regions(tmp_path):
    function = lower(tmp_path, "do i=1,n\na(i)=1\nenddo\ndo i=2,n\na(i)=a(i-1)+1\nenddo\ndo i=1,n\na(i)=a(i)*2\nenddo")
    with pytest.raises(ParallelizationError) as error:
        build_execution_plan(function)
    assert error.value.failure.witness
    assert not error.value.failure.conservative
    plan = build_execution_plan(function, options=HOST)
    assert tuple(type(step) for step in plan.steps) == (ParallelRegion, SequentialRegion, ParallelRegion)
    assert "RAW" in plan.steps[1].reason
    assert [symbol.name for symbol in plan.steps[1].read_symbols] == ["a"]
    assert [symbol.name for symbol in plan.steps[1].write_symbols] == ["a"]


def test_initialized_recurrence_and_scalar_liveout_use_fallback(tmp_path):
    function = lower(tmp_path, "t=0\ndo i=1,n\nt=t+a(i)\nenddo\na(1)=t", "real(knd)::t")
    assert isinstance(build_execution_plan(function, options=HOST).steps[1], SequentialRegion)


def test_uninitialized_recurrence_never_becomes_fallback(tmp_path):
    function = lower(tmp_path, "do i=1,n\nt=t+a(i)\nenddo", "real(knd)::t")
    with pytest.raises(CompilationError, match="read before") as error:
        build_execution_plan(function, options=HOST)
    assert not isinstance(error.value, ParallelizationError)


def test_possibly_empty_loop_does_not_define_liveout(tmp_path):
    function = lower(tmp_path, "do i=1,n\nt=1\nenddo\na(1)=t", "real(knd)::t")
    with pytest.raises(CompilationError, match="read before definition") as error:
        build_execution_plan(function, options=HOST)
    assert not isinstance(error.value, ParallelizationError)


def test_known_nonempty_loop_may_define_liveout(tmp_path):
    function = lower(tmp_path, "do i=1,3\nt=1\nenddo\na(1)=t", "real(knd)::t")
    assert isinstance(build_execution_plan(function, options=HOST).steps[0], SequentialRegion)


def test_zero_stride_expression_is_semantic_error(tmp_path):
    function = lower(tmp_path, "do i=1,n,2-2\na(i)=1\nenddo")
    with pytest.raises(CompilationError, match="stride cannot be zero") as error:
        build_execution_plan(function, options=HOST)
    assert not isinstance(error.value, ParallelizationError)


def test_conservative_nonaffine_proof_is_reported(tmp_path):
    function = lower(
        tmp_path,
        "do i=1,n\na(i)=src(index(i))\nenddo",
        "real(knd),intent(in)::src(:)\ninteger,intent(in)::index(:)",
        "a,n,src,index",
    )
    assert build_execution_plan(function).regions[0].report.conservative


def test_fallback_invalidates_previous_scalar_substitutions(tmp_path):
    function = lower(
        tmp_path,
        "k=1\ndo i=1,n\nk=k+1\nenddo\ndo i=1,k\na(i)=i\nenddo",
        "integer::k",
    )
    plan = build_execution_plan(function, options=HOST)
    assert isinstance(plan.steps[1], SequentialRegion)
    k = next(symbol for symbol in function.symbols if symbol.name == "k")
    assert re.search(rf"h\d+_{k.id}\b", plan.regions[0].report.domain)


def test_internal_failures_do_not_trigger_fallback(tmp_path, monkeypatch):
    from compiler.analysis import planning

    function = lower(tmp_path, "do i=1,n\na(i)=i\nenddo")

    def fail(*args, **kwargs):
        raise RuntimeError("internal proof failure")

    monkeypatch.setattr(planning, "prove_region", fail)
    with pytest.raises(RuntimeError, match="internal proof failure"):
        build_execution_plan(function, options=HOST)


@pytest.mark.native
@pytest.mark.parametrize("backend", ["serial", "openmp", "cuda-simulated"])
def test_runtime_strides_empty_nests_and_partial_fallback_writes(tmp_path, monkeypatch, backend):
    from compiler.driver.pipeline import prepare_function
    from compiler.emission import generate_sources
    from compiler.tests import test_language as language
    from compiler.tests.test_language_cuda import CUDA_RUNTIME

    function = lower(
        tmp_path,
        "do i=1,n\na(i)=a(i)+1\nenddo\nt=0\n"
        "do i=lo,hi,stride\ndo j=1,1,inner_stride\nt=t+a(i)\na(i)=t\nenddo\nenddo\n"
        "a(1)=a(1)+i\ndo i=1,n\na(i)=abs(a(i))*2\nenddo",
        "real(knd)::t\ninteger::j\ninteger,intent(in)::lo,hi,stride,inner_stride",
        "a,n,lo,hi,stride,inner_stride",
    )
    optimized, plan = prepare_function(function, options=HOST)
    assert len(plan.regions) == 2
    assert sum(isinstance(step, SequentialRegion) for step in plan.steps) == 1
    driver = """program main
use language_case
real(knd)::a(6)
a=[1.d0,2.d0,3.d0,4.d0,55.d0,66.d0]
call entry(a,4,4,1,-1,1)
print *,a
call entry(a,4,1,4,2,1)
print *,a
call entry(a,4,4,1,1,0)
print *,a
end program
"""
    if backend == "cuda-simulated":
        (tmp_path / "cuda_runtime.h").write_text(CUDA_RUNTIME)

        def simulated_sources(function, plan):
            sources = generate_sources(function, plan)
            source = sources.cuda.replace("#include <cuda_runtime.h>", '#include "cuda_runtime.h"')
            source, launches = re.subn(
                r"(kernel_region_\d+_device)<<<([^>]+)>>>\(", r"fort_test_launch(\1, \2, ", source
            )
            assert launches == len(plan.regions)
            return replace(sources, cpp=source)

        monkeypatch.setattr(language, "generate_sources", simulated_sources)
    run_reference_and_cpp(tmp_path, optimized, plan, driver, openmp=backend == "openmp")


@pytest.mark.parametrize("valid_source", [False, True])
def test_fallback_cli_publishes_only_valid_source(tmp_path, valid_source):
    import subprocess
    import sys
    from pathlib import Path

    from compiler.tests.test_native import OUTPUT_FILES

    function = mixed_case(tmp_path)[0] if valid_source else lower(tmp_path, "do i=1,n\nt=t+a(i)\nenddo", "real(knd)::t")
    output = tmp_path / "output"
    output.mkdir()
    previous = {name: f"original {name}" for name in OUTPUT_FILES}
    for name, contents in previous.items():
        (output / name).write_text(contents)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            function.source,
            "--kernel",
            "entry",
            "--output-dir",
            str(output),
            "--fallback",
            "host",
            "--verbose",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if valid_source:
        assert result.returncode == 0, result.stderr
        assert "sequential host fallback" in result.stdout
        assert (output / "generated_code.cu").read_text().count("__global__ void") == 2
    else:
        assert result.returncode != 0
        assert "read before" in result.stderr
        assert {path.name: path.read_text() for path in output.iterdir()} == previous


def mixed_case(tmp_path):
    function = lower(
        tmp_path,
        "do i=1,n\na(i)=a(i)+1\nenddo\nt=0\ndo i=n,1,-1\nt=t+a(i)\na(i)=t\n"
        "enddo\na(1)=a(1)+i\ndo i=1,n\na(i)=a(i)*2\nenddo",
        "real(knd)::t",
    )
    driver = """program main
use language_case
real(knd)::a(4)
a=[1.d0,2.d0,3.d0,4.d0]
call entry(a,4)
print *,a
call entry(a,3)
print *,a
a=7
call entry(a,0)
print *,a
end program
"""
    return function, driver


@pytest.mark.native
@pytest.mark.parametrize("openmp", [False, True])
def test_fallback_signed_stride_liveout_and_following_parallel_loop(tmp_path, openmp):
    function, driver = mixed_case(tmp_path)
    run_reference_and_cpp(
        tmp_path,
        function,
        build_execution_plan(function, options=HOST),
        driver,
        openmp=openmp,
    )


@pytest.mark.cuda
def test_mixed_parallel_fallback_plan_compiles_for_cuda(tmp_path):
    function, _ = mixed_case(tmp_path)
    compile_cuda_sources(tmp_path, function, build_execution_plan(function, options=HOST))


@pytest.mark.native
@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_device")
def test_mixed_parallel_fallback_matches_fortran_on_cuda(tmp_path):
    function, driver = mixed_case(tmp_path)
    run_reference_and_cpp(
        tmp_path,
        function,
        build_execution_plan(function, options=HOST),
        driver,
        backend="cuda",
    )


def test_memory_plan_preserves_host_fallback_and_partial_write_coherence(tmp_path):
    from compiler.memory import plan_memory

    function, _ = mixed_case(tmp_path)
    plan = plan_memory(build_execution_plan(function, options=HOST), function.parameters)
    kinds = [operation.kind for operation in plan.run]
    assert kinds == [
        "device",
        "execute",
        "device_write",
        "execute",
        "host",
        "execute",
        "host_write",
        "host",
        "execute",
        "host_write",
        "device",
        "execute",
        "device_write",
    ]
    assert all(operation.symbols == (function.parameters[0],) for operation in plan.run if operation.symbols)
