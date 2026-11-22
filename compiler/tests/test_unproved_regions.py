"""An unproved region is a proof failure with its reason; invalid source is a semantic error."""

import pytest

from compiler.analysis import ParallelizationError, build_execution_plan
from compiler.ir import CompilationError
from compiler.tests.test_language import lower


def test_conflict_is_rejected_with_a_witness(tmp_path):
    function = lower(tmp_path, "do i=1,n\na(i)=1\nenddo\ndo i=2,n\na(i)=a(i-1)+1\nenddo\ndo i=1,n\na(i)=a(i)*2\nenddo")
    with pytest.raises(ParallelizationError, match="RAW") as error:
        build_execution_plan(function)
    assert error.value.failure.witness
    assert not error.value.failure.conservative


def test_initialized_recurrence_with_scalar_liveout_is_a_proof_failure(tmp_path):
    function = lower(tmp_path, "t=0\ndo i=1,n\nt=t+a(i)\nenddo\na(1)=t", "real(knd)::t")
    with pytest.raises(ParallelizationError):
        build_execution_plan(function)


def test_uninitialized_recurrence_is_a_semantic_error(tmp_path):
    function = lower(tmp_path, "do i=1,n\nt=t+a(i)\nenddo", "real(knd)::t")
    with pytest.raises(CompilationError, match="read before") as error:
        build_execution_plan(function)
    assert not isinstance(error.value, ParallelizationError)


def test_possibly_empty_loop_does_not_define_liveout(tmp_path):
    function = lower(tmp_path, "do i=1,n\nt=1\nenddo\na(1)=t", "real(knd)::t")
    with pytest.raises(CompilationError, match="read before definition") as error:
        build_execution_plan(function)
    assert not isinstance(error.value, ParallelizationError)


def test_known_nonempty_loop_defines_liveout_and_is_a_proof_failure(tmp_path):
    function = lower(tmp_path, "do i=1,3\nt=1\nenddo\na(1)=t", "real(knd)::t")
    with pytest.raises(ParallelizationError):
        build_execution_plan(function)


def test_zero_stride_expression_is_semantic_error(tmp_path):
    function = lower(tmp_path, "do i=1,n,2-2\na(i)=1\nenddo")
    with pytest.raises(CompilationError, match="stride cannot be zero") as error:
        build_execution_plan(function)
    assert not isinstance(error.value, ParallelizationError)


def test_conservative_nonaffine_proof_is_reported(tmp_path):
    function = lower(
        tmp_path,
        "do i=1,n\na(i)=src(index(i))\nenddo",
        "real(knd),intent(in)::src(:)\ninteger,intent(in)::index(:)",
        "a,n,src,index",
    )
    assert build_execution_plan(function).regions[0].report.conservative


def test_internal_failures_are_not_proof_failures(tmp_path, monkeypatch):
    from compiler.analysis import planning

    function = lower(tmp_path, "do i=1,n\na(i)=i\nenddo")

    def fail(*args, **kwargs):
        raise RuntimeError("internal proof failure")

    monkeypatch.setattr(planning, "prove_region", fail)
    with pytest.raises(RuntimeError, match="internal proof failure"):
        build_execution_plan(function)
