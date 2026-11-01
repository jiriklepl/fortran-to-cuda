"""Fixed local vector companions grant only original native descriptor evidence."""
from copy import copy

import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_metadata_arrays import fixture, prove


def setup(tmp_path, *, formal="real(8),intent(in)::vector(3)",
          local="real(8)::local_vector(3)", actual="local_vector", body="value=vector(1)",
          second_formal="", second_actual="", extra="", update_arguments="a"):
    call = "call scalar(marks(i)%weight," + actual + second_actual + ")"
    path, _, selection = fixture(tmp_path, action=call)
    source = path.read_text().replace("subroutine update(a)",
        "subroutine scalar(value,vector" + (",other" if second_formal else "") + ")\n"
        "real(8),intent(inout)::value\n" + formal + "\n" + second_formal + "\n" + body +
        "\nend subroutine\nsubroutine update(" + update_arguments + ")", 1)
    source = source.replace("integer::i,n\nn=size(marks)", "integer::i,n\n" + local + "\n" + extra + "\nn=size(marks)")
    path.write_text(source)
    return path, SourceEffects([path]), selection


@pytest.mark.parametrize(("formal_bounds", "actual_bounds", "expected"), [
    ("3", "3", ([1, 3], [1, 3], 3)),
    ("-1:1", "4:6", ([-1, 1], [4, 6], 3)),
    ("0:-1", "2:1", ([0, -1], [2, 1], 0)),
])
def test_exact_fixed_vector_extent_keeps_original_call_and_no_payload_authority(
        tmp_path, formal_bounds, actual_bounds, expected):
    path, analysis, selection = setup(tmp_path,
        formal=f"real(8),intent(in)::vector({formal_bounds})", local=f"real(8)::local_vector({actual_bounds})")
    before = path.read_bytes()
    record = prove(analysis, selection).public()
    call, = record["native_metadata_call_descriptor_requirements"]
    vector, = call["fixed_vector_arguments"]
    assert (vector["formal_bounds"], vector["actual_bounds"], vector["extent"]) == expected
    assert vector["actual_binding"]["resource"].endswith("::local_vector")
    assert vector["descriptor_evidence_only"]
    for item in (record, call, vector):
        assert not item["execution_authorized"]
        assert not item["entry_definition_authority"]
    assert not vector["payload_access_authorized"]
    assert call["whole_original_closure_required"]
    assert "metadata_steps::scalar" in record["source_call_closure"]
    assert path.read_bytes() == before


def test_two_distinct_original_vectors_and_parameter_bounds_are_checked(tmp_path):
    _path, analysis, selection = setup(tmp_path,
        formal="integer,parameter::width=3\nreal(8),intent(in)::vector(width)",
        local="integer,parameter::length=3\nreal(8)::local_vector(length),second(0:2)",
        second_formal="real(8),intent(in)::other(3)", second_actual=",second",
        body="value=sum(vector)+sum(other)")
    call, = prove(analysis, selection).public()["native_metadata_call_descriptor_requirements"]
    assert [vector["formal"] for vector in call["fixed_vector_arguments"]] == ["vector", "other"]
    assert [vector["extent"] for vector in call["fixed_vector_arguments"]] == [3, 3]


@pytest.mark.parametrize("formal", [
    "real(8),intent(in)::vector(:)", "real(8),intent(in)::vector(*)",
    "real(8),intent(in)::vector(4)", "real(8),intent(in)::vector(2,2)",
    "real(8),optional,intent(in)::vector(3)", "real(8),pointer,intent(in)::vector(:)",
    "real(8),allocatable,intent(in)::vector(:)", "real(8),target,intent(in)::vector(3)",
])
def test_dynamic_nonconforming_or_aliasing_formals_remain_unsupported(tmp_path, formal):
    _path, analysis, selection = setup(tmp_path, formal=formal)
    with pytest.raises(CompilationError):
        prove(analysis, selection)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize(("local", "actual", "extra"), [
    ("integer::extent\nreal(8)::local_vector(extent)", "local_vector", ""),
    ("real(8),allocatable::local_vector(:)", "local_vector", ""),
    ("real(8),pointer::local_vector(:)", "local_vector", ""),
    ("real(8),target::local_vector(3)", "local_vector", ""),
    ("real(8)::local_vector(3)", "local_vector(:)", ""),
    ("real(8)::local_vector(3)", "[1.0_8,2.0_8,3.0_8]", ""),
    ("real(8)::local_vector(3),alias(3)", "local_vector", "equivalence(local_vector,alias)"),
    ("real(8)::local_vector(3)", "local_vector", "common /external/ local_vector"),
])
def test_unproved_actual_shape_association_or_storage_remains_unsupported(tmp_path, local, actual, extra):
    _path, analysis, selection = setup(tmp_path, local=local, actual=actual, extra=extra)
    with pytest.raises(CompilationError):
        prove(analysis, selection)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize(("formal", "actual"), [
    ("real(8),intent(in)::other(3)", ",local_vector"),
    ("real(8),intent(in)::other", ",local_vector(1)"),
])
def test_duplicate_or_scalar_vector_overlap_cannot_gain_alias_authority(tmp_path, formal, actual):
    _path, analysis, selection = setup(tmp_path, second_formal=formal, second_actual=actual)
    with pytest.raises(CompilationError, match="overlaps another argument"):
        prove(analysis, selection)


def test_original_dummy_array_is_not_assumed_independent_local_storage(tmp_path):
    path, _, selection = setup(tmp_path, update_arguments="a,local_vector")
    path.write_text(path.read_text().replace("call update(a)",
        "real(8)::forwarded_vector(3)\ncall update(a,forwarded_vector)"))
    analysis = SourceEffects([path])
    with pytest.raises(CompilationError, match="independent fixed local storage"):
        prove(analysis, selection)


@pytest.mark.parametrize("dtype", ["real(4)", "integer", "logical"])
def test_other_fixed_numeric_vectors_remain_descriptor_only(tmp_path, dtype):
    _path, analysis, selection = setup(tmp_path,
        formal=dtype + ",intent(inout)::vector(3)", local=dtype + "::local_vector(3)",
        body="vector=vector")
    record = prove(analysis, selection).public()
    vector, = record["native_metadata_call_descriptor_requirements"][0]["fixed_vector_arguments"]
    assert vector["extent"] == 3
    assert not vector["entry_definition_authority"]
    assert not vector["payload_access_authorized"]


@pytest.mark.parametrize("body", ["call absent(vector)", "allocate(scratch(3))", "deallocate(scratch)"])
def test_fixed_shapes_do_not_skip_original_callee_lifetime_and_unknown_call_checks(tmp_path, body):
    _path, analysis, selection = setup(tmp_path,
        formal="real(8),intent(in)::vector(3)\nreal(8),allocatable::scratch(:)", body=body)
    with pytest.raises(CompilationError):
        prove(analysis, selection)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_next_validation_rejects_replaced_original_vector_binding(tmp_path):
    _path, analysis, selection = setup(tmp_path)
    proof = prove(analysis, selection)
    scope = analysis.routines["metadata_steps::update"].scope
    scope.bindings["local_vector"] = copy(scope.bindings["local_vector"])
    with pytest.raises(CompilationError):
        proof.validate(analysis, "metadata_steps::advance", selection)
