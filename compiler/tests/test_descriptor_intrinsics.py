"""NORM2 storage reads retain exact native evaluation and intrinsic resolution."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_intrinsics import norm2_descriptor_effects
from compiler.ir import CompilationError
from compiler.ir.intrinsics import INTRINSICS
from compiler.tests.test_descriptor_window_proofs import fixture


def original(tmp_path, *, dtype="real(8)", shape="3", actual="v", dim=None,
             dimension="integer::axis", extra="", prefix="", result="real(8)::value"):
    expression = "norm2(" + actual + ("," + dim if dim is not None else "") + ")"
    _path, analysis, _graph, selected = fixture(tmp_path,
        child_decl=f"real(8),intent(inout)::a(:)\n{dtype}::v({shape})\n{dimension}\n{result}\n{prefix}",
        child=f"v=a(1)\nvalue={expression}\na(1)=a(1)+1", extra=extra)
    routine = analysis.routines["intervals::read_modify"]
    node = next(item for item in walk(routine.execution)
                if type(item).__name__ == "Intrinsic_Function_Reference" and str(item.items[0]).lower() == "norm2")
    return analysis, routine, node, selected


@pytest.mark.parametrize("dtype", ["real(4)", "real(8)"])
@pytest.mark.parametrize(("shape", "dim", "result_rank"), [("3", None, 0), ("3", "1", 0),
    ("3,2", "axis", 1), ("3,2", "dim=2", 1)])
def test_original_real_arrays_and_optional_integer_dim_have_readonly_effects(tmp_path, dtype, shape, dim, result_rank):
    analysis, routine, node, _selected = original(tmp_path, dtype=dtype, shape=shape, dim=dim)
    before = analysis.summarize(routine.qualified)
    actuals, record = norm2_descriptor_effects(analysis, routine, node)
    assert len(actuals) == (1 if dim is None else 2)
    assert record["result"]["rank"] == result_rank
    assert record["result"]["kind"] == int(dtype[-2])
    assert record["arguments"][0]["effect"] == "read"
    assert record["requires_original_execution"]
    assert record["all_actual_effects_require_original_source_traversal"]
    for flag in ("storage_writes", "allocation_or_escape", "execution_authorized", "capture_authorized",
                 "entry_definition_authority", "gpu_legality_established", "numerical_lowering_authorized", "motion_authorized"):
        assert record[flag] is False
    assert "norm2" not in INTRINSICS  # No GPU/numerical lowering registration.
    assert analysis.summarize(routine.qualified) == before
    assert not before["complete"]


def test_keyword_order_preserves_original_actuals_not_formal_reordering(tmp_path):
    analysis, routine, node, _selected = original(tmp_path, shape="3,2", actual="dim=axis,array=v")
    actuals, record = norm2_descriptor_effects(analysis, routine, node)
    assert [str(item) for item in actuals] == ["axis", "v"]
    assert [(item["formal"], item["keyword"]) for item in record["arguments"]] == [("dim", "dim"), ("array", "array")]


@pytest.mark.parametrize(("dtype", "actual", "dim", "dimension"), [
    ("integer(4)", "v", None, "integer::axis"),
    ("logical", "v", None, "integer::axis"),
    ("real(8)", "a(1)", None, "integer::axis"),
    ("real(8)", "1.0_8", None, "integer::axis"),
    ("real(8)", "v", "0", "integer::axis"),
    ("real(8)", "v", "2", "integer::axis"),
    ("real(8)", "v", "axis", "real(8)::axis"),
    ("real(8)", "v", "axis", "integer::axis(2)"),
    ("real(8)", "v", "absent()", "integer::axis"),
    ("real(8)", "v", "1,2", "integer::axis"),
    ("real(8)", "v", "array=v", "integer::axis"),
    ("real(8)", "v", "mask=1", "integer::axis"),
    ("real(8)", "array=v,1", None, "integer::axis"),
])
def test_bad_type_rank_dim_and_original_argument_association_reject(tmp_path, dtype, actual, dim, dimension):
    def classify():
        analysis, routine, node, _selected = original(tmp_path, dtype=dtype, actual=actual, dim=dim, dimension=dimension)
        return norm2_descriptor_effects(analysis, routine, node)

    with pytest.raises(CompilationError):
        classify()


def test_explicit_external_name_cannot_borrow_intrinsic_spelling(tmp_path):
    analysis, routine, node, _selected = original(tmp_path, prefix="external::norm2")
    with pytest.raises(CompilationError, match="shadowed"):
        norm2_descriptor_effects(analysis, routine, node)


def test_source_function_with_intrinsic_spelling_is_not_intrinsic_authority(tmp_path):
    analysis, routine, node, _selected = original(tmp_path,
        extra="real(8) function norm2(x)\nreal(8),intent(in)::x(3)\nnorm2=x(1)\nend function")
    with pytest.raises(CompilationError, match="resolution"):
        norm2_descriptor_effects(analysis, routine, node)


def test_unknown_use_exports_do_not_establish_intrinsic_identity(tmp_path):
    analysis, routine, node, _selected = original(tmp_path, prefix="use unavailable")
    with pytest.raises(CompilationError):
        norm2_descriptor_effects(analysis, routine, node)


def test_detached_or_replaced_original_expression_is_not_an_intrinsic_proof(tmp_path):
    analysis, routine, node, _selected = original(tmp_path)
    with pytest.raises(CompilationError, match="exact original"):
        norm2_descriptor_effects(analysis, routine, copy(node))
    with pytest.raises(CompilationError):
        norm2_descriptor_effects(analysis, copy(routine), node)


def test_full_descriptor_consumer_records_native_reads_without_completing_normal_summary(tmp_path):
    from compiler.tests.test_descriptor_window_proofs import ENTRY, prove

    analysis, routine, node, selected = original(tmp_path)
    before = analysis.summarize(routine.qualified)
    proof = prove(analysis, selected)
    item, = proof.public()["native_intrinsic_descriptor_requirements"]
    assert item["source_expression"] == str(node)
    assert item["intrinsic"] == "norm2"
    assert not proof.public()["entry_definition_authority"]
    assert proof.validate(analysis, ENTRY, selected) is proof
    assert analysis.summarize(routine.qualified) == before


def test_intrinsic_does_not_hide_unknown_dim_effects_or_later_deallocation(tmp_path):
    from compiler.tests.test_descriptor_window_proofs import prove

    _path, analysis, _graph, selected = fixture(tmp_path,
        child_decl="real(8),allocatable,intent(inout)::a(:)\nreal(8)::value,v(3)",
        child="v=a(1)\nvalue=norm2(v)\ndeallocate(a)")
    with pytest.raises(CompilationError):
        prove(analysis, selected)
    assert not getattr(analysis, "_descriptor_window_proofs", {})
