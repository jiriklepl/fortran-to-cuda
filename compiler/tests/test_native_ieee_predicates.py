"""Native predicates retain original IEEE resolution, data and source order."""

from copy import copy
from dataclasses import replace

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.native_environment import canonical_intrinsic_export, intrinsic_export
from compiler.frontend.native_intrinsics import native_intrinsic_export, prove_native_predicate
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "renamed_flow::advance"


def analyze(tmp_path, *, use="use, intrinsic :: ieee_arithmetic", declaration="real(8)::p",
            expression="ieee_is_nan(p)", specification="", helpers="", sources=(), result_shape=""):
    source = tmp_path / "consumer.f90"
    source.write_text(f"""module renamed_flow
{specification}
contains
subroutine advance()
{use}
implicit none
{declaration}
logical::result{result_shape}
result={expression}
end subroutine
{helpers}
end module
""")
    analysis = SourceEffects([*sources, source])
    routine = analysis.routines[ENTRY]
    assignment = next(node for node in walk(routine.execution) if type(node).__name__ == "Assignment_Stmt")
    return source, analysis, routine, assignment.items[2]


@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("keyword", [False, True])
def test_original_real_scalar_signature_and_actual_ordinal(tmp_path, precision, keyword):
    expression = "ieee_is_nan(x=p)" if keyword else "ieee_is_nan(p)"
    source, analysis, routine, node = analyze(tmp_path, declaration=f"real({precision})::p", expression=expression)
    original = source.read_bytes()
    proof = prove_native_predicate(analysis, ENTRY, node)
    assert proof.validate(analysis, ENTRY, node) is proof
    assert prove_native_predicate(analysis, ENTRY, node) is proof
    assert proof.intrinsic == "$intrinsic::ieee_arithmetic::ieee_is_nan"
    actual = node.items[1].items[0]
    if keyword:
        actual = actual.items[1]
    assert proof.actuals == (actual,)
    assert proof.actuals[0] is actual
    record = proof.public()
    assert record["arguments"] == [{"position": 1, "formal": "x", "keyword": "x" if keyword else None,
                                   "source_actual": "p", "type": "real", "kind": precision,
                                   "rank": 0, "effect": "read"}]
    assert record["native_only"]
    assert record["actual_effects_require_source_traversal"]
    assert not record["storage_writes"]
    assert not record["exception_observers_authorized"]
    assert not record["gpu_legality_established"]
    assert not record["numerical_lowering_authorized"]
    assert "original" in record["exception_behavior"]
    assert source.read_bytes() == original
    assert native_intrinsic_export(analysis, routine.scope, "ieee_is_nan") == proof.intrinsic
    # The old environment resolver and its reviewed exception flag set stay finite.
    assert intrinsic_export(analysis, routine.scope, "ieee_is_nan") is None
    assert canonical_intrinsic_export(proof.intrinsic) is None


@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("rank", [1, 2, 4])
def test_original_whole_real_array_input_is_elemental_native_read(tmp_path, precision, rank):
    shape = ",".join("2" for _ in range(rank))
    _path, analysis, _routine, node = analyze(tmp_path, declaration=f"real({precision})::p({shape})",
                                          result_shape=f"({shape})")
    proof = prove_native_predicate(analysis, ENTRY, node)
    assert proof.public()["arguments"][0]["rank"] == rank
    assert proof.public()["result"]["rank"] == rank
    assert proof.public()["result"]["kind"] == "original default logical"
    assert not proof.public()["internal_cuts_authorized"]


@pytest.mark.parametrize("expression", ["ieee_is_nan(-p)", "ieee_is_nan(p+p)", "ieee_is_nan(1.0_8)"])
def test_bounded_existing_scalar_signature_keeps_original_expression(tmp_path, expression):
    _path, analysis, _routine, node = analyze(tmp_path, expression=expression)
    proof = prove_native_predicate(analysis, ENTRY, node)
    assert proof.public()["arguments"][0]["rank"] == 0
    assert str(proof.actuals[0]) in str(node)


def test_renames_multifile_reexport_and_host_association(tmp_path):
    exports = tmp_path / "exports.f90"
    exports.write_text("""module predicate_exports
use, intrinsic :: ieee_arithmetic, only: classify=>ieee_is_nan
end module
module predicate_bridge
use predicate_exports, only: finite_test=>classify
end module
""")
    _path, analysis, routine, node = analyze(tmp_path, use="", specification="use predicate_bridge,only:probe=>finite_test",
                                          expression="probe(x=p)", sources=(exports,))
    assert native_intrinsic_export(analysis, routine.scope, "probe") == "$intrinsic::ieee_arithmetic::ieee_is_nan"
    assert prove_native_predicate(analysis, ENTRY, node).public()["arguments"][0]["keyword"] == "x"


def test_unrestricted_rename_excludes_original_export(tmp_path):
    _path, analysis, routine, node = analyze(tmp_path, use="use ieee_arithmetic,probe=>ieee_is_nan",
                                          expression="probe(p)")
    assert prove_native_predicate(analysis, ENTRY, node)
    assert native_intrinsic_export(analysis, routine.scope, "ieee_is_nan") is None


def test_second_unrestricted_import_restores_original_export(tmp_path):
    _path, analysis, routine, node = analyze(tmp_path,
        use="use ieee_arithmetic,probe=>ieee_is_nan\nuse ieee_arithmetic")
    assert prove_native_predicate(analysis, ENTRY, node)
    assert native_intrinsic_export(analysis, routine.scope, "probe") == native_intrinsic_export(
        analysis, routine.scope, "ieee_is_nan")


@pytest.mark.parametrize("use", [
    "use, non_intrinsic :: ieee_arithmetic", "use ieee_exceptions", "use ieee_features",
    "use ieee_arithmetic,only:ieee_is_finite", "use ieee_arithmetic\nuse unavailable",
    "use ieee_arithmetic,only:probe=>ieee_is_nan\nuse ieee_arithmetic,only:probe=>ieee_is_finite",
])
def test_unreviewed_or_ambiguous_imports_cannot_gain_authority(tmp_path, use):
    expression = "probe(p)" if "probe=>" in use else "ieee_is_nan(p)"
    _path, analysis, _routine, node = analyze(tmp_path, use=use, expression=expression)
    with pytest.raises(CompilationError, match="intrinsic resolution"):
        prove_native_predicate(analysis, ENTRY, node)


def test_source_module_named_like_intrinsic_remains_application_source(tmp_path):
    provider = tmp_path / "application.f90"
    provider.write_text("""module ieee_arithmetic
contains
logical function ieee_is_nan(x)
real(8),intent(in)::x
ieee_is_nan=x<0
end function
end module
""")
    _path, analysis, _routine, node = analyze(tmp_path, use="use ieee_arithmetic", sources=(provider,))
    with pytest.raises(CompilationError, match="intrinsic resolution"):
        prove_native_predicate(analysis, ENTRY, node)


@pytest.mark.parametrize("shadow", ["storage", "procedure", "generic"])
def test_local_shadow_cannot_gain_intrinsic_authority(tmp_path, shadow):
    declaration, helpers, specification, use = "real(8)::p", "", "", "use, intrinsic :: ieee_arithmetic"
    if shadow == "storage":
        declaration += "\nreal(8)::ieee_is_nan(2)"
    else:
        use = ""
        helpers = """logical function impostor(x)
real(8)::x
impostor=x<0
end function
"""
        if shadow == "procedure":
            helpers = helpers.replace("impostor", "ieee_is_nan")
        else:
            specification = "interface ieee_is_nan\nmodule procedure impostor\nend interface"
    _path, analysis, _routine, node = analyze(tmp_path, declaration=declaration, helpers=helpers,
                                          specification=specification, use=use)
    with pytest.raises(CompilationError, match="intrinsic resolution"):
        prove_native_predicate(analysis, ENTRY, node)


@pytest.mark.parametrize(("declaration", "expression"), [
    ("integer::p", "ieee_is_nan(p)"), ("logical::p", "ieee_is_nan(p)"),
    ("complex(8)::p", "ieee_is_nan(p)"), ("real(16)::p", "ieee_is_nan(p)"),
    ("real(8)::p(2)", "ieee_is_nan(p(1))"), ("real(8)::p(2)", "ieee_is_nan(p(:))"),
    ("real(8)::p", "ieee_is_nan(p,p)"), ("real(8)::p", "ieee_is_nan(y=p)"),
    ("real(8)::p", "ieee_is_nan(x=p,x=p)"),
])
def test_unproved_signature_or_actual_association_rejects(tmp_path, declaration, expression):
    _path, analysis, _routine, node = analyze(tmp_path, declaration=declaration, expression=expression)
    with pytest.raises(CompilationError, match="signature|actual|association"):
        prove_native_predicate(analysis, ENTRY, node)


def test_forged_copied_cross_analysis_and_stale_tokens_reject(tmp_path):
    path, analysis, _routine, node = analyze(tmp_path)
    proof = prove_native_predicate(analysis, ENTRY, node)
    for forged in (copy(proof), replace(proof), replace(proof, actuals=(F.Name("p"),))):
        with pytest.raises(CompilationError, match="registered original"):
            forged.validate(analysis, ENTRY, node)
    with pytest.raises(CompilationError, match="exact original"):
        prove_native_predicate(analysis, ENTRY, F.Part_Ref("ieee_is_nan(p)"))
    second = SourceEffects([path])
    with pytest.raises(CompilationError, match="registered original"):
        proof.validate(second, ENTRY, node)
    path.write_text(path.read_text().replace("real(8)::p", "real(4)::p"))
    with pytest.raises(CompilationError, match="source changed"):
        proof.validate(analysis, ENTRY, node)


def test_original_nested_guard_remains_with_source_caller(tmp_path):
    path, _analysis, _routine, _node = analyze(tmp_path)
    path.write_text(path.read_text().replace("result=ieee_is_nan(p)", "if (.false.) then\nresult=ieee_is_nan(p)\nend if"))
    analysis = SourceEffects([path])
    routine = analysis.routines[ENTRY]
    node = next(item.items[2] for item in walk(routine.execution) if type(item).__name__ == "Assignment_Stmt")
    proof = prove_native_predicate(analysis, ENTRY, node)
    assert str(proof.actuals[0]) == "p"
    assert "surrounding guards" in proof.public()["argument_evaluation"]
    assert not proof.public()["internal_cuts_authorized"]
    # Issuing classification does not evaluate a protected, possibly undefined P.
    assert "IF (.FALSE.)" in str(routine.execution)


def test_exception_flag_observer_and_environment_calls_remain_separate(tmp_path):
    _path, analysis, routine, node = analyze(tmp_path, expression="ieee_support_flag(p)")
    assert native_intrinsic_export(analysis, routine.scope, "ieee_get_flag") is None
    assert native_intrinsic_export(analysis, routine.scope, "ieee_set_halting_mode") is None
    with pytest.raises(CompilationError, match="intrinsic resolution"):
        prove_native_predicate(analysis, ENTRY, node)


def test_registered_predicate_count_is_bounded(tmp_path):
    path, _analysis, _routine, _node = analyze(tmp_path)
    path.write_text(path.read_text().replace("result=ieee_is_nan(p)", "result=ieee_is_nan(p)\nresult=ieee_is_nan(p)"))
    analysis = SourceEffects([path], operations=1)
    routine = analysis.routines[ENTRY]
    nodes = [item.items[2] for item in walk(routine.execution) if type(item).__name__ == "Assignment_Stmt"]
    first = prove_native_predicate(analysis, ENTRY, nodes[0])
    assert prove_native_predicate(analysis, ENTRY, nodes[0]) is first
    with pytest.raises(CompilationError, match="proof budget"):
        prove_native_predicate(analysis, ENTRY, nodes[1])
