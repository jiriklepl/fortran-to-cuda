"""Source helper completion grants outlining authority, never native effects."""

from dataclasses import replace

import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_inline_source_regions import original_joined_group
from compiler.tests.test_source_numerical_closures import SOURCE


PROCEDURE = "renamed_numerics::advance"


def fixture(tmp_path, source=SOURCE):
    source = source.replace("do block=-3,n,tile", "i=0\n!$omp parallel do private(g,s,i,block) schedule(runtime)\ndo block=-3,n,tile")
    source = source.replace("enddo\ncontains", "enddo\n!$omp end parallel do\ncontains")
    path = tmp_path / "closure.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    return path, analysis, original_joined_group(analysis.routines[PROCEDURE])


def test_pure_private_outputs_and_lexical_reads_have_distinct_completion_authority(tmp_path):
    _path, analysis, group = fixture(tmp_path)
    proof = analysis.numerical_joined_completion(PROCEDURE, group)
    record = proof.public()
    assert record["available"] and record["requires_numerical_lowering"]
    assert record["proof_role"] == "numerical_source_completion"
    assert not record["native_effects_authority"] and not record["gpu_legality_established"]
    assert {item["procedure"] for item in record["helper_closure"]} == {
        PROCEDURE + "::derivative", PROCEDURE + "::invariant"}
    assert {PROCEDURE + "::g", PROCEDURE + "::s"}.issubset(proof.private_roots)
    assert proof.validate(analysis) is proof
    with pytest.raises(CompilationError, match="source-call proof"):
        analysis.joined_completion(PROCEDURE, group)
    with pytest.raises(CompilationError, match="registered original source proof token"):
        analysis.native_sections_for_nodes(PROCEDURE, group, completion=proof)


def test_module_function_closure_is_bounded_and_keeps_original_result_owner(tmp_path):
    source = SOURCE.replace("s=dot_product(g,g)", "s=quadratic(g)")
    source = source.replace("end module", """pure function quadratic(values) result(value)
real(8),intent(in)::values(2)
real(8)::value
value=dot_product(values,values)
end function
end module""")
    _path, analysis, group = fixture(tmp_path, source)
    record = analysis.numerical_joined_completion(PROCEDURE, group).public()
    assert "renamed_numerics::quadratic" in {item["procedure"] for item in record["helper_closure"]}


@pytest.mark.parametrize("before,after,reason", [
    ("pure subroutine derivative", "subroutine derivative", "explicit PURE"),
    ("g(1)=a(i)", "call missing(g)\ng(1)=a(i)", "unresolved synchronous source helper"),
    ("g(1)=a(i)", "call derivative(g,i)\ng(1)=a(i)", "recursive"),
    ("g(1)=a(i)", "a(i)=1\ng(1)=a(i)", "nonlocal or unstable"),
    ("g(1)=a(i)", "print *,i\ng(1)=a(i)", "unsupported I/O"),
    ("g(1)=a(i)", "!$omp task\ng(1)=a(i)\n!$omp end task", "OpenMP work"),
    ("integer,intent(in)::i\ng(1)", "integer,intent(in)::i\nreal(8),allocatable::scratch(:)\nallocate(scratch(2))\ng(1)", "lifetime effects"),
])
def test_pure_alone_cannot_prove_transitive_synchronous_completion(tmp_path, before, after, reason):
    _path, analysis, group = fixture(tmp_path, SOURCE.replace(before, after))
    with pytest.raises(CompilationError, match=reason):
        analysis.numerical_joined_completion(PROCEDURE, group)


def test_numerical_completion_copies_foreign_analyses_and_stale_sources_have_no_authority(tmp_path):
    path, analysis, group = fixture(tmp_path)
    proof = analysis.numerical_joined_completion(PROCEDURE, group)
    with pytest.raises(CompilationError, match="registered original source authority"):
        replace(proof).validate(analysis)
    independent = SourceEffects([path])
    with pytest.raises(CompilationError, match="registered original source authority"):
        proof.validate(independent)
    path.write_text(path.read_text().replace("g(1)=a(i)", "g(1)=2*a(i)"))
    with pytest.raises(CompilationError, match="changed"):
        proof.validate(analysis)


def test_repeated_helper_sites_respect_the_fixed_call_generation_budget(tmp_path):
    source = SOURCE.replace("call invariant(s,g)", "\n".join(["call invariant(s,g)"] * 129))
    _path, analysis, group = fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="call budget"):
        analysis.numerical_joined_completion(PROCEDURE, group)


def test_independent_module_function_metadata_mutation_invalidates_completion(tmp_path):
    source = SOURCE.replace("s=dot_product(g,g)", "s=quadratic(g)")
    source = source.replace("pure subroutine invariant(s,g)",
                            "pure subroutine invariant(s,g)\nuse other_numerics,only:quadratic")
    source += """module other_numerics
implicit none
contains
pure function quadratic(values) result(value)
real(8),intent(in)::values(2)
real(8)::value
value=dot_product(values,values)
end function
end module
"""
    _path, analysis, group = fixture(tmp_path, source)
    proof = analysis.numerical_joined_completion(PROCEDURE, group)
    helper = analysis.numerical_helpers["other_numerics::quadratic"]
    # A function's semantic bindings are outside the original owning routine;
    # unchanged source text alone cannot grant stale helper authority.
    binding = helper.scope.bindings["value"]
    helper.scope.bindings["value"] = replace(binding, attributes=binding.attributes | {"volatile"})
    with pytest.raises(CompilationError, match="unchanged original source-backed helper authority"):
        proof.validate(analysis)
