"""Reached expression queries reuse the original storage projection engine."""
from dataclasses import replace

import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.structured_effects import storage_mutation_effects
from compiler.ir import CompilationError


def fixture(tmp_path, *, function_body="counter=counter+1\ncalculate=sum(x)",
            function_declarations="real(8),intent(in)::x(:)",
            body="a(:)=a(:)+calculate(a)", extra=""):
    path = tmp_path / "function.f90"
    path.write_text(f"""module functions
real(8)::counter
real(8),allocatable::hidden(:)
contains
subroutine caller(a)
real(8),allocatable,intent(inout)::a(:)
{body}
end subroutine
real(8) function calculate(x)
{function_declarations}
{function_body}
end function
{extra}
end module
""")
    source = SourceEffects([path])
    caller = source.routines["functions::caller"]
    expressions = tuple(node for node in walk(caller.execution)
        if type(node).__name__ == "Part_Ref" and str(node.items[0]).lower() == "calculate")
    return path, source, caller, expressions


def test_reached_function_query_allows_impure_payload_writes_without_native_authority(tmp_path):
    _path, source, caller, (expression,) = fixture(tmp_path)
    before = source.summarize(caller.qualified)
    proof = source.storage_mutation_effects(caller.qualified, expression=expression)
    assert proof.preserves_descriptors()
    assert proof.descriptor_unchanged("argument::a")
    public = proof.public()
    assert public["version"] == 2
    assert public["kind"] == "original_storage_function_mutation_effects"
    assert public["original_expression"]["source"] == "calculate(a)"
    assert {member["procedure"] for member in public["closure"]} == {"functions::calculate"}
    assert "independent exact native effects" in public["native_execution_requirement"]
    assert not public["payload_access_authorized"]
    assert not public["execution_authorized"]
    assert not source._native_functions
    assert source.summarize(caller.qualified) == before


def test_same_shared_service_preserves_original_whole_procedure_record(tmp_path):
    _path, source, caller, _expressions = fixture(tmp_path)
    direct = source.storage_mutation_effects(caller.qualified)
    assert storage_mutation_effects(source, caller.qualified) is direct
    assert direct.public()["version"] == 1
    assert "original_expression" not in direct.public()
    assert direct.preserves_descriptors()


def test_query_excludes_unreached_caller_tail_without_lending_whole_caller_authority(tmp_path):
    _path, source, caller, (expression,) = fixture(tmp_path, body="a(:)=calculate(a)\ndeallocate(a)")
    whole = source.storage_mutation_effects(caller.qualified)
    selected = source.storage_mutation_effects(caller.qualified, expression=expression)
    assert not whole.preserves_descriptors()
    assert selected.preserves_descriptors()
    assert all(event["kind"] != "deallocate" for event in selected.public()["mutations"])


@pytest.mark.parametrize(("declarations", "body"), [
    ("real(8),intent(in)::x(:)", "deallocate(hidden)\ncalculate=sum(x)"),
    ("real(8),allocatable,intent(inout)::x(:)", "x=1\ncalculate=sum(x)"),
    ("real(8),intent(in)::x(:)", "call unknown(x)\ncalculate=1"),
    ("real(8),target,intent(in)::x(:)", "calculate=sum(x)"),
    ("real(8),intent(in)::x(:)", "!$omp parallel\ncalculate=sum(x)\n!$omp end parallel"),
])
def test_hidden_lifetime_escape_and_unknown_closures_block_stability(tmp_path, declarations, body):
    _path, source, caller, (expression,) = fixture(tmp_path,
        function_declarations=declarations, function_body=body)
    proof = source.storage_mutation_effects(caller.qualified, expression=expression)
    assert not proof.preserves_descriptors()
    assert not proof.descriptor_unchanged("argument::a")


def test_actual_expression_evaluation_closure_is_not_omitted(tmp_path):
    _path, source, caller, (expression,) = fixture(tmp_path, body="a(:)=calculate(a+mutate())",
        extra="real(8) function mutate()\ndeallocate(hidden)\nmutate=1\nend function")
    proof = source.storage_mutation_effects(caller.qualified, expression=expression)
    assert not proof.preserves_descriptors()
    assert any(event["resource"] == "functions::hidden" for event in proof.public()["mutations"])


def test_private_temporary_lifetime_does_not_mutate_caller_descriptor(tmp_path):
    _path, source, caller, (expression,) = fixture(tmp_path,
        function_declarations="real(8),intent(in)::x(:)\nreal(8),allocatable::scratch(:)",
        function_body="allocate(scratch(2))\nscratch(:)=1\ncalculate=sum(x)+sum(scratch)")
    proof = source.storage_mutation_effects(caller.qualified, expression=expression)
    assert proof.preserves_descriptors()
    assert proof.public()["mutations"]


def test_same_text_calls_are_distinct_original_nodes_and_registry_tokens(tmp_path):
    _path, source, caller, expressions = fixture(tmp_path,
        body="a(:)=calculate(a)\na(:)=calculate(a)")
    first, second = (source.storage_mutation_effects(caller.qualified, expression=node) for node in expressions)
    assert first.identity != second.identity
    assert first.public()["original_expression"]["syntax_path"] != second.public()["original_expression"]["syntax_path"]
    with pytest.raises(CompilationError, match="exact registered"):
        first.validate(source, caller.qualified, expressions[1])


def test_private_source_view_uses_original_issuer_and_no_projected_body_authority(tmp_path):
    _path, source, caller, (expression,) = fixture(tmp_path)
    projected = source.private_projection()
    projected._descriptor_source = source
    assert projected.storage_mutation_effects(caller.qualified, expression=expression) is source.storage_mutation_effects(
        caller.qualified, expression=expression)


@pytest.mark.parametrize("change", ["detached_expression", "registry", "expression_identity"])
def test_exact_parser_membership_and_registered_token_required(tmp_path, change):
    _path, source, caller, (expression,) = fixture(tmp_path)
    proof = source.storage_mutation_effects(caller.qualified, expression=expression)
    if change == "detached_expression":
        detached = type(expression)(str(expression))
        with pytest.raises(CompilationError, match="exact unique original"):
            source.storage_mutation_effects(caller.qualified, expression=detached)
    elif change == "registry":
        source._storage_mutation_effects[proof.identity] = replace(proof)
        with pytest.raises(CompilationError, match="exact registered"):
            proof.validate()
    else:
        object.__setattr__(proof, "_expression", None)
        with pytest.raises(CompilationError, match="original source or resolution changed"):
            proof.validate()
