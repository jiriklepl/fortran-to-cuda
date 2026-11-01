"""Fixed primitive function values do not change an external descriptor."""
import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def fixture(tmp_path, *, declaration="real(8)::value(-1:1)",
            body="value=[x(1),x(2),x(3)]", sealed=True, extra="", caller_body="a(:)=vector(a)",
            module_declarations=""):
    path = tmp_path / "values.f90"
    path.write_text(f"""module values
real(8),allocatable::hidden(:)
{module_declarations}
contains
subroutine caller(a)
real(8),allocatable,intent(inout)::a(:)
{caller_body}
end subroutine
function vector(x) result(value)
real(8),intent(in)::x(:)
{declaration}
{body}
end function
{extra}
end module
""")
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    caller = analysis.routines["values::caller"]
    expression, = (node for node in walk(caller.execution)
                   if type(node).__name__ == "Part_Ref" and str(node.items[0]).lower() == "vector")
    return analysis, caller, expression


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize(("declaration", "body", "bounds", "caller_body"), [
    ("real(8)::value(-1:1)", "value=[x(1),x(2),x(3)]", [[-1, 1]], "a(:)=vector(a)"),
    ("integer::value(3)", "value=[1,2,3]", [[1, 3]], "a(:)=vector(a)"),
    ("logical::value(3)", "value=[.true.,.false.,.true.]", [[1, 3]], "if(any(vector(a))) a(:)=1.0_8"),
    ("real(8)::value(0:1,2:3)", "value=1", [[0, 1], [2, 3]], "a(:)=sum(vector(a))"),
])
def test_fixed_value_result_keeps_full_function_closure_without_execution_grants(
        tmp_path, sealed, declaration, body, bounds, caller_body):
    analysis, caller, expression = fixture(tmp_path, declaration=declaration, body=body,
                                           sealed=sealed, caller_body=caller_body)
    proof = analysis.storage_mutation_effects(caller.qualified, expression=expression)
    assert proof.preserves_descriptors(), proof.public()
    assert proof.descriptor_unchanged("argument::a")
    record = proof.public()
    result, = record["fixed_array_function_results"]
    assert result["procedure"] == "values::vector"
    assert result["original_result"]["name"] == "value"
    assert result["bounds"] == bounds
    assert record["version"] == 11
    assert {item["procedure"] for item in record["closure"]} == {"values::vector"}
    for flag in ("payload_access_authorized", "execution_authorized", "entry_definition_authority",
                 "capture_authorized", "motion_authorized", "gpu_legality_established", "alias_authority"):
        assert record[flag] is False
    assert proof.validate() is proof


@pytest.mark.parametrize(("declaration", "body"), [
    ("real(8),allocatable::value(:)", "allocate(value(3))\nvalue=x(1:3)"),
    ("real(8),pointer::value(:)", "nullify(value)"),
    ("real(8),target::value(3)", "value=1"),
    ("real(8),save::value(3)", "value=1"),
    ("real(8)::value(size(x))", "value=1"),
    ("real(8)::value(257)", "value=1"),
])
def test_unproved_result_storage_is_not_promoted_to_fixed_value(tmp_path, declaration, body):
    analysis, caller, expression = fixture(tmp_path, declaration=declaration, body=body)
    proof = analysis.storage_mutation_effects(caller.qualified, expression=expression)
    assert not proof.preserves_descriptors()
    assert proof.public()["frontiers"]
    assert not proof.public().get("fixed_array_function_results")


@pytest.mark.parametrize("effect", ["deallocate(hidden)", "call unknown(x)", "call mutate()"])
def test_fixed_result_does_not_hide_original_body_or_transitive_lifetime_effects(tmp_path, effect):
    extra = "subroutine mutate()\ndeallocate(hidden)\nend subroutine"
    analysis, caller, expression = fixture(tmp_path, body="value=1\n" + effect, extra=extra)
    proof = analysis.storage_mutation_effects(caller.qualified, expression=expression)
    assert not proof.preserves_descriptors()
    assert not proof.descriptor_unchanged("argument::a")
    assert proof.public()["mutations"] or proof.public()["frontiers"]


def test_function_value_does_not_remove_caller_allocation_assignment(tmp_path):
    # Querying the original function alone never lends authority to other
    # original source statements. Whole ALLOCATABLE LHS assignment remains
    # a possible allocation change, even for a fixed-size result.
    analysis, _caller, _expression = fixture(tmp_path, caller_body="a=vector(a)")
    proof = analysis.storage_mutation_effects("values::caller")
    assert not proof.preserves_descriptors()
    assert any(row["resource"] == "argument::a" for row in proof.public()["mutations"])


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize(("field", "value"), [("kind", 4), ("attributes", frozenset({"pointer"}))])
def test_result_declaration_cannot_change_during_bound_validation(tmp_path, monkeypatch, sealed, field, value):
    import compiler.frontend.storage_mutation_effects as storage

    analysis, caller, expression = fixture(tmp_path, sealed=sealed)
    original = storage._fixed_array_bounds
    reached = []

    def changed(source, binding):
        result = original(source, binding)
        reached.append(binding)
        object.__setattr__(binding, field, value)
        return result

    monkeypatch.setattr(storage, "_fixed_array_bounds", changed)
    with pytest.raises(CompilationError):
        analysis.storage_mutation_effects(caller.qualified, expression=expression)
    assert reached


def test_result_declaration_cannot_change_in_final_publication_callback(tmp_path, monkeypatch):
    analysis, caller, expression = fixture(tmp_path)
    binding = analysis.numerical_helpers["values::vector"].scope.bindings["value"]
    original = analysis.verify_publication
    reached = []

    def changed():
        result = original()
        reached.append(binding)
        object.__setattr__(binding, "kind", 4)
        return result

    monkeypatch.setattr(analysis, "verify_publication", changed)
    with pytest.raises(CompilationError):
        analysis.storage_mutation_effects(caller.qualified, expression=expression)
    assert reached


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("constants", ["integer,parameter::n=3", "integer,parameter::m=2,n=m+1"])
def test_original_host_constant_result_bounds(tmp_path, sealed, constants):
    analysis, caller, expression = fixture(tmp_path, declaration="real(8)::value(n)",
                                           sealed=sealed, module_declarations=constants)
    proof = analysis.storage_mutation_effects(caller.qualified, expression=expression)
    assert proof.preserves_descriptors()
    result, = proof.public()["fixed_array_function_results"]
    assert result["bounds"] == [[1, 3]]
    assert proof.validate() is proof


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize(("constants", "dependency", "field", "replacement"), [
    ("integer,parameter::n=3", "n", "items", ("4", None)),
    ("integer,parameter::m=2,n=m+1", "m", "items", ("4", None)),
    ("integer,parameter::m=3,n=m", "n", "string", "hidden"),
    ("integer,parameter::m=2,n=3", None, "string", "m"),
])
def test_host_bound_constant_cannot_change_in_final_publication(
        tmp_path, monkeypatch, sealed, constants, dependency, field, replacement):
    analysis, caller, expression = fixture(tmp_path, declaration="real(8)::value(n)",
                                           sealed=sealed, module_declarations=constants)
    literal = (analysis.modules["values"].kinds.values[dependency] if dependency else
               analysis.numerical_helpers["values::vector"].scope.bindings["value"].shape_nodes[0].items[1])
    original = analysis.verify_publication
    reached = []

    def changed():
        result = original()
        if not reached:
            reached.append(literal)
            object.__setattr__(literal, field, replacement)
        return result

    monkeypatch.setattr(analysis, "verify_publication", changed)
    with pytest.raises(CompilationError):
        analysis.storage_mutation_effects(caller.qualified, expression=expression)
    assert reached
