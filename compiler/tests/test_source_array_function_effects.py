"""Exact native readonly helper effects for bounded original fixed arrays."""

from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SOURCE_SUMMARY_VERSION, SourceEffects
from compiler.frontend.source_functions import SOURCE_FUNCTION_VERSION
from compiler.ir import CompilationError


def fixture(tmp_path, *, declaration="real(8),intent(in)::a(-1:0,4:5)",
            body="r=a(-1,4)+a(0,5)", result="real(8)",
            caller="real(8),intent(in)::p(-4:-3,7:8)", actual="p", extra="", prefix="pure",
            imports="", cache=None):
    helper = tmp_path / "shapes.f90"
    helper.write_text(f"""module shapes
{imports}
contains
{prefix} function contract(a) result(r)
{declaration}
{result}::r
{extra}
{body}
end function
end module
""")
    client = tmp_path / "application.f90"
    client.write_text(f"""module application
use shapes,only:combine=>contract
contains
subroutine step(p,enabled,value)
{caller}
logical,intent(in)::enabled
{result},intent(inout)::value
if(enabled) value=combine(a={actual})
end subroutine
end module
""")
    return SourceEffects([helper, client], summary_cache=cache)


def test_renamed_multifile_fixed_arrays_preserve_negative_bounds_and_authority(tmp_path):
    analysis = fixture(tmp_path)
    summary = analysis.summarize("application::step")
    assert summary["complete"], summary["reasons"]
    assert summary["summary_version"] == SOURCE_SUMMARY_VERSION == 19
    requirement, = summary["native_function_requirements"]
    assert requirement["schema_version"] == SOURCE_FUNCTION_VERSION == 2
    assert requirement["fixed_arrays"] == [{"resource": "argument::a", "rank": 2,
                                            "bounds": [[-1, 0], [4, 5]], "role": "readonly_argument"}]
    assert requirement["source_call"]["original_arguments"] == [
        {"position": 0, "formal": "a", "keyword": "a", "actual": "p"}]
    assert requirement["guard_frames"] == [{"procedure": "application::step", "condition": "enabled"}]
    assert not requirement["gpu_legality_established"]
    assert not requirement["numerical_lowering_authorized"]
    assert not requirement["native_callback_authorized"]
    assert not summary["cloneable"]
    assert "shapes::contract" not in analysis.routines
    effects = [item for item in summary["ordered_effects"] if item.get("source_function_identity")]
    assert effects
    assert all(item["resource"] == "argument::p" and item["rank"] == 2 for item in effects)
    mapping = effects[0]["view_chain"][0]
    assert mapping["formal_descriptor"]["lower_bounds"] == ("- 1", "4")
    assert mapping["actual_descriptor"]["lower_bounds"] == ("- 4", "7")
    assert all(item["guard_frames"] == requirement["guard_frames"] for item in effects)
    assert not summary["native_sections"]["available"]  # No fabricated exact function footprints.


@pytest.mark.parametrize(("dtype", "body"), [
    ("real(4)", "r=a(1)+a(2)"), ("real(8)", "r=a(1)+a(2)"),
    ("integer(4)", "r=a(1)+a(2)"), ("integer(8)", "r=a(1)+a(2)"),
    ("logical(4)", "r=any(a)"), ("logical(8)", "r=all(a)"),
])
def test_fixed_numeric_and_logical_input_array_kinds(tmp_path, dtype, body):
    analysis = fixture(tmp_path, declaration=f"{dtype},intent(in)::a(2)",
                       caller=f"{dtype},intent(in)::p(2)", body=body, result=dtype)
    summary = analysis.summarize("application::step")
    assert summary["complete"], summary["reasons"]


def test_existing_scalar_target_contract_remains_available(tmp_path):
    analysis = fixture(tmp_path, declaration="real(8),target,intent(in)::a", body="r=a",
                       caller="real(8),intent(in)::p")
    summary = analysis.summarize("application::step")
    assert summary["complete"], summary["reasons"]


def test_private_arrays_and_nested_readonly_leaves_never_become_captures(tmp_path):
    path = tmp_path / "private.f90"
    path.write_text("""module arithmetic
contains
pure function combine(a) result(r)
real(8),intent(in)::a(2,2)
real(8)::r,scratch(-1:0,4:5)
scratch(-1,4)=a(1,1)
scratch(0,4)=a(2,1)
scratch(-1,5)=a(1,2)
scratch(0,5)=a(2,2)
r=leaf(scratch)+dot_product(scratch(:,4),scratch(:,5))
contains
pure function leaf(x) result(y)
real(8),intent(in)::x(2,2)
real(8)::y
y=x(1,1)*x(2,2)-x(2,1)*x(1,2)
end function
end function
subroutine step(p,result)
real(8),intent(in)::p(2,2)
real(8),intent(out)::result
real(8)::buffer(2,2)
buffer(1,1)=p(1,1)
buffer(2,1)=p(2,1)
buffer(1,2)=p(1,2)
buffer(2,2)=p(2,2)
result=combine(buffer)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    summary = analysis.summarize("arithmetic::step")
    assert summary["complete"], summary["reasons"]
    assert {item["callee"] for item in summary["native_function_requirements"]} == {
        "arithmetic::combine", "arithmetic::combine::leaf"}
    assert all(item.get("resource") in {"argument::p", "argument::result"}
               for item in summary["ordered_effects"])
    assert summary["capture_lifetime_requirements"] == []
    assert summary["persistent_state"] == []
    outer = next(item for item in summary["native_function_requirements"] if item["callee"] == "arithmetic::combine")
    assert {item["role"] for item in outer["fixed_arrays"]} == {"readonly_argument", "private_local"}
    proof = analysis._native_functions[outer["proof_identity"]]
    assert all(item["resource"] == "argument::a" for item in proof.effects)


def test_rank_reducing_constant_section_retains_caller_rank_and_view(tmp_path):
    analysis = fixture(tmp_path, declaration="real(8),intent(in)::a(-1:0)", body="r=a(-1)+a(0)",
                       caller="real(8),intent(in)::p(-2:-1,3:4,7:8)", actual="p(-2:-1,4,7)")
    summary = analysis.summarize("application::step")
    assert summary["complete"], summary["reasons"]
    effects = [item for item in summary["ordered_effects"] if item.get("source_function_identity")]
    assert effects
    assert all(item["rank"] == 3 for item in effects)
    mapping = effects[0]["view_chain"][0]
    assert mapping["storage"] == "rectangle"
    assert mapping["section"]["rank"] == 3
    assert mapping["section"]["logical_rank"] == 1
    assert mapping["section"]["retained_axes"] == [0]
    assert mapping["actual"] == "p(- 2 : - 1, 4, 7)"


def test_parameter_bounds_and_whole_default_section_are_source_proved(tmp_path):
    analysis = fixture(tmp_path, imports="integer,parameter::n=2", declaration="real(8),intent(in)::a(n,n)",
                       body="r=a(1,1)+a(n,n)", actual="p(:,:)")
    summary = analysis.summarize("application::step")
    assert summary["complete"], summary["reasons"]
    mapping = summary["native_function_requirements"][0]["source_call"]["resource_mappings"][0]
    assert mapping["storage"] == "rectangle"
    assert mapping["section"]["dependencies"] == [{"resource": "argument::p", "kind": "descriptor_read"}]


@pytest.mark.parametrize(("formal_extent", "complete"), [(2, True), (4, False)])
def test_section_parameter_resolves_in_caller_not_array_defining_module(tmp_path, formal_extent, complete):
    path = tmp_path / "bounds.f90"
    path.write_text(f"""module storage
integer,parameter::n=4
real(8)::field(n)
end module
module operations
contains
pure function combine(a) result(r)
real(8),intent(in)::a({formal_extent})
real(8)::r
r=a(1)
end function
end module
module application
use storage,only:local=>field
use operations,only:combine
contains
subroutine step(value)
real(8),intent(out)::value
integer,parameter::n=2
value=combine(local(1:n))
end subroutine
end module
""")
    analysis = SourceEffects([path])
    summary = analysis.summarize("application::step")
    assert summary["complete"] is complete, summary["reasons"]
    if not complete:
        assert any("shape mismatch" in reason for reason in summary["reasons"])
    else:
        mapping = summary["native_function_requirements"][0]["source_call"]["resource_mappings"][0]
        assert mapping["section"]["axes"][0]["upper"]["value"] == 2


def test_empty_fixed_array_stays_native_and_retains_original_result(tmp_path):
    analysis = fixture(tmp_path, declaration="real(8),intent(in)::a(0)", body="r=0.d0",
                       caller="real(8),intent(in)::p(-2:-3)")
    summary = analysis.summarize("application::step")
    assert summary["complete"], summary["reasons"]
    assert summary["native_function_requirements"][0]["requires_original_execution"]


@pytest.mark.parametrize(("declaration", "caller", "actual", "reason"), [
    ("real(8),intent(out)::a(2,2)", "real(8),intent(in)::p(2,2)", "p", "INTENT(IN)"),
    ("real(8),optional,intent(in)::a(2,2)", "real(8),intent(in)::p(2,2)", "p", "INTENT(IN)"),
    ("real(8),target,intent(in)::a(2,2)", "real(8),intent(in)::p(2,2)", "p", "INTENT(IN)"),
    ("real(8),allocatable,intent(in)::a(:,:)", "real(8),intent(in)::p(2,2)", "p", "INTENT(IN)"),
    ("real(8),intent(in)::a(:,:)", "real(8),intent(in)::p(2,2)", "p", "constant explicit bounds"),
    ("real(8),intent(in)::a(2,2)", "real(8),intent(in)::p(3,2)", "p", "shape mismatch"),
    ("real(8),intent(in)::a(2,2)", "real(4),intent(in)::p(2,2)", "p", "type, kind or rank"),
    ("real(8),intent(in)::a(2,2)", "real(8),intent(in)::p(4)", "p", "type, kind or rank"),
    ("real(8),intent(in)::a(2,2)", "real(8),intent(in)::p(2,2)", "p(1,1)", "scalar payload coherence"),
    ("real(8),intent(in)::a(2,2)", "real(8),allocatable,intent(in)::p(:,:)", "p", "fixed unaliased storage"),
    ("real(8),intent(in)::a(2,2)", "real(8),optional,intent(in)::p(2,2)", "p", "fixed unaliased storage"),
    ("real(8),intent(in)::a(2,2)", "real(8),target,intent(in)::p(2,2)", "p", "fixed unaliased storage"),
    ("real(8),intent(in)::a(2,2)", "real(8),intent(in)::p(2,2)", "p(1:3,:)", "outside original bounds"),
    ("real(8),intent(in)::a(2,2)", "real(8),intent(in)::p(2,2)", "p(1:2:2,:)", "unit stride"),
    ("real(8),intent(in)::a(257)", "real(8),intent(in)::p(257)", "p", "fixed storage budget"),
    ("real(8),intent(in)::a(-2147483647-1:2147483647)", "real(8),intent(in)::p(2)", "p", "INTEGER ABI"),
    ("real(8),intent(in)::a(2147483648:2147483649,1:2)", "real(8),intent(in)::p(2,2)", "p", "INTEGER ABI"),
    ("real(8),intent(in)::a(2,2)", "real(8),intent(in)::p(2,2)\ninteger::n", "p(n:n+1,:)", "fixed bounds"),
])
def test_unproved_array_association_remains_incomplete(tmp_path, declaration, caller, actual, reason):
    analysis = fixture(tmp_path, declaration=declaration, caller=caller, actual=actual)
    summary = analysis.summarize("application::step")
    assert not summary["complete"]
    assert any(reason in item for item in summary["reasons"]), summary["reasons"]
    assert not summary["native_function_requirements"]


@pytest.mark.parametrize(("extra", "reason"), [
    ("real(8)::scratch(257)", "fixed storage budget"),
    ("real(8),save::scratch(2)", "fixed private scalars"),
    ("real(8),target::scratch(2)", "fixed private scalars"),
    ("real(8),allocatable::scratch(:)", "fixed private scalars"),
    ("real(8)::scratch(a(1,1))", "constant explicit bounds"),
])
def test_unproved_private_arrays_cannot_supply_readonly_proof(tmp_path, extra, reason):
    analysis = fixture(tmp_path, extra=extra)
    summary = analysis.summarize("application::step")
    assert not summary["complete"]
    assert any(reason in item for item in summary["reasons"]), summary["reasons"]


def test_dot_product_effects_are_readonly_with_original_slice_dependencies(tmp_path):
    analysis = fixture(tmp_path, body="r=dot_product(a(:,4),a(:,5))")
    summary = analysis.summarize("application::step")
    assert summary["complete"], summary["reasons"]
    assert all(item["kind"] == "read" for item in summary["ordered_effects"]
               if item.get("source_function_identity"))


def test_shadowed_dot_product_requires_actual_source_effects(tmp_path):
    analysis = fixture(tmp_path, body="r=dot_product(a(:,4),a(:,5))",
                       imports="use unavailable_library,only:dot_product")
    summary = analysis.summarize("application::step")
    assert not summary["complete"]
    assert any("DOT_PRODUCT" in item or "dot_product" in item for item in summary["reasons"])


@pytest.mark.parametrize("body", ["a(-1,4)=0.d0\nr=a(0,5)", "call unavailable(a)\nr=a(0,5)"])
def test_fixed_arrays_do_not_authorize_mutation_or_unknown_children(tmp_path, body):
    analysis = fixture(tmp_path, body=body)
    summary = analysis.summarize("application::step")
    assert not summary["complete"]
    assert not summary["native_function_requirements"]


def test_fixed_array_token_is_exact_private_and_invalidated_by_original_shape_mutation(tmp_path):
    from compiler.scopes.segments import _native_analysis

    original = fixture(tmp_path)
    analysis = _native_analysis(original)
    summary = analysis.summarize("application::step")
    assert summary["complete"], summary["reasons"]
    requirement, = summary["native_function_requirements"]
    proof = analysis._native_functions[requirement["proof_identity"]]
    assert original._native_functions == {}
    assert proof.validate(analysis, "application::step", proof._expression) is proof
    with pytest.raises(CompilationError, match="original expression authority"):
        proof.validate(original, "application::step", proof._expression)
    with pytest.raises(CompilationError, match="original expression authority"):
        proof.validate(analysis, "application::step", copy(proof._expression))
    callee = analysis.numerical_helpers["shapes::contract"]
    shape = callee.scope.bindings["a"].shape_nodes[0]
    shape.items = (shape.items[0], shape.items[0])
    with pytest.raises(CompilationError, match="authority"):
        proof.validate(analysis, "application::step", proof._expression)


@pytest.mark.parametrize("callee", [False, True])
def test_fixed_bound_resolution_rejects_forged_declaring_scope(tmp_path, callee):
    analysis = fixture(tmp_path)
    routine = analysis.numerical_helpers["shapes::contract"] if callee else analysis.routines["application::step"]
    binding = routine.scope.bindings["a" if callee else "p"]
    # Public binding values are unchanged; an unregistered scope must still
    # never supply the constant/kind authority for these original dimensions.
    binding.declaring_scope = copy(binding.declaring_scope)
    summary = analysis.summarize("application::step")
    assert not summary["complete"]
    assert any("original resolution authority" in item for item in summary["reasons"]), summary["reasons"]


@pytest.mark.parametrize("replace_binding", [False, True])
def test_original_binding_authority_cannot_be_changed_after_proof_or_cached_summary(tmp_path, replace_binding):
    analysis = fixture(tmp_path, extra="real(8)::scratch(2)")
    summary = analysis.summarize("application::step")
    assert summary["complete"], summary["reasons"]
    proof = analysis._native_functions[summary["native_function_requirements"][0]["proof_identity"]]
    scope = analysis.numerical_helpers["shapes::contract"].scope
    binding = scope.bindings["scratch"]
    if replace_binding:
        scope.bindings["scratch"] = copy(binding)
    else:
        binding.declaring_scope = copy(binding.declaring_scope)
    with pytest.raises(CompilationError, match="original resolution authority"):
        proof.validate(analysis, "application::step", proof._expression)
    changed = analysis.summarize("application::step")
    assert not changed["complete"]
    assert any("original resolution authority" in reason for reason in changed["reasons"])


def test_changed_array_source_cannot_reuse_summary_or_proof_identity(tmp_path):
    cache = tmp_path / "cache"
    first_analysis = fixture(tmp_path, cache=cache)
    first = first_analysis.summarize("application::step")
    second = SourceEffects(first_analysis.inputs.paths, summary_cache=cache).summarize("application::step")
    assert first["complete"]
    assert second["complete"]
    assert first["native_function_requirements"] == second["native_function_requirements"]
    path = tmp_path / "shapes.f90"
    path.write_text(path.read_text().replace("r=a(-1,4)+a(0,5)", "r=a(-1,4)-a(0,5)"))
    third = SourceEffects(first_analysis.inputs.paths, summary_cache=cache).summarize("application::step")
    assert third["complete"]
    assert third["native_function_requirements"] != first["native_function_requirements"]
    assert not list(cache.glob("*.json"))


def test_reached_function_array_retains_original_guard(tmp_path):
    analysis = fixture(tmp_path)
    assignment = next(node for node in walk(analysis.routines["application::step"].execution)
                      if type(node).__name__ == "Assignment_Stmt")
    segment = analysis.segment_summary("application::step", (assignment,))
    assert segment["complete"], segment["reasons"]
    requirement, = segment["native_function_requirements"]
    assert requirement["guard_frames"] == [{"procedure": "application::step", "condition": "enabled"}]


def test_fixed_array_function_requires_exact_descriptor_token(tmp_path):
    analysis = fixture(tmp_path)
    path = tmp_path / "application.f90"
    path.write_text(path.read_text().replace("step(p,enabled,value)", "step(p,enabled,value,store)").replace(
        "logical,intent(in)::enabled", "logical,intent(in)::enabled\nreal(8),allocatable,intent(in)::store(:)").replace(
        "combine(a=p)", "combine(p)"))
    analysis = SourceEffects(analysis.inputs.paths)
    assignment = next(node for node in walk(analysis.routines["application::step"].execution)
                      if type(node).__name__ == "Assignment_Stmt")
    graph = analysis.structure("application::step")
    selected = (graph.node_id(assignment),)
    with pytest.raises(CompilationError, match="registered original readonly source-function proof"):
        prove_descriptor_window(analysis, "application::step", selected, ("argument::store",))
    summary = analysis.summarize("application::step")
    assert summary["complete"], summary["reasons"]
    proof = prove_descriptor_window(analysis, "application::step", selected, ("argument::store",))
    assert proof.public()["native_source_function_requirements"][0]["callee"] == "shapes::contract"
