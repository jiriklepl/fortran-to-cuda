"""Native scalar functions supply source effects, never intrinsic/GPU authority."""

import json

import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def source(tmp_path, *, prefix="elemental", body="r=x/=x", declaration="real(8),intent(in)::x",
           expression="check(p)", extra="", cache=None):
    helper = tmp_path / "arithmetic.f90"
    helper.write_text(f"""module arithmetic
{extra}
contains
logical {prefix} function predicate(x) result(r)
{declaration}
{body}
end function
end module
""")
    caller = tmp_path / "client.f90"
    caller.write_text(f"""module client
use arithmetic,only:check=>predicate
contains
subroutine leaf(p,enabled,result)
real(8),intent(in)::p
logical,intent(in)::enabled
logical,intent(inout)::result
if(enabled) result={expression}
end subroutine
subroutine step(p,enabled,result)
real(8),intent(in)::p
logical,intent(in)::enabled
logical,intent(inout)::result
if(enabled) call leaf(p,enabled,result)
end subroutine
end module
""")
    return SourceEffects([helper, caller], summary_cache=cache)


@pytest.mark.parametrize("prefix", ["pure", "elemental", "pure elemental"])
@pytest.mark.parametrize("expression", ["check(p)", "check(x=p)", "check(p+1.d0)"])
def test_original_source_read_effects_and_argument_order(tmp_path, prefix, expression):
    analysis = source(tmp_path, prefix=prefix, expression=expression)
    summary = analysis.summarize("client::step")
    assert summary["complete"], summary["reasons"]
    requirement, = summary["native_function_requirements"]
    assert requirement["callee"] == "arithmetic::predicate"
    assert requirement["native_only"]
    assert not requirement["gpu_legality_established"]
    assert not requirement["native_callback_authorized"]
    assert requirement["source_expression"].lower().replace(" ", "") == expression
    assert requirement["guard_frames"] == [
        {"procedure": "client::step", "condition": "enabled"},
        {"procedure": "client::leaf", "condition": "enabled"}]
    assert "arithmetic::predicate" not in analysis.routines
    assert not summary["cloneable"]
    assert not analysis.summarize("client::leaf")["cloneable"]
    assert not summary["native_predicate_requirements"]
    reads = [effect for effect in summary["ordered_effects"] if effect["resource"] == "argument::p"]
    assert reads
    assert all(effect["kind"] == "read" for effect in reads)
    assert all(effect["guard_frames"] == requirement["guard_frames"] for effect in reads)
    assert all(operation["kind"] != "call" for operation in analysis.summarize("client::leaf")["operations"])
    with pytest.raises(CompilationError, match="original source-backed procedure authority"):
        analysis.structure("arithmetic::predicate")


def test_kind_generic_resolves_original_source_not_intrinsic_name(tmp_path):
    path = tmp_path / "generic.f90"
    path.write_text("""module ieee_arithmetic
use iso_fortran_env,only:real32,real64
interface ieee_is_nan
module procedure f32,f64
end interface
contains
logical elemental function f32(x) result(r)
real(real32),intent(in)::x
r=x/=x
end function
logical elemental function f64(x) result(r)
real(real64),intent(in)::x
r=x/=x
end function
end module
module client
use ieee_arithmetic,only:predicate=>ieee_is_nan
contains
subroutine leaf(p,q,a,b)
real(4),intent(in)::p
real(8),intent(in)::q
logical,intent(out)::a,b
a=predicate(p)
b=predicate(x=q)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    summary = analysis.summarize("client::leaf")
    assert summary["complete"], summary["reasons"]
    assert [item["callee"] for item in summary["native_function_requirements"]] == [
        "ieee_arithmetic::f32", "ieee_arithmetic::f64"]
    assert summary["native_predicate_requirements"] == []


@pytest.mark.parametrize(("prefix", "declaration", "body", "reason"), [
    ("", "real(8),intent(in)::x", "r=x/=x", "requires PURE"),
    ("impure elemental", "real(8),intent(in)::x", "r=x/=x", "requires PURE"),
    ("pure", "real(8),intent(inout)::x", "r=x/=x", "INTENT(IN)"),
    ("pure", "real(8),optional,intent(in)::x", "r=x/=x", "INTENT(IN)"),
    ("pure", "real(8),pointer,intent(in)::x", "r=x/=x", "INTENT(IN)"),
    ("pure", "real(8),intent(in)::x\nreal(8)::a(2)", "r=x/=x", "fixed private scalars"),
    ("pure", "real(8),intent(in)::x\nreal(8),save::a", "r=x/=x", "fixed private scalars"),
    ("pure", "real(8),intent(in)::x", "print *,x\nr=x/=x", "effects incomplete"),
    ("pure", "real(8),intent(in)::x", "call unknown(x)\nr=x/=x", "effects incomplete"),
])
def test_purity_does_not_replace_complete_source_proof(tmp_path, prefix, declaration, body, reason):
    analysis = source(tmp_path, prefix=prefix, declaration=declaration, body=body)
    summary = analysis.summarize("client::leaf")
    assert not summary["complete"]
    assert any(reason in value for value in summary["reasons"]), summary["reasons"]
    assert not summary["native_function_requirements"]


def test_internal_function_keeps_original_lexical_reads_and_guards(tmp_path):
    path = tmp_path / "lexical.f90"
    path.write_text("""module lexical
contains
subroutine step(p,limit,enabled,result)
real(8),intent(in)::p,limit
logical,intent(in)::enabled
logical,intent(inout)::result
if(enabled) result=check(p)
contains
logical pure function check(x) result(r)
real(8),intent(in)::x
r=.false.
if(x>0.d0) r=x>limit
end function
end subroutine
end module
""")
    analysis = SourceEffects([path])
    summary = analysis.summarize("lexical::step")
    assert summary["complete"], summary["reasons"]
    requirement, = summary["native_function_requirements"]
    assert requirement["callee"] == "lexical::step::check"
    hidden = [effect for effect in summary["ordered_effects"] if effect["resource"] == "argument::limit"]
    assert len(hidden) == 1
    assert hidden[0]["guard_frames"] == [
        {"procedure": "lexical::step", "condition": "enabled"},
        {"procedure": "lexical::step::check", "condition": "x > 0.D0"}]


def test_hidden_allocation_is_not_authorized_by_function_purity(tmp_path):
    analysis = source(tmp_path, prefix="pure", extra="real(8),allocatable::weights(:)", body="r=x>weights(1)")
    summary = analysis.summarize("client::leaf")
    assert not summary["complete"]
    assert any("storage lifetime requires capture proof: arithmetic::weights" in reason for reason in summary["reasons"])


def test_native_source_metadata_reissued_after_cache_and_source_change(tmp_path):
    cache = tmp_path / "cache"
    analysis = source(tmp_path, cache=cache)
    first = analysis.summarize("client::step")
    second = SourceEffects(analysis.inputs.paths, summary_cache=cache).summarize("client::step")
    assert first["complete"]
    assert second["complete"]
    assert first["native_function_requirements"] == second["native_function_requirements"]
    assert not list(cache.glob("*.json"))
    helper = tmp_path / "arithmetic.f90"
    helper.write_text(helper.read_text().replace("r=x/=x", "r=x>0.d0"))
    third = SourceEffects(analysis.inputs.paths, summary_cache=cache).summarize("client::step")
    assert third["complete"]
    assert third["native_function_requirements"] != first["native_function_requirements"]


def test_exact_original_expression_and_source_tokens_required(tmp_path):
    analysis = source(tmp_path)
    summary = analysis.summarize("client::leaf")
    requirement, = summary["native_function_requirements"]
    proof = analysis._native_functions[requirement["proof_identity"]]
    assert proof.validate(analysis, "client::leaf", proof._expression) is proof
    from copy import copy
    with pytest.raises(CompilationError, match="original expression authority"):
        proof.validate(analysis, "client::leaf", copy(proof._expression))
    routine = analysis.numerical_helpers["arithmetic::predicate"]
    assignment = next(node for node in walk(routine.execution) if type(node).__name__ == "Assignment_Stmt")
    assignment.items = (assignment.items[0], assignment.items[1], assignment.items[0])
    with pytest.raises(CompilationError, match="authority"):
        proof.validate(analysis, "client::leaf", proof._expression)


def test_native_analysis_registry_is_private(tmp_path):
    from compiler.scopes.segments import _native_analysis

    original = source(tmp_path)
    native = _native_analysis(original)
    summary = native.summarize("client::leaf")
    assert summary["complete"], summary["reasons"]
    assert native._native_functions
    assert original._native_functions == {}


def test_function_syntax_budget_covers_purely_local_work(tmp_path):
    analysis = source(tmp_path, prefix="pure", declaration="real(8),intent(in)::x\nreal(8)::v",
                      body="v=x\n" + "v=v+1.d0\n" * 100 + "r=v>0.d0")
    summary = analysis.summarize("client::leaf")
    assert not summary["complete"]
    assert any("syntax budget exhausted" in reason for reason in summary["reasons"])


def test_reached_segment_preserves_function_guards_without_synthetic_call(tmp_path):
    analysis = source(tmp_path)
    assignment = next(node for node in walk(analysis.routines["client::leaf"].execution)
                      if type(node).__name__ == "Assignment_Stmt")
    summary = analysis.segment_summary("client::leaf", (assignment,))
    assert summary["complete"], summary["reasons"]
    requirement, = summary["native_function_requirements"]
    assert requirement["guard_frames"] == [{"procedure": "client::leaf", "condition": "enabled"}]
    assert not summary["cloneable"]
    assert "source_function_version" in analysis._summary_authority()
    json.dumps(summary)


def test_proof_effect_payload_is_immutable_and_returned_by_value(tmp_path):
    analysis = source(tmp_path)
    first = analysis.summarize("client::leaf")
    requirement, = first["native_function_requirements"]
    proof = analysis._native_functions[requirement["proof_identity"]]
    effects = proof.effects
    effects[0]["kind"] = "overwrite"
    effects[0]["resource"] = "forged::storage"
    assert proof.effects[0]["kind"] == "read"
    assert proof.validate(analysis, "client::leaf", proof._expression) is proof
    analysis._closures.clear()
    second = analysis.summarize("client::leaf")
    assert first == second


def test_in_memory_helper_mutation_cannot_reuse_valid_caller_summary(tmp_path):
    analysis = source(tmp_path)
    first = analysis.summarize("client::leaf")
    assert first["complete"]
    routine = analysis.numerical_helpers["arithmetic::predicate"]
    assignment = next(node for node in walk(routine.execution) if type(node).__name__ == "Assignment_Stmt")
    assignment.items = (assignment.items[0], assignment.items[1], assignment.items[0])
    second = analysis.summarize("client::leaf")
    assert not second["complete"]
    assert any("authority" in reason for reason in second["reasons"]), second["reasons"]


def test_unused_function_actual_expression_keeps_its_reads(tmp_path):
    analysis = source(tmp_path, prefix="pure", body="r=.false.", expression="check(p+1.d0)")
    summary = analysis.summarize("client::leaf")
    assert summary["complete"], summary["reasons"]
    assert any(effect["resource"] == "argument::p" for effect in summary["ordered_effects"])


def test_repeated_diamond_function_closures_are_bounded_and_reused(tmp_path):
    path = tmp_path / "diamond.f90"
    path.write_text("""module tree
contains
logical pure function leaf(x) result(r)
real(8),intent(in)::x
r=x>0.d0
end function
logical pure function left(x) result(r)
real(8),intent(in)::x
r=leaf(x)
end function
logical pure function right(x) result(r)
real(8),intent(in)::x
r=leaf(x)
end function
subroutine step(x,r)
real(8),intent(in)::x
logical,intent(out)::r
r=left(x).or.right(x)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    summary = analysis.summarize("tree::step")
    assert summary["complete"], summary["reasons"]
    assert summary["closure_depth"] == 3
    assert len(analysis._closures["tree::step"].summaries) == 4
    assert {item["callee"] for item in summary["native_function_requirements"]} == {
        "tree::leaf", "tree::left", "tree::right"}
    limited = SourceEffects([path], procedures=3).summarize("tree::step")
    assert not limited["complete"]
    assert any("procedure budget exhausted" in reason for reason in limited["reasons"])


def test_recursive_pure_function_remains_a_boundary(tmp_path):
    analysis = source(tmp_path, prefix="pure recursive", body="r=predicate(x)")
    summary = analysis.summarize("client::leaf")
    assert not summary["complete"]
    assert any("recursive call" in reason for reason in summary["reasons"])


def test_opaque_readonly_contract_inside_pure_source_function_is_rejected(tmp_path):
    analysis = source(tmp_path, prefix="pure", extra="use library,only:opaque", body="call opaque(x)\nr=x/=x")
    analysis.contracts = {"library::opaque": {"identity": "library-v1", "lifetime": "stable", "escapes": False,
        "ordering": "serial", "complete": True, "descriptor_changes": False,
        "completion": "synchronous", "effects": [{"argument": 0, "kind": "read", "section": "whole"}]}}
    summary = analysis.summarize("client::leaf")
    assert not summary["complete"]
    assert any("completion is unproved" in reason for reason in summary["reasons"]), summary["reasons"]


@pytest.mark.parametrize("first_query", [False, True])
def test_generic_resolution_mutation_cannot_replace_original_function(tmp_path, first_query):
    analysis = source(tmp_path)
    module = analysis.modules["arithmetic"]
    if first_query:
        assert analysis.summarize("client::leaf")["complete"]
    # Change resolution bookkeeping while retaining every original AST token.
    module.procedures["predicate"] = "client::leaf"
    summary = analysis.summarize("client::leaf")
    assert not summary["complete"]
    assert not summary["native_function_requirements"]


def test_module_kind_binding_mutation_invalidates_registered_function_proof(tmp_path):
    analysis = source(tmp_path)
    summary = analysis.summarize("client::leaf")
    proof = analysis._native_functions[summary["native_function_requirements"][0]["proof_identity"]]
    analysis.modules["arithmetic"].kinds.values["invented"] = "8"
    with pytest.raises(CompilationError, match="resolution authority changed"):
        proof.validate(analysis, "client::leaf", proof._expression)


@pytest.mark.parametrize("first_query", [False, True])
def test_generic_cannot_redirect_to_another_authentic_same_signature_helper(tmp_path, first_query):
    initial = source(tmp_path)
    helper = tmp_path / "arithmetic.f90"
    helper.write_text(helper.read_text().replace("contains", "interface chosen\nmodule procedure predicate\nend interface\ncontains")
                      .replace("end module", "logical pure function other(x) result(r)\nreal(8),intent(in)::x\nr=.false.\nend function\nend module"))
    caller = tmp_path / "client.f90"
    caller.write_text(caller.read_text().replace("check=>predicate", "check=>chosen"))
    analysis = SourceEffects(initial.inputs.paths)
    if first_query:
        assert analysis.summarize("client::leaf")["complete"]
    analysis.modules["arithmetic"].generics["chosen"] = [("arithmetic", "other")]
    summary = analysis.summarize("client::leaf")
    assert not summary["complete"]
    assert any("resolution authority changed" in reason for reason in summary["reasons"])
