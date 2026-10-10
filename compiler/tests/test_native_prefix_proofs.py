"""Original pre-owner native calls gain effects, never managed capture facts."""

from copy import copy, deepcopy

import pytest

from compiler.frontend.native_prefix import _fork, prove_native_prefix
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module geometry
real(8),allocatable::spacing(:)
end module
module original_work
use geometry,only:widths=>spacing
contains
subroutine step(a,n,enabled,total)
real(8),intent(in)::a(-2:)
integer,intent(in)::n
logical,intent(in)::enabled
real(8),intent(inout)::total
if(enabled) call middle(a,n,total)
contains
subroutine middle(x,m,result)
real(8),intent(in)::x(-2:)
integer,intent(in)::m
real(8),intent(inout)::result
call leaf(x,m,result)
end subroutine
subroutine leaf(x,m,result)
real(8),intent(in)::x(-2:)
integer,intent(in)::m
real(8),intent(inout)::result
integer::i
if(allocated(widths)) then
do i=0,m-1
result=result+x(i)/widths(i)
enddo
endif
end subroutine
end subroutine
end module
"""


def analysis_from(tmp_path, source=SOURCE, **kwargs):
    path = tmp_path / "original.f90"
    path.write_text(source)
    return SourceEffects([path], **kwargs)


def test_hidden_readonly_module_inputs_leave_original_summary_incomplete(tmp_path):
    analysis = analysis_from(tmp_path)
    original = analysis.summarize("original_work::step")
    assert not original["complete"]
    assert not original["ordered_effects"]
    proof = prove_native_prefix(analysis, "original_work::step")
    assert proof.validate(analysis, "original_work::step") is proof
    assert proof.readonly_module_resources == ("geometry::spacing",)
    assert proof.summary["complete"]
    record = proof.public()
    assert record["readonly_module_resources"] == ["geometry::spacing"]
    assert not record["original_summary_complete"]
    assert record["requires_no_active_context"]
    assert record["requires_original_native_execution"]
    for key in ("execution_authorized", "capture_authority", "managed_definition_authority",
                "allocation_freshness_authority", "alias_authority", "gpu_legality_established",
                "numerical_lowering_authorized"):
        assert record[key] is False
    reads = [effect for effect in proof.effects if effect["resource"] == "geometry::spacing"]
    assert reads
    assert all(effect["kind"] in {"read", "descriptor_read"} for effect in reads)
    payload, = [effect for effect in reads if effect["kind"] == "read"]
    assert payload["source_procedure"] == "original_work::step::leaf"
    assert payload["guard_frames"] == [
        {"procedure": "original_work::step", "condition": "enabled"},
        {"procedure": "original_work::step::leaf", "condition": "ALLOCATED(widths)"},
        {"procedure": "original_work::step::leaf", "condition": "DO i = 0, m - 1"}]
    assert all(item["authorized"] is False for item in record["candidate_lifetime_obligations"])
    assert analysis.summarize("original_work::step") == original
    assert analysis.stable_module_allocatables == frozenset()


@pytest.mark.parametrize("declaration", ["real(4)", "real(8)", "integer(4)", "integer(8)"])
def test_original_numeric_module_storage_kinds(tmp_path, declaration):
    analysis = analysis_from(tmp_path, SOURCE.replace("real(8),allocatable::spacing", declaration + ",allocatable::spacing"))
    assert prove_native_prefix(analysis, "original_work::step").readonly_module_resources == ("geometry::spacing",)


def test_repeated_and_diamond_calls_keep_every_original_read(tmp_path):
    source = SOURCE.replace("call leaf(x,m,result)", "call leaf(x,m,result)\ncall other(x,m,result)").replace(
        "end subroutine\nsubroutine leaf", """end subroutine
subroutine other(x,m,result)
real(8),intent(in)::x(-2:)
integer,intent(in)::m
real(8),intent(inout)::result
call leaf(x,m,result)
call leaf(x,m,result)
end subroutine
subroutine leaf""")
    analysis = analysis_from(tmp_path, source)
    proof = prove_native_prefix(analysis, "original_work::step")
    assert proof.readonly_module_resources == ("geometry::spacing",)
    assert len(proof.public()["closure"]) == 4
    reads = [effect for effect in proof.effects if effect["resource"] == "geometry::spacing" and effect["kind"] == "read"]
    assert len(reads) == 3
    assert prove_native_prefix(analysis, "original_work::step") is proof


@pytest.mark.parametrize("action", ["widths(i)=x(i)", "widths(i)=widths(i)+x(i)",
                                    "deallocate(widths)", "allocate(widths(1:m))",
                                    "call unknown(widths)", "call move_alloc(widths,other)",
                                    "widths=x"])
def test_hidden_write_escape_and_descriptor_change_do_not_disappear(tmp_path, action):
    source = SOURCE.replace("real(8),allocatable::spacing(:)", "real(8),allocatable::spacing(:),other(:)")
    source = source.replace("result=result+x(i)/widths(i)", "result=result+x(i)/widths(i)\n" + action)
    analysis = analysis_from(tmp_path, source)
    with pytest.raises(CompilationError):
        prove_native_prefix(analysis, "original_work::step")
    assert not getattr(analysis, "_native_prefix_proofs", {})
    assert analysis.stable_module_allocatables == frozenset()


@pytest.mark.parametrize("attributes", ["pointer", "allocatable,target", "allocatable,volatile",
                                        "allocatable,asynchronous"])
def test_uncertain_hidden_storage_is_not_provisionally_authorized(tmp_path, attributes):
    analysis = analysis_from(tmp_path, SOURCE.replace("real(8),allocatable::spacing", "real(8)," + attributes + "::spacing"))
    with pytest.raises(CompilationError):
        prove_native_prefix(analysis, "original_work::step")


def test_descriptor_only_original_native_read_needs_no_new_payload_fact(tmp_path):
    source = SOURCE.replace("result=result+x(i)/widths(i)", "result=result+real(size(widths),8)")
    analysis = analysis_from(tmp_path, source)
    proof = prove_native_prefix(analysis, "original_work::step")
    assert not proof.readonly_module_resources
    assert proof.public()["original_summary_complete"]
    assert any(effect["kind"] == "descriptor_read" and effect["resource"] == "geometry::spacing"
               for effect in proof.effects)


def test_native_source_function_keeps_original_expression_and_hidden_reads(tmp_path):
    source = SOURCE.replace("result=result+x(i)/widths(i)", "result=result+value(result)").replace(
        "end module\nmodule original_work", """contains
real(8) pure function value(x) result(r)
real(8),intent(in)::x
r=x/spacing(0)
end function
end module
module original_work""").replace("use geometry,only:widths=>spacing", "use geometry,only:widths=>spacing,value")
    analysis = analysis_from(tmp_path, source)
    proof = prove_native_prefix(analysis, "original_work::step")
    requirement, = proof.summary["native_function_requirements"]
    assert requirement["callee"] == "geometry::value"
    assert requirement["native_only"]
    assert proof.readonly_module_resources == ("geometry::spacing",)
    assert "geometry::value" not in analysis.routines
    assert not analysis._native_functions


def test_source_out_kills_and_writes_to_other_resources_are_retained(tmp_path):
    source = SOURCE.replace("real(8),intent(inout)::total", "real(8),intent(out)::total").replace(
        "if(enabled) call middle", "total=0.d0\nif(enabled) call middle")
    analysis = analysis_from(tmp_path, source)
    proof = prove_native_prefix(analysis, "original_work::step")
    assert proof.effects[0]["kind"] == "definition_change"
    assert proof.effects[0]["resource"] == "argument::total"
    assert any(effect["kind"] in {"write", "overwrite"} and effect["resource"] == "argument::total"
               for effect in proof.effects)


def test_proof_is_frozen_and_bound_to_original_analysis(tmp_path):
    analysis = analysis_from(tmp_path)
    proof = prove_native_prefix(analysis, "original_work::step")
    public = proof.public()
    public["summary"]["ordered_effects"].clear()
    public["readonly_module_resources"].clear()
    proof.effects.clear()
    proof.summary["complete"] = False
    assert proof.effects
    assert proof.summary["complete"]
    for analysis_arg, proof_arg in ((copy(analysis), proof), (analysis, copy(proof)),
                                    (_fork(analysis, ()), proof)):
        with pytest.raises(CompilationError, match="registered"):
            proof_arg.validate(analysis_arg, "original_work::step")
    with pytest.raises(CompilationError, match="registry belongs"):
        prove_native_prefix(copy(analysis), "original_work::step")


@pytest.mark.parametrize("mutation", ["file", "imports", "bindings", "scope", "kind", "contracts"])
def test_source_and_resolution_mutations_invalidate_registered_proof(tmp_path, mutation):
    analysis = analysis_from(tmp_path)
    proof = prove_native_prefix(analysis, "original_work::step")
    scope = analysis.routines["original_work::step"].scope
    if mutation == "file":
        path = tmp_path / "original.f90"
        path.write_text(path.read_text() + "\n! changed\n")
    elif mutation == "imports":
        analysis.modules["original_work"].imports["widths"] = ("geometry", "other")
    elif mutation == "bindings":
        binding = analysis.modules["geometry"].bindings["spacing"]
        analysis.modules["geometry"].bindings["spacing"] = copy(binding)
    elif mutation == "scope":
        analysis.modules["geometry"] = copy(analysis.modules["geometry"])
    elif mutation == "kind":
        scope.bindings["a"].kind = 4
    else:
        analysis.contracts["unknown"] = {"complete": True}
    with pytest.raises(CompilationError):
        proof.validate(analysis, "original_work::step")


@pytest.mark.parametrize("before_issue", [False, True])
@pytest.mark.parametrize("replace_binding", [False, True])
def test_all_original_scope_binding_authority_is_checked_without_changing_usage(
        tmp_path, before_issue, replace_binding):
    source = "module untouched\nreal(8)::value\nend module\n" + SOURCE
    analysis = analysis_from(tmp_path, source)
    used = set(analysis._source_function_used_scopes)
    proof = None if before_issue else prove_native_prefix(analysis, "original_work::step")
    assert analysis._source_function_used_scopes == used
    scope = analysis.modules["untouched"]
    binding = scope.bindings["value"]
    if replace_binding:
        scope.bindings["value"] = copy(binding)
    else:
        binding.declaring_scope = copy(scope)
    action = prove_native_prefix if before_issue else proof.validate
    with pytest.raises(CompilationError, match="native prefix original binding resolution changed"):
        action(analysis, "original_work::step")
    assert analysis._source_function_used_scopes == used


def test_native_fork_never_mutates_parent_caches_tokens_or_capture_authority(tmp_path):
    analysis = analysis_from(tmp_path)
    original = analysis.summarize("original_work::step")
    analysis.structure("original_work::step")
    names = ("summaries", "_closures", "_structures", "_segments", "_native_sections",
             "_descriptor_proofs", "_joined_completions", "_native_functions", "_native_predicates",
             "_source_provenance", "_allocation_authorizations")
    previous = {name: dict(getattr(analysis, name)) for name in names}
    stats = analysis._summary_cache.stats
    used_scopes = set(analysis._source_function_used_scopes)
    authority = deepcopy(analysis._summary_authority())
    prove_native_prefix(analysis, "original_work::step")
    for name in names:
        assert getattr(analysis, name) == previous[name], name
        for key, value in previous[name].items():
            assert getattr(analysis, name)[key] is value
    assert analysis._summary_cache.stats == stats
    assert analysis._source_function_used_scopes == used_scopes
    assert analysis._summary_authority() == authority
    assert analysis.summarize("original_work::step") == original


TEAM = """module original_team
real(8),allocatable::spacing(:)
contains
subroutine step(a,n,total,bad)
real(8),intent(in)::a(:)
integer,intent(in)::n
real(8),intent(out)::total
logical,intent(out)::bad
integer::i
real(8)::p
total=0.d0
bad=.false.
if(allocated(spacing)) then
!$omp parallel do private(i,p) reduction(max:total) reduction(.or.:bad)
do i=1,n
p=a(i)/spacing(i)
total=max(total,p)
if(p/=p) bad=.true.
enddo
!$omp end parallel do
endif
end subroutine
end module
"""


def test_paired_original_native_real_max_logical_or_joins_before_owner(tmp_path):
    analysis = analysis_from(tmp_path, TEAM)
    proof = prove_native_prefix(analysis, "original_team::step")
    assert proof.readonly_module_resources == ("original_team::spacing",)
    assert proof.summary["native_completion"]["available"] is False
    completion = proof.public()["synchronous_return"]
    assert completion["available"]
    assert completion["requires_no_active_context"]
    group, = completion["procedures"][0]["original_joined_groups"]
    assert [(item["operator"], item["binding"]["type"]) for item in group["reductions"]] == [
        ("max", "real"), (".or.", "logical")]
    assert group["join"].lower() == "!$omp end parallel do"
    assert not completion["numerical_reassociation_authorized"]
    assert not completion["execution_authorized"]


@pytest.mark.parametrize(("operator", "update"), [("min", "min(total,p)"), ("+", "total+p")])
def test_other_predeclared_numeric_reduction_operators(tmp_path, operator, update):
    source = TEAM.replace("reduction(max:total)", "reduction(" + operator + ":total)").replace("max(total,p)", update)
    proof = prove_native_prefix(analysis_from(tmp_path, source), "original_team::step")
    group, = proof.public()["synchronous_return"]["procedures"][0]["original_joined_groups"]
    assert group["reductions"][0]["operator"] == operator


@pytest.mark.parametrize("change", ["nowait", "missing_join", "unknown_operator", "wrong_type",
                                   "unknown_clause", "duplicate", "task", "threadprivate", "unknown_call"])
def test_unproved_native_team_completion_rejects_without_weakening_ordinary_rules(tmp_path, change):
    source = TEAM
    if change == "nowait":
        source = source.replace("!$omp end parallel do", "!$omp end parallel do nowait")
    elif change == "missing_join":
        source = source.replace("!$omp end parallel do", "")
    elif change == "unknown_operator":
        source = source.replace("reduction(max:total)", "reduction(user_max:total)")
    elif change == "wrong_type":
        source = source.replace("reduction(max:total)", "reduction(.or.:total)")
    elif change == "unknown_clause":
        source = source.replace("private(i,p)", "firstprivate(i,p)")
    elif change == "duplicate":
        source = source.replace("private(i,p)", "private(i,p) shared(p)")
    elif change == "task":
        source = source.replace("p=a(i)/spacing(i)", "!$omp task\np=a(i)/spacing(i)\n!$omp end task")
    elif change == "threadprivate":
        source = source.replace("contains", "!$omp threadprivate(spacing)\ncontains", 1)
    else:
        source = source.replace("p=a(i)/spacing(i)", "call unknown(a)\np=a(i)/spacing(i)")
    analysis = analysis_from(tmp_path, source)
    with pytest.raises(CompilationError):
        prove_native_prefix(analysis, "original_team::step")
    assert not getattr(analysis, "_native_prefix_proofs", {})


def test_serial_original_caller_can_complete_native_team_but_team_cannot_nest_it(tmp_path):
    caller = """subroutine caller(a,n,total,bad)
real(8),intent(in)::a(:)
integer,intent(in)::n
real(8),intent(out)::total
logical,intent(out)::bad
call step(a,n,total,bad)
end subroutine
"""
    analysis = analysis_from(tmp_path, TEAM.replace("end module", caller + "end module"))
    proof = prove_native_prefix(analysis, "original_team::caller")
    assert proof.public()["synchronous_return"]["available"]
    assert len(proof.public()["closure"]) == 2
    nested = caller.replace("call step(a,n,total,bad)", """!$omp parallel do private(i)
do i=1,n
call step(a,n,total,bad)
enddo
!$omp end parallel do""").replace("logical,intent(out)::bad", "logical,intent(out)::bad\ninteger::i")
    second = tmp_path / "nested"
    second.mkdir()
    analysis = analysis_from(second, TEAM.replace("end module", nested + "end module"))
    with pytest.raises(CompilationError, match="nested team"):
        prove_native_prefix(analysis, "original_team::caller")


def test_imported_threadprivate_native_input_does_not_gain_uniform_authority(tmp_path):
    source = TEAM.replace("module original_team", """module hidden_values
real(8)::offset
!$omp threadprivate(offset)
end module
module original_team
use hidden_values,only:offset""").replace("p=a(i)/spacing(i)", "p=a(i)/spacing(i)+offset")
    analysis = analysis_from(tmp_path, source)
    with pytest.raises(CompilationError, match="ownership"):
        prove_native_prefix(analysis, "original_team::step")


@pytest.mark.parametrize("budget", [{"procedures": 2}, {"depth": 2}, {"operations": 5}])
def test_original_source_budgets_remain_bounded(tmp_path, budget):
    analysis = analysis_from(tmp_path, **budget)
    with pytest.raises(CompilationError):
        prove_native_prefix(analysis, "original_work::step")


def test_opaque_declared_contract_is_not_a_source_completion_proof(tmp_path):
    source = SOURCE.replace("result=result+x(i)/widths(i)", "result=result+x(i)/widths(i)\ncall external_work(x)").replace(
        "module original_work\n", "module original_work\nuse opaque_library,only:external_work\n")
    contract = {"lifetime": "stable", "escapes": False, "ordering": "serial", "complete": True,
                "descriptor_changes": False, "identity": "caller-assertion",
                "effects": [{"kind": "read", "argument": 0, "section": "whole"}]}
    analysis = analysis_from(tmp_path, source, contracts={"opaque_library::external_work": contract})
    with pytest.raises(CompilationError, match="nonopaque"):
        prove_native_prefix(analysis, "original_work::step")
