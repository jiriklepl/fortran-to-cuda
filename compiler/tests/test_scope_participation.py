"""Source-bound participation grants authority only to exact full-team calls."""

from copy import deepcopy
from hashlib import sha256

import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.participation import verify_collective_participation

PROGRAM = """module operators
implicit none
contains
subroutine numerical(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=2*a(i)
enddo
end subroutine
subroutine step(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer,intent(in)::n
call numerical(a,b,n)
end subroutine
end module
module callers
use operators,only:local_step=>step
implicit none
contains
subroutine qualified(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
!$omp parallel default(none) shared(a,b,n) num_threads(4)
call local_step(a,b,n)
!$omp end parallel
end subroutine
subroutine unknown(a,b,n,flag)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
logical,intent(in)::flag
if(flag) call local_step(a,b,n)
end subroutine
end module
"""

ARRAY_FACT = {"storage": "stable", "initialized": "whole", "escapes": False,
              "allocation_changes": False, "association": "shared_whole_storage", "descriptor_uniform": True}
CONTROL_FACT = {"storage": "stable", "escapes": False, "allocation_changes": False,
                "association": "shared_immutable_control"}


def case(tmp_path, source=PROGRAM):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    lines = source.splitlines(keepends=True)
    first = next(index for index, line in enumerate(lines, 1) if line.startswith("call local_step"))
    team_first = next(index for index, line in enumerate(lines, 1) if line.startswith("!$omp parallel"))
    team_last = next(index for index, line in enumerate(lines, 1) if line.startswith("!$omp end parallel"))
    facts = {"schema_version": 2, "sources": analysis.sources,
             "participation": {"kind": "omp_full_team", "dispatch": "qualified_companion",
                               "entry": "operators::step", "expected_omp_level": 1, "host_threads": 4,
                               "call_sites": [{"source": str(path), "first_line": first, "last_line": first,
                                               "span_sha256": sha256(lines[first - 1].encode()).hexdigest(),
                                               "team_first_line": team_first, "team_last_line": team_last,
                                               "uniform_guard": "unconditional"}]},
             "captures": {"argument::a": deepcopy(ARRAY_FACT), "argument::b": deepcopy(ARRAY_FACT),
                          "argument::n": deepcopy(CONTROL_FACT)}}
    return path, analysis, facts


def verify(analysis, facts):
    return verify_collective_participation(analysis, "operators::step", facts, host_threads=4)


def test_verified_site_resolves_use_rename_and_leaves_unknown_callers_untouched(tmp_path):
    path, analysis, facts = case(tmp_path)
    proof = verify(analysis, facts)
    site, = proof.sites
    assert site.routine.qualified == "callers::qualified"
    assert site.shared_array_roots == ("argument::a", "argument::b")
    assert site.immutable_control_roots == ("argument::n",)
    assert site.bindings["argument::a"].name == "a"
    assert site.call_text("fort_collective_step") == "call fort_collective_step(a, b, n)\n"
    public = proof.public()
    assert public["other_callers"] == "unchanged"
    assert public["call_sites"][0]["resource_mapping"] == {
        "argument::a": "argument::a", "argument::b": "argument::b", "argument::n": "argument::n"}
    assert "one_context_and_ordered_collective_execution" in public["runtime_requirements"]
    assert path.read_text() == PROGRAM
    assert "collective" not in path.read_text()


@pytest.mark.parametrize("clause", ["private(a)", "firstprivate(a)", "reduction(+:n)", "if(n>0)",
                                    "default(private)", "num_threads(3)", "shared(a(1))"])
def test_unsupported_parallel_clauses_and_private_captures_are_rejected(tmp_path, clause):
    source = PROGRAM.replace("default(none) shared(a,b,n) num_threads(4)", clause)
    _, analysis, facts = case(tmp_path, source)
    with pytest.raises(CompilationError):
        verify(analysis, facts)


@pytest.mark.parametrize("body", [
    "if(n>0) then\ncall local_step(a,b,n)\nendif",
    "do k=1,2\ncall local_step(a,b,n)\nenddo",
    "!$omp single\ncall local_step(a,b,n)\n!$omp end single",
    "!$omp master\ncall local_step(a,b,n)\n!$omp end master",
    "!$omp critical\ncall local_step(a,b,n)\n!$omp end critical",
    "!$omp task\ncall local_step(a,b,n)\n!$omp end task",
    "!$omp parallel\ncall local_step(a,b,n)\n!$omp end parallel",
    "call unavailable(a)\ncall local_step(a,b,n)",
    "call local_step(a,b,n); call local_step(a,b,n)",
])
def test_partial_unknown_nested_and_ambiguous_call_sites_cannot_claim_unconditional(tmp_path, body):
    source = PROGRAM.replace("call local_step(a,b,n)\n!$omp end parallel", body + "\n!$omp end parallel")
    _, analysis, facts = case(tmp_path, source)
    with pytest.raises(CompilationError):
        verify(analysis, facts)


@pytest.mark.parametrize("actual", ["a(:)", "a(1)", "a+1", "a=a"])
def test_only_whole_name_positional_actuals_are_admitted(tmp_path, actual):
    source = PROGRAM.replace("call local_step(a,b,n)", f"call local_step({actual},b,n)")
    _, analysis, facts = case(tmp_path, source)
    with pytest.raises(CompilationError, match="whole-name"):
        verify(analysis, facts)


@pytest.mark.parametrize("mutation", [
    lambda facts: facts.update(schema_version=1),
    lambda facts: facts.update(sources={}),
    lambda facts: facts["participation"].update(host_threads=2),
    lambda facts: facts["participation"].update(expected_omp_level=True),
    lambda facts: facts["participation"].update(dispatch="global_entry"),
    lambda facts: facts["participation"].update(entry="other::step"),
    lambda facts: facts["participation"]["call_sites"][0].update(span_sha256="0" * 64),
    lambda facts: facts["participation"]["call_sites"][0].update(source_sha256="0" * 64),
    lambda facts: facts["participation"]["call_sites"][0].update(team_span_sha256="0" * 64),
    lambda facts: facts["participation"]["call_sites"][0].update(source=[]),
    lambda facts: facts["participation"]["call_sites"][0].update(caller="other::caller"),
    lambda facts: facts["participation"]["call_sites"][0].update(uniform_guard="flag"),
    lambda facts: facts["participation"]["call_sites"][0].update(first_line=True),
    lambda facts: facts["captures"]["argument::a"].update(descriptor_uniform=False),
    lambda facts: facts["captures"]["argument::a"].update(association="private"),
    lambda facts: facts["captures"]["argument::n"].update(association="shared_whole_storage"),
    lambda facts: facts["captures"].pop("argument::b"),
])
def test_public_contract_validation(tmp_path, mutation):
    _, analysis, facts = case(tmp_path)
    mutation(facts)
    with pytest.raises(CompilationError):
        verify(analysis, facts)


def test_scalar_control_written_in_earlier_team_call_is_not_immutable(tmp_path):
    writer = """subroutine change(n)
integer,intent(inout)::n
n=n+1
end subroutine
"""
    source = PROGRAM.replace("subroutine qualified(a,b,n)", writer + "subroutine qualified(a,b,n)")
    source = source.replace("call local_step(a,b,n)\n!$omp end parallel",
                            "call change(n)\ncall local_step(a,b,n)\n!$omp end parallel")
    _, analysis, facts = case(tmp_path, source)
    with pytest.raises(CompilationError, match="immutable"):
        verify(analysis, facts)


def test_writable_actual_alias_is_rejected(tmp_path):
    source = PROGRAM.replace("call local_step(a,b,n)", "call local_step(a,a,n)")
    _, analysis, facts = case(tmp_path, source)
    with pytest.raises(CompilationError, match="alias"):
        verify(analysis, facts)


def test_readonly_actual_alias_is_not_a_false_write_dependency(tmp_path):
    source = PROGRAM.replace("real(8),intent(out)::b(:)", "real(8),intent(in)::b(:)")
    source = source.replace("b(i)=2*a(i)", "if(a(i)>b(i)) continue")
    source = source.replace("call local_step(a,b,n)", "call local_step(a,a,n)")
    _, analysis, facts = case(tmp_path, source)
    proof = verify(analysis, facts)
    assert proof.sites[0].shared_array_roots == ("argument::a",)


def test_threadprivate_module_capture_is_rejected(tmp_path):
    source = PROGRAM.replace("module callers\nuse operators,only:local_step=>step\nimplicit none\ncontains",
                             "module callers\nuse operators,only:local_step=>step\nimplicit none\n"
                             "real(8),save::a(8)\n!$omp threadprivate(a)\ncontains")
    source = source.replace("subroutine qualified(a,b,n)\nreal(8),intent(in)::a(:)", "subroutine qualified(b,n)")
    _, analysis, facts = case(tmp_path, source)
    with pytest.raises(CompilationError, match="threadprivate"):
        verify(analysis, facts)


def test_default_none_requires_shared_scalar_too(tmp_path):
    source = PROGRAM.replace("shared(a,b,n)", "shared(a,b)")
    _, analysis, facts = case(tmp_path, source)
    with pytest.raises(CompilationError, match="explicitly shared"):
        verify(analysis, facts)


def test_an_unlisted_serial_call_is_not_returned_for_rewrite(tmp_path):
    source = PROGRAM.replace("!$omp parallel default", "call local_step(a,b,n)\n!$omp parallel default")
    _, analysis, facts = case(tmp_path, source)
    record = facts["participation"]["call_sites"][0]
    record["first_line"] += 2
    record["last_line"] += 2
    proof = verify(analysis, facts)
    assert len(proof.sites) == 1
    assert proof.sites[0].first_line == record["first_line"]


def test_source_changed_after_analysis_cannot_authorize_edits(tmp_path):
    path, analysis, facts = case(tmp_path)
    path.write_text(PROGRAM + "! changed\n")
    with pytest.raises(CompilationError, match="changed"):
        verify(analysis, facts)


def test_configured_source_retains_original_edit_and_team_spans(tmp_path):
    path, _, facts = case(tmp_path, "! original-only comment\n" + PROGRAM)
    prepared = tmp_path / "prepared.f90"
    prepared.write_text(PROGRAM)
    record = {"schema_version": 1, "source_inputs": facts["sources"], "preserves_source_order": True,
              "configuration": {}, "dependencies": {},
              "entries": [{"source": str(path), "path": str(prepared),
                           "sha256": sha256(prepared.read_bytes()).hexdigest(),
                           "line_map": list(range(2, len(PROGRAM.splitlines()) + 2))}]}
    analysis = SourceEffects([path], analysis_sources=record)
    proof = verify(analysis, facts)
    site, = proof.sites
    assert site.first_line == 30
    assert site.team_first_line == 29
    assert site.team_last_line == 31
    assert site.source == path
    assert site.source_sha256 == sha256(path.read_bytes()).hexdigest()
    prepared.write_text(PROGRAM + "! stale configured text\n")
    with pytest.raises(CompilationError, match="configured analysis source changed"):
        verify(analysis, facts)


def test_hidden_immutable_control_is_matched_through_import_rename(tmp_path):
    source = PROGRAM.replace("module operators\nimplicit none", "module operators\nimplicit none\ninteger::gain")
    source = source.replace("b(i)=2*a(i)", "b(i)=gain*a(i)")
    source = source.replace("only:local_step=>step", "only:local_step=>step, local_gain=>gain")
    source = source.replace("shared(a,b,n)", "shared(a,b,n,local_gain)")
    _, analysis, facts = case(tmp_path, source)
    facts["captures"]["operators::gain"] = deepcopy(CONTROL_FACT)
    site, = verify(analysis, facts).sites
    assert site.bindings["operators::gain"].root == "operators::gain"
    assert "operators::gain" in site.immutable_control_roots


def test_hidden_mutable_control_is_not_accepted_as_immutable(tmp_path):
    source = PROGRAM.replace("module operators\nimplicit none", "module operators\nimplicit none\ninteger::gain")
    source = source.replace("b(i)=2*a(i)", "b(i)=gain*a(i)").replace("do i=1,n", "gain=gain+1\ndo i=1,n")
    source = source.replace("only:local_step=>step", "only:local_step=>step, gain")
    source = source.replace("shared(a,b,n)", "shared(a,b,n,gain)")
    _, analysis, facts = case(tmp_path, source)
    facts["captures"]["operators::gain"] = deepcopy(CONTROL_FACT)
    with pytest.raises(CompilationError, match="immutable"):
        verify(analysis, facts)


def test_generic_overload_ambiguity_is_not_resolved_by_contract_entry(tmp_path):
    source = PROGRAM.replace("module operators\nimplicit none\ncontains",
                             "module operators\nimplicit none\ninterface entry\n"
                             "module procedure step, other_step\nend interface\ncontains")
    other = """subroutine other_step(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer,intent(in)::n
call numerical(a,b,n)
end subroutine
"""
    source = source.replace("end module\nmodule callers", other + "end module\nmodule callers")
    source = source.replace("local_step=>step", "local_step=>entry")
    _, analysis, facts = case(tmp_path, source)
    with pytest.raises(CompilationError, match="ambiguous"):
        verify(analysis, facts)


def test_exact_call_hash_does_not_allow_incorrect_team_boundary(tmp_path):
    _, analysis, facts = case(tmp_path)
    facts["participation"]["call_sites"][0]["team_first_line"] -= 1
    with pytest.raises(CompilationError, match="boundaries"):
        verify(analysis, facts)


def test_repeated_certified_site_is_not_a_second_dispatch(tmp_path):
    _, analysis, facts = case(tmp_path)
    records = facts["participation"]["call_sites"]
    records.append(deepcopy(records[0]))
    with pytest.raises(CompilationError, match="repeats"):
        verify(analysis, facts)


def test_mapped_effect_expansion_is_bounded_across_team_calls(tmp_path):
    source = PROGRAM.replace("call local_step(a,b,n)\n!$omp end parallel",
                             "call local_step(a,b,n)\n" * 128 + "!$omp end parallel")
    _, analysis, facts = case(tmp_path, source)
    with pytest.raises(CompilationError, match="effect budget exhausted"):
        verify(analysis, facts)


def test_generated_companion_call_respects_free_form_line_limit(tmp_path):
    import re
    name = "capture_" + "a" * 49
    source = re.sub(r"\ba\b", name, PROGRAM)
    _, analysis, facts = case(tmp_path, source)
    facts["captures"]["argument::" + name] = facts["captures"].pop("argument::a")
    site, = verify(analysis, facts).sites
    text = site.call_text("companion_" + "c" * 53)
    assert "&" in text
    assert all(len(line) <= 132 for line in text.splitlines())
    with pytest.raises(CompilationError, match="procedure name"):
        site.call_text("invalid%component")
