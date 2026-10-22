"""Failed lexical planning retains its own bounded original-source reasons."""

from hashlib import sha256

import pytest

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.frontend import source_effects
from compiler.frontend.source_effects import _children
from compiler.frontend.summary_cache import SummaryCache
from compiler.ir import CompilationError
from compiler.scopes.lexical import LexicalOwner, lexical_attempt_diagnostics
from compiler.scopes.source import ScopeBuilder

SOURCE = """module diagnostic_owner
implicit none
contains
subroutine step(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=2*a(i)+real(i,8)
enddo
end subroutine
end module
"""
NATIVE_SOURCE = SOURCE.replace("integer::i\ndo i=1,n\na(i)=2*a(i)+real(i,8)\nenddo",
                               "call manage()").replace("end subroutine\nend module", """contains
subroutine manage()
if(n>0) then
call opaque_one(a,n)
endif
if(n>1) then
call opaque_two(a,n)
endif
if(n>2) then
call opaque_three(a,n)
endif
end subroutine
end subroutine
end module""")
REJECTED_SOURCE = SOURCE.replace("a(i)=2*a(i)+real(i,8)", "a(i)=adjust(a(i))").replace(
    "end module", """real(8) function adjust(x)
real(8),intent(in)::x
adjust=x+1
end function
end module""")
NATIVE_TEAM_SOURCE = SOURCE.replace("do i=1,n", "!$omp parallel do private(i)\ndo i=1,n,2").replace(
    "enddo", "enddo\n!$omp end parallel do")


@pytest.fixture(autouse=True)
def isolated_summary_cache(monkeypatch):
    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())


def subject(tmp_path, text=SOURCE, *, policy="sections", profile=None):
    path = tmp_path / "original.f90"
    path.write_text(text)
    digest = sha256(path.read_bytes()).hexdigest()
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): digest},
             "captures": {"argument::a": {"storage": "stable", "initialized": "whole",
                                          "escapes": False, "allocation_changes": False}}}
    builder = ScopeBuilder([path], "diagnostic_owner::step", facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy=policy, profile=profile, scope_execution="reached"))
    return path, builder


def prepare(builder):
    nodes = builder.inline.prepare(_children(builder.entry.execution))
    builder.generated.update(builder.inline.generated)
    builder.numerical_ir.update(builder.inline.ir)
    builder.numerical_reasons.update({procedure: None for procedure in builder.inline.regions})
    return nodes


def test_native_only_internal_member_decline_is_retained_separately(tmp_path):
    path, builder = subject(tmp_path, NATIVE_SOURCE)
    outputs, report = builder.run()
    attempt = report["lexical_owner_attempt"]
    assert attempt["outcome"] == "no_numerical_leaves"
    assert not attempt["planning_succeeded"]
    assert not attempt["admitted"]
    assert not attempt["artifacts_published"]
    assert attempt["source_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert [row["procedure"] for row in attempt["members"]] == [
        "diagnostic_owner::step", "diagnostic_owner::step::manage"]
    assert len(attempt["boundaries"]) == 3
    for row, name in zip(attempt["boundaries"], ("opaque_one", "opaque_two", "opaque_three"), strict=True):
        assert row["procedure"] == "diagnostic_owner::step::manage"
        assert row["source"] == str(path)
        assert name in row["reason"]
        assert path.read_text().splitlines()[row["first_line"] - 1].startswith("call " + name)
        assert row["last_line"] == row["first_line"]
    assert all(row.get("fallback") != "existing bounded source scopes" for row in attempt["boundaries"])
    assert report["scope_count"] == 0
    assert not any("fort_owner_context_" in text for text in outputs.values())
    attempt["boundaries"][0]["reason"] = "caller modified public JSON"
    assert builder.lexical_owner_attempt["boundaries"][0]["reason"] != "caller modified public JSON"


def test_numerical_preparation_rejection_and_lexical_decline_keep_distinct_positions(tmp_path):
    path, builder = subject(tmp_path, REJECTED_SOURCE)
    _outputs, report = builder.run()
    attempt = report["lexical_owner_attempt"]
    assert attempt["outcome"] == "no_numerical_leaves"
    candidates = [row for row in report["boundaries"] if row.get("phase") == "inline_numerical_extraction"]
    assert any("explicit PURE" in row["reason"] for row in candidates)
    assert attempt["boundaries"]
    assert all(row["procedure"] == "diagnostic_owner::step" for row in attempt["boundaries"])
    assert all(row.get("phase") != "inline_numerical_extraction" for row in attempt["boundaries"])
    first = REJECTED_SOURCE.splitlines().index("do i=1,n") + 1
    last = REJECTED_SOURCE.splitlines().index("enddo") + 1
    assert any((row["first_line"], row["last_line"]) == (first, last) for row in attempt["boundaries"])
    assert attempt["source"] == str(path)
    assert not attempt["compiler_source_owner_admitted"]


def test_valid_native_team_retains_its_declined_numerical_route_without_a_boundary(tmp_path):
    path, builder = subject(tmp_path, NATIVE_TEAM_SOURCE)
    owner = LexicalOwner(builder)
    # Exercise the whole-team consumer directly on exact original nodes. Its
    # physical effect mapper requires unit stride; native completion does not.
    assert owner.run(_children(builder.entry.execution)) is False
    assert len(owner.units) == 1
    unit, = owner.units
    assert not unit.leaves
    assert not unit.scope.calls
    assert unit.scope.native
    assert "argument::a" in unit.arrays
    assert not owner.boundaries
    attempt = lexical_attempt_diagnostics(
        builder, owner, outcome="no_numerical_leaves", scope_count_before=0,
        inline_before={"attempts": 0, "operations": 0}, finalize=False)
    assert attempt["boundaries"] == []
    assert attempt["numerical_team_decline_count"] == 2
    assert not attempt["numerical_team_declines_truncated"]
    decline, mixed_decline = attempt["numerical_team_declines"]
    first = NATIVE_TEAM_SOURCE.splitlines().index("!$omp parallel do private(i)") + 1
    last = NATIVE_TEAM_SOURCE.splitlines().index("!$omp end parallel do") + 1
    assert (decline["first_line"], decline["last_line"]) == (first, last)
    reason = "numerical effect iteration guards require unit constant stride"
    assert decline["reason"] == reason
    assert decline["route"] == "whole_numerical_team"
    assert mixed_decline["route"] == "mixed_joined_team"
    assert mixed_decline["reason"]
    assert (mixed_decline["first_line"], mixed_decline["last_line"]) == (first, last)
    assert all(row["continuation"].startswith("proved original native operation;")
               for row in attempt["numerical_team_declines"])
    assert decline["procedure"] == "diagnostic_owner::step"
    assert decline["source"] == str(path)
    assert decline["source_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert not attempt["compiler_source_owner_admitted"]
    assert not builder.outputs
    assert not builder.edits
    decline["reason"] = "modified public diagnostic"
    assert owner.numerical_team_declines[0]["reason"] == reason
    # Serialization has its own shared cap and never removes native units.
    builder.analysis.operation_limit = 0
    limited = lexical_attempt_diagnostics(
        builder, owner, outcome="no_numerical_leaves", scope_count_before=0,
        inline_before={"attempts": 0, "operations": 0}, finalize=False)
    assert limited["numerical_team_decline_count"] == 2
    assert limited["numerical_team_decline_limit"] == 0
    assert limited["numerical_team_declines"] == []
    assert limited["numerical_team_declines_truncated"]
    assert len(owner.units) == 1


def test_failed_team_alternatives_restore_artifacts_without_refunding_preparation(tmp_path, monkeypatch):
    from compiler.scopes import combined_team, joined_team

    source = SOURCE.replace("do i=1,n", "!$omp parallel private(i)\n!$omp do\ndo i=1,n").replace(
        "enddo", "enddo\n!$omp end do\n!$omp end parallel")
    _path, builder = subject(tmp_path, source)
    before = builder.scope_checkpoint()
    whole, mixed = combined_team.CombinedNumericalTeam, joined_team.JoinedTeam

    def fail_whole(builder, nodes):
        whole(builder, nodes)
        builder.outputs["rejected-whole.txt"] = "unpublished whole alternative"
        raise CompilationError("whole alternative sentinel")

    def fail_mixed(builder, nodes, **keywords):
        mixed(builder, nodes, **keywords)
        builder.outputs["rejected-mixed.txt"] = "unpublished mixed alternative"
        raise CompilationError("mixed alternative sentinel")

    monkeypatch.setattr(combined_team, "CombinedNumericalTeam", fail_whole)
    monkeypatch.setattr(joined_team, "JoinedTeam", fail_mixed)
    owner = LexicalOwner(builder)
    assert owner.run(_children(builder.entry.execution)) is False
    assert len(owner.units) == 1
    assert owner.units[0].scope.native
    assert not owner.units[0].leaves
    assert not owner.boundaries
    assert builder.outputs == before[0]
    assert builder.edits == before[1]
    assert builder.generated == before[4]
    assert builder.numerical_ir == before[6]
    assert builder.inline.attempts > 0
    assert builder.inline.operations > 0
    assert [(row["route"], row["reason"]) for row in owner.numerical_team_declines] == [
        ("whole_numerical_team", "whole alternative sentinel"),
        ("mixed_joined_team", "mixed alternative sentinel")]


@pytest.mark.parametrize(("policy", "outcome", "admitted"), [
    ("sections", "accepted", True), ("auto", "successful_native", False)])
def test_accepted_source_owner_and_successful_native_are_distinct(tmp_path, policy, outcome, admitted):
    from compiler.tests.test_offload_profile import scoped_profile

    # Keep exercising the owner diagnostic path with a transfer profile that
    # lacks numerical compute calibration. An absent profile exits earlier.
    _path, builder = subject(tmp_path, policy=policy, profile=scoped_profile() if policy == "auto" else None)
    _outputs, report = builder.run()
    attempt = report["lexical_owner_attempt"]
    assert attempt["outcome"] == outcome
    assert attempt["planning_succeeded"]
    assert attempt["compiler_source_owner_admitted"] is admitted
    assert attempt["admitted"] is admitted
    assert attempt["source_scope_count_delta"] == 1
    assert attempt["finalize_requested"]
    assert attempt["artifacts_published"]
    assert attempt["error"] is None
    assert attempt["execution"] == "source planning only; no application or GPU execution"
    if policy == "auto":
        assert report["scopes"][0]["automatic_preflight"]["generated_owner_invoked"] is False
    else:
        assert report["scopes"][0]["owner_variant"] is not None


def test_failed_finalization_replaces_accepted_snapshot_before_existing_rollback(tmp_path, monkeypatch):
    _path, builder = subject(tmp_path)
    nodes = prepare(builder)
    before = builder.scope_checkpoint()

    def fail_finish():
        raise CompilationError("diagnostic finalization sentinel")

    monkeypatch.setattr(builder, "finish", fail_finish)
    assert builder._attempt_lexical_owner(nodes, finalize=True) is False
    attempt = builder.lexical_owner_attempt
    assert attempt["outcome"] == "error"
    assert attempt["error"] == "diagnostic finalization sentinel"
    assert not attempt["admitted"]
    assert not attempt["artifacts_published"]
    assert builder.outputs == before[0]
    assert builder.edits == before[1]
    assert builder.boundaries[-1] == {
        "reason": "lexical owner unavailable: diagnostic finalization sentinel",
        "fallback": "existing bounded source scopes"}


def test_boundary_cap_applies_across_members_without_mutating_original_rows(tmp_path):
    _path, builder = subject(tmp_path, NATIVE_SOURCE)
    owner = LexicalOwner(builder)
    assert owner.run(prepare(builder)) is False
    # Reduce only serialization capacity after the genuine, normally bounded scan.
    builder.analysis.operation_limit = 2
    attempt = lexical_attempt_diagnostics(
        builder, owner, outcome="no_numerical_leaves", scope_count_before=0,
        inline_before={"attempts": 0, "operations": 0}, finalize=False)
    assert attempt["boundary_count"] == 3
    assert attempt["boundary_limit"] == 2
    assert len(attempt["boundaries"]) == 2
    assert attempt["boundaries_truncated"]
    assert not attempt["members_truncated"]
    assert not attempt["artifacts_published"]
    attempt["boundaries"][0]["reason"] = "changed snapshot"
    child = owner.members["diagnostic_owner::step::manage"]
    assert "opaque_one" in child.boundaries[0]["reason"]


def test_source_file_mutation_still_fails_publication_hard(tmp_path, monkeypatch):
    path, builder = subject(tmp_path)
    original = LexicalOwner.run

    def mutate(owner, nodes):
        path.write_text(path.read_text() + "! changed during lexical planning\n")
        return original(owner, nodes)

    monkeypatch.setattr(LexicalOwner, "run", mutate)
    with pytest.raises(CompilationError, match="source changed"):
        builder.run()


@pytest.mark.parametrize(("policy", "outcome", "admitted"), [
    ("sections", "accepted", True), ("auto", "successful_native", False)])
def test_source_attempt_without_finalization_stages_without_publication(tmp_path, monkeypatch, policy, outcome, admitted):
    path, builder = subject(tmp_path, policy=policy)
    original_digest = sha256(path.read_bytes()).hexdigest()

    def forbidden_finish():
        pytest.fail("source-only attempt must not publish compiler artifacts")

    monkeypatch.setattr(builder, "finish", forbidden_finish)
    assert builder._attempt_lexical_owner(prepare(builder), finalize=False) is True
    attempt = builder.lexical_owner_attempt
    assert attempt["outcome"] == outcome
    assert attempt["planning_succeeded"]
    assert attempt["compiler_source_owner_admitted"] is admitted
    assert attempt["source_scope_count_delta"] == 1
    assert not attempt["finalize_requested"]
    assert not attempt["artifacts_published"]
    assert "scope-manifest.json" not in builder.outputs
    assert sha256(path.read_bytes()).hexdigest() == original_digest
    builder.analysis.inputs.verify()
