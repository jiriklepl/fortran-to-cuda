"""Original-position intervals reuse closed-selection control and proofs."""

from copy import copy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.closed_selection import ClosedIntervalOwner, interval_candidates
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_interval_draft import SOURCE
from compiler.tests.test_source_scopes import FACT

MIXED_SOURCE = SOURCE.replace(
    "endif\n!$omp parallel do private(i) schedule(runtime)\ndo i=0,n-1\ns(i)",
    "endif\n!$omp parallel private(i) shared(a,v,s,n) default(none)\n"
    "!$omp do schedule(runtime)\ndo i=0,n-1\nv(i)=v(i)+sum(a)\nenddo\n!$omp end do\n"
    "!$omp do schedule(runtime)\ndo i=0,n-1\ns(i)",
).replace(
    "s(i)=v(i)+a(i)\nenddo\n!$omp end parallel do",
    "s(i)=v(i)+a(i)\nenddo\n!$omp end do\n!$omp end parallel",
)


def builder(path, *, repeated=False, prefix="", suffix="", source=None):
    source = source or SOURCE
    start = "call producer(old,viscosity,n)"
    end = "call momentum(old,viscosity,star,out,n,correcting)"
    if repeated:
        source = source.replace(start, "integer:: repetition\ndo repetition=1,2\n" + prefix + start)
        source = source.replace(end, end + "\n" + suffix + "enddo")
    else:
        if prefix:
            source = source.replace(start, prefix + start)
        if suffix:
            source = source.replace(end, end + "\n" + suffix)
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"interval_fields::" + name: dict(FACT)
                          for name in ("old", "viscosity", "star", "out")}}
    return ScopeBuilder([path], "interval_fields::advance", facts=facts,
                        options=CompilerOptions(gpu_policy="sections"),
                        config=OffloadConfig(policy="sections", scope_execution="reached"))


def call_pair(selected):
    return next(site for site in interval_candidates(selected)
                if sum(type(node).__name__ == "Call_Stmt" for node in site.originals) == 2
                and all(type(node).__name__ in {"Call_Stmt", "Comment"} for node in site.originals))


def owner_for(selected, site):
    return ClosedIntervalOwner(selected, site.originals, site.prefix, site.controls)


def emit_repeated_interval(path, *, mixed=True):
    """Shared host/CUDA fixture through the original selected-site owner."""
    selected = builder(path, repeated=True, source=MIXED_SOURCE if mixed else SOURCE)
    site = call_pair(selected)
    owner = owner_for(selected, site)
    assert owner.run(site.originals)
    outputs, report = selected.finish()
    return selected, owner, outputs, report


def test_repeated_interval_keeps_complete_mixed_team_native_correction(tmp_path):
    from compiler.tests.test_joined_team_source import teams

    selected, owner, outputs, report = emit_repeated_interval(tmp_path / "owner.f90")
    assert owner.execution_plan.numerical_invocations == 3
    team, = teams(report)
    assert team[0]["completion"]["retains_original_team_and_directives"]
    assert len(team[0]["numerical_units"]) == 1
    assert not any(member.boundaries for member in owner.execution_plan.members)
    text = outputs[report["sources"][str(tmp_path / "owner.f90")]["replacement"]]
    assert "v(i)=v(i)+sum(a)" in text
    assert "visits=visits+1" in text
    assert selected.analysis.source_text(tmp_path / "owner.f90") == (tmp_path / "owner.f90").read_text()


@pytest.mark.parametrize("repeated", [False, True])
def test_original_call_interval_uses_one_checked_plan_at_each_reached_site(tmp_path, repeated):
    selected = builder(tmp_path / "owner.f90", repeated=repeated)
    site = call_pair(selected)
    owner = owner_for(selected, site)
    assert owner.run(site.originals)
    outputs, report = selected.finish()
    plan = owner.execution_plan
    assert plan.validate() is plan
    assert plan.numerical_invocations == 3
    assert len(plan.members) == 3
    assert not any(member.boundaries for member in plan.members)
    assert len(plan.public()["resources"]) == 4
    text = outputs[report["sources"][str(tmp_path / "owner.f90")]["replacement"]]
    if repeated:
        start = text.index("do repetition=1,2")
        assert text.index("block", start) < text.index("enddo", start)
        assert text.index(owner.context + " = 0", start) < text.index("enddo", start)
        assert text.index("end block", start) < text.index("enddo", start)
    assert "fort_window_spent_" not in text
    assert text.count("visits=visits+1") == 1
    assert "v(0)=v(0)+a(0)" in text
    assert "fort_scope_host_begin" in text
    assert "fort_scope_host_end" in text


def test_original_interval_rejects_forged_or_noncontiguous_source(tmp_path):
    selected = builder(tmp_path / "owner.f90")
    site = call_pair(selected)
    for originals in ((copy(site.originals[0]), site.originals[1]),
                      tuple(reversed(site.originals))):
        with pytest.raises(CompilationError, match="exact contiguous"):
            ClosedIntervalOwner(selected, originals, site.prefix, site.controls)
    with pytest.raises(CompilationError, match="prefix or enclosing controls"):
        ClosedIntervalOwner(selected, site.originals, (site.originals[0],), site.controls)
    assert not selected.edits


def test_original_interval_does_not_bless_prior_native_prefix(tmp_path):
    selected = builder(tmp_path / "owner.f90", prefix="visits=visits+1\n")
    site = call_pair(selected)
    owner = owner_for(selected, site)
    with pytest.raises(CompilationError, match="prefix changes storage"):
        owner._prepare_lexical(site.originals)


@pytest.mark.parametrize("position", ["prefix", "suffix", "loop_header", "loop_end"])
def test_interval_edit_boundaries_cannot_absorb_same_line_statements(tmp_path, position):
    start = "call producer(old,viscosity,n)"
    end = "call momentum(old,viscosity,star,out,n,correcting)"
    source = SOURCE.replace(start, "integer::local_count\n" + start)
    if position == "prefix":
        source = source.replace(start, "local_count=n; call producer(old,viscosity,local_count)")
    elif position == "suffix":
        source = source.replace(end, end + "; local_count=n")
    else:
        source = source.replace(start, "do local_count=1,2" + ("; " if position == "loop_header" else "\n") + start)
        source = source.replace(end, end + ("; " if position == "loop_end" else "\n") + "enddo")
    selected = builder(tmp_path / "owner.f90", source=source)
    site = call_pair(selected)
    with pytest.raises(CompilationError, match="edit boundary shares a source line"):
        owner_for(selected, site)
    assert not selected.edits
    assert not selected.inline.generated


def test_interval_rejects_shared_statement_line_inside_its_selection(tmp_path):
    start = "call producer(old,viscosity,n)"
    source = SOURCE.replace(start,
        "integer::local_count\nlocal_count=n; call producer(old,viscosity,local_count)")
    selected = builder(tmp_path / "owner.f90", source=source)
    site = next(site for site in interval_candidates(selected) if len(site.originals) == 3
                and type(site.originals[0]).__name__ == "Assignment_Stmt")
    assert not site.prefix
    with pytest.raises(CompilationError, match="statements share a source edit line"):
        owner_for(selected, site)
    assert not selected.edits


@pytest.mark.parametrize("separator", ["; ", "\n", " ! harmless ; comment\n"])
def test_interval_borrowed_child_requires_complete_statement_lines(tmp_path, separator):
    wrapper = """subroutine bridge(a,v,n)
real(8),intent(in)::a(-1:)
real(8),intent(inout)::v(-1:)
integer,intent(in)::n
integer::local_count
local_count=n""" + separator + """call producer(a,v,local_count)
end subroutine
"""
    source = SOURCE.replace("call producer(old,viscosity,n)", "call bridge(old,viscosity,n)")
    source = source.replace("subroutine producer(a,v,n)", wrapper + "subroutine producer(a,v,n)")
    selected = builder(tmp_path / "owner.f90", source=source)
    site = call_pair(selected)
    owner = owner_for(selected, site)
    if separator == "; ":
        with pytest.raises(CompilationError, match="statements share a source edit line"):
            owner.run(site.originals)
        assert not selected.edits
        assert not selected.scopes
    else:
        assert owner.run(site.originals)
        outputs, report = selected.finish()
        text = outputs[report["sources"][str(tmp_path / "owner.f90")]["replacement"]]
        body = text.split("subroutine bridge", 1)[1].split("end subroutine", 1)[0]
        assert body.index("local_count=n") < body.index("call fort_scope_entry_")


@pytest.mark.parametrize("comment", [" ! semicolon ; in a comment", " ! quoted 'literal;value'"])
def test_interval_edit_boundaries_allow_semicolons_that_are_not_statements(tmp_path, comment):
    source = SOURCE.replace("call producer(old,viscosity,n)", "call producer(old,viscosity,n)" + comment)
    source = source.replace("call momentum(old,viscosity,star,out,n,correcting)",
        "call momentum(old,viscosity,star,out,n,correcting)" + comment)
    selected = builder(tmp_path / "owner.f90", source=source)
    owner = owner_for(selected, call_pair(selected))
    assert owner.run(owner.closed_selection)


def test_interval_statement_span_check_allows_quoted_semicolon(tmp_path):
    source = SOURCE.replace("call producer(old,viscosity,n)", "call producer(old,viscosity,n,'literal;value')")
    source = source.replace("subroutine producer(a,v,n)",
        "subroutine producer(a,v,n,label)\ncharacter(*),intent(in)::label")
    selected = builder(tmp_path / "owner.f90", source=source)
    site = call_pair(selected)
    assert owner_for(selected, site).closed_selection == site.originals


@pytest.mark.parametrize("suffix", ["visits=visits+1\n", "old=0.d0\n", "call unknown()\n", "cycle\n"])
def test_repeated_interval_cannot_reuse_entry_facts_after_unproved_suffix(tmp_path, suffix):
    selected = builder(tmp_path / "owner.f90", repeated=True, suffix=suffix)
    site = call_pair(selected)
    owner = owner_for(selected, site)
    with pytest.raises(CompilationError, match="repeated original interval"):
        owner._prepare_lexical(site.originals)


@pytest.mark.parametrize("statement", ["return", "stop", "cycle"])
def test_interval_selection_with_early_exit_cannot_open_context(tmp_path, statement):
    selected = builder(tmp_path / "owner.f90", suffix=statement + "\n")
    site = next(site for site in interval_candidates(selected)
                if any(str(node).strip().lower() == statement for node in site.originals))
    with pytest.raises(CompilationError, match="lifetime changes or early exits"):
        owner_for(selected, site)
    assert not selected.edits
