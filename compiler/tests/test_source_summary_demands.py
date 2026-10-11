"""Synchronous summaries share frozen authority without retaining demand state."""

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend import source_effects
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.source_syntax import SourceSyntaxSnapshot
from compiler.frontend.summary_cache import SummaryCache
from compiler.ir import CompilationError

ENTRY = "closure_source::update"
CHILD = "closure_source::adjust"
SOURCE = """module closure_source
integer,parameter::marker=3
contains
subroutine update(a,b)
real(8),intent(inout)::a(:),b(:)
call adjust(a,b)
call adjust(b,a)
end subroutine
subroutine adjust(x,y)
real(8),intent(inout)::x(:)
real(8),intent(in)::y(:)
x(1)=y(1)+1
end subroutine
end module
"""


@pytest.fixture
def analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    path = tmp_path / "closure.f90"
    path.write_text(SOURCE)
    return path, SourceEffects([path])


def test_recursive_and_cached_summaries_have_fresh_entry_and_exit_checks(analysis, monkeypatch):
    _path, subject = analysis
    original_checkpoint = SourceSyntaxSnapshot.checkpoint
    checkpoints = []

    def checkpoint(snapshot):
        result = original_checkpoint(snapshot)
        if snapshot is subject._source_syntax:
            checkpoints.append(result)
        return result

    monkeypatch.setattr(SourceSyntaxSnapshot, "checkpoint", checkpoint)
    summary = subject.summarize(ENTRY)
    assert summary["complete"], summary["reasons"]
    assert len(checkpoints) == 2
    assert checkpoints[0] is not checkpoints[1]
    assert set(subject._closures[ENTRY].summaries) == {ENTRY, CHILD}
    assert set(vars(subject._closures[ENTRY])) == {"summaries", "operations"}
    assert not any(value is check for value in vars(subject).values() for check in checkpoints)

    cached = subject.summarize(ENTRY)
    assert cached == summary
    assert len(checkpoints) == 4
    assert len({id(checkpoint) for checkpoint in checkpoints}) == 4


def test_next_call_recomputes_in_place_private_projection_and_restores_source_cache(analysis):
    _path, subject = analysis
    original = subject.summarize(ENTRY)
    statement, = walk(subject.routines[CHILD].execution, F.Assignment_Stmt)
    expression = statement.items[2]
    original_items = expression.items
    expression.items = (original_items[0], "-", original_items[2])

    projected = subject.summarize(ENTRY)
    assert projected["complete"], projected["reasons"]
    assert projected["summary_role"] == "private_projection"
    assert projected["summary_identity"] != original["summary_identity"]
    with pytest.raises(CompilationError, match="source.*authority"):
        subject.structure(ENTRY)

    expression.items = original_items
    restored = subject.summarize(ENTRY)
    assert restored["summary_role"] == "source"
    assert restored["summary_identity"] == original["summary_identity"]


def test_source_function_projection_does_not_inherit_parent_syntax_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "function.f90"
    path.write_text("""module closure_function
contains
pure logical function predicate(x) result(r)
real(8),intent(in)::x
r=x/=x
end function
subroutine update(x,r)
real(8),intent(in)::x
logical,intent(out)::r
r=predicate(x)
end subroutine
end module
""")
    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    subject = SourceEffects([path])
    original_authority = SourceEffects._summary_authority
    uses = []

    def authority(current, *, _syntax=None):
        uses.append((current, _syntax))
        return original_authority(current, _syntax=_syntax)

    monkeypatch.setattr(SourceEffects, "_summary_authority", authority)
    summary = subject.summarize("closure_function::update")
    assert summary["complete"], summary["reasons"]
    assert summary["native_function_requirements"]
    projected = [(current, syntax) for current, syntax in uses if current is not subject]
    assert projected
    assert all(syntax is None for _current, syntax in projected)
    assert any(syntax is not None for current, syntax in uses if current is subject)


@pytest.mark.parametrize("change", ["module", "execution", "binding", "contracts", "nested_contract", "file"])
def test_mid_demand_mutation_rejects_before_store_and_restores_section_cache(analysis, monkeypatch, change):
    path, subject = analysis
    if change == "nested_contract":
        subject.contracts["opaque"] = {"identity": "original-contract"}
    subject.native_sections(CHILD)
    original_sections = dict(subject._native_sections)
    declaration = next(node for node in walk(subject.modules["closure_source"].node, F.Type_Declaration_Stmt)
                       if "marker" in str(node).lower())
    literal, = walk(declaration, F.Int_Literal_Constant)
    statement, = walk(subject.routines[CHILD].execution, F.Assignment_Stmt)
    expression = statement.items[2]
    literal_items, expression_items = literal.items, expression.items
    binding = subject.routines[CHILD].scope.bindings["x"]
    original_kind = binding.kind
    original_stamp = SourceEffects._stamp_summary
    changed = False

    def stamp(current, summary, *, _syntax=None, _authority=None):
        nonlocal changed
        result = original_stamp(current, summary, _syntax=_syntax, _authority=_authority)
        if current is subject and summary["procedure"] == CHILD and not changed:
            changed = True
            if change == "module":
                literal.items = ("4", literal_items[1])
            elif change == "execution":
                expression.items = (expression_items[0], "-", expression_items[2])
            elif change == "binding":
                binding.kind = 4
            elif change == "contracts":
                subject.contracts["opaque"] = {"identity": "changed-during-demand"}
            elif change == "nested_contract":
                subject.contracts["opaque"]["identity"] = "changed-during-demand"
            else:
                path.write_text(SOURCE + "! changed during demand\n")
        return result

    monkeypatch.setattr(SourceEffects, "_stamp_summary", stamp)
    with pytest.raises(CompilationError, match="source.*(authority|changed)"):
        subject.summarize(ENTRY)
    assert changed
    assert not subject._closures
    assert not subject.summaries
    assert subject._native_sections == original_sections
    assert subject._summary_cache.stats["stores"] == 0
    assert subject._summary_cache.stats["disk_writes"] == 0

    literal.items, expression.items = literal_items, expression_items
    binding.kind = original_kind
    subject.contracts.clear()
    path.write_text(SOURCE)
    recovered = subject.summarize(ENTRY)
    assert recovered["complete"], recovered["reasons"]
    assert recovered["summary_role"] == "source"
    assert subject._summary_cache.stats["stores"] > 0


@pytest.mark.parametrize("change", ["contracts", "budget"])
def test_local_cache_still_rechecks_non_syntax_authority(analysis, change):
    _path, subject = analysis
    original = subject.summarize(ENTRY)
    if change == "contracts":
        subject.contracts["opaque"] = {"identity": "new-contract"}
    else:
        subject.operation_limit = 1
    changed = subject.summarize(ENTRY)
    assert changed["summary_identity"] != original["summary_identity"]
    if change == "contracts":
        assert changed["complete"], changed["reasons"]
    else:
        assert not changed["complete"]
        assert any("budget" in reason for reason in changed["reasons"])


def test_imported_closure_rechecks_authority_before_adoption(analysis, monkeypatch):
    path, original = analysis
    original.contracts["opaque"] = {"identity": "original-contract"}
    expected = original.summarize(ENTRY)
    subject = SourceEffects([path])
    subject.contracts["opaque"] = {"identity": "original-contract"}
    original_lookup = SummaryCache.lookup
    stores = subject._summary_cache.stats["stores"]
    changed = False

    def lookup(cache, identity, requested):
        nonlocal changed
        payload = original_lookup(cache, identity, requested)
        if cache is subject._summary_cache and requested == ENTRY and payload is not None and not changed:
            changed = True
            subject.contracts["opaque"]["identity"] = "changed-during-cache-import"
        return payload

    monkeypatch.setattr(SummaryCache, "lookup", lookup)
    with pytest.raises(CompilationError, match="source authority changed during summary demand"):
        subject.summarize(ENTRY)
    assert changed
    assert not subject._closures
    assert not subject.summaries
    assert not subject._native_sections
    assert subject._summary_cache.stats["stores"] == stores

    subject.contracts["opaque"]["identity"] = "original-contract"
    assert source_effects._canonical(subject.summarize(ENTRY)) == source_effects._canonical(expected)


def test_summary_checks_whole_module_before_overlapping_routine_subtrees(analysis, monkeypatch):
    _path, subject = analysis
    module = subject.modules["closure_source"].node
    original_layout = SourceSyntaxSnapshot._layout
    layouts = []

    def layout(snapshot, node):
        if snapshot is subject._source_syntax:
            layouts.append(node)
        return original_layout(snapshot, node)

    monkeypatch.setattr(SourceSyntaxSnapshot, "_layout", layout)
    original = subject.summarize(ENTRY)
    assert original["complete"], original["reasons"]
    assert len(layouts) == 2
    assert all(node is module for node in layouts)

    assert subject.summarize(ENTRY) == original
    assert len(layouts) == 4
    assert all(node is module for node in layouts)

    statement, = walk(subject.routines[CHILD].execution, F.Assignment_Stmt)
    expression = statement.items[2]
    items = expression.items
    expression.items = (items[0], "-", items[2])
    changed = subject.summarize(ENTRY)
    assert changed["summary_role"] == "private_projection"
    assert changed["summary_identity"] != original["summary_identity"]
    with pytest.raises(CompilationError, match="source.*authority"):
        subject.structure(ENTRY)


@pytest.mark.parametrize("cached", [False, True])
def test_authority_construction_mutation_cannot_publish_after_module_memo(analysis, monkeypatch, cached):
    _path, subject = analysis
    if cached:
        subject.summarize(ENTRY)
    stores = subject._summary_cache.stats["stores"]
    statement, = walk(subject.routines[CHILD].execution, F.Assignment_Stmt)
    expression = statement.items[2]
    items = expression.items
    original_authority = SourceEffects._summary_authority
    changed = False

    def authority(current, *, _syntax=None):
        nonlocal changed
        document = original_authority(current, _syntax=_syntax)
        if current is subject and not changed:
            changed = True
            expression.items = (items[0], "-", items[2])
        return document

    monkeypatch.setattr(SourceEffects, "_summary_authority", authority)
    with pytest.raises(CompilationError, match="source authority changed during summary demand"):
        subject.summarize(ENTRY)
    assert changed
    assert subject._summary_cache.stats["stores"] == stores

    expression.items = items
    restored = subject.summarize(ENTRY)
    assert restored["complete"], restored["reasons"]
    assert restored["summary_role"] == "source"
