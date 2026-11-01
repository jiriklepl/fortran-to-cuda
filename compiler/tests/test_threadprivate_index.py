"""Repeated ownership queries retain exact original and projected directives."""

from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects, _part
from compiler.ir import CompilationError
from compiler.scopes import participation

SOURCE = """module values
real(8),save::shared_value
!$omp threadprivate(shared_value)
end module
module work
use values,only:renamed=>shared_value
contains
subroutine step()
real(8),save::a,b
!$omp threadprivate(a)
a=a+renamed
end subroutine
end module
"""


def analysis_from(tmp_path, source=SOURCE):
    path = tmp_path / "source.f90"
    path.write_text(source)
    return SourceEffects([path])


def original_roots(analysis):
    """The pre-index algorithm, including original binding resolution."""
    roots = set()
    scopes = [*analysis.modules.values(), *(routine.scope for routine in analysis.routines.values())]
    for scope in scopes:
        for node in walk(_part(scope.node, "Specification_Part")):
            directive = participation._directive(node)
            if directive is not None and directive.startswith("threadprivate"):
                for name in directive.partition("(")[2].removesuffix(")").split(","):
                    binding = analysis._binding(scope, name.strip())
                    assert binding is not None
                    roots.add(binding.root)
    return roots


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("source", [SOURCE, SOURCE.replace("!$omp threadprivate(a)", ""),
                                    SOURCE.replace("!$omp threadprivate(shared_value)", "")])
def test_index_retains_exact_canonical_roots_and_imports(tmp_path, source, sealed):
    analysis = analysis_from(tmp_path, source)
    expected = original_roots(analysis)
    if sealed:
        analysis.seal_session()
    for _ in range(3):
        assert participation._threadprivate(analysis) == expected


def test_sealed_queries_do_not_repeat_specification_walks(tmp_path, monkeypatch):
    analysis = analysis_from(tmp_path)
    expected = original_roots(analysis)
    analysis.seal_session()

    def unexpected(*args, **kwargs):
        pytest.fail("an original specification was walked again")

    monkeypatch.setattr(participation, "walk", unexpected)
    for current in (analysis, analysis.private_projection(), analysis.private_projection(routines=True)):
        assert participation._threadprivate(current) == expected


def test_foreign_projection_syntax_is_walked_and_never_receives_original_authority(tmp_path, monkeypatch):
    analysis = analysis_from(tmp_path)
    analysis.seal_session()
    other = tmp_path / "other"
    other.mkdir()
    foreign = analysis_from(other, SOURCE.replace("threadprivate(a)", "threadprivate(b)"))
    projected = analysis.private_projection(routines=True)
    routine = copy(analysis.routines["work::step"])
    routine.scope = copy(routine.scope)
    routine.scope.node = foreign.routines["work::step"].scope.node
    registry = projected.routines
    with pytest.raises(CompilationError, match="routine projection is immutable; use project_routine"):
        registry[routine.qualified] = routine
    assert projected.routines is registry
    projected.project_routine(routine.qualified, routine)
    scans = []
    original_walk = participation.walk

    def recording(node):
        scans.append(node)
        return original_walk(node)

    monkeypatch.setattr(participation, "walk", recording)
    for _ in range(2):
        assert participation._threadprivate(projected) == {"values::shared_value", "work::step::b"}
    assert scans == [_part(routine.scope.node, "Specification_Part")] * 2
    with pytest.raises(CompilationError, match="original source-backed"):
        projected._require_original(routine.qualified)
    assert participation._threadprivate(analysis) == {"values::shared_value", "work::step::a"}


def test_syntax_index_does_not_cache_projected_binding_resolution(tmp_path):
    analysis = analysis_from(tmp_path)
    analysis.seal_session()
    projected = analysis.private_projection(routines=True)
    routine = copy(analysis.routines["work::step"])
    routine.scope = copy(routine.scope)
    routine.scope.bindings["a"] = routine.scope.bindings["b"]
    registry = projected.routines
    with pytest.raises(CompilationError, match="routine projection is immutable; use project_routine"):
        registry[routine.qualified] = routine
    assert projected.routines is registry
    projected.project_routine(routine.qualified, routine)
    assert participation._threadprivate(projected) == {"values::shared_value", "work::step::b"}
    assert participation._threadprivate(analysis) == {"values::shared_value", "work::step::a"}


@pytest.mark.parametrize("directive", ["threadprivate(missing)", "threadprivate(/state/)"])
def test_unresolved_directives_remain_lazy_errors(tmp_path, directive):
    analysis = analysis_from(tmp_path, SOURCE.replace("threadprivate(a)", directive))
    analysis.seal_session()
    with pytest.raises(CompilationError, match="unresolved threadprivate"):
        participation._threadprivate(analysis)


def test_replaced_original_registry_cannot_reuse_syntax_index(tmp_path):
    analysis = analysis_from(tmp_path)
    analysis.seal_session()
    expected = participation._threadprivate(analysis)
    registry = analysis.modules
    replacement = dict(registry)
    with pytest.raises(CompilationError, match="original source registry is immutable"):
        analysis.modules = replacement
    assert analysis.modules is registry
    assert participation._threadprivate(analysis) == expected
    # Preserve the syntax-index consumer check under deliberate internal corruption.
    object.__setattr__(analysis, "modules", replacement)
    with pytest.raises(CompilationError, match="original registry changed"):
        participation._threadprivate(analysis)


def test_generated_collective_artifacts_are_identical_to_fresh_walks(tmp_path, monkeypatch):
    from compiler.frontend import source_effects
    from compiler.frontend.source_session import SourceSession
    from compiler.frontend.summary_cache import SummaryCache
    from compiler.tests.test_collective_source_scopes import generate

    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    _, indexed, indexed_report, _ = generate(tmp_path)

    def fresh_walks(self, analysis):
        self.verify(analysis)
        return {}

    monkeypatch.setattr(SourceSession, "specification_comments", fresh_walks)
    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    _, original, original_report, _ = generate(tmp_path)
    assert indexed == original
    assert indexed_report == original_report
