"""Invocation environment scans share syntax work only within one fresh demand."""

from types import SimpleNamespace

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.region_dispatch import InlineRegions

ENTRY = "dispatch_origin::advance"
VISITED = {ENTRY, "dispatch_branches::left", "dispatch_branches::right", "dispatch_leaf::adjust"}


def fixture(tmp_path, *, ieee=False):
    leaf = tmp_path / "leaf.f90"
    use = ("use, intrinsic::ieee_exceptions,only:set_mode=>ieee_set_halting_mode,ieee_invalid\n"
           if ieee else "")
    operation = "call set_mode(ieee_invalid,.false.)\n" if ieee else ""
    leaf.write_text("""module dispatch_leaf
contains
subroutine adjust(a)
""" + use + """real(8),intent(inout)::a(:)
""" + operation + """a=a+1
end subroutine
subroutine unused(a)
real(8),intent(inout)::a(:)
a=a+2
end subroutine
end module
""")
    branches = tmp_path / "branches.f90"
    branches.write_text("""module dispatch_branches
use dispatch_leaf,only:renamed=>adjust
contains
subroutine left(a)
real(8),intent(inout)::a(:)
call renamed(a)
end subroutine
subroutine right(a)
real(8),intent(inout)::a(:)
call renamed(a)
end subroutine
end module
""")
    root = tmp_path / "root.f90"
    root.write_text("""module dispatch_origin
use dispatch_branches,only:imported_left=>left,right
contains
subroutine advance(a)
real(8),intent(inout)::a(:)
call imported_left(a)
call right(a)
end subroutine
end module
""")
    analysis = SourceEffects([root, branches, leaf])
    builder = SimpleNamespace(analysis=analysis, entry=analysis.routines[ENTRY], _numerical_environment_root=ENTRY)
    return (root, branches, leaf), analysis, InlineRegions(builder)


@pytest.mark.parametrize("ieee", [False, True])
def test_one_traversal_uses_two_fresh_checkpoints_and_retains_original_ieee_resolution(tmp_path, monkeypatch, ieee):
    _paths, analysis, inline = fixture(tmp_path, ieee=ieee)
    checkpoints, requests = [], []
    original_checkpoint = analysis._source_syntax.checkpoint
    original_require = analysis._require_original

    def checkpoint():
        result = original_checkpoint()
        checkpoints.append(result)
        return result

    def require(procedure, *, _syntax=None):
        requests.append((procedure, _syntax))
        return original_require(procedure, _syntax=_syntax)

    monkeypatch.setattr(analysis._source_syntax, "checkpoint", checkpoint)
    monkeypatch.setattr(analysis, "_require_original", require)
    assert inline.native_environment_required() is ieee
    assert len(checkpoints) == 2
    entered = {procedure for procedure, syntax in requests if syntax is checkpoints[0]}
    exited = {procedure for procedure, syntax in requests if syntax is checkpoints[1]}
    assert entered == exited
    assert ENTRY in entered
    assert "dispatch_leaf::adjust" in entered
    assert entered <= VISITED
    assert all(syntax is checkpoints[0] or syntax is checkpoints[1] for _procedure, syntax in requests)
    if not ieee:
        assert entered == VISITED
    # Keep the existing per-dispatcher Boolean memo; no checkpoint is retained.
    before = tuple(requests)
    assert inline.native_environment_required() is ieee
    assert tuple(requests) == before
    assert len(checkpoints) == 2


@pytest.mark.parametrize("limit", ["procedure_limit", "operation_limit"])
def test_bounded_environment_scan_remains_conservative(tmp_path, limit):
    _paths, analysis, inline = fixture(tmp_path)
    setattr(analysis, limit, 1)
    assert inline.native_environment_required()


def test_exit_rechecks_a_visited_child_namespace_before_memoizing(tmp_path, monkeypatch):
    _paths, analysis, inline = fixture(tmp_path)
    child = analysis.routines["dispatch_branches::right"]
    binding = child.scope.bindings["a"]
    original_kind = binding.kind
    original_candidates = analysis._candidates

    def candidates(scope, name, visited=frozenset()):
        result = original_candidates(scope, name, visited)
        if scope is child.scope:
            binding.kind = 4
        return result

    monkeypatch.setattr(analysis, "_candidates", candidates)
    with pytest.raises(CompilationError, match="original source-backed procedure authority"):
        inline.native_environment_required()
    assert inline._native_environment_required is None
    binding.kind = original_kind
    monkeypatch.setattr(analysis, "_candidates", original_candidates)
    assert not inline.native_environment_required()


def test_exit_rechecks_unvisited_module_syntax_before_memoizing(tmp_path, monkeypatch):
    _paths, analysis, inline = fixture(tmp_path)
    unused = analysis.routines["dispatch_leaf::unused"]
    assignment, = walk(unused.execution, F.Assignment_Stmt)
    original_items = assignment.items
    original_candidates = analysis._candidates

    def candidates(scope, name, visited=frozenset()):
        result = original_candidates(scope, name, visited)
        assignment.items = (*original_items[:2], original_items[0])
        return result

    monkeypatch.setattr(analysis, "_candidates", candidates)
    with pytest.raises(CompilationError, match="unchanged original source-backed module authority"):
        inline.native_environment_required()
    assert inline._native_environment_required is None
    assignment.items = original_items
    monkeypatch.setattr(analysis, "_candidates", original_candidates)
    assert not inline.native_environment_required()


def test_new_dispatcher_checks_fresh_original_source_instead_of_reusing_another_builders_memo(tmp_path):
    paths, analysis, inline = fixture(tmp_path)
    assert not inline.native_environment_required()
    paths[-1].write_text(paths[-1].read_text() + "! changed original source\n")
    fresh = InlineRegions(SimpleNamespace(analysis=analysis, entry=analysis.routines[ENTRY],
                                         _numerical_environment_root=ENTRY))
    with pytest.raises(CompilationError, match="source changed"):
        fresh.native_environment_required()
    assert fresh._native_environment_required is None
