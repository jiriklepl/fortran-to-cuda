"""Descriptor closures share authentication only until their fresh exit fence."""

from copy import copy
from dataclasses import replace

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend import structured_effects
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.source_syntax import SourceSyntaxSnapshot
from compiler.ir import CompilationError

ENTRY = "descriptor_driver::advance"
LEAF = "leaf_library::touch"
SOURCE = """module leaf_library
contains
subroutine touch(x)
real(8),allocatable,intent(inout)::x(:)
x(lbound(x,1))=x(lbound(x,1))+1
end subroutine
subroutine unused(spare)
real(8),allocatable,intent(inout)::spare(:)
spare(1)=0
end subroutine
end module
module reexports
use leaf_library,only:through=>touch
end module
module branch_library
use reexports,only:through
contains
subroutine left(x)
real(8),allocatable,intent(inout)::x(:)
call through(x)
end subroutine
subroutine right(x)
real(8),allocatable,intent(inout)::x(:)
call through(x)
end subroutine
end module
module descriptor_driver
use branch_library,only:left,right
contains
subroutine advance(a)
real(8),allocatable,intent(inout)::a(:)
integer::local
local=1
associate(alias=>local)
alias=2
end associate
call left(a)
call right(a)
end subroutine
end module
"""


def fixture(tmp_path, **limits):
    path = tmp_path / "renamed_descriptor.f90"
    path.write_text(SOURCE)
    return path, SourceEffects([path], summary_cache=tmp_path / "cache", **limits)


def oracle(analysis):
    # The default implementation still makes independent public structure
    # demands. Compare semantic records and original parser-backed graph IDs.
    return structured_effects.descriptor_stability(analysis, ENTRY)


def test_one_closure_and_cached_query_have_fresh_entry_and_exit(tmp_path, monkeypatch):
    _path, analysis = fixture(tmp_path)
    expected = oracle(analysis)
    originals = dict(analysis._structures)
    original_verify = analysis.inputs.verify
    original_authority = analysis._summary_authority
    original_checkpoint = SourceSyntaxSnapshot.checkpoint
    verifications, authorities, checkpoints = [], [], []

    def verify():
        verifications.append(None)
        return original_verify()

    def authority(*, _syntax=None):
        authorities.append(_syntax)
        return original_authority(_syntax=_syntax)

    def checkpoint(snapshot):
        result = original_checkpoint(snapshot)
        if snapshot is analysis._source_syntax:
            checkpoints.append(result)
        return result

    monkeypatch.setattr(analysis.inputs, "verify", verify)
    monkeypatch.setattr(analysis, "_summary_authority", authority)
    monkeypatch.setattr(SourceSyntaxSnapshot, "checkpoint", checkpoint)
    for _ in range(2):
        result = analysis.descriptor_stability(ENTRY)
        assert result == expected
        result["resources"][0]["reasons"].append("caller mutation")
    assert len(verifications) == len(authorities) == len(checkpoints) == 4
    assert all(left is right for left, right in zip(authorities, checkpoints, strict=True))
    assert len({id(checkpoint) for checkpoint in checkpoints}) == 4
    assert all(analysis._structures[key] is graph for key, graph in originals.items())
    assert not any(value is checkpoint for value in vars(analysis).values() for checkpoint in checkpoints)
    assert analysis.operation_count == 0
    assert analysis._source_function_used_scopes == set()

    before = len(verifications)
    graph = analysis.structure(ENTRY)
    assert graph.identity == expected["structured_identity"]
    assert len(verifications) == before + 1


def test_cold_closure_preserves_records_graphs_and_cache_accounting(tmp_path):
    path, analysis = fixture(tmp_path)
    independent = SourceEffects([path], summary_cache=tmp_path / "oracle-cache")
    expected = oracle(independent)
    result = analysis.descriptor_stability(ENTRY)
    assert result == expected
    assert {key: graph.public() for key, graph in analysis._structures.items()} == {
        key: graph.public() for key, graph in independent._structures.items()}
    assert analysis._summary_cache.stats == independent._summary_cache.stats
    assert len(analysis._structures) == 4  # Root, diamond branches, one shared leaf.
    for graph in analysis._structures.values():
        routine = analysis.routines[graph.procedure]
        assert graph.source_nodes(graph.root_id) == tuple(routine.execution.content)
    assert analysis.operation_count == independent.operation_count == 0


def mutate(path, analysis, change):
    if change == "file":
        original = path.read_bytes()
        path.write_bytes(original + b"! changed during closure\n")
        return lambda: path.write_bytes(original)
    if change in {"ast", "span"}:
        statement, = walk(analysis.routines[LEAF].execution, F.Assignment_Stmt)
        if change == "span":
            original = statement.item.span
            statement.item.span = (1, 1)
            return lambda: setattr(statement.item, "span", original)
        expression = statement.items[2]
        original = expression.items
        expression.items = (original[0], "-", original[2])
        return lambda: setattr(expression, "items", original)
    if change == "imports":
        scope = analysis.modules["reexports"]
        original = scope.imports["through"]
        scope.imports["through"] = ("leaf_library", "unused")
        return lambda: scope.imports.__setitem__("through", original)
    if change == "module_scope":
        original = analysis.modules["reexports"]
        analysis.modules["reexports"] = copy(original)
        return lambda: analysis.modules.__setitem__("reexports", original)
    if change == "routine_scope":
        routine = analysis.routines[LEAF]
        original = routine.scope
        routine.scope = copy(original)
        return lambda: setattr(routine, "scope", original)
    if change == "associate_scope":
        key, original = next((key, record) for key, record in analysis._associate_scopes.items() if record.available)
        analysis._associate_scopes[key] = replace(original, scope=copy(original.scope))
        return lambda: analysis._associate_scopes.__setitem__(key, original)
    if change in {"binding", "associate_binding"}:
        scope = (analysis.routines[LEAF].scope if change == "binding" else
                 next(record.scope for record in analysis._associate_scopes.values() if record.available))
        name = "x" if change == "binding" else "alias"
        original = scope.bindings[name]
        scope.bindings[name] = copy(original)
        assert scope.bindings[name].public() == original.public()
        return lambda: scope.bindings.__setitem__(name, original)
    if change == "contracts":
        original = analysis.contracts["unused_contract"]["identity"]
        analysis.contracts["unused_contract"]["identity"] = "changed"
        return lambda: analysis.contracts["unused_contract"].__setitem__("identity", original)
    assert change == "budget"
    original = analysis.operation_limit
    analysis.operation_limit += 1
    return lambda: setattr(analysis, "operation_limit", original)


@pytest.mark.parametrize("change", [
    "file", "ast", "span", "imports", "binding", "associate_binding", "contracts", "budget",
    "module_scope", "routine_scope", "associate_scope",
])
def test_mid_closure_changes_never_publish_provisional_graphs_or_metadata(tmp_path, monkeypatch, change):
    path, analysis = fixture(tmp_path)
    analysis.contracts["unused_contract"] = {"identity": "original"}
    original_build = structured_effects.build_structure
    restores = []

    def build(current, routine, digest):
        graph = original_build(current, routine, digest)
        if current is analysis and routine.qualified == LEAF and not restores:
            restores.append(mutate(path, analysis, change))
        return graph

    monkeypatch.setattr(structured_effects, "build_structure", build)
    with pytest.raises(CompilationError, match="(source.*(authority|changed)|resolution authority changed)"):
        analysis.descriptor_stability(ENTRY)
    assert len(restores) == 1
    assert not analysis._structures
    assert not analysis._descriptor_proofs
    assert analysis._summary_cache.stats["stores"] == 0
    assert analysis._summary_cache.stats["disk_writes"] == 0
    assert not (tmp_path / "cache").exists()
    assert not analysis._allocation_authorizations
    assert analysis.stable_module_allocatables == frozenset()
    assert analysis.operation_count == 0
    restores[0]()
    monkeypatch.setattr(structured_effects, "build_structure", original_build)
    assert analysis.descriptor_stability(ENTRY) == oracle(analysis)


@pytest.mark.parametrize("change", [
    "file", "ast", "imports", "binding", "associate_binding", "module_scope", "routine_scope", "associate_scope",
])
def test_next_cached_query_reauthenticates_source_and_namespaces(tmp_path, change):
    path, analysis = fixture(tmp_path)
    expected = analysis.descriptor_stability(ENTRY)
    graphs, proofs = dict(analysis._structures), dict(analysis._descriptor_proofs)
    restore = mutate(path, analysis, change)
    with pytest.raises(CompilationError, match="(source.*(authority|changed)|resolution authority changed)"):
        analysis.descriptor_stability(ENTRY)
    assert analysis._structures == graphs
    assert analysis._descriptor_proofs == proofs
    restore()
    assert analysis.descriptor_stability(ENTRY) == expected


def test_unrelated_private_projection_does_not_become_a_namespace_obligation(tmp_path):
    _path, analysis = fixture(tmp_path)
    unused = analysis.routines["leaf_library::unused"]
    unused.execution = copy(unused.execution)
    unused.scope.bindings["spare"] = copy(unused.scope.bindings["spare"])
    assert analysis._summary_authority()["role"] == "private_projection"
    assert analysis.descriptor_stability(ENTRY) == oracle(analysis)


def test_private_descriptor_source_delegates_without_sharing_its_context(tmp_path, monkeypatch):
    _path, original = fixture(tmp_path)
    projected = copy(original)
    projected._descriptor_source = original
    expected = original.descriptor_stability(ENTRY)

    def no_projected_authentication(**_kwargs):
        pytest.fail("private projection entered the original descriptor demand")

    monkeypatch.setattr(projected, "_require_original_global", no_projected_authentication)
    assert projected.descriptor_stability(ENTRY) == expected


@pytest.mark.parametrize("limits", [{"depth": 1}, {"procedures": 2}, {"operations": 1}])
def test_existing_closure_and_structure_caps_keep_their_original_declines(tmp_path, limits):
    _path, analysis = fixture(tmp_path, **limits)
    expected = oracle(analysis)
    assert not expected["complete"]
    assert analysis.descriptor_stability(ENTRY) == expected
    assert len(analysis._structures) <= analysis.procedure_limit
    assert analysis.operation_count == 0
