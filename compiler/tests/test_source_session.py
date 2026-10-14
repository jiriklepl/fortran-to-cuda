"""The active compiler owns immutable source; projections own mutable work."""
from copy import copy, deepcopy
from hashlib import sha256

import pytest

from compiler.frontend.immutable_graph import sealed
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.source_session import ProjectedRoutines
from compiler.ir import CompilationError
from compiler.tests.test_source_syntax_authority import fixture


@pytest.mark.parametrize("mutation", [
    lambda a, r: setattr(r.scope.bindings["a"], "kind", 4),
    lambda a, r: setattr(r, "arguments", ()),
    lambda a, r: setattr(r.scope, "parent", None),
    lambda a, r: r.scope.bindings.clear(),
    lambda a, r: r.scope.imports.update({"x": ("other", "x")}),
    lambda a, r: r.scope.parent.kinds.values.update({"marker": 9}),
    lambda a, r: a.routines.clear(),
    lambda a, r: a._source_scopes.clear(),
    lambda a, r: a.inputs.sources.clear(),
    lambda a, r: a._source_syntax._nodes.clear(),
])
def test_original_semantic_graph_is_read_only(tmp_path, mutation):
    _path, analysis, routine = fixture(tmp_path)
    analysis.seal_session()
    with pytest.raises(CompilationError, match="immutable"):
        mutation(analysis, routine)
    assert analysis._require_original(routine.qualified) is routine


def test_sealing_detaches_retained_semantic_state_and_containers(tmp_path):
    _path, analysis, routine = fixture(tmp_path)
    binding = routine.scope.bindings["a"]
    state, bindings, kinds = vars(binding), routine.scope.bindings, routine.scope.parent.kinds.values
    old_registry = analysis.routines
    identity = analysis.seal_session()
    state["kind"] = 4
    bindings.clear()
    kinds.clear()
    old_registry.clear()
    assert analysis.session_identity == identity
    assert routine.scope.bindings["a"] is binding
    assert binding.kind == 8
    assert routine.scope.parent.kinds.values["marker"] is not None
    assert analysis._require_original(routine.qualified) is routine
    with pytest.raises(TypeError):
        vars(binding)["kind"] = 4
    binding.__getstate__()["kind"] = 4
    assert binding.kind == 8


@pytest.mark.parametrize("copier", [copy, deepcopy])
def test_private_semantic_copies_are_mutable_but_have_no_original_authority(tmp_path, copier):
    _path, analysis, routine = fixture(tmp_path)
    analysis.seal_session()
    projected = copier(routine.scope.bindings["a"])
    assert not sealed(projected)
    projected.kind = 4
    assert routine.scope.bindings["a"].kind == 8
    assert not analysis._source_session.contains(projected)
    local = copy(routine.scope)
    local.bindings["a"] = projected
    assert routine.scope.bindings["a"] is not projected


def test_private_routine_overrides_authenticate_only_their_own_projection(tmp_path):
    _path, analysis, routine = fixture(tmp_path)
    identity = analysis.seal_session()
    projected = analysis.private_projection(routines=True)
    assert isinstance(projected.routines, ProjectedRoutines)
    assert projected.session_identity == identity
    local = copy(routine)
    local.arguments = ()
    projected.routines[routine.qualified] = local
    assert projected._summary_authority()["role"] == "private_projection"
    assert analysis._summary_authority()["role"] == "source"
    with pytest.raises(CompilationError, match="original source-backed"):
        projected._require_original(routine.qualified)
    second = projected.private_projection()
    second.routines[routine.qualified] = routine
    assert projected.routines[routine.qualified] is local
    assert second._require_original(routine.qualified) is routine


@pytest.mark.parametrize("name", ["modules", "routines", "_source_scopes", "sources"])
def test_equal_looking_replacement_registry_cannot_borrow_session_authority(tmp_path, name):
    _path, analysis, routine = fixture(tmp_path)
    analysis.seal_session()
    setattr(analysis, name, dict(getattr(analysis, name)))
    with pytest.raises(CompilationError, match="original registry changed"):
        analysis._require_original(routine.qualified)


def test_captured_text_is_used_until_live_publication_authentication(tmp_path):
    path, analysis, routine = fixture(tmp_path)
    original = path.read_text()
    analysis.seal_session()
    path.write_text(original + "! changed input\n")
    assert analysis.source_text(path) == original
    assert analysis._require_original(routine.qualified) is routine
    with pytest.raises(CompilationError, match="source changed"):
        analysis.verify_publication()
    path.write_text(original)
    analysis.verify_publication()


def test_queries_do_not_rehash_or_reformat_the_original_graph(tmp_path, monkeypatch):
    path, analysis, routine = fixture(tmp_path)
    analysis.seal_session()
    graph = analysis.structure(routine.qualified)
    from compiler.frontend import source_functions, source_objects
    from compiler.frontend.source_inputs import SourceInputs

    def unexpected(*args, **kwargs):
        pytest.fail("an immutable original graph or live file was scanned again")

    monkeypatch.setattr(SourceInputs, "verify_live", unexpected)
    monkeypatch.setattr(SourceEffects, "_routine_signature", staticmethod(unexpected))
    monkeypatch.setattr(source_functions, "_resolution", unexpected)
    monkeypatch.setattr(source_objects, "_specification", unexpected)
    monkeypatch.setattr(source_objects, "_kind_state", unexpected)
    for _ in range(3):
        assert analysis.structure(routine.qualified) is graph
        source_functions._validate_scope(analysis, routine.scope)
        source_objects._TypeBuilder(analysis).scope(routine.scope)
        assert analysis.source_text(path)


def configured(tmp_path):
    path, _analysis, _routine = fixture(tmp_path)
    prepared = tmp_path / "prepared.f90"
    prepared.write_text(path.read_text())
    include = tmp_path / "config.inc"
    include.write_text("! selected configuration\n")
    def digest(name):
        return sha256(name.read_bytes()).hexdigest()
    document = {"schema_version": 1, "source_inputs": {str(path): digest(path)},
                "configuration": {"selected": ["original"]}, "preserves_source_order": True,
                "dependencies": {str(include): digest(include)},
                "entries": [{"source": str(path), "path": str(prepared), "sha256": digest(prepared),
                             "line_map": list(range(1, len(prepared.read_text().splitlines()) + 1))}]}
    analysis = SourceEffects([path], analysis_sources=document)
    analysis.seal_session()
    return analysis, path, prepared, include


@pytest.mark.parametrize("selected", [1, 2, 3])
def test_publication_checks_original_prepared_and_dependency_files(tmp_path, selected):
    values = configured(tmp_path)
    analysis, path = values[0], values[selected]
    path.write_text(path.read_text() + "! changed\n")
    with pytest.raises(CompilationError, match="source changed|analysis source changed"):
        analysis.verify_publication()


def test_configuration_and_line_map_are_immutable_but_public_copies_are_mutable(tmp_path):
    analysis, path, prepared, _include = configured(tmp_path)
    with pytest.raises(CompilationError, match="immutable"):
        analysis.inputs.configuration["selected"].append("different")
    with pytest.raises(CompilationError, match="immutable"):
        analysis.inputs.entries[str(path)]["line_map"][0] = 2
    public = analysis.inputs.public()
    public["configuration"]["selected"].append("private")
    assert list(analysis.inputs.configuration["selected"]) == ["original"]
    assert analysis.source_text(prepared) == prepared.read_text()


def test_standalone_analysis_retains_fresh_validation(tmp_path):
    path, analysis, routine = fixture(tmp_path)
    assert analysis.session_identity is None
    path.write_text(path.read_text() + "! changed\n")
    assert analysis.source_text(path).endswith("! changed\n")
    with pytest.raises(CompilationError, match="source changed"):
        analysis._require_original(routine.qualified)


def test_container_methods_cannot_be_shadowed(tmp_path):
    _path, analysis, routine = fixture(tmp_path)
    analysis.seal_session()
    with pytest.raises(AttributeError):
        analysis.routines.get = lambda name: routine
    with pytest.raises(AttributeError):
        routine.scope.wildcards.index = lambda name: 0
    with pytest.raises(AttributeError):
        routine.execution.content.index = lambda node: 0


def test_freezing_private_copies_does_not_grant_source_authority(tmp_path):
    from compiler.frontend.immutable_graph import GraphFreezer
    from compiler.frontend.source_effects import Routine
    from compiler.frontend.source_inputs import SourceInputs

    path, analysis, routine = fixture(tmp_path)
    analysis.seal_session()
    private = analysis.private_projection(routines=True)
    local = copy(routine)
    local.arguments = ()
    GraphFreezer(owned_types=(Routine,), borrowed=(local.scope, local.execution)).freeze(local)
    private.routines[routine.qualified] = local
    assert private._routine_signature(local) != analysis._routine_signature(routine)
    assert private._summary_authority()["role"] == "private_projection"
    with pytest.raises(CompilationError, match="original source-backed"):
        private._require_original(routine.qualified)

    copied_inputs = copy(analysis.inputs)
    GraphFreezer(owned_types=(SourceInputs,)).freeze(copied_inputs)
    path.write_text(path.read_text() + "! changed\n")
    with pytest.raises(CompilationError, match="source changed"):
        copied_inputs.verify()


def test_scope_roles_bind_only_the_session_and_explicit_overrides(tmp_path):
    from compiler.frontend.source_functions import _scope_roles, _validate_scope_roles
    _path, analysis, routine = fixture(tmp_path)
    analysis.seal_session()
    roles = _scope_roles(analysis)
    assert roles.overrides == ()
    _validate_scope_roles(analysis, roles, {id(routine.scope)})
    projection = analysis.private_projection(routines=True)
    roles = _scope_roles(projection)
    projection.routines[routine.qualified] = copy(routine)
    with pytest.raises(CompilationError, match="resolution authority changed"):
        _validate_scope_roles(projection, roles, {id(routine.scope)})


def test_transaction_shares_original_roots_and_rolls_back_only_mutable_analysis(tmp_path):
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.tests.test_generation_transaction import builder_fixture
    parent = builder_fixture(tmp_path / "source")
    session = parent.analysis._source_session
    roots = dict(session.roots)
    with GenerationTransaction(parent) as transaction:
        candidate = transaction.builder
        assert candidate.analysis.session_identity == session.identity
        assert all(getattr(candidate.analysis, name) is value for name, value in roots.items())
        candidate.analysis.summaries["private-test-entry"] = {"private": True}
    assert "private-test-entry" not in parent.analysis.summaries
    assert all(getattr(parent.analysis, name) is value for name, value in roots.items())


def test_parser_uses_captured_input_during_a_live_change_and_restore(tmp_path, monkeypatch):
    from compiler.frontend.source_inputs import SourceInputs
    from compiler.tests.test_source_syntax_authority import ENTRY, SOURCE

    path = tmp_path / "original.f90"
    path.write_text(SOURCE)
    parse = SourceInputs.parse

    def changed_while_parsing(inputs, original):
        captured = path.read_text()
        path.write_text(captured.replace("b(1)+1", "b(1)+999"))
        try:
            return parse(inputs, original)
        finally:
            path.write_text(captured)

    monkeypatch.setattr(SourceInputs, "parse", changed_while_parsing)
    analysis = SourceEffects([path])
    analysis.seal_session()
    assert "999" not in str(analysis.routines[ENTRY].execution)
    assert "b(1) + 1" in str(analysis.routines[ENTRY].execution)
    analysis.verify_publication()


def test_raw_include_remains_standalone_only_until_preexpanded_capture(tmp_path):
    path = tmp_path / "included.f90"
    include = tmp_path / "offset.inc"
    include.write_text("real(8),parameter::offset=1.d0\n")
    path.write_text("""module included
implicit none
include 'offset.inc'
contains
subroutine advance(a)
real(8),intent(inout)::a(:)
a(1)=a(1)+offset
end subroutine
end module
""")
    analysis = SourceEffects([path])
    assert "offset" in analysis.modules["included"].bindings
    with pytest.raises(CompilationError, match="captured prepared text for INCLUDE"):
        analysis.seal_session()

    prepared = tmp_path / "prepared.f90"
    prepared.write_text(path.read_text().replace("include 'offset.inc'", include.read_text().strip()))
    document = {"schema_version": 1, "preserves_source_order": True,
                "source_inputs": {str(path): sha256(path.read_bytes()).hexdigest()},
                "configuration": {"selected": "original"},
                "dependencies": {str(include): sha256(include.read_bytes()).hexdigest()},
                "entries": [{"source": str(path), "path": str(prepared),
                             "sha256": sha256(prepared.read_bytes()).hexdigest(),
                             "line_map": [1, 2, None, 4, 5, 6, 7, 8, 9]}]}
    configured_analysis = SourceEffects([path], analysis_sources=document)
    configured_analysis.seal_session()
    configured_analysis.verify_publication()
