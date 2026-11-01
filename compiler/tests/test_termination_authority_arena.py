"""Immutable terminating facts keep private exact callsite handles."""
from copy import copy
from hashlib import sha256

import pytest
from fparser.two.utils import walk

from compiler.frontend import descriptor_termination as termination
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_termination import fixture


def sealed_fixture(tmp_path, **kwargs):
    data = fixture(tmp_path, **kwargs)
    data[1].seal_session()
    return data


def issue(data, analysis=None):
    _path, original, routine, call, _selected = data
    return termination.prove_descriptor_termination(original if analysis is None else analysis, routine.qualified, call)


def test_one_complete_termination_derivation_per_closed_context(tmp_path, monkeypatch):
    data = sealed_fixture(tmp_path)
    original = termination._derive
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(termination, "_derive", counted)
    proof = issue(data)
    public = proof.public()
    for _ in range(5):
        assert issue(data) is proof
        assert proof.validate(data[1], data[2].qualified, data[3]) is proof
        proof.require_attachments()
    assert len(calls) == 1
    assert proof._arena.derivation_count("original_terminating_call") == 1
    assert proof.public() == public
    assert type(proof._derived.record) is str
    assert all(type(identity) is int for identity in proof._derived.source_nodes)


def test_public_record_matches_original_fresh_issuer(tmp_path):
    data = fixture(tmp_path)
    old = issue(data).public()
    fresh = SourceEffects([data[0]])
    fresh.seal_session()
    routine = fresh.routines[data[2].qualified]
    call, = [node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt"]
    new = termination.prove_descriptor_termination(fresh, routine.qualified, call)
    assert new.public() == old
    assert new.identity != new._derived.identity
    assert "dependencies" not in new.public()


def test_equivalent_native_forks_share_fact_but_never_handle_authority(tmp_path):
    from compiler.scopes.segments import _native_analysis

    data = sealed_fixture(tmp_path)
    left_analysis, right_analysis = (_native_analysis(data[1]) for _ in range(2))
    left, right = issue(data, left_analysis), issue(data, right_analysis)
    assert left._context is right._context
    assert left._derived is right._derived
    assert left is not right
    assert left._arena.derivation_count("original_terminating_call") == 1
    with pytest.raises(CompilationError):
        left.validate(right_analysis, data[2].qualified, data[3])
    left_analysis.operation_limit += 1
    with pytest.raises(CompilationError, match="stale or foreign"):
        left.validate(left_analysis, data[2].qualified, data[3])
    assert right.validate(right_analysis, data[2].qualified, data[3]) is right


@pytest.mark.parametrize("field", ["operation_limit", "source_graph_operation_limit", "_stable_module_allocatables"])
def test_policy_changes_select_new_fact_context_and_reject_old_handle(tmp_path, field):
    data = sealed_fixture(tmp_path)
    analysis = data[1]
    first = issue(data)
    value = analysis.operation_limit + 1 if field == "operation_limit" else {data[2].qualified: 8} if field == "source_graph_operation_limit" else {"terminals::stable"}
    setattr(analysis, field, value)
    with pytest.raises(CompilationError, match="stale or foreign"):
        first.validate(analysis, data[2].qualified, data[3])
    second = issue(data)
    assert second is not first
    assert second._context is not first._context
    assert second._derived is not first._derived
    assert second.validate(analysis, data[2].qualified, data[3]) is second


@pytest.mark.parametrize("fault", ["copy", "registry_copy", "foreign_fact", "record", "receipt", "call"])
def test_copied_substituted_or_detached_authority_is_rejected(tmp_path, fault):
    from dataclasses import replace

    data = sealed_fixture(tmp_path)
    proof = issue(data)
    analysis, procedure, call = data[1], data[2].qualified, data[3]
    if fault == "copy":
        proof = copy(proof)
    elif fault == "registry_copy":
        analysis._descriptor_terminating_calls[procedure, id(call)] = copy(proof)
    elif fault == "foreign_fact":
        object.__setattr__(proof, "_derived", replace(proof._derived))
    elif fault == "record":
        object.__setattr__(proof, "_record", copy(dict(proof._record)))
    elif fault == "receipt":
        object.__setattr__(proof, "_source_receipt", copy(proof._source_receipt))
    else:
        call = copy(call)
    with pytest.raises(CompilationError):
        proof.validate(analysis, procedure, call)


def imported_fixture(tmp_path):
    data = fixture(tmp_path, actual="datum")
    path = data[0]
    path.write_text("module data\ninteger,parameter::datum=1\nend module\n" + path.read_text().replace(
        "subroutine work(a)\n", "subroutine work(a)\nuse data,only:datum\n"))
    analysis = SourceEffects([path])
    analysis.seal_session()
    routine = analysis.routines["terminals::work"]
    call, = [node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt"]
    return path, analysis, routine, call, ()


@pytest.mark.parametrize("fault", ["imported_kind", "parameter", "import", "spec_replacement", "body", "parent", "call_movement"])
def test_cached_fact_rejects_changed_import_declaration_body_and_lexical_structure(tmp_path, fault):
    data = imported_fixture(tmp_path)
    proof = issue(data)
    analysis, routine, call = data[1:4]
    module = analysis.modules["data"]
    if fault == "imported_kind":
        object.__setattr__(module.bindings["datum"], "kind", 8)
    elif fault == "parameter":
        literal, = [node for node in walk(module.node) if type(node).__name__ == "Int_Literal_Constant"]
        object.__setattr__(literal, "items", ("2", None))
    elif fault == "import":
        object.__setattr__(routine.scope, "imports", {})
    elif fault == "spec_replacement":
        contents = module.node.content
        object.__setattr__(module.node, "content", tuple(copy(node) if type(node).__name__ == "Specification_Part" else node for node in contents))
    elif fault == "body":
        target = proof._callee.execution.content[-1]
        object.__setattr__(target, "items", ("STOP", None))
    elif fault == "parent":
        object.__setattr__(call, "parent", None)
    else:
        object.__setattr__(routine.execution, "content", ())
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine.qualified, call)


def test_unknown_or_returning_complete_body_never_publishes_a_fact(tmp_path):
    data = sealed_fixture(tmp_path, body="call unknown()")
    context = data[1].source_authority_context()
    for _ in range(2):
        with pytest.raises(CompilationError):
            issue(data)
    assert context.arena.derivation_count("original_terminating_call") == 0
    assert not data[1]._descriptor_terminating_calls


def test_cached_handle_checks_dependencies_after_context_callbacks(tmp_path, monkeypatch):
    from compiler.frontend.authority_context import SourceAuthorityContext

    data = imported_fixture(tmp_path)
    proof = issue(data)
    original = SourceAuthorityContext.require
    reached = []

    def changed(self, analysis):
        result = original(self, analysis)
        if self is proof._context and not reached:
            reached.append(analysis)
            object.__setattr__(analysis.modules["data"].bindings["datum"], "kind", 8)
        return result

    monkeypatch.setattr(SourceAuthorityContext, "require", changed)
    with pytest.raises(CompilationError):
        proof.validate(data[1], data[2].qualified, data[3])
    assert reached


def test_completed_facts_survive_rollback_but_child_handles_do_not(tmp_path):
    from compiler._legacy.config import LegacyCompilerOptions, LegacyOffloadConfig
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.scopes.source import ScopeBuilder

    path, _analysis, _routine, _call, _selected = fixture(tmp_path)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()}, "captures": {}}
    builder = ScopeBuilder([path], "terminals::advance", facts=facts,
                           options=LegacyCompilerOptions(), config=LegacyOffloadConfig(policy="sections", scope_execution="graph"))
    analysis = builder.analysis
    routine = analysis.routines["terminals::work"]
    call, = [node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt"]
    context = analysis.source_authority_context()
    parent_arena = context.derivation_arena(analysis)
    with GenerationTransaction(builder):
        temporary = termination.prove_descriptor_termination(analysis, routine.qualified, call)
        assert temporary._arena is not parent_arena
    with pytest.raises(CompilationError):
        temporary.validate(analysis, routine.qualified, call)
    replacement = termination.prove_descriptor_termination(analysis, routine.qualified, call)
    assert replacement is not temporary
    assert replacement._derived is temporary._derived
    assert parent_arena.derivation_count("original_terminating_call") == 1
    assert replacement.validate(analysis, routine.qualified, call) is replacement


def test_cached_fact_does_not_replace_final_live_source_publication(tmp_path):
    data = sealed_fixture(tmp_path)
    proof = issue(data)
    data[0].write_text(data[0].read_text() + "! changed after source capture\n")
    assert proof.validate(data[1], data[2].qualified, data[3]) is proof
    with pytest.raises(CompilationError, match="source changed"):
        data[1].verify_publication()


def test_final_attachment_fence_never_invokes_context_verifier_callbacks(tmp_path, monkeypatch):
    from compiler.frontend.authority_context import SourceAuthorityContext

    data = sealed_fixture(tmp_path)
    proof = issue(data)

    def unexpected(*_args):
        pytest.fail("context callback after trusted publication boundary")

    monkeypatch.setattr(SourceAuthorityContext, "require", unexpected)
    proof.require_attachments()
    data[1].operation_limit += 1
    with pytest.raises(CompilationError):
        proof.require_attachments()


@pytest.mark.parametrize("callback", ["tick", "context", "factory_context"])
def test_cached_handle_rejects_registry_replacement_across_callbacks(tmp_path, monkeypatch, callback):
    from compiler.frontend.authority_context import SourceAuthorityContext

    data = sealed_fixture(tmp_path)
    proof = issue(data)
    analysis, routine, call = data[1:4]
    registry = analysis._descriptor_terminating_calls
    changed = []

    def replace_registry():
        if not changed:
            changed.append(True)
            analysis._descriptor_terminating_calls = dict(registry)

    if callback == "tick":
        def validate():
            return proof.validate(analysis, routine.qualified, call, tick=replace_registry)
    else:
        original = SourceAuthorityContext.require

        def replace_after_context(self, owner):
            result = original(self, owner)
            if self is proof._context and owner is analysis:
                replace_registry()
            return result

        monkeypatch.setattr(SourceAuthorityContext, "require", replace_after_context)
        def validate():
            return issue(data) if callback == "factory_context" else proof.validate(analysis, routine.qualified, call)
    with pytest.raises(CompilationError, match="registry changed"):
        validate()
    assert changed
    assert analysis._descriptor_terminating_calls is not registry
    assert analysis._descriptor_terminating_calls[routine.qualified, id(call)] is proof


@pytest.mark.parametrize("fault", ["limit", "source_owner", "new_policy"])
def test_raw_policy_and_source_owner_changes_after_callback_reject(tmp_path, fault):
    data = sealed_fixture(tmp_path)
    proof = issue(data)
    analysis, routine, call = data[1:4]

    def change():
        if fault == "limit":
            object.__setattr__(analysis, "operation_limit", analysis.operation_limit + 1)
        elif fault == "source_owner":
            object.__setattr__(analysis, "sources", dict(analysis.sources))
        else:
            object.__setattr__(analysis, "_descriptor_source", analysis)

    with pytest.raises(CompilationError, match="retained source or policy changed"):
        proof.validate(analysis, routine.qualified, call, tick=change)


@pytest.mark.parametrize("fault", ["add_use", "remove_use", "new_interface", "callee_parent"])
def test_cached_fact_rejects_resolution_topology_changes(tmp_path, fault):
    data = imported_fixture(tmp_path)
    proof = issue(data)
    routine = data[2]
    specification, = [node for node in routine.scope.node.content if type(node).__name__ == "Specification_Part"]
    if fault in {"add_use", "remove_use"}:
        use = next(node for node in specification.content if type(node).__name__ == "Use_Stmt")
        content = tuple(specification.content)
        object.__setattr__(specification, "content", (*content, copy(use)) if fault == "add_use" else tuple(node for node in content if node is not use))
    elif fault == "new_interface":
        from fparser.common.readfortran import FortranStringReader
        from fparser.two.parser import ParserFactory

        extra = tmp_path / "interface.f90"
        extra.write_text("module other\ninterface added\nmodule procedure hidden\nend interface\ncontains\nsubroutine hidden()\nend subroutine\nend module\n")
        parsed = ParserFactory().create(std="f2008")(FortranStringReader(extra.read_text()))
        interface, = [node for node in walk(parsed) if type(node).__name__ == "Interface_Block"]
        object.__setattr__(specification, "content", (*specification.content, interface))
    else:
        object.__setattr__(proof._callee.scope.node, "parent", None)
    with pytest.raises(CompilationError):
        proof.validate(data[1], routine.qualified, data[3])


@pytest.mark.parametrize("phase", ["before", "context_callback"])
def test_cached_fact_retains_imported_kind_expression_copy(tmp_path, monkeypatch, phase):
    from compiler.frontend.authority_context import SourceAuthorityContext

    path = tmp_path / "original.f90"
    path.write_text("module kinds\ninteger,parameter::k4=4,k8=8,selected_kind=k4\nend module\n"
                    "module terminals\nuse kinds,only:selected_kind\nuse iso_fortran_env,only:error_unit\ncontains\n"
                    "subroutine fatal(number)\ninteger,intent(in)::number\n"
                    "write(error_unit,'(i0)')number\nstop\nend subroutine\n"
                    "subroutine work()\ncall fatal(1_selected_kind)\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    analysis.seal_session()
    routine = analysis.routines["terminals::work"]
    call, = routine.execution.content
    proof = termination.prove_descriptor_termination(analysis, routine.qualified, call)
    initializer = analysis.modules["kinds"].kinds.values["selected_kind"]
    assert initializer.string == "k4"
    assert proof._arena.derivation_count("original_terminating_call") == 1
    changed = []

    def mutate():
        if not changed:
            changed.append(True)
            object.__setattr__(initializer, "string", "k8")

    if phase == "before":
        mutate()
    else:
        original = SourceAuthorityContext.require

        def change_after_context(self, owner):
            result = original(self, owner)
            if self is proof._context and owner is analysis:
                mutate()
            return result

        monkeypatch.setattr(SourceAuthorityContext, "require", change_after_context)
    with pytest.raises(CompilationError):
        termination.prove_descriptor_termination(analysis, routine.qualified, call)
    assert changed
    assert proof._arena.derivation_count("original_terminating_call") == 1


def test_same_analysis_legacy_issue_then_seal_issues_fresh_arena_handle(tmp_path):
    data = fixture(tmp_path)
    legacy = issue(data)
    assert legacy._context is None
    old_record = legacy.public()
    data[1].seal_session()
    current = issue(data)
    assert current is not legacy
    assert current._context is data[1].source_authority_context()
    assert current.public() == old_record
    assert current._arena.derivation_count("original_terminating_call") == 1
    assert issue(data) is current
    with pytest.raises(CompilationError):
        legacy.validate(data[1], data[2].qualified, data[3])


@pytest.mark.parametrize("fault", ["remove_caller", "remove_callee", "reorder_part"])
def test_cached_handle_retains_exact_enclosing_part_membership_and_order(tmp_path, fault):
    data = sealed_fixture(tmp_path)
    proof = issue(data)
    routine = data[2]
    target = routine.scope.node if fault == "remove_caller" else proof._callee.scope.node
    part = target.parent
    assert type(part).__name__ == "Module_Subprogram_Part"
    original = tuple(part.content)
    assert any(node is target for node in original)
    if fault == "reorder_part":
        replacement = (original[0], *reversed(original[1:]))
        assert replacement != original
    else:
        replacement = tuple(node for node in original if node is not target)
    object.__setattr__(part, "content", replacement)
    assert target.parent is part  # A retained parent pointer alone is insufficient.
    with pytest.raises(CompilationError):
        proof.validate(data[1], routine.qualified, data[3])
