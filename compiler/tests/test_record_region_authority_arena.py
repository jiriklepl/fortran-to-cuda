"""Shared immutable association facts retain private exact consuming handles."""
import json
from copy import copy
from hashlib import sha256

import pytest

from compiler.frontend import storage_record_regions as regions
from compiler.ir import CompilationError
from compiler.tests.test_storage_record_regions import SOURCE, fixture, prove


def test_one_complete_association_derivation_per_closed_context(tmp_path, monkeypatch):
    data = fixture(tmp_path, sealed=True)
    original = regions._derive
    derived = []

    def counted(*args, **kwargs):
        derived.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(regions, "_derive", counted)
    proof = prove(data)
    public = proof.public()
    for _ in range(5):
        assert prove(data) is proof
        assert proof.validate() is proof
        regions.RecordRegionCheckpoint(data[1], proof).require()
    assert len(derived) == 1
    assert proof._arena.derivation_count("original_owned_record_region") == 1
    assert type(proof._derived.record) is str
    assert all(type(identity) is int for identity in proof._derived.source_nodes)
    assert proof.public() == public
    assert not public["payload_access_authorized"]
    assert not public["native_execution_authorized"]


def test_public_record_and_digest_equal_original_fresh_derivation(tmp_path):
    data = fixture(tmp_path, sealed=True)
    fresh_bytes = regions._derive(data[1], data[2].qualified, data[3], data[4])
    new = prove(data)
    assert new._record == fresh_bytes
    assert new.identity == sha256(fresh_bytes.encode()).hexdigest()
    assert new.public() == {"identity": new.identity, **json.loads(fresh_bytes)}
    assert "dependencies" not in new.public()


def test_equivalent_native_forks_reuse_fact_and_issue_private_handles(tmp_path):
    from compiler.scopes.segments import _native_analysis

    data = fixture(tmp_path, sealed=True)
    original = prove(data)
    left_analysis, right_analysis = (_native_analysis(data[1]) for _ in range(2))
    left = regions.prove_owned_record_region(left_analysis, data[2].qualified, data[3], data[4])
    right = regions.prove_owned_record_region(right_analysis, data[2].qualified, data[3], data[4])
    assert left._context is right._context is original._context
    assert left._derived is right._derived is original._derived
    assert left is not right
    assert left is not original
    assert left._arena.derivation_count("original_owned_record_region") == 1
    assert left_analysis._owned_record_regions is not right_analysis._owned_record_regions
    with pytest.raises(CompilationError):
        left.validate(right_analysis)
    left_analysis.operation_limit += 1
    with pytest.raises(CompilationError):
        left.validate()
    assert right.validate() is right


@pytest.mark.parametrize("field", ["operation_limit", "source_graph_operation_limit", "_stable_module_allocatables"])
def test_supported_policy_changes_create_new_fact_context(tmp_path, field):
    data = fixture(tmp_path, sealed=True)
    first = prove(data)
    analysis = data[1]
    value = analysis.operation_limit + 1 if field == "operation_limit" else {data[2].qualified: 8} if field == "source_graph_operation_limit" else {"regions::field"}
    setattr(analysis, field, value)
    with pytest.raises(CompilationError):
        first.validate()
    second = prove(data)
    assert second is not first
    assert second._context is not first._context
    assert second._derived is not first._derived
    assert second.public()["original_call"] == first.public()["original_call"]
    assert prove(data) is second
    assert second.validate() is second


@pytest.mark.parametrize("fault", ["copy", "registry_token", "record", "receipt", "roles", "derived"])
def test_detached_or_substituted_handle_never_consumes_fact(tmp_path, fault):
    from dataclasses import replace

    data = fixture(tmp_path, sealed=True)
    proof = prove(data)
    if fault == "copy":
        proof = copy(proof)
    elif fault == "registry_token":
        data[1]._owned_record_regions[proof.identity] = copy(proof)
    elif fault == "record":
        object.__setattr__(proof, "_record", "{}")
    elif fault == "receipt":
        object.__setattr__(proof, "_dependencies", copy(proof._dependencies))
    elif fault == "roles":
        object.__setattr__(proof, "_roles", tuple(list(proof._roles)))
    else:
        object.__setattr__(proof, "_derived", replace(proof._derived))
    with pytest.raises(CompilationError):
        proof.validate()


@pytest.mark.parametrize("phase", ["before", "context"])
@pytest.mark.parametrize("fault", ["member_map", "schema_map", "new_schema_map", "binding", "body_name", "spec", "parent", "membership"])
def test_cached_fact_retains_exact_declarations_schema_and_original_structure(tmp_path, monkeypatch, phase, fault):
    from compiler.frontend.authority_context import SourceAuthorityContext

    data = fixture(tmp_path, sealed=True)
    proof = prove(data)
    module = data[1].modules["regions"]
    changed = []

    def mutate():
        if changed:
            return
        changed.append(True)
        if fault == "member_map":
            schema = module.component_types["cell"]
            object.__setattr__(schema, "members", dict(schema.members))
        elif fault == "schema_map":
            object.__setattr__(module, "component_types", dict(module.component_types))
        elif fault == "new_schema_map":
            object.__setattr__(proof._roles[1].scope, "component_types", module.component_types)
        elif fault == "binding":
            object.__setattr__(module.bindings["records"], "rank", 2)
        elif fault == "body_name":
            name = proof._roles[7][0].items[-1]
            object.__setattr__(name, "string", "index")
        elif fault == "spec":
            specification = next(node for node in module.node.content if type(node).__name__ == "Specification_Part")
            object.__setattr__(specification, "content", tuple(reversed(specification.content)))
        elif fault == "parent":
            object.__setattr__(data[3], "parent", None)
        else:
            callee = proof._roles[1].scope.node
            part = callee.parent
            object.__setattr__(part, "content", tuple(node for node in part.content if node is not callee))
            assert callee.parent is part

    if phase == "before":
        mutate()
    else:
        original = SourceAuthorityContext.require

        def after_context(self, analysis):
            result = original(self, analysis)
            if self is proof._context and analysis is data[1]:
                mutate()
            return result

        monkeypatch.setattr(SourceAuthorityContext, "require", after_context)
    with pytest.raises(CompilationError):
        prove(data)
    assert changed
    assert proof._arena.derivation_count("original_owned_record_region") == 1


@pytest.mark.parametrize("fault", ["registry", "token", "policy", "source_owner"])
def test_cached_factory_retains_authority_before_tick_callbacks(tmp_path, fault):
    data = fixture(tmp_path, sealed=True)
    proof = prove(data)
    analysis = data[1]
    before = analysis._owned_record_regions
    fired = []

    def changed():
        fired.append(True)
        if fault == "registry":
            analysis._owned_record_regions = dict(before)
        elif fault == "token":
            before[proof.identity] = copy(proof)
        elif fault == "policy":
            object.__setattr__(analysis, "operation_limit", analysis.operation_limit + 1)
        else:
            object.__setattr__(analysis, "sources", dict(analysis.sources))

    with pytest.raises(CompilationError):
        prove(data, tick=changed)
    assert fired


def test_complete_facts_survive_generation_rollback_but_handles_do_not(tmp_path):
    from compiler._legacy.config import LegacyCompilerOptions, LegacyOffloadConfig
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.scopes.source import ScopeBuilder

    path, *_rest = fixture(tmp_path)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()}, "captures": {}}
    builder = ScopeBuilder([path], "regions::advance", facts=facts,
                           options=LegacyCompilerOptions(), config=LegacyOffloadConfig(policy="sections", scope_execution="graph"))
    analysis = builder.analysis
    caller = analysis.routines["regions::advance"]
    call, = caller.execution.content
    numeric = (analysis.modules["regions"].bindings["field"],)
    context = analysis.source_authority_context()
    parent_arena = context.derivation_arena(analysis)
    with GenerationTransaction(builder):
        temporary = regions.prove_owned_record_region(analysis, caller.qualified, call, numeric)
        assert temporary._arena is not parent_arena
    with pytest.raises(CompilationError):
        temporary.validate()
    current = regions.prove_owned_record_region(analysis, caller.qualified, call, numeric)
    assert current is not temporary
    assert current._derived is temporary._derived
    assert parent_arena.derivation_count("original_owned_record_region") == 1
    assert current.validate() is current


def test_same_analysis_legacy_then_sealed_issues_new_legal_handle(tmp_path):
    data = fixture(tmp_path)
    old = prove(data)
    assert old._context is None
    data[1].seal_session()
    current = prove(data)
    assert current is not old
    assert current._context is not None
    assert current._record == regions._derive(data[1], data[2].qualified, data[3], data[4])
    assert current._arena.derivation_count("original_owned_record_region") == 1
    with pytest.raises(CompilationError):
        old.validate()


def test_incomplete_association_never_installs_immutable_result(tmp_path):
    data = fixture(tmp_path, SOURCE.replace("private(i,p)", "private(i)"), sealed=True)
    context = data[1].source_authority_context()
    for _ in range(2):
        with pytest.raises(CompilationError):
            prove(data)
    assert context.arena.derivation_count("original_owned_record_region") == 0
    assert not data[1]._owned_record_regions


def test_cached_validation_still_checks_live_source_at_publication(tmp_path):
    data = fixture(tmp_path, sealed=True)
    proof = prove(data)
    data[0].write_text(data[0].read_text() + "! changed original\n")
    with pytest.raises(CompilationError, match="source changed"):
        proof.validate()


@pytest.mark.parametrize("fault", ["add_use", "remove_use", "interface", "call_substitution"])
def test_cached_fact_rejects_changed_resolution_namespace_or_callsite(tmp_path, fault):
    data = fixture(tmp_path, SOURCE.replace("implicit none", "use iso_fortran_env,only:int64\nimplicit none"), sealed=True)
    proof = prove(data)
    module = data[1].modules["regions"]
    specification = next(node for node in module.node.content if type(node).__name__ == "Specification_Part")
    use = next(node for node in specification.content if type(node).__name__ == "Use_Stmt")
    if fault == "add_use":
        object.__setattr__(specification, "content", (*specification.content, copy(use)))
    elif fault == "remove_use":
        object.__setattr__(specification, "content", tuple(node for node in specification.content if node is not use))
    elif fault == "interface":
        object.__setattr__(module, "generics", {"assignment(=)": ("regions::touch",)})
    else:
        call = data[3]
        object.__setattr__(data[2].execution, "content", (copy(call),))
    with pytest.raises(CompilationError):
        proof.validate()


@pytest.mark.parametrize("phase", ["first_resolution", "first_context", "cached_context"])
def test_imported_kind_initializer_is_retained_before_callbacks(tmp_path, monkeypatch, phase):
    from compiler.frontend.authority_context import SourceAuthorityContext

    source = ("module kinds\ninteger,parameter::k4=4,k8=8,chosen=k8\nend module\n"
              + SOURCE.replace("implicit none", "use kinds,only:chosen\nimplicit none").replace("real(8)", "real(chosen)"))
    data = fixture(tmp_path, source, sealed=True)
    initializer = data[1].modules["kinds"].kinds.values["chosen"]
    changed = []

    def mutate():
        if not changed:
            changed.append(True)
            object.__setattr__(initializer, "string", "k4")

    if phase == "first_resolution":
        original = regions._original

        def after_original(analysis, name):
            result = original(analysis, name)
            mutate()
            return result

        monkeypatch.setattr(regions, "_original", after_original)
    else:
        proof = prove(data) if phase == "cached_context" else None
        original = SourceAuthorityContext.require

        def after_context(self, analysis):
            result = original(self, analysis)
            if analysis is data[1] and (proof is None or self is proof._context):
                mutate()
            return result

        monkeypatch.setattr(SourceAuthorityContext, "require", after_context)
    with pytest.raises(CompilationError):
        prove(data)
    assert changed


def test_final_private_fence_has_no_source_or_context_verifier_callbacks(tmp_path, monkeypatch):
    from compiler.frontend.authority_context import SourceAuthorityContext

    data = fixture(tmp_path, sealed=True)
    proof = prove(data)

    def unexpected(*_args):
        pytest.fail("delegated verifier after final publication boundary")

    monkeypatch.setattr(SourceAuthorityContext, "require", unexpected)
    monkeypatch.setattr(data[1], "verify_publication", unexpected)
    regions._require_region_attachments(proof)
    object.__setattr__(data[1].modules["regions"], "component_types", {})
    with pytest.raises(CompilationError):
        regions._require_region_attachments(proof)


def test_unrelated_sibling_body_is_not_a_dependency_but_membership_is(tmp_path):
    source = SOURCE.replace("contains\nsubroutine advance", "contains\nsubroutine unused()\nprint *,42\nend subroutine\nsubroutine advance", 1)
    data = fixture(tmp_path, source, sealed=True)
    proof = prove(data)
    unused = data[1].routines["regions::unused"]
    literal = unused.execution.content[0].items[-1].items[0]
    object.__setattr__(literal, "items", ("43", None))
    # No consulted name, declaration, or body depends on this sibling body.
    assert proof.validate() is proof
    target = proof._roles[1].scope.node
    part = target.parent
    object.__setattr__(part, "content", tuple(node for node in part.content if node is not target))
    with pytest.raises(CompilationError):
        proof.validate()


def test_cached_fact_needs_no_encoder_but_retains_final_file_check(tmp_path, monkeypatch):
    data = fixture(tmp_path, sealed=True)
    proof = prove(data)

    def unexpected(*_args):
        pytest.fail("cached record association invoked serializer")

    monkeypatch.setattr(regions, "_encoded", unexpected)
    assert prove(data) is proof
    assert proof.validate() is proof

    def changed():
        data[0].write_text(data[0].read_text() + "! changed before final file verification\n")

    with pytest.raises(CompilationError, match="source changed"):
        prove(data, tick=changed)


@pytest.mark.parametrize("role", [0, 1], ids=["caller", "callee"])
@pytest.mark.parametrize("fault", ["arguments", "execution", "qualified", "scope"])
def test_cached_fact_retains_consumed_routine_fields(tmp_path, role, fault):
    data = fixture(tmp_path, sealed=True)
    proof = prove(data)
    routine = proof._roles[role]
    value = {"arguments": ("changed",), "execution": copy(routine.execution),
             "qualified": "changed::routine", "scope": copy(routine.scope)}[fault]
    object.__setattr__(routine, fault, value)
    with pytest.raises(CompilationError):
        proof.validate()


@pytest.mark.parametrize("role", ["caller", "callee"])
@pytest.mark.parametrize("fault", ["arguments", "execution", "qualified", "scope"])
def test_shared_dependency_capture_retains_terminating_routine_fields(tmp_path, role, fault):
    from compiler.tests.test_termination_authority_arena import issue, sealed_fixture

    data = sealed_fixture(tmp_path)
    proof = issue(data)
    routine = data[2] if role == "caller" else proof._callee
    value = {"arguments": ("changed",), "execution": copy(routine.execution),
             "qualified": "changed::routine", "scope": copy(routine.scope)}[fault]
    object.__setattr__(routine, fault, value)
    with pytest.raises(CompilationError):
        proof.validate(data[1], data[2].qualified if role != "caller" else proof._procedure, data[3])
