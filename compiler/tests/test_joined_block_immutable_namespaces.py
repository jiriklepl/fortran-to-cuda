"""Sealed namespace receipts borrow owned syntax, not validation callbacks."""
from copy import copy

import pytest

from compiler.frontend import descriptor_associations, descriptor_block_storage
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_block_storage import compose, fixture, source


def with_unreached_helper():
    return source().replace(
        "end module\n",
        "subroutine unrelated(a)\nreal(8)::a(:)\na(1)=2._8\nend subroutine\nend module\n",
    )


def test_sealed_receipts_use_existing_immutable_source_without_walking_unrelated_bodies(tmp_path, monkeypatch):
    _path, analysis, *_ = fixture(tmp_path, text=with_unreached_helper())
    called = []
    original = descriptor_associations._syntax_attachments

    def inspect(node):
        called.append(node)
        return original(node)

    monkeypatch.setattr(descriptor_associations, "_syntax_attachments", inspect)
    saved = descriptor_block_storage._scopes(analysis)
    assert descriptor_block_storage._same_scopes(analysis, saved)
    assert not called
    assert analysis._source_session is not None


def test_unsealed_diagnostic_receipt_still_rejects_unrelated_mutable_syntax_changes(tmp_path):
    _path, analysis, *_ = fixture(tmp_path, text=with_unreached_helper(), sealed=False)
    saved = descriptor_block_storage._scopes(analysis)
    statement = analysis.routines["blocks::unrelated"].execution.content[0]
    statement.items = tuple(copy(item) if index == 2 else item for index, item in enumerate(statement.items))
    assert not descriptor_block_storage._same_scopes(analysis, saved)


def test_sealed_original_syntax_and_namespace_replacement_remain_unsupported(tmp_path):
    _path, analysis, routine, *_ = fixture(tmp_path)
    saved = descriptor_block_storage._scopes(analysis)
    statement = routine.execution.content[0]
    with pytest.raises(CompilationError, match="immutable"):
        statement.parent = None
    with pytest.raises(CompilationError, match="immutable"):
        routine.scope.node = copy(routine.scope.node)
    with pytest.raises(CompilationError, match="immutable"):
        routine.scope.imports["other"] = ("unknown", "other")
    assert descriptor_block_storage._same_scopes(analysis, saved)


def test_equal_source_foreign_session_cannot_reuse_the_captured_namespace_receipt(tmp_path):
    path, analysis, *_ = fixture(tmp_path)
    saved = descriptor_block_storage._scopes(analysis)
    other = SourceEffects([path])
    other.seal_session()
    assert other.session_identity == analysis.session_identity
    assert not descriptor_block_storage._same_scopes(other, saved)


def test_policy_replacement_invalidates_receipt_without_changing_immutable_source(tmp_path):
    _path, analysis, *_ = fixture(tmp_path)
    saved = descriptor_block_storage._scopes(analysis)
    analysis.operation_limit += 1
    with pytest.raises(CompilationError, match="stale or foreign"):
        descriptor_block_storage._same_scopes(analysis, saved)


def test_supported_native_projection_uses_the_same_owned_original_syntax(tmp_path):
    _path, analysis, *_ = fixture(tmp_path)
    projected = analysis.private_projection(routines=True)
    saved = descriptor_block_storage._scopes(projected)
    assert descriptor_block_storage._same_scopes(projected, saved)
    assert projected._source_session is analysis._source_session


@pytest.mark.parametrize("sealed", [False, True])
def test_changed_live_source_is_still_rejected_by_standalone_validation(tmp_path, sealed):
    data = fixture(tmp_path, sealed=sealed)
    path = data[0]
    proof = compose(data)
    path.write_text(path.read_text() + "! changed build input\n")
    with pytest.raises(CompilationError, match="source changed"):
        proof.validate()


@pytest.mark.parametrize("sealed", [False, True])
def test_source_change_from_a_declaration_consumer_is_rejected_before_issuance(tmp_path, sealed):
    data = fixture(tmp_path, sealed=sealed)
    path = data[0]
    fired = False

    def inspect(_scope):
        nonlocal fired
        if not fired:
            fired = True
            path.write_text(path.read_text() + "! changed in declaration consumer\n")

    with pytest.raises(CompilationError, match="source changed"):
        compose(data, scope_validator=inspect)
    assert fired


@pytest.mark.parametrize("consumer", ["receipt", "compose"])
def test_issued_namespace_context_is_reused_without_context_serialization(tmp_path, monkeypatch, consumer):
    data = fixture(tmp_path)
    _path, analysis, routine, *_ = data
    context = analysis._source_authority_context
    assert context is not None
    original_imports = dict(routine.scope.imports)
    original_authority = context.authority
    original_summary = analysis._summary_authority

    def inspect(*args, **kwargs):
        # Fresh context construction calls this serializer without a syntax
        # checkpoint. Graph-key computation legitimately supplies one.
        assert kwargs.get("_syntax") is not None, "issued context must not be serialized again"
        assert analysis._source_authority_context is context
        return original_summary(*args, **kwargs)

    monkeypatch.setattr(analysis, "_summary_authority", inspect)
    descriptor_block_storage._scopes(analysis) if consumer == "receipt" else compose(data)
    assert analysis._source_authority_context is context
    assert context.authority == original_authority
    assert dict(routine.scope.imports) == original_imports
