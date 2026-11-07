"""Byte field transport stays attached to its exact original tile team."""
from dataclasses import replace

import pytest

from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.ir import CompilationError
from compiler.tests.test_byte_scalar_transport import byte_fixture


def issue(builder, nodes, storage_transport):
    return source_device_task(builder.analysis, builder.entry, nodes, builder=builder,
                              storage_transport=storage_transport)


@pytest.mark.parametrize("initialized", ["whole", "none"])
def test_storage_attachment_preserves_source_and_scalar_read_authority(tmp_path, initialized):
    builder, nodes = byte_fixture(tmp_path)
    for name in ("a", "b"):
        builder.facts["captures"]["argument::" + name]["initialized"] = initialized
    strict, storage = issue(builder, nodes, False), issue(builder, nodes, True)
    assert strict.identity != storage.identity
    assert issue(builder, nodes, False) is strict
    assert issue(builder, nodes, True) is storage
    assert storage.team is strict.team
    record = storage.public()
    assert record["version"] == 3
    assert record["transaction"]["runtime"] == "fort_scope_source_execute_v2"
    assert record["scalar_captures"] == strict.public()["scalar_captures"]
    assert record["definitions_authorized"] is False
    assert all(resource["initialization_authorized"] is False for resource in record["resources"])
    assert all("without initialization" in resource["regions"] for resource in record["resources"])
    provider = emit_tile_task_provider(storage.team, readable_scalars=storage, access_envelopes=True)
    meta = provider.public()
    assert meta["version"] == 10
    assert meta["field_transport"]["source_proof_identity"] == storage.identity
    assert meta["field_transport"]["definitions_authorized"] is False
    assert meta["scalar_transport"]["resources"] == ["argument::scale"]
    assert provider.cuda_source.count("status=fort_scope_source_execute_v2(") == 1
    assert "fort_scope_source_execute_v1(" not in provider.cuda_source
    assert provider.cuda_source.index("const int access_status=prepare_tile_access(") < provider.cuda_source.index("status=fort_scope_source_execute_v2(")
    with pytest.raises(CompilationError, match="registered original proof"):
        replace(storage, storage_transport=False).validate()
    object.__setattr__(storage, "storage_transport", False)
    with pytest.raises(CompilationError, match="lost original source"):
        storage.validate()


@pytest.mark.parametrize("options", [{}, {"access_envelopes": True, "queued_single": True},
    {"access_envelopes": True, "queued_team": True}, {"access_envelopes": True, "batch_target_bytes": 4194304}])
def test_storage_proof_cannot_borrow_an_unqualified_transport(tmp_path, options):
    builder, nodes = byte_fixture(tmp_path)
    proof = issue(builder, nodes, True)
    with pytest.raises(CompilationError, match="source storage transport requires"):
        emit_tile_task_provider(proof.team, readable_scalars=proof, **options)


@pytest.mark.parametrize("mode", [1, "direct", None])
def test_storage_attachment_requires_an_exact_boolean(tmp_path, mode):
    builder, nodes = byte_fixture(tmp_path)
    with pytest.raises(CompilationError, match="original tile team"):
        issue(builder, nodes, mode)
