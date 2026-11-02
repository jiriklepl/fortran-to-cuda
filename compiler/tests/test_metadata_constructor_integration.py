"""A private metadata constructor keeps original reads without defining arrays."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_external_device_tasks import digest
from compiler.tests.test_external_device_tasks_v2 import OPAQUE_SOURCE, opaque_fixture
from compiler.tests.test_graph_source_owner import FACT


def fixture(tmp_path, *, pure=True):
    source = OPAQUE_SOURCE.replace("real(8)::scratch", "real(8)::scratch\nreal(8)::lengths(3)")
    source = source.replace("use external_types", "use external_types\nreal(8)::metadata(3)=[1.0_8,2.0_8,3.0_8]", 1)
    source = source.replace("function make_state()", ("pure " if pure else "") + "function make_state(values)")
    source = source.replace("type(state_type)::value", "real(8),intent(in)::values(3)\ntype(state_type)::value")
    source = source.replace("value%scratch=3.0_8", "value%scratch=3.0_8\nvalue%lengths=values")
    source = source.replace("cache=make_state()", "cache=make_state(metadata)")
    return opaque_fixture(tmp_path, source)


def emit(tmp_path, *, managed=False, pure=True):
    path, analysis, _, _, _, _, _ = fixture(tmp_path, pure=pure)
    captures = {"argument::a": dict(FACT), "argument::b": dict(FACT)}
    if managed:
        captures["external_fixture::metadata"] = dict(FACT)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): digest(path)},
             "captures": captures}
    builder = ScopeBuilder([path], "external_fixture::advance", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"), contracts=analysis.contracts)
    outputs, report = builder.run()
    return builder, outputs[report["sources"][str(path)]["replacement"]], report


@pytest.mark.parametrize("managed", [False, True])
def test_reached_metadata_constructor_retains_assignment_and_exact_native_read(tmp_path, managed):
    _, text, report = emit(tmp_path, managed=managed)
    scope, = report["scopes"]
    assert not any("whole derived" in item["reason"] for item in report["boundaries"])
    assert "cache=make_state(metadata)" in text.lower()
    assert "if(.not.called)then" in text.lower()
    operations = [operation for unit in scope["planning_segments"] for operation in unit["operations"]["native_operations"]
                  if operation["metadata_constructors"]]
    operation, = operations
    proof, = operation["metadata_constructors"]
    assert operation["resource_effects"] == {"external_fixture::metadata": ["read"]}
    assert {item["resource"] for item in proof["metadata_reads"]} == {"external_fixture::metadata"}
    assert all(proof[key] is False for key in proof if key.endswith("_authorized"))
    assert "external_fixture::advance::cache" not in scope["resource_bindings"]
    if managed:
        assert operation["managed_resources"] == ["external_fixture::metadata"]
        assignment = text.lower().index("cache=make_state(metadata)")
        assert text.rfind("fort_scope_access_begin_v1", 0, assignment) >= 0
        assert text.find("fort_scope_host_end", assignment) > assignment
    else:
        assert operation["managed_resources"] == []
        assert [item["resource"] for item in operation["native_opaque_accesses"]["accesses"]] == ["external_fixture::metadata"]
        assert operation["native_opaque_accesses"]["managed_definitions"] is False


def test_unproved_constructor_keeps_original_close_and_does_not_resurrect_entry_facts(tmp_path):
    _, text, report = emit(tmp_path, pure=False)
    assert any("explicit nonrecursive PURE" in item["reason"] for item in report["boundaries"])
    assignment = text.lower().index("cache=make_state(metadata)")
    assert text.rfind("fort_scope_resident_close_v1", 0, assignment) >= 0


def test_metadata_constructor_projection_retains_exact_issuer_and_rejects_copied_assignment(tmp_path):
    _, analysis, routine, _, _, _, _ = fixture(tmp_path)
    assignment, = [node for node in walk(routine.execution) if type(node).__name__ == "Assignment_Stmt"
                   and str(node.items[0]).lower() == "cache"]
    proof = analysis.metadata_constructor(routine.qualified, assignment)
    selected = analysis.segment_summary(routine.qualified, (assignment,), capture_locals=True)
    marker, = [item for item in selected["operations"] if item["kind"] == "native_metadata_constructor"]
    assert selected["complete"] is True
    assert marker["proof"] == proof.public()
    assert selected["definition_changes"] == []
    assert selected["guaranteed_whole_overwrites"] == []
    with pytest.raises(CompilationError, match="exact qualified opaque receiver"):
        analysis.metadata_constructor(routine.qualified, copy(assignment))
