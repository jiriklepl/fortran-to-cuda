"""A complete source team stays an issuer-owned graph/emission authority."""
from copy import copy
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_device_tasks import source_device_task
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_tile_private import SOURCE
from compiler.tests.test_tile_private_team import team_fixture


def test_source_task_wraps_real_complete_team_without_definition_or_lifetime_mutation_authority(tmp_path):
    path, analysis, routine, nodes = team_fixture(tmp_path)
    before = path.read_bytes()
    proof = source_device_task(analysis, routine, nodes)
    assert proof.validate() is proof
    assert source_device_task(analysis, routine, nodes) is proof
    record = proof.public()
    assert record["kind"] == "original_tile_private_team"
    assert [(item["resource"], item["access"]) for item in record["resources"]] == [
        ("argument::a", "read"), ("argument::b", "readwrite")]
    assert record["team"] == proof.team.public()
    assert record["definitions_authorized"] is False
    assert record["motion_authorized"] is False
    assert record["placement_authorized"] is False
    assert record["pricing"] == "unpriced; explicit sections only"
    assert path.read_bytes() == before
    for forged in (copy(proof), replace(proof)):
        with pytest.raises(CompilationError, match="registered original proof"):
            forged.validate()
    with pytest.raises(CompilationError, match="registered original proof"):
        proof.validate(nodes=nodes[1:])


def test_source_task_public_record_cannot_mutate_original_numerical_work_or_field_access(tmp_path):
    _, analysis, routine, nodes = team_fixture(tmp_path)
    proof = source_device_task(analysis, routine, nodes)
    record = proof.public()
    record["team"]["unit_proof_identities"] = []
    record["resources"][0]["access"] = "overwrite"
    assert proof.public()["team"]["unit_proof_identities"]
    assert proof.public()["resources"][0]["access"] == "read"
    assert proof.validate() is proof


def test_conditional_scalar_distinguishes_byte_storage_from_whole_definition_and_keeps_FP_guard(tmp_path):
    source = SOURCE.replace("advance(a,b,n,t)", "advance(a,b,n,t,scale)")
    source = source.replace("integer,intent(in)::n,t", "integer,intent(in)::n,t\nreal(8),intent(in)::scale")
    source = source.replace("  b(i)=b(i)+(tmp(li+1)-tmp(li-1))", """  if(mod(i,2)==0)then
   b(i)=b(i)+scale*(tmp(li+1)-tmp(li-1))
  else
   b(i)=b(i)-(tmp(li+1)-tmp(li-1))
  endif""")
    path, analysis, routine, nodes = team_fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="independent whole-defined"):
        source_device_task(analysis, routine, nodes)
    facts = {"schema_version": 1, "participation": "serial", "sources": dict(analysis.sources),
             "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "scale")}}
    facts["captures"]["argument::scale"]["initialized"] = "none"
    builder = ScopeBuilder([path], routine.qualified, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"))
    analysis, routine = builder.analysis, builder.entry
    nodes = tuple(routine.execution.content)
    byte_proof = source_device_task(analysis, routine, nodes, builder=builder)
    assert [b.root for b in byte_proof.byte_scalars] == ["argument::scale"]
    assert all(b.root != "argument::scale" for b in byte_proof.readable_scalars)
    facts["captures"]["argument::scale"]["initialized"] = "whole"
    builder = ScopeBuilder([path], routine.qualified, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"))
    proof = source_device_task(builder.analysis, builder.entry, tuple(builder.entry.execution.content), builder=builder)
    capture, = [item for item in proof.public()["scalar_captures"] if item["resource"] == "argument::scale"]
    assert capture["authority"]["kind"] == "explicit checked whole-defined scalar entry fact"
    assert capture["authority"]["finite_or_constant_authorized"] is False
    assert any(binding.root == "argument::scale" for binding in proof.readable_scalars)
    builder.facts["captures"]["argument::scale"]["initialized"] = "none"
    with pytest.raises(CompilationError, match="lost original"):
        proof.validate()


def test_nested_module_scalar_fact_uses_exact_child_builder_and_original_module_binding(tmp_path):
    source = SOURCE.replace("module stencil\n", "module stencil\nreal(8)::scale\n")
    source = source.replace("b(i)=b(i)+(tmp(li+1)-tmp(li-1))",
                            "if(mod(i,2)==0)b(i)=b(i)+scale*(tmp(li+1)-tmp(li-1))")
    source = source.rsplit("end module", 1)[0] + """subroutine outer(left,right,n,t)
integer,intent(in)::n,t
real(8),intent(in)::left(-3:)
real(8),intent(inout)::right(-3:)
call advance(left,right,n,t)
end subroutine
end module
"""
    path = tmp_path / "source.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {root: dict(FACT) for root in ("argument::left", "argument::right", "stencil::scale")}}
    builder = ScopeBuilder([path], "stencil::outer", facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"))
    _outputs, report = builder.run()
    owner, = report["scopes"]
    task, = owner["source_device_tasks"]
    assert task["procedure"] == "stencil::advance"
    capture, = [item for item in task["scalar_captures"] if item["resource"] == "stencil::scale"]
    assert capture["authority"]["kind"] == "explicit checked whole-defined scalar entry fact"
    assert capture["authority"]["fact"]["initialized"] == "whole"
    assert capture["authority"]["finite_or_constant_authorized"] is False
    assert task["execution_binding"]["canonical_resources"] == {
        "argument::a": "argument::left", "argument::b": "argument::right"}
