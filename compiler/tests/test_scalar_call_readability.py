"""Companion scalar inputs retain original associations, never literal specialization."""
import json
from copy import copy
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.reached_scalar_definitions import _constant_call_input
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_reached_scalar_definitions import branching_source


def call_builder(tmp_path, calls=None, declarations="", callee=None):
    calls = calls or "\n".join("call advance(a,mask,nx,ny,nz," + str(n) + ")" for n in (1, 1, 2, 3))
    source = (branching_source() if callee is None else callee) + """
module caller
use lines, only:advance
implicit none
contains
subroutine step(a,mask,nx,ny,nz)
integer,intent(in)::nx,ny,nz
real(8),intent(inout)::a(-2:,-2:,-2:)
integer,intent(in)::mask(-2:,-2:,-2:)
integer::unknown
""" + declarations + "\n" + calls + "\nend subroutine\nend module\n"
    path = tmp_path / "source.f90"
    path.write_text(source)
    return ScopeBuilder([path], "caller::step", facts={"schema_version": 1, "participation": "serial",
        "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
        "captures": {"argument::"+n: dict(FACT) for n in ("a", "mask", "nx", "ny", "nz")}},
        options=CompilerOptions(), config=OffloadConfig(policy="sections", scope_execution="graph"))


def proofs(builder):
    return [proof for proof in getattr(builder.analysis, "_source_device_task_proofs", {}).values()
            if proof._routine.qualified == "lines::advance"]


@pytest.mark.parametrize("keyword", [False, True])
def test_all_original_calls_remain_value_unspecialized(tmp_path, keyword):
    calls = "\n".join(("call advance(dir="+str(n)+",a=a,mask=mask,nz=nz,ny=ny,nx=nx)" if keyword else
                       "call advance(a,mask,nx,ny,nz,"+str(n)+")") for n in (1, 1, 2, 3))
    builder = call_builder(tmp_path, calls)
    original_facts = copy(builder.facts["captures"])
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report
    tasks = report["scopes"][0]["source_device_tasks"]
    assert len(tasks) == 3, report
    assert builder.facts["captures"] == original_facts
    for proof in proofs(builder):
        assert proof.validate() is proof
        prefix = proof._scalar_prefix.public()
        authority = next(row["authority"] for row in prefix["dependencies"] if row["resource"] == "argument::dir")
        assert authority["kind"] == "original source-constant scalar call inputs"
        actuals = [next(row["actual"] for row in call["source_call"]["resource_mappings"]
                        if row["formal"] == "dir") for call in authority["calls"]]
        assert actuals == ["1", "1", "2", "3"]
        assert authority["value_specialization_authorized"] is False
        assert authority["entry_fact_authorized"] is False
    rewritten = next(text for name, text in outputs.items() if name.endswith(".f90") and "if(dir==1)then" in text)
    assert "if(dir==1)then" in rewritten
    assert "else if(dir==2)then" in rewritten
    if not keyword:
        package = tmp_path / "package"
        package.mkdir()
        for name, content in outputs.items():
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        (tmp_path / "facts.json").write_text(json.dumps(builder.facts, indent=2) + "\n")


def test_parameter_actual_and_skipped_original_guard_preserved(tmp_path):
    calls = "call advance(a,mask,nx,ny,nz,one)\nif(nx>0)then\ncall advance(a,mask,nx,ny,nz,2)\nendif"
    builder = call_builder(tmp_path, calls, "integer,parameter::one=1")
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report
    assert len(report["scopes"][0]["source_device_tasks"]) == 3, report
    assert any("if(nx>0)then" in text.lower().replace(" ", "") for text in outputs.values())
    for proof in proofs(builder):
        authority = next(row["authority"] for row in proof._scalar_prefix.public()["dependencies"]
                         if row["resource"] == "argument::dir")
        assert len(authority["calls"]) == 2
        assert all(call["selected_node_ids"] for call in authority["calls"])


@pytest.mark.parametrize("actual", ["unknown", "nx", "1_8", "1.0_8"])
def test_later_unknown_or_wrong_kind_never_enters_readable_companion(tmp_path, actual):
    builder = call_builder(tmp_path, "call advance(a,mask,nx,ny,nz,1)\ncall advance(a,mask,nx,ny,nz,"+actual+")")
    outputs, report = builder.run()
    for scope in report["scopes"]:
        if actual in {"unknown", "nx"}:
            assert not scope["source_device_tasks"], report
        else:
            # Invalid type/kind remains the original native statement; it
            # cannot invalidate a separately checked preceding CALL.
            calls = [call for member in scope["checked_execution_plan"]["procedures"]
                     if member["procedure"] == "lines::advance" for call in member["calls"]]
            assert len(calls) == 1, report
            assert any("call advance(a,mask,nx,ny,nz,"+actual+")" in text
                       for name, text in outputs.items() if name.endswith(".f90"))


def test_readwrite_scalar_formal_cannot_consume_literal_definition(tmp_path):
    text = branching_source().replace("integer,intent(in)::nx,ny,nz,dir", "integer,intent(in)::nx,ny,nz\ninteger,intent(inout)::dir")
    builder = call_builder(tmp_path, callee=text)
    _outputs, report = builder.run()
    assert not any(scope["source_device_tasks"] for scope in report["scopes"]), report


def test_exact_call_owner_and_formal_required(tmp_path):
    builder = call_builder(tmp_path)
    _outputs, report = builder.run()
    assert report["scope_count"] == 1
    proof = proofs(builder)[0]
    child = proof._builder
    binding = child.entry.scope.bindings["dir"]
    with pytest.raises(CompilationError):
        _constant_call_input(copy(child), binding)
    with pytest.raises(CompilationError):
        _constant_call_input(child, replace(binding))


def test_registered_prefix_cannot_be_replaced_or_resealed_with_forged_calls(tmp_path):
    builder = call_builder(tmp_path)
    builder.run()
    proof = proofs(builder)[0]
    prefix = proof._scalar_prefix
    registry = builder.analysis._reached_scalar_definition_proofs
    forged = replace(prefix, identity="0"*64)
    registry[forged.identity] = forged
    with pytest.raises(CompilationError, match="registered original"):
        forged.validate(proof._builder, proof._nodes, prefix.demanded)
    del registry[prefix.identity]
    with pytest.raises(CompilationError, match="registered original"):
        proof.validate()


def test_original_source_change_is_rejected_at_publication(tmp_path):
    builder = call_builder(tmp_path)
    builder.run()
    path = tmp_path / "source.f90"
    path.write_text(path.read_text().replace("nz,3)", "nz,unknown)"))
    with pytest.raises(CompilationError, match="configured source changed"):
        builder.analysis.verify_publication()


def test_even_skipped_unknown_actual_cannot_become_an_entry_fact(tmp_path):
    builder = call_builder(tmp_path, "call advance(a,mask,nx,ny,nz,1)\nif(.false.)then\ncall advance(a,mask,nx,ny,nz,unknown)\nendif")
    _outputs, report = builder.run()
    assert not any(scope["source_device_tasks"] for scope in report["scopes"]), report


def test_final_sealing_covers_calls_and_exact_admission_source(tmp_path, monkeypatch):
    from compiler.scopes import execution_plan

    original = execution_plan._seal_execution
    checked = []

    def seal(owner):
        member = owner.members["lines::advance"]
        task = member.source_device_tasks[0].proof
        assert len(member.call_sites) == 4
        task.validate()
        for changes in ({"caller": "foreign::step"}, {"first_line": 9999, "last_line": 9999}):
            member.call_sites.append({**member.call_sites[-1], **changes})
            try:
                with pytest.raises(CompilationError, match="every emitted companion call"):
                    task.validate()
            finally:
                member.call_sites.pop()
        node = member.original_source_call
        member.original_source_call = copy(node)
        try:
            with pytest.raises(CompilationError, match="original companion admission call"):
                task.validate()
        finally:
            member.original_source_call = node
        checked.append(True)
        return original(owner)

    monkeypatch.setattr(execution_plan, "_seal_execution", seal)
    builder = call_builder(tmp_path)
    _outputs, report = builder.run()
    assert checked
    assert report["scope_count"] == 1, report
