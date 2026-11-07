"""Storage-only real captures do not become host value/definition authority."""
from dataclasses import replace

import pytest

from compiler.driver.options import CompilerOptions
from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_tile_private_emission import original_source
from compiler.tests.test_tile_private_team import team_fixture


def byte_fixture(tmp_path, kind=8, *, attribute="", initialized="none"):
    source = original_source(kind, branch=True).replace("advance(a,b,n,t)", "advance(a,b,n,t,scale)")
    source = source.replace("integer,intent(in)::n,t", f"integer,intent(in)::n,t\nreal({kind}),intent(in){attribute}::scale")
    source = source.replace("b(i)=b(i)+(tmp(li+1)-tmp(li-1))", "b(i)=b(i)+scale*(tmp(li+1)-tmp(li-1))")
    path, analysis, routine, _nodes = team_fixture(tmp_path, source)
    facts = {"schema_version": 1, "participation": "serial", "sources": dict(analysis.sources),
             "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "scale")}}
    facts["captures"]["argument::scale"]["initialized"] = initialized
    builder = ScopeBuilder([path], routine.qualified, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"))
    nodes = tuple(builder.entry.execution.content)
    return builder, nodes


def issue(builder, nodes):
    return source_device_task(builder.analysis, builder.entry, nodes, builder=builder)


@pytest.mark.parametrize("kind", [4, 8])
@pytest.mark.parametrize("mapping", ["thread", "cooperative"])
def test_byte_capture_retains_original_conditional_load_without_readability(tmp_path, kind, mapping):
    builder, nodes = byte_fixture(tmp_path, kind)
    proof = issue(builder, nodes)
    assert proof.public()["version"] == 2
    assert [b.root for b in proof.byte_scalars] == ["argument::scale"]
    assert all(b.root != "argument::scale" for b in proof.readable_scalars)
    with pytest.raises(CompilationError, match="registered original proof"):
        replace(proof, readable_scalars=proof.readable_scalars + proof.byte_scalars).validate()
    provider = emit_tile_task_provider(proof.team, readable_scalars=proof, mapping=mapping, access_envelopes=True)
    assert provider.public()["version"] == 9
    assert provider.public()["scalar_transport"]["resources"] == ["argument::scale"]
    parameter = next(p for p in provider.parameters if p.binding.root == "argument::scale")
    code = provider.cuda_source
    assert f"state.{parameter.name}=*{parameter.name}" not in code
    assert f"std::memcpy(state.{parameter.name}.bytes,{parameter.name},sizeof(state.{parameter.name}.bytes));" in code
    assert "fort_scalar_bytes<" in code
    # The decode is emitted at the numerical expression inside the original IF.
    kernel = code[code.index("void fort_tile_one_") if mapping == "thread" else code.index("void kernel_tile_private_cta_"):]
    assert kernel.index("if ((::generated_kernels::numeric::mod") < kernel.index("fort_scalar_value(")
    assert "initialized" not in next(c["authority"]["fact"] for c in proof.public()["scalar_captures"]
                                     if c["resource"] == "argument::scale")


@pytest.mark.parametrize("options", [{}, {"access_envelopes": True, "queued_single": True},
                                    {"access_envelopes": True, "queued_team": True},
                                    {"access_envelopes": True, "batch_target_bytes": 16777216}])
def test_byte_capture_does_not_borrow_an_unqualified_transport(tmp_path, options):
    builder, nodes = byte_fixture(tmp_path)
    proof = issue(builder, nodes)
    with pytest.raises(CompilationError, match="byte scalar capture requires"):
        emit_tile_task_provider(proof.team, readable_scalars=proof, **options)


@pytest.mark.parametrize("attribute", [",target", ",optional", ",volatile", ",asynchronous"])
def test_byte_capture_keeps_association_and_storage_restrictions(tmp_path, attribute):
    builder, nodes = byte_fixture(tmp_path, attribute=attribute)
    with pytest.raises(CompilationError):
        issue(builder, nodes)


def test_storage_change_invalidates_byte_attachment(tmp_path):
    builder, nodes = byte_fixture(tmp_path)
    proof = issue(builder, nodes)
    builder.facts["captures"]["argument::scale"]["escapes"] = True
    with pytest.raises(CompilationError, match="stable storage"):
        proof.validate()
