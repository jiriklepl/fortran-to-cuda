"""The sealed graph explicitly owns reached source-team backend dispatch."""
from dataclasses import replace

import pytest

from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.ir import CompilationError
from compiler.tests.test_graph_source_owner import walk_graph
from compiler.tests.test_source_native_pricing import live_emitted


def placements(schedule):
    for item in schedule["placements"]:
        yield item
        for child in item.get("children", []):
            yield from placements(child)


@pytest.mark.parametrize("child", [False, True])
def test_source_graph_has_real_cpu_and_gpu_implementations_and_unknown_deferred_cost(tmp_path, child):
    path, (outputs, report) = live_emitted(tmp_path, child)
    scope, = report["scopes"]
    source, = scope["source_device_tasks"]
    binding = source["placement"]
    deferred = {"version": 1, "policy": binding["policy"], "identity": binding["identity"]}
    graph_task, = [task for member in scope["graph_execution"]["procedures"]
                  for task in walk_graph(member["task_graph"]) if task["id"] == binding["task_id"]]
    assert graph_task["implementations"] == ["cpu", "gpu"]
    assert graph_task["ordered"] is True
    assert graph_task["deferred_placement"] == deferred
    planned, = [task for member in scope["graph_execution"]["procedures"]
                for task in placements(member["schedule"]) if task["task"] == binding["task_id"]]
    assert planned["target"] is None
    assert planned["estimated_cost"] is None
    assert planned["deferred_placement"] == deferred
    provider = source["execution_binding"]["provider"]
    assert provider["metadata"]["placement"] == binding
    assert provider["metadata"]["selected_backend"] == {"argument": "selected_backend", "cpu": 0, "gpu": 1}
    code = outputs[report["sources"][str(path)]["replacement"]]
    assert "if (fort_source_selected == 1) then" in code
    assert "if (fort_source_selected == 0 .and. fort_source_registered) then" in code
    native = code.index("if (fort_source_selected == 0 .and. fort_source_registered) then")
    assert native < code.index("fort_scope_source_host_start_v1(", native)
    assert "source GPU selection failed; replay prohibited" in code
    assert scope["estimate_available"] is False
    cu = outputs[provider["entry"] + ".cu"]
    assert "*selected_backend=0;" in cu
    assert cu.index("if (report->started) *selected_backend=1;") > cu.index("status=fort_scope_source_execute_v1(")
    assert max(len(provider[key]) for key in ("entry", "module")) <= 63


def test_only_registered_exact_source_graph_token_can_enable_dispatch(tmp_path, monkeypatch):
    import compiler.frontend.source_placement as issuer

    real, captured = issuer.source_placement, []

    def capture(*args):
        token = real(*args)
        captured.append(token)
        return token

    monkeypatch.setattr(issuer, "source_placement", capture)
    live_emitted(tmp_path, False)
    token, = captured
    token.validate()
    with pytest.raises(CompilationError, match="registered graph/source"):
        replace(token).validate()
    with pytest.raises(CompilationError, match="registered graph/source"):
        replace(token, identity="0"*64).validate()
    with pytest.raises(CompilationError, match="registered graph/source"):
        token.validate(object())
    with pytest.raises(CompilationError, match="registered graph/source/pricing"):
        emit_tile_task_provider(token.proof.team, readable_scalars=token.proof,
                                pricing=token.pricing, placement=token.public())
