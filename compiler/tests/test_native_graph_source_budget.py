"""A complete native graph is bounded per unique original source template."""
import json

import pytest

from compiler.frontend.native_completion import prove_graph_joined_completion
from compiler.frontend.resource_reachability import (
    VERSION,
    ResourceReachabilityBudgetExceeded,
    prove_resource_reachability,
)
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def setup(tmp_path, *, assignments=100, callees=3, calls=None, operations=256):
    names = [f"helper{index}" for index in range(callees)]
    calls = names if calls is None else calls
    bodies = [f"subroutine {name}()\n" + "a(1)=a(1)+1.0_8\n" * assignments + "end subroutine\n"
              for name in names]
    path = tmp_path / "source.f90"
    path.write_text("module work\nimplicit none\nreal(8)::a(1)\ncontains\nsubroutine owner()\n"
                    + "".join(f"call {name}()\n" for name in calls) + "end subroutine\n"
                    + "".join(bodies) + "end module\n")
    analysis = SourceEffects([path], operations=operations)
    analysis.seal_session()
    originals = tuple(analysis.routines["work::owner"].execution.content)
    demanded = (analysis.modules["work"].bindings["a"],)
    return analysis, originals, demanded


def test_complete_graph_exceeds_single_template_statement_limit_without_truncation(tmp_path):
    analysis, originals, demanded = setup(tmp_path)
    proof = prove_resource_reachability(analysis, "work::owner", originals, demanded)
    assert not proof.barriers
    graph = proof.native_graph.public()
    assert graph["schema_version"] == 4
    assert proof.public()["schema_version"] == VERSION
    assert graph["source_work"]["statements"] == 303 > analysis.operation_limit
    assert len(graph["templates"]) == 4
    assert len(graph["instances"]) == 4
    assert len(proof.calls) == 3
    assert all(template["source_work"]["statements"] <= 256 for template in graph["templates"])
    assert graph["limits"]["per_template"] == {"source_statements": 256, "source_nodes": 65536}
    assert graph["limits"]["aggregate"] == {"source_statements": 1024, "source_nodes": 262144}
    assert graph["limits"]["unique_templates"] == 4
    assert graph["limits"]["source_policy_version"] == VERSION
    assert {template["scope"] for template in graph["templates"]} == {"exact original root selection", "complete callee"}
    assert len({template["identity"] for template in graph["templates"]}) == 4
    assert graph["source_identity"] == analysis.inputs.identity()
    assert not proof.public()["execution_authorized"]
    transport = json.loads(json.dumps(graph))
    assert transport["identity"] == proof.native_graph.identity
    assert transport["schema_version"] == 4
    transport["limits"]["aggregate"]["source_statements"] = 1000000
    with pytest.raises(CompilationError, match="issued native effect graph"):
        prove_graph_joined_completion(analysis, "work::owner", originals, transport)


def test_repeated_template_is_counted_once_but_original_calls_stay_distinct(tmp_path):
    analysis, originals, demanded = setup(tmp_path, callees=1, calls=["helper0"] * 3)
    proof = prove_resource_reachability(analysis, "work::owner", originals, demanded)
    assert not proof.barriers
    assert dict(proof.native_graph.source_work)["statements"] == 103
    assert len(proof.native_graph.templates) == 2
    assert len(proof.native_graph.instances) == 4
    assert len({id(call.original) for call in proof.calls}) == 3


def test_oversized_individual_template_still_rejects_the_whole_query(tmp_path):
    analysis, originals, demanded = setup(tmp_path, assignments=257, callees=1)
    with pytest.raises(ResourceReachabilityBudgetExceeded, match="source scan budget"):
        prove_resource_reachability(analysis, "work::owner", originals, demanded)
    assert not getattr(analysis, "_native_effect_graphs", {})
    assert not getattr(analysis, "_resource_reachability_proofs", {})


@pytest.mark.parametrize("field", ["operation_limit", "procedure_limit", "depth_limit"])
def test_changed_existing_budget_invalidates_complete_graph(tmp_path, field):
    analysis, originals, demanded = setup(tmp_path)
    proof = prove_resource_reachability(analysis, "work::owner", originals, demanded)
    setattr(analysis, field, getattr(analysis, field) + 1)
    with pytest.raises(CompilationError):
        proof.validate(analysis, "work::owner", originals, demanded)
