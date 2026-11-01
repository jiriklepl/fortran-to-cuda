"""Original structured graphs retain all bounded source control nesting."""

import json

import pytest
from fparser.two.utils import walk

from compiler.frontend import structured_effects
from compiler.frontend.source_effects import SourceEffects
from compiler.tests.test_native_group_structure import fixture as native_fixture
from compiler.tests.test_native_group_structure import team, worksharing
from compiler.tests.test_structured_effects import source


def recursive_sequences(sequence, initial):
    """Reference the previous recursive child-before-parent construction."""
    frame, result = sequence(*initial), None
    try:
        while True:
            try:
                child = frame.send(result)
            except StopIteration as done:
                return done.value
            result = recursive_sequences(sequence, child)
    finally:
        frame.close()


@pytest.mark.parametrize("depth", [9, 16, 24])
@pytest.mark.parametrize("kind", ["if", "loop", "mixed"])
def test_nested_control_is_not_limited_by_call_closure_depth(tmp_path, depth, kind):
    body = "a(1)=b(1)\n"
    for index in range(depth):
        if kind == "if" or kind == "mixed" and index % 2:
            body = "if(flag)then\n" + body + "endif\n"
        else:
            body = f"do j{index}=1,n\n" + body + "enddo\n"
    specification = "integer::" + ",".join(f"j{index}" for index in range(depth))
    _, analysis = source(tmp_path, body, specification=specification)
    analysis.seal_session()
    graph = analysis.structure("varied::advance")
    assert analysis.depth_limit == 8
    assert graph.available, graph.reasons
    assert not any(node.kind == "boundary" for node in graph.nodes.values())
    assert sum(node.kind in {"branch", "loop"} for node in graph.nodes.values()) == depth
    original = next(node for node in walk(analysis.routines[graph.procedure].execution)
                    if type(node).__name__ == "Assignment_Stmt")
    leaf = graph.nodes[graph.node_id(original)]
    assert len(leaf.guard) == depth
    assert graph.source_nodes(leaf.id) == (original,)
    assert graph.operation_count == depth + 1


@pytest.mark.parametrize("body", [
    "a(1)=b(1)\na(2)=b(2)",
    "if(flag)then\na(1)=b(1)\nelse if(n>1)then\nb(1)=a(1)\nelse\na(2)=0\nendif",
    "if(flag)a(1)=b(1)\ndo i=1,n\na(i)=b(i)\nenddo",
    "associate(x=>a(1))\nx=b(1)\nend associate",
    "do 10 i=1,n\na(i)=b(i)\n10 continue",
])
def test_public_records_and_original_bindings_match_recursive_order(tmp_path, monkeypatch, body):
    _, analysis = source(tmp_path, body)
    routine = analysis.routines["varied::advance"]
    authority = analysis._summary_authority()
    digest = "a" * 64
    actual = structured_effects.build_structure(analysis, routine, digest)
    monkeypatch.setattr(structured_effects, "_complete_sequences", recursive_sequences)
    expected = structured_effects.build_structure(analysis, routine, digest)
    assert json.dumps(actual.public(), sort_keys=True) == json.dumps(expected.public(), sort_keys=True)
    assert list(actual.nodes) == list(expected.nodes)
    assert all(actual.source_nodes(identity) == expected.source_nodes(identity) for identity in actual.nodes)
    assert actual._ids == expected._ids
    assert analysis._summary_authority() == authority


def test_deferred_native_groups_keep_identical_public_records(tmp_path, monkeypatch):
    _, analysis = native_fixture(tmp_path, team("if(flag)then\n" + worksharing() * 6 + "endif\n"))
    routine = analysis.routines["renamed_boundary::apply"]
    actual = structured_effects.build_structure(analysis, routine, "a" * 64)
    assert actual.native_groups
    monkeypatch.setattr(structured_effects, "_complete_sequences", recursive_sequences)
    expected = structured_effects.build_structure(analysis, routine, "a" * 64)
    assert actual.public() == expected.public()
    assert list(actual.nodes) == list(expected.nodes)
    assert [group.public() for group in actual.native_groups.values()] == [
        group.public() for group in expected.native_groups.values()]


def test_work_budget_exhaustion_keeps_complete_source_as_unavailable_boundary(tmp_path):
    body = "if(flag)then\n" * 16 + "a(1)=b(1)\n" + "endif\n" * 16
    _, analysis = source(tmp_path, body, operations=12)
    graph = analysis.structure("varied::advance")
    assert not graph.available
    assert graph.operation_count == 1  # The complete-source boundary is one operation.
    assert graph.reasons == ("bounded structured source operation budget exhausted",)
    assert len(graph.nodes) == 2
    assert graph.nodes[graph.root_id].kind == "boundary"
    assert graph.source_nodes(graph.root_id) == tuple(analysis.routines[graph.procedure].execution.content)


def test_source_nesting_does_not_change_separate_call_closure_budget(tmp_path):
    path = tmp_path / "recursive.f90"
    path.write_text("module work\ncontains\nrecursive subroutine step(a)\n"
                    "real(8)::a(:)\nif(a(1)>0)call step(a)\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    graph = analysis.structure("work::step")
    assert graph.available
    assert analysis.depth_limit == 8
    assert not analysis.summarize("work::step")["complete"]
