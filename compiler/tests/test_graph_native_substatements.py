"""Native source children without their own edit span remain ordered."""

import pytest

from compiler.tests.test_graph_source_owner import SOURCE, emit, walk_graph


@pytest.mark.parametrize(("declaration", "assignment"), [
    ("real(8),intent(inout)::delta", "delta=0.0_8"),
    ("integer,intent(inout)::delta", "delta=0"),
])
def test_original_single_line_if_action_needs_no_independent_motion_span(tmp_path, declaration, assignment):
    source = SOURCE.replace("advance(a,b,x,n)", "advance(a,b,x,n,delta)")
    source = source.replace("integer::i", "integer::i\n" + declaration)
    source = source.replace("do i=2,n\nx(i)=x(i-1)+1\nenddo", f"if(n>0) {assignment}")
    path, outputs, report = emit(tmp_path, source)
    assert report["scope_count"] == 1, report["boundaries"]
    owner, = report["scopes"]
    assert not owner["boundaries"]
    graph = owner["graph_execution"]
    procedure, = graph["procedures"]
    native = [task for task in walk_graph(procedure["task_graph"])
              if ":native:" in task["id"]]
    assert native
    assert all(task["ordered"] for task in native)
    assert graph["source_motion"] == []
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.lower().count(assignment.lower()) == 1
    assert text.lower().count(f"if(n>0) {assignment}".lower()) == 1
