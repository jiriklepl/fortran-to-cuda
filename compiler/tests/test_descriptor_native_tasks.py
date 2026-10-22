"""Descriptor inquiries retain their source dependencies without payload hooks."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from compiler.scopes.definitions import DefinitionSummary
from compiler.scopes.graph import GraphOwner
from compiler.scopes.lexical import original_descriptor_scalar_operation
from compiler.scopes.segments import Native
from compiler.tests.test_graph_source_owner import emit, walk_graph

SOURCE = """module graph_fixture
implicit none
contains
subroutine advance(a,b,x,n)
real(8),intent(inout)::a(:,:,:),b(:,:,:)
integer,intent(inout)::x(:)
integer,intent(in)::n
integer::i,j,k,mini,maxi,minj,maxj,mink,maxk
do k=1,n
do j=1,n
do i=1,n
b(i,j,k)=2.0_8*a(i,j,k)
enddo
enddo
enddo
mini = lbound(a,1) + 1
maxi = ubound(a,1) - 1
minj = lbound(a,2) + 1
maxj = ubound(a,2) - 1
mink = lbound(a,3) + 1
maxk = ubound(a,3) - 1
do k=mink,maxk
do j=minj,maxj
do i=mini,maxi
a(i,j,k)=b(i,j,k)+1.0_8
enddo
enddo
enddo
end subroutine
end module
"""


def candidate():
    # The classifier consumes only complete fragment proofs constructed by
    # admit(); these small records exercise its conservative shape predicate.
    assignment = type("Assignment_Stmt", (), {})()
    operation = Native(
        nodes=(assignment,), effects={}, overwrites=set(),
        bindings={"argument::a": SimpleNamespace(dtype="real"),
                  "graph_fixture::advance::mini": SimpleNamespace(dtype="integer")},
        summary={"complete": True, "native_completion": {"available": True}, "definition_changes": [],
                 "operations": [{"kind": "descriptor_read", "rank": 3, "resource": "argument::a"},
                                {"kind": "write", "rank": 0, "resource": "graph_fixture::advance::mini"}]},
        sections=SimpleNamespace(resources=()),
    )
    return SimpleNamespace(calls=[], native=[operation], tree=[operation]), DefinitionSummary(
        frozenset(), frozenset(), frozenset())


def test_complete_descriptor_and_scalar_effects_need_no_runtime_payload():
    scope, definitions = candidate()
    assert original_descriptor_scalar_operation(scope, definitions)


@pytest.mark.parametrize("change", [
    {"effects": {"argument::a": {"read"}}},
    {"overwrites": {"argument::a"}},
    {"host_metadata": {"argument::a": object()}},
    {"native_environment": (object(),)},
    {"preserve_original": True},
    {"bindings": {"argument::a": SimpleNamespace(dtype="character")}},
    {"sections": SimpleNamespace(resources=[SimpleNamespace(reads=(object(),), writes=(), overwrites=())])},
    {"nodes": (type("Call_Stmt", (), {})(),)},
])
def test_payload_opaque_environment_and_nonassignment_operations_stay_managed(change):
    scope, definitions = candidate()
    operation = replace(scope.native[0], **change)
    scope.native = scope.tree = [operation]
    assert not original_descriptor_scalar_operation(scope, definitions)


@pytest.mark.parametrize(("kind", "rank"), [
    ("read", 1), ("write", 1), ("overwrite", 1), ("call", 0), ("control", 0),
    ("boundary", 0), ("native_environment", 0), ("allocation", 0),
])
def test_complete_summary_cannot_hide_elements_calls_or_lifetime_effects(kind, rank):
    scope, definitions = candidate()
    scope.native[0].summary["operations"].append({"kind": kind, "rank": rank})
    assert not original_descriptor_scalar_operation(scope, definitions)


@pytest.mark.parametrize("case", ["incomplete", "completion", "kill", "requires", "returns", "changes", "tree", "call"])
def test_proof_and_definition_obligations_cannot_be_removed(case):
    scope, definitions = candidate()
    if case == "incomplete":
        scope.native[0].summary["complete"] = False
    elif case == "completion":
        scope.native[0].summary["native_completion"]["available"] = False
    elif case == "kill":
        scope.native[0].summary["definition_changes"] = ["argument::a"]
    elif case in {"requires", "returns", "changes"}:
        field = {"requires": "required_whole", "returns": "whole_on_return", "changes": "changes"}[case]
        definitions = replace(definitions, **{field: frozenset({"argument::a"})})
    elif case == "tree":
        scope.tree = [object()]
    elif case == "call":
        scope.calls = [object()]
    assert not original_descriptor_scalar_operation(scope, definitions)


def test_six_original_inquiries_are_graph_tasks_with_descriptor_and_scalar_dependencies(tmp_path):
    owners = []
    initialize = GraphOwner.__init__

    def retain_owner(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        owners.append(self)

    with patch.object(GraphOwner, "__init__", retain_owner):
        path, outputs, report = emit(tmp_path, SOURCE)
    owner, = report["scopes"]
    assert len(owner["planning_segments"]) == 2
    procedure, = owner["graph_execution"]["procedures"]
    native = [task for task in walk_graph(procedure["task_graph"]) if ":native:" in task["id"]]
    assert len(native) == 6
    checked = owners[0].task_graphs["graph_fixture::advance"]
    native_checked = [task for task in checked.tasks if ":native:" in task.id]
    assert len(native_checked) == 6
    assert all(task.effects.metadata_reads == frozenset({"argument::a"}) for task in native_checked)
    assert all(task.effects.ordered for task in native_checked)
    assert all(not task["reads"] and not task["may_writes"] for task in native)
    assert {root.rsplit("::", 1)[-1] for task in native for root in task["scalar_writes"]} == {
        "mini", "maxi", "minj", "maxj", "mink", "maxk"}
    text = outputs[report["sources"][str(path)]["replacement"]]
    start, end = text.index("mini = lbound(a,1) + 1"), text.index("maxk = ubound(a,3) - 1")
    assert start < end
    assert "fort_scope_" not in text[start:end]
    assert text.count("mini = lbound(a,1) + 1") == 1
    assert text.count("fort_reached_0: block") == text.count("fort_reached_1: block") == 1
    assert "fort_reached_2: block" not in text
    assert text.index("fort_reached_0: block") < start < text.index("fort_reached_1: block")


@pytest.mark.parametrize(("case", "segments"), [("size", 2), ("element-read", 3), ("guarded-action", 3)])
def test_source_inquiries_do_not_hide_element_reads_or_original_action_guards(tmp_path, case, segments):
    source = SOURCE
    if case == "size":
        source = source.replace("mini = lbound(a,1) + 1", "mini = size(a,1) - 3")
    elif case == "element-read":
        source = source.replace("mini = lbound(a,1) + 1", "mini = x(1)")
    else:
        source = source.replace("mini = lbound(a,1) + 1", "if(n>0)mini = lbound(a,1) + 1")
        source = source.replace("do k=mink,maxk", "if(n>0)then\ndo k=mink,maxk")
        source = source.replace("end subroutine", "endif\nend subroutine")
    path, outputs, report = emit(tmp_path, source)
    owner, = report["scopes"]
    assert len(owner["planning_segments"]) == segments
    text = outputs[report["sources"][str(path)]["replacement"]]
    if case == "size":
        assert text.count("mini = size(a,1) - 3") == 1
    elif case == "element-read":
        native = [operation for segment in owner["planning_segments"]
                  for operation in segment["operations"]["native_operations"]]
        assert any(operation["resource_effects"].get("argument::x") == ["read"] for operation in native)
    else:
        assert "if (n > 0) then" in text
        assert "mini=lbound(a,1)+1" in text.lower().replace(" ", "")


@pytest.mark.native
def test_descriptor_only_source_with_gpu_before_and_after_compiles(tmp_path):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    if fc is None:
        pytest.skip("Fortran compiler unavailable")
    _path, outputs, report = emit(tmp_path / "inputs", SOURCE)
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        artifact = build / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, item in enumerate(report["build_sources"]):
            if item["language"] != "fortran" or item["role"] != role:
                continue
            result = subprocess.run([fc, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-c",
                                     str(build / item["path"]), "-I", str(build), "-J", str(build),
                                     "-o", str(build / f"source-{index}.o")],
                                    cwd=build, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
