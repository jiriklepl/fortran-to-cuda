"""Per-call line bindings preserve original control and distinct allocations."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_reached_scalar_definitions import branching_source


def multi_field_builder(tmp_path, *, keyword=False, repeat=False, separate_sizes=False):
    calls = []
    size_names = [f"n{axis}{field}" for field in "abc" for axis in "xyz"] if separate_sizes else ["nx", "ny", "nz"]
    for a, mask in [("a", "ma"), *(([("a", "ma")]) if repeat else []), ("b", "mb"), ("c", "mc")]:
        nx, ny, nz = [f"n{axis}{a}" for axis in "xyz"] if separate_sizes else ("nx", "ny", "nz")
        calls.append(
            f"call relay(mask={mask},x={a},nz={nz},ny={ny},nx={nx})"
            if keyword
            else f"call relay({a},{mask},{nx},{ny},{nz})"
        )
    source = (
        branching_source()
        + """
module caller
use lines,only:advance
implicit none
contains
subroutine relay(x,mask,nx,ny,nz)
integer,intent(in)::nx,ny,nz
real(8),intent(inout)::x(-2:,-2:,-2:)
integer,intent(in)::mask(-2:,-2:,-2:)
if(nx==1)then
 call advance(x,mask,nx,ny,nz,2)
 call advance(x,mask,nx,ny,nz,3)
else if(ny==1)then
 call advance(x,mask,nx,ny,nz,1)
 call advance(x,mask,nx,ny,nz,3)
else if(nz==1)then
 call advance(x,mask,nx,ny,nz,3)
 call advance(x,mask,nx,ny,nz,3)
else
 call advance(x,mask,nx,ny,nz,1)
 call advance(x,mask,nx,ny,nz,2)
 call advance(x,mask,nx,ny,nz,3)
endif
end subroutine
"""
        + "subroutine step(a,b,c,ma,mb,mc,"
        + ",".join(size_names)
        + ")\n"
        + "integer,intent(in)::"
        + ",".join(size_names)
        + "\n"
        + "real(8),intent(inout)::a(-2:,-2:,-2:),b(-2:,-2:,-2:),c(-2:,-2:,-2:)\n"
        + "integer,intent(in)::ma(-2:,-2:,-2:),mb(-2:,-2:,-2:),mc(-2:,-2:,-2:)\n"
        + "\n".join(calls)
        + "\nend subroutine\nend module\n"
    )
    path = tmp_path / "source.f90"
    path.write_text(source)
    facts = {
        "schema_version": 1,
        "participation": "serial",
        "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
        "captures": {"argument::" + n: dict(FACT) for n in ["a", "b", "c", "ma", "mb", "mc", *size_names]},
    }
    return ScopeBuilder(
        [path],
        "caller::step",
        facts=facts,
        options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"),
    )


@pytest.mark.parametrize(
    "keyword,repeat,separate_sizes",
    [(False, False, False), (True, False, False), (False, True, False), (False, True, True)],
)
def test_original_three_field_call_dag_has_all_direction_and_root_bindings(tmp_path, keyword, repeat, separate_sizes):
    builder = multi_field_builder(tmp_path, keyword=keyword, repeat=repeat, separate_sizes=separate_sizes)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report["boundaries"]
    (scope,) = report["scopes"]
    assert len(scope["source_device_tasks"]) == 3
    for task in scope["source_device_tasks"]:
        projection = task["execution_binding"]["call_projection"]
        assert (
            projection["identity"]
            == sha256(
                json.dumps(
                    {k: v for k, v in projection.items() if k != "identity"}, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        )
        assert len(projection["members"]) == 3
        assert len(projection["calls"]) == 12 + int(repeat)
        if separate_sizes:
            root_calls = [call for call in projection["calls"] if call["caller"] == "caller::step"]
            assert [
                [
                    arg["actual"].lower()
                    for arg in call["source_call"]["original_arguments"]
                    if arg["formal"] in {"nx", "ny", "nz"}
                ]
                for call in root_calls
            ] == [["nxa", "nya", "nza"], ["nxa", "nya", "nza"], ["nxb", "nyb", "nzb"], ["nxc", "nyc", "nzc"]]
        assert projection["field_roots"] == {
            "argument::a": ["argument::a", "argument::b", "argument::c"],
            "argument::mask": ["argument::ma", "argument::mb", "argument::mc"],
        }
        assert set(projection["root_registrations"]) == {
            "argument::" + name for name in ["a", "b", "c", "ma", "mb", "mc"]
        }
        assert all(
            projection[key] is False
            for key in [
                "definition_authorized",
                "lifetime_authorized",
                "alias_authorized",
                "value_specialization_authorized",
            ]
        )
        child = [row for row in projection["calls"] if row["callee"] == "lines::advance"]
        assert len(child) == 9
        values = [
            next(a["actual"] for a in row["source_call"]["original_arguments"] if a["formal"] == "dir") for row in child
        ]
        assert values == ["2", "3", "1", "3", "3", "3", "1", "2", "3"]
    expected_reads = {"argument::" + name for name in ["a", "b", "c", "ma", "mb", "mc"]}
    expected_writes = {"argument::" + name for name in ["a", "b", "c"]}

    def graph_rows(value):
        if isinstance(value, dict):
            if ":source_device:" in value.get("id", ""):
                yield value
            for child in value.values():
                yield from graph_rows(child)
        elif isinstance(value, list):
            for child in value:
                yield from graph_rows(child)

    rows = list(graph_rows(scope["graph_execution"]))
    assert len(rows) == 3
    for row in rows:
        assert set(row["reads"]) == set(row["metadata_reads"]) == expected_reads
        assert set(row["may_writes"]) == expected_writes
    for task in scope["source_device_tasks"]:
        effects = task["execution_binding"]["call_projection"]["conservative_effects"]
        assert set(effects["reads"]) == set(effects["metadata_reads"]) == expected_reads
        assert set(effects["may_writes"]) == expected_writes
        assert effects["must_writes"] == effects["overwrites"] == []
    if not keyword and (not repeat or separate_sizes):
        package = tmp_path / "package"
        package.mkdir()
        for name, value in outputs.items():
            target = package / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(value)
        (tmp_path / "facts.json").write_text(json.dumps(builder.facts, indent=2) + "\n")
    text = "\n".join(v for k, v in outputs.items() if k.endswith(".f90"))
    assert "fort_storage_identity_" in text
    assert "fort_scope_source_execute_v1" in "\n".join(v for k, v in outputs.items() if k.endswith(".cu"))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_receipt",
        "missing_call",
        "missing_both",
        "foreign_receipt",
        "changed_source",
        "changed_handle",
        "changed_identity",
        "changed_resolved",
        "unknown_receipt",
        "unknown_namespace",
    ],
)
def test_missing_or_changed_original_forwarding_is_rejected_before_sealing(tmp_path, monkeypatch, mutation):
    from compiler.scopes import execution_plan
    from compiler.scopes.source_task_bindings import call_projection

    original = execution_plan._seal_execution
    checked = []

    def seal(owner):
        if checked:
            return original(owner)
        checked.append(True)
        member = owner.members["lines::advance"]
        relay = owner.members["caller::relay"]
        task = member.source_device_tasks[0]
        expected = call_projection(member, task)
        assert expected is not None
        receipts = owner.source_companion_calls
        receipt = receipts[-1]
        restore = None
        if mutation in {"missing_receipt", "missing_both"}:
            removed = receipts.pop()
            removed_call = relay.call_sites.pop() if mutation == "missing_both" else None

            def restore():
                receipts.append(removed)
                if removed_call is not None:
                    relay.call_sites.append(removed_call)
        elif mutation == "missing_call":
            removed = relay.call_sites.pop()

            def restore():
                relay.call_sites.append(removed)
        elif mutation in {"foreign_receipt", "changed_source", "unknown_receipt"}:
            receipts[-1] = (
                replace(receipt, callee=member)
                if mutation == "foreign_receipt"
                else replace(receipt, source=receipt.source + "! changed")
                if mutation == "changed_source"
                else {}
            )

            def restore():
                receipts[-1] = receipt
        elif mutation == "changed_handle":
            slots = relay.call_sites[-1]["handle_bindings"]
            key = next(iter(slots))
            old = slots[key]
            slots[key] = old + 999

            def restore():
                slots[key] = old
        elif mutation == "changed_identity":
            key = next(iter(relay.resource_slot_roots))
            old = relay.resource_slot_roots[key]
            relay.resource_slot_roots[key] = "foreign::allocation"

            def restore():
                relay.resource_slot_roots[key] = old
        elif mutation == "changed_resolved":
            site = relay.call_sites[-1]
            key = (site["caller"], (site["first_line"], site["last_line"]))
            old = owner.builder.resolved_calls[key]
            owner.builder.resolved_calls[key] = {}

            def restore():
                owner.builder.resolved_calls[key] = old
        elif mutation == "unknown_namespace":
            old = relay.resource_slot_namespace
            relay.resource_slot_namespace = owner

            def restore():
                relay.resource_slot_namespace = old

        try:
            with pytest.raises(CompilationError):
                call_projection(member, task)
        finally:
            restore()
        assert call_projection(member, task) == expected
        return original(owner)

    monkeypatch.setattr(execution_plan, "_seal_execution", seal)
    outputs, report = multi_field_builder(tmp_path).run()
    assert report["scope_count"] == 1, report["boundaries"]
    assert checked


def test_per_field_closure_handles_large_diamond_without_expanding_paths():
    from compiler.scopes.source_task_bindings import _field_roots

    nodes = {
        "owner": {
            "parameterized": False,
            "resource_slots": {"a": 1, "b": 2},
            "owner_resources": {"a": "argument::a", "b": "argument::b"},
        }
    }
    calls = []
    lookups = []

    class Mapping(dict):
        def get(self, key):
            lookups.append(key)
            return super().get(key)

    previous = ["owner"]
    for depth in range(80):
        current = [f"level{depth}_{side}" for side in range(2)]
        for name in current:
            nodes[name] = {"parameterized": True, "resource_slots": {"x": 1, "y": 2}}
            for parent in previous:
                calls.append({"caller": parent, "callee": name, "handle_bindings": Mapping({"1": 1, "2": 2})})
        previous = current
    assert _field_roots(nodes, calls, previous[0], {"x": 1, "y": 2}) == {"x": ["argument::a"], "y": ["argument::b"]}
    # 2**79 original paths; each reached (procedure, slot) is processed once.
    assert len(lookups) <= 4 * len(calls)


@pytest.mark.parametrize("failure", ["cycle", "missing_terminal", "missing_slot", "missing_owner"])
def test_per_field_closure_rejects_incomplete_or_cyclic_effects(failure):
    from compiler.scopes.source_task_bindings import _field_roots

    nodes = {
        "child": {"parameterized": True, "resource_slots": {"a": 1}},
        "owner": {"parameterized": False, "resource_slots": {"a": 2}, "owner_resources": {"a": "argument::a"}},
    }
    calls = [{"caller": "owner", "callee": "child", "handle_bindings": {"1": 2}}]
    if failure == "cycle":
        calls[0].update(caller="child", handle_bindings={"1": 1})
    elif failure == "missing_terminal":
        calls.clear()
    elif failure == "missing_slot":
        calls[0]["handle_bindings"] = {}
    elif failure == "missing_owner":
        nodes["owner"]["owner_resources"] = {}
    with pytest.raises(CompilationError):
        _field_roots(nodes, calls, "child", {"a": 1})
