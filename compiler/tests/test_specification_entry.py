"""Original bounds must be coherent before entering a retained child."""
from dataclasses import replace

import pytest

from compiler.scopes import conditional_return
from compiler.tests.test_conditional_return import compile_fixture
from compiler.tests.test_native_ancestor import SOURCE, fixture

ENTRY_SOURCE = SOURCE.replace("real(8)::observed", "real(8)::observed\ninteger::tile(1)=[8]").replace(
    "integer::i\ncall producer", "integer::i\nreal(8)::scratch(tile(1))\ncall producer")


def test_bound_reads_get_a_checked_phase_before_child_entry(tmp_path):
    outputs, _report, scope, proof = compile_fixture(tmp_path, ENTRY_SOURCE)
    assert proof["version"] == 5
    assert proof["specification_entry_policy_version"] == 1
    middle, = [row for row in proof["members"] if row["procedure"] == "ancestor_fixture::middle"]
    source = middle["entry_exit"]["specification_reads"]
    assert source["array_resources"] == ["ancestor_fixture::tile"]
    assert not source["execution_authorized"]
    route, = [route for row in proof["members"] for route in row["routes"] if "specification_entry" in route]
    assert route["specification_entry"]["proof_identity"] == source["identity"]
    assert "scratch" not in str(scope["resource_bindings"])
    text = "\n".join(value for name, value in outputs.items() if name.startswith("sources/"))
    coordinator, = [row for row in scope["module_coordinators"] if row["procedure"] == middle["procedure"]]
    assert text.index("fort_specification_status = fort_scope_observe_all_v1") < text.index("call " + coordinator["entry"])
    assert "real(8)::scratch(tile(1))" in text


@pytest.mark.parametrize("mutation", ["proof", "source", "publication", "order"])
def test_entry_authority_cannot_change_before_final_seal(tmp_path, monkeypatch, mutation):
    seal = conditional_return.seal

    def corrupt(owner):
        middle = owner.members["ancestor_fixture::middle"]
        if mutation == "proof":
            middle.specification_reads = replace(middle.specification_reads)
        elif mutation == "source":
            object.__setattr__(middle.specification_reads._routine.scope.bindings["scratch"], "rank", 2)
        else:
            call = owner.return_calls[0]
            from compiler.scopes.specification_entry import publication
            block = "\n".join(publication(owner.context)) + "\n"
            changed = call.source.replace(block, "")
            if mutation == "order":
                changed += block
            owner.patches[:] = [(a, b, changed if text is call.source else text) for a, b, text in owner.patches]
            owner.return_calls[0] = replace(call, source=changed)
        return seal(owner)

    monkeypatch.setattr(conditional_return, "seal", corrupt)
    _, builder = fixture(tmp_path, source=ENTRY_SOURCE)
    _outputs, report = builder.run()
    assert not report["scope_count"]
    assert not report["source_edits"]


@pytest.mark.parametrize("bound", ["source_length()", "size(a)", "tile(:)"])
def test_unproved_specification_execution_still_declines(tmp_path, bound):
    _, builder = fixture(tmp_path, source=ENTRY_SOURCE.replace("scratch(tile(1))", f"scratch({bound})"))
    _outputs, report = builder.run()
    assert not report["scope_count"]
    assert not report["source_edits"]
