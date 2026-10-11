"""Module descriptors are borrowed only at an authenticated reached window."""

from copy import deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.reached_windows import prove_reached_window
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_reached_window_owner import ENTRY, SOURCE
from compiler.tests.test_source_scopes import FACT

MODULE_ROOTS = ("renamed_stage::a", "renamed_stage::b")
MODULE_SOURCE = SOURCE.replace(
    "implicit none\n", "implicit none\nreal(8),allocatable::a(:),b(:)\n", 1
).replace("subroutine step(a,b,n,iterations)", "subroutine step(n,iterations)").replace(
    "real(8),allocatable,intent(inout)::a(:),b(:)\n", ""
)
MODULE_BOUNDARY_SOURCE = MODULE_SOURCE.replace(
    "enddo\ndo i=-2,n-3\na(i)", "enddo\nif(n>0) then\nb(0)=b(0)+0.5_8\nendif\ndo i=-2,n-3\na(i)"
)


def module_fixture(directory, source=MODULE_SOURCE, *, policy="sections", device_budget=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "stage.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {root: {**deepcopy(FACT), "lifetime": "selected_window"} for root in MODULE_ROOTS}}
    if device_budget is not None:
        facts["device_budget_bytes"] = device_budget
    return path, ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
                              config=OffloadConfig(policy=policy, scope_execution="reached"))


def original_loops(builder):
    outer = next(node for node in builder.entry.execution.content
                 if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    return tuple(node for node in outer.content[1:-1] if type(node).__name__ == "Block_Nonlabel_Do_Construct")


def test_selected_window_facts_do_not_authorize_invocation_lifetime(tmp_path):
    _path, builder = module_fixture(tmp_path)
    assert builder.analysis.stable_module_allocatables == frozenset()
    for root in MODULE_ROOTS:
        binding = builder.visible_binding(builder.entry, root)
        assert builder.capture_fact(binding)["initialized"] == "whole"
        with pytest.raises(CompilationError, match="registered selected-position lifetime"):
            builder.capture(binding)
    assert not builder.analysis.window_module_allocatables(ENTRY)


@pytest.mark.parametrize("lifetime", [[], {}, 1, None, "unknown"])
def test_invalid_capture_lifetime_has_a_controlled_rejection(tmp_path, lifetime):
    _path, builder = module_fixture(tmp_path)
    builder.facts["captures"][MODULE_ROOTS[0]]["lifetime"] = lifetime
    binding = builder.visible_binding(builder.entry, MODULE_ROOTS[0])
    with pytest.raises(CompilationError, match="unsupported scoped capture lifetime"):
        builder.capture_fact(binding)


def test_malformed_module_facts_cannot_trigger_discovery_or_crash(tmp_path):
    from compiler.scopes.window_discovery import discover_reached_window

    _path, builder = module_fixture(tmp_path)
    builder.facts["captures"] = {root: [] for root in MODULE_ROOTS}
    assert not discover_reached_window(builder)
    assert not builder.outputs
    assert not builder.edits


def test_explicit_window_borrows_original_module_descriptors_and_closes_before_swap(tmp_path):
    path, builder = module_fixture(tmp_path)
    facts = deepcopy(builder.facts)
    selection = original_loops(builder)
    proof = prove_reached_window(builder.analysis, ENTRY, selection, MODULE_ROOTS, facts)
    assert proof.public()["module_descriptor_resources"] == list(MODULE_ROOTS)
    assert all(state["resource_kind"] == "original_module_allocatable" for state in proof.public()["states"])
    owner = ReachedWindowOwner(builder, proof, selection)
    assert owner.run()
    outputs, report = owner.builder.finish()
    text = outputs[report["sources"][str(path)]["replacement"]]
    activation = text.index(f"if (.not. {owner.spent}) then")
    assert text.index("call native_prefix(a)") < activation
    assert text.index("allocated(a)", activation) < text.index("lbound(a", activation)
    assert text.index("allocated(b)", activation) < text.index("lbound(b", activation)
    assert text.index("fort_scope_close(") < text.index("call move_alloc(a,temporary)")
    assert len(report["scopes"][0]["gpu_leaves"]) == 2
    assert builder.analysis.stable_module_allocatables == frozenset()
    assert not builder.analysis.window_module_allocatables(ENTRY)
    assert builder.facts == facts
    assert path.read_text() == MODULE_SOURCE


def test_public_discovery_admits_module_window_without_global_authorization(tmp_path):
    _path, builder = module_fixture(tmp_path)
    _outputs, report = builder.run()
    assert report["reached_window_discovery"]["selected"]
    assert report["reached_window_discovery"]["trigger"]["selected_window_module_resources"] == list(MODULE_ROOTS)
    scope, = report["scopes"]
    assert scope["reached_window"]["automatic_discovery"]
    assert len(scope["gpu_leaves"]) == 2
    assert builder.analysis.stable_module_allocatables == frozenset()


@pytest.mark.parametrize("single_line", [False, True])
def test_reached_native_partial_update_retains_same_module_owner(tmp_path, single_line):
    source = MODULE_BOUNDARY_SOURCE
    if single_line:
        source = source.replace("if(n>0) then\nb(0)=b(0)+0.5_8\nendif", "if(n>0) b(0)=b(0)+0.5_8")
    path, builder = module_fixture(tmp_path, source)
    outputs, report = builder.run()
    assert report["reached_window_discovery"]["selected"]
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    boundary, = scope["boundaries"]
    cut, = scope["reached_window"]["planned_closures"]["close_points"]
    assert boundary["original_node_ids"] == [cut["node_id"]]
    original, = builder.analysis.structure(ENTRY).source_nodes(cut["node_id"])
    assert str(original).lower() == "call move_alloc(a, temporary)"
    assert cut["original_source_kind"] == "Call_Stmt"
    producer, correction, consumer = scope["planning_segments"]
    assert [bool(segment["gpu_leaves"]) for segment in (producer, correction, consumer)] == [True, False, True]
    assert producer["last_line"] < correction["first_line"]
    assert correction["last_line"] < consumer["first_line"]
    assert consumer["last_line"] < boundary["first_line"]
    assert not scope["reopen_after_boundary"]
    text = outputs[report["sources"][str(path)]["replacement"]]
    compact = "".join(text.lower().split())
    native_edit = compact.index("b(0)=b(0)+0.5_8")
    replacement = compact.index("callmove_alloc(a,temporary)")
    close = compact.rindex("fort_scope_close(", 0, replacement)
    assert compact.index("fort_scope_host_begin(") < native_edit < close < replacement
    assert text.count("call move_alloc(a,temporary)") == 1
    assert path.read_text() == source
    assert builder.analysis.stable_module_allocatables == frozenset()


def test_module_allocated_state_never_supplies_payload_definition(tmp_path):
    _path, builder = module_fixture(tmp_path)
    builder.facts["captures"][MODULE_ROOTS[0]]["initialized"] = "none"
    with pytest.raises(CompilationError, match="incomplete definition"):
        prove_reached_window(builder.analysis, ENTRY, original_loops(builder), MODULE_ROOTS, builder.facts)
    assert not getattr(builder.analysis, "_reached_window_proofs", {})


def test_module_descriptor_replacement_inside_window_is_rejected(tmp_path):
    source = MODULE_SOURCE.replace("a(i)=2.0_8*b(i)", "call replace_hidden()\na(i)=2.0_8*b(i)")
    source = source.replace("subroutine step(n,iterations)",
                            "subroutine replace_hidden()\ndeallocate(a)\nallocate(a(8))\nend subroutine\n"
                            "subroutine step(n,iterations)")
    _path, builder = module_fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="allocation|replace|unknown"):
        prove_reached_window(builder.analysis, ENTRY, original_loops(builder), MODULE_ROOTS, builder.facts)
    assert not getattr(builder.analysis, "_reached_window_proofs", {})


@pytest.mark.parametrize("attributes", ["target", "pointer", "volatile", "asynchronous"])
def test_uncertain_module_storage_cannot_be_blessed_by_selected_facts(tmp_path, attributes):
    source = MODULE_SOURCE.replace("real(8),allocatable::a(:),b(:)",
                                   "real(8),allocatable," + attributes + "::a(:),b(:)")
    _path, builder = module_fixture(tmp_path, source)
    with pytest.raises(CompilationError):
        prove_reached_window(builder.analysis, ENTRY, original_loops(builder), MODULE_ROOTS, builder.facts)
    assert not builder.analysis.stable_module_allocatables


def test_missing_auto_calibration_keeps_module_source_native(tmp_path):
    path, builder = module_fixture(tmp_path, policy="auto")
    outputs, report = builder.run()
    assert not report["reached_window_discovery"]["selected"]
    replacement = report["sources"].get(str(path), {}).get("replacement")
    assert (outputs[replacement] if replacement else path.read_text()) == MODULE_SOURCE
    assert builder.analysis.stable_module_allocatables == frozenset()
