"""Partial source definitions feed real reached-window execution."""

from copy import copy

import pytest

from compiler.frontend.reached_windows import prove_reached_window
from compiler.ir import CompilationError
from compiler.tests.test_module_reached_windows import MODULE_ROOTS, MODULE_SOURCE, module_fixture, original_loops
from compiler.tests.test_reached_window_owner import ENTRY

PARTIAL_BOUNDARY_SOURCE = MODULE_SOURCE.replace(
    "enddo\ndo i=-2,n-3\na(i)",
    "enddo\nb(-3)=a(-3)+0.5_8\nb(n-2)=a(n-2)+0.25_8\ndo i=-2,n-3\na(i)",
)


def partial_fixture(directory, source=MODULE_SOURCE, *, device_budget=None, policy="sections", precision=8):
    assert precision in {4, 8}
    if precision == 4:
        source = source.replace("real(8)", "real(4)").replace("_8", "_4").replace(",8)", ",4)")
    path, builder = module_fixture(directory, source, device_budget=device_budget, policy=policy)
    builder.facts["captures"][MODULE_ROOTS[1]]["initialized"] = "none"
    return path, builder


def selection(builder):
    outer = next(node for node in builder.entry.execution.content
                 if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    body = tuple(outer.content[1:-1])
    tail = next(index for index, node in enumerate(body) if type(node).__name__ == "Call_Stmt")
    return body[:tail]


@pytest.mark.parametrize("source", [MODULE_SOURCE, PARTIAL_BOUNDARY_SOURCE])
def test_partial_producer_consumer_is_emitted_without_whole_initialization(tmp_path, source):
    path, builder = partial_fixture(tmp_path, source)
    outputs, report = builder.run()
    assert report["reached_window_discovery"]["selected"]
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    closes_before_swap = source == MODULE_SOURCE
    if closes_before_swap:
        boundary, = scope["boundaries"]
        assert "move_alloc" in boundary["reason"]
    else:
        assert not scope["boundaries"]
    proof = builder.validate_reached_window()
    state = proof.capture_state(MODULE_ROOTS[1])
    assert state["initialized"] == "none"
    assert state["coverage_on_return"] == ("unknown_after_close" if closes_before_swap else "sections")
    assert bool(state["defined_sections_on_return"]) is not closes_before_swap
    # The longer discovered window closes before the allocation exchange;
    # ordinary complete geometry retains its shorter original contract.
    complete = prove_reached_window(builder.analysis, ENTRY, selection(builder), MODULE_ROOTS, builder.facts)
    complete_state = complete.capture_state(MODULE_ROOTS[1])
    assert complete_state["coverage_on_return"] == "sections"
    assert len(complete_state["defined_sections_on_return"]) == (3 if source == PARTIAL_BOUNDARY_SOURCE else 1)
    assert proof.public()["native_definition_proof_identities"]
    assert scope["reached_window"]["source_proof_version"] == 10
    public_state = next(item for item in scope["reached_window"]["defined_coverage"]
                        if item["resource"] == MODULE_ROOTS[1])
    assert public_state["on_activation"] == "none"
    assert public_state["on_return"] == state["coverage_on_return"]
    assert public_state["return_sections"] == state["defined_sections_on_return"]
    assert builder.facts["captures"][MODULE_ROOTS[1]]["initialized"] == "none"
    assert not builder.analysis.stable_module_allocatables
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert "0_c_int, fort_buffer_2)" in text
    assert text.index("fort_scope_close(") < text.index("call move_alloc(a,temporary)")


def test_missing_halo_definition_rejects_the_window(tmp_path):
    source = MODULE_SOURCE.replace("2.0_8*b(i)", "2.0_8*b(i+1)")
    _path, builder = partial_fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="section read follows incomplete definition"):
        prove_reached_window(builder.analysis, ENTRY, selection(builder), MODULE_ROOTS, builder.facts)
    assert not getattr(builder.analysis, "_reached_window_proofs", {})


def test_one_conditional_producer_cannot_define_all_paths(tmp_path):
    source = MODULE_SOURCE.replace("do stage=1,iterations\n", "do stage=1,iterations\nif(n>0) then\n").replace(
        "enddo\ndo i=-2,n-3\na(i)", "enddo\nendif\ndo i=-2,n-3\na(i)")
    _path, builder = partial_fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="incomplete definition"):
        prove_reached_window(builder.analysis, ENTRY, selection(builder), MODULE_ROOTS, builder.facts)


def test_both_original_branches_retain_only_common_definitions(tmp_path):
    producer = "do i=-2,n-3\nb(i)=a(i)+real(i+4,8)\nenddo"
    source = MODULE_SOURCE.replace(producer, "if(n>8) then\n" + producer + "\nelse\n" + producer + "\nendif")
    _path, builder = partial_fixture(tmp_path, source)
    proof = prove_reached_window(builder.analysis, ENTRY, selection(builder), MODULE_ROOTS, builder.facts)
    assert proof.capture_state(MODULE_ROOTS[1])["coverage_on_return"] == "sections"


def test_mutated_bound_cannot_relabel_earlier_definition(tmp_path):
    source = MODULE_SOURCE.replace("integer::stage,i", "integer::stage,i,limit").replace(
        "do stage=1,iterations", "limit=n\ndo stage=1,iterations").replace(
        "do i=-2,n-3", "do i=-2,limit-3").replace(
        "enddo\ndo i=-2,limit-3\na(i)", "enddo\nlimit=limit+1\ndo i=-2,limit-3\na(i)")
    _path, builder = partial_fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="incomplete definition"):
        prove_reached_window(builder.analysis, ENTRY, selection(builder), MODULE_ROOTS, builder.facts)


def test_unproved_partial_prefix_cannot_seed_runtime_registration(tmp_path):
    source = MODULE_SOURCE.replace("do stage=1,iterations", "b(-3)=1.0_8\ndo stage=1,iterations").replace(
        "b(i)=a(i)+real(i+4,8)", "b(i)=b(-3)+a(i)+real(i+4,8)")
    _path, builder = partial_fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="incomplete definition"):
        prove_reached_window(builder.analysis, ENTRY, selection(builder), MODULE_ROOTS, builder.facts)


def test_partial_geometry_does_not_replace_original_registered_authority(tmp_path):
    _path, builder = partial_fixture(tmp_path)
    originals = original_loops(builder)
    proof = prove_reached_window(builder.analysis, ENTRY, originals, MODULE_ROOTS, builder.facts)
    with pytest.raises(CompilationError, match="registered exact original"):
        copy(proof).validate(builder.analysis, ENTRY, originals)


def test_missing_calibration_keeps_partial_window_automatic_execution_native(tmp_path):
    path, builder = partial_fixture(tmp_path, policy="auto")
    outputs, report = builder.run()
    assert not report["reached_window_discovery"]["selected"]
    replacement = report["sources"].get(str(path), {}).get("replacement")
    assert (outputs[replacement] if replacement else path.read_text()) == MODULE_SOURCE
