"""Source-position consumers preserve admitted facts and selected authority."""

from copy import copy
from hashlib import sha256

import pytest
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.frontend.reached_windows import prove_reached_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.segments import _native_analysis
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_reached_window_proofs import ENTRY, ROOT, fixture


def bound(tmp_path, *, prefix="counter=2", attributes="", initialized="whole", suffix=""):
    path, _analysis, _routine, _selected, facts = fixture(
        tmp_path, prefix=prefix, attributes=attributes, initialized=initialized, suffix=suffix)
    facts["participation"] = "serial"
    builder = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    selected = (next(node for node in builder.entry.execution.content
                     if type(node).__name__ == "Call_Stmt"),)
    proof = prove_reached_window(builder.analysis, ENTRY, selected, (ROOT,), facts)
    return builder, selected, proof, builder.bind_reached_window(proof, selected)


def test_reached_coverage_is_private_to_source_position(tmp_path):
    builder, _selected, _proof, proxy = bound(tmp_path, prefix="a=3.0_8", initialized="none")
    binding = builder.entry.scope.bindings["a"]
    assert builder.capture(binding)["initialized"] == "none"
    assert proxy.capture(binding)["initialized"] == "whole"
    assert builder.facts is proxy.facts
    assert proxy.inline is not builder.inline
    assert proxy.inline.builder is proxy
    assert proxy.region_owners[builder.entry.qualified] is proxy.inline
    assert builder.facts["captures"][ROOT]["initialized"] == "none"
    assert not builder.edits


def test_registered_window_is_required_in_exact_analysis(tmp_path):
    builder, selected, proof, _proxy = bound(tmp_path)
    with pytest.raises(CompilationError, match="registered original window proof"):
        builder.bind_reached_window(proof.public(), selected)
    with pytest.raises(CompilationError, match="registered exact original"):
        builder.bind_reached_window(copy(proof), selected)


def test_binding_requires_the_same_admitted_entry_facts(tmp_path):
    builder, selected, proof, proxy = bound(tmp_path)
    builder.facts["captures"][ROOT]["initialized"] = "none"
    with pytest.raises(CompilationError, match="entry facts differ"):
        builder.bind_reached_window(proof, selected)
    with pytest.raises(CompilationError, match="entry facts changed"):
        proxy.capture(builder.entry.scope.bindings["a"])


def test_unproved_managed_root_never_receives_window_coverage(tmp_path):
    builder, _selected, _proof, proxy = bound(tmp_path)
    binding = builder.entry.scope.bindings["counter"]
    builder.facts["captures"][binding.root] = dict(builder.facts["captures"][ROOT])
    # Changing the admitted document invalidates the token before any added
    # capture can masquerade as proof-covered storage.
    with pytest.raises(CompilationError, match="entry facts changed"):
        proxy.capture(binding)


def test_window_summary_reissues_descriptor_proof_in_same_private_analysis(tmp_path):
    builder, selected, proof, proxy = bound(tmp_path, attributes=",allocatable", suffix="deallocate(a)")
    assert not builder.analysis.descriptor_stability(ENTRY)["complete"]
    native = _native_analysis(builder.analysis)
    original_tokens = dict(builder.analysis._descriptor_window_proofs)
    result = proxy.segment_summary(native, ENTRY, selected, capture_locals=True)
    assert result["complete"]
    assert result["descriptor_window_identity"]
    assert proof.validate(builder.analysis, ENTRY, selected) is proof
    assert not builder.analysis.descriptor_stability(ENTRY)["complete"]
    assert builder.analysis._descriptor_window_proofs == original_tokens
    token = native._descriptor_window_proofs[result["descriptor_window_identity"]]
    assert token is not original_tokens[token.identity]
    assert not builder.edits


def test_excluded_prefix_or_tail_cannot_borrow_selected_descriptor_authority(tmp_path):
    builder, selected, _proof, proxy = bound(tmp_path, attributes=",allocatable", suffix="deallocate(a)")
    prefix = builder.entry.execution.content[0]
    tail = builder.entry.execution.content[-1]
    for excluded in (prefix, tail):
        with pytest.raises(CompilationError, match="summary escapes"):
            proxy.segment_summary(builder.analysis, ENTRY, (excluded,), capture_locals=True)
    assert proxy.segment_summary(builder.analysis, ENTRY, selected, capture_locals=True)["complete"]


def test_changed_original_source_invalidates_source_position_capture(tmp_path):
    builder, _selected, _proof, proxy = bound(tmp_path)
    captured = proxy.capture(builder.entry.scope.bindings["a"])
    builder.entry.scope.path.write_text(builder.entry.scope.path.read_text().replace("counter=2", "counter=3"))
    assert proxy.capture(builder.entry.scope.bindings["a"]) == captured
    with pytest.raises(CompilationError, match="source changed"):
        proxy.finish()
    assert not builder.outputs
    assert not builder.edits


def test_child_builder_cannot_reuse_parent_window_authority(tmp_path):
    builder, _selected, _proof, proxy = bound(tmp_path)
    child = copy(proxy)
    child.entry = builder.analysis.routines["renamed_windows::consume"]
    with pytest.raises(CompilationError, match="registered exact original"):
        child.validate_reached_window()


def test_copied_binding_cannot_borrow_canonical_resource_authority(tmp_path):
    builder, _selected, _proof, proxy = bound(tmp_path)
    with pytest.raises(CompilationError, match="exact original storage binding"):
        proxy.capture(copy(builder.entry.scope.bindings["a"]))


def test_reparsed_equal_source_ids_cannot_borrow_registered_window_authority(tmp_path):
    builder, _selected, _proof, proxy = bound(tmp_path)
    foreign = SourceEffects([builder.entry.scope.path])
    selection = (next(node for node in foreign.routines[ENTRY].execution.content
                      if type(node).__name__ == "Call_Stmt"),)
    assert foreign.sources == builder.analysis.sources
    with pytest.raises(CompilationError, match="exact original source objects"):
        proxy.segment_summary(foreign, ENTRY, selection, capture_locals=True)


def test_reached_branch_descendants_keep_their_original_selected_authority(tmp_path):
    path, _analysis, _routine, _selected, facts = fixture(tmp_path)
    path.write_text(path.read_text().replace("counter=2\ncall consume(a)",
                                            "counter=2\nif(flag) then\ncall consume(a)\nelse\n"
                                            "call consume(a)\nendif"))
    facts["sources"] = {str(path): sha256(path.read_bytes()).hexdigest()}
    facts["participation"] = "serial"
    builder = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    selected = (next(node for node in builder.entry.execution.content
                     if type(node).__name__ == "If_Construct"),)
    proof = prove_reached_window(builder.analysis, ENTRY, selected, (ROOT,), facts)
    proxy = builder.bind_reached_window(proof, selected)
    for child in walk(selected):
        if type(child).__name__ == "Call_Stmt":
            assert proxy.segment_summary(builder.analysis, ENTRY, (child,), capture_locals=True)["complete"]
