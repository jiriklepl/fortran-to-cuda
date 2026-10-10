"""Source-position coverage is propagated, never recreated by fresh handles."""

from copy import copy
from dataclasses import replace

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.reached_windows import prove_reached_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "renamed_windows::advance"
ROOT = "argument::a"


def fixture(tmp_path, prefix="counter=2", selected="call consume(a)", suffix="", *, intent="inout",
            attributes="", declaration="", helper="", initialized="whole", operations=256):
    source = tmp_path / "windows.f90"
    source.write_text(f"""module renamed_windows
contains
subroutine consume(a)
real(8),intent(inout)::a(:)
a=a+1.0_8
end subroutine
{helper}
subroutine advance(a,flag)
real(8){attributes},intent({intent})::a(:)
logical,intent(in)::flag
integer::counter
{declaration}
{prefix}
{selected}
{suffix}
end subroutine
end module
""")
    analysis = SourceEffects([source], operations=operations)
    routine = analysis.routines[ENTRY]
    calls = tuple(node for node in routine.execution.content if type(node).__name__ == "Call_Stmt")
    selection = (next(node for node in calls if str(node.items[0]).lower() == "consume"),)
    facts = {"schema_version": 1, "sources": dict(analysis.sources), "captures": {
        ROOT: {"storage": "stable", "escapes": False, "allocation_changes": False, "initialized": initialized}}}
    return source, analysis, routine, selection, facts


def prove(analysis, selection, facts, roots=(ROOT,)):
    return prove_reached_window(analysis, ENTRY, selection, roots, facts)


def test_exact_later_window_preserves_complete_scalar_native_prefix(tmp_path):
    source, analysis, _routine, selected, facts = fixture(tmp_path, suffix="call unknown_tail()")
    original = source.read_bytes()
    proof = prove(analysis, selected, facts)
    assert proof.validate(analysis, ENTRY, selected) is proof
    assert prove(analysis, selected, facts) is proof
    assert proof.capture_state(ROOT)["initialized"] == "whole"
    assert proof.capture_state(ROOT)["coverage_on_return"] == "whole"
    assert proof.capture_state(ROOT)["requires_runtime_layout_and_alias_guards"]
    assert not proof.capture_state(ROOT)["address_or_shape_freshness_inferred"]
    public = proof.public()
    assert public["prefix_node_ids"]
    assert public["lifecycle"].endswith("no reopening")
    assert not public["execution_authorized"]
    assert not public["gpu_legality_established"]
    assert not public["definition_facts_reset"]
    assert public["entry_facts_authority"].endswith("independent admission required")
    assert source.read_bytes() == original
    assert not hasattr(analysis, "context")


@pytest.mark.parametrize("prefix", ["a=3.0_8", "if(flag) then\na=3.0_8\nelse\na=4.0_8\nend if"])
def test_original_whole_writes_establish_source_position_coverage(tmp_path, prefix):
    _path, analysis, _routine, selected, facts = fixture(tmp_path, prefix, initialized="none")
    proof = prove(analysis, selected, facts)
    assert proof.capture_state(ROOT)["initialized"] == "whole"
    assert facts["captures"][ROOT]["initialized"] == "none"


@pytest.mark.parametrize("prefix", ["a=a+1.0_8", "a(1)=3.0_8", "if(flag) a=3.0_8"])
def test_partial_or_one_arm_definition_does_not_manufacture_whole(tmp_path, prefix):
    _path, analysis, _routine, selected, facts = fixture(tmp_path, prefix, initialized="none")
    with pytest.raises(CompilationError, match="incomplete definition"):
        prove(analysis, selected, facts)


def test_original_entry_out_kills_supplied_whole_fact(tmp_path):
    _path, analysis, _routine, selected, facts = fixture(tmp_path, intent="out")
    with pytest.raises(CompilationError, match="incomplete definition"):
        prove(analysis, selected, facts)


@pytest.mark.parametrize("whole", [False, True])
def test_original_child_out_then_definition_is_transferred_in_order(tmp_path, whole):
    target = "a" if whole else "a(1)"
    helper = f"""subroutine initialize(a)
real(8),intent(out)::a(:)
{target}=3.0_8
end subroutine
"""
    _path, analysis, _routine, selected, facts = fixture(tmp_path, "call initialize(a)", helper=helper)
    if whole:
        assert prove(analysis, selected, facts).capture_state(ROOT)["initialized"] == "whole"
    else:
        with pytest.raises(CompilationError, match="incomplete definition"):
            prove(analysis, selected, facts)


def test_explicit_shape_child_cannot_restore_full_actual_after_out(tmp_path):
    helper = """subroutine initialize(a)
real(8),intent(out)::a(1)
a=3.0_8
end subroutine
"""
    _path, analysis, _routine, selected, facts = fixture(tmp_path, "call initialize(a)", helper=helper)
    with pytest.raises(CompilationError, match="incomplete definition"):
        prove(analysis, selected, facts)


@pytest.mark.parametrize("prefix", ["call unavailable(a)", "call unavailable()", "allocate(a(7))"])
def test_unknown_effects_or_allocation_prefix_never_resets_facts(tmp_path, prefix):
    attributes = ",allocatable" if prefix.startswith("allocate") else ""
    _path, analysis, _routine, selected, facts = fixture(tmp_path, prefix, attributes=attributes)
    with pytest.raises(CompilationError, match="unknown effects|lifetime authority|complete prefix"):
        prove(analysis, selected, facts)
    assert not getattr(analysis, "_reached_window_proofs", {})


def test_same_shape_reallocation_in_suffix_is_honestly_unavailable(tmp_path):
    _path, analysis, _routine, selected, facts = fixture(tmp_path, suffix="deallocate(a)\nallocate(a(7))",
                                                     attributes=",allocatable")
    with pytest.raises(CompilationError, match="source-relative lifetime authority"):
        prove(analysis, selected, facts)


def test_unchanged_allocatable_input_retains_original_descriptor_guard(tmp_path):
    _path, analysis, _routine, selected, facts = fixture(tmp_path, attributes=",allocatable")
    proof = prove(analysis, selected, facts)
    assert proof.capture_state(ROOT)["requires_original_allocation_presence_guard"]


def test_untracked_arrays_cannot_disappear_as_unrelated_work(tmp_path):
    _path, analysis, _routine, selected, facts = fixture(tmp_path, "b=3.0_8", declaration="real(8)::b(7)")
    with pytest.raises(CompilationError, match="untracked array.*alias authority"):
        prove(analysis, selected, facts)


def test_all_array_dependencies_and_runtime_nonalias_obligations_are_explicit(tmp_path):
    _path, analysis, _routine, selected, facts = fixture(tmp_path, "b=3.0_8\na=b", declaration="real(8)::b(7)",
                                                     initialized="none")
    other = ENTRY + "::b"
    facts["captures"][other] = {"storage": "stable", "escapes": False, "allocation_changes": False,
                                 "initialized": "none"}
    proof = prove(analysis, selected, facts, roots=(ROOT, other))
    assert proof.capture_state(ROOT)["initialized"] == "whole"
    assert proof.capture_state(other)["initialized"] == "whole"
    assert all(state["requires_runtime_layout_and_alias_guards"] for state in proof.public()["states"])


def test_full_original_sweep_uses_existing_whole_definition_proof(tmp_path):
    prefix = "do counter=lbound(a,1),ubound(a,1)\na(counter)=3.0_8\nend do"
    _path, analysis, _routine, selected, facts = fixture(tmp_path, prefix, initialized="none")
    assert prove(analysis, selected, facts).capture_state(ROOT)["initialized"] == "whole"


def test_complete_control_paths_are_met_without_evaluating_flags(tmp_path):
    prefix = "if(flag) then\na=3.0_8\nelse\na(1)=4.0_8\nend if"
    _path, analysis, routine, selected, facts = fixture(tmp_path, prefix, initialized="none")
    with pytest.raises(CompilationError, match="incomplete definition"):
        prove(analysis, selected, facts)
    assert "IF (flag)" in str(routine.execution)


def test_nested_and_partial_joined_selections_need_distinct_authority(tmp_path):
    _path, analysis, routine, _selected, facts = fixture(tmp_path, prefix="do counter=1,3\ncall consume(a)\nend do")
    nested = next(node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt")
    with pytest.raises(CompilationError, match="top-level"):
        prove(analysis, (nested,), facts)
    _path, analysis, routine, selected, facts = fixture(tmp_path, prefix="!$omp parallel\na=3.0_8\n!$omp end parallel")
    with pytest.raises(CompilationError, match="whole-completion authority"):
        prove(analysis, selected, facts)


@pytest.mark.parametrize(("field", "value"), [
    ("initialized", "sections"), ("initialized", []), ("escapes", True), ("allocation_changes", True),
])
def test_incomplete_seed_authority_rejects(tmp_path, field, value):
    _path, analysis, _routine, selected, facts = fixture(tmp_path)
    facts["captures"][ROOT][field] = value
    with pytest.raises(CompilationError, match="entry coverage/no-escape"):
        prove(analysis, selected, facts)


def test_selection_and_registry_are_exact_and_source_bound(tmp_path):
    path, analysis, routine, selected, facts = fixture(tmp_path)
    proof = prove(analysis, selected, facts)
    for forged in (copy(proof), replace(proof)):
        with pytest.raises(CompilationError, match="registered exact original"):
            forged.validate(analysis, ENTRY, selected)
    for selection in ((F.Call_Stmt("call consume(a)"),), (selected[0], selected[0]),
                      (routine.execution.content[0], selected[0], routine.execution.content[0])):
        with pytest.raises(CompilationError, match="original|contiguous"):
            prove(analysis, selection, facts)
    with pytest.raises(CompilationError, match="registered exact original"):
        proof.validate(SourceEffects([path]), ENTRY, selected)
    with pytest.raises(CompilationError, match="capture root"):
        proof.capture_state("argument::unproved")
    public = proof.public()
    public["states"][0]["initialized"] = "none"
    assert proof.capture_state(ROOT)["initialized"] == "whole"
    path.write_text(path.read_text().replace("counter=2", "counter=3"))
    with pytest.raises(CompilationError, match="source changed"):
        proof.capture_state(ROOT)


def test_proof_requires_matching_seed_sources_and_finite_facts(tmp_path):
    _path, analysis, _routine, selected, facts = fixture(tmp_path)
    facts["sources"] = {}
    with pytest.raises(CompilationError, match="matching invocation-entry"):
        prove(analysis, selected, facts)
    facts["bad"] = float("nan")
    with pytest.raises(CompilationError, match="finite source facts"):
        prove(analysis, selected, facts)


def test_multiple_complete_original_statements_form_one_window(tmp_path):
    _path, analysis, routine, _selected, facts = fixture(tmp_path, selected="call consume(a)\ncall consume(a)")
    selected = tuple(node for node in routine.execution.content if type(node).__name__ == "Call_Stmt")
    proof = prove(analysis, selected, facts)
    assert len(proof.public()["selected_node_ids"]) == 2
    with pytest.raises(CompilationError, match="contiguous original"):
        prove(analysis, tuple(reversed(selected)), facts)


def test_source_point_undefined_output_can_be_registered_without_inventing_definition(tmp_path):
    _path, analysis, routine, _selected, facts = fixture(tmp_path, prefix="counter=2", selected="a=3.0_8\ncall consume(a)",
                                                     intent="out")
    selected = tuple(routine.execution.content[1:])
    proof = prove(analysis, selected, facts)
    state = proof.capture_state(ROOT)
    assert state["initialized"] == "none"
    assert state["coverage_on_return"] == "whole"
    assert not proof.public()["definition_facts_reset"]


@pytest.mark.parametrize(("prefix", "declaration", "attributes"), [
    ("counter=2", "real(8)::scratch(7)\ncommon /hidden/ scratch", ""),
    ("q=>a", "real(8),pointer::q(:)", ",target"),
])
def test_storage_alias_or_escape_requires_distinct_authority(tmp_path, prefix, declaration, attributes):
    _path, analysis, _routine, selected, facts = fixture(tmp_path, prefix, declaration=declaration,
                                                     attributes=attributes)
    with pytest.raises(CompilationError, match="alias authority|unknown effects"):
        prove(analysis, selected, facts)


def test_seed_snapshot_cannot_be_changed_after_proof_issue(tmp_path):
    _path, analysis, _routine, selected, facts = fixture(tmp_path)
    proof = prove(analysis, selected, facts)
    facts["captures"][ROOT]["initialized"] = "none"
    assert proof.capture_state(ROOT)["initialized"] == "whole"
    assert proof.validate(analysis, ENTRY, selected) is proof
