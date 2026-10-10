"""Whole original teams may read a coherent, shared constant array point."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.segments import grouped_nodes
from compiler.scopes.source import form_source_scopes
from compiler.tests.test_source_scopes import FACT

ENTRY = "switch_owner::advance"


def source_text(*, condition="gate(0)>0", gate="real(8),intent(inout)::gate(-2:)",
                specification="integer,parameter::slot=0", update="", private="i", branch=None,
                module_specification="", imports=""):
    unit = "!$omp do\ndo i=1,n\nb(i)=b(i)+a(i)\n" + update + "\nenddo\n!$omp end do nowait\n"
    branch = branch or "if(" + condition + ") then\n" + unit + "else\n" + unit + "endif\n"
    return """module switch_owner
implicit none
""" + module_specification + """
contains
subroutine advance(a,gate,b,out,n,flag)
""" + imports + """
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),out(:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
""" + gate + "\n" + specification + """
continue
!$omp parallel private(""" + private + ")\n" + branch + """!$omp end parallel
end subroutine
end module
"""


def analyzed(tmp_path, text=None, **kwargs):
    path = tmp_path / "switches.f90"
    path.write_text(text if text is not None else source_text(**kwargs))
    analysis = SourceEffects([path])
    original = tuple(node for node in analysis.routines[ENTRY].execution.content
                     if type(node).__name__ != "Continue_Stmt")
    group, = grouped_nodes(original)
    return path, analysis, group


@pytest.mark.parametrize("dtype", ["real(4)", "real(8)", "integer"])
@pytest.mark.parametrize(("shape", "storage"), [("(-2:2)", "fixed_explicit_shape"),
                                           ("(-2:)", "stable_original_assumed_shape")])
def test_original_constant_point_uniformity_has_explicit_runtime_requirements(tmp_path, dtype, shape, storage):
    path, analysis, group = analyzed(tmp_path, gate=dtype + ",intent(inout)::gate" + shape)
    before = path.read_bytes()
    proof = analysis.joined_completion(ENTRY, group)
    public = proof.public()
    assert public["schema_version"] == 5
    assert public["uniform_array_read_contract"] == "fixed-rank-one-shared-constant-point-v1"
    fact, = public["uniform_array_reads"]
    assert fact["resource"] == "argument::gate"
    assert fact["subscript"] == 0
    assert fact["storage"] == storage
    assert fact["shared_and_unwritten_in_complete_team"]
    assert fact["requires_registered_storage_and_alias_validation"]
    assert fact["requires_host_coherence_before_original_team"]
    assert not public["guarded_fixed_bound_array_conditions_authorized"]
    assert not public["guarded_assumed_shape_array_conditions_authorized"]
    assert not public["gpu_independence_established"]
    public["uniform_array_reads"].clear()
    assert proof.public()["uniform_array_reads"]
    assert path.read_bytes() == before


def test_assumed_shape_condition_effects_include_exact_point_before_original_team(tmp_path):
    _, analysis, group = analyzed(tmp_path, condition="gate(-1)>0")
    proof = analysis.joined_completion(ENTRY, group)
    summary = analysis.segment_summary(ENTRY, group, capture_locals=True)
    effects = [item for item in summary["operations"] if item.get("resource") == "argument::gate"]
    assert any(item["kind"] == "read" for item in effects)
    assert not any(item["kind"] in {"write", "overwrite"} for item in effects)
    sections = analysis.native_sections_for_nodes(ENTRY, group, completion=proof, capture_locals=True)
    assert sections.available, sections.reason
    resource, = [item for item in sections.resources if item.resource == "argument::gate"]
    point, = resource.reads
    assert point.axes[0].point
    assert point.axes[0].lower.value == point.axes[0].upper.value == -1
    assert resource.lower_bounds == (-2,)
    assert resource.writes == resource.overwrites == ()


def test_imported_renamed_parameter_uses_its_original_declaring_scope(tmp_path):
    header = """module immutable_indices
implicit none
integer,parameter::chosen=-1
end module
"""
    text = header + source_text(condition="gate(alias)>0", specification="",
                                 imports="use immutable_indices,only:alias=>chosen\n")
    _, analysis, group = analyzed(tmp_path, text)
    proof = analysis.joined_completion(ENTRY, group)
    fact, = proof.public()["uniform_array_reads"]
    assert fact["subscript"] == -1
    assert fact["parameter_resource"] == "immutable_indices::chosen"
    sections = analysis.native_sections_for_nodes(ENTRY, group, completion=proof, capture_locals=True)
    assert sections.available, sections.reason
    resource, = [item for item in sections.resources if item.resource == "argument::gate"]
    assert resource.reads[0].axes[0].lower.value == -1


@pytest.mark.parametrize(("condition", "specification", "reason"), [
    ("gate(slot)>0", "integer::slot", "original INTEGER PARAMETER"),
    ("gate(i)>0", "", "original INTEGER PARAMETER"),
    ("gate(slot+1)>0", "integer,parameter::slot=0", "literal or original INTEGER PARAMETER"),
    ("gate(int(0.0))>0", "", "unproved uniform|uniform scalar|literal or original"),
    ("gate(slot(1))>0", "integer::slot(1)", "literal or original INTEGER PARAMETER"),
    ("gate(0:0)>0", "", "literal or original INTEGER PARAMETER"),
    ("gate(0,0)>0", "", "one constant scalar subscript"),
    ("gate(2147483648_8)>0", "", "INTEGER"),
])
def test_uncertain_or_unbounded_subscripts_remain_boundaries(tmp_path, condition, specification, reason):
    _, analysis, group = analyzed(tmp_path, condition=condition, specification=specification)
    with pytest.raises(CompilationError, match=reason):
        analysis.joined_completion(ENTRY, group)


def test_local_shadow_does_not_borrow_host_parameter_identity(tmp_path):
    text = source_text(condition="gate(slot)>0", specification="integer::slot",
                       module_specification="integer,parameter::slot=-1")
    _, analysis, group = analyzed(tmp_path, text)
    with pytest.raises(CompilationError, match="original INTEGER PARAMETER"):
        analysis.joined_completion(ENTRY, group)


@pytest.mark.parametrize(("attributes", "shape"), [
    ("allocatable", "(:)"), ("pointer", "(:)"), ("optional", "(-2:)"),
    ("target", "(-2:)"), ("volatile", "(-2:)"), ("asynchronous", "(-2:)"),
])
def test_uncertain_storage_or_association_is_not_uniform_authority(tmp_path, attributes, shape):
    _, analysis, group = analyzed(tmp_path, gate=f"real(8),intent(inout),{attributes}::gate{shape}")
    with pytest.raises(CompilationError, match="fixed shared rank-one numeric storage"):
        analysis.joined_completion(ENTRY, group)


@pytest.mark.parametrize(("condition", "gate", "specification", "reason"), [
    ("gate(0)>0", "real(8)::gate(-2:2,2)", "", "rank-one"),
    ("gate(0)", "logical::gate(-2:2)", "", "numeric storage"),
    ("gate(3)>0", "real(8)::gate(-2:2)", "", "outside original fixed bounds"),
    ("gate(0)>0", "real(8)::gate(-2:n)", "", "unresolved INTEGER kind parameter"),
])
def test_other_array_forms_do_not_gain_the_narrow_proof(tmp_path, condition, gate, specification, reason):
    _, analysis, group = analyzed(tmp_path, condition=condition, gate=gate, specification=specification)
    with pytest.raises(CompilationError, match=reason):
        analysis.joined_completion(ENTRY, group)


def test_private_array_and_any_whole_team_write_are_rejected(tmp_path):
    text = source_text(gate="", specification="real(8)::local(-2:2)", private="i,local",
                       condition="local(0)>0")
    for program in (text, source_text(update="gate(0)=a(i)")):
        _, analysis, group = analyzed(tmp_path, program)
        with pytest.raises(CompilationError, match="private or changes"):
            analysis.joined_completion(ENTRY, group)


def test_write_in_inactive_alternative_still_prevents_uniform_proof(tmp_path):
    unit = "!$omp do\ndo i=1,n\nb(i)=a(i)\nenddo\n!$omp end do\n"
    branch = "if(gate(0)>0) then\n" + unit + "else\n"
    branch += "!$omp do\ndo i=1,n\ngate(1)=a(i)\nenddo\n!$omp end do\nendif\n"
    _, analysis, group = analyzed(tmp_path, branch=branch)
    with pytest.raises(CompilationError, match="private or changes"):
        analysis.joined_completion(ENTRY, group)


def test_imported_threadprivate_original_storage_remains_a_boundary(tmp_path):
    text = """module per_worker
implicit none
integer::selector(-2:2)
!$omp threadprivate(selector)
end module
""" + source_text(gate="", imports="use per_worker,only:gate=>selector\n")
    _, analysis, group = analyzed(tmp_path, text)
    with pytest.raises(CompilationError, match="THREADPRIVATE"):
        analysis.joined_completion(ENTRY, group)


def test_storage_association_is_rejected_before_uniform_payload_authority(tmp_path):
    text = source_text(gate="", specification="integer::gate(-2:2),other(-2:2)\n"
                                                "equivalence(gate,other)")
    _, analysis, group = analyzed(tmp_path, text)
    with pytest.raises(CompilationError, match="storage association"):
        analysis.joined_completion(ENTRY, group)


def test_canonical_resource_identity_boundary_is_preserved(tmp_path, monkeypatch):
    _, analysis, group = analyzed(tmp_path)
    monkeypatch.setattr(analysis, "resource_identity_boundary", lambda binding:
                        "unproved original storage alias" if binding.name == "gate" else None)
    with pytest.raises(CompilationError, match="unproved original storage alias"):
        analysis.joined_completion(ENTRY, group)


@pytest.mark.parametrize("style", ["nested", "elseif"])
def test_guarded_points_are_not_eagerly_refined_before_an_original_condition(tmp_path, style):
    unit = "!$omp do\ndo i=1,n\nb(i)=a(i)\nenddo\n!$omp end do\n"
    if style == "nested":
        branch = "if(flag) then\nif(gate(0)>0) then\n" + unit + "endif\nendif\n"
    else:
        branch = "if(flag) then\n" + unit + "else if(gate(0)>0) then\n" + unit + "endif\n"
    _, analysis, group = analyzed(tmp_path, branch=branch)
    with pytest.raises(CompilationError, match="guarded uniform array condition"):
        analysis.joined_completion(ENTRY, group)


@pytest.mark.parametrize("style", ["nested", "elseif", "compound"])
def test_guarded_fixed_storage_points_have_proved_valid_coordinates(tmp_path, style):
    unit = "!$omp do\ndo i=1,n\nb(i)=a(i)\nenddo\n!$omp end do\n"
    if style == "nested":
        branch = "if(flag) then\nif(gate(slot)>0) then\n" + unit + "endif\nendif\n"
    elif style == "elseif":
        branch = "if(flag) then\n" + unit + "else if(gate(slot)>0) then\n" + unit + "endif\n"
    else:
        branch = "if(flag.and.gate(slot)>0) then\n" + unit + "endif\n"
    # Configuration storage is physically present independently of which
    # original branch is reached. No condition payload is evaluated here.
    _, analysis, group = analyzed(tmp_path, gate="", specification="integer,parameter::slot=2",
                                  module_specification="integer::gate(6)", branch=branch)
    proof = analysis.joined_completion(ENTRY, group)
    fact, = proof.public()["uniform_array_reads"]
    assert fact["resource"] == "switch_owner::gate"
    assert fact["storage"] == "fixed_explicit_shape"
    assert (fact["original_lower_bound"], fact["original_upper_bound"]) == (1, 6)
    assert fact["subscript"] == 2
    assert proof.public()["guarded_fixed_bound_array_conditions_authorized"] == (style != "compound")


def test_repeated_fixed_point_preserves_each_condition_context_in_public_proof(tmp_path):
    unit = "!$omp do\ndo i=1,n\nb(i)=a(i)\nenddo\n!$omp end do\n"
    branch = "if(flag) then\nif(gate(slot)>0) then\n" + unit + "endif\n"
    branch += "else if(gate(slot)>0) then\n" + unit + "endif\n"
    branch += "if(flag.and.gate(slot)>0) then\n" + unit + "endif\n"
    # The later unconditional read must not overwrite the guarded fact for
    # exactly the same canonical storage and source point.
    branch += "if(gate(slot)>0) then\n" + unit + "endif\n"
    _, analysis, group = analyzed(tmp_path, gate="", specification="integer,parameter::slot=2",
                                  module_specification="integer::gate(6)", branch=branch)
    public = analysis.joined_completion(ENTRY, group).public()
    facts = public["uniform_array_reads"]
    assert len(facts) == 3
    assert {item["resource"] for item in facts} == {"switch_owner::gate"}
    assert {item["subscript"] for item in facts} == {2}
    assert {(item["guarded_condition"], item["compound_logical_condition"]) for item in facts} == {
        (True, False), (False, True), (False, False)}
    assert public["guarded_fixed_bound_array_conditions_authorized"]
    assert not public["guarded_assumed_shape_array_conditions_authorized"]


def test_assumed_shape_point_cannot_borrow_a_short_circuit_guard(tmp_path):
    _, analysis, group = analyzed(tmp_path, condition="flag.and.gate(0)>0")
    with pytest.raises(CompilationError, match="compound uniform array condition"):
        analysis.joined_completion(ENTRY, group)


def test_proof_copy_and_changed_source_cannot_supply_new_payload_authority(tmp_path):
    path, analysis, group = analyzed(tmp_path)
    proof = analysis.joined_completion(ENTRY, group)
    with pytest.raises(CompilationError, match="whole original joined-group authority"):
        analysis.native_sections_for_nodes(ENTRY, group, completion=replace(proof), capture_locals=True)
    path.write_text(path.read_text().replace("gate(0)>0", "gate(1)>0"))
    with pytest.raises(CompilationError, match="changed"):
        analysis.native_sections_for_nodes(ENTRY, group, completion=proof, capture_locals=True)


def test_reached_source_hooks_publish_the_point_before_the_unchanged_complete_team(tmp_path):
    text = source_text()
    text = text.replace("continue\n!$omp parallel", """do i=1,n
gate(i)=a(i)
b(i)=2*a(i)+real(i,8)
enddo
!$omp parallel""")
    text = text.replace("!$omp end parallel\nend subroutine", """!$omp end parallel
do i=1,n
out(i)=b(i)+b(1)
enddo
end subroutine""")
    path = tmp_path / "producer_branch_consumer.f90"
    path.write_text(text)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: FACT for name in ("a", "gate", "b", "out")}}
    outputs, report = form_source_scopes([path], ENTRY, facts=facts,
        options=CompilerOptions(fallback="host", gpu_policy="sections", memory_model="scoped", scope_execution="reached"),
        config=OffloadConfig("sections"))
    owner, = report["scopes"]
    assert len(owner["gpu_leaves"]) == 2
    assert not owner.get("boundaries"), owner.get("boundaries")
    operation, = [item for item in owner["native_operations"] if item["kind"] == "joined native OpenMP"]
    assert operation["sections"]["available"], operation["sections"]
    point, = [item for item in operation["sections"]["resources"] if item["resource"] == "argument::gate"]
    assert point["reads"][0]["axes"][0]["lower"]["value"] == 0
    generated = outputs[report["sources"][str(path)]["replacement"]]
    begin = generated.index("fort_scope_host_begin(")
    opening = generated.lower().index("!$omp parallel private(i)", begin)
    ending = generated.lower().index("!$omp end parallel", opening)
    assert begin < opening
    assert generated.find("fort_scope_host_end(", ending) >= 0
    assert "if(gate(0)>0)then" in generated.lower().replace(" ", "")
    assert text == path.read_text()
