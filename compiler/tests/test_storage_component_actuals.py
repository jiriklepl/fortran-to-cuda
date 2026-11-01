"""Fixed scalar record-field CALL association does not replace complete callee effects."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_original_call
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_component_reads import numeric_component_actual, numeric_component_read
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError


def fixture(tmp_path, *, intent="inout", body="x=x+1", actual="points(i)%value", component="real(8)::value=0",
            formal_attributes="", root_attributes="allocatable", extra="", module_extra="", sealed=True):
    path = tmp_path / "actuals.f90"
    path.write_text(f"""module field_storage
implicit none
type record
{component}
real(8),allocatable::unrelated(:)
end type
type(record),{root_attributes}::points(:)
real(8),allocatable::hidden(:)
{module_extra}
contains
subroutine drive(i)
integer,intent(in)::i
call compute({actual})
end subroutine
subroutine compute(x)
real(8),intent({intent}){formal_attributes}::x
{body}
end subroutine
{extra}
end module
""")
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines["field_storage::drive"]
    call = next(node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt")
    return path, analysis, routine, call


def project(analysis):
    return prove_storage_mutation_effects(analysis, "field_storage::drive")


def component(analysis, routine, resolved, association, indices=None):
    return numeric_component_actual(analysis, routine, resolved, association,
        visit_index=(lambda _node: None) if indices is None else indices.append,
        visit_scope=lambda _scope: None, tick=lambda: None)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize(("intent", "body"), [("in", "continue"), ("out", "x=2"), ("inout", "x=x+1")])
def test_numeric_scalar_field_calls_keep_complete_original_callee_without_payload_authority(tmp_path, intent, body, sealed):
    _path, analysis, _routine, _call = fixture(tmp_path, intent=intent, body=body, sealed=sealed)
    proof = project(analysis)
    assert proof.preserves_descriptors()
    public = proof.public()
    assert public["version"] == 18
    assert {row["procedure"] for row in public["closure"]} == {"field_storage::drive", "field_storage::compute"}
    assert public["mutations"] == public["escapes"] == public["frontiers"] == []
    association, = public["original_numeric_component_associations"]
    assert association["resource"] == "field_storage::points%value"
    assert association["formal"] == "x"
    assert association["callee"] == "field_storage::compute"
    for name in ("allocation_or_definition_inferred", "descriptor_mutation_authorized", "payload_access_authorized",
                 "layout_authorized", "capture_authorized", "execution_authorized", "motion_authorized", "alias_authority"):
        assert association[name] is False
    drive = next(row for row in public["closure"] if row["procedure"] == "field_storage::drive")
    assert drive["calls"][0]["canonical_resources"]["field_storage::compute::argument::x"] == "field_storage::points%value"


@pytest.mark.parametrize(("intent", "body"), [("in", "allocate(hidden(2))"), ("inout", "allocate(hidden(2))\nx=1"),
                                                ("out", "allocate(hidden(2))\nx=1")])
def test_scalar_intent_never_hides_callee_allocation(tmp_path, intent, body):
    _path, analysis, _routine, _call = fixture(tmp_path, intent=intent, body=body)
    proof = project(analysis)
    assert not proof.preserves_descriptors()
    assert proof.public()["original_numeric_component_associations"]
    assert any(row["resource"] == "field_storage::hidden" for row in proof.public()["mutations"])


def test_transitive_original_callee_storage_and_escape_effects_remain_in_closure(tmp_path):
    extra = """subroutine leaf(x)
real(8),intent(inout)::x
allocate(hidden(3))
saved=>other
x=2
end subroutine"""
    _path, analysis, _routine, _call = fixture(tmp_path, body="call leaf(x)", extra=extra,
        module_extra="real(8),pointer::saved\nreal(8),target::other")
    proof = project(analysis)
    public = proof.public()
    assert not proof.preserves_descriptors()
    assert {row["procedure"] for row in public["closure"]} == {
        "field_storage::drive", "field_storage::compute", "field_storage::leaf"}
    assert any(row["resource"] == "field_storage::hidden" for row in public["mutations"])
    assert public["escapes"]


def test_unknown_callee_escape_is_not_replaced_with_scalar_field_assumption(tmp_path):
    _path, analysis, _routine, _call = fixture(tmp_path, body="call unknown(x)")
    proof = project(analysis)
    assert not proof.preserves_descriptors()
    assert any("unresolved original call" in row["kind"] for row in proof.public()["frontiers"])


def test_optional_callee_branch_still_keeps_hidden_allocation(tmp_path):
    _path, analysis, _routine, _call = fixture(tmp_path,
        body="if(change) allocate(hidden(2))\nx=x+1", module_extra="logical::change")
    proof = project(analysis)
    assert not proof.preserves_descriptors()
    assert any(row["resource"] == "field_storage::hidden" for row in proof.public()["mutations"])


@pytest.mark.parametrize("attributes", [",pointer", ",target", ",allocatable", ",optional", ",value", ",volatile"])
def test_nonordinary_scalar_formals_keep_their_separate_association_requirements(tmp_path, attributes):
    _path, analysis, _routine, _call = fixture(tmp_path, formal_attributes=attributes)
    proof = project(analysis)
    assert not proof.preserves_descriptors()
    assert not proof.public().get("original_numeric_component_associations")


@pytest.mark.parametrize("component_declaration", ["real(8),pointer::value", "real(8),allocatable::value"])
def test_selected_pointer_or_allocatable_field_cannot_borrow_fixed_scalar_actual_authority(tmp_path, component_declaration):
    _path, analysis, _routine, _call = fixture(tmp_path, component=component_declaration)
    proof = project(analysis)
    assert not proof.preserves_descriptors()
    assert not proof.public().get("original_numeric_component_associations")


@pytest.mark.parametrize("actual", ["points%value", "points(:)%value", "points([i])%value"])
def test_field_actual_requires_one_original_scalar_element_not_whole_array_or_section(tmp_path, actual):
    _path, analysis, _routine, _call = fixture(tmp_path, actual=actual)
    proof = project(analysis)
    assert not proof.preserves_descriptors()
    assert not proof.public().get("original_numeric_component_associations")


@pytest.mark.parametrize("attributes", ["target,allocatable", "pointer", "volatile,allocatable"])
def test_aliasable_root_does_not_acquire_new_component_actual_capability(tmp_path, attributes):
    _path, analysis, _routine, _call = fixture(tmp_path, root_attributes=attributes)
    proof = project(analysis)
    assert not proof.preserves_descriptors()
    assert not proof.public().get("original_numeric_component_associations")


def test_exact_selector_nodes_are_scanned_and_call_occurrence_still_is_not_an_ordinary_read(tmp_path):
    _path, analysis, routine, call = fixture(tmp_path, actual="points(i+1)%value")
    resolved = resolve_original_call(analysis, routine.scope, call)
    item, = resolved.associations
    indices = []
    result = component(analysis, routine, resolved, item, indices)
    assert tuple(indices) == result.indices == (item.actual.items[0].items[1].items[0],)
    assert str(indices[0]) == "i + 1"
    with pytest.raises(CompilationError, match="address"):
        numeric_component_read(analysis, routine, item.actual, visit_index=lambda _node: None,
                               visit_scope=lambda _scope: None, tick=lambda: None)


@pytest.mark.parametrize("selector", ["unknown()", "select_index()"])
def test_unknown_or_impure_selector_cannot_gain_scalar_actual_authority(tmp_path, selector):
    extra = "integer function select_index()\nallocate(hidden(2))\nselect_index=1\nend function"
    _path, analysis, _routine, _call = fixture(tmp_path, actual=f"points({selector})%value", extra=extra)
    proof = project(analysis)
    assert not proof.preserves_descriptors()
    assert proof.public()["frontiers"]
    assert not proof.public().get("original_numeric_component_associations")


@pytest.mark.parametrize("second", ["value", "other"])
def test_multiple_numeric_formals_retain_exact_field_mapping_without_disjointness_inference(tmp_path, second):
    path, _analysis, _routine, _call = fixture(tmp_path, component="real(8)::value=0,other=0")
    source = path.read_text().replace("call compute(points(i)%value)",
        f"call compute(points(i)%value,points(i)%{second})").replace("subroutine compute(x)",
        "subroutine compute(x,y)").replace("real(8),intent(inout)::x", "real(8),intent(inout)::x,y")
    path.write_text(source)
    analysis = SourceEffects([path])
    analysis.seal_session()
    proof = project(analysis)
    assert proof.preserves_descriptors()
    public = proof.public()
    first, second_actual = public["original_numeric_component_associations"]
    assert first["resource"] == "field_storage::points%value"
    assert second_actual["resource"] == "field_storage::points%" + second
    assert first["alias_authority"] is second_actual["alias_authority"] is False


@pytest.mark.parametrize("change", ["copied_call", "copied_association", "foreign_call"])
def test_component_actual_requires_exact_original_issued_call_and_association(tmp_path, change):
    path, analysis, routine, call = fixture(tmp_path)
    resolved = resolve_original_call(analysis, routine.scope, call)
    item, = resolved.associations
    if change == "copied_call":
        resolved = copy(resolved)
    elif change == "copied_association":
        item = copy(item)
    else:
        other = SourceEffects([path])
        other.seal_session()
        foreign = other.routines[routine.qualified]
        foreign_call = next(node for node in walk(foreign.execution) if type(node).__name__ == "Call_Stmt")
        resolved = resolve_original_call(other, foreign.scope, foreign_call)
        item, = resolved.associations
    with pytest.raises(CompilationError, match="original|association"):
        component(analysis, routine, resolved, item)


def test_changed_original_file_is_rejected_at_publication_and_foreign_context_cannot_consume_fact(tmp_path):
    path, analysis, _routine, _call = fixture(tmp_path)
    proof = project(analysis)
    assert proof.preserves_descriptors()
    other = SourceEffects([path])
    other.seal_session()
    with pytest.raises(CompilationError):
        proof.validate(other)
    public = proof.public()
    public["original_numeric_component_associations"].clear()
    assert proof.public()["original_numeric_component_associations"]
    path.write_text(path.read_text().replace("x=x+1", "allocate(hidden(2))\nx=x+1"))
    with pytest.raises(CompilationError):
        analysis.verify_publication()


@pytest.mark.parametrize("sealed", [False, True])
def test_completed_static_scope_checks_are_shared_only_inside_one_sealed_storage_demand(tmp_path, monkeypatch, sealed):
    from collections import Counter

    import compiler.frontend.storage_component_reads as reads

    path, _analysis, _routine, _call = fixture(tmp_path)
    path.write_text(path.read_text().replace("call compute(points(i)%value)",
                                           "\n".join(["call compute(points(i)%value)"] * 20)))
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    counts = Counter()
    original = reads._component_scope_syntax

    def checked(scope, tick):
        counts[id(scope)] += 1
        return original(scope, tick)

    monkeypatch.setattr(reads, "_component_scope_syntax", checked)
    proof = project(analysis)
    assert proof.preserves_descriptors()
    assert len(proof.public()["original_numeric_component_associations"]) == 20
    count = counts[id(analysis.modules["field_storage"])]
    assert count == 1 if sealed else count > 1


def test_sealed_scope_sharing_keeps_the_complete_public_storage_projection(tmp_path):
    path, mutable, _routine, _call = fixture(tmp_path, sealed=False)
    expected = project(mutable).public()
    sealed = SourceEffects([path])
    sealed.seal_session()
    assert project(sealed).public() == expected


@pytest.mark.parametrize("addition", ["common /shared/ extra", "!$omp threadprivate(extra)"])
def test_added_alias_or_openmp_declaration_after_cached_check_rejects_at_publication(tmp_path, monkeypatch, addition):
    from compiler.frontend.source_dependency_receipts import DependencyCapture
    from compiler.frontend.storage_mutation_effects import _Scan

    _path, analysis, _routine, _call = fixture(tmp_path, module_extra="real(8)::extra")
    owner = analysis.modules["field_storage"]
    specification = next(part for part in owner.node.content if type(part).__name__ == "Specification_Part")
    foreign_path = tmp_path / "added.f90"
    foreign_path.write_text(f"module added\nreal(8)::extra\n{addition}\nend module\n")
    foreign = SourceEffects([foreign_path])
    kind = "Common_Stmt" if addition.startswith("common") else "Comment"
    declaration = next(node for node in walk(foreign.modules["added"].node)
                       if type(node).__name__ == kind)
    state = {"capture": None, "changed": False}
    scope_check, receipt = _Scan.component_scope, DependencyCapture.receipt

    def remember(scan, scope, check):
        scope_check(scan, scope, check)
        if scope is owner:
            assert id(scope) in scan.component_scope_checks
            state["capture"] = scan.declaration_dependencies

    def publish(capture):
        retained = receipt(capture)
        if capture is state["capture"] and not state["changed"]:
            object.__setattr__(specification, "content", (*specification.content, declaration))
            state["changed"] = True
        return retained

    monkeypatch.setattr(_Scan, "component_scope", remember)
    monkeypatch.setattr(DependencyCapture, "receipt", publish)
    with pytest.raises(CompilationError):
        project(analysis)
    assert state["changed"]
