"""Assignment indices are original reads, without field-write or payload credit."""
import json
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_component_reads import numeric_component_read
from compiler.frontend.storage_mutation_effects import _original, prove_storage_mutation_effects
from compiler.ir import CompilationError


def fixture(tmp_path, *, body="field(points(i)%xi,points(i)%yj,points(i)%zk)=1", sealed=True,
            field_type="integer::xi,yj,zk", root_attributes="allocatable", extra=""):
    path = tmp_path / "selectors.f90"
    path.write_text(f"""module records
implicit none
type point
 {field_type}
 real(8),allocatable::unrelated(:)
end type
type(point),{root_attributes}::points(:)
type(point)::current
real(8),allocatable::field(:,:,:),hidden(:)
integer,allocatable::indices(:)
contains
subroutine advance(i)
integer,intent(in)::i
{body}
end subroutine
{extra}
end module
""")
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = _original(analysis, "records::advance")
    nodes = [node for node in walk(routine.execution) if type(node).__name__ == "Data_Ref"]
    return path, analysis, routine, nodes


def read(analysis, routine, node, callback=lambda _node: None):
    return numeric_component_read(analysis, routine, node, visit_index=callback,
        visit_scope=lambda _scope: None, tick=lambda: None)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("body", [
    "field(points(i)%xi,points(i)%yj,points(i)%zk)=1",
    "field(1+points(i)%xi,1,1)=2",
    "field(points(i)%xi:points(i)%yj,1,1)=2",
    "field(indices(points(i)%xi),1,1)=2",
    "field(current%xi,1,1)=2",
])
def test_original_array_target_selectors_are_reads_with_no_payload_authority(tmp_path, sealed, body):
    _path, analysis, routine, nodes = fixture(tmp_path, body=body, sealed=sealed)
    for node in nodes:
        record = read(analysis, routine, node).public()
        assert record["terminal"] == {"type": "integer", "kind": 4, "rank": 0}
        for name in ("descriptor_mutation_authorized", "allocation_or_definition_inferred",
                     "payload_access_authorized", "execution_authorized", "alias_authority"):
            assert record[name] is False
    proof = prove_storage_mutation_effects(analysis, routine.qualified)
    assert proof.preserves_descriptors(), proof.public()["frontiers"]
    assert proof.public()["mutations"] == proof.public()["escapes"] == []


@pytest.mark.parametrize("body", [
    "points(i)%xi=1",
    "points(i)%xi=points(i)%yj",
    "call unknown(points(i)%xi)",
    "read(*,*)field(points(i)%xi,1,1)",
    "allocate(field(points(i)%xi,1,1))",
    "field(unknown(points(i)%xi),1,1)=1",
    "field(.custom.points(i)%xi,1,1)=1",
])
def test_real_writes_address_contexts_and_unproved_index_functions_still_reject(tmp_path, body):
    _path, analysis, routine, nodes = fixture(tmp_path, body=body)
    with pytest.raises(CompilationError):
        read(analysis, routine, nodes[0])


@pytest.mark.parametrize("field_type", ["integer,pointer::xi", "integer,allocatable::xi"])
def test_selected_owned_or_pointer_field_does_not_become_fixed_index_authority(tmp_path, field_type):
    _path, analysis, routine, nodes = fixture(tmp_path, body="field(points(i)%xi,1,1)=1", field_type=field_type)
    with pytest.raises(CompilationError):
        read(analysis, routine, nodes[0])


def test_original_record_element_indices_still_reach_the_same_scanner(tmp_path):
    _path, analysis, routine, nodes = fixture(tmp_path, body="field(points(i+1)%xi,1,1)=1")
    seen = []
    result = read(analysis, routine, nodes[0], seen.append)
    assert seen == list(result.indices)
    assert str(seen[0]) == "i + 1"


def test_sibling_selector_function_mutations_are_not_hidden_by_component_read(tmp_path):
    _path, analysis, routine, _nodes = fixture(tmp_path,
        body="field(points(i)%xi,select_index(),1)=1", extra="""integer function select_index()
allocate(hidden(2))
select_index=1
end function""")
    proof = prove_storage_mutation_effects(analysis, routine.qualified)
    assert not proof.preserves_descriptors()
    assert any(row["resource"] == "records::hidden" for row in proof.public()["mutations"])
    assert {row["procedure"] for row in proof.public()["closure"]} == {"records::advance", "records::select_index"}


def test_unknown_record_element_selector_does_not_gain_type_authority(tmp_path):
    _path, analysis, routine, _nodes = fixture(tmp_path, body="field(points(unknown())%xi,1,1)=1")
    proof = prove_storage_mutation_effects(analysis, routine.qualified)
    assert not proof.preserves_descriptors()
    assert proof.public()["frontiers"]


def test_foreign_or_copied_expression_cannot_use_original_selector_position(tmp_path):
    _path, analysis, routine, nodes = fixture(tmp_path)
    with pytest.raises(CompilationError, match="original executable occurrence"):
        read(analysis, routine, copy(nodes[0]))


def test_changed_selected_field_declaration_is_rejected_after_index_callback(tmp_path):
    _path, analysis, routine, nodes = fixture(tmp_path)
    changed = False

    def mutate(_index):
        nonlocal changed
        if not changed:
            binding = analysis.modules["records"].bindings["points"]
            object.__setattr__(binding, "dtype", "real")
            changed = True

    with pytest.raises(CompilationError):
        read(analysis, routine, nodes[0], mutate)
    assert changed


def test_reparenting_selector_to_actual_target_rejects_before_return(tmp_path):
    _path, analysis, routine, nodes = fixture(tmp_path, sealed=False)
    assignment = next(node for node in walk(routine.execution) if type(node).__name__ == "Assignment_Stmt")
    changed = False

    def mutate(_index):
        nonlocal changed
        if not changed:
            assignment.items = (nodes[0], *assignment.items[1:])
            changed = True

    with pytest.raises(CompilationError):
        read(analysis, routine, nodes[0], mutate)
    assert changed


def test_source_publication_and_policy_changes_invalidate_the_existing_fact(tmp_path, monkeypatch):
    import compiler.frontend.storage_component_reads as components

    path, analysis, routine, _nodes = fixture(tmp_path)
    proof = prove_storage_mutation_effects(analysis, routine.qualified)
    assert proof.preserves_descriptors()
    context = analysis.source_authority_context()
    version = components.STORAGE_COMPONENT_READ_POSITION_VERSION
    assert json.loads(context.authority)["storage_component_read_position_version"] == version
    monkeypatch.setattr(components, "STORAGE_COMPONENT_READ_POSITION_VERSION", version + 1)
    changed = SourceEffects([path])
    changed.seal_session()
    with pytest.raises(CompilationError):
        proof.validate(changed, routine.qualified)
    monkeypatch.setattr(components, "STORAGE_COMPONENT_READ_POSITION_VERSION", version)
    path.write_text(path.read_text().replace("points(i)%xi", "points(i)%xi+1"))
    with pytest.raises(CompilationError):
        analysis.verify_publication()


def test_authentic_hierarchical_retirement_emits_and_survives_strict_transport(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    from compiler.scopes.lexical import LexicalBoundaryPatch
    from compiler.tests.test_hierarchical_child_regions import SOURCE
    from compiler.tests.test_hierarchical_child_regions import fixture as hierarchy

    loop = "do i=1,n\nright(points(i)%xi)=sample(points(i),right)\nenddo"
    function = """real(8) function sample(point,values)
type(point_record),intent(in)::point
real(8),intent(inout)::values(:)
sample=values(point%xi)+1.0_8
end function
"""
    source = (SOURCE.replace("implicit none\n", """implicit none
type point_record
integer::xi
real(8),allocatable::unrelated(:)
end type
type(point_record),allocatable::points(:)
""", 1)
        .replace("if(change)call relocate(left,right)", loop)
        .replace("end module", function + "end module"))
    path, builder = hierarchy(tmp_path, source)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    outputs = dict(outputs)
    owner = builder._conditional_return_draft.validate().owner
    middle, = [member for member in owner.members.values() if member.routine.qualified == "hierarchical::middle"]
    patches = [patch for _first, _last, patch in middle.patches
               if isinstance(patch, LexicalBoundaryPatch) and loop in patch.original_source]
    patch, = patches
    assert patch.lifecycle.preserves_descriptors()
    assert patch.lifecycle.public()["frontiers"] == []
    assert patch.validate(middle, require_source=True) is patch
    generated = outputs[report["sources"][str(path)]["replacement"]]
    assert patch.counted_entry is not None
    header, body = loop.split("\n", 1)
    rendered = patch.render(middle)
    assert rendered.count(header + "\n") == 1
    assert body in rendered
    assert rendered in generated
    assert patch.counted_entry.authority["empty_loop_preserves_domain"] is True
    validate_checked_execution_plans(report)
    validate_checked_execution_plans(json.loads(outputs["scope-manifest.json"]))
    assert builder.finish()
