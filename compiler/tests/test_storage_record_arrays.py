"""Original record-array metadata and element reads carry no payload authority."""

import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import _original, prove_storage_mutation_effects
from compiler.ir import CompilationError


def fixture(tmp_path, *, body="value=size(points)+points(i)%eps+fraction(points(i))",
            attributes="allocatable", component="real(8)::eps=0", extra="", sealed=True,
            transform=lambda source: source):
    path = tmp_path / "arrays.f90"
    path.write_text(f"""module records
implicit none
type record
 {component}
end type
type(record),{attributes}::points(:)
real(8),allocatable::hidden(:)
logical::change
contains
subroutine advance(i,out)
integer,intent(in)::i
real(8),intent(out)::out
out=evaluate(i)
end subroutine
function evaluate(i) result(value)
integer,intent(in)::i
real(8)::value
{body}
end function
function fraction(point) result(value)
type(record),intent(in)::point
real(8)::value
value=point%eps
end function
{extra}
end module
""")
    path.write_text(transform(path.read_text()))
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines["records::advance"]
    expression = next(node for node in walk(routine.execution)
                      if type(node).__name__ in {"Part_Ref", "Function_Reference"}
                      and str(node.items[0]).lower() == "evaluate")
    return path, analysis, expression


def project(analysis, expression):
    return prove_storage_mutation_effects(analysis, "records::advance", expression=expression)


@pytest.mark.parametrize("sealed", [False, True])
def test_record_array_inquiry_and_element_read_retain_complete_function_closure(tmp_path, sealed):
    _path, analysis, expression = fixture(tmp_path, sealed=sealed)
    proof = project(analysis, expression)
    public = proof.public()
    assert proof.preserves_descriptors()
    assert public["complete"]
    assert public["frontiers"] == public["mutations"] == public["escapes"] == []
    assert {row["procedure"] for row in public["closure"]} == {"records::evaluate", "records::fraction"}
    assert public["version"] == 18
    reads = public["original_record_array_reads"]
    assert len(reads) == 2
    assert {row["kind"] for row in reads} == {"original_record_array_inquiry", "original_numeric_component_read"}
    assert reads[0]["resource"] == "records::points"
    assert reads[1]["resource"] == "records::points%eps"
    for row in reads:
        assert row["payload_access_authorized"] is False
        assert row["allocation_or_definition_inferred"] is False
        assert row["execution_authorized"] is False
        assert row["alias_authority"] is False


def test_array_element_indices_are_exact_original_nodes_visited_by_the_ordinary_scanner(tmp_path):
    from compiler.frontend.storage_component_reads import numeric_component_read
    from compiler.frontend.storage_mutation_effects import _original

    _path, analysis, _expression = fixture(tmp_path, body="value=points(i)%eps")
    routine = _original(analysis, "records::evaluate")
    node = next(node for node in walk(routine.execution) if type(node).__name__ == "Data_Ref")
    indices = []
    result = numeric_component_read(analysis, routine, node, visit_index=indices.append,
                                    visit_scope=lambda _scope: None, tick=lambda: None)
    assert tuple(indices) == result.indices == (node.items[0].items[1].items[0],)
    assert result.public()["version"] == 3


def test_multidimensional_original_scalar_record_element_is_admitted(tmp_path):
    _path, analysis, expression = fixture(tmp_path, body="value=size(points,2)+points(i,1)%eps",
        transform=lambda text: text.replace("points(:)", "points(:,:)"))
    assert project(analysis, expression).preserves_descriptors()


@pytest.mark.parametrize("inquiry", ["size(points)", "size(array=points,dim=1,kind=8)",
                                    "lbound(points,1)", "ubound(points,dim=1)", "sum(shape(points))"])
def test_exact_intrinsic_metadata_inquiries_do_not_classify_whole_record_values(tmp_path, inquiry):
    _path, analysis, expression = fixture(tmp_path, body=f"value={inquiry}")
    assert project(analysis, expression).preserves_descriptors()


@pytest.mark.parametrize("body", ["value=points%eps", "value=sum(points(:)%eps)",
                                 "value=points([i])%eps", "value=points(1:2)%eps",
                                 "value=size(points(:))", "value=size(points,dim=1.0)"])
def test_sections_vectors_whole_record_arrays_and_nonscalar_integer_selectors_reject(tmp_path, body):
    _path, analysis, expression = fixture(tmp_path, body=body)
    proof = project(analysis, expression)
    assert not proof.preserves_descriptors()
    assert proof.public()["frontiers"]


@pytest.mark.parametrize("attributes", ["pointer", "target,allocatable", "volatile,allocatable",
                                       "asynchronous,allocatable"])
def test_aliasable_record_array_roots_do_not_borrow_read_capability(tmp_path, attributes):
    _path, analysis, expression = fixture(tmp_path, attributes=attributes)
    assert not project(analysis, expression).preserves_descriptors()


def test_optional_record_array_dummy_requires_separate_presence_authority(tmp_path):
    def optional(text):
        return (text.replace("type(record),allocatable::points(:)", "")
                .replace("subroutine advance(i,out)", "subroutine advance(i,out,points)\ntype(record),intent(in)::points(:)")
                .replace("out=evaluate(i)", "out=evaluate(points,i)")
                .replace("function evaluate(i)", "function evaluate(points,i)")
                .replace("function evaluate(points,i) result(value)",
                         "function evaluate(points,i) result(value)\ntype(record),optional,intent(in)::points(:)"))

    _path, analysis, expression = fixture(tmp_path, transform=optional)
    assert not project(analysis, expression).preserves_descriptors()


@pytest.mark.parametrize(("component", "body", "preserves"), [
    ("real(8),pointer::eps", "value=size(points)", False),
    ("real(8),pointer::eps", "value=points(i)%eps", False),
    ("real(8),allocatable::eps", "value=size(points)", True),
    ("real(8),allocatable::eps", "value=points(i)%eps", False),
    ("real(8)::eps\nreal(8),allocatable::unused(:)", "value=size(points)", True),
    ("real(8)::eps\nreal(8),allocatable::unused(:)", "value=points(i)%eps", True),
])
def test_array_read_checks_the_selected_field_without_borrowing_component_ownership(
        tmp_path, component, body, preserves):
    _path, analysis, expression = fixture(tmp_path, component=component, body=body)
    proof = project(analysis, expression)
    assert proof.preserves_descriptors() is preserves
    if preserves:
        assert proof.public()["mutations"] == proof.public()["escapes"] == []
        for row in proof.public()["original_record_array_reads"]:
            assert row["payload_access_authorized"] is False
            assert row["allocation_or_definition_inferred"] is False


INDEX = """function select_index() result(j)
integer::j
allocate(hidden(2))
j=1
end function"""


@pytest.mark.parametrize("body", ["value=points(select_index())%eps", "value=size(points,select_index())"])
def test_original_selector_effects_are_scanned_before_read_classification(tmp_path, body):
    _path, analysis, expression = fixture(tmp_path, body=body, extra=INDEX)
    proof = project(analysis, expression)
    assert not proof.preserves_descriptors()
    assert any(row["resource"] == "records::hidden" for row in proof.public()["mutations"])


@pytest.mark.parametrize("body", ["value=points(unknown())%eps", "value=size(points,unknown())"])
def test_unknown_selectors_remain_frontiers(tmp_path, body):
    _path, analysis, expression = fixture(tmp_path, body=body)
    proof = project(analysis, expression)
    assert not proof.preserves_descriptors()
    assert proof.public()["frontiers"]


@pytest.mark.parametrize("inquiry", ["size(array=points,select_index())",
                                     "size(points,array=points,dim=select_index())",
                                     "size(points,unknown=select_index())"])
def test_invalid_intrinsic_argument_order_or_keywords_fall_back_with_selector_effects(tmp_path, inquiry):
    _path, analysis, expression = fixture(tmp_path, body=f"value={inquiry}", extra=INDEX)
    proof = project(analysis, expression)
    assert not proof.preserves_descriptors()
    assert proof.public()["frontiers"]
    assert any(row["resource"] == "records::hidden" for row in proof.public()["mutations"])


@pytest.mark.parametrize("mutation", ["allocate(hidden(2))", "deallocate(points)",
                                     "call mutate()"])
def test_sibling_branch_mutation_is_not_removed_by_an_admitted_read(tmp_path, mutation):
    extra = "subroutine mutate()\nallocate(hidden(2))\nend subroutine"
    _path, analysis, expression = fixture(tmp_path,
        body=f"if(change) then\n{mutation}\nendif\nvalue=size(points)+points(i)%eps", extra=extra)
    proof = project(analysis, expression)
    assert not proof.preserves_descriptors()
    assert proof.public()["mutations"]


def test_source_shadowed_size_uses_original_function_effects(tmp_path):
    extra = """function size(array) result(n)
type(record),intent(in)::array(:)
integer::n
allocate(hidden(2))
n=1
end function"""
    _path, analysis, expression = fixture(tmp_path, body="value=size(points)", extra=extra)
    proof = project(analysis, expression)
    assert not proof.preserves_descriptors()
    assert any(row["resource"] == "records::hidden" for row in proof.public()["mutations"])


def test_record_array_fact_keeps_original_context_and_changed_file_publication_guard(tmp_path):
    path, analysis, expression = fixture(tmp_path)
    proof = project(analysis, expression)
    assert proof.preserves_descriptors()
    foreign = SourceEffects([path])
    foreign.seal_session()
    with pytest.raises(CompilationError):
        proof.validate(foreign, "records::advance", expression)
    public = proof.public()
    public["original_record_array_reads"].clear()
    assert len(proof.public()["original_record_array_reads"]) == 2
    path.write_text(path.read_text().replace("value=size(points)", "value=2+size(points)"))
    with pytest.raises(CompilationError):
        analysis.verify_publication()


def test_record_array_capability_version_is_part_of_issued_source_authority(tmp_path, monkeypatch):
    import json

    import compiler.frontend.storage_mutation_effects as storage

    path, analysis, expression = fixture(tmp_path)
    proof = project(analysis, expression)
    assert proof.preserves_descriptors()
    original = analysis.source_authority_context()
    assert json.loads(original.authority)["storage_record_array_read_version"] == 18
    monkeypatch.setattr(storage, "STORAGE_RECORD_ARRAY_READ_VERSION", storage.STORAGE_RECORD_ARRAY_READ_VERSION + 1)
    changed = SourceEffects([path])
    changed.seal_session()
    context = changed.source_authority_context()
    assert json.loads(context.authority)["storage_record_array_read_version"] == 19
    assert original._intern_key != context._intern_key
    with pytest.raises(CompilationError):
        proof.validate(changed, "records::advance", expression)


OWNED_COMPONENT = "real(8)::eps=0\nreal(8),allocatable::unused(:)"


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("body", ["value=size(points)", "value=lbound(points,1)",
                                 "value=ubound(points,1)", "value=sum(shape(points))",
                                 "value=points(i)%eps"])
def test_unrelated_allocatable_field_does_not_turn_original_read_into_storage_mutation(tmp_path, body, sealed):
    _path, analysis, expression = fixture(tmp_path, body=body, component=OWNED_COMPONENT, sealed=sealed)
    proof = project(analysis, expression)
    assert proof.preserves_descriptors()
    public = proof.public()
    assert public["complete"]
    assert public["frontiers"] == public["mutations"] == public["escapes"] == []


@pytest.mark.parametrize("body", ["points(i)=points(i)\nvalue=0", "points(i)%eps=1\nvalue=0"])
def test_read_projection_does_not_authorize_aggregate_or_component_mutation(tmp_path, body):
    _path, analysis, expression = fixture(tmp_path, body=body, component=OWNED_COMPONENT)
    proof = project(analysis, expression)
    assert not proof.preserves_descriptors()
    assert proof.public()["frontiers"] or proof.public()["mutations"]


@pytest.mark.parametrize("body", ["call take(points(i)%eps)\nvalue=0", "points(i)%eps=1\nvalue=0"])
def test_array_component_read_helper_cannot_authorize_call_actual_or_assignment_target(tmp_path, body):
    from compiler.frontend.storage_component_reads import numeric_component_read

    _path, analysis, _expression = fixture(tmp_path, body=body, component=OWNED_COMPONENT)
    routine = _original(analysis, "records::evaluate")
    node = next(node for node in walk(routine.execution) if type(node).__name__ == "Data_Ref")
    with pytest.raises(CompilationError, match="writes|address"):
        numeric_component_read(analysis, routine, node, visit_index=lambda _node: None,
                               visit_scope=lambda _scope: None, tick=lambda: None)


@pytest.mark.parametrize(("attributes", "preserves"), [("", True), (",allocatable", False)])
def test_selected_nested_allocatable_object_keeps_its_ownership_barrier(tmp_path, attributes, preserves):
    def nested(text):
        return text.replace("type record\n", "type nested_record\nreal(8)::value\nend type\ntype record\n")

    _path, analysis, expression = fixture(tmp_path, body="value=points(i)%child%value",
        component=f"type(nested_record){attributes}::child\n{OWNED_COMPONENT}", transform=nested)
    assert project(analysis, expression).preserves_descriptors() is preserves


@pytest.mark.parametrize("body", ["value=size(points)", "value=points(i)%eps"])
def test_unrelated_allocation_does_not_bypass_original_finalizer_barrier(tmp_path, body):
    def finalizer(text):
        return text.replace("real(8),allocatable::unused(:)\nend type",
                            "real(8),allocatable::unused(:)\ncontains\nfinal::finish\nend type")

    _path, analysis, expression = fixture(tmp_path, body=body, component=OWNED_COMPONENT,
        transform=finalizer, extra="subroutine finish(point)\ntype(record)::point\nend subroutine")
    proof = project(analysis, expression)
    assert not proof.preserves_descriptors()
    assert any("finalization" in row["kind"] for row in proof.public()["frontiers"])


@pytest.mark.parametrize("body", ["value=points(select_index())%eps", "value=size(points,select_index())"])
def test_unrelated_owned_component_does_not_hide_selector_mutation(tmp_path, body):
    _path, analysis, expression = fixture(tmp_path, body=body, component=OWNED_COMPONENT, extra=INDEX)
    proof = project(analysis, expression)
    assert not proof.preserves_descriptors()
    assert any(row["resource"] == "records::hidden" for row in proof.public()["mutations"])


@pytest.mark.parametrize("mutation", ["schema", "field"])
def test_owned_component_read_rejects_replaced_original_schema_or_selected_field(tmp_path, mutation):
    from copy import copy

    from compiler.frontend.storage_component_reads import numeric_component_read

    _path, analysis, _expression = fixture(tmp_path, body="value=points(i)%eps", component=OWNED_COMPONENT,
                                         sealed=False)
    routine = _original(analysis, "records::evaluate")
    node = next(node for node in walk(routine.execution) if type(node).__name__ == "Data_Ref")
    owner = analysis.modules["records"]
    schema = owner.component_types["record"]
    if mutation == "schema":
        owner.component_types["record"] = copy(schema)
    else:
        dtype, attributes, entity = schema.members["eps"]
        schema.members["eps"] = (dtype, attributes, copy(entity))
    with pytest.raises(CompilationError, match="original|schema|component"):
        numeric_component_read(analysis, routine, node, visit_index=lambda _node: None,
                               visit_scope=lambda _scope: None, tick=lambda: None)


@pytest.mark.parametrize("body", ["allocate(points(i)%unused(2))\nvalue=0",
                                 "deallocate(points(i)%unused)\nvalue=0"])
def test_array_component_allocation_keeps_existing_hard_unsupported_boundary(tmp_path, body):
    from compiler.frontend.component_bindings import ComponentAccessUnsupported

    _path, analysis, expression = fixture(tmp_path, body=body, component=OWNED_COMPONENT)
    with pytest.raises(ComponentAccessUnsupported, match="one scalar variable object"):
        project(analysis, expression)
