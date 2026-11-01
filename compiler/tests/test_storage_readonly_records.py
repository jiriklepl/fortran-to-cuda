"""Readonly record arguments retain full source-function storage effects."""

import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError


def fixture(
    tmp_path,
    *,
    body="value=interpolate(point,values,-1)",
    inner="",
    outer="",
    declarations="",
    pointer="allocatable",
    intent="in",
    actual="points(i)",
    root_attributes="allocatable,save",
    type_attributes="",
    component="type(sample)",
    caller="values(1,1,1)=evaluate(points(i),values)",
    module_effect="",
    sealed=False,
    transform=lambda text: text,
):
    path = tmp_path / "records.f90"
    path.write_text(f"""module records
implicit none
integer,parameter::rk=8
type {type_attributes} sample
 integer::x,y,z
 real(rk)::weight
end type
type record
 integer::count
 type(sample),{pointer}::samples(:)
end type
type(record),{root_attributes}::points(:)
real(rk),allocatable::hidden(:)
contains
subroutine advance(values,i)
real(rk),allocatable,intent(inout)::values(:,:,:)
integer,intent(in)::i
{caller.replace("points(i)", actual)}
end subroutine
function evaluate(point,values) result(value)
type(record),intent({intent})::point
real(rk),intent(in)::values(-1:,-1:,-1:)
real(rk)::value
{declarations}
{outer}
{body}
end function
function interpolate(point,values,lb) result(value)
type(record),intent(in)::point
integer,intent(in)::lb
real(rk),intent(in)::values(lb:,lb:,lb:)
real(rk)::value
integer::j
value=0
do j=1,point%count
 value=value+point%samples(j)%weight*values(point%samples(j)%x,point%samples(j)%y,point%samples(j)%z)
enddo
{inner}
end function
{module_effect}
end module
""")
    path.write_text(transform(path.read_text()))
    source = SourceEffects([path])
    if sealed:
        source.seal_session()
    routine = source.routines["records::advance"]
    expression = next(
        node
        for node in walk(routine.execution)
        if type(node).__name__ in {"Part_Ref", "Function_Reference"} and str(node.items[0]).lower() == "evaluate"
    )
    return path, source, expression


@pytest.mark.parametrize("sealed", [False, True])
def test_original_scalar_element_to_readonly_record_has_complete_function_closure(tmp_path, sealed):
    _path, source, expression = fixture(tmp_path, sealed=sealed)
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    public = proof.public()
    assert public["frontiers"] == []
    assert public["mutations"] == []
    assert public["escapes"] == []
    assert proof.preserves_descriptors()
    assert {row["procedure"] for row in public["closure"]} == {"records::evaluate", "records::interpolate"}
    assert public["version"] == 14
    assert len(public["readonly_record_associations"]) == 2
    assert public["readonly_record_associations"][0]["actual"] == "points(i)"
    for name in ("payload_access_authorized", "entry_definition_authority", "execution_authorized", "alias_authority"):
        assert public[name] is False


@pytest.mark.parametrize("mutation", ["deallocate(hidden)", "call unknown(point)", "allocate(hidden(2))"])
def test_hidden_mutation_and_unknown_calls_are_retained(tmp_path, mutation):
    _path, source, expression = fixture(tmp_path, inner=mutation)
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert not proof.preserves_descriptors()
    public = proof.public()
    assert any(
        mutation.lower() in event["original_source"].lower()
        for bucket in ("mutations", "escapes", "frontiers")
        for event in public[bucket]
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("type(record),intent(in)::point", "type(record),intent(inout)::point", 1),
        lambda text: text.replace("type(record),intent(in)::point", "type(record),target,intent(in)::point", 1),
        lambda text: text.replace("type(record),intent(in)::point", "class(record),intent(in)::point", 1),
        lambda text: text.replace("type(sample),allocatable::samples(:)", "type(sample),pointer::samples(:)"),
        lambda text: text.replace("type  sample", "type,bind(c)::sample"),
        lambda text: text.replace("type  sample", "type sample\nsequence"),
        lambda text: text.replace("points(i),values", "points(:),values"),
        lambda text: text.replace("points(i),values", "points([i]),values"),
        lambda text: text.replace(
            "value=interpolate(point,values,-1)", "value=interpolate(point,values,-1)\ncall unknown(point)"
        ),
        lambda text: text.replace(
            "value=interpolate(point,values,-1)", "value=interpolate(point,values,-1)\npoint%count=0"
        ),
    ],
)
def test_association_lifecycle_and_alias_frontiers_remain_explicit(tmp_path, change):
    _path, source, expression = fixture(tmp_path, transform=change)
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert not proof.preserves_descriptors()
    assert proof.public()["frontiers"] or proof.public()["mutations"] or proof.public()["escapes"]


@pytest.mark.parametrize("call", ["call change(values)", "call capture(point)"])
def test_complete_source_callees_retain_mutations_and_pointer_exports(tmp_path, call):
    helpers = """subroutine change(a)
real(8),allocatable,intent(inout)::a(:,:,:)
deallocate(a)
end subroutine
subroutine capture(point)
type(record),intent(in),target::point
type(record),pointer::saved
saved=>point
end subroutine"""
    _path, source, expression = fixture(tmp_path, inner=call, module_effect=helpers)
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert not proof.preserves_descriptors()
    assert proof.public()["mutations"] or proof.public()["escapes"]


def test_untypable_original_index_retains_known_function_mutation(tmp_path):
    def change(text):
        return text.replace("samples(j)%weight", "samples(select_index())%weight")

    _path, source, expression = fixture(
        tmp_path,
        transform=change,
        module_effect="""integer function select_index()
deallocate(hidden)
select_index=1
end function""",
    )
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    public = proof.public()
    assert not proof.preserves_descriptors()
    assert any(event["resource"] == "records::hidden" for event in public["mutations"])
    assert any(row["procedure"] == "records::select_index" for row in public["closure"])
    assert any("source expression authority: select_index()" in event["kind"] for event in public["frontiers"])


@pytest.mark.parametrize("selected", ["actual", "formal", "schema", "schema_map", "member", "index", "kind"])
@pytest.mark.parametrize("sealed", [False, True])
def test_final_publication_rejects_consumed_declaration_schema_and_index_changes(
    tmp_path, monkeypatch, selected, sealed
):
    import compiler.frontend.storage_component_reads as reads

    _path, source, expression = fixture(tmp_path, sealed=sealed)
    original = reads.readonly_record_declaration
    consumed = False
    changed = False

    def record(*args, **kwargs):
        nonlocal consumed
        result = original(*args, **kwargs)
        consumed = True
        return result

    monkeypatch.setattr(reads, "readonly_record_declaration", record)
    old_verify = source.verify_publication

    def verify():
        nonlocal changed
        result = old_verify()
        if consumed and not changed:
            changed = True
            owner = source.modules["records"]
            schema = owner.component_types["record"]
            if selected == "actual":
                object.__setattr__(owner.bindings["points"], "root", "forged::points")
            elif selected == "formal":
                object.__setattr__(
                    source.numerical_helpers["records::interpolate"].scope.bindings["point"], "intent", "inout"
                )
            elif selected == "schema":
                object.__setattr__(schema, "members", dict(schema.members))
            elif selected == "schema_map":
                object.__setattr__(owner, "component_types", dict(owner.component_types))
            elif selected == "member":
                altered = dict(schema.members)
                altered["count"] = tuple(list(altered["count"]))
                object.__setattr__(schema, "members", altered)
            elif selected == "index":
                index = next(
                    node
                    for node in walk(source.numerical_helpers["records::interpolate"].execution)
                    if type(node).__name__ == "Name" and str(node) == "j"
                )
                object.__setattr__(index, "string", "changed_index")
            else:
                parameter = owner.kinds.values["rk"]
                object.__setattr__(parameter, "items", ("4", None))
        return result

    monkeypatch.setattr(source, "verify_publication", verify)
    with pytest.raises(CompilationError):
        prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert consumed
    assert changed


@pytest.mark.parametrize(("limit", "expected"), [(64, True), (8, False)])
def test_real_unique_callee_budget_is_still_enforced(tmp_path, limit, expected):
    helpers = "\n".join(
        f"subroutine level{i}()\n" + (f"call level{i + 1}()\n" if i < 9 else "") + "end subroutine"
        for i in range(1, 10)
    )
    _path, source, expression = fixture(tmp_path, inner="call level1()", module_effect=helpers)
    source.procedure_limit = limit
    source.depth_limit = 16
    if expected:
        proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
        assert proof.preserves_descriptors()
        assert len(proof.public()["closure"]) == 11
    else:
        with pytest.raises(CompilationError, match="bounded call closure"):
            prove_storage_mutation_effects(source, "records::advance", expression=expression)


def test_source_publication_rejects_changed_original_file_after_fact_reuse(tmp_path):
    path, source, expression = fixture(tmp_path, sealed=True)
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert proof.preserves_descriptors()
    path.write_text(path.read_text().replace("value=0", "value=1"))
    assert proof.validate() is proof
    with pytest.raises(CompilationError):
        source.verify_publication()


def test_component_deallocation_retains_existing_hard_storage_frontier(tmp_path):
    from compiler.frontend.component_bindings import ComponentStorageUnsupported

    _path, source, expression = fixture(tmp_path, outer="deallocate(point%samples)")
    with pytest.raises(ComponentStorageUnsupported, match="association or lifetime is uncertain: samples"):
        prove_storage_mutation_effects(source, "records::advance", expression=expression)


@pytest.mark.parametrize("sealed", [False, True])
def test_existing_descriptor_window_consumes_readonly_function_storage_projection(tmp_path, sealed):
    from compiler.frontend.descriptor_windows import prove_descriptor_window

    _path, source, _expression = fixture(tmp_path, sealed=sealed)
    routine = source.routines["records::advance"]
    graph = source.structure(routine.qualified)
    selected = tuple(graph.node_id(node) for node in routine.execution.content)
    proof = prove_descriptor_window(source, routine.qualified, selected, ("argument::values",))
    public = proof.public()
    assert public["native_storage_function_requirements"]
    assert public["native_storage_function_requirements"][0]["version"] == 14
    assert public["execution_authorized"] is False


def imported_sample(text):
    return text.replace("module records", "module record_types", 1).replace(
        "end type\ntype record",
        "end type\nend module\nmodule records\nuse record_types,only:sample,rk\nimplicit none\ntype record",
        1,
    )


@pytest.mark.parametrize("sealed", [False, True])
def test_imported_component_schema_and_kind_use_the_existing_source_receipts(tmp_path, sealed):
    _path, source, expression = fixture(tmp_path, sealed=sealed, transform=imported_sample)
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert proof.preserves_descriptors()
    assert "records::points%samples%weight" in proof.public()["resources"]


@pytest.mark.parametrize("selected", ["kind", "component"])
@pytest.mark.parametrize("sealed", [False, True])
def test_imported_component_dependencies_cannot_change_at_publication(tmp_path, monkeypatch, selected, sealed):
    import compiler.frontend.storage_component_reads as reads

    _path, source, expression = fixture(tmp_path, sealed=sealed, transform=imported_sample)
    consumed = False
    changed = False
    original = reads.readonly_record_declaration

    def declaration(*args, **kwargs):
        nonlocal consumed
        value = original(*args, **kwargs)
        consumed = True
        return value

    monkeypatch.setattr(reads, "readonly_record_declaration", declaration)
    old = source.verify_publication

    def verify():
        nonlocal changed
        result = old()
        if consumed and not changed:
            changed = True
            owner = source.modules["record_types"]
            if selected == "kind":
                object.__setattr__(owner.kinds.values["rk"], "items", ("4", None))
            else:
                schema = owner.component_types["sample"]
                object.__setattr__(schema, "members", dict(schema.members))
        return result

    monkeypatch.setattr(source, "verify_publication", verify)
    with pytest.raises(CompilationError):
        prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert consumed
    assert changed


@pytest.mark.parametrize("sealed", [False, True])
def test_changed_original_kind_before_first_record_inspection_does_not_seed_a_new_baseline(
    tmp_path, monkeypatch, sealed
):
    import compiler.frontend.storage_mutation_effects as mutation

    _path, source, expression = fixture(tmp_path, sealed=sealed, transform=imported_sample)
    original = mutation._scope_roles
    changed = False

    def roles(analysis):
        nonlocal changed
        result = original(analysis)
        if not changed:
            changed = True
            object.__setattr__(source.modules["record_types"].kinds.values["rk"], "items", ("4", None))
        return result

    monkeypatch.setattr(mutation, "_scope_roles", roles)
    # Mutable original-call receipts reject before the new classifier. The
    # sealed diagnostic path can instead retain an explicit unsupported record;
    # neither path may report preserved descriptors from the changed kind.
    if sealed:
        proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
        assert not proof.preserves_descriptors()
        assert any(
            "storage record original declaring namespace changed" in event["kind"]
            for event in proof.public()["frontiers"]
        )
    else:
        with pytest.raises(CompilationError, match="source|namespace|original|kind|declaration"):
            prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert changed


@pytest.mark.parametrize("selected", ["fields", "registry"])
@pytest.mark.parametrize("sealed", [False, True])
def test_window_cannot_publish_after_consumed_record_call_attachment_replacement(
    tmp_path, monkeypatch, sealed, selected
):
    from compiler.frontend.descriptor_windows import prove_descriptor_window
    from compiler.frontend.storage_mutation_effects import StorageMutationEffects

    _path, source, _expression = fixture(tmp_path, sealed=sealed)
    consumed = False
    changed = False
    resolved = None
    original = StorageMutationEffects.descriptor_actuals

    def actuals(proof, routine, node):
        nonlocal consumed, resolved
        result = original(proof, routine, node)
        resolved = next(call for call in proof._calls if call._caller is routine and call._original is node)
        consumed = True
        return result

    monkeypatch.setattr(StorageMutationEffects, "descriptor_actuals", actuals)
    from compiler.frontend.source_inputs import SourceInputs

    old = SourceInputs.verify_live

    def verify(inputs):
        nonlocal changed
        result = old(inputs)
        if consumed and not changed:
            changed = True
            if selected == "fields":
                object.__setattr__(resolved, "associations", tuple(list(resolved.associations)))
            else:
                owner = resolved._analysis
                object.__setattr__(owner, "_original_call_associations", dict(owner._original_call_associations))
        return result

    monkeypatch.setattr(SourceInputs, "verify_live", verify)
    routine = source.routines["records::advance"]
    graph = source.structure(routine.qualified)
    selected_ids = tuple(graph.node_id(node) for node in routine.execution.content)
    with pytest.raises(CompilationError, match="attachment|association|source|consumed|registry"):
        prove_descriptor_window(source, routine.qualified, selected_ids, ("argument::values",))
    assert consumed
    assert changed


def test_complete_nonpure_numeric_hidden_writes_stay_independent_of_descriptor_authority(tmp_path):
    def transform(text):
        return text.replace("real(rk),allocatable::hidden(:)", "real(rk),allocatable::hidden(:)\ninteger::counter")

    _path, source, expression = fixture(tmp_path, inner="counter=counter+1", transform=transform)
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert proof.preserves_descriptors()
    assert "records::counter" in proof.public()["resources"]
    assert proof.public()["payload_access_authorized"] is False
    assert proof.public()["execution_authorized"] is False


def test_transient_consumed_imported_kind_is_rejected_even_when_restored_before_final_comparison(tmp_path):
    from compiler.frontend.storage_component_reads import numeric_component_read

    _path, source, _expression = fixture(tmp_path, sealed=True, transform=imported_sample)
    routine = source.numerical_helpers["records::interpolate"]
    node = next(
        node
        for node in walk(routine.execution)
        if type(node).__name__ == "Data_Ref" and str(node).lower().replace(" ", "").endswith("%weight")
    )
    parameter = source.modules["record_types"].kinds.values["rk"]
    original_items = parameter.items
    original_string = parameter.string
    changed = False
    restored = False

    def index(_index):
        nonlocal changed
        if not changed:
            object.__setattr__(parameter, "items", ("4", None))
            object.__setattr__(parameter, "string", "4")
            changed = True

    def scope(_scope):
        nonlocal restored
        if changed and not restored:
            object.__setattr__(parameter, "items", original_items)
            object.__setattr__(parameter, "string", original_string)
            restored = True

    with pytest.raises(CompilationError, match="declaration authority changed during traversal"):
        numeric_component_read(
            source,
            routine,
            node,
            readonly_record=routine.scope.bindings["point"],
            visit_index=index,
            visit_scope=scope,
            tick=lambda: None,
        )
    assert changed
    assert restored
    assert parameter.items is original_items


@pytest.mark.parametrize("sealed", [False, True])
def test_original_structural_specification_check_does_not_write_the_session_cache(tmp_path, sealed):
    from copy import copy

    _path, source, _expression = fixture(tmp_path, sealed=sealed)
    snapshot = source._source_syntax
    specification = next(
        node for node in source.modules["records"].node.content if type(node).__name__ == "Specification_Part"
    )
    cache = snapshot._layouts
    before = tuple(cache.items())
    assert snapshot.original_subtree_unchanged(specification)
    assert not snapshot.original_subtree_unchanged(copy(specification))
    parameter = source.modules["records"].kinds.values["rk"]
    old = parameter.items
    object.__setattr__(parameter, "items", tuple(list(old)))
    assert not snapshot.original_subtree_unchanged(specification)
    object.__setattr__(parameter, "items", old)
    assert snapshot.original_subtree_unchanged(specification)
    assert snapshot._layouts is cache
    assert tuple(cache.items()) == before
