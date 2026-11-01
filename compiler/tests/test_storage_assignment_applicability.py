"""Visible unrelated record assignments do not replace primitive assignment."""

import json

import pytest

from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.tests.test_storage_readonly_records import fixture as record_fixture

GENERICS = """interface assignment(=)
module procedure record_assignment, other_record_assignment
end interface
"""
SPECIFICS = """subroutine record_assignment(left,right)
type(record),intent(out)::left
type(record),intent(in)::right
deallocate(hidden)
end subroutine
subroutine other_record_assignment(left,right)
type(sample),intent(out)::left
type(record),intent(in)::right
allocate(hidden(2))
end subroutine
"""


def fixture(tmp_path, *, sealed=False, transform=lambda text: text):
    def generics(text):
        text = text.replace("contains\nsubroutine advance", GENERICS + "contains\nsubroutine advance", 1)
        return transform(text)
    return record_fixture(tmp_path, sealed=sealed, transform=generics, module_effect=SPECIFICS)


@pytest.mark.parametrize("sealed", [False, True])
def test_real_function_assignments_exclude_concrete_record_lhs_specifics(tmp_path, sealed):
    _path, source, expression = fixture(tmp_path, sealed=sealed)
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    public = proof.public()
    (tmp_path / "complete-storage-record.json").write_text(json.dumps(public, sort_keys=True, indent=2) + "\n")
    assert proof.preserves_descriptors(), public["frontiers"]
    assert public["mutations"] == public["escapes"] == public["frontiers"] == []
    assert {row["procedure"] for row in public["closure"]} == {"records::evaluate", "records::interpolate"}
    assert public["version"] == 15
    rows = public["original_inapplicable_assignments"]
    assert len(rows) == 3
    assert all(row["actual_lhs_type"] == "real" for row in rows)
    assert all({item["procedure"] for item in row["excluded_specifics"]} ==
               {"records::record_assignment", "records::other_record_assignment"} for row in rows)
    assert not public["payload_access_authorized"]
    assert not public["execution_authorized"]


@pytest.mark.parametrize("formal", ["real(8)", "class(record)", "class(*)"])
def test_matching_or_uncertain_lhs_signature_stays_a_frontier(tmp_path, formal):
    def change(text):
        return text.replace("type(record),intent(out)::left", formal + ",intent(out)::left", 1)
    _path, source, expression = fixture(tmp_path, transform=change)
    public = prove_storage_mutation_effects(source, "records::advance", expression=expression).public()
    assert public["frontiers"]
    assert any("defined assignment" in row["kind"] for row in public["frontiers"])
    assert "original_inapplicable_assignments" not in public


@pytest.mark.parametrize("route", ["direct", "reexport"])
def test_unavailable_namespace_cannot_complete_exclusions(tmp_path, route):
    def change(text):
        if route == "direct":
            return text.replace("module records\n", "module records\nuse unknown,only:assignment(=)\n", 1)
        return ("module imported\nuse unknown,only:assignment(=)\nend module\n" +
                text.replace("module records\n", "module records\nuse imported\n", 1))
    _path, source, expression = fixture(tmp_path, transform=change)
    public = prove_storage_mutation_effects(source, "records::advance", expression=expression).public()
    assert any("closed-name namespace is unavailable" in row["kind"] for row in public["frontiers"])
    assert "original_inapplicable_assignments" not in public


IMPORTED = """module imported
{access}
interface assignment(=)
module procedure assign_logical
end interface
contains
subroutine assign_logical(left,right)
real(8),intent(out)::left
logical,intent(in)::right
left=1
end subroutine
end module
"""


@pytest.mark.parametrize("private", [False, True])
def test_all_exported_host_and_use_generics_are_considered(tmp_path, private):
    def change(text):
        return IMPORTED.format(access="private" if private else "") + text.replace(
            "module records\n", "module records\nuse imported\n", 1
        )
    _path, source, expression = fixture(tmp_path, sealed=True, transform=change)
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert proof.preserves_descriptors() is private
    assert bool(proof.public()["frontiers"]) is not private


@pytest.mark.parametrize("sealed", [False, True])
def test_rhs_and_original_function_bodies_keep_hidden_storage_mutations(tmp_path, sealed):
    def change(text):
        return text.replace("value=0\n", "deallocate(hidden)\nvalue=0\n", 1)
    _path, source, expression = fixture(tmp_path, sealed=sealed, transform=change)
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    public = proof.public()
    assert not proof.preserves_descriptors()
    assert {row["resource"] for row in public["mutations"]} == {"records::hidden"}
    assert public["frontiers"] == []
    assert public["original_inapplicable_assignments"]
    assert any(row["procedure"] == "records::interpolate" for row in public["closure"])


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("fault", ["formal", "generic", "schema", "membership"])
def test_final_publication_retains_specific_declaration_and_namespace(tmp_path, monkeypatch, sealed, fault):
    from compiler.frontend.storage_mutation_effects import _Scan
    from compiler.ir import CompilationError

    _path, source, expression = fixture(tmp_path, sealed=sealed)
    old_classify = _Scan.assignment_inapplicable
    consumed = changed = False

    def classify(*args, **kwargs):
        nonlocal consumed
        result = old_classify(*args, **kwargs)
        consumed |= result
        return result

    monkeypatch.setattr(_Scan, "assignment_inapplicable", classify)
    old_verify = source.verify_publication

    def verify():
        nonlocal changed
        result = old_verify()
        if consumed and not changed:
            changed = True
            owner = source.modules["records"]
            specific = source.routines["records::record_assignment"]
            if fault == "formal":
                object.__setattr__(specific.scope.bindings["left"], "dtype", "real")
            elif fault == "generic":
                object.__setattr__(owner, "generics", {"assignment(=)": [("records", "missing")]})
            elif fault == "schema":
                schema = owner.component_types["record"]
                object.__setattr__(schema, "members", dict(schema.members))
            else:
                part = specific.scope.node.parent
                object.__setattr__(part, "content", tuple(item for item in part.content if item is not specific.scope.node))
        return result

    monkeypatch.setattr(source, "verify_publication", verify)
    with pytest.raises(CompilationError):
        prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert consumed
    assert changed


def test_original_file_change_cannot_reuse_assignment_exclusions(tmp_path):
    from compiler.ir import CompilationError

    path, source, expression = fixture(tmp_path, sealed=True)
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert proof.preserves_descriptors()
    path.write_text(path.read_text() + "\n!changed original source\n")
    assert proof.validate() is proof
    with pytest.raises(CompilationError):
        source.verify_publication()


def test_actual_matching_numeric_from_record_assignment_stays_unavailable(tmp_path):
    def change(text):
        return text.replace("type(record),intent(out)::left", "real(8),intent(out)::left", 1).replace(
            "value=interpolate(point,values,-1)", "value=point", 1
        )
    _path, source, expression = fixture(tmp_path, transform=change)
    public = prove_storage_mutation_effects(source, "records::advance", expression=expression).public()
    assert any("defined assignment" in row["kind"] for row in public["frontiers"])
    assert any(row["original_source"] == "value = point" for row in public["frontiers"])


@pytest.mark.parametrize("sealed", [False])
@pytest.mark.parametrize("target", ["actual", "specific"])
def test_transient_consumed_types_cannot_be_restored_into_an_accepted_proof(tmp_path, monkeypatch, sealed, target):
    from compiler.frontend.call_bindings import _OriginalTypes
    from compiler.frontend.source_objects import _TypeBuilder

    _path, source, expression = fixture(tmp_path, sealed=sealed)
    changed = False
    restored = False
    retained = None
    previous = None
    original_actual = _OriginalTypes.actual
    original_binding = _TypeBuilder.binding

    def actual(reader, scope, node, *args, **kwargs):
        nonlocal changed, retained, previous
        result = original_actual(reader, scope, node, *args, **kwargs)
        if target == "actual" and not changed and str(node).lower() == "value" and result.binding is not None:
            changed = True
            retained, previous = result.binding, result.binding.dtype
            object.__setattr__(retained, "dtype", "integer")
        return result

    def binding(reader, formal, *args, **kwargs):
        nonlocal changed, retained, previous
        result = original_binding(reader, formal, *args, **kwargs)
        if target == "specific" and not changed and formal.name == "left":
            changed = True
            retained, previous = formal, formal.dtype
            object.__setattr__(retained, "dtype", "real")
        return result

    old_verify = source.verify_publication

    def verify():
        nonlocal restored
        if changed and not restored:
            object.__setattr__(retained, "dtype", previous)
            restored = True
        return old_verify()

    monkeypatch.setattr(_OriginalTypes, "actual", actual)
    monkeypatch.setattr(_TypeBuilder, "binding", binding)
    monkeypatch.setattr(source, "verify_publication", verify)
    from compiler.ir import CompilationError
    if not sealed:
        # The unsealed original namespace check rejects before publication and
        # therefore before this test's restoration callback can be reached.
        with pytest.raises(CompilationError, match="original resolution authority changed"):
            prove_storage_mutation_effects(source, "records::advance", expression=expression)
        assert changed
        assert not restored
        return
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    public = proof.public()
    assert changed
    assert restored
    assert public["frontiers"]
    with pytest.raises(CompilationError, match="original source or resolution changed"):
        proof.preserves_descriptors()


@pytest.mark.parametrize("reexport", [False, True])
def test_unknown_local_import_cannot_hide_host_generic_applicability(tmp_path, reexport):
    def change(text):
        imported = "use unknown,only:assignment(=)"
        prefix = ""
        if reexport:
            prefix = "module reexport\nuse unknown,only:assignment(=)\nend module\n"
            imported = "use reexport"
        return prefix + text.replace("function evaluate(point,values) result(value)\n",
                                     "function evaluate(point,values) result(value)\n" + imported + "\n", 1)
    _path, source, expression = fixture(tmp_path, sealed=True, transform=change)
    public = prove_storage_mutation_effects(source, "records::advance", expression=expression).public()
    assert any("closed-name namespace is unavailable" in row["kind"] for row in public["frontiers"])


def separate_interfaces(reverse=False):
    names = ["record_assignment", "other_record_assignment"]
    if reverse:
        names.reverse()
    return "".join(f"interface assignment(=)\nmodule procedure {name}\nend interface\n" for name in names)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_matching_specific_in_either_separate_interface_block_is_retained(tmp_path, sealed, reverse):
    def change(text):
        return text.replace(GENERICS, separate_interfaces(reverse), 1).replace(
            "type(record),intent(out)::left\ntype(record),intent(in)::right",
            "real(8),intent(out)::left\nlogical,intent(in)::right", 1
        ).replace("value=interpolate(point,values,-1)", "value=.true.", 1)
    _path, source, expression = fixture(tmp_path, sealed=sealed, transform=change)
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert not proof.preserves_descriptors()
    assert any(row["original_source"] == "value = .TRUE." and "defined assignment" in row["kind"]
               for row in proof.public()["frontiers"])


@pytest.mark.parametrize("sealed", [False, True])
def test_separate_derived_interface_blocks_keep_every_excluded_specific(tmp_path, sealed):
    def change(text):
        return text.replace(GENERICS, separate_interfaces(), 1)
    _path, source, expression = fixture(tmp_path, sealed=sealed, transform=change)
    proof = prove_storage_mutation_effects(source, "records::advance", expression=expression)
    assert proof.preserves_descriptors()
    assert len(source.modules["records"].generics["assignment(=)"]) == 2
    assert all(len(row["excluded_specifics"]) == 2 for row in proof.public()["original_inapplicable_assignments"])


@pytest.mark.parametrize("reverse", [False, True])
def test_repeated_ordinary_generic_blocks_use_existing_typed_call_resolution(tmp_path, reverse):
    from fparser.two.utils import walk

    from compiler.frontend.call_bindings import resolve_original_function
    from compiler.frontend.source_effects import SourceEffects

    names = ["integer_value", "real_value"]
    if reverse:
        names.reverse()
    interfaces = "".join(f"interface value\nmodule procedure {name}\nend interface\n" for name in names)
    path = tmp_path / "generic.f90"
    path.write_text("module generic\n" + interfaces + """contains
subroutine advance(a,i)
real(8),intent(inout)::a
integer,intent(in)::i
a=value(i)+value(a)
end subroutine
integer function integer_value(i)
integer,intent(in)::i
integer_value=i
end function
real(8) function real_value(a)
real(8),intent(in)::a
real_value=a
end function
end module
""")
    source = SourceEffects([path])
    source.seal_session()
    routine = source.routines["generic::advance"]
    calls = [node for node in walk(routine.execution) if type(node).__name__ == "Part_Ref"]
    resolved = [resolve_original_function(source, routine.scope, node).procedure for node in calls]
    assert resolved == ["generic::integer_value", "generic::real_value"]
