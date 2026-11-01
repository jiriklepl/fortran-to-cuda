"""One metadata root retains declaring-type identity without repeated lookup credit."""
from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend import descriptor_windows as windows
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_metadata_arrays import ENTRY, fixture, prove


def metadata_check(tmp_path, *, sealed=False, imported=False):
    path, analysis, _ = fixture(tmp_path)
    if imported:
        path.write_text(path.read_text().replace("type(record),allocatable::marks(:)",
            "end module\nmodule metadata_bridge\nuse metadata_types,only:local_record=>record\n"
            "type(local_record),allocatable::marks(:)").replace(
            "use metadata_types\ncontains", "use metadata_bridge\ncontains"))
        analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    binding = analysis.modules["metadata_bridge" if imported else "metadata_types"].bindings["marks"]
    return path, analysis, binding, windows._Check(analysis)


@pytest.mark.parametrize("sealed", [False, True])
def test_repeated_original_fields_preserve_full_proof_and_source_work(tmp_path, monkeypatch, sealed):
    _path, analysis, selection = fixture(tmp_path, action="\n".join(
        "a(marks(i)%index)=marks(i)%weight" for _ in range(35)))
    if sealed:
        analysis.seal_session()
    checks = []
    original = windows._Check.publish

    def publish(check, *args):
        result = original(check, *args)
        checks.append(check)
        return result

    monkeypatch.setattr(windows._Check, "publish", publish)
    public = prove(analysis, selection).public()
    check, = checks
    assert check.object_builder.visits == 1
    assert public["resolution_validation"]["procedure_limit"] == analysis.procedure_limit
    assert check.operations >= 35
    assert check.nodes == public["source_traversal_entries"]
    assert set(public["native_metadata_descriptor_requirements"][0]["selected_fields"]) == {"index", "weight"}
    assert not public["execution_authorized"]
    assert not public["entry_definition_authority"]


def test_reuse_matches_uncached_public_oracle_and_every_charged_counter(tmp_path, monkeypatch):
    _path, analysis, selection = fixture(tmp_path)
    cached = windows._derive(analysis, ENTRY, selection, ("argument::a",))
    original = windows._Check.metadata_schema
    publish = windows._Check.publish
    publishing = set()
    forced_resolutions = 0

    def always_resolve(check, *args):
        nonlocal forced_resolutions
        # Recompute during the original semantic scan. Publication retains
        # exact consumed receipt identities and must authenticate them intact.
        if check not in publishing:
            check.metadata_schemas.clear()
            forced_resolutions += 1
        return original(check, *args)

    def retain_consumed_receipts(check, *args, **kwargs):
        publishing.add(check)
        try:
            return publish(check, *args, **kwargs)
        finally:
            publishing.remove(check)

    monkeypatch.setattr(windows._Check, "metadata_schema", always_resolve)
    monkeypatch.setattr(windows._Check, "publish", retain_consumed_receipts)
    uncached = windows._derive(analysis, ENTRY, selection, ("argument::a",))
    assert forced_resolutions > 1
    assert cached == uncached
    assert cached["source_traversal_entries"] > 0
    assert cached["resolution_validation"]["work_entries"] > 0


def test_publication_cannot_replace_retained_metadata_receipts_with_fresh_resolutions(tmp_path, monkeypatch):
    _path, analysis, selection = fixture(tmp_path)
    original = windows._Check.metadata_schema

    def replace_consumed_receipts(check, *args):
        check.metadata_schemas.clear()
        return original(check, *args)

    monkeypatch.setattr(windows._Check, "metadata_schema", replace_consumed_receipts)
    with pytest.raises(CompilationError, match="consuming source attachment changed at publication"):
        windows._derive(analysis, ENTRY, selection, ("argument::a",))


def test_another_original_object_still_spends_shared_resolution_budget(tmp_path):
    path, _, _ = fixture(tmp_path, module_extra="type(record),allocatable::other(:)")
    analysis = SourceEffects([path], procedures=1)
    owner = analysis.modules["metadata_types"]
    check = windows._Check(analysis)
    check.metadata_object(owner.bindings["marks"])
    for _ in range(80):
        check.metadata_object(owner.bindings["marks"])
    assert check.object_builder.visits == 1
    with pytest.raises(CompilationError, match="type resolution budget exhausted"):
        check.metadata_object(owner.bindings["other"])
    assert check.object_builder.visits == 2


def test_declaration_traversal_remains_charged_after_type_reuse(tmp_path):
    _path, _analysis, binding, check = metadata_check(tmp_path)
    check.metadata_object(binding)
    first = check.nodes
    check.metadata_object(binding)
    assert check.nodes == 2 * first
    check.nodes = 65536
    with pytest.raises(CompilationError, match="original-source traversal budget exhausted"):
        check.metadata_object(binding)
    assert check.object_builder.visits == 1


@pytest.mark.parametrize("mutation", [
    "binding", "owner", "binding_kind", "imports", "module_role", "type_copy", "extra_type",
    "members", "schema_node", "schema_scope", "schema_name", "schema_private", "schema_reason",
    "kind_scope", "budget", "session", "analysis", "check",
])
def test_reused_metadata_receipt_rejects_changed_exact_authority(tmp_path, mutation):
    _path, analysis, binding, check = metadata_check(tmp_path, imported=True)
    check.metadata_object(binding)
    owner = analysis.modules["metadata_types"]
    schema = owner.component_types["record"]
    if mutation == "binding":
        binding = copy(binding)
    elif mutation == "owner":
        binding.declaring_scope = owner
    elif mutation == "binding_kind":
        binding.kind = 8
    elif mutation == "imports":
        binding.declaring_scope.imports["local_record"] = ("metadata_types", "absent")
    elif mutation == "module_role":
        analysis.modules["metadata_types"] = copy(owner)
    elif mutation == "type_copy":
        owner.component_types["record"] = copy(schema)
    elif mutation == "extra_type":
        owner.component_types["ambiguous"] = replace(schema, name="ambiguous")
    elif mutation == "members":
        schema.members.pop("weight")
    elif mutation.startswith("schema_"):
        attribute = mutation.removeprefix("schema_")
        value = {"node": copy(schema.node), "scope": binding.declaring_scope,
                 "name": "changed", "private": not schema.private, "reason": "changed"}[attribute]
        object.__setattr__(schema, attribute, value)
    elif mutation == "kind_scope":
        owner.kinds.values["new_kind"] = 8
    elif mutation == "budget":
        analysis.procedure_limit += 1
    elif mutation == "session":
        analysis.seal_session()
    elif mutation == "analysis":
        check.analysis = copy(analysis)
    else:
        replacement = windows._Check(analysis)
        replacement.metadata_schemas = check.metadata_schemas.copy()
        check = replacement
    with pytest.raises(CompilationError):
        check.metadata_object(binding)


@pytest.mark.parametrize("sealed", [False, True])
def test_imported_alias_receipt_keeps_all_original_declaring_scopes(tmp_path, sealed):
    _path, analysis, binding, check = metadata_check(tmp_path, sealed=sealed, imported=True)
    check.metadata_object(binding)
    visits = check.object_builder.visits
    assert visits == 2
    check.metadata_object(binding)
    assert check.object_builder.visits == visits
    receipt, = check.metadata_schemas.values()
    assert {scope.module for scope, *_ in receipt.scopes} == {"metadata_types", "metadata_bridge"}
    assert receipt.schema is analysis.modules["metadata_types"].component_types["record"]


@pytest.mark.parametrize("mutation", ["members", "type_copy", "imports", "file"])
def test_final_publication_rechecks_metadata_after_last_reuse(tmp_path, monkeypatch, mutation):
    path, analysis, selection = fixture(tmp_path)
    original = windows._Check.publish
    changed = False

    def publish(check, *args):
        nonlocal changed
        assert check.metadata_schemas
        owner = analysis.modules["metadata_types"]
        if mutation == "members":
            owner.component_types["record"].members.pop("weight")
        elif mutation == "type_copy":
            owner.component_types["record"] = copy(owner.component_types["record"])
        elif mutation == "imports":
            owner.imports["changed"] = ("absent", "changed")
        else:
            path.write_bytes(path.read_bytes() + b"\n! changed after final metadata lookup\n")
        changed = True
        return original(check, *args)

    monkeypatch.setattr(windows._Check, "publish", publish)
    with pytest.raises(CompilationError):
        prove(analysis, selection)
    assert changed
    assert not getattr(analysis, "_descriptor_window_proofs", {})
