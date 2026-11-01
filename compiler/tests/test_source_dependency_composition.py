"""Compose expected source receipts without rebuilding or reseeding them."""
from copy import copy

import pytest

from compiler.frontend import source_dependency_receipts as receipts
from compiler.frontend import storage_record_regions as regions
from compiler.ir import CompilationError
from compiler.tests.test_storage_record_regions import SOURCE, fixture, prove

EXTRA = """module widths
integer,parameter::wp=8
end module
""" + SOURCE.replace("end module\n", """subroutine extra(value)
use widths,only:wp
real(wp),intent(inout)::value
value=value+1
end subroutine
end module
""")


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("existing_closure", [False, True])
def test_checkpoint_reuses_existing_expected_receipts_without_resolution_or_syntax_capture(
        tmp_path, monkeypatch, sealed, existing_closure):
    import compiler.frontend.descriptor_teams as teams
    import compiler.frontend.source_functions as functions

    data = fixture(tmp_path, sealed=sealed)
    proof = prove(data)
    before = proof.public()

    def unexpected(*_args, **_kwargs):
        pytest.fail("an existing consumed dependency was rebuilt")

    monkeypatch.setattr(teams, "_group_syntax_checkpoint", unexpected)
    monkeypatch.setattr(functions, "_validate_resolution", unexpected)
    extra = proof._roles[:2] if existing_closure else ()
    checkpoint = regions.RecordRegionCheckpoint(data[1], proof, extra)
    checkpoint.require()
    assert proof.public() == before
    assert checkpoint.dependencies is not proof._dependencies
    if sealed:
        assert checkpoint.dependencies.ownership is proof._dependencies.ownership


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("when", ["before_composition", "after_composition"])
@pytest.mark.parametrize("fault", ["binding", "schema", "body", "membership"])
def test_imported_base_expected_rows_are_never_reseeded(tmp_path, sealed, when, fault):
    data = fixture(tmp_path, sealed=sealed)
    proof = prove(data)
    module = data[1].modules["regions"]

    def mutate():
        if fault == "binding":
            object.__setattr__(module.bindings["field"], "kind", 4)
        elif fault == "schema":
            schema = module.component_types["cell"]
            object.__setattr__(schema, "members", dict(schema.members))
        elif fault == "body":
            object.__setattr__(proof._roles[7][0].items[-1], "string", "index")
        else:
            node = proof._roles[1].scope.node
            part = node.parent
            object.__setattr__(part, "content", tuple(item for item in part.content if item is not node))

    if when == "before_composition":
        mutate()
    checkpoint = regions.RecordRegionCheckpoint(data[1], proof)
    if when == "after_composition":
        mutate()
    with pytest.raises(CompilationError, match="source dependency"):
        checkpoint.require()


def test_extra_closure_captures_only_new_namespaces_and_keeps_one_pure_suffix(tmp_path, monkeypatch):
    from compiler.frontend import descriptor_intrinsics as intrinsics

    data = fixture(tmp_path, EXTRA, sealed=True)
    proof = prove(data)
    extra = data[1].routines["regions::extra"]
    old_scopes = {id(row[0]) for row in proof._dependencies.saved[0]}
    original_capture = intrinsics.intrinsic_scope_attachments
    captures = []

    def captured(scope, **kwargs):
        captures.append(scope)
        return original_capture(scope, **kwargs)

    monkeypatch.setattr(intrinsics, "intrinsic_scope_attachments", captured)
    checkpoint = regions.RecordRegionCheckpoint(data[1], proof, (extra, *proof._roles[:2], extra))
    assert captures
    assert all(id(scope) not in old_scopes for scope in captures)
    assert extra.scope in captures
    assert data[1].modules["widths"] in captures
    calls = []
    original_require = receipts._require_dependencies

    def checked(dependencies):
        calls.append(dependencies)
        return original_require(dependencies)

    monkeypatch.setattr(receipts, "_require_dependencies", checked)
    checkpoint.require()
    assert len(calls) == 1
    scopes, syntax, declarations, parents, layouts, schemas, routines = calls[0]
    assert extra in [row[0] for row in routines]
    assert len(syntax) == 1
    for rows in (*syntax[0], scopes, declarations, parents, layouts, schemas, routines):
        assert len({id(row[0]) for row in rows}) == len(rows)


@pytest.mark.parametrize("fault", ["formal", "imported_binding", "imported_constant", "extra_body", "policy"])
def test_new_closure_and_imported_kind_dependencies_remain_exact(tmp_path, fault):
    data = fixture(tmp_path, EXTRA, sealed=True)
    proof = prove(data)
    extra = data[1].routines["regions::extra"]
    checkpoint = regions.RecordRegionCheckpoint(data[1], proof, (extra,))
    if fault == "formal":
        object.__setattr__(extra.scope.bindings["value"], "kind", 4)
    elif fault == "imported_binding":
        object.__setattr__(data[1].modules["widths"].bindings["wp"], "kind", 8)
    elif fault == "imported_constant":
        constant = data[1].modules["widths"].kinds.values["wp"]
        object.__setattr__(constant, "string", "4")
    elif fault == "extra_body":
        node = extra.execution.content[0].items[0]
        object.__setattr__(node, "string", "hidden")
    else:
        data[1].operation_limit += 1
    with pytest.raises(CompilationError, match="source dependency"):
        checkpoint.require()


@pytest.mark.parametrize("fault", ["registry", "token", "receipt"])
def test_composition_preserves_exact_private_issuer_and_registry_checks(tmp_path, fault):
    data = fixture(tmp_path, sealed=True)
    proof = prove(data)
    checkpoint = regions.RecordRegionCheckpoint(data[1], proof)
    if fault == "registry":
        data[1]._owned_record_regions = dict(data[1]._owned_record_regions)
    elif fault == "token":
        data[1]._owned_record_regions[proof.identity] = copy(proof)
    else:
        object.__setattr__(proof, "_dependencies", copy(proof._dependencies))
    with pytest.raises(CompilationError, match="record region"):
        checkpoint.require()


def test_private_composition_is_not_an_arena_fact_or_foreign_analysis_authority(tmp_path):
    first = fixture(tmp_path, sealed=True)
    proof = prove(first)
    composition = receipts.DependencyCapture(first[1], components=True, inherited=proof._dependencies)
    with pytest.raises(CompilationError, match="not a shared arena fact"):
        composition.recipe()
    other = fixture(tmp_path, sealed=True)
    with pytest.raises(CompilationError, match="same analysis"):
        receipts.DependencyCapture(other[1], components=True, inherited=proof._dependencies)
