"""Original scalar aliases are declaration evidence, never body authority."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_associations import original_fixed_association
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module scalar_aliases
type reference_t
integer::mode
integer::point(3)
end type
type settings_t
type(reference_t)::reference
real(8)::coefficient
end type
type(settings_t),save::settings
integer::count
contains
subroutine advance(a)
real(8),allocatable,intent(inout)::a(:)
associate(ref=>settings%reference)
if(ref%mode>0)a(1)=a(1)+ref%point(1)
end associate
deallocate(a)
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE, *, sealed=False):
    path = tmp_path / "scalar_aliases.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines["scalar_aliases::advance"]
    original = next(node for node in walk(routine.execution) if type(node).__name__ == "Associate_Construct")
    return path, analysis, routine, original


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("selector", ["settings%reference", "settings", "count"])
def test_exact_fixed_scalar_alias_keeps_original_scope_and_has_no_body_or_payload_authority(tmp_path, sealed, selector):
    source = SOURCE.replace("ref=>settings%reference", "ref=>"+selector)
    if selector == "settings":
        source = source.replace("ref%mode", "ref%reference%mode").replace("ref%point", "ref%reference%point")
    elif selector == "count":
        source = source.replace("if(ref%mode>0)a(1)=a(1)+ref%point(1)", "a(1)=a(1)+ref")
    path, analysis, routine, original = fixture(tmp_path, source, sealed=sealed)
    before = path.read_bytes()
    proof = original_fixed_association(analysis, routine, original)
    assert proof.validate() is proof
    record = proof.public()
    assert record["selectors"][0]["alias"] == "ref"
    assert record["requires_complete_original_body"] is True
    for flag in ("payload_access_authorized", "entry_definition_authority", "alias_separation_authorized",
                 "descriptor_lifetime_authorized", "execution_authorized", "motion_authorized"):
        assert record[flag] is False
    assert path.read_bytes() == before


@pytest.mark.parametrize(("selector", "declaration"), [
    ("array", "integer::array(3)"),
    ("item", "integer,pointer::item"),
    ("item", "integer,allocatable::item"),
    ("item", "integer,target::item"),
    ("item", "integer,parameter::item=3"),
    ("item", "integer,external::item"),
    ("count+1", ""),
    ("mystery()", "external mystery\ninteger::mystery"),
])
def test_dynamic_array_pointer_target_and_expression_selectors_do_not_get_fixed_receipt(tmp_path, selector, declaration):
    source = SOURCE.replace("integer::count", "integer::count\n"+declaration).replace("ref=>settings%reference", "ref=>"+selector)
    _path, analysis, routine, original = fixture(tmp_path, source)
    with pytest.raises(CompilationError):
        original_fixed_association(analysis, routine, original)


def test_unknown_body_remains_an_independent_obligation(tmp_path):
    source = SOURCE.replace("if(ref%mode>0)a(1)=a(1)+ref%point(1)", "call unknown_escape(ref)")
    _path, analysis, routine, original = fixture(tmp_path, source)
    proof = original_fixed_association(analysis, routine, original)
    assert proof.public()["requires_complete_original_body"] is True
    assert analysis.summarize(routine.qualified)["complete"] is False


@pytest.mark.parametrize("change", ["scope", "selector", "projected", "record"])
def test_early_scope_callback_cannot_reseed_original_alias_attachments(tmp_path, change):
    _path, analysis, routine, original = fixture(tmp_path)
    association = analysis._associate_scopes[id(original)]
    changed = False

    def observe(_scope):
        nonlocal changed
        if changed:
            return
        changed = True
        if change == "scope":
            object.__setattr__(association, "scope", copy(association.scope))
        elif change == "selector":
            object.__setattr__(association.selectors[0], "selector", copy(association.selectors[0].selector))
        elif change == "projected":
            association.scope.bindings["ref"] = copy(association.scope.bindings["ref"])
        else:
            analysis._associate_scopes[id(original)] = copy(association)

    with pytest.raises(CompilationError):
        original_fixed_association(analysis, routine, original, scope_validator=observe)


@pytest.mark.parametrize("change", ["record", "alias", "header", "schema"])
def test_final_publication_cannot_replace_equal_public_alias_source(tmp_path, monkeypatch, change):
    _path, analysis, routine, original = fixture(tmp_path)
    proof = original_fixed_association(analysis, routine, original)
    association = analysis._associate_scopes[id(original)]
    publish = analysis.verify_publication

    def observe():
        publish()
        if change == "record":
            analysis._associate_scopes[id(original)] = copy(association)
        elif change == "alias":
            association.scope.bindings["ref"] = copy(association.scope.bindings["ref"])
        elif change == "header":
            original.content = [copy(original.content[0]), *original.content[1:]]
        else:
            scope = analysis.modules["scalar_aliases"]
            scope.component_types["reference_t"] = copy(scope.component_types["reference_t"])

    monkeypatch.setattr(analysis, "verify_publication", observe)
    with pytest.raises(CompilationError):
        proof.validate()


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("change", ["association", "selector", "part"])
def test_late_declaration_callback_cannot_detach_consumed_header_children(tmp_path, monkeypatch, sealed, change):
    import compiler.frontend.descriptor_associations as associations

    _path, analysis, routine, original = fixture(tmp_path, sealed=sealed)
    proof = original_fixed_association(analysis, routine, original)
    validate_roles = associations._validate_scope_roles

    def observe(*args):
        validate_roles(*args)
        values = original.content[0].items[1]
        if change == "association":
            object.__setattr__(values, "items", (copy(values.items[0]),))
        else:
            item = values.items[0]
            if change == "selector":
                object.__setattr__(item, "items", (*item.items[:2], copy(item.items[2])))
            else:
                selector = item.items[2]
                object.__setattr__(selector, "items", (copy(selector.items[0]), *selector.items[1:]))

    monkeypatch.setattr(associations, "_validate_scope_roles", observe)
    with pytest.raises(CompilationError, match="attachment changed"):
        proof.validate()
