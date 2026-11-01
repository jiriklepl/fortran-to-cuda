"""Fixed lexical selectors retain the complete unchanged storage-effects walk."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.call_bindings import _OriginalTypes
from compiler.frontend.descriptor_associations import original_fixed_association
from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.storage_component_reads import numeric_component_read
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_associations import SOURCE, fixture


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("selector", ["settings%reference", "settings", "count"])
def test_whole_original_association_body_is_checked_without_payload_or_definition_credit(tmp_path, sealed, selector):
    source = SOURCE.replace("ref=>settings%reference", "ref=>" + selector)
    if selector == "settings":
        source = source.replace("ref%mode", "ref%reference%mode").replace("ref%point", "ref%reference%point")
    elif selector == "count":
        source = source.replace("if(ref%mode>0)a(1)=a(1)+ref%point(1)", "ref=3\na(1)=a(1)+ref")
    path, analysis, routine, original = fixture(tmp_path, source, sealed=sealed)
    before = path.read_bytes()
    _graph, selected = analysis._selected_source(routine.qualified, (original,))
    window = prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",)).public()
    assert window["schema_version"] == 15
    association, = window["native_fixed_scalar_associations"]
    assert association["requires_complete_original_body"] is True
    storage = prove_storage_mutation_effects(analysis, routine.qualified, selection=(original,))
    record = storage.public()
    assert record["version"] == 6
    assert record["complete"] is True, record["frontiers"]
    assert storage.preserves_descriptors()
    assert len(record["fixed_scalar_associations"]) == 1
    assert analysis._summary_authority()["descriptor_window_association_version"] == 15
    for flag in ("payload_access_authorized", "entry_definition_authority", "capture_authorized",
                 "execution_authorized", "motion_authorized", "alias_authority"):
        assert record[flag] is False
    assert path.read_bytes() == before


@pytest.mark.parametrize("tail", ["deallocate(a)", "call unknown_escape(a)", "call unknown_escape(ref)",
                                  "a(1)=unknown(ref)", "ref=settings%reference"])
def test_later_alias_body_mutation_and_whole_object_escape_are_never_hidden(tmp_path, tail):
    source = SOURCE.replace("end associate", tail + "\nend associate")
    _path, analysis, routine, original = fixture(tmp_path, source)
    record = prove_storage_mutation_effects(analysis, routine.qualified, selection=(original,)).public()
    assert record["frontiers"] or record["mutations"] or record["escapes"]
    _graph, selected = analysis._selected_source(routine.qualified, (original,))
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))


def test_unsupported_selector_still_scans_later_known_allocation_mutation(tmp_path):
    source = SOURCE.replace("ref=>settings%reference", "ref=>settings%reference%point").replace(
        "end associate", "deallocate(a)\nend associate")
    _path, analysis, routine, original = fixture(tmp_path, source)
    record = prove_storage_mutation_effects(analysis, routine.qualified, selection=(original,)).public()
    assert record["frontiers"]
    assert any(item["kind"] == "deallocate" and item["resource"] == "argument::a" for item in record["mutations"])


def component_fixture(tmp_path):
    source = SOURCE.replace("ref=>settings%reference", "ref=>settings").replace(
        "ref%mode", "ref%reference%mode").replace("ref%point", "ref%reference%point")
    _path, analysis, routine, original = fixture(tmp_path, source)
    token = original_fixed_association(analysis, routine, original)
    node = next(item for item in walk(original) if type(item).__name__ == "Data_Ref"
                and str(item).lower().replace(" ", "") == "ref%reference%mode")
    return analysis, routine, original, token, node


def test_lexical_component_service_is_explicit_and_does_not_relax_the_original_type_resolver(tmp_path):
    analysis, routine, _original, token, node = component_fixture(tmp_path)
    scope = analysis.source_scope_for(node, routine.scope)
    with pytest.raises(CompilationError):
        _OriginalTypes(analysis).actual(scope, node)
    with pytest.raises(CompilationError):
        numeric_component_read(analysis, routine, node, visit_index=lambda _node: None,
                               visit_scope=lambda _scope: None, tick=lambda: None)
    result = numeric_component_read(analysis, routine, node, visit_index=lambda _node: None,
                                    visit_scope=lambda _scope: None, tick=lambda: None, lexical_alias=token)
    assert result.public()["resource"] == "scalar_aliases::settings%reference%mode"
    assert result.public()["capture_authorized"] is False


@pytest.mark.parametrize("mutation", ["foreign_analysis", "foreign_routine", "wrong_construct", "not_typed"])
def test_lexical_component_context_requires_the_exact_original_alias_owner(tmp_path, mutation):
    analysis, routine, _original, token, node = component_fixture(tmp_path)
    if mutation == "not_typed":
        token = object()
    else:
        token = copy(token)
        if mutation == "foreign_analysis":
            object.__setattr__(token, "_analysis", copy(analysis))
        elif mutation == "foreign_routine":
            object.__setattr__(token, "_routine", copy(routine))
        else:
            object.__setattr__(token, "_original", copy(token._original))
    with pytest.raises(CompilationError):
        numeric_component_read(analysis, routine, node, visit_index=lambda _node: None,
                               visit_scope=lambda _scope: None, tick=lambda: None, lexical_alias=token)


def test_alias_private_declaration_cannot_bypass_namelist_observer(tmp_path):
    source = SOURCE.replace("integer::count", "integer::count\nnamelist /observe/ settings")
    _path, analysis, routine, original = fixture(tmp_path, source)
    record = prove_storage_mutation_effects(analysis, routine.qualified, selection=(original,)).public()
    assert record["frontiers"]


@pytest.mark.parametrize("mutation", ["proof", "alias", "selector", "scope"])
def test_component_tick_cannot_reseed_an_earlier_lexical_context(tmp_path, mutation):
    analysis, routine, _original, token, node = component_fixture(tmp_path)
    changed = False

    def tick():
        nonlocal changed
        if changed:
            return
        changed = True
        if mutation == "proof":
            object.__setattr__(token, "_association", copy(token._association))
        elif mutation == "alias":
            aliases = token._association.scope.bindings
            aliases["ref"] = copy(aliases["ref"])
        elif mutation == "selector":
            header = token._original.content[0]
            object.__setattr__(header.items[1], "items", (copy(header.items[1].items[0]),))
        else:
            object.__setattr__(token._association, "scope", copy(token._association.scope))

    with pytest.raises(CompilationError):
        numeric_component_read(analysis, routine, node, visit_index=lambda _node: None,
                               visit_scope=lambda _scope: None, tick=tick, lexical_alias=token)


def test_cached_storage_token_cannot_replace_its_original_alias_selection(tmp_path):
    analysis, routine, original, _token, _node = component_fixture(tmp_path)
    storage = prove_storage_mutation_effects(analysis, routine.qualified, selection=(original,))
    object.__setattr__(storage, "_selection", (copy(original),))
    with pytest.raises(CompilationError):
        storage.validate()
