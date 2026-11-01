"""Returned scalar component associations retain source identity, not view identity."""
from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend.call_bindings import resolve_source_call
from compiler.ir import CompilationError
from compiler.tests.test_hierarchical_child_regions import SOURCE, fixture, owner_at


def component_source(*, nested=False, intent="in", component="real(8)::dt"):
    declarations = "type clock_state\n" + component + "\nreal(8)::other\nend type\n"
    if nested:
        declarations += "type wrapper\ntype(clock_state)::inner\nend type\ntype(wrapper)::clock,other_clock\n"
        actual = "clock%inner%dt"
    else:
        declarations += "type(clock_state)::clock,other_clock\n"
        actual = "clock%dt"
    return (SOURCE.replace("real(8)::observed", "real(8)::observed\n" + declarations)
            .replace("subroutine middle(n,change)", "subroutine middle(n,change,dt)\nreal(8),intent(" + intent + ")::dt")
            .replace("call middle(n,change)", "call middle(n,change," + actual + ")")
            .replace("observed=0.0_8", "observed=dt"))


def retained(tmp_path, **options):
    path, builder = fixture(tmp_path, component_source(**options))
    owner = owner_at(builder)
    assert owner.run(builder.inline.prepare(owner.closed_selection))
    call, = [item for item in owner.return_calls if item.resolved.procedure == "hierarchical::middle"]
    mapping, = [item for item in call.resolved.mappings if item.formal == "dt"]
    return path, builder, owner, call, mapping


@pytest.mark.parametrize("nested", [False, True])
def test_readonly_scalar_component_retains_original_source_association(tmp_path, nested):
    path, builder, owner, call, mapping = retained(tmp_path, nested=nested)
    scope = builder.analysis.source_scope_for(call.node, owner.routine.scope)
    current = resolve_source_call(builder.analysis, scope, call.node)
    fresh, = [item for item in current.mappings if item.formal == "dt"]
    assert current.public() == call.resolved.public()
    assert fresh.binding is not mapping.binding
    assert fresh.actual is mapping.actual
    assert fresh.binding.component_object is mapping.binding.component_object
    assert call.validate(owner) is call
    outputs, report = builder.finish()
    assert report["scope_count"] == 1
    assert "".join(str(mapping.actual).lower().split()) in "".join(call.source.lower().split())
    assert report["scopes"][0]["checked_execution_plan"]["storage_epoch"]["return_contract"]["version"] == 2
    assert path.exists()


@pytest.mark.parametrize("mutation", ["copy", "different_object", "selector", "formal_declaration"])
def test_changed_or_forged_mapping_cannot_borrow_scalar_receipt(tmp_path, mutation):
    _path, builder, owner, call, mapping = retained(tmp_path)
    binding = copy(mapping.binding)
    if mutation == "different_object":
        binding.component_object = builder.analysis.modules["hierarchical"].bindings["other_clock"]
    elif mutation == "selector":
        from fparser.two.Fortran2003 import Data_Ref
        binding.component_selector = Data_Ref("clock%other")
    if mutation == "formal_declaration":
        forged = replace(mapping, formal_binding=copy(mapping.formal_binding))
    else:
        forged = replace(mapping, binding=binding)
    assert forged.public() == mapping.public()
    resolved = replace(call.resolved, mappings=tuple(forged if item is mapping else item for item in call.resolved.mappings))
    with pytest.raises(CompilationError, match="original actual association"):
        replace(call, resolved=resolved).validate(owner)


@pytest.mark.parametrize("mutation", ["object", "selector", "schema", "field_declaration"])
def test_retained_component_receipt_requires_exact_original_source_links(tmp_path, mutation):
    _path, builder, owner, call, _mapping = retained(tmp_path)
    (index, receipt), = call.scalar_components
    originals = list(receipt.originals)
    if mutation == "object":
        originals[0] = builder.analysis.modules["hierarchical"].bindings["other_clock"]
    elif mutation == "selector":
        originals[1] = copy(originals[1])
    elif mutation == "schema":
        originals[2] = copy(originals[2])
    else:
        originals[-1] = copy(originals[-1])
    forged = replace(receipt, originals=tuple(originals))
    with pytest.raises(CompilationError, match="original actual association"):
        replace(call, scalar_components=((index, forged),)).validate(owner)


def test_ordinary_array_actual_keeps_exact_binding_identity(tmp_path):
    _path, _builder, owner, _call, _mapping = retained(tmp_path)
    middle = owner.members["hierarchical::middle"]
    call, = [item for item in middle.return_calls if item.resolved.procedure == "hierarchical::consumer"]
    mapping, = [item for item in call.resolved.mappings if item.formal == "b"]
    forged = replace(mapping, binding=copy(mapping.binding))
    assert forged.public() == mapping.public()
    resolved = replace(call.resolved, mappings=tuple(forged if item is mapping else item for item in call.resolved.mappings))
    with pytest.raises(CompilationError, match="original actual association"):
        replace(call, resolved=resolved).validate(middle)


@pytest.mark.parametrize("component", ["real(8),pointer::dt", "real(8),allocatable::dt"])
def test_associated_scalar_component_still_declines(tmp_path, component):
    _path, builder = fixture(tmp_path, component_source(component=component))
    owner = owner_at(builder)
    with pytest.raises(CompilationError, match="unproved lifetime/escape"):
        owner.run(builder.inline.prepare(owner.closed_selection))
    assert not hasattr(owner, "return_contract")


@pytest.mark.parametrize("intent", ["out", "inout"])
def test_component_receipt_does_not_admit_writable_scalar_formals(tmp_path, intent):
    _path, builder = fixture(tmp_path, component_source(intent=intent))
    owner = owner_at(builder)
    with pytest.raises(CompilationError, match="original actual association"):
        owner.run(builder.inline.prepare(owner.closed_selection))
    assert not hasattr(owner, "return_contract")


def test_changed_original_file_still_blocks_publication(tmp_path):
    path, builder, _owner, _call, _mapping = retained(tmp_path)
    path.write_text(path.read_text().replace("real(8)::dt", "real(4)::dt"))
    with pytest.raises(CompilationError, match="changed|source"):
        builder.finish()


@pytest.mark.parametrize("mutation", ["schema_container", "foreign_declaring_scope"])
def test_existing_component_declaration_authority_rejects_rebinding(tmp_path, mutation):
    from fparser.two.utils import walk

    from compiler.frontend.source_effects import SourceEffects
    from compiler.scopes.conditional_return import _scalar_component_source

    path = tmp_path / "source.f90"
    path.write_text(component_source())
    analysis = SourceEffects([path])
    routine = analysis.routines["hierarchical::advance"]
    call = next(node for node in walk(routine.execution)
                if type(node).__name__ == "Call_Stmt" and str(node.items[0]).lower() == "middle")
    resolved = resolve_source_call(analysis, routine.scope, call)
    mapping, = [item for item in resolved.mappings if item.formal == "dt"]
    assert _scalar_component_source(analysis, routine.scope, mapping)
    module = analysis.modules["hierarchical"]
    if mutation == "schema_container":
        module.component_types["clock_state"] = copy(module.component_types["clock_state"])
    else:
        module.bindings["clock"].declaring_scope = copy(module)
    with pytest.raises(CompilationError, match="schema changed|declaration scope authority"):
        _scalar_component_source(analysis, routine.scope, mapping)
