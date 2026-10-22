"""Native sections consume the same issued interval as numerical placement."""

from copy import copy
from hashlib import sha256

import pytest

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.frontend.native_demand import prove_native_demand
from compiler.ir import CompilationError
from compiler.scopes.interval_draft import prepare_interval_draft
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.segments import StructuredScope, native_demand_fragment
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

SOURCE = """module fields
implicit none
real(8)::old(0:5),mid(0:5),output(0:5)
integer::visits=0
contains
subroutine advance()
integer::i
do i=1,4
mid(i)=2*old(i)
enddo
mid(0)=mid(1)
visits=visits+1
do i=1,4
output(i)=mid(i)+mid(i-1)
enddo
end subroutine
end module
"""


def fixture(tmp_path):
    path = tmp_path / "fields.f90"
    path.write_text(SOURCE)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"fields::" + name: {**FACT, "initialized": initialized}
                          for name, initialized in (("old", "whole"), ("mid", "none"), ("output", "none"))}}
    builder = ScopeBuilder([path], "fields::advance", facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    owner = LexicalOwner(builder)
    originals = tuple(builder.entry.execution.content)
    draft = prepare_interval_draft(owner, originals)
    return builder, owner, draft, originals


def test_partial_native_halo_correction_keeps_exact_read_and_write(tmp_path):
    builder, owner, draft, originals = fixture(tmp_path)
    context = draft.native_context(owner)
    correction = (originals[1],)
    operation = native_demand_fragment(builder, correction, draft.demand, context=context)
    assert operation.sections.available, operation.sections.reason
    resource, = operation.sections.resources
    assert resource.resource == "fields::mid"
    assert resource.reads[0].axes[0].lower.value == 1
    assert resource.writes[0].axes[0].lower.value == 0
    assert resource.overwrites == resource.writes
    assert operation.overwrites == set()  # No whole-allocation definition.
    scope = StructuredScope.from_native(builder, correction, operation)
    scope.check_conservative_definitions(operation.bindings)
    assert scope.tree == [operation]
    assert operation.demand_effects.validate(builder.analysis, builder.entry.qualified,
        correction, context.local_bindings) is operation.demand_effects


def test_original_scalar_write_is_visible_without_device_coherence(tmp_path):
    builder, owner, draft, originals = fixture(tmp_path)
    context = draft.native_context(owner)
    operation = native_demand_fragment(builder, (originals[2],), draft.demand, context=context)
    assert operation.coherence_effects == {}
    assert {row["kind"] for row in operation.summary["operations"]
            if row["resource"] == "fields::visits"} == {"read", "write"}
    assert operation.preserve_original


@pytest.mark.parametrize("change", ["copied_context", "changed_demand", "copied_node"])
def test_interval_native_effects_reject_foreign_authority(tmp_path, change):
    builder, owner, draft, originals = fixture(tmp_path)
    context = draft.native_context(owner)
    selected, demanded = (originals[1],), context.local_bindings
    if change == "copied_context":
        context = copy(context)
    elif change == "changed_demand":
        demanded = demanded[:-1]
    else:
        selected = (copy(selected[0]),)
    with pytest.raises(CompilationError):
        prove_native_demand(builder.analysis, builder.entry.qualified, selected, demanded, context=context)


def test_unknown_section_cannot_borrow_partial_definition_authority(tmp_path):
    builder, owner, draft, originals = fixture(tmp_path)
    context = draft.native_context(owner)
    operation = native_demand_fragment(builder, (originals[1],), draft.demand, context=context)
    # The ordinary emitter must reject an exact footprint it cannot map rather
    # than silently reverting to a whole-array access on an undefined halo.
    scope = StructuredScope.from_native(builder, operation.nodes, operation)
    with pytest.raises(CompilationError, match="buffer mapping is unavailable"):
        scope.emit({}, {}, {}, {}, selector=None)
