"""Selected descriptor stability preserves each original declaration's lifetime."""
import pytest

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.local_allocatable_storage import original_local_allocation
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError
from compiler.scopes.conditional_return import _descriptor_record
from compiler.tests.test_hierarchical_child_regions import SOURCE, fixture

CLASSES = """module classes
implicit none
real(8),allocatable::shared(:)
contains
subroutine inspect(formal,n)
real(8),allocatable,intent(inout)::formal(:)
integer,intent(in)::n
real(8),allocatable,save::saved(:)
real(8),allocatable::automatic(:)
logical::ready
ready=allocated(shared).and.allocated(formal).and.allocated(saved).and.allocated(automatic)
end subroutine
end module
"""
ENTRY = "classes::inspect"


def classified(tmp_path):
    path = tmp_path / "classes.f90"
    path.write_text(CLASSES)
    analysis = SourceEffects([path])
    routine = analysis.routines[ENTRY]
    selection = tuple(routine.execution.content)
    _graph, identities = analysis._selected_source(ENTRY, selection)
    storage = prove_storage_mutation_effects(analysis, ENTRY, selection=identities)
    assert storage.preserves_descriptors()
    bindings = tuple(analysis._binding(routine.scope, name) for name in ("shared", "formal", "saved", "automatic"))
    record = _descriptor_record(analysis, ENTRY, identities, storage, bindings, ENTRY, "classification-only")
    return analysis, routine, identities, bindings, record


def test_only_original_module_declarations_enter_the_module_inventory(tmp_path):
    _analysis, _routine, _selection, bindings, record = classified(tmp_path)
    assert record["module_roots"] == ["classes::shared"]
    assert {row["resource"] for row in record["resources"]} == {binding.root for binding in bindings}
    assert record["entry_exit_lifecycle_authorized"] is False
    assert record["payload_access_authorized"] is False
    assert record["entry_definition_authority"] is False
    assert record["retention_requires_sealed_return_contract"] is True


def test_save_local_uses_existing_local_source_authority_without_value_credit(tmp_path):
    analysis, routine, selected, bindings, record = classified(tmp_path)
    saved = bindings[2]
    local = original_local_allocation(analysis, routine, saved)
    assert local.public()["payload_definition_authority"] is False
    assert local.public()["cross_frontier_retention_authorized"] is False
    assert saved.root not in record["module_roots"]
    window = prove_descriptor_window(analysis, ENTRY, selected, (saved.root,))
    resource = window.resource_state(saved.root)
    assert resource["resource_kind"] == "original_local_save_allocatable"
    assert resource["entry_definition_authority"] is False
    assert resource["requires_original_allocation_presence_and_alias_guards"] is True


def test_automatic_local_cannot_acquire_save_or_return_lifetime(tmp_path):
    analysis, routine, selected, bindings, record = classified(tmp_path)
    automatic = bindings[3]
    assert "save" not in automatic.attributes
    assert automatic.root not in record["module_roots"]
    with pytest.raises(CompilationError, match="local SAVE ALLOCATABLE"):
        original_local_allocation(analysis, routine, automatic)
    with pytest.raises(CompilationError, match="local SAVE ALLOCATABLE"):
        prove_descriptor_window(analysis, ENTRY, selected, (automatic.root,))
    assert record["entry_exit_lifecycle_authorized"] is False


@pytest.mark.parametrize(("index", "kind"), [(0, "original_module_allocatable"), (1, "original_allocatable_formal")])
def test_module_and_formal_keep_their_existing_descriptor_issuer(tmp_path, index, kind):
    analysis, _routine, selected, bindings, _record = classified(tmp_path)
    binding = bindings[index]
    proof = prove_descriptor_window(analysis, ENTRY, selected, () if index == 0 else (binding.root,),
                                    module_roots=(binding.root,) if index == 0 else ())
    assert proof.resource_state(binding.root)["resource_kind"] == kind


def test_hierarchical_child_automatic_scratch_stays_out_of_returned_resources(tmp_path):
    source = SOURCE.replace("integer::i\nobserved=0.0_8", "integer::i\nreal(8),allocatable::scratch(:)\n"
                            "if(.not.allocated(scratch))allocate(scratch(n))\nobserved=0.0_8")
    _path, builder = fixture(tmp_path, source)
    _outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    scope, = report["scopes"]
    resources = scope["checked_execution_plan"]["resources"]
    assert not any(row["resource"].endswith("::scratch") for row in resources)
    returned = scope["checked_execution_plan"]["storage_epoch"]["return_contract"]
    middle, = [member for member in returned["members"] if member["procedure"] == "hierarchical::middle"]
    assert any(route["kind"] == "reached_retirement" for route in middle["routes"])
