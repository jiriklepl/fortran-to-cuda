"""Component READs use the same lifetime scanner, never payload authority."""
import pytest

from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.tests.test_storage_component_reads import fixture


@pytest.mark.parametrize("body", ["out=current%tempfl+current%tempflsgs",
    "if(allocated(current%tempfl))out=sqrt(current%tempfl**2)",
    "write(*,*)current%tempfl\nout=0.0_8"])
def test_numeric_allocatable_component_reads_are_lifetime_inert_without_payload_proof(tmp_path, body):
    _path, analysis, routine, _node = fixture(tmp_path, body=body)
    token = prove_storage_mutation_effects(analysis, routine.qualified)
    record = token.public()
    assert record["complete"], record["frontiers"]
    assert token.preserves_descriptors()
    assert "declared::current_profiles%tempfl" in record["resources"]
    assert not getattr(analysis, "_native_functions", {})
    for key in ("payload_access_authorized", "entry_definition_authority", "capture_authorized",
                "execution_authorized", "motion_authorized", "gpu_legality_established", "alias_authority"):
        assert record[key] is False


@pytest.mark.parametrize("body", ["current%tempfl=out", "current%tempfl(:)=out",
    "call unknown(current%tempfl)", "out=unknown(current%tempfl)",
    "out=c_loc(current%tempfl)", "read(*,*)current%tempfl"])
def test_component_mutation_and_address_frontiers_cannot_borrow_read_classification(tmp_path, body):
    _path, analysis, routine, _node = fixture(tmp_path, body=body)
    token = prove_storage_mutation_effects(analysis, routine.qualified)
    assert not token.public()["complete"]
    assert not token.preserves_descriptors()


def test_component_iostat_is_a_write_designator_not_a_read(tmp_path):
    _path, analysis, routine, _node = fixture(tmp_path, field_type="integer(4)", field_shape="",
        body="write(*,*,iostat=current%tempfl)n")
    record = prove_storage_mutation_effects(analysis, routine.qualified).public()
    assert not record["complete"]
    assert any("I/O output" in event["kind"] for event in record["frontiers"])


def test_component_source_function_index_remains_unavailable_without_original_type_authority(tmp_path):
    _path, analysis, routine, _node = fixture(tmp_path, body="out=current%tempfl(selector())",
        module_declarations="real(8),allocatable::hidden(:)", extra="""integer function selector()
use declared,only:hidden
deallocate(hidden)
selector=1
end function
""")
    token = prove_storage_mutation_effects(analysis, routine.qualified)
    record = token.public()
    # The original type resolver currently rejects this selector before the
    # helper reaches its index callback. Do not manufacture an index Binding,
    # omit an unknown function effect, or credit any retained lifetime here.
    assert not record["complete"]
    assert any("source expression authority: selector()" in event["kind"] for event in record["frontiers"])
    assert not token.preserves_descriptors()


def test_original_numeric_component_indices_join_the_same_resource_scan(tmp_path):
    _path, analysis, routine, _node = fixture(tmp_path, body="out=current%tempfl(n-1)")
    token = prove_storage_mutation_effects(analysis, routine.qualified)
    assert token.public()["complete"]
    assert "argument::n" in token.public()["resources"]
    assert token.preserves_descriptors()


def test_whole_allocatable_numeric_lhs_still_records_automatic_reallocation(tmp_path):
    _path, analysis, routine, _node = fixture(tmp_path, body="out=current%tempfl")
    # Use an authentic separate declaration, rather than changing a parsed
    # binding to pretend that an explicit-shape array is allocatable.
    path = routine.scope.path
    path.write_text(path.read_text().replace("real(8)::out(0:n)", "real(8),allocatable::out(:)"))
    from compiler.frontend.source_effects import SourceEffects

    original = SourceEffects([path])
    token = prove_storage_mutation_effects(original, routine.qualified)
    assert token.public()["complete"]
    assert any(event["resource"] == "argument::out" and "whole allocatable" in event["kind"]
               for event in token.public()["mutations"])
    assert not token.preserves_descriptors()
