"""Bound reads use original forwarded metadata without publishing other fields."""
import pytest

from compiler.tests.test_conditional_return import compile_fixture
from compiler.tests.test_native_ancestor import SOURCE

MAPPED_SOURCE = SOURCE.replace("real(8)::observed", "real(8)::observed\ninteger::tile(1)=[8]").replace(
    "observed=0.0_8\ndo i=1,n\nobserved=observed+b(i)\nenddo", "observed=0.0_8").replace(
    "b(i)=2.0_8*a(i)\nenddo", "b(i)=2.0_8*a(i)\nenddo\ndo i=1,1\ntile(i)=n\nenddo").replace(
    "subroutine consumer(a,b,n)\nreal(8),intent(inout)::a(:)",
    "subroutine consumer(a,b,n)\nreal(8)::scratch(tile(1))\nreal(8),intent(inout)::a(:)").replace(
    "a(i)=b(i)+1.0_8", "a(i)=b(i)+real(tile(1),8)")


def test_original_metadata_mapping_selects_only_its_forwarded_canonical_handle(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    _outputs, report, scope, proof = compile_fixture(tmp_path, MAPPED_SOURCE)
    validate_checked_execution_plans(report)
    route, = [route for member in proof["members"] for route in member["routes"] if "specification_entry" in route]
    phase = route["specification_entry"]
    assert phase["version"] == 2
    assert phase["runtime"] == "fort_scope_observe_v1"
    binding, = phase["bindings"]
    assert binding["resource"] == binding["canonical_resource"] == "ancestor_fixture::tile"
    assert route["handle_forwarding"][str(binding["callee_slot"])] == binding["caller_slot"]
    edit, = [e for e in report["source_edits"] if [e["first_line"], e["last_line"]] == phase["source_span"]]
    text = edit["replacement"]
    assert f'fort_specification_buffers(1) = fort_buffer_{binding["caller_slot"]}' in text
    assert "fort_specification_buffers(2)" not in text
    assert "fort_scope_buffer_valid_v1(" in text
    assert "if (fort_specification_precise) then" in text
    assert "else\nfort_specification_status = fort_scope_observe_all_v1(" in text
    coordinator, = [row for row in scope["module_coordinators"] if row["procedure"] == route["procedure"]]
    assert text.index("fort_scope_observe_v1(") < text.index("call " + coordinator["entry"])
    assert "scratch" not in str(scope["resource_bindings"])


@pytest.mark.parametrize("unmapped", [False, True])
def test_multiple_bound_arrays_are_all_observed_or_use_the_domain_fallback(tmp_path, unmapped):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    source = MAPPED_SOURCE.replace("integer::tile(1)=[8]",
        "integer::tile(1)=[8], tile_y(1)=[8], tile_z(1)=[8]").replace(
        "tile(i)=n", "tile(i)=n\ntile_y(i)=n\ntile_z(i)=n").replace(
        "scratch(tile(1))", "scratch(tile(1),tile_y(1),tile_z(1))")
    if not unmapped:
        source = source.replace("a(i)=b(i)+real(tile(1),8)",
                                "a(i)=b(i)+real(tile(1)+tile_y(1)+tile_z(1),8)")
    _, report, scope, proof = compile_fixture(tmp_path, source)
    validate_checked_execution_plans(report)
    route, = [r for member in proof["members"] for r in member["routes"] if "specification_entry" in r]
    phase = route["specification_entry"]
    edit, = [e for e in report["source_edits"] if [e["first_line"], e["last_line"]] == phase["source_span"]]
    text = edit["replacement"]
    if unmapped:
        assert phase["bindings"] == []
        assert phase["runtime"] == "fort_scope_observe_all_v1"
        assert "fort_specification_buffers" not in text
    else:
        assert [r["resource"] for r in phase["bindings"]] == [
            "ancestor_fixture::tile", "ancestor_fixture::tile_y", "ancestor_fixture::tile_z"]
        for row in phase["bindings"]:
            assert route["handle_forwarding"][str(row["callee_slot"])] == row["caller_slot"]
        assert "fort_specification_buffers(3)" in text
        assert "c_loc(fort_specification_buffers), 3_c_size_t)" in text
    assert "scratch" not in str(scope["resource_bindings"])
