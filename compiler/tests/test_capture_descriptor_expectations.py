"""External physical coverage can require an exact original descriptor."""

from copy import deepcopy

import pytest

from compiler.ir import CompilationError
from compiler.scopes.registration import expected_descriptor_conditions
from compiler.tests.test_collective_source_scopes import generate as collective_generate
from compiler.tests.test_lexical_source_owner import MODULE_SOURCE, NATIVE_METADATA_SOURCE, SOURCE, emit
from compiler.tests.test_module_reached_windows import MODULE_ROOTS, module_fixture, original_loops
from compiler.tests.test_ordinary_numerical_regions import ordinary_numerical_fixture
from compiler.tests.test_source_scopes import FACT

EXPECTED = {"lower_bounds": [-2], "extents": [12]}


@pytest.mark.parametrize("expected", [
    None, {}, {"lower_bounds": [-4]}, {"lower_bounds": [-4], "extents": [12], "extra": 1},
    {"lower_bounds": [-4, -4], "extents": [12]},
    {"lower_bounds": [True], "extents": [12]}, {"lower_bounds": [-4], "extents": [False]},
    {"lower_bounds": [-(2**63) - 1], "extents": [12]},
    {"lower_bounds": [2**63], "extents": [12]},
    {"lower_bounds": [-4], "extents": [-1]}, {"lower_bounds": [-4], "extents": [2**63]},
])
def test_malformed_expected_descriptor_cannot_authorize_capture(tmp_path, expected):
    _paths, builder = ordinary_numerical_fixture(tmp_path)
    builder.facts["captures"]["argument::a"]["expected_descriptor"] = expected
    with pytest.raises(CompilationError, match="expected capture descriptor"):
        builder.capture_fact(builder.entry.scope.bindings["a"])
    assert not builder.outputs
    assert not builder.edits


def test_signed64_endpoint_is_rendered_without_out_of_range_positive_literal(tmp_path):
    expected = {"lower_bounds": [-(2**63)], "extents": [2**63 - 1]}
    _paths, builder = ordinary_numerical_fixture(tmp_path, captures={
        "argument::a": {"expected_descriptor": expected}})
    fact = builder.capture_fact(builder.entry.scope.bindings["a"])
    lower, extent = expected_descriptor_conditions(fact, 1, "a")
    assert "(-9223372036854775807_c_int64_t - 1_c_int64_t)" in lower
    assert "9223372036854775808_c_int64_t" not in lower
    assert "9223372036854775807_c_int64_t" in extent


def test_descriptor_expectation_guards_original_layout_without_definition_credit(tmp_path):
    capture = {**FACT, "initialized": "none", "expected_descriptor": EXPECTED}
    path, outputs, report = emit(tmp_path, captures={"argument::out": capture})
    owner, = report["scopes"]
    assert report["capture_descriptor_expectations"]["schema_version"] == 1
    assert report["capture_descriptor_expectations"]["definition_authority"] is False
    resource = next(row for row in owner["resource_bindings"]["resources"] if row["resource"] == "argument::out")
    assert resource["expected_descriptor"] == EXPECTED
    assert resource["descriptor_expectation_guard"]["definition_authority"] is False
    assert capture["initialized"] == "none"
    text = outputs[report["sources"][str(path)]["replacement"]]
    check = text.index("lbound(out, 1, kind=c_int64_t) == -2_c_int64_t")
    size_check = text.index("size(out, 1, kind=c_int64_t) == 12_c_int64_t", check)
    close = text.index("fort_scope_close(", size_check)
    register = text.index("fort_scope_register(", close)
    assert text.index("is_contiguous(out)") < check < size_check < close < register
    # The producer has already been emitted; failure closes only this later
    # unit and reaches its unchanged original consumer once.
    assert "fort_scope_register(" in text[:check]
    assert "b(i)=2*a(i)+real(i,8)" not in text[check:]
    assert text[close:register].count("exit fort_reached_") == 1
    assert owner["reopen_after_boundary"] is False


def test_module_actual_guard_precedes_association_and_does_not_check_rebased_dummy(tmp_path):
    path, outputs, report = emit(tmp_path, MODULE_SOURCE, captures={
        "argument::b": {**FACT, "expected_descriptor": EXPECTED}})
    owner, = report["scopes"]
    child, = owner["module_coordinators"]
    text = outputs[report["sources"][str(path)]["replacement"]]
    caller = text[text.index("subroutine step"):]
    check = caller.index("fort_actuals_contiguous = lbound(b, 1, kind=c_int64_t) == -2_c_int64_t")
    close = caller.index("fort_scope_close(", check)
    call = caller.index("call " + child["entry"], close)
    assert caller.index("fort_actuals_contiguous = is_contiguous(b)") < check < close < call
    assert "lbound(x, 1, kind=c_int64_t) == -2_c_int64_t" not in text
    assert child["resource_mappings"] == {"argument::x": "argument::b"}


def test_selected_unallocated_module_descriptor_is_not_inquired_before_allocation_guard(tmp_path):
    from compiler.frontend.reached_windows import prove_reached_window
    from compiler.scopes.reached_owner import ReachedWindowOwner

    path, builder = module_fixture(tmp_path)
    for root in MODULE_ROOTS:
        builder.facts["captures"][root]["expected_descriptor"] = {"lower_bounds": [-2], "extents": [12]}
    original_facts = deepcopy(builder.facts)
    selection = original_loops(builder)
    proof = prove_reached_window(builder.analysis, builder.entry.qualified, selection, MODULE_ROOTS, builder.facts)
    owner = ReachedWindowOwner(builder, proof, selection)
    assert owner.run()
    outputs, report = owner.builder.finish()
    text = outputs[report["sources"][str(path)]["replacement"]]
    for name in ("a", "b"):
        allocation = text.index(f"if (fort_can) fort_can = allocated({name})")
        inquiry = text.index(f"lbound({name}, 1, kind=c_int64_t) == -2_c_int64_t", allocation)
        assert "if (.not. fort_can) then" in text[allocation:inquiry]
        assert "if (fort_can) fort_can = lbound" in text[allocation:inquiry + 12]
    assert builder.facts == original_facts


@pytest.mark.parametrize("mutation", ["source", "expectation"])
def test_selected_descriptor_guard_cannot_use_stale_source_or_fact_authority(tmp_path, mutation):
    from compiler.frontend.reached_windows import prove_reached_window
    from compiler.scopes.reached_owner import ReachedWindowOwner

    path, builder = module_fixture(tmp_path)
    builder.facts["captures"][MODULE_ROOTS[0]]["expected_descriptor"] = deepcopy(EXPECTED)
    selection = original_loops(builder)
    proof = prove_reached_window(builder.analysis, builder.entry.qualified, selection, MODULE_ROOTS, builder.facts)
    if mutation == "source":
        path.write_text(path.read_text() + "! changed original source\n")
    else:
        builder.facts["captures"][MODULE_ROOTS[0]]["expected_descriptor"]["extents"][0] = 13
    with pytest.raises(CompilationError):
        ReachedWindowOwner(builder, proof, selection)
    assert not builder.outputs
    assert not builder.edits


def test_optional_array_remains_unsupported_without_early_descriptor_evaluation(tmp_path):
    source = SOURCE.replace("real(8),intent(in)::a(-2:)", "real(8),optional,intent(in)::a(-2:)")
    source = source.replace("visits=visits+1", "visits=visits+1\nif(present(a)) then").replace(
        "end subroutine", "endif\nend subroutine")
    path, outputs, report = emit(tmp_path, source, captures={
        "argument::a": {**FACT, "expected_descriptor": EXPECTED}})
    assert not report["scopes"]
    assert not report["source_edits"]
    assert all("lbound(a, 1, kind=c_int64_t)" not in content.lower() for content in outputs.values())
    assert path.read_text() == source


def test_qualified_collective_explicitly_rejects_unenforced_expectation(tmp_path):
    def facts_change(facts):
        facts["captures"]["argument::a"]["expected_descriptor"] = {"lower_bounds": [1], "extents": [12]}

    _path, _outputs, report, _text = collective_generate(tmp_path, facts_change=facts_change)
    assert not report["scopes"]
    assert any("expected capture descriptor" in row["reason"]
               and "qualified collective owner is unsupported" in row["reason"] for row in report["boundaries"])


def test_opaque_native_metadata_cannot_advertise_an_unenforced_descriptor_guard(tmp_path):
    _path, _outputs, report = emit(tmp_path, NATIVE_METADATA_SOURCE, captures={
        "local_owner::metadata": {**FACT, "expected_descriptor": EXPECTED}})
    owner, = report["scopes"]
    assert any("expected capture descriptor is unsupported for opaque native metadata" in row["reason"]
               for row in owner["boundaries"])
    assert all("descriptor_expectation_guard" not in row for row in owner["resource_bindings"]["resources"])
