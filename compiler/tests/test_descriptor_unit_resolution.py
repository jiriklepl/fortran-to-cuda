"""Bounded original semantic units, without eager transitive name expansion."""

from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_windows import _Check, prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.source_functions import _validate_resolution, _walk
from compiler.ir import CompilationError

ENTRY = "application::advance"
ROOT = "argument::a"


def fixture(tmp_path, body="a(1)=a(1)+1.0_8", *, imports="", declarations="",
            children="", sources=(), operations=256, procedures=64, selected_count=None):
    paths = []
    for index, source in enumerate(sources):
        path = tmp_path / f"dependency{index}.f90"
        path.write_text(source)
        paths.append(path)
    path = tmp_path / "application.f90"
    path.write_text(f"""module application
{imports}
contains
subroutine advance(a,n)
real(8),allocatable,intent(inout)::a(:)
integer,intent(in)::n
integer::i
{declarations}
{body}
end subroutine
{children}
end module
""")
    paths.append(path)
    analysis = SourceEffects(paths, operations=operations, procedures=procedures)
    graph = analysis.structure(ENTRY)
    originals = tuple(node for node in analysis.routines[ENTRY].execution.content
                      if type(node).__name__ != "Comment")
    selected = originals if selected_count is None else originals[:selected_count]
    return analysis, tuple(graph.node_id(node) for node in selected), originals, paths


def prove(analysis, selected):
    return prove_descriptor_window(analysis, ENTRY, selected, (ROOT,))


def constants(count):
    return "module values\n" + "\n".join(
        f"real(8),parameter::v{index}=1.0_8" for index in range(count)) + "\nend module\n"


def test_large_sequence_spends_per_unit_and_aggregate_resolution_budgets(tmp_path):
    body = "\n".join(f"a(1)=v{index}+v{index+1}" for index in range(0, 280, 2))
    analysis, selected, originals, _ = fixture(tmp_path, body, imports="use values", sources=(constants(280),))
    # The removed eager whole-selected-subtree query really exceeds the old
    # lookup cap. Every actual original assignment is independently bounded.
    with pytest.raises(CompilationError, match="resolution budget exhausted"):
        _validate_resolution(analysis, analysis.routines[ENTRY].scope, _walk(originals))
    record = prove(analysis, selected).public()
    resolution = record["resolution_validation"]
    assert record["schema_version"] == 5
    assert record["source_operations"] == 140
    assert resolution["units"] == 140
    assert resolution["lookup_transitions"] > 256
    assert resolution["syntax_entries"] > 256
    assert resolution["work_entries"] == resolution["lookup_transitions"] + resolution["syntax_entries"]
    assert record["source_traversal_entries"] > resolution["work_entries"]
    assert resolution["per_unit_lookup_limit"] == 256
    assert resolution["depth_limit"] == 8
    assert resolution["procedure_limit"] == 64
    assert resolution["aggregate_source_scan_limit"] == 65536
    assert not record["execution_authorized"]
    assert not record["entry_definition_authority"]


def test_one_large_assignment_keeps_target_and_all_operands_in_one_unit(tmp_path):
    body = "a(1)=" + "+".join(f"v{index}" for index in range(260))
    analysis, selected, _, _ = fixture(tmp_path, body, imports="use values", sources=(constants(260),))
    with pytest.raises(CompilationError, match="resolution budget exhausted"):
        prove(analysis, selected)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_call_target_and_all_original_actuals_form_one_unit(tmp_path):
    arguments = ",".join(f"v{index}" for index in range(130))
    names = ",".join(f"x{index}" for index in range(130))
    child = f"subroutine consume({names})\nreal(8),intent(in)::{names}\ncontinue\nend subroutine"
    analysis, selected, _, _ = fixture(tmp_path, f"call consume({arguments})", children=child,
                                        imports="use values", sources=(constants(130),))
    with pytest.raises(CompilationError, match="resolution budget exhausted"):
        prove(analysis, selected)


def test_counted_header_all_control_operands_form_one_unit(tmp_path):
    count = 130
    source = "module values\n" + "\n".join(f"integer,parameter::v{i}=1" for i in range(count)) + "\nend module"
    bound = "+".join(f"v{index}" for index in range(count))
    analysis, selected, _, _ = fixture(tmp_path, f"do i=1,{bound}\na(i)=1.0_8\nend do",
                                        imports="use values", sources=(source,))
    with pytest.raises(CompilationError, match="resolution budget exhausted"):
        prove(analysis, selected)


def test_aggregate_scan_counter_includes_syntax_and_lookup_transitions(tmp_path):
    analysis, _, originals, _ = fixture(tmp_path)
    checker = _Check(analysis)
    # Put the existing shared budget one entry below its limit. Both real
    # syntax and namespace transitions must spend it; no per-unit reset occurs.
    checker.nodes = 65535
    with pytest.raises(CompilationError, match="original-source traversal budget exhausted"):
        checker.resolution(analysis.routines[ENTRY], originals[0])
    assert checker.nodes == 65537
    assert checker.resolution_units == 1
    assert checker.resolution_syntax + checker.resolution_lookups == 2


def test_descriptor_operation_cap_remains_aggregate_across_child_calls(tmp_path):
    child = "subroutine increment(a)\nreal(8),intent(inout)::a(:)\na(1)=a(1)+1.0_8\nend subroutine"
    analysis, selected, _, _ = fixture(tmp_path, "\n".join("call increment(a)" for _ in range(8)),
                                        children=child, operations=8)
    with pytest.raises(CompilationError, match="source operation budget exhausted"):
        prove(analysis, selected)


@pytest.mark.parametrize(("levels", "accepted"), [(6, True), (8, False)])
def test_import_resolution_depth_cap_is_unchanged(tmp_path, levels, accepted):
    sources = ["module chain0\nreal(8),parameter::value=1.0_8\nend module"]
    sources.extend(f"module chain{i}\nuse chain{i-1}\nend module" for i in range(1, levels))
    analysis, selected, _, _ = fixture(tmp_path, "a(1)=value", imports=f"use chain{levels-1}", sources=sources)
    if accepted:
        assert prove(analysis, selected).public()["resolution_validation"]["depth_limit"] == 8
    else:
        with pytest.raises(CompilationError, match="resolution budget exhausted"):
            prove(analysis, selected)


@pytest.mark.parametrize("mutation", ["rename", "private", "unknown_import", "shadow", "binding", "owner", "source"])
def test_original_resolution_and_source_mutations_invalidate_units(tmp_path, mutation):
    sources = ("module values\nreal(8),parameter::value=1.0_8\nend module",)
    analysis, selected, _, paths = fixture(tmp_path, "a(1)=renamed", imports="use values,only:renamed=>value",
                                            sources=sources)
    proof = prove(analysis, selected)
    scope = analysis.modules["application"]
    if mutation == "rename":
        scope.imports["renamed"] = ("values", "other")
    elif mutation == "private":
        analysis.modules["values"].access["value"] = False
    elif mutation == "unknown_import":
        scope.imports["renamed"] = ("unknown", "value")
    elif mutation == "shadow":
        scope.bindings["renamed"] = copy(analysis.modules["values"].bindings["value"])
    elif mutation == "binding":
        analysis.modules["values"].bindings["value"] = copy(analysis.modules["values"].bindings["value"])
    elif mutation == "owner":
        analysis.modules["values"].bindings["value"].declaring_scope = scope
    else:
        paths[0].write_text(paths[0].read_text().replace("1.0_8", "2.0_8"))
    with pytest.raises(CompilationError, match="authority|source.*changed|input.*changed"):
        proof.validate(analysis, ENTRY, selected)
    with pytest.raises(CompilationError, match="authority|source.*changed|input.*changed"):
        prove(analysis, selected)


@pytest.mark.parametrize("mutation", ["access", "binding", "owner", "rename"])
def test_unused_requested_module_root_authenticates_original_reexport_visibility(tmp_path, mutation):
    sources = ("module storage\nreal(8),allocatable::field(:)\nend module",
               "module facade\nuse storage,only:exposed=>field\nend module")
    analysis, selected, _, _ = fixture(tmp_path, "a(1)=2.0_8", imports="use facade,only:slot=>exposed", sources=sources)
    proof = prove_descriptor_window(analysis, ENTRY, selected, (ROOT,), module_roots=("storage::field",))
    assert proof.resource_state("storage::field")["original_visible_spelling"] == "slot"
    if mutation == "access":
        analysis.modules["facade"].access["exposed"] = False
    elif mutation == "binding":
        analysis.modules["storage"].bindings["field"] = copy(analysis.modules["storage"].bindings["field"])
    elif mutation == "owner":
        analysis.modules["storage"].bindings["field"].declaring_scope = analysis.modules["facade"]
    else:
        analysis.modules["facade"].imports["exposed"] = ("storage", "other")
    with pytest.raises(CompilationError, match="authority|originally visible|original numeric"):
        proof.validate(analysis, ENTRY, selected)


def test_unsupported_excluded_tail_is_not_an_executable_lookup_unit(tmp_path):
    analysis, selected, _, _ = fixture(tmp_path, "a(1)=2.0_8\ncall missing(a)", selected_count=1)
    record = prove(analysis, selected).public()
    assert record["source_operations"] == 1
    assert record["resolution_validation"]["units"] == 1
    assert not record["prefix_effects_proved"]
    assert record["requires_close_before_excluded_tail"]


def test_complete_native_child_still_checks_unsupported_tail(tmp_path):
    child = "subroutine child(a)\nreal(8),intent(inout)::a(:)\na(1)=2.0_8\ncall missing(a)\nend subroutine"
    analysis, selected, _, _ = fixture(tmp_path, "call child(a)", children=child)
    with pytest.raises(CompilationError, match="unknown selected storage|boundary|source calls"):
        prove(analysis, selected)


def test_true_child_declaration_bounds_remain_original_lookup_units(tmp_path):
    child = """subroutine child(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
real(8)::scratch(n)
scratch(1)=a(1)
a(1)=scratch(1)
end subroutine"""
    analysis, selected, _, _ = fixture(tmp_path, "call child(a,n)", children=child)
    proof = prove(analysis, selected)
    record = proof.public()
    assert record["source_call_closure"].keys() == {"application::child"}
    assert record["resolution_validation"]["units"] >= 4  # Call, bound and assignments.
    binding = analysis.routines["application::child"].scope.bindings["n"]
    binding.declaring_scope = analysis.modules["application"]
    with pytest.raises(CompilationError, match="authority"):
        proof.validate(analysis, ENTRY, selected)


@pytest.mark.parametrize("declaration", [
    "real(width),intent(inout)::a(:)",
    "real(8),intent(inout)::a(:)\ninteger,parameter::local_width=width",
    "real(8),intent(inout)::a(:)\ninteger::local_width\nparameter(local_width=width)",
])
def test_child_kind_and_constant_expressions_authenticate_original_imports(tmp_path, declaration):
    sources = ("module kinds\ninteger,parameter::width=8\nend module",)
    child = f"subroutine child(a)\n{declaration}\na(1)=1.0_8\nend subroutine"
    analysis, selected, _, _ = fixture(tmp_path, "call child(a)", children=child,
                                        imports="use kinds,only:width", sources=sources)
    proof = prove(analysis, selected)
    analysis.modules["kinds"].bindings["width"].declaring_scope = analysis.modules["application"]
    with pytest.raises(CompilationError, match="original resolution authority changed"):
        proof.validate(analysis, ENTRY, selected)
    with pytest.raises(CompilationError, match="original resolution authority changed"):
        prove(analysis, selected)


def test_child_definition_spellings_are_not_an_eager_resolution_unit(tmp_path):
    declarations = "\n".join(f"real(8),parameter::unused{index}=1.0_8" for index in range(280))
    child = f"subroutine child(a)\nreal(8),intent(inout)::a(:)\n{declarations}\na(1)=1.0_8\nend subroutine"
    analysis, selected, _, _ = fixture(tmp_path, "call child(a)", children=child)
    public = prove(analysis, selected).public()
    assert public["source_operations"] == 2
    assert public["resolution_validation"]["units"] > 256
    assert public["resolution_validation"]["lookup_transitions"] < 256


def test_diamond_and_repeated_children_reuse_only_complete_original_closures(tmp_path):
    children = """subroutine leaf(a)
real(8),intent(inout)::a(:)
a(1)=a(1)+1.0_8
end subroutine
subroutine left(a)
real(8),intent(inout)::a(:)
call leaf(a)
end subroutine
subroutine right(a)
real(8),intent(inout)::a(:)
call leaf(a)
end subroutine"""
    analysis, selected, _, _ = fixture(tmp_path, "call left(a)\ncall right(a)\ncall leaf(a)", children=children)
    record = prove(analysis, selected).public()
    assert set(record["source_call_closure"]) == {"application::left", "application::right", "application::leaf"}
    assert record["source_operations"] == 6
    assert record["resolution_validation"]["procedure_limit"] == 64


def test_complete_registered_readonly_function_validation_spends_aggregate_budget(tmp_path):
    helper = """module arithmetic
contains
pure function evaluate(x) result(r)
real(8),intent(in)::x
real(8)::r
r=x*x
end function
end module"""
    analysis, selected, originals, _ = fixture(tmp_path, "a(1)=evaluate(value)", imports="use arithmetic",
                                                declarations="real(8),parameter::value=2.0_8",
                                                sources=(helper,))
    summary = analysis.summarize(ENTRY)
    assert summary["native_function_requirements"], summary["reasons"]
    proof, = analysis._native_functions.values()
    expression = next(node for node in walk(originals) if node is proof._expression)
    checker = _Check(analysis)
    proof.validate(analysis, ENTRY, expression, resolution_tick=checker.resolution_tick)
    assert checker.resolution_syntax > len(tuple(_walk(expression)))  # Complete child body, too.
    assert checker.resolution_lookups > 0
    assert checker.nodes == checker.resolution_work
    checker.nodes = 65535
    with pytest.raises(CompilationError, match="original-source traversal budget exhausted"):
        proof.validate(analysis, ENTRY, expression, resolution_tick=checker.resolution_tick)
    record = prove(analysis, selected).public()
    assert len(record["native_source_function_requirements"]) == 1


def test_inherited_configured_procedure_limit_is_not_raised(tmp_path):
    child = "subroutine child(a)\nreal(8),intent(inout)::a(:)\na(1)=a(1)+1.0_8\nend subroutine"
    analysis, selected, _, _ = fixture(tmp_path, "call child(a)", children=child, procedures=1)
    with pytest.raises(CompilationError, match="direct call closure budget exhausted"):
        prove(analysis, selected)
