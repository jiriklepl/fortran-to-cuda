"""Combined numerical groups use the original participants and whole join."""

from copy import copy
from dataclasses import replace

import pytest

from compiler.ir import CompilationError
from compiler.scopes.combined_team import (
    CombinedNumericalTeam,
    _check_generated_names,
    _narrow_runtime_imports,
    _source_partition,
    is_combined_group,
)
from compiler.scopes.segments import grouped_nodes
from compiler.tests.test_reached_numerical_effects import (
    CHILD,
    CHILD_SOURCE,
    MATH_SOURCE,
    emit_reached_numerical_fixture,
    reached_numerical_fixture,
)

ENVIRONMENT_USE = "use,intrinsic::ieee_exceptions,only:ieee_set_halting_mode,ieee_invalid\n"
ENVIRONMENT_TEAM = """!$omp parallel
call ieee_set_halting_mode(ieee_invalid,.false.)
!$omp end parallel
"""


def emit_combined_team_fixture(directory, **options):
    """Keep the native environment setter and force original-team admission."""
    source = options.pop("child_source", CHILD_SOURCE)
    assert "implicit none\n" in source
    assert "i=0\n" in source
    source = source.replace("implicit none\n", ENVIRONMENT_USE + "implicit none\n", 1)
    source = source.replace("i=0\n", "i=0\n" + ENVIRONMENT_TEAM, 1)
    return emit_reached_numerical_fixture(directory, child_source=source, **options)


def combined_units(owner):
    return tuple(unit for unit in owner.members[CHILD].units if isinstance(unit.scope, CombinedNumericalTeam))


def emitted_child(paths, outputs, report):
    return outputs[report["sources"][str(paths[1])]["replacement"]]


def team_text(text):
    """The original setter has no generated environment-vote marker."""
    body = text.index("call cell_formula(g,s,a(i),i)")
    marker = text.rfind("fort_team_environment_supported = .true.", 0, body)
    assert marker >= 0
    first = text.rfind("!$omp parallel", 0, marker)
    last = text.index("!$omp end parallel", marker) + len("!$omp end parallel")
    assert first >= 0
    return text[first:last]


def test_combined_team_requires_actual_participants_and_keeps_native_environment(tmp_path):
    paths, outputs, report, owner = emit_combined_team_fixture(tmp_path)
    units = combined_units(owner)
    assert units
    assert any(unit.scope.prepared.requires_original_team_environment_preflight for unit in units)
    text = emitted_child(paths, outputs, report)
    assert paths[1].read_text().count(ENVIRONMENT_TEAM) == 1
    assert "!$omp parallel\ncall ieee_set_halting_mode" in text
    assert not owner.members[CHILD].boundaries
    for unit in units:
        scope = unit.scope
        assert is_combined_group(scope.builder, scope.original_nodes)
        assert not scope.native
        assert not scope.guarded
        public = scope.public()["combined_numerical_team"]
        assert public["expected_omp_level"] == 1
        assert public["host_threads"] == 4
        assert not public["native_effects_authority"]
        assert not public["nested_team_created"]
        assert public["closed_is_absorbing"]
        assert "generated CPU" in public["worker_resource_failure"]
        assert public["completion"]["proof_identity"] == scope.prepared.completion_proof.identity


def test_combined_preparation_and_query_follow_the_all_participant_vote(tmp_path):
    paths, outputs, report, _owner = emit_combined_team_fixture(tmp_path)
    text = team_text(emitted_child(paths, outputs, report))
    reset = text.index("fort_team_environment_supported = .true.")
    qualification = text.index("fort_team_omp_level() /= 1")
    probe = text.index("fort_scope_numerical_environment_supported() == 0")
    veto = text.index("fort_team_environment_supported = .false.")
    block = text.index(": block", veto)
    descriptor = text.index("is_contiguous(", block)
    geometry = text.index("fort_precondition_", descriptor)
    registration = text.index("fort_scope_register(", descriptor)
    reset_plan = text.index("fort_scope_plan_reset_mode(", registration)
    query = text.index("fort_inline_query(", registration)
    execution = text.index("fort_inline_team_run(", query)
    wait = text.index("fort_scope_wait(", execution)
    assert reset < qualification < probe < veto < block < descriptor < geometry < registration < query < execution < wait
    assert "if (fort_status == FORT_SCOPE_OK) then" in text[reset_plan:query]
    assert "!$omp atomic write\nfort_team_environment_supported = .false." in text
    assert text[probe:block].count("!$omp barrier") >= 1
    assert text[query:execution].count("!$omp barrier") >= 1
    assert "if (fort_native_ready) then" in text[:probe]
    assert "if (.not. fort_team_environment_supported) then" in text[block:descriptor]
    assert "error stop 'combined original team numerical execution failed'" in text
    assert "error stop 'combined original team completion failed'" in text
    assert text.count("!$omp parallel") == 1
    assert text.count("!$omp end parallel") == 1


def test_combined_partition_preserves_supported_original_clauses_and_native_body(tmp_path):
    source = CHILD_SOURCE.replace(
        "!$omp parallel do private(block,i,g,s) schedule(static)",
        "!$omp parallel do private(block,i,g,s) shared(a,b,n,tile) default(none) collapse(1) schedule(runtime)")
    paths, outputs, report, owner = emit_combined_team_fixture(tmp_path, child_source=source)
    unit, = (unit for unit in combined_units(owner) if unit.scope.prepared.extraction.numerical_helpers)
    public = unit.scope.public()["combined_numerical_team"]
    assert public["parallel_clauses"] == ["private(block,i,g,s)", "shared(a,b,n,tile)", "default(none)"]
    assert public["do_clauses"] == ["collapse(1)", "schedule(runtime)"]
    text = team_text(emitted_child(paths, outputs, report))
    assert "!$omp parallel private(block,i,g,s) shared(a,b,n,tile) default(none)" in text
    assert "!$omp do collapse(1) schedule(runtime)" in text
    assert text.count("call cell_formula(g,s,a(i),i)") == 1
    assert "do block=-3,n,tile\ndo i=block,min(block+tile-1,n)" in text
    assert "fort_team_buffer_" in text
    assert "if (fort_team_owned) then" in emitted_child(paths, outputs, report)


@pytest.mark.parametrize("authority", ["source", "binding", "configuration", "partition"])
def test_combined_source_and_configuration_mutation_remain_hard(tmp_path, authority):
    paths, _outputs, _report, owner = emit_combined_team_fixture(tmp_path)
    scope = combined_units(owner)[0].scope
    if authority == "source":
        paths[1].write_text(paths[1].read_text() + "! changed source\n")
    elif authority == "binding":
        original = scope.builder.entry.scope.bindings["n"]
        scope.builder.entry.scope.bindings["n"] = replace(original, attributes=original.attributes | {"volatile"})
    elif authority == "configuration":
        scope.builder.config = replace(scope.builder.config, host_threads=2)
    else:
        scope.partition = {**scope.partition, "body": scope.partition["body"] + "! invented native body\n"}
    with pytest.raises(CompilationError):
        scope._validate()


def test_combined_partition_scan_limit_is_a_hard_failure(tmp_path, monkeypatch):
    _paths, _outputs, _report, owner = emit_combined_team_fixture(tmp_path)
    scope = combined_units(owner)[0].scope
    monkeypatch.setattr("compiler.scopes.combined_team.NATIVE_GROUP_SCAN_LIMIT", 1)
    with pytest.raises(CompilationError, match="source scan budget"):
        _source_partition(scope.builder, scope.original_nodes)


def test_combined_partial_selection_cannot_replace_registered_whole_join(tmp_path):
    _paths, _outputs, _report, owner = emit_combined_team_fixture(tmp_path)
    scope = combined_units(owner)[0].scope
    loop, = (node for node in scope.original_nodes if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    with pytest.raises(CompilationError):
        CombinedNumericalTeam(scope.builder, (loop,))


def test_combined_generated_names_cannot_shadow_original_references(tmp_path):
    _paths, _outputs, _report, owner = emit_combined_team_fixture(tmp_path)
    scope = next(unit.scope for unit in combined_units(owner) if unit.scope.prepared.extraction.numerical_helpers)
    with pytest.raises(CompilationError, match="generated name conflicts with original reference: n"):
        _check_generated_names(scope.original_nodes, ["integer :: n"], [], [], "generated_block")
    with pytest.raises(CompilationError, match="generated name conflicts with original reference: n"):
        _check_generated_names(scope.original_nodes, ["integer(c_int), parameter :: n = 1"], [], [], "generated_block")
    _check_generated_names(scope.original_nodes, ["integer :: n_extra"], [], [], "generated_block")
    _check_generated_names(scope.original_nodes, ["integer(n), parameter :: unrelated = 1"], [], [], "generated_block")


@pytest.mark.parametrize("alias", ["fort_team_context", "fort_mode"])
def test_combined_imported_alias_cannot_be_shadowed_by_generated_storage(tmp_path, alias):
    math = MATH_SOURCE.replace("implicit none\n", "implicit none\ninteger,parameter::numeric_bias=2\n", 1)
    child = CHILD_SOURCE.replace("use quartz_math,only:cell_formula=>profile_cell",
        "use quartz_math,only:cell_formula=>profile_cell," + alias + "=>numeric_bias")
    child = child.replace("b(i)=s+real(i,8)", "b(i)=s+real(i,8)+real(" + alias + ",8)")
    with pytest.raises(CompilationError, match="generated name conflicts with original reference: " + alias):
        emit_combined_team_fixture(tmp_path, math_source=math, child_source=child)


def test_combined_unused_shared_alias_cannot_rebind_to_generated_storage(tmp_path):
    math = MATH_SOURCE.replace("implicit none\n", "implicit none\ninteger::numeric_bias=2\n", 1)
    child = CHILD_SOURCE.replace("use quartz_math,only:cell_formula=>profile_cell",
        "use quartz_math,only:cell_formula=>profile_cell,fort_team_context=>numeric_bias")
    child = child.replace("private(block,i,g,s) schedule(static)",
        "private(block,i,g,s) shared(fort_team_context) schedule(static)")
    with pytest.raises(CompilationError, match="generated name conflicts with original reference: fort_team_context"):
        emit_combined_team_fixture(tmp_path, math_source=math, child_source=child)


def test_combined_implicit_end_cannot_drop_a_same_line_serial_sibling(tmp_path):
    child = CHILD_SOURCE.replace("enddo\nenddo\n!$omp end parallel do\nif(n>=-3)",
        "enddo\nenddo;middle_visits=middle_visits+1\nif(n>=-3)")
    _paths, builder, _proof, _originals, _provider = reached_numerical_fixture(tmp_path, child_source=child)
    local = copy(builder)
    local.entry = builder.analysis.routines[CHILD]
    group, = (node for node in grouped_nodes(local.entry.execution.content) if isinstance(node, tuple))
    with pytest.raises(CompilationError, match="semicolon source boundary"):
        _source_partition(local, group)


def test_combined_runtime_imports_introduce_only_required_exact_exports():
    imports = ["use fort_scoped_memory", "use iso_c_binding"]
    _narrow_runtime_imports(imports, ["integer(c_int) :: fort_status"], ["fort_status = FORT_SCOPE_OK"])
    assert imports == ["use fort_scoped_memory, only: fort_scope_ok", "use iso_c_binding, only: c_int"]
    assert all("fort_scope_execution" not in line for line in imports)
