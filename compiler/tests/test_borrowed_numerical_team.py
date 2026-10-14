"""Whole numerical helpers borrow one context across distinct original calls."""

import json
import re
import shutil
import subprocess
from copy import copy
from dataclasses import replace

import pytest

from compiler.ir import CompilationError
from compiler.scopes.region_dispatch import InlineRegions
from compiler.scopes.segments import fragment, original_roots
from compiler.tests.support.borrowed_numerical_team import (
    DUMMY_LOWER_BOUNDS,
    ENTRY,
    HELPER_SOURCE,
    INITIALIZER,
    OWNER_SOURCE,
    SCALER,
    EmittedBorrowedTeam,
    borrowed_team_fixture,
    borrowed_team_launch_counts,
    emit_borrowed_team_fixture,
)


def _workers(report):
    return {procedure["procedure"]: [variant["name"] for variant in procedure["variants"]
                                    if variant["role"] == "call_worker"]
            for procedure in report["implementation_variants"]["procedures"]
            if procedure["procedure"] in {INITIALIZER, SCALER}}


def _body(text, name):
    lower = text.lower()
    first = lower.index("subroutine " + name.lower() + "(")
    last = lower.index("end subroutine " + name.lower(), first)
    return lower[first:last]


def _imports(body, module):
    logical, pending = [], ""
    for line in body.splitlines():
        current = pending + line.strip().lstrip("& ")
        if current.endswith("&"):
            pending = current[:-1]
        else:
            logical.append(current)
            pending = ""
    return {name.strip().split("=>")[0].strip() for line in logical
            if line.startswith("use " + module + ", only:")
            for name in line.split("only:", 1)[1].split(",")}


def _explicit_shared_source():
    helper = HELPER_SOURCE.replace("private(plane)\ndo plane=1,size(values,3)\nvalues(:,:,plane)=seed",
        "private(plane) shared(values,seed) default(none)\ndo plane=1,size(values,3)\nvalues(:,:,plane)=seed")
    return helper.replace("private(plane)\ndo plane=1,size(values,3)\nvalues(:,:,plane)=values",
        "private(plane) shared(values,factor) default(none)\ndo plane=1,size(values,3)\nvalues(:,:,plane)=values")


@pytest.fixture(scope="module")
def borrowed_team_emission(tmp_path_factory):
    paths, outputs, report, builder = emit_borrowed_team_fixture(
        tmp_path_factory.mktemp("borrowed_numerical_team"))
    borrowed_team_launch_counts(builder, report)
    return EmittedBorrowedTeam.capture(paths, outputs, report)


@pytest.fixture(scope="module")
def explicit_shared_emission(tmp_path_factory):
    paths, outputs, report, _builder = emit_borrowed_team_fixture(
        tmp_path_factory.mktemp("borrowed_explicit_shared"), helper_source=_explicit_shared_source())
    return EmittedBorrowedTeam.capture(paths, outputs, report)


def test_one_original_team_worker_is_reused_for_three_distinct_actuals(borrowed_team_emission):
    report = borrowed_team_emission.report()
    owner, = report["scopes"]
    children = {child["procedure"]: child for child in report["borrowed_source_coordinators"]}
    assert set(children) == {INITIALIZER, SCALER}
    assert not owner["boundaries"], owner["boundaries"]
    assert not owner["module_coordinators"]
    assert set().union(*(set(child["gpu_leaves"]) for child in children.values())) <= set(owner["gpu_leaves"])
    assert any(leaf.startswith(ENTRY + "#") for leaf in owner["gpu_leaves"])
    assert all(len(names) == 1 for names in _workers(report).values())
    for child in children.values():
        assert child["ownership"]["close_count"] == 0
        assert len(child["gpu_leaves"]) == 1
    assert children[INITIALIZER]["requirements"]["whole_initialized_resources"] == []
    assert children[SCALER]["requirements"]["whole_initialized_resources"] == ["argument::values"]
    assert children[INITIALIZER]["ordered_definitions"]["whole_on_return"] == ["argument::values"]


def test_original_call_maps_preserve_entry_bounds_and_scalar_association(borrowed_team_emission):
    report = borrowed_team_emission.report()
    calls = [call for call in report["resolved_calls"]
             if call["caller"] == ENTRY and call["procedure"] in {INITIALIZER, SCALER}]
    assert len(calls) == 6
    for procedure in (INITIALIZER, SCALER):
        selected = [call for call in calls if call["procedure"] == procedure]
        assert {call["resource_mapping"]["argument::values"] for call in selected} == {
            "argument::b", "argument::c", "argument::d"}
        for call in selected:
            mapping, = (item for item in call["resource_mappings"] if item["formal"] == "values")
            actual = mapping["actual"]
            assert tuple(int(bound.replace(" ", "")) for bound in
                         mapping["actual_descriptor"]["lower_bounds"]) == DUMMY_LOWER_BOUNDS[actual]
            assert mapping["formal_descriptor"]["lower_bounds"] == ["1", "1", "1"]
            assert mapping["storage"] == "whole"
    scales = [call for call in calls if call["procedure"] == SCALER]
    assert {call["actual_arguments"][1].lower() for call in scales} == {"scale_factor"}
    assert all(call["actual_evaluation"] == "original call position and guards; no analysis-time evaluation"
               for call in scales)
    text = borrowed_team_emission.source(1).lower().replace(" ", "")
    for coefficient in ("0.25_8", "0.5_8", "0.75_8"):
        assert text.count("scale_factor=gain+" + coefficient) == 1


def test_actual_participants_vote_before_query_and_native_alternative_runs_once(borrowed_team_emission):
    report = borrowed_team_emission.report()
    helper_text = borrowed_team_emission.source(0)
    for procedure, names in _workers(report).items():
        name, = names
        body = _body(helper_text, name)
        probe = body.index("fort_scope_numerical_environment_supported()")
        query = body.index("fort_inline_query(")
        execute = body.index("fort_inline_team_run(")
        begin = body.index("fort_scope_host_begin(")
        end = body.index("fort_scope_host_end(")
        native = ("values(:,:,plane)=seed" if procedure == INITIALIZER
                  else "values(:,:,plane)=values(:,:,plane)*factor")
        compact = body.replace(" ", "")
        assert compact.count(native) == 1
        original = compact.index(native)
        assert probe < query < execute
        assert compact.index("fort_scope_host_begin(") < original < compact.index("fort_scope_host_end(")
        assert begin < end
        assert body.count("!$omp parallel") == 1
        assert body.count("!$omp end parallel") == 1
        assert "!$omp atomic write" in body[:query]
        assert "!$omp barrier" in body[probe:query]
        assert "!$omp barrier" in body[query:execute]
        assert "intent(in) :: fort_context" in body
        assert "fort_scope_create(" not in body
        assert "fort_scope_close(" not in body
        assert "error stop" in body[execute:]
        assert "fort_scope_wait(" in body[execute:]
        assert {"c_size_t", "c_double"} <= _imports(body, "iso_c_binding")
        if procedure == INITIALIZER:
            forget = body.index("fort_scope_forget_definition(")
            assert forget < body.index("!$omp parallel")
            assert "fort_scope_forget_definition" in _imports(body, "fort_scoped_memory")
    owner_text = borrowed_team_emission.source(1).lower().replace(" ", "")
    assert owner_text.count("callieee_get_halting_mode(ieee_invalid,original_mask)") == 1
    assert owner_text.count("work(i,j,k)=work(i,j,k)+a(i,j,k)+1.0_8") == 1
    assert owner_text.count("work(i,j,k)=work(i,j,k)+b(i-2,j-2,k-2)+c(i-5,j-5,k-5)+d(i+4,j-8,k+4)") == 1


@pytest.mark.parametrize("mutation", ["source", "binding", "configuration"])
def test_borrowed_team_retains_fresh_source_and_configuration_authority(tmp_path, mutation):
    paths, _outputs, _report, builder = emit_borrowed_team_fixture(tmp_path)
    child = builder.coordinators[SCALER]
    if mutation == "source":
        prior_outputs = dict(builder.outputs)
        paths[0].write_text(paths[0].read_text() + "! original source changed\n")
        child.scope._validate()
        with pytest.raises(CompilationError, match="source changed"):
            builder.finish()
        assert builder.outputs == prior_outputs
    elif mutation == "binding":
        scope = child.routine.scope
        binding = copy(scope.bindings["factor"])
        binding.attributes |= {"volatile"}
        with pytest.raises(CompilationError):
            scope.bindings["factor"] = binding
    else:
        child.builder.config = replace(child.builder.config, host_threads=2)
        with pytest.raises(CompilationError):
            child.scope._validate()


def test_partial_out_helper_cannot_grant_whole_definition(tmp_path):
    initializer = HELPER_SOURCE.replace("do plane=1,size(values,3)", "do plane=1,size(values,3)-1", 1)
    paths, builder = borrowed_team_fixture(tmp_path, helper_source=initializer)
    outputs, report = builder.run()
    owner, = report["scopes"]
    assert owner["boundaries"], report
    decline, = (item for item in report["source_coordinator_boundaries"] if item["procedure"] == INITIALIZER)
    assert "native fallback requires undefined INTENT(OUT) input" in decline["reason"]
    child = builder.coordinators.get(INITIALIZER)
    if child is not None:
        assert "argument::values" not in child.definition_summary.whole_on_return
    assert builder.facts["captures"]["argument::b"]["initialized"] == "none"
    original = paths[1].read_text()
    assert original == OWNER_SOURCE
    emitted = outputs[report["sources"][str(paths[1])]["replacement"]].lower()
    assert emitted.count("work(i,j,k)=work(i,j,k)+a(i,j,k)+1.0_8") == 1


def test_conditional_scalar_writes_cannot_grant_whole_out_definition(tmp_path):
    initializer = HELPER_SOURCE.replace("integer::plane\n", "integer::plane,row,column\n", 1)
    initializer = initializer.replace("!$omp parallel do private(plane)",
                                      "!$omp parallel do private(plane,row,column)", 1)
    initializer = initializer.replace("values(:,:,plane)=seed", """do row=1,size(values,2)
do column=1,size(values,1)
if(seed>=0) values(column,row,plane)=seed
enddo
enddo""", 1)
    _paths, builder = borrowed_team_fixture(tmp_path, helper_source=initializer)
    local = copy(builder)
    local.entry = builder.analysis.routines[INITIALIZER]
    local.inline = InlineRegions(local)
    group, = local.inline.grouped_nodes(local.entry.execution.content)
    originals = original_roots(local.inline.original_selection(group))
    native = fragment(local, originals, kind="joined native OpenMP")
    assert not native.overwrites
    assert native.summary["guaranteed_whole_overwrites"] == []
    _outputs, report = builder.run()
    assert builder.coordinators.get(INITIALIZER) is None
    assert not _workers(report).get(INITIALIZER)
    assert builder.facts["captures"]["argument::b"]["initialized"] == "none"
    assert report["scopes"][0]["boundaries"]


def test_original_private_and_explicit_shared_clauses_do_not_conflict(explicit_shared_emission):
    report = explicit_shared_emission.report()
    owner, = report["scopes"]
    assert not owner["boundaries"]
    text = explicit_shared_emission.source(0)
    for procedure, names in _workers(report).items():
        name, = names
        body = _body(text, name)
        header = body[body.index("!$omp parallel"):body.index("!$omp master")]
        assert header.count("private(plane)") == 1
        assert "default(none)" in header
        shared = [name.strip() for clause in re.findall(r"shared\(([^()]*)\)", header)
                  for name in clause.split(",")]
        assert shared.count("values") == 1
        assert shared.count("seed" if procedure == INITIALIZER else "factor") == 1
        assert "plane" not in shared


def test_public_fortran_sources_compile_with_original_data_sharing_clauses(tmp_path, explicit_shared_emission):
    compiler = shutil.which("gfortran-15") or shutil.which("gfortran")
    if compiler is None:
        pytest.skip("Fortran frontend required for emitted source syntax checks")
    report = explicit_shared_emission.report()
    output, build = tmp_path / "sources", tmp_path / "modules"
    output.mkdir()
    build.mkdir()
    for name, text in explicit_shared_emission.artifacts:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    original_order = {report["sources"][path]["replacement"]: index
                      for index, (path, _source) in enumerate(explicit_shared_emission.original_sources)}
    commands = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        records = [record for record in report["build_sources"]
                   if record["role"] == role and record["language"] == "fortran"]
        if role == "original_source":
            records.sort(key=lambda record: original_order[record["path"]])
        for record in records:
            command = [compiler, "-fsyntax-only", "-fopenmp", "-std=f2018", "-O0", "-I", str(build),
                       "-I", str(output), str(output / record["path"])]
            result = subprocess.run(command, cwd=build, capture_output=True, text=True, timeout=30, check=False)
            commands.append({"role": role, "path": record["path"], "argv": command,
                             "cwd": str(build), "exit_code": result.returncode,
                             "stdout": result.stdout, "stderr": result.stderr})
            (tmp_path / "fortran-syntax-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            assert result.returncode == 0, result.stdout + result.stderr
    assert {command["role"] for command in commands} == {"common_runtime", "shared_entry", "original_source"}


@pytest.mark.parametrize("alias", ["plane_coefficient", "fort_team_omp_level"], ids=["original-import", "generated-collision"])
def test_original_imports_remain_available_without_shadowing_generated_aliases(tmp_path, alias):
    constants = """module amber_constants
implicit none
integer,parameter::normalization=2
end module
"""
    helper = constants + HELPER_SOURCE.replace("subroutine scale_planes(values,factor)\n",
        "subroutine scale_planes(values,factor)\nuse amber_constants,only:" + alias + "=>normalization\n")
    helper = helper.replace("values(:,:,plane)=values(:,:,plane)*factor",
                            "values(:,:,plane)=values(:,:,plane)*factor*real(" + alias + ",8)")
    paths, outputs, report, _builder = emit_borrowed_team_fixture(tmp_path, helper_source=helper)
    children = {child["procedure"]: child for child in report["borrowed_source_coordinators"]}
    if alias == "plane_coefficient":
        assert SCALER in children
        assert not report["scopes"][0]["boundaries"]
        name, = _workers(report)[SCALER]
        text = outputs[report["sources"][str(paths[0])]["replacement"]]
        body = _body(text, name).replace(" ", "")
        assert "useamber_constants,only:plane_coefficient=>normalization" in body
        assert body.count("values(:,:,plane)=values(:,:,plane)*factor*real(plane_coefficient,8)") == 1
    else:
        assert SCALER not in children
        assert any("generated name conflicts with original reference: fort_team_omp_level" in item["reason"]
                   for item in report["boundaries"])
        assert SCALER not in _workers(report)
