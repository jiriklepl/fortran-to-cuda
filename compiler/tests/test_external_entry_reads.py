"""Trusted must-read prerequisites are separate from provider permissions."""
import json
import shutil
import subprocess
from copy import deepcopy

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.device_tasks import external_device_task
from compiler.frontend.external_entry_reads import (
    external_entry_read_requirements,
    provider_read_binding,
    requirement_scope,
)
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_external_device_tasks import digest, requalify
from compiler.tests.test_external_device_tasks_v2 import OPAQUE_SOURCE, opaque_fixture

CLAIM = "unconditional original entry reads before any same-root write under accepted query and layout predicates"
FLAGS = {"may_read_permissions_are_entry_requirements": False, "unreached_definition_authority": False,
         "halo_definition_authority": False, "alias_authority": False,
         "lifetime_authority": False, "execution_authority": False}


def qualified(tmp_path, source=OPAQUE_SOURCE):
    path, analysis, routine, node, task, manifest, implementation = opaque_fixture(tmp_path, source)
    provider = json.loads(manifest.read_text())
    provider["metadata_query"]["configuration_components"].append({"component": "extent", "kind": "int64", "count": 1})
    task["configuration"] = []
    manifest.write_text(json.dumps(provider))
    requalify(task, manifest)
    evidence = tmp_path / "reviewed-library-contract.json"
    evidence.write_text(json.dumps({"scope": "test-only explicit reviewed original source contract",
        "original": str(path), "source_sha256": digest(path),
        "claim": "original output=3*input requires all input elements on entry under matching query extents"}))
    required = {key: deepcopy(task["resources"][0]["effects"][0][key]) for key in ("lower", "upper")}
    required["formal"] = "input"
    q = {"version": 1, "kind": "qualified_original_external_entry_reads", "claim": CLAIM, "authority": deepcopy(FLAGS),
         "source_inputs": analysis.inputs.identity(), "task_scope": requirement_scope(task),
         "metadata_query": provider["metadata_query"], "provider_binding": deepcopy(provider_read_binding(provider)), "required_reads": [required],
         "layout_equalities": [{"component": "extent", "index": 1, "descriptor": {
             "descriptor": "size", "formal": name, "axis": 1, "offset": 0}} for name in ("input", "output")],
         "evidence": [{"path": str(evidence), "sha256": digest(evidence)}]}
    qualification = tmp_path / "entry-read-qualification.json"
    qualification.write_text(json.dumps(q))
    task["entry_read_requirements"] = {"path": str(qualification), "sha256": digest(qualification)}
    requalify(task, manifest)
    return path, analysis, routine, node, task, manifest, qualification, q


def rewrite(task, manifest, qualification, q):
    qualification.write_text(json.dumps(q))
    task["entry_read_requirements"]["sha256"] = digest(qualification)
    requalify(task, manifest)


def test_requires_distinct_trusted_entry_contract_and_retains_no_grants(tmp_path):
    _, analysis, routine, node, _, _, _, _ = qualified(tmp_path)
    task = external_device_task(analysis, routine, node)
    proof = external_entry_read_requirements(task)
    assert proof.validate(analysis, routine, node) is proof
    public = proof.public()
    assert public["canonical_resources"] == {"input": "argument::b"}
    assert all(public[key] is False for key in FLAGS)
    assert task.public()["definitions_authorized"] is False
    assert task.public()["source_effects"]["entry_definition_kills"] == ["output"]


@pytest.mark.parametrize("change", ["may_read", "conditional", "source", "query", "scope", "region", "out", "halo", "missing_axis", "duplicate_guard", "evidence", "nonboolean", "unhashable_formal", "query_artifact", "backend"])
def test_rejects_unqualified_or_changed_entry_prerequisites(tmp_path, change):
    _, analysis, routine, node, task, manifest, qualification, q = qualified(tmp_path)
    if change == "query_artifact":
        provider = json.loads(manifest.read_text())
        artifact = tmp_path / "changed-query.f90"
        artifact.write_text("! same typed query name, different implementation\n")
        provider["artifacts"] = {str(artifact): digest(artifact)}
        manifest.write_text(json.dumps(provider))
    elif change == "backend":
        q["provider_binding"]["pressure_backend"] = {"manifest": "/tmp/other-backend", "sha256": "0" * 64}
    elif change == "may_read":
        q["authority"]["may_read_permissions_are_entry_requirements"] = True
    elif change == "conditional":
        q["claim"] = "possibly read under a nested source predicate"
    elif change == "source":
        q["source_inputs"] = {}
    elif change == "query":
        q["metadata_query"]["procedure"] = "different_query"
    elif change == "scope":
        q["task_scope"] = "0" * 64
    elif change == "region":
        q["required_reads"][0]["lower"][0] = 1
    elif change == "out":
        q["required_reads"][0]["formal"] = "output"
    elif change == "halo":
        q["authority"]["halo_definition_authority"] = True
    elif change == "missing_axis":
        q["layout_equalities"].pop()
    elif change == "duplicate_guard":
        q["layout_equalities"].append(deepcopy(q["layout_equalities"][0]))
    elif change == "nonboolean":
        q["authority"]["halo_definition_authority"] = 0
    elif change == "unhashable_formal":
        q["layout_equalities"][0]["descriptor"]["formal"] = []
    else:
        q["evidence"][0]["sha256"] = "0" * 64
    rewrite(task, manifest, qualification, q)
    with pytest.raises(CompilationError):
        external_device_task(analysis, routine, node)


@pytest.mark.parametrize("change", ["call", "configuration", "qualification_file", "original_file"])
def test_retained_exact_external_token_rejects_changed_authority(tmp_path, change):
    path, analysis, routine, node, task, _, qualification, _ = qualified(tmp_path)
    proof = external_entry_read_requirements(external_device_task(analysis, routine, node))
    if change == "call":
        node.items[0].string = "other_transform"
    elif change == "configuration":
        task["configuration"].append({"component": "other", "kind": "int64", "values": [1]})
    elif change == "qualification_file":
        qualification.write_text("{}")
    else:
        path.write_text(path.read_text() + "! changed original\n")
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine, node)


def emitted(tmp_path, *, conditional=False):
    source = OPAQUE_SOURCE
    if conditional:
        source = source.replace("call original_transform(cache,b,a)", "if(n>0)then\ncall original_transform(cache,b,a)\nendif")
    path, _, _, _, task, _, _, _ = qualified(tmp_path, source)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): digest(path)},
             "captures": {"argument::" + name: {"storage": "stable", "escapes": False,
                 "allocation_changes": False, "initialized": "none"} for name in ("a", "b")}}
    builder = ScopeBuilder([path], "external_fixture::advance", facts=facts,
        options=CompilerOptions(), config=OffloadConfig(policy="sections", scope_execution="graph"),
        contracts={"external_fixture::advance": {"device_tasks": [task]}})
    outputs, report = builder.run()
    for relative, content in outputs.items():
        destination = tmp_path / "generated" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    (tmp_path / "generated-source.f90").write_text(outputs[report["sources"][str(path)]["replacement"]])
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2))
    return outputs[report["sources"][str(path)]["replacement"]], report, facts


def test_generated_order_retains_NONE_guards_create_admission_outkill_transaction(tmp_path):
    text, report, facts = emitted(tmp_path)
    assert all(fact["initialized"] == "none" for fact in facts["captures"].values())
    create = text.index("fort_external_status = fort_external_create(")
    success = text.index("if (fort_external_status == FORT_SCOPE_OK) then", create)
    admission = text.index("fort_external_status = fort_scope_admit_source_reads_v1(", success)
    kill = text.index("fort_external_status = fort_scope_forget_definition(", admission)
    execute = text.index("fort_external_status = fort_external_execute(", kill)
    assert create < success < admission < kill < execute
    assert text.index("fort_external_status = fort_external_metadata(") < create
    assert text.index("int(fort_external_config%extent,c_int64_t) ==") < create
    assert "external provider unavailable at original call" in text or report["scopes"]
    assert "if (.not. fort_external_done) then" in text
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    validate_checked_execution_plans(report)


def test_original_false_guard_remains_outside_admission(tmp_path):
    text, report, _ = emitted(tmp_path, conditional=True)
    lowered = text.lower()
    admission = lowered.index("fort_external_status = fort_scope_admit_source_reads_v1(")
    assert "n>0" in "".join(lowered[:admission].split())
    assert report["scope_count"] == 1


@pytest.mark.parametrize("change", ["record", "missing", "mapping", "authority"])
def test_strict_reader_rejects_changed_entry_prerequisite_even_after_outer_rehash(tmp_path, change):
    from elmm_pipeline.external_entry_read_record import validate_external_entry_reads

    _, analysis, routine, node, _, _, _, _ = qualified(tmp_path)
    task = external_device_task(analysis, routine, node).public()
    record = task["source_effects"]["entry_read_requirements"]
    ranks = {item["formal"]: item["formal_descriptor"]["rank"] for item in task["call"]["associations"]
             if item["formal_descriptor"]["rank"]}
    associations = task["call"]["associations"]
    validate_external_entry_reads(record, task, ranks, associations)
    if change == "record":
        record["qualification_record"]["required_reads"][0]["lower"][0] = 1
    elif change == "missing":
        record.pop("qualification")
    elif change == "mapping":
        record["canonical_resources"]["input"] = "argument::a"
    else:
        record["halo_definition_authority"] = True
    with pytest.raises(ValueError, match="external entry|exact record fields"):
        validate_external_entry_reads(record, task, ranks, associations)


def test_new_admission_uses_real_typed_Fortran_pointer_interface(tmp_path):
    from compiler.tests.test_external_device_tasks import PROVIDER_INTERFACE
    from compiler.tests.test_external_device_tasks_v2 import TYPES

    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    assert fortran is not None, "This qualified GNU fixture requires its pinned compiler"
    _, report, _ = emitted(tmp_path)
    build = tmp_path / "compile"
    build.mkdir()
    types, provider, query = (build / name for name in ("types.f90", "provider.f90", "query.f90"))
    types.write_text(TYPES)
    provider.write_text(PROVIDER_INTERFACE)
    query.write_text("""module external_metadata
use iso_c_binding
use external_types
use external_provider
implicit none
contains
integer(c_int) function metadata_query(config,d) result(status)
type(provider_config),intent(inout)::config
type(state_type),intent(in)::d
config%coefficient=d%scratch
config%extent=8
status=0
end function
end module
""")
    artifacts = [item for item in report["build_sources"] if item["language"] == "fortran"]
    paths = [types, *(tmp_path / "generated" / item["path"] for item in artifacts if item["role"] == "common_runtime"),
             provider, query, *(tmp_path / "generated" / item["path"] for item in artifacts if item["role"] != "common_runtime")]
    commands = []
    for index, path in enumerate(paths):
        command = [fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-J", str(build), "-I", str(build),
                   "-c", str(path), "-o", str(build / f"source-{index}.o")]
        result = subprocess.run(command, cwd=build, text=True, capture_output=True, timeout=30)
        (build / f"source-{index}.stderr").write_text(result.stderr)
        commands.append({"argv": command, "returncode": result.returncode, "source_sha256": digest(path)})
        (build / "commands.json").write_text(json.dumps(commands, indent=2))
        assert result.returncode == 0, result.stderr
