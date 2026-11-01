"""A complete three-unit source team keeps one checked queue transaction."""
import json
import shutil
import subprocess
from copy import deepcopy

import pytest

from compiler.frontend.source_device_tasks import source_device_task
from compiler.ir import CompilationError
from compiler.scopes import source_device_tasks
from compiler.scopes.queued_team_source_binding import execution_record
from compiler.tests import test_queued_source_binding as single
from compiler.tests.test_queued_team_task_provider import three_units
from compiler.tests.test_tile_private_team import team_fixture


def issued(directory):
    _path, analysis, routine, nodes = team_fixture(directory, three_units())
    analysis.seal_session()
    proof = source_device_task(analysis, routine, nodes)
    provider = source_device_tasks._provider(proof, transport="queued_team")
    resources = {binding.root: {"resource": binding.root, "descriptor": binding.public(),
                 "device_replica_authorized": True, "registration_identity": index}
                 for index, (binding, _written) in enumerate(proof.fields, 1)}
    binding = {"owner": routine.qualified, "canonical_resources": {root: root for root in resources},
               "registration_slots": {root: value["registration_identity"] for root, value in resources.items()},
               "provider": source_device_tasks._provider_record(provider)}
    record = execution_record(proof, binding)
    member = {"procedure": routine.qualified, "source_graph": proof.public()["source_graph"]}
    manifest = {"artifacts_sha256": {provider.entry + ".cu": binding["provider"]["cuda_source_sha256"],
                                   provider.module + ".f90": binding["provider"]["fortran_source_sha256"]}}
    return record, member, {"procedure": routine.qualified, "calls": []}, analysis.inputs.identity(), resources, manifest


def test_complete_original_three_unit_checked_transport_round_trip(tmp_path):
    values = issued(tmp_path)
    single.validate(values)
    record = values[0]
    metadata = record["execution_binding"]["provider"]["metadata"]
    assert len(record["source_proof"]["team"]["unit_proof_identities"]) == 3
    assert record["transport"]["original_units"] == len(metadata["code_keys"]) == 3
    assert metadata["consumer_wiring"] == "checked_source_device_task_queue_team_v1"
    assert "expected_launches" not in record["transport"]
    assert metadata["effects"]["initialization_authorized"] is False
    assert record["source_proof"]["transaction"]["runtime"] == "fort_scope_source_execute_v1"


@pytest.mark.parametrize("fault", ["version", "pricing", "source_role", "single_kind", "direct_backend",
                                  "complete_units", "bool_units", "missing_unit", "ordered_unit_keys",
                                  "replay_status", "queue", "flush_boundary", "overwrite", "initialize",
                                  "resource_budget", "truncation", "scratch", "completion", "metadata_standalone",
                                  "artifact", "parameter", "missing_abi", "bool_abi", "canonical", "registration"])
def test_resealed_whole_team_transport_cannot_drop_units_or_gain_old_authority(tmp_path, fault):
    values = issued(tmp_path)
    single.validate(values)
    record, _member, _procedure, _inputs, resources, manifest = values
    metadata = record["execution_binding"]["provider"]["metadata"]
    if fault == "version":
        record["version"] = True
    elif fault == "pricing":
        record["pricing"] = {}
    elif fault == "source_role":
        record["source_proof_role"] = "direct execution or queue pricing"
    elif fault == "single_kind":
        record["kind"] = "queued_single_original_tile_private_team"
    elif fault == "direct_backend":
        record["transport"]["backend"] = "fort_scope_source_execute_v1"
    elif fault == "complete_units":
        record["transport"]["original_units"] = 2
    elif fault == "bool_units":
        record["transport"]["original_units"] = True
    elif fault == "missing_unit":
        metadata["unit_proof_identities"].pop()
    elif fault == "ordered_unit_keys":
        metadata["code_keys"].reverse()
    elif fault == "replay_status":
        record["transport"]["recoverable_preaccept_statuses"].append(6)
    elif fault == "queue":
        metadata["restriction"]["empty_queue_at_entry"] = False
    elif fault == "flush_boundary":
        metadata["effects"]["atomic_commit"] = "flush after each kernel"
    elif fault == "overwrite":
        metadata["effects"]["overwrites_authorized"] = True
    elif fault == "initialize":
        metadata["effects"]["initialization_authorized"] = True
    elif fault == "resource_budget":
        metadata["restriction"]["metadata_budget_bytes"] += 1
    elif fault == "truncation":
        metadata["restriction"]["maximum_launches"] = 64
    elif fault == "scratch":
        metadata["scratch"] = "release after first enqueue"
    elif fault == "completion":
        metadata["report"]["committed"] = "device completed"
    elif fault == "metadata_standalone":
        metadata.update(version=1, consumer_wiring=False)
    elif fault == "artifact":
        manifest["artifacts_sha256"].clear()
    elif fault == "parameter":
        metadata["parameters"][0]["kind"] = "c_double"
    elif fault == "missing_abi":
        metadata["launch_abi"]["units"].pop()
    elif fault == "bool_abi":
        metadata["launch_abi"]["units"][0]["parameters"] = True
    elif fault == "canonical":
        roots = list(resources)
        record["execution_binding"]["canonical_resources"][roots[0]] = roots[1]
    else:
        root = next(iter(resources))
        record["execution_binding"]["registration_slots"][root] += 1
    single.reseal(record)
    with pytest.raises(ValueError, match="queued|source"):
        single.validate(values)


def test_direct_cost_or_placement_cannot_select_whole_team(tmp_path):
    _path, analysis, routine, nodes = team_fixture(tmp_path, three_units())
    proof = source_device_task(analysis, routine, nodes)
    for kwargs in ({"pricing": object()}, {"placement": object()}):
        with pytest.raises(CompilationError, match="pricing|placement"):
            source_device_tasks._provider(proof, transport="queued_team", **kwargs)


def test_genuine_checked_three_unit_owner_emits_closed_manifest_and_original_zero_guards(tmp_path, monkeypatch):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    monkeypatch.setattr(source_device_tasks, "admit_source_device_task",
                        lambda member, nodes: source_device_tasks._admit_source_device_task(
                            member, nodes, transport="queued_team"))
    outputs, manifest, facts = single.emit(tmp_path / "inputs", monkeypatch, queued=False, source=three_units())
    assert manifest["scope_count"] == 1
    task, = manifest["scopes"][0]["source_device_tasks"]
    assert task["kind"] == "queued_original_tile_private_team"
    assert task["transport"]["original_units"] == 3
    validate_checked_execution_plans(manifest, capture_facts=facts)
    original = next(text for name, text in outputs.items() if name.startswith("sources/"))
    assert "fort_tile_queue_report_v1" in original
    assert "%started" not in original
    assert "fort_scope_abandon" in original
    assert original.index("if ((t) > 0)") < original.index("fort_source_status = fort_tile_task_")
    assert original.index("if ((1) <= (n))") < original.index("fort_source_status = fort_tile_task_")
    poisoned = deepcopy(manifest)
    changed = poisoned["scopes"][0]["source_device_tasks"][0]
    changed["transport"]["original_units"] = 1
    single.reseal(changed)
    with pytest.raises(ValueError, match="queued|source"):
        validate_checked_execution_plans(poisoned, capture_facts=facts)
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        artifact = build / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (tmp_path / "facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    fortran = shutil.which("gfortran-15")
    assert fortran is not None, "required original compiler cannot skip"
    commands = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["language"] != "fortran" or artifact["role"] != role:
                continue
            argv = [fortran, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-c",
                    str(build / artifact["path"]), "-J", str(build), "-I", str(build),
                    "-o", str(build / f"source-{index}.o")]
            result = subprocess.run(argv, cwd=build, capture_output=True, text=True, timeout=30)
            commands.append({"argv": argv, "returncode": result.returncode})
            (build / f"compile-{index}.stdout").write_text(result.stdout)
            (build / f"compile-{index}.stderr").write_text(result.stderr)
            (build / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            assert result.returncode == 0, result.stdout + result.stderr
