"""Real payload/receipt integration for fixed numerical policies, without app runs.

Build command receipts are explicit synthetic fixtures, as in application-build
tests. No compiler/build/numerical authority is mocked by these tests.
"""

import json
import os
import struct
from pathlib import Path

import pytest

from compiler.offload.application_build import (
    application_build_validation,
    digest,
    preflight_application_build,
)
from compiler.offload.selection_cost import process_evidence_decision
from compiler.offload.selection_validation import (
    POLICY_PROTOCOL,
    RECEIPT_PROTOCOL,
    VTK_FORMAT,
    _compare_field,
    _Reader,
    normalize_policy,
)
from compiler.tests.test_application_build import (
    diagnostic_fixture,
    hashed,
    observed_build,
    prepared_receipts,
    save,
    synthetic_receipt,
    write,
)
from compiler.tests.test_selection_cost import process_fixture


def reference(path):
    return {"path": str(path), "sha256": hashed(path)}


@pytest.fixture
def prepared(tmp_path):
    return prepared_receipts(tmp_path)


def rewrite(ref, content):
    path = Path(ref["path"])
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(json.dumps(content, sort_keys=True))
    ref["sha256"] = hashed(path)


def vtk(delta=0.0, material=2.0):
    result = b"# vtk DataFile Version 2.0\nfixture\nBINARY\nDATASET RECTILINEAR_GRID\nDIMENSIONS 2 2 2\n"
    for axis in "XYZ":
        result += f"{axis}_COORDINATES 2 double\n".encode() + struct.pack(">2d", 0.0, 1.0) + b"\n"
    result += b"POINT_DATA 8\n"
    for name, components, value in (("temperature", 1, 1.0 + delta), ("velocity", 3, 2.0 + delta),
                                     ("material", 1, material)):
        header = (f"SCALARS {name} double\nLOOKUP_TABLE default\n" if components == 1
                  else f"VECTORS {name} double\n")
        result += header.encode() + struct.pack(">" + str(8 * components) + "d", *([value] * (8 * components))) + b"\n"
    return result


def bind_selection(p, contract):
    """Issue the fixture's actual build manifest for this complete contract."""
    scope = p["manifest"]["scopes"][0]["exact_window"]
    scope["contract"], scope["contract_sha256"] = contract, digest(contract)
    artifact_root = Path(p["preflight"]["request"]["artifact_root"])
    selector = artifact_root / "selector.cu"
    selector.write_text("#ifndef FORT_EXACT_WINDOW_BUILD_" + digest(contract) + "\n#error missing\n#endif\n")
    p["manifest"]["artifacts_sha256"]["selector.cu"] = hashed(selector)
    request = p["preflight"]["request"]
    request["manifest"], request["manifest_sha256"] = save(artifact_root / "scope-manifest.json", p["manifest"])
    p["preflight"] = preflight_application_build(request)
    original = next(iter(p["arguments"]["sources"]))
    source = Path(request["destinations"][original])
    p["actual"] = synthetic_receipt(Path(request["build_root"]), source.parent, source,
        [source.parent / "driver.f90", *map(Path, p["preflight"]["generated"])],
        macro=p["preflight"]["macros"][str(selector)], header=source.parent / "values.inc")
    p["receipt"], p["receipt_sha256"] = save(p["root"] / "actual-receipt.json", p["actual"])


@pytest.fixture
def campaign(prepared):
    p = prepared
    contract, evidence = process_fixture()
    contract["selection_bundle"]["groups"] = [{"selection_identity": contract["selection_identity"]}]
    contract["selection_bundle"]["identity"] = digest({key: value for key, value in contract["selection_bundle"].items()
                                                       if key != "identity"})
    contract["sources"] = p["arguments"]["sources"]
    contract["build_contract"] = p["contract"]
    policy = {"schema_version": 1, "protocol": POLICY_PROTOCOL, "dtype": "<f8",
              "fields": {"state": {"payload": "fields.bin", "metadata": "fields.shape"}},
              "metadata_format": "storage-shape-bounds-v1", "atol": 1e-6, "rtol": 1e-5,
              "finite_only": True, "exact_layouts": True,
              "outputs": {"format": VTK_FORMAT, "files": {
                  "output/final.vtk": ["temperature", "velocity"],
                  "output/average.vtk": ["temperature", "velocity"]}},
              "driver": {"original_source": p["driver"], "original_sha256": hashed(p["driver"]),
                         "replacement_sha256": digest_text("reviewed dump driver\n")}}
    contract["validation_policy"] = normalize_policy(policy, build_contract=p["contract"])
    item = evidence["campaigns"][0]
    workload = item["workload"]
    workload["files"] = {"case.conf": digest_text("unchanged configuration\n")}
    workload["identity"] = digest({key: value for key, value in workload.items() if key != "identity"})
    bind_selection(p, contract)
    builds = {mode: observed_build(p, mode) for mode in ("native", "resident")}
    diagnostics = {mode: diagnostic_fixture(p, build, mode)[1] for mode, build in builds.items()}
    receipt = {"schema_version": 1, "protocol": RECEIPT_PROTOCOL, "contract_sha256": digest(contract),
               "policy_identity": contract["validation_policy"]["identity"], "workload_identity": workload["identity"],
               "selection_bundle_identity": contract["selection_bundle"]["identity"], "variants": {}, "pairs": []}
    identity = item["identity"]
    cpu = item["observations"][0]["observed_cpu_identity"]

    def run(directory, mode, binary, seconds):
        directory.mkdir(parents=True)
        write(directory / "case.conf", "unchanged configuration\n")
        delta = 1e-7 if mode == "resident" else 0.0
        outputs = {}
        for name in policy["outputs"]["files"]:
            path = directory / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(vtk(delta))
            outputs[name] = reference(path)
        # Additional recorded files must not invalidate required comparisons.
        snapshot = write(directory / "other-output.txt", "unselected output\n")
        invocations = [{"ordinal": index, "selection": contract["selection_identity"], "key": [value],
                        "seconds": .01, "outcome": mode, "launches": int(mode == "resident")}
                       for index, value in enumerate((96, 128, 96))]
        log = directory / "stderr.log"
        log.write_text("".join("FORT_SELECTION_OBSERVATION " + json.dumps(
            {**{key: value for key, value in row.items() if key != "ordinal"},
             "identity": identity if mode == "resident" else None,
             "cpu_identity": cpu if mode == "native" else None}) + "\n" for row in invocations))
        value = {"argv": [binary, *workload["argv"][1:]], "binary_sha256": hashed(binary),
                 "exit_code": 0, "timed_out": False, "workload_identity": workload["identity"], "traced": False,
                 "configuration_sha256": workload["files"], "environment": workload["environment"],
                 "cwd": str(directory), "affinity": identity["affinity"], "threads": identity["threads"],
                 "completion": workload["completion"], "diagnostic_complete_process_seconds": seconds,
                 "output_hashes": {name: ref["sha256"] for name, ref in outputs.items()} | {snapshot.name: hashed(snapshot)}}
        path, recorded = save(directory / "run.json", value)
        return {"run": {"path": path, "sha256": recorded}, "observations": reference(log), "outputs": outputs}

    for mode, diagnostic in diagnostics.items():
        build = json.loads(Path(diagnostic["build"]["path"]).read_text())
        variant = run(p["root"] / (mode + "-dump-run"), mode, build["binary"], 1.0)
        directory = Path(variant["run"]["path"]).parent
        data = [1.0 + (1e-7 if mode == "resident" else 0.0)] * 8
        (directory / "fields.bin").write_bytes(struct.pack("<8d", *data))
        write(directory / "fields.shape", "64\n2 2 2\n-1 0 2\n0 1 3\n")
        receipt["variants"][mode] = {**variant, "diagnostic_build": diagnostic,
            "fields": {"state": {"payload": reference(directory / "fields.bin"),
                                  "metadata": reference(directory / "fields.shape")}}}
    for row in item["observations"]:
        mode = row["mode"]
        row["build"] = builds[mode]
        row["process"].update(workload_identity=workload["identity"], files=workload["files"],
                              selection_bundle_identity=contract["selection_bundle"]["identity"])
        if len(receipt["pairs"]) <= row["round"]:
            receipt["pairs"].append({"round": row["round"]})
        attestation = json.loads(Path(builds[mode]["attestation"]).read_text())
        receipt["pairs"][row["round"]][mode] = run(
            p["root"] / f"round-{row['round']}-{mode}", mode, attestation["binary"], row["seconds"])
        row["agreement"] = False
        row["fields_sha256"] = digest(mode)  # Neither old flag nor fabricated equal hash grants authority.
    path, recorded = save(p["root"] / "numerical.json", receipt)
    item["numerical_validation"] = {"path": path, "sha256": recorded}
    return {"contract": contract, "evidence": evidence, "receipt": receipt, "prepared": p}


def digest_text(text):
    from hashlib import sha256

    return sha256(text.encode()).hexdigest()


def decide(campaign):
    ref = campaign["evidence"]["campaigns"][0]["numerical_validation"]
    rewrite(ref, campaign["receipt"])
    return process_evidence_decision(campaign["evidence"], campaign["contract"])


def test_real_payloads_and_attestations_allow_distinct_hashes_at_fixed_tolerance(campaign):
    result = decide(campaign)
    assert result["available"], result
    assert result["selection"] == "resident", result
    report = result["campaigns"][0]["numerical_validation"]
    assert len(report["timed_pairs"]) == 9
    assert report["fields"]["state"]["layout"]["lower_bounds"] == [-1, 0, 2]
    assert report["fields"]["state"]["max_absolute_error"] > 0
    assert "material" in report["diagnostic_outputs"]["output/final.vtk"]["fields"]
    variants = campaign["receipt"]["variants"]
    assert variants["native"]["fields"]["state"]["payload"]["sha256"] != variants["resident"]["fields"]["state"]["payload"]["sha256"]


@pytest.mark.parametrize("change", ["halo", "interior", "nan", "bounds", "truncated", "payload_hash", "tolerance",
                                   "binary", "gpu_outcome", "sequence", "missing_output", "extra_field_value",
                                   "extra_field_inventory", "pair_order", "selection_contract"])
def test_changed_actual_payload_or_evidence_never_authorizes_placement(campaign, change):
    receipt = campaign["receipt"]
    variant = receipt["variants"]["resident"]
    field = variant["fields"]["state"]
    if change in {"halo", "interior", "nan", "truncated"}:
        values = [1.0] * 8
        values[0 if change == "halo" else 3] = float("nan") if change == "nan" else 5.0
        rewrite(field["payload"], struct.pack("<8d", *values)[:8] if change == "truncated" else struct.pack("<8d", *values))
    elif change == "bounds":
        Path(field["metadata"]["path"]).write_text("64\n2 2 2\n0 0 2\n1 1 3\n")
        field["metadata"]["sha256"] = hashed(field["metadata"]["path"])
    elif change == "payload_hash":
        field["payload"]["sha256"] = "0" * 64
    elif change == "tolerance":
        campaign["contract"]["validation_policy"]["atol"] = 100.0
    elif change == "selection_contract":
        campaign["contract"]["selection_identity"] = "0" * 64
    elif change == "binary":
        run = json.loads(Path(variant["run"]["path"]).read_text())
        run["binary_sha256"] = "0" * 64
        rewrite(variant["run"], run)
    elif change in {"gpu_outcome", "sequence"}:
        log = Path(variant["observations"]["path"])
        lines = log.read_text().splitlines()
        if change == "sequence":
            lines.pop()
        else:
            lines[0] = lines[0].replace('"launches": 1', '"launches": 0')
        log.write_text("\n".join(lines) + "\n")
        variant["observations"]["sha256"] = hashed(log)
    elif change == "missing_output":
        receipt["pairs"][4]["native"]["outputs"].pop("output/final.vtk")
    elif change == "pair_order":
        receipt["pairs"][2]["round"] = 7
    else:
        selected = receipt["pairs"][0]["resident"]
        output = selected["outputs"]["output/final.vtk"]
        content = vtk(material=5.0) if change == "extra_field_value" else vtk().replace(b"SCALARS material", b"SCALARS different")
        rewrite(output, content)
        run = json.loads(Path(selected["run"]["path"]).read_text())
        run["output_hashes"]["output/final.vtk"] = output["sha256"]
        rewrite(selected["run"], run)
    result = decide(campaign)
    assert not result["available"], result
    assert result["selection"] == "native", result
    expected = {
        "halo": "exceeds source-bound tolerance", "interior": "exceeds source-bound tolerance",
        "nan": "nonfinite numerical value", "bounds": "bounds or shape differ", "truncated": "byte count",
        "payload_hash": "hash differs", "tolerance": "selection contract differs",
        "binary": "run executable or arguments differ", "gpu_outcome": "selected key, invocation",
        "sequence": "selected invocation sequence", "missing_output": "output file inventory differs",
        "extra_field_value": "exceeds source-bound tolerance", "extra_field_inventory": "field declaration differs",
        "pair_order": "output pair order differs", "selection_contract": "selection contract differs",
    }
    assert expected[change] in result["reason"], result


def test_missing_comparison_cannot_be_replaced_by_pass_flags(campaign):
    item = campaign["evidence"]["campaigns"][0]
    item.pop("numerical_validation")
    for row in item["observations"]:
        row["agreement"], row["fields_sha256"] = True, "a" * 64
    result = process_evidence_decision(campaign["evidence"], campaign["contract"])
    assert not result["available"], result
    assert result["selection"] == "native", result


@pytest.mark.parametrize("endian", ["<", ">"])
def test_full_storage_payload_is_checked_in_declared_byte_order(tmp_path, endian):
    metadata = write(tmp_path / "shape", "64 3 -1 1\n")
    native, resident = tmp_path / "native.bin", tmp_path / "resident.bin"
    native.write_bytes(struct.pack(endian + "3d", 1.0, 2.0, 3.0))
    resident.write_bytes(struct.pack(endian + "3d", 1.0 + 1e-7, 2.0, 3.0))
    result = _compare_field({"payload": reference(native), "metadata": reference(metadata)},
                            {"payload": reference(resident), "metadata": reference(metadata)},
                            {"dtype": endian + "f8", "atol": 1e-6, "rtol": 1e-5})
    assert result["values"] == 3
    assert result["layout"]["lower_bounds"] == [-1]


def test_reader_rejects_fifo_without_waiting_for_writer(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        _Reader({"path": str(fifo), "sha256": "0" * 64})


def test_reader_rejects_mutation_of_the_opened_payload(tmp_path):
    path = write(tmp_path / "data", "original")
    def mutate_opened_file():
        with _Reader(reference(path)) as reader:
            reader.read(8)
            path.write_text("modified")

    with pytest.raises(ValueError, match="changed while reading"):
        mutate_opened_file()


def test_selection_contract_binding_is_checked_even_after_build_memoization(prepared):
    from compiler.offload.application_build import validate_application_build_observation

    observation = observed_build(prepared, "resident")
    with application_build_validation():
        validate_application_build_observation(observation, contract=prepared["contract"], mode="resident")
        with pytest.raises(ValueError, match="selection contract"):
            validate_application_build_observation(observation, contract=prepared["contract"], mode="resident",
                                                   selection_contract_identity="0" * 64)
