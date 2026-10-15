"""Application cost build recipes retain semantics and bind final artifacts."""

import json
import subprocess
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.numerical_contract import cuda_compile_options
from compiler.offload.application_build import (
    PROTOCOL,
    application_build_validation,
    attest_application_build,
    digest,
    preflight_application_build,
    require_application_build_contract,
    validate_application_build_observation,
    validate_diagnostic_relink,
)


def hashed(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return str(path), hashed(path)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def synthetic_receipt(root, application, original, generated, *, macro=None, header=None):
    """Small explicit JSON evidence fixture; does not claim a compiled program."""
    root.mkdir(parents=True, exist_ok=True)
    compiler = {"path": sys.executable, "sha256": hashed(sys.executable),
                "version": "synthetic-receipt-test", "version_returncode": 0}
    units, inputs, files = [], {}, {sys.executable: compiler["sha256"]}
    environment = {"FCFLAGS": "-fprotect-parens", "NVCC_APPEND_FLAGS": None}

    def command(argv, inpaths, output):
        write(output, json.dumps(argv))
        value = {"schema_version": 1, "kind": "executed-build-command", "status": "completed", "returncode": 0,
                 "shell": False, "argv": argv, "cwd": str(application), "compiler": compiler,
                 "environment": environment, "inputs": [{"path": str(p), "sha256": hashed(p)} for p in inpaths],
                 "outputs": [{"path": str(output), "sha256": hashed(output)}]}
        files.update({str(p): hashed(p) for p in [*inpaths, output]})
        return value

    objects = []
    for i, source in enumerate([original, *generated]):
        output = root / f"unit{i}.o"
        flags = (["-O3", *cuda_compile_options()] if source.suffix == ".cu" else
                 ["-O3", "-fopenmp", "-I" + str(root), "-J" + str(root)])
        if macro and "selector" in source.name:
            flags.append("-D" + macro)
        argv = [sys.executable, *flags, "-c", str(source), "-o", str(output)]
        units.append({"source": str(source), "object": str(output), "execution": command(argv, [source], output)})
        objects.append(output)
        inputs[str(source)] = hashed(source)
    binary = root / "program"
    link = command([sys.executable, "-O3", *map(str, objects), "-lm", "-o", str(binary)], objects, binary)
    if header:
        inputs[str(header)] = hashed(header)
        files[str(header)] = hashed(header)
    value = {"schema_version": 1, "kind": "application-build-receipts", "status": "complete", "placement_authority": False,
             "binary": str(binary), "units": units, "link": link, "source_inputs": inputs, "files_sha256": files,
             "limitations": ["No placement authority", "Implicit system headers and libraries are not inventoried"]}
    return {**value, "identity": digest(value)}


def prepared_receipts(tmp_path):
    current = write(tmp_path / "original" / "source.f90", "original\n")
    driver = write(current.parent / "driver.f90", "original driver\n")
    header = write(current.parent / "values.inc", "integer, parameter :: n=4\n")
    native = write(tmp_path / "native" / "source.f90", current.read_text())
    native_driver = write(native.parent / "driver.f90", driver.read_text())
    native_header = write(native.parent / "values.inc", header.read_text())
    resident = write(tmp_path / "resident" / "source.f90", "old approved replacement\n")
    resident_driver = write(resident.parent / "driver.f90", driver.read_text())
    worker = write(tmp_path / "old-artifacts" / "worker.cu", "numerical worker\n")
    interface = write(worker.parent / "interface.f90", "interface\n")
    old_manifest = {"sources": {str(current): {"sha256": hashed(current), "replacement": "old.f90"}},
                    "artifacts_sha256": {"old.f90": hashed(resident), "worker.cu": hashed(worker), "interface.f90": hashed(interface)}}
    manifest_path, manifest_hash = save(tmp_path / "old-manifest.json", old_manifest)
    n = synthetic_receipt(tmp_path / "native-build", native.parent, native, [native_driver], header=native_header)
    r = synthetic_receipt(tmp_path / "resident-build", resident.parent, resident, [resident_driver, interface, worker])
    np, nh = save(tmp_path / "native-receipt.json", n)
    rp, rh = save(tmp_path / "resident-receipt.json", r)
    request = {"schema_version": 1, "protocol": PROTOCOL, "origins": {
        "native": {"receipt": np, "receipt_sha256": nh, "build_root": str(tmp_path / "native-build"),
                   "source_mapping": {str(native): str(current), str(native_header): str(header), str(native_driver): str(driver)}},
        "resident": {"receipt": rp, "receipt_sha256": rh, "build_root": str(tmp_path / "resident-build"),
                     "source_mapping": {str(resident): str(current), str(resident_driver): str(driver)},
                     "manifest": manifest_path, "manifest_sha256": manifest_hash}}}
    arguments = dict(sources={str(current): hashed(current)}, source_inputs_identity="1" * 64,
                     configuration_identity="2" * 64, compiler_identity="3" * 64, options_identity="4" * 64,
                     worker_artifacts={"entries/leaf/shared_entry.cu": hashed(worker)}, runtime_identity="5" * 64)
    contract = require_application_build_contract(request, **arguments)
    artifact_root = tmp_path / "new-artifacts"
    new_worker = write(artifact_root / "entries/leaf/shared_entry.cu", worker.read_text())
    new_interface = write(artifact_root / "interface.f90", interface.read_text())
    replacement = write(artifact_root / "replacement.f90", "new compiler approved replacement\n")
    selection_contract = {"protocol": "checked-selection-process-v2", "sources": arguments["sources"], "build_contract": contract}
    selection_identity = digest(selection_contract)
    selector = write(artifact_root / "selector.cu", "#ifndef FORT_EXACT_WINDOW_BUILD_" + selection_identity + "\n#error attestation required\n#endif\n")
    files = [new_worker, new_interface, replacement, selector]
    manifest = {"source_inputs": arguments["sources"], "runtime": {"runtime_id": arguments["runtime_identity"]},
                "scopes": [{"exact_window": {"contract": selection_contract, "contract_sha256": selection_identity,
                                            "capture": True, "capture_variant": "resident"}}],
                "sources": {str(current): {"sha256": hashed(current), "replacement": "replacement.f90"}},
                "artifacts_sha256": {p.relative_to(artifact_root).as_posix(): hashed(p) for p in files},
                "build_sources": [{"path": p.relative_to(artifact_root).as_posix(), "language": "cuda" if p.suffix == ".cu" else "fortran",
                                   "role": "original_source" if p == replacement else "shared_entry"} for p in files]}
    mp, mh = save(artifact_root / "scope-manifest.json", manifest)
    destination = write(tmp_path / "new-application" / "source.f90", replacement.read_text())
    destination_header = write(destination.parent / "values.inc", header.read_text())
    destination_driver = write(destination.parent / "driver.f90", driver.read_text())
    build_root = tmp_path / "new-build"
    preflight_request = {"manifest": mp, "manifest_sha256": mh, "artifact_root": str(artifact_root), "build_root": str(build_root),
                         "destinations": {str(current): str(destination), str(header): str(destination_header),
                                          str(driver): str(destination_driver)}}
    preflight = preflight_application_build(preflight_request)
    actual = synthetic_receipt(build_root, destination.parent, destination, [destination_driver, new_worker, new_interface, selector],
                               macro=preflight["macros"][str(selector)], header=destination_header)
    ap, ah = save(tmp_path / "actual-receipt.json", actual)
    return {"request": request, "arguments": arguments, "contract": contract, "preflight": preflight,
            "actual": actual, "receipt": ap, "receipt_sha256": ah, "manifest": manifest, "root": tmp_path,
            "driver": str(driver)}


@pytest.fixture
def prepared(tmp_path):
    return prepared_receipts(tmp_path)


def test_relocated_exact_build_contract_and_post_build_observation(prepared):
    p = prepared
    attestation = attest_application_build(p["preflight"], p["receipt"], p["receipt_sha256"])
    path, recorded = save(p["root"] / "attestation.json", attestation)
    observation = {"receipt": p["receipt"], "receipt_sha256": p["receipt_sha256"], "attestation": path,
                   "attestation_sha256": recorded, "binary_sha256": attestation["binary_sha256"],
                   "contract_identity": attestation["contract_identity"], "variant": "resident"}
    assert validate_application_build_observation(observation, contract=p["contract"], mode="resident") == attestation["identity"]
    with pytest.raises(ValueError, match="variant"):
        validate_application_build_observation(observation, contract=p["contract"], mode="native")
    p["request"]["origins"]["native"]["source_mapping"].clear()
    assert p["contract"]["origins"]["native"]["source_mapping"]


@pytest.mark.parametrize("change", ["flag", "environment", "link", "macro", "unmapped_object"])
def test_actual_semantic_command_changes_are_not_path_normalization(prepared, change):
    p = prepared
    actual = deepcopy(p["actual"])
    if change == "flag":
        actual["units"][0]["execution"]["argv"].insert(1, "-ffast-math")
    elif change == "environment":
        actual["units"][0]["execution"]["environment"]["FCFLAGS"] = "-Ofast"
    elif change == "link":
        actual["link"]["argv"].insert(1, "-Wl,--as-needed")
    elif change == "macro":
        unit = next(u for u in actual["units"] if "selector" in u["source"])
        unit["execution"]["argv"] = [a for a in unit["execution"]["argv"] if not a.startswith("-DFORT_EXACT")]
    else:
        actual["units"][0]["source"] = str(p["root"] / "unmapped.f90")
    actual["identity"] = digest({k: v for k, v in actual.items() if k != "identity"})
    path, recorded = save(p["root"] / "altered.json", actual)
    with pytest.raises(ValueError, match="application build"):
        attest_application_build(p["preflight"], path, recorded)


@pytest.mark.parametrize("which", ["original", "header", "worker", "binary"])
def test_changed_authenticated_payloads_fail_closed(prepared, which):
    p = prepared
    if which in {"original", "header"}:
        original = next(name for name in p["contract"]["original_inputs"] if name.endswith(".f90" if which == "original" else ".inc"))
        path = p["preflight"]["request"]["destinations"][original]
    elif which == "worker":
        path = next(name for name in p["preflight"]["generated"] if name.endswith("shared_entry.cu"))
    else:
        path = p["actual"]["binary"]
    Path(path).write_text("changed")
    with pytest.raises(ValueError, match="application build"):
        attest_application_build(p["preflight"], p["receipt"], p["receipt_sha256"])


def test_native_origin_cannot_map_different_current_sources(prepared):
    p = prepared
    current = next(iter(p["arguments"]["sources"]))
    Path(current).write_text("different original")
    with pytest.raises(ValueError, match="native original"):
        require_application_build_contract(p["request"], **p["arguments"])


def test_legacy_fixture_contract_is_not_an_application_contract():
    with pytest.raises(ValueError, match="versioned"):
        require_application_build_contract({"schema_version": 4}, sources={}, source_inputs_identity="", configuration_identity="",
                                           compiler_identity="", options_identity="", worker_artifacts={}, runtime_identity="")


def test_preflight_requires_every_admitted_bundle_group(prepared):
    p = prepared
    manifest = deepcopy(p["manifest"])
    first = manifest["scopes"][0]["exact_window"]
    first["contract"]["selection_identity"] = "a" * 64
    bundle = {"schema_version": 1, "entry": "step", "groups": [
        {"selection_identity": "a" * 64}, {"selection_identity": "b" * 64}]}
    bundle["identity"] = digest(bundle)
    first["contract"]["selection_bundle"] = bundle
    first["contract_sha256"] = digest(first["contract"])
    second = deepcopy(first)
    second["contract"]["selection_identity"] = "b" * 64
    second["contract_sha256"] = digest(second["contract"])
    manifest["scopes"].append({"exact_window": second})
    root = Path(p["preflight"]["request"]["artifact_root"])
    for name, selection in [("selector.cu", first), ("other_selector.cu", second)]:
        source = write(root / name, "#ifndef FORT_EXACT_WINDOW_BUILD_" + selection["contract_sha256"] + "\n#error missing\n#endif\n")
        manifest["artifacts_sha256"][name] = hashed(source)
    manifest["build_sources"].append({"path": "other_selector.cu", "language": "cuda", "role": "shared_entry"})
    request = deepcopy(p["preflight"]["request"])
    request["manifest"], request["manifest_sha256"] = save(root / "scope-manifest.json", manifest)
    result = preflight_application_build(request)
    assert len(result["macros"]) == 2
    assert len(result["selection_contract_identities"]) == 2
    manifest["scopes"].pop()
    request["manifest"], request["manifest_sha256"] = save(root / "scope-manifest.json", manifest)
    with pytest.raises(ValueError, match="bundle is incomplete"):
        preflight_application_build(request)


def test_native_capture_binds_resident_identity_without_its_removed_workers(prepared):
    p = prepared
    manifest = deepcopy(p["manifest"])
    manifest["scopes"][0]["exact_window"]["capture_variant"] = "native"
    observer = Path(p["preflight"]["request"]["artifact_root"]) / "selector.cu"
    observer.write_text('extern "C" void original_native_observer() {}\n')
    manifest["artifacts_sha256"]["selector.cu"] = hashed(observer)
    name = "entries/leaf/shared_entry.cu"
    del manifest["artifacts_sha256"][name]
    manifest["build_sources"] = [unit for unit in manifest["build_sources"] if unit["path"] != name]
    request = deepcopy(p["preflight"]["request"])
    request["manifest"], request["manifest_sha256"] = save(Path(request["manifest"]), manifest)
    result = preflight_application_build(request)
    assert result["variant"] == "native"
    assert result["macros"] == {}
    assert name in result["contracts"][0]["worker_artifacts"]
    assert all(not path.endswith(name) for path in result["generated"])
    destination = next(Path(path) for path in result["expected_inputs"] if path.endswith("source.f90"))
    header = next(Path(path) for path in result["expected_inputs"] if path.endswith(".inc"))
    receipt = synthetic_receipt(Path(request["build_root"]), destination.parent, destination,
                                [destination.parent / "driver.f90", *[Path(path) for path in result["generated"]]], header=header)
    path, recorded = save(p["root"] / "native-capture-receipt.json", receipt)
    attestation = attest_application_build(result, path, recorded)
    assert attestation["variant"] == "native"


def test_public_verifier_cli_publishes_only_successful_result(prepared):
    p = prepared
    path, _ = save(p["root"] / "public-request.json", p["preflight"]["request"])
    output = p["root"] / "public-result.json"
    command = [sys.executable, "-m", "compiler.offload.application_build", "preflight", "--request", path, "--output", str(output)]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["identity"] == p["preflight"]["identity"]
    output.unlink()
    request = deepcopy(p["preflight"]["request"])
    request["manifest_sha256"] = "0" * 64
    save(Path(path), request)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert not result.stdout
    assert not output.exists()
    assert "file changed" in result.stderr


def observed_build(p, mode):
    """Build an authenticated file/receipt fixture, not measured execution."""
    if mode == "resident":
        preflight, receipt, path, recorded = p["preflight"], p["actual"], p["receipt"], p["receipt_sha256"]
    else:
        artifact_root = p["root"] / "native-artifacts"
        manifest = deepcopy(p["manifest"])
        manifest["scopes"][0]["exact_window"]["capture_variant"] = "native"
        old_root = Path(p["preflight"]["request"]["artifact_root"])
        for name in manifest["artifacts_sha256"]:
            target = write(artifact_root / name, (old_root / name).read_text())
            if name == "selector.cu":
                target.write_text('extern "C" void native_observer() {}\n')
            manifest["artifacts_sha256"][name] = hashed(target)
        request = deepcopy(p["preflight"]["request"])
        request.update(artifact_root=str(artifact_root), build_root=str(p["root"] / "native-capture-build"))
        request["manifest"], request["manifest_sha256"] = save(artifact_root / "scope-manifest.json", manifest)
        request["destinations"] = {original: str(write(p["root"] / "native-application" / Path(path).name,
                                                       Path(path).read_text()))
                                   for original, path in request["destinations"].items()}
        preflight = preflight_application_build(request)
        source = Path(request["destinations"][next(iter(p["arguments"]["sources"]))])
        header = source.parent / "values.inc"
        receipt = synthetic_receipt(Path(request["build_root"]), source.parent, source,
                                    [source.parent / "driver.f90", *map(Path, preflight["generated"])], header=header)
        path, recorded = save(p["root"] / "native-actual.json", receipt)
    attestation = attest_application_build(preflight, path, recorded)
    ap, ah = save(p["root"] / (mode + "-attestation.json"), attestation)
    return {"receipt": path, "receipt_sha256": recorded, "attestation": ap, "attestation_sha256": ah,
            "binary_sha256": attestation["binary_sha256"], "contract_identity": attestation["contract_identity"],
            "variant": mode}


def diagnostic_fixture(p, observation, mode):
    """Two exact command receipts for a policy-approved driver replacement."""
    receipt = json.loads(Path(observation["receipt"]).read_text())
    attestation = json.loads(Path(observation["attestation"]).read_text())
    original = p["driver"]
    destination = attestation["preflight"]["request"]["destinations"][original]
    unit = next(unit for unit in receipt["units"] if unit["source"] == destination)
    output = p["root"] / (mode + "-diagnostic")
    source = write(output / "dump.f90", "reviewed dump driver\n")
    obj, binary = write(output / "dump.o", "diagnostic object\n"), write(output / "program", "diagnostic " + mode)
    module = write(output / "numeric.mod", "recorded diagnostic module\n")
    commands = [deepcopy(unit["execution"]), deepcopy(receipt["link"])]
    replacements = [{destination: str(source), unit["object"]: str(obj)},
                    {unit["object"]: str(obj), receipt["binary"]: str(binary)}]
    references = []
    for index, (command, mapping) in enumerate(zip(commands, replacements, strict=True)):
        command["argv"] = [mapping.get(arg, arg) for arg in command["argv"]]
        for role in ("inputs", "outputs"):
            command[role] = [{"path": mapping.get(item["path"], item["path"]),
                              "sha256": hashed(mapping.get(item["path"], item["path"]))}
                             for item in command[role]]
        command["inputs"].append({"path": str(module), "sha256": hashed(module)})
        path, recorded = save(output / f"command-{index}.json", command)
        references.append({"path": path, "sha256": recorded})
    plan = {"application_build_identity": receipt["identity"], "application_build_receipt": observation["receipt"],
            "application_binary": receipt["binary"], "diagnostic_binary": str(binary),
            "module_inputs": {str(module): hashed(module)}, "files_sha256": {str(source): hashed(source)},
            "relink": {"compile_argv": commands[0]["argv"], "link_argv": commands[1]["argv"],
                       "cwd": commands[0]["cwd"]}}
    pp, ph = save(output / "plan.json", plan)
    bp, bh = save(output / "build.json", {"status": "passed", "plan_sha256": ph,
                                         "binary": str(binary), "binary_sha256": hashed(binary)})
    driver = {"original_source": original, "original_sha256": hashed(original), "replacement_sha256": hashed(source)}
    return driver, {"plan": {"path": pp, "sha256": ph}, "build": {"path": bp, "sha256": bh}, "commands": references}


@pytest.mark.parametrize("mode", ["native", "resident"])
def test_reviewed_diagnostic_relink_preserves_actual_numerical_objects(prepared, mode):
    observation = observed_build(prepared, mode)
    driver, diagnostic = diagnostic_fixture(prepared, observation, mode)
    with application_build_validation():
        result = validate_diagnostic_relink(diagnostic, application_build=observation,
                                            contract=prepared["contract"], driver=driver, mode=mode)
    assert result["binary_sha256"] == hashed(result["binary"])
    assert result["binary_sha256"] != observation["binary_sha256"]


@pytest.mark.parametrize("change", ["flag", "link_object", "environment", "driver", "module"])
def test_diagnostic_relink_rejects_changes_beyond_the_approved_driver(prepared, change):
    observation = observed_build(prepared, "resident")
    driver, diagnostic = diagnostic_fixture(prepared, observation, "resident")
    if change in {"flag", "link_object", "environment"}:
        reference = diagnostic["commands"][int(change == "link_object")]
        command = json.loads(Path(reference["path"]).read_text())
        if change == "flag":
            command["argv"].insert(1, "-ffast-math")
        elif change == "environment":
            command["environment"]["FCFLAGS"] = "-Ofast"
        else:
            command["argv"][2] = str(write(prepared["root"] / "foreign.o", "foreign numerical object"))
        _, reference["sha256"] = save(Path(reference["path"]), command)
    elif change == "driver":
        driver["replacement_sha256"] = "0" * 64
    else:
        plan = json.loads(Path(diagnostic["plan"]["path"]).read_text())
        Path(next(iter(plan["module_inputs"]))).write_text("changed module")
    with pytest.raises(ValueError, match="application build"), application_build_validation():
        validate_diagnostic_relink(diagnostic, application_build=observation,
                                   contract=prepared["contract"], driver=driver, mode="resident")


def test_repeated_observation_is_verified_once_then_every_dependency_rechecked(prepared, monkeypatch):
    import compiler.offload.application_build as module

    observation = observed_build(prepared, "resident")
    original = module._raw_hash
    reads = []

    def counted(path):
        reads.append(str(path))
        return original(path)

    monkeypatch.setattr(module, "_raw_hash", counted)
    with application_build_validation():
        first = validate_application_build_observation(observation, contract=prepared["contract"], mode="resident")
        before = len(reads)
        assert validate_application_build_observation(observation, contract=prepared["contract"], mode="resident") == first
        assert len(reads) == before
    assert len(reads) > before
    def mutate_checked_build():
        with application_build_validation():
            validate_application_build_observation(observation, contract=prepared["contract"], mode="resident")
            Path(prepared["actual"]["binary"]).write_text("changed after verification")

    with pytest.raises(ValueError, match="before build validation completed"):
        mutate_checked_build()
