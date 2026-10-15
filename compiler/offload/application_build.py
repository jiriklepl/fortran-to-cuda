"""Executed application build applicability, shared by the public verifier CLI.

This deliberately uses existing command receipts. It does not claim an implicit
system-header/toolchain closure, and never supplies numerical legality. The
pre-build contract excludes the selector and final binary to avoid a hash cycle;
the post-build attestation binds both through the finalized compiler manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

PROTOCOL = "executed-application-build-v1"
SELECTION_PROTOCOL = "checked-selection-process-v2"
LIMITATION = "Implicit system headers and libraries are not inventoried"
SOURCE_SUFFIXES = {".f", ".for", ".ftn", ".f90", ".f95", ".f03", ".f08", ".c", ".cc", ".cpp", ".cxx", ".cu"}
HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".cuh", ".inc"}
_VALIDATION = ContextVar("application_build_validation", default=None)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _raw_hash(path):
    with Path(path).open("rb") as stream:
        from hashlib import file_digest

        return file_digest(stream, "sha256").hexdigest()


def _remember(path, recorded):
    scope = _VALIDATION.get()
    if scope is not None:
        name = str(Path(path).absolute())
        _check(name not in scope["files"] or scope["files"][name] == recorded,
               "input changed during build validation: " + name)
        scope["files"][name] = recorded
    return recorded


def _hash(path):
    return _remember(path, _raw_hash(path))


def _bytes(path):
    raw = Path(path).read_bytes()
    _remember(path, sha256(raw).hexdigest())
    return raw


@contextmanager
def application_build_validation():
    """Share checked observations for one decision, then recheck every input.

    This scope has no persistent cache. A successful context exit, after fresh
    dependency hashes, is required before its caller can publish a decision.
    """
    _check(_VALIDATION.get() is None, "build validation scopes must not nest")
    scope = {"files": {}, "observations": {}}
    token = _VALIDATION.set(scope)
    try:
        yield
        for path, recorded in scope["files"].items():
            _check(_raw_hash(path) == recorded, "input changed before build validation completed: " + path)
    finally:
        _VALIDATION.reset(token)


def _check(condition, message):
    if not condition:
        raise ValueError("application build: " + message)


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _identified(value):
    _check(type(value) is dict and value.get("identity") == digest({k: v for k, v in value.items() if k != "identity"}),
           "record identity differs")
    return value


def _read(path, expected):
    raw = _bytes(path)
    _check(_sha(expected) and sha256(raw).hexdigest() == expected, "receipt or manifest file changed")
    return json.loads(raw)


def _receipt(path, expected):
    value = _identified(_read(path, expected))
    _check(value.get("schema_version") == 1 and value.get("kind") == "application-build-receipts"
           and value.get("status") == "complete" and value.get("placement_authority") is False,
           "complete successful executed receipts required")
    _check(type(value.get("files_sha256")) is dict and value["files_sha256"], "receipt file inventory missing")
    for name, recorded in value["files_sha256"].items():
        _check(_sha(recorded) and _hash(name) == recorded, "receipted file changed: " + name)
    _check(type(value.get("source_inputs")) is dict and value["source_inputs"], "source/header inventory missing")
    for name, recorded in value["source_inputs"].items():
        _check(value["files_sha256"].get(name) == recorded, "source inventory not bound to executed receipt")
    _check(type(value.get("units")) is list and value["units"], "original compilation units missing")
    seen = set()
    for unit in value["units"]:
        _check(unit["object"] not in seen, "ambiguous object")
        seen.add(unit["object"])
        command = unit["execution"]
        _command(command)
        _check(all(value["files_sha256"].get(item["path"]) == item["sha256"] for item in
                   [command["compiler"], *command["inputs"], *command["outputs"]]), "command files outside receipt inventory")
        _check(command["argv"].count("-c") == 1, "unit is not one compilation")
        _check({x["path"] for x in command["outputs"]} == {unit["object"]}, "unit output differs")
        _check(unit["source"] in value["source_inputs"] and value["source_inputs"][unit["source"]] == next(
            (x["sha256"] for x in command["inputs"] if x["path"] == unit["source"]), None),
            "compiled source has no input identity")
    _command(value["link"])
    _check(all(value["files_sha256"].get(item["path"]) == item["sha256"] for item in
               [value["link"]["compiler"], *value["link"]["inputs"], *value["link"]["outputs"]]),
           "link files outside receipt inventory")
    _check("-c" not in value["link"]["argv"] and {x["path"] for x in value["link"]["outputs"]} == {value["binary"]},
           "final executable link missing")
    _check({x["path"] for x in value["link"]["inputs"] if Path(x["path"]).suffix == ".o"} == seen,
           "link objects differ from executed compilation units")
    return value


def _command(command):
    _check(command.get("kind") == "executed-build-command" and command.get("schema_version") == 1
           and command.get("status") == "completed" and command.get("returncode") == 0
           and type(command.get("returncode")) is int and command.get("shell") is False,
           "unsuccessful or shell build command")
    _check(type(command.get("argv")) is list and command["argv"]
           and all(type(x) is str and "\0" not in x for x in command["argv"]), "invalid actual argv")
    _check(command["argv"].count("-o") == 1 and command.get("compiler", {}).get("version_returncode") == 0,
           "explicit compiler and output required")
    _check(type(command.get("environment")) is dict, "explicit build environment required")


def _path(argument, cwd):
    return str((Path(cwd) / argument).resolve())


def _recipe(command, root, substitutions, *, macro=None, source_mapping=None):
    """Normalize only exact operands and explicit -I/-J paths below build root."""
    root = Path(root).resolve()
    argv = []
    removed = 0
    for argument in command["argv"][1:]:
        if macro is not None and argument == "-D" + macro:
            removed += 1
            continue
        if not argument.startswith("-"):
            resolved = _path(argument, command["cwd"])
            if resolved in substitutions:
                argv.append(substitutions[resolved])
                continue
        prefix = next((p for p in ("-I", "-J") if argument.startswith(p) and len(argument) > len(p)), "")
        candidate = argument[len(prefix):] if prefix else argument
        if prefix or not argument.startswith("-"):
            resolved = Path(_path(candidate, command["cwd"]))
            if resolved.is_relative_to(root):
                argv.append(prefix + "<BUILD>/" + resolved.relative_to(root).as_posix())
                continue
        argv.append(argument)
    if macro is not None:
        _check(removed == 1, "selector attestation macro missing or repeated")
    directories = {}
    for original, current in (source_mapping or {}).items():
        old, new = str(Path(original).parent), str(Path(current).parent)
        _check(old not in directories or directories[old] == new, "ambiguous source working-directory mapping")
        directories[old] = new
    cwd = Path(command["cwd"]).resolve()
    normalized_cwd = ("<BUILD>/" + cwd.relative_to(root).as_posix() if cwd.is_relative_to(root)
                      else directories.get(str(cwd), str(cwd)))
    compiler = deepcopy(command["compiler"])
    inputs = {x["path"]: x["sha256"] for x in command["inputs"] if x["path"] not in substitutions}
    return {"argv": argv, "compiler": compiler, "environment": deepcopy(command["environment"]),
            "cwd": normalized_cwd, "inputs": inputs}


def _unit_recipe(unit, root, *, macro=None, source_mapping=None):
    return _recipe(unit["execution"], root, {unit["source"]: "<SOURCE>", unit["object"]: "<OBJECT>"},
                   macro=macro, source_mapping=source_mapping)


def _language(path):
    return "cuda" if Path(path).suffix.lower() == ".cu" else "fortran" if Path(path).suffix.lower().startswith(".f") else "other"


def _link_recipe(receipt, root, source_mapping):
    substitutions = {receipt["binary"]: "<BINARY>"}
    for unit in receipt["units"]:
        source = source_mapping.get(unit["source"])
        substitutions[unit["object"]] = "<ORIGINAL>:" + source if source else "<GENERATED>"
    recipe = _recipe(receipt["link"], root, substitutions, source_mapping=source_mapping)
    argv = recipe["argv"]
    indices = [i for i, argument in enumerate(argv) if argument == "<GENERATED>"]
    _check(indices and indices == list(range(indices[0], indices[-1] + 1)), "generated objects require one explicit link block")
    recipe["argv"] = argv[:indices[0]] + ["<GENERATED>"] + argv[indices[-1] + 1:]
    return recipe


def require_application_build_contract(requested, *, sources, source_inputs_identity, configuration_identity,
                                       compiler_identity, options_identity, worker_artifacts, runtime_identity):
    """Issue a detached expected contract from existing executed build origins."""
    _check(type(requested) is dict and requested.get("schema_version") == 1 and requested.get("protocol") == PROTOCOL,
           "versioned native/resident executed origins required")
    origins = requested.get("origins")
    _check(type(origins) is dict and set(origins) == {"native", "resident"}, "both independently built origins required")
    records, mappings = {}, {}
    for mode, origin in origins.items():
        records[mode] = _receipt(origin["receipt"], origin["receipt_sha256"])
        _check(Path(origin["build_root"]).resolve() == Path(records[mode]["binary"]).parent
               and all(Path(unit["object"]).is_relative_to(Path(origin["build_root"]).resolve())
                       for unit in records[mode]["units"]), "declared origin build root differs from executed outputs")
        mapping = origin["source_mapping"]
        _check(type(mapping) is dict and mapping and all(type(k) is str and type(v) is str for k, v in mapping.items()),
               "explicit original source mapping required")
        _check(len(set(mapping.values())) == len(mapping), "ambiguous original source mapping")
        mappings[mode] = {str(Path(k).resolve()): str(Path(v).resolve(strict=True)) for k, v in mapping.items()}
        _check(len(mappings[mode]) == len(mapping), "ambiguous normalized source mapping")
        _check(Path(origin["build_root"]).is_absolute(), "explicit absolute build root required")
    native, resident = records["native"], records["resident"]
    original_inputs = {}
    for name, recorded in native["source_inputs"].items():
        if name in mappings["native"]:
            target = mappings["native"][name]
            _check(_hash(target) == recorded, "native original source/header differs: " + target)
            original_inputs[target] = recorded
        else:
            _check(Path(name).suffix.lower() not in SOURCE_SUFFIXES | HEADER_SUFFIXES,
                   "unmapped original source/header: " + name)
    _check(all(original_inputs.get(name) == recorded for name, recorded in sources.items()),
           "current authenticated source set differs from native original inputs")
    original_units = {}
    for unit in native["units"]:
        target = mappings["native"].get(unit["source"])
        _check(target is not None and target not in original_units, "unmapped or ambiguous original compilation unit")
        original_units[target] = _unit_recipe(unit, origins["native"]["build_root"], source_mapping=mappings["native"])
    resident_units, generated = {}, {}
    # Old replacements establish only the origin of the old executed command.
    # They are never reused as new source or numerical authority.
    manifest = _read(origins["resident"]["manifest"], origins["resident"]["manifest_sha256"])
    replacements = {item["sha256"]: manifest["artifacts_sha256"][item["replacement"]]
                    for item in manifest.get("sources", {}).values()}
    for unit in resident["units"]:
        target = mappings["resident"].get(unit["source"])
        recipe = _unit_recipe(unit, origins["resident"]["build_root"], source_mapping=mappings["resident"])
        if target is not None:
            _check(target in original_units and target not in resident_units, "unmapped or ambiguous resident original unit")
            old = resident["source_inputs"][unit["source"]]
            _check(old == original_inputs[target] or replacements.get(original_inputs[target]) == old,
                   "resident origin source lacks exact original or approved replacement")
            baseline = original_units[target]
            _check(all(recipe[k] == baseline[k] for k in ("argv", "compiler", "inputs", "cwd")),
                   "native/resident original compiler or ordered flags differ: " + target)
            resident_units[target] = recipe
        else:
            language = _language(unit["source"])
            _check(language in {"fortran", "cuda"}, "unsupported generated language")
            _check(resident["source_inputs"][unit["source"]] in manifest["artifacts_sha256"].values(),
                   "generated origin source is not compiler-issued")
            _check(language not in generated or generated[language] == recipe, "generated language has ambiguous compile policy")
            generated[language] = recipe
    _check(set(resident_units) == set(original_units) and set(generated) == {"fortran", "cuda"},
           "resident origin must cover all originals and both generated languages")
    # Require separate-arithmetic options already present in the executed CUDA policy.
    from compiler.numerical_contract import cuda_compile_options

    _check(all(option in generated["cuda"]["argv"] for option in cuda_compile_options()), "numerical CUDA options missing")
    value = {"schema_version": 1, "protocol": PROTOCOL, "origins": deepcopy(origins),
             "sources": dict(sources), "source_inputs_identity": source_inputs_identity,
             "configuration_identity": configuration_identity, "compiler_identity": compiler_identity,
             "options_identity": options_identity, "worker_artifacts": dict(worker_artifacts), "runtime_identity": runtime_identity,
             "original_inputs": original_inputs, "original_units": resident_units, "generated": generated,
             "link": _link_recipe(resident, origins["resident"]["build_root"], mappings["resident"]),
             "coverage": "exact recorded local application build; common generated/runtime link envelope for both measured modes",
             "limitations": [LIMITATION]}
    return {**value, "identity": digest(value)}


def _contract(value):
    _identified(value)
    _check(value.get("protocol") == PROTOCOL and value.get("schema_version") == 1
           and value.get("limitations") == [LIMITATION], "unsupported build contract")
    return value


def preflight_application_build(request):
    """Bind a finalized compiler manifest and exact application-copy destinations."""
    manifest = _read(request["manifest"], request["manifest_sha256"])
    scopes = [scope.get("exact_window") for scope in manifest.get("scopes", []) if "exact_window" in scope]
    _check(scopes, "checked application selections required")
    contracts, identities = [], set()
    for selection in scopes:
        _check(selection["contract"].get("protocol") == SELECTION_PROTOCOL
               and selection["contract_sha256"] == digest(selection["contract"]), "unsupported selection contract")
        _check(selection["contract_sha256"] not in identities, "duplicate selection contract")
        identities.add(selection["contract_sha256"])
        contracts.append(_contract(selection["contract"]["build_contract"]))
    contract = contracts[0]
    def base(value):
        return {k: v for k, v in value.items() if k not in {"worker_artifacts", "identity"}}
    _check(all(base(item) == base(contract) for item in contracts), "selection bundle build policies differ")
    variants = {selection.get("capture_variant") if selection.get("capture") else "production" for selection in scopes}
    _check(len(variants) == 1, "mixed build capture variants")
    variant = variants.pop()
    _check(variant in {"native", "resident", "production"}, "invalid selected build variant")
    bundles = [selection["contract"].get("selection_bundle") for selection in scopes]
    if len(scopes) > 1 or bundles[0] is not None:
        bundle = _identified(bundles[0])
        _check(all(item == bundle for item in bundles) and bundle.get("schema_version") == 1,
               "selection bundle differs across scopes")
        groups = bundle.get("groups", [])
        selected = [selection["contract"]["selection_identity"] for selection in scopes]
        _check(len(groups) == len(scopes) and len(set(selected)) == len(selected)
               and {group["selection_identity"] for group in groups} == set(selected), "selection bundle is incomplete")
    _check(all(selection["contract"]["sources"] == contract["sources"] for selection in scopes)
           and manifest["source_inputs"] == contract["sources"],
           "final manifest source binding differs")
    _check(manifest.get("runtime", {}).get("runtime_id") == contract["runtime_identity"], "final runtime differs")
    destinations = request["destinations"]
    _check(type(destinations) is dict and set(destinations) == set(contract["original_inputs"])
           and len(set(destinations.values())) == len(destinations), "complete unambiguous original destination mapping required")
    root = Path(request["artifact_root"]).resolve(strict=True)
    artifacts = {}
    for name, recorded in manifest["artifacts_sha256"].items():
        path = (root / name).resolve(strict=True)
        _check(path.is_relative_to(root) and _hash(path) == recorded, "final artifact differs or escapes its directory")
        artifacts[name] = str(path)
    for selection, current in zip(scopes, contracts, strict=True):
        if variant != "native" and (selection.get("capture") or selection.get("selection") == "resident"):
            for name, recorded in current["worker_artifacts"].items():
                _check(manifest["artifacts_sha256"].get(name) == recorded, "prepared numerical worker differs from final artifact")
    expected = {str(Path(destinations[name]).resolve()): recorded for name, recorded in contract["original_inputs"].items()}
    for source, replacement in manifest["sources"].items():
        _check(source in contract["sources"] and replacement["sha256"] == contract["sources"][source], "foreign replacement")
        expected[str(Path(destinations[source]).resolve())] = manifest["artifacts_sha256"][replacement["replacement"]]
    for name, recorded in expected.items():
        _check(_hash(name) == recorded, "application destination does not contain approved source: " + name)
    units = {}
    for unit in manifest["build_sources"]:
        if unit["role"] == "original_source":
            continue
        _check(unit["language"] in {"fortran", "cuda"} and unit["path"] in artifacts, "unsupported generated build source")
        path = artifacts[unit["path"]]
        _check(path not in units, "duplicate generated build source")
        units[path] = {"language": unit["language"], "sha256": manifest["artifacts_sha256"][unit["path"]]}
    macros = {}
    for selection in scopes:
        if variant == "native" or not selection.get("capture") and selection.get("selection") == "native":
            continue
        name = "FORT_EXACT_WINDOW_BUILD_" + selection["contract_sha256"]
        selectors = [path for path, unit in units.items() if unit["language"] == "cuda" and name in _bytes(path).decode()]
        _check(len(selectors) == 1 and selectors[0] not in macros, "one distinct compiler-issued attestation guard per selection required")
        macros[selectors[0]] = name + "=1"
    _check(macros or variant == "native", "no emitted application selector requires build attestation")
    value = {"schema_version": 1, "protocol": PROTOCOL, "request": deepcopy(request), "contracts": contracts,
             "selection_contract_identities": sorted(identities), "variant": variant,
             "expected_inputs": expected, "generated": units, "macros": macros,
             "artifacts": {artifacts[k]: v for k, v in manifest["artifacts_sha256"].items()},
             "manifest_identity": digest(manifest)}
    return {**value, "identity": digest(value)}


def attest_application_build(preflight, receipt, receipt_sha256):
    """Recheck actual ordered compiler/link commands before delivering a binary."""
    _identified(preflight)
    _check(preflight_application_build(preflight["request"]) == preflight, "preflight source/manifest changed")
    value = _receipt(receipt, receipt_sha256)
    contract = preflight["contracts"][0]
    build_root = preflight["request"]["build_root"]
    _check(Path(build_root).resolve() == Path(value["binary"]).parent
           and all(Path(unit["object"]).is_relative_to(Path(build_root).resolve()) for unit in value["units"]),
           "declared build root differs from executed outputs")
    mapping = {str(Path(path).resolve()): source for source, path in preflight["request"]["destinations"].items()}
    seen_original, seen_generated = set(), set()
    for unit in value["units"]:
        source = unit["source"]
        if source in mapping:
            original = mapping[source]
            _check(original in contract["original_units"] and original not in seen_original, "unexpected original object")
            seen_original.add(original)
            expected = contract["original_units"][original]
            actual = _unit_recipe(unit, build_root, source_mapping=mapping)
        else:
            _check(source in preflight["generated"] and source not in seen_generated, "unexpected generated object")
            seen_generated.add(source)
            expected = contract["generated"][preflight["generated"][source]["language"]]
            actual = _unit_recipe(unit, build_root, macro=preflight["macros"].get(source),
                                  source_mapping=mapping)
        _check(actual == expected, "executed compiler, ordered argv, environment or inputs differ: " + source)
    _check(seen_original == set(contract["original_units"]) and seen_generated == set(preflight["generated"]),
           "missing original or generated object")
    expected_inputs = {**preflight["expected_inputs"], **preflight["artifacts"]}
    for name, recorded in value["source_inputs"].items():
        if Path(name).suffix.lower() in SOURCE_SUFFIXES | HEADER_SUFFIXES:
            _check(expected_inputs.get(name) == recorded, "unapproved source/header input: " + name)
    for name, recorded in preflight["expected_inputs"].items():
        _check(value["source_inputs"].get(name) == recorded, "original source/header absent from build inventory")
    for name, recorded in preflight["artifacts"].items():
        if Path(name).suffix.lower() in HEADER_SUFFIXES:
            _check(value["source_inputs"].get(name) == recorded, "compiler-owned header absent from build inventory")
    _check(_link_recipe(value, build_root, mapping) == contract["link"], "actual final link policy differs")
    result = {"schema_version": 1, "protocol": PROTOCOL, "preflight": preflight,
              "receipt": str(Path(receipt).resolve()), "receipt_sha256": receipt_sha256,
              "receipt_identity": value["identity"], "binary": value["binary"],
              "binary_sha256": value["files_sha256"][value["binary"]],
              "contract_identity": digest(sorted(item["identity"] for item in preflight["contracts"])),
              "contract_identities": sorted(item["identity"] for item in preflight["contracts"]),
              "selection_contract_identities": preflight["selection_contract_identities"], "variant": preflight["variant"],
              "limitations": [LIMITATION]}
    return {**result, "identity": digest(result)}


def validate_application_build_observation(observation_build, *, contract, mode, selection_contract_identity=None):
    """Reconstruct a measured build attestation; a claimed success flag is insufficient."""
    _contract(contract)
    _check(type(observation_build) is dict and set(observation_build) == {
        "receipt", "receipt_sha256", "attestation", "attestation_sha256", "binary_sha256", "contract_identity", "variant"},
        "complete executed observation build required")
    _check(mode in {"native", "resident"} and observation_build["variant"] == mode, "observation build variant differs")
    scope = _VALIDATION.get()
    key = digest({"observation": observation_build, "contract": contract, "mode": mode})
    if scope is not None and key in scope["observations"]:
        checked = scope["observations"][key]
        _check(selection_contract_identity is None or _sha(selection_contract_identity)
               and selection_contract_identity in checked["selection_contract_identities"],
               "observation selection contract differs")
        return checked["identity"]
    attestation = _identified(_read(observation_build["attestation"], observation_build["attestation_sha256"]))
    _check(contract in attestation["preflight"]["contracts"], "observation expected build differs")
    if selection_contract_identity is not None:
        _check(_sha(selection_contract_identity)
               and selection_contract_identity in attestation["selection_contract_identities"],
               "observation selection contract differs")
    actual = attest_application_build(attestation["preflight"], observation_build["receipt"], observation_build["receipt_sha256"])
    _check(actual == attestation and all(actual[name] == observation_build[name] for name in
                                      ("binary_sha256", "contract_identity", "variant")), "observation attestation differs")
    if scope is not None:
        scope["observations"][key] = {name: actual[name] for name in ("identity", "selection_contract_identities")}
    return actual["identity"]


def _reference(reference):
    _check(type(reference) is dict and set(reference) == {"path", "sha256"}
           and type(reference["path"]) is str and Path(reference["path"]).is_absolute(),
           "absolute diagnostic receipt reference required")
    return _read(reference["path"], reference["sha256"])


def _command_files(command, role):
    values = command[role]
    _check(type(values) is list and all(type(item) is dict and set(item) == {"path", "sha256"}
           and type(item["path"]) is str and Path(item["path"]).is_absolute()
           and _sha(item["sha256"]) for item in values), "invalid diagnostic command file inventory")
    result = {item["path"]: item["sha256"] for item in values}
    _check(len(result) == len(values), "duplicate diagnostic command input/output")
    return result


def _operand(argv, path, cwd):
    indices = [index for index, arg in enumerate(argv[1:], 1)
               if not arg.startswith("-") and _path(arg, cwd) == path]
    _check(len(indices) == 1, "diagnostic replacement requires one exact original operand")
    return indices[0]


def validate_diagnostic_relink(observation, *, application_build, contract, driver, mode):
    """Authenticate one reviewed diagnostic source replacement and its relink.

    The source hash authorizes the explicitly trusted validation driver; it does
    not prove the driver's numerical noninterference. Every other linked object
    and ordered compiler/link argument is preserved. Diagnostic module inputs
    are freshly checked, but existing application receipts do not establish an
    original compile-time module inventory or an implicit toolchain closure.
    """
    authority = validate_application_build_observation(application_build, contract=contract, mode=mode)
    _check(type(driver) is dict and set(driver) == {"original_source", "original_sha256", "replacement_sha256"}
           and type(driver["original_source"]) is str and _sha(driver["original_sha256"])
           and _sha(driver["replacement_sha256"]), "source-bound diagnostic driver required")
    original = driver["original_source"]
    _check(original in contract["original_units"] and contract["original_inputs"].get(original) == driver["original_sha256"]
           and _hash(original) == driver["original_sha256"], "diagnostic driver original source differs")
    _check(type(observation) is dict and set(observation) == {"plan", "build", "commands"}
           and type(observation["commands"]) is list and len(observation["commands"]) == 2,
           "diagnostic plan, build and two executed command receipts required")
    plan, build = _reference(observation["plan"]), _reference(observation["build"])
    commands = [_reference(item) for item in observation["commands"]]
    _check(len({item["path"] for item in observation["commands"]}) == 2, "duplicate diagnostic command receipt")
    for command in commands:
        _command(command)
        for path, recorded in {**_command_files(command, "inputs"), **_command_files(command, "outputs"),
                               command["compiler"]["path"]: command["compiler"]["sha256"]}.items():
            _check(_sha(recorded) and _hash(path) == recorded, "diagnostic command input/output changed: " + path)
    compiled = [command for command in commands if command["argv"].count("-c") == 1]
    linked = [command for command in commands if "-c" not in command["argv"]]
    _check(len(compiled) == len(linked) == 1, "one diagnostic compile and link required")
    compiled, linked = compiled[0], linked[0]
    attestation = _read(application_build["attestation"], application_build["attestation_sha256"])
    receipt = _read(application_build["receipt"], application_build["receipt_sha256"])
    destination = attestation["preflight"]["request"]["destinations"][original]
    units = [unit for unit in receipt["units"] if unit["source"] == destination]
    _check(len(units) == 1 and receipt["source_inputs"][destination] == driver["original_sha256"],
           "diagnostic driver must replace one unchanged original compilation unit")
    unit = units[0]
    before_compile, before_link = unit["execution"], receipt["link"]
    _check(plan["application_build_identity"] == receipt["identity"]
           and plan["application_build_receipt"] == application_build["receipt"]
           and plan["application_binary"] == receipt["binary"], "diagnostic plan application build differs")
    _check(build.get("status") == "passed" and build["plan_sha256"] == observation["plan"]["sha256"],
           "diagnostic build does not bind its plan")
    for current, previous in ((compiled, before_compile), (linked, before_link)):
        _check(all(current[name] == previous[name] for name in ("cwd", "compiler", "environment")),
               "diagnostic compiler, working directory or environment differs")
    source_index = _operand(before_compile["argv"], destination, before_compile["cwd"])
    object_index = before_compile["argv"].index("-o") + 1
    binary_index = before_link["argv"].index("-o") + 1
    link_object_index = _operand(before_link["argv"], unit["object"], before_link["cwd"])
    _check(len(compiled["argv"]) == len(before_compile["argv"])
           and len(linked["argv"]) == len(before_link["argv"]), "diagnostic ordered arguments differ")
    source, obj, binary = (_path(compiled["argv"][source_index], compiled["cwd"]),
                           _path(compiled["argv"][object_index], compiled["cwd"]),
                           _path(linked["argv"][binary_index], linked["cwd"]))
    _check(len({source, obj, binary}) == 3 and all(path not in receipt["files_sha256"] for path in (source, obj, binary)),
           "diagnostic artifacts overwrite an application input/output")
    expected_compile, expected_link = list(before_compile["argv"]), list(before_link["argv"])
    expected_compile[source_index], expected_compile[object_index] = source, obj
    expected_link[link_object_index], expected_link[binary_index] = obj, binary
    _check(compiled["argv"] == expected_compile and linked["argv"] == expected_link,
           "diagnostic arguments must preserve every other source, object, flag and link position")
    compile_outputs, link_outputs = _command_files(compiled, "outputs"), _command_files(linked, "outputs")
    _check(set(compile_outputs) == {obj} and set(link_outputs) == {binary}
           and _hash(source) == driver["replacement_sha256"], "diagnostic source or command output differs")
    modules = plan["module_inputs"]
    _check(type(modules) is dict and all(type(path) is str and Path(path).is_absolute()
           and Path(path).suffix.lower() in {".mod", ".smod"} and _sha(recorded)
           and _hash(path) == recorded for path, recorded in modules.items()), "diagnostic module inputs differ")
    expected_inputs = _command_files(before_compile, "inputs")
    del expected_inputs[destination]
    expected_inputs[source] = driver["replacement_sha256"]
    for path, recorded in modules.items():
        _check(path not in expected_inputs or expected_inputs[path] == recorded, "diagnostic module identity conflicts")
        expected_inputs[path] = recorded
    _check(_command_files(compiled, "inputs") == expected_inputs, "diagnostic compile inputs differ")
    expected_inputs = _command_files(before_link, "inputs")
    del expected_inputs[unit["object"]]
    expected_inputs[obj] = compile_outputs[obj]
    for path, recorded in modules.items():
        _check(path not in expected_inputs or expected_inputs[path] == recorded, "diagnostic module identity conflicts")
        expected_inputs[path] = recorded
    _check(_command_files(linked, "inputs") == expected_inputs, "diagnostic link inputs differ")
    relink = plan["relink"]
    _check(relink["compile_argv"] == compiled["argv"] and relink["link_argv"] == linked["argv"]
           and relink["cwd"] == compiled["cwd"] == linked["cwd"]
           and plan["diagnostic_binary"] == build["binary"] == binary
           and build["binary_sha256"] == link_outputs[binary], "diagnostic plan/build differs from executed commands")
    for path, recorded in plan["files_sha256"].items():
        _check(_sha(recorded) and _hash(path) == recorded, "diagnostic plan input changed: " + path)
    value = {"binary": binary, "binary_sha256": link_outputs[binary], "application_build_identity": authority}
    return {**value, "identity": digest({"schema_version": 1, "observation": observation,
                                        "driver": driver, "mode": mode, **value})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "attest"))
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        request = json.loads(Path(args.request).read_text())
        result = (preflight_application_build(request) if args.phase == "preflight" else
                  attest_application_build(request["preflight"], request["receipt"], request["receipt_sha256"]))
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": "verified", "identity": result["identity"], "output": args.output}))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
