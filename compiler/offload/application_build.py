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
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

PROTOCOL = "executed-application-build-v1"
SELECTION_PROTOCOL = "checked-selection-process-v1"
LIMITATION = "Implicit system headers and libraries are not inventoried"
SOURCE_SUFFIXES = {".f", ".for", ".ftn", ".f90", ".f95", ".f03", ".f08", ".c", ".cc", ".cpp", ".cxx", ".cu"}
HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".cuh", ".inc"}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _hash(path):
    with Path(path).open("rb") as stream:
        from hashlib import file_digest

        return file_digest(stream, "sha256").hexdigest()


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
    _check(_sha(expected) and _hash(path) == expected, "receipt or manifest file changed")
    return json.loads(Path(path).read_text())


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
        selectors = [path for path, unit in units.items() if unit["language"] == "cuda" and name in Path(path).read_text()]
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


def validate_application_build_observation(observation_build, *, contract, mode):
    """Reconstruct a measured build attestation; a claimed success flag is insufficient."""
    _contract(contract)
    _check(type(observation_build) is dict and set(observation_build) == {
        "receipt", "receipt_sha256", "attestation", "attestation_sha256", "binary_sha256", "contract_identity", "variant"},
        "complete executed observation build required")
    _check(mode in {"native", "resident"} and observation_build["variant"] == mode, "observation build variant differs")
    attestation = _identified(_read(observation_build["attestation"], observation_build["attestation_sha256"]))
    _check(contract in attestation["preflight"]["contracts"], "observation expected build differs")
    actual = attest_application_build(attestation["preflight"], observation_build["receipt"], observation_build["receipt_sha256"])
    _check(actual == attestation and all(actual[name] == observation_build[name] for name in
                                      ("binary_sha256", "contract_identity", "variant")), "observation attestation differs")
    return actual["identity"]


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
