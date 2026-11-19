"""Read a saved scope package the way a build consumer does.

Revalidates the manifest and every artifact it names against their recorded
hashes, and returns the runtime description, the sources to build with their
roles, and the hashes of the shared artifacts.
"""
import hashlib
import json
import re
from pathlib import Path

from compiler.numerical_contract import numerical_build_contract, require_numerical_build_contract

from .checked_execution_plan import validate_checked_execution_plans


class PackageError(RuntimeError):
    pass


def _checked_execution_plans(manifest, *, capture_facts=None):
    try:
        validate_checked_execution_plans(manifest, capture_facts=capture_facts)
    except ValueError as error:
        raise PackageError(str(error)) from error


def _backend_request(request):
    if (not isinstance(request, dict) or set(request) != {"manifest", "sha256"}
            or not isinstance(request["manifest"], str) or not Path(request["manifest"]).is_absolute()
            or "\0" in request["manifest"] or not isinstance(request["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", request["sha256"]) is None):
        raise ValueError("Pressure backend requires a SHA-bound prepared manifest")
    return dict(request)


def combine_pressure_backends(requests):
    """Combine explicit checked requests; never choose a provider by its name."""
    result = None
    for request in requests:
        if request is None:
            continue
        request = _backend_request(request)
        if result is not None and result != request:
            raise ValueError("Scope packages require one identical pressure backend")
        result = request
    return result


def pressure_backend_from_manifest(manifest):
    """Project build inputs only after checking the complete issued epoch.

    The duplicated source-facing external task list is not a build authority.
    Numerical/source/execution authority stays with the compiler's issued plan.
    """
    validate_checked_execution_plans(manifest)
    requests = []
    for scope in manifest.get("scopes", []):
        if not isinstance(scope, dict):
            continue
        plan = scope.get("checked_execution_plan", {})
        for member in plan.get("storage_epoch", {}).get("members", []):
            for task in member.get("device_tasks", []):
                request = task["provider"].get("pressure_backend")
                if request is not None:
                    requests.append(request)
    return combine_pressure_backends(requests)


def _pressure_backend(manifest):
    try:
        return pressure_backend_from_manifest(manifest)
    except ValueError as error:
        raise PackageError(str(error)) from error


def _numerical_contract(contract):
    require_numerical_build_contract(contract)
    return numerical_build_contract()


def _numerical_build_fields(item):
    if not isinstance(item, dict):
        raise PackageError("Compiler numerical build metadata requires a source descriptor")
    if "numerical_contract" not in item:
        return {}
    if item.get("language") not in {"cuda", "c++"}:
        raise PackageError("Compiler numerical contract requires a CUDA or C++ build source")
    try:
        return {"numerical_contract": _numerical_contract(item["numerical_contract"])}
    except ValueError as error:
        raise PackageError(str(error)) from error


def _scope_artifact(root, name):
    """Validate a public artifact path without interpreting its source."""
    if not isinstance(name,str) or Path(name).is_absolute():
        raise PackageError("Scope artifacts require relative paths")
    try:
        path=(root/name).resolve(strict=True)
    except OSError as error:
        raise PackageError("Missing compiler scope artifact: " + name) from error
    if not path.is_relative_to(root) or not path.is_file():
        raise PackageError("Scope artifact escapes its output directory: " + name)
    return path


def scope_package_build(metadata):
    """Revalidate saved public artifacts and return their declared build roles."""
    if not isinstance(metadata, dict) or not isinstance(metadata.get("scope_manifest"), dict):
        raise PackageError("Scope composition requires compiler package metadata")
    manifest = metadata["scope_manifest"]
    _checked_execution_plans(manifest)
    if metadata.get("pressure_backend") != _pressure_backend(manifest):
        raise PackageError("Scope pressure backend differs from its checked epoch")
    try:
        root = Path(metadata["artifact_root"]).resolve(strict=True)
    except (KeyError, TypeError, OSError) as error:
        raise PackageError("Scope composition requires an existing artifact directory") from error
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or type(manifest.get("scope_count")) is not int or manifest["scope_count"] < 0
            or not isinstance(manifest.get("source_inputs"), dict) or not isinstance(manifest.get("sources"), dict)
            or not isinstance(manifest.get("artifacts_sha256"), dict)):
        raise PackageError("Scope composition requires a versioned artifact manifest")
    try:
        recorded = json.loads((root / "scope-manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise PackageError("Cannot read saved compiler scope manifest") from error
    if recorded != manifest:
        raise PackageError("Saved compiler scope manifest differs from its package")
    artifact_hashes = {}
    for name, digest in manifest["artifacts_sha256"].items():
        artifact = _scope_artifact(root, name)
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
            raise PackageError("Approved scope artifact changed: " + name)
        artifact_hashes[artifact.relative_to(root).as_posix()] = digest
    units = manifest.get("build_sources", [])
    if not isinstance(units, list):
        raise PackageError("Scope composition requires public build roles")
    runtime = manifest.get("runtime")
    if manifest.get("scope_count", 0):
        if (not isinstance(runtime, dict) or runtime.get("schema_version") != 1
                or runtime.get("abi_version") != 1 or runtime.get("link_once") is not True
                or not isinstance(runtime.get("source_sha256"), dict) or not runtime["source_sha256"]
                or not isinstance(runtime.get("sources"), list)):
            raise PackageError("Scope composition requires a compatible link-once runtime")
        hashes = runtime["source_sha256"]
        if any(not isinstance(name, str) or not isinstance(digest, str) for name, digest in hashes.items()):
            raise PackageError("Scope runtime identity requires source hashes")
        identity = hashlib.sha256("\n".join(f"{name}:{digest}" for name, digest in sorted(hashes.items())).encode()).hexdigest()
        if (runtime.get("runtime_id") != identity
                or any(manifest["artifacts_sha256"].get(name) != digest for name, digest in hashes.items())):
            raise PackageError("Scope runtime identity differs from its approved artifacts")
        if any(not isinstance(item, dict) or not isinstance(item.get("path"), str)
               or item["path"] not in hashes or item.get("language") not in {"cuda", "fortran"}
               for item in runtime["sources"]):
            raise PackageError("Scope runtime sources require public paths and languages")
        for item in runtime["sources"]:
            _numerical_build_fields(item)
        expected_runtime = {(item["path"], item["language"]) for item in runtime["sources"]}
        if len(expected_runtime) != len(runtime["sources"]):
            raise PackageError("Scope runtime source descriptors must not repeat")
    else:
        if runtime is not None:
            raise PackageError("Native scope package unexpectedly declares a runtime")
        expected_runtime = set()
    actual_runtime, build, replacements = set(), [], set()
    for item in units:
        if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
                or item["path"] not in manifest["artifacts_sha256"]
                or item.get("language") not in {"cuda", "fortran"}
                or item.get("role") not in {"common_runtime", "shared_entry", "original_source"}):
            raise PackageError("Scope build source is missing a public artifact/role")
        name, language, role = (item[key] for key in ("path", "language", "role"))
        numerical = _numerical_build_fields(item)
        source = _scope_artifact(root, name)
        if role == "common_runtime":
            actual_runtime.add((name, language))
            if runtime is None or name not in runtime["source_sha256"]:
                raise PackageError("Common runtime build source lacks runtime identity")
            declared = next((entry for entry in runtime["sources"]
                             if (entry["path"], entry["language"]) == (name, language)), None)
            if declared is None or _numerical_build_fields(declared) != numerical:
                raise PackageError("Common runtime numerical build contract differs from its declared source")
        if role == "original_source":
            replacements.add(source.relative_to(root).as_posix())
        build.append({"path": source.relative_to(root).as_posix(), "source": str(source),
                      "language": language, "role": role, "sha256": manifest["artifacts_sha256"][name],
                      **numerical})
    if actual_runtime != expected_runtime:
        raise PackageError("Common runtime build roles differ from its declared sources")
    # Complete original-source replacements compose by destination below;
    # their incidental artifact paths do not identify shared helper code.
    return runtime, build, {path: digest for path, digest in artifact_hashes.items() if path not in replacements}
