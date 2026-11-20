"""Prepare and check the pinned ELMM and PoisFFT sources every comparison builds from, without changing upstream checkouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "source_basis.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def application_paths(app_root):
    root = Path(app_root).resolve(strict=True)
    candidates = [root / name for name in ("elmm", "ELMM") if (root / name / "src/SConstruct").is_file()]
    if len(candidates) != 1 or not (root / "PoisFFT/src/SConstruct").is_file():
        raise ValueError("app_root must contain one ELMM (elmm/ or ELMM/) and sibling PoisFFT/")
    return {"elmm": candidates[0], "PoisFFT": root / "PoisFFT"}


def contract_path(paths, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2 or relative.parts[0] not in paths:
        raise ValueError("invalid portable contract source: " + name)
    path = (paths[relative.parts[0]] / Path(*relative.parts[1:])).resolve(strict=True)
    if not path.is_relative_to(paths[relative.parts[0]].resolve()):
        raise ValueError("contract source escapes its application tree: " + name)
    return path


def verify_contract(app_root, contract):
    """Check every pinned source, include and build file hash of the source basis."""
    paths = application_paths(app_root)
    if contract.get("schema_version") != 2 or not contract.get("source_sha256"):
        raise ValueError("unsupported or incomplete source basis")
    hashes = {}
    for table in ("source_sha256", "include_sha256", "build_sha256"):
        for name, expected in contract[table].items():
            path = contract_path(paths, name)
            if digest(path) != expected:
                raise ValueError("source basis hash mismatch: " + name)
            hashes[str(path)] = expected
    return paths, hashes


def repository_provenance(paths, contract, command):
    """Require pinned pristine checkouts, including files the basis does not hash."""
    repositories = {}
    for name, root in paths.items():
        def git(*arguments):
            text, _elapsed = command(["git", "-C", root, *arguments], timeout=30)
            return text.rstrip("\n")

        if Path(git("rev-parse", "--show-toplevel")).resolve() != root.resolve():
            raise ValueError("application source must be its own Git checkout: " + name)
        revision = git("rev-parse", "--verify", "HEAD")
        if revision != contract["repositories"][name]["revision"]:
            raise ValueError("application repository revision mismatch: " + name)
        if git("status", "--porcelain=v1", "--untracked-files=no", "--ignore-submodules=none"):
            raise ValueError("application repository has changed tracked inputs: " + name)
        # Include ignored files in this query. An ignored Fortran/include/build
        # input must not silently enter the copied application either.
        omitted = []
        for relative in filter(None, git("ls-files", "--others", "-z").split("\0")):
            path = Path(relative)
            if (path.suffix not in {".o", ".mod", ".pyc"}
                    and "__pycache__" not in path.parts and not path.name.startswith(".sconsign")):
                raise ValueError("application repository has untracked input: " + name + "/" + relative)
            omitted.append(relative)
        hashes = {}
        for relative in filter(None, git("ls-files", "-z").split("\0")):
            path = contract_path(paths, name + "/" + relative)
            if not path.is_file():
                raise ValueError("unsupported tracked application input: " + name + "/" + relative)
            hashes[relative] = digest(path)
        repositories[name] = {**contract["repositories"][name], "head": revision, "tracked_clean": True,
                              "tracked_sha256": hashes, "omitted_untracked_artifacts": sorted(omitted)}
    return {"schema_version": 1, "repositories": repositories}


def preparation_receipt(root):
    root = Path(root)
    return root.parent / (root.name + "-preparation.json")


def prepare_application_basis(upstream_root, destination, *, contract_path=None):
    """Create clean checkouts of the pinned revisions, or strictly validate existing ones.

    The sources are the upstream ones, unchanged: both executables of a
    comparison are built from them.
    """
    contract_path = Path(contract_path or DEFAULT_CONTRACT)
    contract = json.loads(contract_path.read_text())
    destination = Path(destination).resolve()
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    commands = []

    def command(argv, *, timeout=120):
        argv = [str(value) for value in argv]
        started = time.monotonic()
        result = subprocess.run(argv, text=True, capture_output=True, env=environment, timeout=timeout, check=False)
        elapsed = time.monotonic() - started
        commands.append({"argv": argv, "returncode": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError("application source preparation failed: " + result.stderr)
        return result.stdout, elapsed

    def git(root, *arguments):
        text, _elapsed = command(["git", "-c", "core.hooksPath=/dev/null", "-C", root, *arguments])
        return text.rstrip("\n")

    def validate():
        paths, _hashes = verify_contract(destination, contract)
        return repository_provenance(paths, contract, command)

    if destination.exists():
        validate()
        return destination
    upstream = application_paths(upstream_root)
    if any(destination == root or destination.is_relative_to(root) for root in upstream.values()):
        raise ValueError("prepared application basis must stay outside upstream repositories")
    receipt_path = preparation_receipt(destination)
    if receipt_path.exists():
        raise ValueError("prior application preparation evidence exists; choose a new destination")
    destination.mkdir(parents=True)
    receipt = {"schema_version": 2, "status": "in_progress", "no_builds_or_applications": True,
               "recipe_sha256": digest(__file__), "contract_sha256": digest(contract_path), "commands": commands}
    try:
        for name, origin in upstream.items():
            target = destination / name
            command(["git", "-c", "core.hooksPath=/dev/null", "clone", "--no-checkout", "--no-hardlinks",
                     "--", origin, target])
            git(target, "checkout", "--detach", contract["repositories"][name]["revision"])
            if git(target, "status", "--porcelain=v1", "--untracked-files=all"):
                raise ValueError("pinned upstream checkout is not clean")
        receipt.update(status="prepared_clean_common_basis", repositories=validate())
    except BaseException as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True,
                        help="Prepared upstream elmm/PoisFFT siblings; neither repository is changed")
    parser.add_argument("--output", type=Path, required=True,
                        help="New clean comparison checkouts, or an existing basis to validate")
    args = parser.parse_args()
    print(prepare_application_basis(args.upstream_root, args.output))


if __name__ == "__main__":
    main()
