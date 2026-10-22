"""Prepare the reviewed common ELMM source basis without changing upstream caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = HERE / "patches/elmm-deterministic-top.patch"
BASIS = {
    "base": "ddb426fca5b020216083f02a218292fea70c1df2",
    "revision": "be38bbbec88756ef0f32d67b44dc412a71e0e6e8",
    "tree": "3c2c2dffbc5bda3c17cbaa8cea5d44798e992131",
    "original_pressure_sha256": "0548a1a1b33284d5811335beff0051c60faf273f1f9535ef737b34be141bf9af",
    "pressure_sha256": "65d6ee76dc4b63b4545a1bfd481dbf875cb168f55a484bac7d0a39b78f21f7d0",
    "patch_sha256": "707f87819df0841831f082a6a915e4d630b767fcaf1a8336cf006fe63a621805",
    "name": "ELMM comparison basis",
    "email": "comparison-basis@example.invalid",
    "date": "1792627200 +0000",
    "message": "Use approved deterministic TOP pressure reference across comparison modes\n",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preparation_receipt(root):
    root = Path(root)
    return root.parent / (root.name + "-preparation.json")


def prepare_application_basis(upstream_root, destination, *, contract_path=None):
    """Create a clean local derivative, or strictly validate an existing one.

    The derivative is reconstructed from a pinned upstream commit and the
    supplied patch. It is not fetched as an alleged upstream revision. Explicit
    --app-root callers bypass this function and retain the normal source checks.
    """
    from sgs_experiment import (
        DEFAULT_CONTRACT,
        application_paths,
        repository_provenance,
        verify_contract,
    )

    contract_path = Path(contract_path or DEFAULT_CONTRACT)
    contract = json.loads(contract_path.read_text())
    if (contract["repositories"]["elmm"]["revision"] != BASIS["revision"]
            or contract["source_sha256"]["elmm/src/pressure.f90"] != BASIS["pressure_sha256"]):
        raise ValueError("application contract differs from the reviewed common source basis")
    if digest(PATCH) != BASIS["patch_sha256"]:
        raise ValueError("approved application source patch changed")
    destination = Path(destination).resolve()
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null",
                       GIT_AUTHOR_NAME=BASIS["name"], GIT_COMMITTER_NAME=BASIS["name"],
                       GIT_AUTHOR_EMAIL=BASIS["email"], GIT_COMMITTER_EMAIL=BASIS["email"],
                       GIT_AUTHOR_DATE=BASIS["date"], GIT_COMMITTER_DATE=BASIS["date"])
    commands = []

    def command(argv, *, timeout=120, data=None):
        argv = [str(value) for value in argv]
        started = time.monotonic()
        result = subprocess.run(argv, input=data, text=True, capture_output=True,
                                env=environment, timeout=timeout, check=False)
        elapsed = time.monotonic() - started
        commands.append({"argv": argv, "returncode": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError("application source preparation failed: " + result.stderr)
        return result.stdout, elapsed

    def git(root, *arguments, data=None):
        text, _elapsed = command(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                                  "-c", "i18n.commitEncoding=UTF-8", "-C", root, *arguments], data=data)
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
    receipt = {"schema_version": 1, "status": "in_progress", "no_builds_or_applications": True,
               "recipe_sha256": digest(__file__), "contract_sha256": digest(contract_path),
               "patch_sha256": digest(PATCH), "identity": BASIS, "commands": commands}
    try:
        for name, origin in upstream.items():
            target = destination / name
            command(["git", "-c", "core.hooksPath=/dev/null", "clone", "--no-checkout", "--no-hardlinks",
                     "--", origin, target])
            revision = BASIS["base"] if name == "elmm" else contract["repositories"][name]["revision"]
            git(target, "checkout", "--detach", revision)
            if git(target, "status", "--porcelain=v1", "--untracked-files=all"):
                raise ValueError("pinned upstream checkout is not clean")
        elmm = destination / "elmm"
        if digest(elmm / "src/pressure.f90") != BASIS["original_pressure_sha256"]:
            raise ValueError("original pressure source differs from the reviewed preimage")
        git(elmm, "apply", "--check", PATCH)
        git(elmm, "apply", PATCH)
        if (digest(elmm / "src/pressure.f90") != BASIS["pressure_sha256"]
                or git(elmm, "diff", "--name-only") != "src/pressure.f90"):
            raise ValueError("approved patch changed unexpected application sources")
        git(elmm, "add", "--", "src/pressure.f90")
        tree = git(elmm, "write-tree")
        if tree != BASIS["tree"]:
            raise ValueError("approved application tree differs from the deterministic identity")
        commit = git(elmm, "commit-tree", tree, "-p", BASIS["base"], data=BASIS["message"])
        if commit != BASIS["revision"]:
            raise ValueError("approved application commit differs from the deterministic identity")
        git(elmm, "update-ref", "refs/heads/approved-deterministic-top", commit)
        git(elmm, "checkout", "--detach", commit)
        if git(elmm, "diff", "--name-only", BASIS["base"], commit) != "src/pressure.f90":
            raise ValueError("approved derivative contains unrelated source changes")
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
