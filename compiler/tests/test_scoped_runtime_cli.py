"""Runtime artifacts travel through the public compiler CLI, not Python IR."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from compiler.emission.common.resources import read_scoped_runtime

COMPILER = Path(__file__).resolve().parents[1]


def test_runtime_manifest_hashes_and_interfaces():
    sources, manifest = read_scoped_runtime()
    assert manifest["abi_version"] == 1
    assert "#define FORT_SCOPE_ABI_VERSION 1" in sources["scoped_runtime.h"]
    assert manifest["link_once"] is True
    for name, text in sources.items():
        assert hashlib.sha256(text.encode()).hexdigest() == manifest["source_sha256"][name]
    assert all(source["path"] in sources for source in manifest["sources"])


def test_public_runtime_cli_from_independent_checkout(tmp_path):
    checkout = tmp_path / "independent-compiler"
    shutil.copytree(COMPILER, checkout / "compiler", ignore=shutil.ignore_patterns("__pycache__", ".*cache"))
    caller = tmp_path / "independent-caller"
    caller.mkdir()
    destination = caller / "generated"
    environment = {**os.environ, "PYTHONPATH": str(checkout)}
    result = subprocess.run([sys.executable, "-m", "compiler", "--emit-scoped-runtime", "--json",
                             "--output-dir", str(destination)], cwd=caller, env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["supported"] is True
    manifest = json.loads((destination / "scoped-runtime.json").read_text())
    assert report["runtime"] == manifest
    for name, digest in manifest["source_sha256"].items():
        assert (destination / name).is_file()
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
    assert set(report["outputs"]) == set(manifest["source_sha256"]) | {"scoped-runtime.json"}
