"""The public shared ABI never publishes or overwrites an ordinary backend."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).parent / "fixtures/fill_array.f90"
ORDINARY = ("generated_code.cu", "generated_cpp_impl.cpp", "generated_interface.f90")


def command(directory):
    return [sys.executable, "-m", "compiler", "--input", str(SOURCE), "--kernel", "fill_array",
            "--memory-model", "scoped", "--gpu-policy", "sections", "--json", "--output-dir", str(directory)]


def test_shared_manifest_cannot_select_stale_ordinary_artifacts(tmp_path):
    for name in ORDINARY:
        (tmp_path / name).write_text("previous ordinary artifact\n")
    result = subprocess.run(command(tmp_path), cwd=ROOT, text=True, capture_output=True,
                            check=False, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["artifact_schema_version"] == 2
    assert report["memory_model"] == "scoped"
    assert "offload" not in report
    assert set(ORDINARY).isdisjoint(report["outputs"])
    units = report["build_sources"]
    assert all(x["role"] in {"common_runtime", "shared_entry"} for x in units)
    assert {x["path"] for x in units if x["role"] == "shared_entry"} == {
        "shared_entry.cu", "shared_interface.f90"}
    assert all((tmp_path / x["path"]).is_file() for x in units)
    assert all((tmp_path / name).read_text() == "previous ordinary artifact\n" for name in ORDINARY)


def test_default_generation_is_the_shared_graph_backend(tmp_path):
    args = command(tmp_path)
    for option in ("--memory-model", "--gpu-policy"):
        index = args.index(option)
        del args[index:index + 2]
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["memory_model"] == "scoped"
    assert report["scoped"]["entry_abi_version"] == 3
    assert "offload" not in report
    assert not set(ORDINARY) & set(report["outputs"])


@pytest.mark.parametrize("option,value", [("--memory-model", "call"),
    ("--scope-transfers", "pipelined"),
    ("--gpu-policy", "always"), ("--gpu-policy", "chunked"), ("--gpu-policy", "hybrid")])
def test_retired_execution_paths_fail_before_publication(tmp_path, option, value):
    sentinel = tmp_path / "preserved"
    sentinel.write_text("caller-owned data")
    result = subprocess.run([*command(tmp_path), option, value], cwd=ROOT,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert "removed" in result.stderr
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "caller-owned data"


@pytest.mark.parametrize("option", ["--cuda-output", "--cpp-output", "--fortran-output"])
def test_shared_abi_rejects_removed_ordinary_output_options_before_writes(tmp_path, option):
    result = subprocess.run([*command(tmp_path), option, "unused.out"], cwd=ROOT,
                            text=True, capture_output=True, check=False, timeout=30)
    assert result.returncode != 0
    assert "ordinary --cuda-output/--cpp-output/--fortran-output options do not apply" in result.stderr
    assert not list(tmp_path.iterdir())
