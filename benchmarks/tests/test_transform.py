"""Shared recipe I/O must protect sources and record the exact generated code."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.harness import transform


@pytest.mark.parametrize("alias", [False, True])
def test_rejects_overwriting_input(tmp_path, alias):
    source = tmp_path / "kernel.f90"
    source.write_text("original Fortran\n")
    output = tmp_path / "alias.f90" if alias else source
    if alias:
        output.symlink_to(source)
    with pytest.raises(ValueError, match="Output must differ from input"):
        transform.run("loki", source, "CDU", output)
    assert source.read_text() == "original Fortran\n"


def test_rejects_provenance_overwriting_input(tmp_path):
    source = tmp_path / "kernel.json"
    source.write_text("original Fortran\n")
    with pytest.raises(ValueError, match="Output must differ from input"):
        transform.run("psyclone", source, "CDU", source.with_suffix(".f90"))
    assert source.read_text() == "original Fortran\n"


@pytest.mark.parametrize(("tool", "vector_length"), [("loki", 64), ("psyclone", None)])
def test_shared_manifest_and_options(tmp_path, monkeypatch, tool, vector_length):
    original, generated = "original Fortran\n", "transformed Fortran\n"
    source, output = tmp_path / "input.f90", tmp_path / "build/kernel.f90"
    source.write_text(original)

    def recipe(code, case, **options):
        assert code == original
        assert case == "CDU"
        assert options == {"target": "cpu", "fuse": False, "vector_length": vector_length or 128}
        return generated, {"tool": tool, "loop_nests": 4}

    def import_recipe(name):
        assert name == f"benchmarks.tools.{tool}"
        return SimpleNamespace(transform=recipe)

    monkeypatch.setattr(transform.importlib, "import_module", import_recipe)
    manifest = transform.run(tool, source, "CDU", output, target="cpu", fuse=False, vector_length=vector_length)
    assert source.read_text() == original
    assert output.read_text() == generated
    assert json.loads(output.with_suffix(".json").read_text()) == manifest
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["generated_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["loop_nests"] == 4
    assert manifest["original_source_edits"] == 0


@pytest.mark.parametrize(("tool", "length"), [("psyclone", 128), ("loki", 0)])
def test_cli_rejects_unsupported_vector_length(monkeypatch, capsys, tool, length):
    monkeypatch.setattr(
        "sys.argv",
        [
            "transform",
            "--tool",
            tool,
            "--input",
            "input.f90",
            "--case",
            "CDU",
            "--output",
            "output.f90",
            "--vector-length",
            str(length),
        ],
    )
    with pytest.raises(SystemExit) as error:
        transform.main()
    assert error.value.code == 2
    assert "--vector-length must be positive and is only supported by Loki" in capsys.readouterr().err
