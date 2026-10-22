"""Independent graph roots share one original source and compiled runtime."""

from __future__ import annotations

import copy
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.compose import _apply, compose_graph_scopes
from compiler.scopes.source import form_source_scopes


@pytest.fixture
def components(tmp_path):
    path = tmp_path / "original.f90"
    routines = """subroutine ROOT(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=a(i)+2.0_8
enddo
end subroutine
"""
    path.write_text(
        "module graph_roots\ncontains\n"
        + routines.replace("ROOT", "first")
        + routines.replace("ROOT", "second")
        + "end module\n"
    )
    facts = {
        "schema_version": 1,
        "participation": "serial",
        "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
        "captures": {
            "argument::a": {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
        },
    }
    options = CompilerOptions(gpu_policy="sections", memory_model="scoped", scope_execution="graph")
    return [
        form_source_scopes(
            [path],
            "graph_roots::" + root,
            facts=facts,
            options=options,
            config=OffloadConfig(policy="sections", scope_execution="graph"),
        )
        for root in ("first", "second")
    ]


@pytest.mark.native
def test_composed_roots_compile_one_shared_original_source(tmp_path, components):
    outputs, report = compose_graph_scopes(components)
    assert report["scope_count"] == 2
    assert report["entry"] == ["graph_roots::first", "graph_roots::second"]
    assert len([item for item in report["build_sources"] if item["role"] == "original_source"]) == 1
    assert len([item for item in report["build_sources"] if item["path"] == "scoped_runtime.cu"]) == 1
    for name, text in outputs.items():
        path = tmp_path / "composed" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] == role and item["language"] == "fortran":
                result = subprocess.run(
                    [fortran, "-std=f2018", "-fopenmp", "-c", item["path"]],
                    cwd=tmp_path / "composed",
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert result.returncode == 0, result.stderr


def test_composition_rejects_changed_artifact_before_build(components):
    damaged = copy.deepcopy(components)
    outputs, report = damaged[1]
    source = next(item["path"] for item in report["build_sources"] if item["role"] == "shared_entry")
    outputs[source] += "\n! tampered\n"
    with pytest.raises(CompilationError, match="artifact does not match"):
        compose_graph_scopes(damaged)


def test_composition_rechecks_all_original_input_bytes(components):
    from pathlib import Path

    original = Path(next(iter(components[0][1]["source_inputs"])))
    original.write_text(original.read_text() + "! changed after compilation\n")
    with pytest.raises(CompilationError, match="source changed"):
        compose_graph_scopes(components)


@pytest.mark.parametrize("insert", [False, True])
def test_composition_rejects_conflicting_original_spans(insert):
    edits = [
        {"first_line": 2, "last_line": 4, "replacement": "replacement\n"},
        {"first_line": 3, "last_line": 2 if insert else 5, "replacement": "conflict\n"},
    ]
    with pytest.raises(CompilationError, match="inside|overlap"):
        _apply("a\nb\nc\nd\ne\nf\n", edits)
