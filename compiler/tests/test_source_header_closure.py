"""Numerical entry packages carry their complete quoted header dependencies."""
import shutil
import subprocess

import pytest

from compiler.emission.common.resources import read_scoped_runtime
from compiler.ir import CompilationError
from compiler.scopes.source import _quoted_include_closure
from compiler.tests.test_source_scopes import generate


def test_local_header_closure_handles_nested_includes_and_cycles():
    artifacts = {"worker.cpp": '#include "nested/first.hpp"\n',
                 "nested/first.hpp": '#include "../second.hpp"\n',
                 "second.hpp": '#include "nested/first.hpp"\n#include <vector>\n',
                 "unused.hpp": "unused\n"}
    assert set(_quoted_include_closure(artifacts, ["worker.cpp"])) == set(artifacts) - {"unused.hpp"}
    with pytest.raises(CompilationError, match="missing generated local include"):
        _quoted_include_closure({"worker.cpp": '#include "missing.hpp"\n'}, ["worker.cpp"])
    with pytest.raises(CompilationError, match="escapes its package"):
        _quoted_include_closure({"worker.cpp": '#include "../outside.hpp"\n'}, ["worker.cpp"])


@pytest.mark.native
def test_common_runtime_compiles_with_only_manifest_headers(tmp_path):
    compiler = shutil.which("g++-14") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    outputs, manifest = read_scoped_runtime()
    for name in ["scoped_runtime.cu", *manifest["headers"]]:
        (tmp_path / name).write_text(outputs[name])
    result = subprocess.run([compiler, "-std=c++17", "-fopenmp", "-DFORT_SCOPE_CPU_TEST",
                             "-fsyntax-only", "-x", "c++", "scoped_runtime.cu"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.native
def test_ordinary_entry_headers_compile_without_aggregate_include_directory(tmp_path):
    compiler = shutil.which("g++-14") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    _, output, _ = generate(tmp_path / "source", scope_execution="graph", entry="original::step")
    entries = sorted(output.glob("entries/*/scoped_entry.hpp"))
    assert entries, "fixture must exercise ordinary numerical entry packaging"
    for number, header in enumerate(entries):
        isolated = tmp_path / ("isolated-" + str(number))
        shutil.copytree(header.parent, isolated)
        assert (isolated / "scoped_regions.hpp").is_file()
        source = isolated / "header-check.cpp"
        source.write_text('#include "scoped_entry.hpp"\n#include "scoped_team_observer.hpp"\nint main() { return 0; }\n')
        result = subprocess.run([compiler, "-std=c++17", "-fopenmp", "-fsyntax-only", source.name],
                                cwd=isolated, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
