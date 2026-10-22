"""Complete original numerical teams are also graph task roots."""

import shutil
import subprocess

import pytest

from compiler.offload.config import OffloadConfig
from compiler.scopes.numerical_roots import admit_numerical_roots
from compiler.tests.test_closed_selection import SOURCE, builder


@pytest.mark.native
@pytest.mark.parametrize("explicit_sharing", [False, True], ids=["default-shared", "default-none"])
def test_original_parallel_tasks_keep_checked_graph_plans(tmp_path, explicit_sharing):
    source = SOURCE
    if explicit_sharing:
        source = source.replace("!$omp parallel do private", "!$omp parallel do default(none) "
                                "shared(a,b,nx,ny,nz,tx,ty,tz,coefficient) private")
    selected = builder(tmp_path / "original.f90", source,
                       config=OffloadConfig(policy="sections", scope_execution="graph"))
    assert admit_numerical_roots(selected)
    outputs, report = selected.finish()
    assert report["scope_count"] == 2
    for scope in report["scopes"]:
        assert scope["graph_root_kind"] == "original complete numerical task"
        assert scope["graph_execution"]["source_windows"] is False
        assert len(scope["gpu_leaves"]) == 1
    for name, content in outputs.items():
        path = tmp_path / "generated" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] == role and item["language"] == "fortran":
                result = subprocess.run([fortran, "-std=f2018", "-fopenmp", "-c", item["path"]],
                                        cwd=tmp_path / "generated", capture_output=True, text=True, timeout=30)
                assert result.returncode == 0, result.stderr
