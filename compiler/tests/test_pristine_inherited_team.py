"""An inherited CPU choice runs the exact original team before generated barriers."""

import re
import shutil
import subprocess

import pytest

from compiler.tests.test_source_query_regions import SOURCE, emit


def source(*, tiled=False):
    text = SOURCE.replace("integer::i,tnx", "integer::i,bi,tnx")
    loop = "do bi=1,n,tnx\ndo i=bi,min(bi+tnx-1,n)" if tiled else "do i=1,n"
    sharing = "a,b,n,tnx,coefficient" if tiled else "a,b,n,coefficient"
    private = "i,bi" if tiled else "i"
    text = text.replace("do i=1,n\nb(i)",
        f"!$omp parallel do private({private}) shared({sharing}) default(none)\n{loop}\nb(i)", 1)
    text = text.replace("b(i)=a(i)*coefficient\nenddo",
        "b(i)=a(i)*coefficient\nenddo\n" + ("enddo\n" if tiled else "") + "!$omp end parallel do", 1)
    text = text.replace("do i=1,n\nc(i)",
        "!$omp parallel do private(i) shared(b,c,n) default(none)\ndo i=1,n\nc(i)", 1)
    return text.replace("c(i)=b(i)+1.0_8\nenddo", "c(i)=b(i)+1.0_8\nenddo\n!$omp end parallel do", 1)


@pytest.mark.parametrize("tiled", [False, True])
def test_native_range_is_acquired_before_pristine_original_fork(tmp_path, tiled):
    _outputs, report, text, proofs = emit(tmp_path, source(tiled=tiled))
    assert len(proofs) == 1
    proof = proofs[0].validate()
    first = next(step for step in proof.steps if step.call is not None)
    alternative = first.scope.public()["combined_numerical_team"]["original_cpu_alternative"]
    assert alternative["available"]
    assert alternative["pristine_inherited_query"]["available"]
    assert alternative["pristine_inherited_query"]["original_span_sha256"]
    assert report["scopes"][0]["query_regions"][0]["workers"] == 2
    start = text.index("fort_team_pristine = .false.")
    acquire = text.index("fort_status = fort_team_native_begin(", start)
    branch = text.index("if (fort_team_pristine) then", acquire)
    assert "if (fort_team_query_plan) then" in text[start:acquire]
    assert text.index("fort_team_native = .false.", start) < acquire
    assert text.index("fort_status = FORT_SCOPE_OK", start) < acquire
    fork = text.index("!$omp parallel do", branch)
    join = text.index("!$omp end parallel do", fork)
    commit = text.index("fort_scope_host_end(", join)
    fallback = text.index("else", commit)
    assert acquire < branch < fork < join < commit < fallback
    assert "!$omp master" not in text[branch:commit]
    assert "!$omp barrier" not in text[branch:commit]
    assert "fort_team_run = .false." in text[fallback:]
    if tiled:
        assert first.scope.effect.preconditions
        assert "fort_numerical_guard" in text[start:acquire]


def test_labeled_original_team_keeps_one_original_label(tmp_path):
    original = source().replace("b(i)=a(i)*coefficient", "100 b(i)=a(i)*coefficient", 1)
    _outputs, report, text, proofs = emit(tmp_path, original)
    # The current original numerical frontend conservatively declines this
    # labeled group already. No retained query may introduce a second label.
    assert not proofs
    assert all(not item.get("query_regions") for item in report["scopes"])
    assert len(re.findall(r"^\s*100\s+b\(", text, re.M)) == 1


@pytest.mark.native
@pytest.mark.parametrize("tiled", [False, True])
def test_pristine_and_generated_default_none_branches_compile(tmp_path, tiled):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if fortran is None:
        pytest.skip("Fortran compiler required")
    outputs, report, _text, _proofs = emit(tmp_path, source(tiled=tiled))
    for name, text in outputs.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] != role or item["language"] != "fortran":
                continue
            result = subprocess.run([fortran, "-O2", "-std=f2018", "-fopenmp", "-c",
                str(tmp_path / item["path"]), "-o", str(tmp_path / (item["path"].replace("/", "_") + ".o"))],
                cwd=tmp_path, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stdout + result.stderr
