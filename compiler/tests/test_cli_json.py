"""The public CLI exposes authoritative generation results without IR imports."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = {"generated_code.cu", "generated_cpp_impl.cpp", "generated_interface.f90", "common_functions.cuh"}


def compile_report(tmp_path, body, *arguments, entry="advance"):
    source = tmp_path / "input.f90"
    source.write_text(
        f"module ordinary\ncontains\nsubroutine {entry}(a,b,n,m)\n"
        "real(8),intent(inout)::a(:,:),b(:,:)\ninteger,intent(in)::n,m\n"
        "integer::i,j\nreal(8)::temporary\n"
        f"{body}\nend subroutine\nend module\n"
    )
    output = tmp_path / "generated"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(source),
            "--kernel",
            entry,
            "--output-dir",
            str(output),
            "--fallback",
            "error",
            "--json",
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return result, json.loads(result.stdout), output


def test_json_generation_preserves_ordered_dependences_and_verbose_stderr(tmp_path):
    result, report, output = compile_report(
        tmp_path,
        """
do j=1,m
  do i=1,n
    b(i,j)=a(i,j)*2.0_8
  enddo
enddo
do j=1,m
  do i=2,n
    a(i,j)=b(i-1,j)+3.0_8
  enddo
enddo
""",
        "--verbose",
    )
    assert result.returncode == 0, result.stderr
    assert report["supported"]
    assert report["reason"] is None
    assert report["parallel_regions"] == 2
    assert report["kernel"] == "advance"
    assert set(report["outputs"]) == OUTPUTS
    assert {path.name for path in output.iterdir()} == OUTPUTS
    assert report["execution_plan"].count("parallel legality PROVEN") == 2
    memory = report["memory_plan"]
    assert memory.index("execute: region 0") < memory.index("execute: region 1")
    assert "acquire [pooled]" in memory
    assert "Normalized IR:" in result.stderr


def test_json_generation_reports_conditional_and_imperfect_regions(tmp_path):
    result, report, _ = compile_report(
        tmp_path,
        """
if(n>1) then
  do j=1,m
    temporary=a(1,j)
    do i=2,n
      if(a(i,j)>0.0_8) then
        b(i,j)=a(i,j)+temporary
      else
        b(i,j)=temporary
      endif
    enddo
    b(1,j)=temporary
  enddo
else
  do j=1,m
    b(1,j)=a(1,j)
  enddo
endif
""",
    )
    assert result.returncode == 0, result.stderr
    assert report["parallel_regions"] == 2
    assert "host conditional" in report["execution_plan"]
    assert "retained sequential loops: 1" in report["execution_plan"]
    assert "branch:" in report["memory_plan"]


@pytest.mark.parametrize(
    ("entry", "body", "reason"),
    [
        ("advance", "do i=2,n\na(i,1)=a(i-1,1)\nenddo", "RAW"),
        ("start_hot", "do i=1,n\na(i,1)=2.0_8\nenddo", "public interface"),
    ],
)
def test_json_rejection_does_not_publish_or_replace_outputs(tmp_path, entry, body, reason):
    output = tmp_path / "generated"
    output.mkdir()
    before = {filename: "existing " + filename for filename in OUTPUTS}
    for filename, text in before.items():
        (output / filename).write_text(text)
    result, report, _ = compile_report(tmp_path, body, entry=entry)
    assert result.returncode != 0
    assert report["supported"] is False
    assert reason in report["reason"]
    assert report["outputs"] == []
    assert {path.name: path.read_text() for path in output.iterdir()} == before


def test_json_generation_distinguishes_valid_host_only_entry(tmp_path):
    result, report, _ = compile_report(tmp_path, "a(1,1)=b(1,1)+1.0_8")
    assert result.returncode == 0, result.stderr
    assert report["supported"]
    assert report["parallel_regions"] == 0


def test_json_candidate_list_retains_existing_record_shape(tmp_path):
    compile_report(tmp_path, "do i=1,n\na(i,1)=2.0_8\nenddo")
    result = subprocess.run(
        [sys.executable, "-m", "compiler", "--input", str(tmp_path / "input.f90"), "--list-candidates", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    (record,) = json.loads(result.stdout)
    assert set(record) == {"module", "name", "qualified_name", "path", "line", "annotated", "supported", "reason"}
    assert record["supported"]
