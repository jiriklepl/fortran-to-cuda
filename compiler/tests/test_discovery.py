"""Marker-free selection, honest eligibility, and declared REAL kinds."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from compiler.driver.pipeline import prepare_function
from compiler.frontend import discover_file, lower_file
from compiler.ir import CompilationError, ScalarType
from compiler.tests.test_language import run_reference_and_cpp


def write_module(tmp_path, routines, specification=""):
    path = tmp_path / "ordinary.f90"
    path.write_text(f"module ordinary\n{specification}\nimplicit none\ncontains\n{routines}\nend module\n")
    return path


def test_unmarked_helpers_are_inlined_and_unrelated_io_is_lazy(tmp_path):
    source = write_module(
        tmp_path,
        """
subroutine helper(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=a(i)+1.0_8
enddo
end subroutine
subroutine advance(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
call helper(a,n)
end subroutine
subroutine log_state
print *, 'logging is native Fortran'
end subroutine
""",
    )
    function, plan = prepare_function(lower_file(source, "advance"))
    assert len(plan.regions) == 1
    assert function.body.statements[0].location.call_stack
    records = {item.name: item for item in discover_file(source)}
    assert records["helper"].lowerable
    assert records["advance"].lowerable
    assert not any(item.annotated for item in records.values())
    assert not records["log_state"].lowerable
    assert "Print_Stmt" in records["log_state"].reason


@pytest.mark.parametrize(
    ("specification", "kind", "expected"),
    [
        ("", "4", ScalarType.REAL32),
        ("", "8", ScalarType.REAL),
        ("integer,parameter::knd=kind(1.0)", "knd", ScalarType.REAL32),
        ("integer,parameter::precision=kind(1.0d0), wp=precision", "wp", ScalarType.REAL),
        ("use, intrinsic::iso_fortran_env, only: wp=>real64", "wp", ScalarType.REAL),
        ("use, intrinsic::iso_c_binding, only: wp=>c_float", "wp", ScalarType.REAL32),
    ],
)
def test_real_precision_comes_from_declarations(tmp_path, specification, kind, expected):
    # USE precedes IMPLICIT NONE; PARAMETER follows it in ordinary source.
    path = tmp_path / "kinds.f90"
    path.write_text(
        f"module precision_case\n{specification}\ncontains\n"
        f"subroutine advance(a)\nreal({kind}),intent(out)::a(:)\na(1)=0.25_{kind}\n"
        "end subroutine\nend module\n"
    )
    function = lower_file(path, "advance")
    assert function.parameters[0].dtype is expected
    assert function.body.statements[0].value.dtype is expected


@pytest.mark.parametrize(
    ("specification", "selector", "message"),
    [
        ("", "knd", "unresolved INTEGER kind parameter knd"),
        ("integer,parameter::wp=16", "wp", "unsupported REAL kind 16"),
        ("integer,parameter::wp=unknown", "wp", "unresolved INTEGER kind parameter unknown"),
        ("integer,parameter::wp=other,other=wp", "wp", "cyclic kind parameter"),
        ("real::wp", "wp", "unresolved INTEGER kind parameter wp"),
    ],
)
def test_unresolved_or_unsupported_kind_is_never_assumed_double(tmp_path, specification, selector, message):
    path = tmp_path / "kinds.f90"
    path.write_text(
        f"module example\n{specification}\ncontains\nsubroutine entry(a)\n"
        f"real({selector})::a(:)\na(1)=1\nend subroutine\nend module\n"
    )
    with pytest.raises(CompilationError, match=message):
        lower_file(path, "entry")


def test_kind_scope_is_restored_after_inlining(tmp_path):
    source = write_module(
        tmp_path,
        """
subroutine helper(a)
integer,parameter::wp=4
real(4),intent(out)::a(:)
a(1)=0.25_wp
end subroutine
subroutine entry(a,b)
integer,parameter::wp=8
real(4),intent(out)::a(:)
real(8),intent(out)::b(:)
call helper(a)
b(1)=0.25_wp
end subroutine
""",
    )
    function = lower_file(source, "entry")
    assert [stmt.value.dtype for stmt in function.body.statements] == [ScalarType.REAL32, ScalarType.REAL]


def test_module_qualification_disambiguates_same_named_entries(tmp_path):
    path = tmp_path / "modules.f90"
    path.write_text(
        "\n".join(
            f"module {module}\ncontains\nsubroutine entry\nend subroutine\nend module" for module in ("left", "right")
        )
    )
    with pytest.raises(CompilationError, match="ambiguous"):
        lower_file(path, "entry")
    assert lower_file(path, "RIGHT::ENTRY").module == "right"


def test_cli_discovery_checks_dependence_legality_without_publishing(tmp_path):
    source = write_module(
        tmp_path,
        """
subroutine independent(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=a(i)+1.0_8
enddo
end subroutine
subroutine recurrence(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=2,n
a(i)=a(i-1)+1.0_8
enddo
end subroutine
""",
    )
    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(source),
            "--list-candidates",
            "--json",
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=True,
    )
    records = {record["name"]: record for record in json.loads(result.stdout)}
    assert records["independent"]["supported"]
    assert not records["recurrence"]["supported"]
    assert "RAW" in records["recurrence"]["reason"]
    assert not output.exists()


@pytest.mark.native
@pytest.mark.parametrize("rank", [1, 2])
def test_unannotated_helper_native_execution_matches_fortran(tmp_path, rank):
    dimensions = ":" if rank == 1 else ":,:"
    extents = "7" if rank == 1 else "7,5"
    subscript = "i" if rank == 1 else "i,j"
    body = f"do i=1,size(a,1)\na({subscript})=a({subscript})*2.0_8+0.125_8\nenddo"
    if rank == 2:
        body = f"do j=1,size(a,2)\n{body}\nenddo"
    source = write_module(
        tmp_path,
        f"""
subroutine helper(a)
real(8),intent(inout)::a({dimensions})
integer::i,j
{body}
end subroutine
subroutine advance(a)
real(8),intent(inout)::a({dimensions})
call helper(a)
end subroutine
""",
    )
    function, plan = prepare_function(lower_file(source, "advance"))
    driver = f"""program verify
use ordinary, only: advance
implicit none
real(8)::a({extents})
a=0.1_8
call advance(a)
print *,a
end program
"""
    run_reference_and_cpp(tmp_path, function, plan, driver, openmp=True)
