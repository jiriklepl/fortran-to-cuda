"""Whole-source preparation for a resident build: anchors, preprocessing, loop rewrites and entry guards."""
import shutil
from pathlib import Path

import pytest

from compiler.ir import CompilationError
from compiler.resident.anchors import (
    find_all_calls,
    find_call_span,
    find_once,
    find_routine_span,
    find_unique_span,
    parse_fortran,
    replace_once,
)
from compiler.resident.configure import configure_sources
from compiler.resident.rewrites import inline_tile_scratch_nests, promote_line_snapshots
from compiler.scopes.resident_entries import (
    compile_resident_entry,
    export_resident_entries,
    guard_entries,
    pass_scalars_by_value,
)

STEP = """module stepping
  implicit none
contains
  subroutine run_step(a, b, c)
    integer :: a, b, c
    call compute_flux( &
      a, &
      b, c)   ! fluxes first
    if (a > 0) then
      call finish(a)
    end if
  end subroutine run_step
  subroutine finish(a)
    integer :: a
    a = 0
  end subroutine finish
end module stepping
"""


def test_an_anchor_survives_reformatting_and_comments_but_must_be_unique():
    exact = "    if (a > 0) then\n"
    assert STEP[slice(*find_unique_span(STEP, exact))] == exact
    # Different spacing, case and a trailing comment still name the same statement.
    start, end = find_unique_span(STEP, "IF ( a>0 ) THEN   ! positive only\n")
    assert STEP[start:end].strip() == "if (a > 0) then"
    assert find_once(STEP, "call finish(a)") == STEP.index("call finish(a)")
    with pytest.raises(ValueError, match="exactly one occurrence"):
        find_unique_span(STEP, "integer :: a")
    with pytest.raises(ValueError, match="exactly one occurrence"):
        find_unique_span(STEP, "call missing()")
    with pytest.raises(ValueError, match="no flux call"):
        find_unique_span(STEP, "call missing()", message="no flux call")


def test_text_is_replaced_or_inserted_relative_to_an_anchor():
    anchor = "      call finish(a)\n"
    assert "      call finish(b)\n" in replace_once(STEP, anchor, "      call finish(b)\n")
    before = replace_once(STEP, anchor, "      a = a + 1\n", insert=True)
    assert "      a = a + 1\n      call finish(a)\n" in before
    after = replace_once(STEP, anchor, "      a = a + 1\n", insert=True, after=True)
    assert "      call finish(a)\n      a = a + 1\n" in after


def test_procedures_and_calls_are_located_through_the_parse():
    start, end = find_routine_span(STEP, "run_step")
    assert STEP[start:end].lstrip().startswith("subroutine run_step(a, b, c)")
    assert STEP[start:end].endswith("end subroutine run_step")
    first, last, text, arguments = find_call_span(STEP, "compute_flux", within_routine="run_step")
    assert STEP[first:last] == text
    assert arguments == ["a", "b", "c"]
    assert len(find_all_calls(STEP, "finish")) == 1
    with pytest.raises(ValueError, match="one complete original"):
        find_routine_span(STEP, "absent")


def test_blocks_of_an_undefined_macro_are_dropped_before_parsing():
    source = """module exchange
contains
  subroutine halo(a)
    real :: a(:)
#ifdef DISTRIBUTED
    call exchange_with_neighbours(a
#else
    a(1) = a(size(a))
#endif
  end subroutine halo
end module exchange
"""
    tree = parse_fortran(source, undefined=("DISTRIBUTED",))
    assert "a(1) = a(SIZE(a))" in str(tree)
    assert "exchange_with_neighbours" not in str(tree)
    start, end = find_routine_span(source, "halo", undefined=("DISTRIBUTED",))
    assert source[start:end].lstrip().startswith("subroutine halo(a)")


TILED = """module m_tile
  implicit none
contains
  subroutine tiled_step(nx, ny, nz, bx, a, b)
    integer, intent(in) :: nx, ny, nz, bx
    real(8), intent(in) :: a(0:nx+1, 0:ny+1, 0:nz+1)
    real(8), intent(out) :: b(1:nx, 1:ny, 1:nz)
    real(8) :: scratch(1:bx+1, 1:bx, 1:bx)
    integer :: i, j, k, ib, jb, kb, tx, ty, tz, rx, ry, rz
    do kb = 1, nz, bx
      do jb = 1, ny, bx
        do ib = 1, nx, bx
          do k = kb, min(kb + bx - 1, nz)
            tz = k - kb + 1
            do j = jb, min(jb + bx - 1, ny)
              ty = j - jb + 1
              do i = ib, min(ib + bx, nx + 1)
                tx = i - ib + 1
                scratch(tx, ty, tz) = 0.25d0 * a(i, j, k)
              end do
            end do
          end do
          do k = kb, min(kb + bx - 1, nz)
            rz = k - kb + 1
            do j = jb, min(jb + bx - 1, ny)
              ry = j - jb + 1
              do i = ib, min(ib + bx - 1, nx)
                rx = i - ib + 1
                b(i, j, k) = scratch(rx + 1, ry, rz) - scratch(rx, ry, rz)
              end do
            end do
          end do
        end do
      end do
    end do
  end subroutine tiled_step
end module m_tile
"""


def test_a_tile_scratch_array_is_inlined_into_its_consumer():
    rewritten, changes = inline_tile_scratch_nests(TILED)
    assert len(changes) == 1
    body = rewritten.split("integer :: i, j, k, ib, jb, kb, tx, ty, tz, rx, ry, rz", 1)[1]
    # The producer's formula now stands where the scratch values were read.
    assert "scratch(" not in body
    assert "0.25d0" in body.lower()
    # A source without the producer and consumer pair is left alone.
    assert inline_tile_scratch_nests(STEP) == (STEP, [])


LINES = """module filters_mod
  implicit none
contains
  subroutine smooth(U, dir)
    real(8), dimension(-2:, -2:, -2:), intent(inout) :: U
    integer, intent(in) :: dir
    integer :: i, j, k, ii, jj, kk
    integer :: maxi, maxj, maxk
    real(8) :: S
    real(8) :: tmp(-2:ubound(U,dir))

    maxi = ubound(U, 1) - 3
    maxj = ubound(U, 2) - 3
    maxk = ubound(U, 3) - 3

    if (dir == 1) then
      do k = 1, maxk
        do j = 1, maxj
          tmp = U(:, j, k)
          do i = 1, maxi
            S = 0._8
            do ii = i - 1, i + 1
              S = S + tmp(ii)
            end do
            U(i, j, k) = S / 3
          enddo
        enddo
      enddo
    else if (dir == 2) then
      do k = 1, maxk
        do i = 1, maxi
          tmp = U(i, :, k)
          do j = 1, maxj
            S = 0._8
            do jj = j - 1, j + 1
              S = S + tmp(jj)
            end do
            U(i, j, k) = S / 3
          enddo
        enddo
      enddo
    else
      do j = 1, maxj
        do i = 1, maxi
          tmp = U(i, j, :)
          do k = 1, maxk
            S = 0._8
            do kk = k - 1, k + 1
              S = S + tmp(kk)
            end do
            U(i, j, k) = S / 3
          enddo
        enddo
      enddo
    end if
  end subroutine smooth
end module filters_mod
"""


def test_a_line_snapshot_becomes_a_whole_snapshot_that_a_directive_releases():
    rewritten, edits = promote_line_snapshots(LINES)
    assert [(edit["routine"], edit["snapshot"], edit["source_array"]) for edit in edits] == [("smooth", "tmp", "U")]
    compact = rewritten.replace(" ", "").lower()
    assert "allocatable::tmp(:,:,:)" in compact
    assert "allocate(tmp(" in compact
    # Each read of the copied line now reads the same element of the snapshot.
    assert "tmp((i+ii),j,k)" in compact or "tmp(ii,j,k)" in compact
    assert "tmp=u(:,j,k)" not in compact
    # The snapshot's device copy is retired before the procedure returns and deallocates it.
    assert rewritten.index("    !$fort release(tmp)\n") < rewritten.lower().rindex("end subroutine smooth")


CONFIGURED = """module numerics
contains
  subroutine selected(a)
    real(KIND_OF_REAL) :: a(:)
    integer :: i
    !$omp parallel do
    do i = 1, size(a)
      a(i) = a(i) + SHIFT
    end do
    !$omp end parallel do
  end subroutine selected
  subroutine other(a)
    real(KIND_OF_REAL) :: a(:)
    integer :: i
    !$omp parallel do
    do i = 1, size(a)
#include "body.inc"
    end do
    !$omp end parallel do
  end subroutine other
end module numerics
"""


def test_sources_are_preprocessed_and_only_selected_procedures_lose_their_directives(tmp_path):
    fortran = shutil.which("gfortran")
    if fortran is None:
        pytest.skip("GNU Fortran is required for preprocessing")
    source = tmp_path / "numerics.F90"
    source.write_text(CONFIGURED)
    (tmp_path / "body.inc").write_text("      a(i) = 2 * a(i)\n")
    record = configure_sources([source], ["numerics::selected"], tmp_path / "configured", fortran=fortran,
                               defines=["KIND_OF_REAL=8", "SHIFT=1.5d0"])
    item, = record["sources"]
    assert item["original"]["path"] == str(source)
    prepared = Path(item["prepared"]["path"])
    assert prepared.parent == tmp_path / "configured"
    text = prepared.read_text()
    # Macros and includes are expanded; the original is untouched.
    assert "real(8) :: a(:)" in text
    assert "a(i) = a(i) + 1.5d0" in text
    assert "a(i) = 2 * a(i)" in text
    assert source.read_text() == CONFIGURED
    # Line numbers of the selected procedure are kept: its directives become empty lines.
    selected, other = text.split("subroutine other", 1)
    assert "!$omp" not in selected
    assert other.count("!$omp") == 2
    assert [row["procedure"] for row in item["serialized_directives"]] == ["numerics::selected"] * 2
    assert record["procedures"] == ["numerics::selected"]
    with pytest.raises(FileExistsError):
        configure_sources([source], [], tmp_path / "configured", fortran=fortran)


HALO = """module halo
integer :: nx
contains
subroutine wrap(a)
real(8),intent(inout)::a(-1:)
integer::i
do i=1,2
a(nx+i)=a(i)
enddo
end subroutine
end module
"""


def test_a_guard_states_what_the_loop_bounds_do_not_show(tmp_path):
    source = tmp_path / "halo.f90"
    source.write_text(HALO)
    exported = export_resident_entries([source], ["halo::wrap"], tmp_path / "exported")
    entry, = exported["entries"]
    # Whether the copied cells overlap depends on nx, which the bounds do not constrain.
    with pytest.raises(CompilationError, match="Cannot parallelize"):
        compile_resident_entry(entry["input_path"], entry["entry"], tmp_path / "unguarded")
    guarded = guard_entries(exported, tmp_path / "guarded", "nx >= 2")
    item, = guarded["entries"]
    assert item["runtime_guards"] == [*entry["runtime_guards"], "nx >= 2"]
    assert item["guard"]["original"]["path"] == entry["input_path"]
    lines = Path(item["input_path"]).read_text().splitlines()
    opening = next(index for index, line in enumerate(lines) if line.strip() == "if (nx >= 2) then")
    assert lines[opening - 1].strip().startswith("DO i")
    assert lines[opening + 2].strip() == "endif"
    assert lines[opening + 3].strip() == "END DO"
    assert Path(entry["input_path"]).read_text().count("nx >= 2") == 0
    # The guard composes with the value ABI, which device placement needs for the lower bound.
    item, = pass_scalars_by_value(guarded, tmp_path / "values", resources=("halo::nx",))["entries"]
    report = compile_resident_entry(item["input_path"], item["entry"], tmp_path / "entry")
    assert all(region["gpu_available"] for region in report["scoped"]["region_execution"])
