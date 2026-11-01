"""Tile scratch source coverage and reached storage arithmetic, not scalarization."""

from copy import copy
from dataclasses import replace

import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.tile_private import prove_tile_private
from compiler.ir import CompilationError, ScalarType, Symbol
from compiler.ir.tile_private import ScratchDescriptor, ScratchResourceError, plan_scratch

SOURCE = """module stencil
contains
subroutine advance(a,b,n,t)
integer,intent(in)::n,t
real(8),intent(in)::a(-3:)
real(8),intent(inout)::b(-3:)
real(8)::tmp(-1:t+2)
integer::bi,i,li
!$omp parallel private(bi,i,li,tmp)
!$omp do schedule(runtime)
do bi=1,n,t
 do i=bi-1,min(bi+t-1,n)+2
  li=i-bi+1
  tmp(li)=3.0_8*a(i)
 enddo
 do i=bi,min(bi+t-1,n)
  li=i-bi+1
  b(i)=b(i)+(tmp(li+1)-tmp(li-1))
 enddo
enddo
!$omp end do
!$omp end parallel
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE):
    p = tmp_path / "source.f90"
    p.write_text(source)
    a = SourceEffects([p])
    r = a.routines["stencil::advance"]
    node = next(x for x in walk(r.execution) if type(x).__name__ == "Block_Nonlabel_Do_Construct")
    return p, a, r, node


def test_original_dynamic_neighbor_phases_are_issued_and_not_recomputed(tmp_path):
    p, a, r, n = fixture(tmp_path)
    before = p.read_bytes()
    proof = prove_tile_private(a, r, n)
    assert proof.validate() is proof
    assert prove_tile_private(a, r, n) is proof
    public = proof.public()
    assert public["axes"][0]["producer_lower_offset"] == -1
    assert public["axes"][0]["producer_upper_guarantee"] == 2
    assert public["axes"][0]["scratch_index_lower"] == 0
    assert public["axes"][0]["scratch_index_upper_width_offset"] == 2
    assert public["scratch_axis_order"] == {"tmp": ["bi"]}
    assert public["recomputation_authorized"] is False
    assert public["definitions_authorized"] is False
    assert public["placement_authorized"] is False
    assert public["pricing"] == "unpriced; forced sections only"
    assert p.read_bytes() == before
    for forged in (copy(proof), replace(proof)):
        with pytest.raises(CompilationError, match="exact compiler-issued"):
            forged.validate()
    public["axes"][0]["producer_lower_offset"] = -99
    assert proof.public()["axes"][0]["producer_lower_offset"] == -1


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace("tmp(li+1)", "tmp(li+3)"),
        SOURCE.replace("do i=bi-1,", "do i=bi,"),
        SOURCE.replace("tmp(li)=3.0_8*a(i)", "if(i>1)tmp(li)=3.0_8*a(i)"),
        SOURCE.replace("tmp(li)=3.0_8*a(i)", "tmp(li)=tmp(li-1)+a(i)"),
        SOURCE.replace("li=i-bi+1", "li=i+1"),
        SOURCE.replace("private(bi,i,li,tmp)", "private(bi,i,li)"),
        SOURCE.replace("real(8)::tmp", "real(8),save::tmp"),
        SOURCE.replace("tmp(li)=3.0_8*a(i)", "call opaque(tmp)"),
        SOURCE.replace("b(i)=b(i)+", "b(i+1)=b(i)+"),
        SOURCE.replace("3.0_8*a(i)", "3.0_8*b(i)"),
        SOURCE.replace("!$omp end do", "!$omp end do\nwrite(*,*) tmp(1)"),
    ],
)
def test_incomplete_private_coverage_and_external_dependence_fail_before_edits(tmp_path, source):
    p, a, r, n = fixture(tmp_path, source)
    before = p.read_bytes()
    with pytest.raises(CompilationError):
        prove_tile_private(a, r, n)
    assert p.read_bytes() == before


def test_changed_source_invalidates_issued_tile_proof(tmp_path):
    p, a, r, n = fixture(tmp_path)
    proof = prove_tile_private(a, r, n)
    p.write_text(SOURCE + "! changed\n")
    with pytest.raises(CompilationError):
        proof.validate()


def test_zero_tiles_never_query_private_storage():
    def forbidden():
        raise AssertionError("unreached descriptor query")

    p = plan_scratch(0, forbidden, 0)
    assert p.lease_bytes == 0
    assert list(p.batches()) == []


def test_real4_real8_original_bounds_padding_and_odd_batches():
    x = Symbol(1, "x", ScalarType.REAL32, rank=1)
    y = Symbol(2, "y", ScalarType.REAL, rank=3)
    p = plan_scratch(5, lambda: (ScratchDescriptor(x, (-1,), (3,)), ScratchDescriptor(y, (-1, 0, 1), (1, 1, 2))), 400)
    assert p.arrays[0].payload_bytes == 20
    assert p.arrays[1].offset_bytes == 24
    assert p.arrays[1].payload_bytes == 96
    assert p.bytes_per_tile == 120
    assert p.lease_bytes == 360
    assert list(p.batches()) == [(0, 3), (3, 2)]
    assert p.byte_offset(2, 1, (0, 1, 2)) == 240 + 24 + (1 + 3 + 6) * 8
    with pytest.raises(IndexError):
        p.byte_offset(0, 1, (-2, 0, 1))


@pytest.mark.parametrize(
    ("bounds", "budget"),
    [
        (((0,), (0,)), 7),
        (((0,), (-1,)), 1024),
        (((0, 0, 0), ((1 << 31) - 1,) * 3), (1 << 64) - 1),
        (((True,), (1,)), 1024),
        (((-(1 << 31) - 1,), (0,)), 1024),
    ],
)
def test_empty_overflow_and_insufficient_storage_are_prework_resource_failures(bounds, budget):
    x = Symbol(1, "x", ScalarType.REAL, rank=len(bounds[0]))
    with pytest.raises(ScratchResourceError):
        plan_scratch(1, lambda: (ScratchDescriptor(x, *bounds),), budget)


SOURCE_3D = """module stencil
contains
subroutine advance(a,b,nx,ny,nz,tx,ty,tz)
integer,intent(in)::nx,ny,nz,tx,ty,tz
real(8),intent(in)::a(:,:,:)
real(8),intent(inout)::b(:,:,:)
real(8)::tmp(-1:tx+2,-1:ty+2,-1:tz+2)
integer::bi,bj,bk,i,j,k,li,lj,lk
!$omp parallel private(bi,bj,bk,i,j,k,li,lj,lk,tmp)
!$omp do collapse(3) schedule(runtime)
do bk=1,nz,tz
 do bj=1,ny,ty
  do bi=1,nx,tx
   do k=bk,min(bk+tz-1,nz)
    do j=bj,min(bj+ty-1,ny)
     do i=bi-1,min(bi+tx-1,nx)+2
      li=i-bi+1
      lj=j-bj+1
      lk=k-bk+1
      tmp(li,lj,lk)=3.0_8*a(i,j,k)
     enddo
    enddo
   enddo
   do k=bk,min(bk+tz-1,nz)
    do j=bj,min(bj+ty-1,ny)
     do i=bi,min(bi+tx-1,nx)
      li=i-bi+1
      lj=j-bj+1
      lk=k-bk+1
      b(i,j,k)=b(i,j,k)+(tmp(li+1,lj,lk)-tmp(li-1,lj,lk))
     enddo
    enddo
   enddo
  enddo
 enddo
enddo
!$omp end do
!$omp end parallel
end subroutine
end module
"""


def test_three_dimensional_source_keeps_original_loop_and_expression_order(tmp_path):
    _, analysis, routine, node = fixture(tmp_path, SOURCE_3D)
    proof = prove_tile_private(analysis, routine, node)
    record = proof.public()
    assert [x["tile"] for x in record["axes"]] == ["bk", "bj", "bi"]
    assert record["scratch"][0]["rank"] == 3
    assert record["scratch_axis_order"] == {"tmp": ["bi", "bj", "bk"]}
    assert [(axis["scratch_index_lower"], axis["scratch_index_upper_width_offset"]) for axis in record["axes"]] == [(1, 0), (1, 0), (0, 2)]
    assert record["original_tile"] == str(node)
    assert record["producer"] == str(proof.phases[0])
    assert record["consumer"] == str(proof.phases[1])


@pytest.mark.parametrize(
    "source",
    [
        SOURCE_3D.replace("b(i,j,k)=b(i,j,k)", "b(j,i,k)=b(i,j,k)"),
        SOURCE.replace("b(i)=b(i)+", "if(tmp(li+3)>0) b(i)=b(i)+"),
        SOURCE.replace("b(i)=b(i)+", "if(a(i)>0)li=li+1\n  b(i)=b(i)+"),
    ],
)
def test_transposed_read_write_and_conditional_indices_cannot_hide_dependences(tmp_path, source):
    _, analysis, routine, node = fixture(tmp_path, source)
    with pytest.raises(CompilationError):
        prove_tile_private(analysis, routine, node)


@pytest.mark.parametrize(("upper", "lower", "upper_offset", "coverage"), [
    ("min(bi+t-1,n)+2", 0, 2, 2),
    ("min(bi+t,n)+1", 0, 2, 1),
    ("min(bi+t-2,n)+3", 0, 2, 2),
])
def test_descriptor_upper_is_a_bound_on_all_original_production_not_only_required_reads(
    tmp_path, upper, lower, upper_offset, coverage
):
    source = SOURCE.replace("min(bi+t-1,n)+2", upper)
    _, analysis, routine, node = fixture(tmp_path, source)
    # The second spelling conservatively guarantees only +1 near clipping, so
    # use an original +1 consumer rather than granting its absent +2 coverage.
    proof = prove_tile_private(analysis, routine, node)
    axis = proof.public()["axes"][0]
    assert axis["scratch_index_lower"] == lower
    assert axis["scratch_index_upper_width_offset"] == upper_offset
    assert axis["producer_upper_guarantee"] == coverage


@pytest.mark.parametrize("observer", ["if(i>0)call observe(i)", "i=0", "write(*,*)li"])
def test_native_private_scalar_uses_outside_complete_tile_units_fail_closed(tmp_path, observer):
    source = SOURCE.replace("!$omp end parallel", observer + "\n!$omp end parallel")
    _, analysis, routine, node = fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="private scalar has an observer"):
        prove_tile_private(analysis, routine, node)


@pytest.mark.parametrize("source", [
    SOURCE_3D.replace("do bi=1,nx,tx", "do bi=bj,nx,tx"),
    SOURCE_3D.replace("do bi=1,nx,tx", "do bi=1,nx-bj,tx").replace("min(bi+tx-1,nx)", "min(bi+tx-1,nx-bj)"),
    SOURCE.replace("do bi=1,n,t", "do bi=1,n,bi").replace("bi+t-1", "bi+bi-1"),
])
def test_nonrectangular_outer_tile_domains_cannot_use_independent_axis_snapshots(tmp_path, source):
    _, analysis, routine, node = fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="rectangular outer tile domains"):
        prove_tile_private(analysis, routine, node)
