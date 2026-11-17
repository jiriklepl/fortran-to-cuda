"""Source-backed batching reuses numerical kernels and complete data-flow proofs."""


import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig
from compiler.offload.profile import SCOPED_BATCH_PAYLOADS, SCOPED_TRANSFER_COST_NAMES
from compiler.tests.test_scoped_planning_entries import calibration

FLAT = """module numerical
contains
subroutine advance(a,b,c,n,m)
real(8),intent(in)::a(:,:)
real(8),intent(inout)::b(:,:),c(:,:)
integer,intent(in)::n,m
integer::i,j
do j=2,m-1
 do i=2,n-1
  b(i,j)=a(i-1,j)+a(i,j)+a(i+1,j)+real(i,8)
 enddo
enddo
do j=2,m-1
 do i=2,n-1
  c(i,j)=2*b(i,j)+a(i,j)
 enddo
enddo
end subroutine
end module
"""

CHAIN = """module original
contains
subroutine first(a,b,n,m)
real(8),intent(in)::a(:,:)
real(8),intent(out)::b(:,:)
integer,intent(in)::n,m
integer::i,j
do j=2,m-1
 do i=2,n-1
  b(i,j)=a(i-1,j)+a(i,j)+a(i+1,j)+real(i,8)
 enddo
enddo
end subroutine
subroutine second(b,c,n,m)
real(8),intent(in)::b(:,:)
real(8),intent(out)::c(:,:)
integer,intent(in)::n,m
integer::i,j
do j=2,m-1
 do i=2,n-1
  c(i,j)=2*b(i,j)+real(j,8)
 enddo
enddo
end subroutine
subroutine step(a,b,c,n,m)
real(8),intent(in)::a(:,:)
real(8),intent(inout)::b(:,:),c(:,:)
integer,intent(in)::n,m
call first(a,b,n,m)
call second(b,c,n,m)
end subroutine
end module
"""


def transfer_profile():
    value = calibration()
    value["hardware"]["async_engine_count"] = 2
    costs = {name: 1e-7 for name in SCOPED_TRANSFER_COST_NAMES}
    costs.update(staging_cold_seconds=[1e-6]*4, staging_reuse_seconds=[1e-7]*4,
                 pack_bytes_per_second=1e11, unpack_bytes_per_second=1e11,
                 pack_row_seconds=0.0, unpack_row_seconds=0.0)
    value["scoped"]["transfers"] = {"schema_version": 1, "batch_payload_bytes": list(SCOPED_BATCH_PAYLOADS),
                                      "max_slot_bytes": SCOPED_BATCH_PAYLOADS[-1], "costs": costs}
    return value


def numerical(tmp_path, source=FLAT, mode="pipelined", profile=None):
    path = tmp_path / "numerical.f90"
    path.write_text(source)
    options = CompilerOptions(opt_level=0, gpu_policy="sections", memory_model="scoped", scope_transfers=mode)
    function, plan = prepare_function(lower_file(path, "advance"), options=options)
    config = OffloadConfig("sections", transfer_profile() if profile is None else profile, 4, scope_transfers=mode)
    return generate_scoped(function, plan, config, "common_functions.cuh", runtime_id=read_scoped_runtime()[1]["runtime_id"])


@pytest.mark.parametrize("mode", ["direct", "pinned"])
def test_synchronous_controls_do_not_emit_batch_workers(tmp_path, mode):
    emitted = numerical(tmp_path, mode=mode)
    assert not emitted.report["batch_execution"]["eligible"]
    assert "fort_scope_batch_execute_v1(" not in emitted.cuda
    if mode == "pinned":
        assert "fort_scope_set_transfer_costs_v1(" in emitted.cuda
        assert emitted.report["automatic_estimate_available"]
