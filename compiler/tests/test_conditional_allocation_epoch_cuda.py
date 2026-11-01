"""Branch-local lifetime cuts preserve the path that does not replace storage."""
import os
import re

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_graph_storage_epochs import SOURCE, emit

CONDITIONAL_SOURCE = SOURCE.replace(
    "subroutine advance(left,right,n,steps)",
    "subroutine advance(left,right,n,steps,change)",
).replace(
    "integer,intent(in)::n,steps",
    "integer,intent(in)::n,steps\nlogical,intent(in)::change",
).replace(
    "call producer(right,n)\ncall consumer(left,right,n)",
    "call producer(right,n)\nif(change)then\ncall relocate(left,right)\nendif\ncall consumer(left,right,n)",
)

DRIVER = """program exercise
use epochs
use iso_fortran_env,only:error_unit
implicit none
real(8),allocatable::left(:),right(:)
integer::case_index,n,i,mode
logical::change
do mode=0,1
change=mode==1
write(error_unit,'(A)')'BRANCH_EPOCH_ZERO_BEGIN'
call advance(left,right,257,0,change)
write(error_unit,'(A)')'BRANCH_EPOCH_ZERO_END'
if(allocated(left).or.allocated(right))error stop 'zero trip inspected storage'
do case_index=1,3
select case(case_index)
case(1)
n=1
case(2)
n=7
case default
n=257
end select
write(error_unit,'(A,1X,I0,1X,I0)')'BRANCH_EPOCH_CASE_BEGIN',mode,n
call advance(left,right,n,2,change)
write(error_unit,'(A,1X,I0,1X,I0)')'BRANCH_EPOCH_CASE_END',mode,n
if(any(shape(left)/=[n]).or.any(shape(right)/=[n]))error stop 'shape changed'
! Two stages restore the original physical bounds in either branch.
if(lbound(left,1)/=-3.or.lbound(right,1)/=2)error stop 'allocation bounds lost'
do i=1,n
if(change)then
if(left(i-4)/=0.0_8.or.right(i+1)/=0.0_8)error stop 'replaced storage value'
else
if(left(i-4)/=real(i,8))error stop 'producer value'
if(right(i+1)/=2.0_8*real(i,8))error stop 'consumer value'
endif
enddo
deallocate(left,right)
enddo
enddo
print *, 'CONDITIONAL_ALLOCATION_EPOCH_FIELDS_OK'
end program
"""


@pytest.fixture(scope="module")
def conditional_epoch_binaries(tmp_path_factory):
    def conditional_emit(path):
        original, outputs, manifest = emit(path, CONDITIONAL_SOURCE)
        assert manifest["scope_count"] == 1, manifest.get("graph_owner_attempt")
        return original, outputs, manifest

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "emit", conditional_emit)
        patch.setattr(base, "DRIVER", DRIVER)
        return base.epoch_binaries.__wrapped__(tmp_path_factory)


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True], ids=["enabled", "hidden"])
def test_conditional_allocation_cut_only_executes_on_original_branch(conditional_epoch_binaries, hidden):
    build, candidate, native = conditional_epoch_binaries
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        env["CUDA_VISIBLE_DEVICES"] = ""
    mode = "hidden" if hidden else "enabled"
    reference = base.checked([str(native)], build, mode + "-branch-native", env)
    result = base.checked([str(candidate)], build, mode + "-branch-candidate", env)
    assert result.stdout == reference.stdout
    assert "CONDITIONAL_ALLOCATION_EPOCH_FIELDS_OK" in result.stdout
    zeros = re.findall(r"BRANCH_EPOCH_ZERO_BEGIN\n(.*?)BRANCH_EPOCH_ZERO_END", result.stderr, re.S)
    assert len(zeros) == 2
    for events in zeros:
        assert "FORT_SCOPED domain_create " not in events
        assert "FORT_SCOPED register " not in events
        assert "FORT_SCOPED launch " not in events
    cases = re.findall(r"BRANCH_EPOCH_CASE_BEGIN (\d+) (\d+)\n(.*?)BRANCH_EPOCH_CASE_END \1 \2", result.stderr, re.S)
    assert len(cases) == 6
    for changed, _size, events in cases:
        lines = events.splitlines()
        launches = [index for index, line in enumerate(lines) if line.startswith("FORT_SCOPED launch ")]
        if hidden:
            assert not launches
            assert not any(line.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download ")) for line in lines)
            continue
        assert len(launches) == 4, events
        for iteration in range(2):
            between = lines[launches[2 * iteration] + 1:launches[2 * iteration + 1]]
            copies = [line for line in between if line.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download "))]
            if changed == "0":
                assert not copies, events
            else:
                assert any(line.startswith("FORT_SCOPED download ") for line in copies), events
                assert any(line.startswith("FORT_SCOPED upload ") for line in copies), events
