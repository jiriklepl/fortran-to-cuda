"""Retained checked children, descriptor exchanges and cold absent domains."""
import json
import os
import re
import subprocess

import pytest

from compiler.tests.test_native_ancestor import SOURCE
from compiler.tests.test_native_ancestor_runtime import build_native_ancestor

EXCHANGE = """subroutine exchange_alloc(left,right)
real(8),allocatable,intent(inout)::left(:),right(:)
real(8),allocatable::temporary(:)
call move_alloc(left,temporary)
call move_alloc(right,left)
call move_alloc(temporary,right)
end subroutine
"""

SOURCE_WITH_SWAP = SOURCE.replace("real(8),intent(inout)::a(:),b(:)",
    "real(8),allocatable,intent(inout)::a(:),b(:)", 1).replace("call middle(a,b,n,flag)", """if(allocated(a))then
 if(allocated(b))then
  call middle(a,b,n,flag)
  call exchange_alloc(a,b)
  call consumer(a,b,n)
 endif
endif""", 1).replace("end module", EXCHANGE + "end module")

DRIVER = """program exercise
use ancestor_fixture
implicit none
integer,parameter::extent=27
real(8),allocatable::a(:),b(:),ea(:),eb(:),temporary(:)
real(8)::expected_probe
integer::i,stage,n,flag_integer
logical::flag
character(16)::argument
call get_command_argument(1,argument)
read(argument,*)n
call get_command_argument(2,argument)
read(argument,*)flag_integer
flag=flag_integer/=0
if(n<0)then
 call advance(a,b,0,flag)
 call advance(a,b,0,flag)
 if(allocated(a).or.allocated(b))error stop 'unallocated control'
 print *, 'CONDITIONAL_SWAP_UNALLOCATED_OK'
 stop
endif
allocate(a(-3:23),b(7:33),ea(-3:23),eb(7:33))
do i=1,extent
 a(i-4)=real(17*i-3,8)/32.0_8
 b(i+6)=real(500-i,8)/16.0_8
enddo
ea=a
eb=b
do stage=1,2
 expected_probe=0
 do i=1,n
  eb(lbound(eb,1)+i-1)=2.0_8*ea(lbound(ea,1)+i-1)
  expected_probe=expected_probe+eb(lbound(eb,1)+i-1)
  if(flag)eb(lbound(eb,1)+i-1)=eb(lbound(eb,1)+i-1)+3.0_8
  ea(lbound(ea,1)+i-1)=eb(lbound(eb,1)+i-1)+1.0_8
 enddo
 call move_alloc(ea,temporary)
 call move_alloc(eb,ea)
 call move_alloc(temporary,eb)
 do i=1,n
  ea(lbound(ea,1)+i-1)=eb(lbound(eb,1)+i-1)+1.0_8
 enddo
 call advance(a,b,n,flag)
 if(lbound(a,1)/=lbound(ea,1).or.ubound(a,1)/=ubound(ea,1))error stop 'first allocation identity'
 if(lbound(b,1)/=lbound(eb,1).or.ubound(b,1)/=ubound(eb,1))error stop 'second allocation identity'
 if(any(a/=ea).or.any(b/=eb))error stop 'complete original arrays and halos'
 if(observed/=expected_probe)error stop 'original readonly probe'
 if(opaque_visits/=stage*flag_integer)error stop 'original synchronous call exactly once'
enddo
print *, 'CONDITIONAL_SWAP_FULL_FIELDS_OK',n,flag_integer
end program
"""


@pytest.fixture(scope="module")
def conditional_swap_binaries(tmp_path_factory):
    return build_native_ancestor(tmp_path_factory.mktemp("conditional_swap"), source=SOURCE_WITH_SWAP,
                                 driver_source=DRIVER, selected_window=True)


@pytest.mark.cuda
@pytest.mark.parametrize(("mode", "n", "cut", "hidden"), [
    ("enabled-live-swap", 19, False, False),
    ("enabled-retire-swap", 19, True, False),
    ("hidden-live-swap", 19, False, True),
    ("cold-zero", 0, False, False),
    ("cold-unallocated", -1, False, False),
])
def test_conditional_return_carries_physical_allocations_and_respects_cold_absence(
        conditional_swap_binaries, mode, n, cut, hidden):
    build, candidate, native = conditional_swap_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    results = []
    for name, binary in (("native", native), ("candidate", candidate)):
        argv = [str(binary), str(n), str(int(cut))]
        (build / f"{mode}-{name}.command.json").write_text(json.dumps({"argv": argv,
            "OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES")}) + "\n")
        result = subprocess.run(argv, cwd=build, env=environment, capture_output=True, text=True, timeout=30)
        for suffix in ("stdout", "stderr"):
            (build / f"{mode}-{name}.{suffix}").write_text(getattr(result, suffix))
        (build / f"{mode}-{name}.completion.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr
        results.append(result)
    reference, result = results
    assert result.stdout == reference.stdout
    assert "CONDITIONAL_SWAP_" in result.stdout
    lines = result.stderr.splitlines()
    ledger = {kind: {"count": sum(line.startswith("FORT_SCOPED " + kind + " ") for line in lines),
        "bytes": sum(int(re.search(r"\bbytes=(\d+)", line)[1]) for line in lines
                     if line.startswith("FORT_SCOPED " + kind + " "))} for kind in ("upload", "download")}
    ledger.update(launches=sum(line.startswith("FORT_SCOPED launch ") for line in lines),
                  swaps=sum(line.startswith("FORT_SCOPED identity_swap ") for line in lines),
                  domains=sum('"event":"domain_create"' in line for line in lines),
                  retirements=sum('"event":"domain_retire"' in line for line in lines))
    (build / f"{mode}-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    if hidden or n <= 0:
        assert ledger["launches"] == 0, result.stderr
        assert ledger["upload"]["bytes"] == 0, result.stderr
        assert ledger["download"]["bytes"] == 0, result.stderr
        if n <= 0:
            assert not any(line.startswith("FORT_SCOPED initialize") for line in lines), result.stderr
    else:
        if cut:
            # Reopening inside an assumed-shape child loses the caller's
            # allocation lower-bound origin. Its reached swap preflight must
            # publish before mutation; NONE does not recredit definitions.
            assert ledger["swaps"] == 0, result.stderr
            assert ledger["launches"] == 6, result.stderr
            assert ledger["domains"] == 6, result.stderr
            assert ledger["upload"]["bytes"] == 912, result.stderr
            assert ledger["download"]["bytes"] == 912, result.stderr
        else:
            assert ledger["launches"] == 6, result.stderr
            assert ledger["swaps"] == 2, result.stderr
            assert ledger["domains"] == 2, result.stderr
            assert ledger["upload"]["bytes"] == 304, result.stderr
            # The post-swap consumer rewrites the allocation already mirrored
            # by the readonly probe, so its final publication is required too.
            assert ledger["download"]["bytes"] == 912, result.stderr
