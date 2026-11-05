"""Exact plane reads preserve untouched storage and reuse their GPU replica."""
import os
import re
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests import test_graph_storage_epochs_cuda as support

SOURCE = """module face_fields
implicit none
real(8)::field(-2:7,-2:7,-2:7)
contains
subroutine advance(n,plane)
integer,intent(in)::n,plane
integer::j,k
do k=1,n
do j=1,n
field(0,j,k)=field(plane,j,k)+1.d0
enddo
enddo
do k=1,n
do j=1,n
field(-1,j,k)=field(plane,j,k)+2.d0
enddo
enddo
end subroutine
end module
"""

DRIVER = """program verify
use face_fields
use iso_fortran_env,only:error_unit
implicit none
integer,parameter::sizes(4)=[0,1,5,5],planes(4)=[1,1,5,0]
integer::step,i,j,k,unit
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do step=1,4
do k=-2,7
do j=-2,7
do i=-2,7
field(i,j,k)=real(i+16*j+256*k+step,8)/8.d0
enddo
enddo
enddo
write(error_unit,'(a,i0)')'FACE_BEGIN ',step
call advance(sizes(step),planes(step))
write(error_unit,'(a,i0)')'FACE_END ',step
write(unit)lbound(field),ubound(field),field
enddo
close(unit)
end program
"""


def emit(directory):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    path = directory / "source.f90"
    path.write_text(SOURCE)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"face_fields::field": {"storage": "stable", "initialized": "none",
                 "escapes": False, "allocation_changes": False}}}
    builder = ScopeBuilder([path], "face_fields::advance", facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"))
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    validate_checked_execution_plans(report)
    return path, outputs, report


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(support, "emit", emit)
        patch.setattr(support, "DRIVER", DRIVER)
        return support.epoch_binaries.__wrapped__(tmp_path_factory)


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True], ids=["gpu", "hidden"])
def test_reached_faces_upload_only_the_incoming_plane_once(binaries, hidden):
    directory, candidate, native = binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    mode = "hidden" if hidden else "gpu"
    support.checked([str(native)], directory, mode + "-native", environment)
    expected = (directory / "fields.bin").read_bytes()
    result = support.checked([str(candidate)], directory, mode + "-candidate", environment)
    assert (directory / "fields.bin").read_bytes() == expected
    traces = re.findall(r"FACE_BEGIN (\d+)\n(.*?)FACE_END \1\n", result.stderr, re.S)
    assert [step for step, _trace in traces] == ["1", "2", "3", "4"]
    for (step, trace), size in zip(traces, (0, 1, 5, 5), strict=True):
        copies = {kind: [dict(re.findall(r"(\w+)=([^\s]+)", line)) for line in trace.splitlines()
                         if line.startswith("FORT_SCOPED " + kind + " ")]
                  for kind in ("upload", "download")}
        for rows in copies.values():
            assert all(row["provenance_version"] == row["domain_version"] == row["generation"] == "1"
                       and row["buffer"] == "1" and int(row["buffer_handle"]) > 0 for row in rows)
        assert trace.count("FORT_SCOPED launch ") == (2 if size and not hidden else 0), (step, trace)
        assert len(copies["upload"]) == (1 if size and not hidden else 0), trace
        assert sum(int(row["bytes"]) for row in copies["upload"]) == (size * size * 8 if not hidden else 0), trace
        assert sum(int(row["bytes"]) for row in copies["download"]) == (size * size * 16 if not hidden else 0), trace
        if not hidden:
            assert '"reason":"uninitialized_read"' not in trace
