"""Live child query grouping on CUDA; synthetic prices are not timing evidence."""
import json
import os
import re
import shutil
import subprocess

import pytest

from compiler.offload.config import OffloadConfig
from compiler.tests.test_borrowed_query_regions import NESTED
from compiler.tests.test_native_ancestor_runtime import build_native_ancestor
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile

DRIVER = """program exercise
use ancestor_fixture
use iso_fortran_env,only:error_unit
implicit none
real(8),allocatable::a(:),b(:),ea(:),eb(:)
integer::i,n,shape,repeat,mode
logical::flag
character(1)::argument
call get_command_argument(1,argument)
read(argument,*)mode
flag=mode/=0
do shape=1,2
 n=257*(shape-1)
 allocate(a(-3:n+4),b(7:n+14),ea(-3:n+4),eb(7:n+14))
 do i=1,n+8
  a(i-4)=real(17*i-3,8)/32.0_8
  b(i+6)=real(500-i,8)/16.0_8
 enddo
 do repeat=1,2
  ea=a
  eb=b
  do i=1,n
   eb(i+6)=2.0_8*a(i-4)
   if(flag)ea(i-4)=ea(i-4)+eb(i+6)+1.0_8
  enddo
  write(error_unit,'(A,1X,I0,1X,I0)')'CHILD_QUERY_BEGIN',n,repeat
  call advance(a,b,n,flag)
  write(error_unit,'(A,1X,I0,1X,I0)')'CHILD_QUERY_END',n,repeat
  if(any(a/=ea).or.any(b/=eb))error stop 'child query fields or halos'
  if(opaque_visits/=0)error stop 'unexpected original call'
  ea=a
  eb=b
  if(flag)then
   do i=1,n
    eb(i+6)=2.0_8*a(i-4)
    ea(i-4)=ea(i-4)+eb(i+6)+1.0_8
   enddo
  endif
  write(error_unit,'(A)')'CHILD_ORDINARY_BEGIN'
  call middle(a,b,n,flag)
  write(error_unit,'(A)')'CHILD_ORDINARY_END'
  if(any(a/=ea).or.any(b/=eb))error stop 'ordinary child fields or halos'
 enddo
 deallocate(a,b,ea,eb)
enddo
print *, 'BORROWED_QUERY_FULL_FIELDS_OK'
end program
"""


@pytest.fixture(scope="module", params=["ordinary", "partial_metadata"])
def binaries(request, tmp_path_factory):
    from compiler.tests import test_native_ancestor_runtime as support

    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14")
    if not nvcc or not host:
        pytest.skip("CUDA toolchain required")
    directory = tmp_path_factory.mktemp("borrowed_query")
    profile = actual_profile(directory, nvcc, host)
    original = support.fixture
    partial = request.param == "partial_metadata"
    source = NESTED
    if partial:
        source = source.replace("integer::opaque_visits=0",
            "integer::opaque_visits=0\ninteger,protected::tiles(-2:5)=[1,1,1,1,1,1,1,1]")
        source = source.replace("integer::i\ndo i=1,n\nb(i)=",
            "integer::i,tn\ntn=tiles(1)\ndo i=1,n-tn+1\nb(i)=")

    def fixture(*args, **kwargs):
        path, builder = original(*args, **kwargs)
        builder.config = OffloadConfig("auto", profile)
        if partial:
            builder.facts["captures"]["ancestor_fixture::tiles"] = {
                "storage": "stable", "initialized": "none", "escapes": False, "allocation_changes": False}
        return path, builder

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(support, "fixture", fixture)
        return build_native_ancestor(directory, source=source, driver_source=DRIVER, selected_window=partial)


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("entered", [False, True])
def test_child_query_uses_live_replicas_and_only_its_reached_branch(binaries, hidden, entered):
    build, candidate, native = binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    results = []
    mode = f"hidden{int(hidden)}-entered{int(entered)}"
    for name, binary in (("native", native), ("candidate", candidate)):
        result = subprocess.run([str(binary), str(int(entered))], cwd=build, env=environment,
                                capture_output=True, text=True, timeout=30)
        for suffix in ("stdout", "stderr"):
            (build / f"{mode}-{name}.{suffix}").write_text(getattr(result, suffix))
        assert result.returncode == 0, result.stdout + result.stderr
        results.append(result)
    assert results[0].stdout == results[1].stdout
    assert "BORROWED_QUERY_FULL_FIELDS_OK" in results[1].stdout
    ordinary = re.findall(r"CHILD_ORDINARY_BEGIN\n(.*?)CHILD_ORDINARY_END", results[1].stderr, re.S)
    assert len(ordinary) == 4
    assert all("FORT_SCOPED " not in events for events in ordinary)
    cases = re.findall(r"CHILD_QUERY_BEGIN (\d+) (\d+)\n(.*?)CHILD_QUERY_END \1 \2", results[1].stderr, re.S)
    assert len(cases) == 4
    for size, _repeat, events in cases:
        lines = events.splitlines()
        launches = [i for i, line in enumerate(lines)
                    if line.startswith(("FORT_SCOPED launch ", "FORT_SCOPED graph_kernel "))]
        copies = [line for line in lines if line.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download "))]
        decisions = [json.loads(line.split("FORT_SCOPED evidence ", 1)[1])
                     for line in lines if line.startswith("FORT_SCOPED evidence ")]
        decisions = [row for row in decisions if row["event"] == "decision"]
        if hidden or not int(size):
            assert not launches, events
            assert not copies, events
            continue
        assert len(launches) == (3 if entered else 1), events
        assert any(row["gpu_units"] == 2 for row in decisions) is entered, events
        assert not any(line.startswith("FORT_SCOPED download ") for line in lines[:launches[-1]])
        assert sum(int(re.search(r"\bbytes=(\d+)\b", line)[1]) for line in copies
                   if line.startswith("FORT_SCOPED upload ")) == 8*int(size), events
