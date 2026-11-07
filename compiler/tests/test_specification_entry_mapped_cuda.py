"""Read-only bounds preserve device replicas without downloading other fields."""
import os
import re
import subprocess

import pytest

from compiler.tests.test_native_ancestor_runtime import build_native_ancestor
from compiler.tests.test_specification_entry_mapped import MAPPED_SOURCE

DRIVER = """program exercise
use ancestor_fixture
use iso_fortran_env, only:error_unit
implicit none
real(8),allocatable::a(:),b(:),ea(:),eb(:)
integer::i,n,shape,repeat,flag_integer
logical::flag
character(1)::argument
call get_command_argument(1,argument)
read(argument,*)flag_integer
flag=flag_integer/=0
do shape=1,2
 n=19*(shape-1)
 allocate(a(-3:n+4),b(7:n+14),ea(-3:n+4),eb(7:n+14))
 do i=1,n+8
  a(i-4)=real(17*i-3,8)/32.0_8
  b(i+6)=real(500-i,8)/16.0_8
 enddo
 do repeat=1,2
  ea=a
  eb=b
  do i=1,n
   eb(i+6)=2.0_8*a(i-4)+real(n,8)
   if(flag)eb(i+6)=eb(i+6)+3.0_8
   ea(i-4)=eb(i+6)+real(n,8)
  enddo
  tile(1)=n
  write(error_unit,'(A,1X,I0,1X,I0)')'MAPPED_ENTRY_BEGIN',n,repeat
  call advance(a,b,n,flag)
  write(error_unit,'(A,1X,I0,1X,I0)')'MAPPED_ENTRY_END',n,repeat
  if(any(a/=ea).or.any(b/=eb))error stop 'mapped entry fields or halos'
  if(tile(1)/=n.or.observed/=0)error stop 'original bound metadata'
  if(opaque_visits/=flag_integer*(2*(shape-1)+repeat))error stop 'original source replay'
 enddo
 deallocate(a,b,ea,eb)
enddo
print *, 'MAPPED_ENTRY_FULL_FIELDS_OK'
end program
"""


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    from compiler.tests import test_native_ancestor_runtime as support

    # Original integer-array writes currently retire a checked source domain.
    # Both numerical workers instead read this unchanged metadata, so this
    # fixture tests live read-only replicas and a dirty unrelated field.
    source = MAPPED_SOURCE.replace("\ndo i=1,1\ntile(i)=n\nenddo", "").replace(
        "b(i)=2.0_8*a(i)", "b(i)=2.0_8*a(i)+real(tile(1),8)")
    original_fixture = support.fixture

    def fixture(*args, **kwargs):
        path, builder = original_fixture(*args, **kwargs)
        # The driver initializes this fixed module array before each call;
        # neither the original source nor the driver changes its allocation.
        builder.facts["captures"]["ancestor_fixture::tile"] = {
            "storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
        return path, builder

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(support, "fixture", fixture)
        return build_native_ancestor(tmp_path_factory.mktemp("mapped_specification"),
                                     source=source, driver_source=DRIVER)


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("cut", [False, True])
def test_readonly_bounds_retain_unrelated_dirty_payload(binaries, hidden, cut):
    build, candidate, native = binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    results = []
    mode = f"hidden{int(hidden)}-cut{int(cut)}"
    for name, binary in (("native", native), ("candidate", candidate)):
        result = subprocess.run([str(binary), str(int(cut))], cwd=build, env=environment,
                                capture_output=True, text=True, timeout=30)
        for suffix in ("stdout", "stderr"):
            (build / f"{mode}-{name}.{suffix}").write_text(getattr(result, suffix))
        assert result.returncode == 0, result.stdout + result.stderr
        results.append(result)
    reference, actual = results
    assert actual.stdout == reference.stdout
    assert "MAPPED_ENTRY_FULL_FIELDS_OK" in actual.stdout
    cases = re.findall(r"MAPPED_ENTRY_BEGIN (\d+) (\d+)\n(.*?)MAPPED_ENTRY_END \1 \2", actual.stderr, re.S)
    assert len(cases) == 4
    for size, _repeat, events in cases:
        lines = events.splitlines()
        launches = [i for i, line in enumerate(lines) if line.startswith("FORT_SCOPED launch ")]
        copies = [line for line in lines if line.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download "))]
        if hidden:
            assert not launches
            assert not copies
            continue
        n = int(size)
        # The proved native writer retires the first domain; its checked
        # successor may open a fresh one after that original call completes.
        expected_launches = 2 if n else 0
        assert len(launches) == expected_launches, events
        if cut:
            continue
        downloads = [line for line in copies if line.startswith("FORT_SCOPED download ")]
        sizes = [int(re.search(r"\bbytes=(\d+)\b", line)[1]) for line in downloads]
        assert sorted(sizes) == ([8*n, 8*n] if n else []), events
        if n:
            before_consumer = [line for line in lines[:launches[-1]] if line.startswith("FORT_SCOPED download ")]
            assert not before_consumer
            assert sum(int(re.search(r"\bbytes=(\d+)\b", line)[1]) for line in copies
                       if line.startswith("FORT_SCOPED upload ")) == 8*n + 4
