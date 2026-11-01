"""Actual child kernels retain replicas across host reads and retire before swaps."""
import json
import os
import re
import shutil
import subprocess

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_hierarchical_child_regions import READ_FIRST_SOURCE, SOURCE, fixture

DRIVER = """program exercise
use hierarchical
use iso_fortran_env,only:error_unit
implicit none
integer::case_index,n,mode
logical::change
character(8)::argument
call get_command_argument(1,argument)
if(trim(argument)=='fault')then
call advance(7,2,.true.)
error stop 'failed retirement returned to the original continuation'
endif
do mode=0,1
change=mode==1
write(error_unit,'(A)')'HIER_ZERO_BEGIN'
call advance(257,0,change)
write(error_unit,'(A)')'HIER_ZERO_END'
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
write(error_unit,'(A,1X,I0,1X,I0)')'HIER_CASE_BEGIN',mode,n
call advance(n,2,change)
write(error_unit,'(A,1X,I0,1X,I0)')'HIER_CASE_END',mode,n
if(any(shape(left)/=[n]).or.any(shape(right)/=[n]))error stop 'shape changed'
if(lbound(left,1)/=-3.or.lbound(right,1)/=2)error stop 'allocation bounds lost'
write(*,'(6(I0,1X))')mode,n,lbound(left,1),ubound(left,1),lbound(right,1),ubound(right,1)
write(*,'(*(ES26.17E3,1X))')observed,left,right
deallocate(left,right)
enddo
enddo
print *, 'HIERARCHICAL_CHILD_FIELDS_BOUNDS_AND_HOST_READ_OK'
end program
"""

# Test-only failure after actual work and successful domain publication.  The
# changed reported status must stop source execution, never replay either child.
FAILURE = r"""#include <cstdint>
#include <cstdio>
extern "C" int __real_fort_scope_domain_close_v1(std::uint64_t);
extern "C" int __wrap_fort_scope_domain_close_v1(std::uint64_t domain) {
    const int status = __real_fort_scope_domain_close_v1(domain);
    if (status != 0) return status;
    std::fputs("HIER_TEST_POST_SUBMISSION_RETIREMENT_ERROR\n", stderr);
    return 6;
}
"""


@pytest.fixture(scope="module", params=["overwrite", "read-first"])
def hierarchy_binaries(tmp_path_factory, request):
    source = SOURCE if request.param == "overwrite" else READ_FIRST_SOURCE

    def emit(directory):
        original, builder = fixture(directory, source)
        outputs, manifest = builder.run()
        assert manifest["scope_count"] == 1
        scope, = manifest["scopes"]
        assert scope["checked_execution_plan"]["storage_epoch"]["return_contract"]["version"] == 2
        assert {name.split("#")[0] for name in scope["gpu_leaves"]} == {
            "hierarchical::producer", "hierarchical::consumer"}
        return original, outputs, manifest

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "emit", emit)
        patch.setattr(base, "DRIVER", DRIVER)
        build, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    manifest = json.loads((build / "scope-manifest.json").read_text())
    provenance = {row["id"]: row for row in manifest["runtime_provenance"]["records"]}
    wrapper = build / "retirement-failure.cpp"
    wrapper.write_text(FAILURE)
    wrapper_object = build / "retirement-failure.o"
    host = shutil.which("g++-14") or shutil.which("g++")
    base.checked([host, "-std=c++17", "-c", str(wrapper), "-o", str(wrapper_object)], build, "compile-failure")
    command = json.loads((build / "link-candidate.command.json").read_text())
    fault = build / "fault-candidate"
    command[command.index("-o") + 1] = str(fault)
    base.checked([*command, str(wrapper_object), "-Wl,--wrap=fort_scope_domain_close_v1"], build, "link-failure")
    return request.param, build, candidate, native, fault, provenance


def events_in(text):
    result = []
    for line in text.splitlines():
        if not line.startswith("FORT_SCOPED "):
            continue
        if line.startswith("FORT_SCOPED domain "):
            record = json.loads(line.split(" ", 2)[2])
            result.append({"kind": record["event"], **record})
            continue
        fields = dict(re.findall(r"(\w+)=(\S+)", line))
        result.append({"kind": line.split()[1], **fields})
    return result


def procedure(event, provenance):
    return provenance.get(event.get("procedure"), {}).get("procedure")


def copy_bytes(events, kind):
    return sum(int(event["bytes"]) for event in events if event["kind"] == kind)


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True], ids=["enabled", "hidden"])
def test_child_host_read_replica_and_checked_retirement(hierarchy_binaries, hidden):
    variant, build, candidate, native, _fault, provenance = hierarchy_binaries
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        env["CUDA_VISIBLE_DEVICES"] = ""
    mode = "hidden" if hidden else "enabled"
    reference = base.checked([str(native)], build, mode + "-native", env)
    result = base.checked([str(candidate)], build, mode + "-candidate", env)
    assert result.stdout == reference.stdout
    assert "HIERARCHICAL_CHILD_FIELDS_BOUNDS_AND_HOST_READ_OK" in result.stdout
    zeros = re.findall(r"HIER_ZERO_BEGIN\n(.*?)HIER_ZERO_END", result.stderr, re.S)
    assert len(zeros) == 2
    for text in zeros:
        assert not any(event["kind"] in {"domain_create", "register", "launch"} for event in events_in(text))
    cases = re.findall(r"HIER_CASE_BEGIN (\d+) (\d+)\n(.*?)HIER_CASE_END \1 \2", result.stderr, re.S)
    assert len(cases) == 6
    ledger = []
    for changed, width, text in cases:
        n = int(width)
        events = events_in(text)
        launches = [index for index, event in enumerate(events) if event["kind"] == "launch"]
        row = {"changed": bool(int(changed)), "n": n,
               "launches": [procedure(events[index], provenance) for index in launches],
               "upload_bytes": copy_bytes(events, "upload"),
               "download_bytes": copy_bytes(events, "download"), "events": events}
        ledger.append(row)
        # Retain raw full ledgers even if a stricter retention assertion fails.
        (build / (mode + "-ledger.json")).write_text(json.dumps(ledger, indent=2) + "\n")
        if hidden:
            assert not launches, text
            assert row["upload_bytes"] == row["download_bytes"] == 0, text
            continue
        expected = ["hierarchical::producer", "hierarchical::consumer"] * 2
        if variant == "read-first" and n == 1:
            expected = ["hierarchical::consumer"] * 2
        assert row["launches"] == expected, text
        if variant == "read-first" and n == 1:
            continue
        for stage in range(2):
            first, last = launches[2 * stage:2 * stage + 2]
            between = events[first + 1:last]
            producer, consumer = events[first], events[last]
            written = n if variant == "overwrite" else n - 2
            assert copy_bytes(between, "download") == 8 * written, text
            if changed == "0":
                assert producer["domain"] == consumer["domain"], text
                assert not any(event["kind"] == "domain_retire" for event in between), text
                # A partial producer may still need the two untouched end values.
                assert copy_bytes(between, "upload") == (0 if variant == "overwrite" else 16), text
            else:
                assert producer["domain"] != consumer["domain"], text
                assert any(event["kind"] == "domain_retire" for event in between), text
                assert copy_bytes(between, "upload") == 8 * n, text
        if variant == "overwrite":
            assert row["download_bytes"] == 32 * n, text
            assert row["upload_bytes"] == (0 if changed == "0" else 16 * n), text


@pytest.mark.cuda
def test_reported_retirement_error_after_submission_never_replays(hierarchy_binaries):
    _variant, build, _candidate, _native, fault, provenance = hierarchy_binaries
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run([str(fault), "fault"], cwd=build, env=env,
                            capture_output=True, text=True, timeout=30)
    (build / "failure-candidate.stdout").write_text(result.stdout)
    (build / "failure-candidate.stderr").write_text(result.stderr)
    assert result.returncode != 0
    assert result.stderr.count("HIER_TEST_POST_SUBMISSION_RETIREMENT_ERROR") == 1
    assert "lexical owner publication failed" in result.stderr
    assert "failed retirement returned" not in result.stderr
    launches = [procedure(event, provenance) for event in events_in(result.stderr) if event["kind"] == "launch"]
    assert launches == ["hierarchical::producer"], result.stderr
