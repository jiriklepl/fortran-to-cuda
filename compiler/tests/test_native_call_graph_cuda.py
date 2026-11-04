"""Actual graph-native CALL retention, physical copies and safe prework failure."""
import json
import os
import re
import shutil

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as build_support
from compiler.tests.test_native_call_graph import generate
from compiler.tests.test_native_opaque_write_cuda import API_OBSERVER

DRIVER = """program verify
use native_graph
use iso_fortran_env,only:error_unit
implicit none
integer::i,j,k,step,unit
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do step=1,2
do k=-1,6
do j=-1,6
do i=-1,6
a(i,j,k)=real(i+7*j+31*k+step,8)/16.d0
enddo
enddo
enddo
b=-111.d0
c=-113.d0
points%flux=-117.d0
points%aux=-119.d0
visits=0
write(error_unit,*)'CALL_BEGIN',step
call advance()
write(error_unit,*)'CALL_END',step
if(visits/=1)error stop 'native CALL repeated or omitted'
if(any(lbound(a)/=-1).or.any(ubound(a)/=6))error stop 'input bounds'
if(any(lbound(b)/=-1).or.any(ubound(b)/=6))error stop 'producer bounds'
if(any(lbound(c)/=-1).or.any(ubound(c)/=6))error stop 'consumer bounds'
if(lbound(points,1)/=-1.or.ubound(points,1)/=2)error stop 'record bounds'
write(unit)a,b,c,points%flux,points%aux,visits
enddo
close(unit)
print *,'COMPLETE_NATIVE_CALL_FIELDS_OK'
end program
"""


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    checked = build_support.checked

    def emit(directory):
        from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

        source, _, outputs, manifest = generate(directory)
        assert len(manifest["scopes"]) == 1, manifest.get("graph_owner_attempt")
        assert not manifest["scopes"][0]["boundaries"]
        validate_checked_execution_plans(manifest)
        return source, outputs, manifest

    def instrument(command, build, stem, env=None):
        if stem == "link-candidate":
            source = build / "reservation-observer.cpp"
            source.write_text(API_OBSERVER)
            obj = build / "reservation-observer.o"
            host = shutil.which("g++-14") or shutil.which("g++")
            checked([host, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(build),
                     "-c", str(source), "-o", str(obj)], build, "compile-reservation-observer")
            command = [*command, str(obj), "-Wl,--wrap=fort_scope_register", "-Wl,--wrap=fort_scope_access_begin_v1"]
        return checked(command, build, stem, env)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(build_support, "emit", emit)
        patch.setattr(build_support, "DRIVER", DRIVER)
        patch.setattr(build_support, "checked", instrument)
        return build_support.epoch_binaries.__wrapped__(tmp_path_factory)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["gpu", "untraced", "hidden", "resource", "alias"])
def test_complete_native_call_keeps_fields_and_readonly_replica(binaries, mode):
    directory, candidate, native = binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}
    for name in ("FORT_RUNTIME_TRACE", "CUDA_VISIBLE_DEVICES", "FORT_C3_TEST_FAULT"):
        environment.pop(name, None)
    build_support.checked([str(native)], directory, mode + "-native", environment)
    expected = (directory / "fields.bin").read_bytes()
    assert len(expected) == 2 * ((3 * 8**3 + 8) * 8 + 4)
    if mode != "untraced":
        environment["FORT_RUNTIME_TRACE"] = "1"
    if mode == "hidden":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    if mode in {"resource", "alias"}:
        environment["FORT_C3_TEST_FAULT"] = mode
    result = build_support.checked([str(candidate)], directory, mode + "-candidate", environment)
    assert "COMPLETE_NATIVE_CALL_FIELDS_OK" in result.stdout
    assert (directory / "fields.bin").read_bytes() == expected
    if mode in {"hidden", "untraced"}:
        assert not re.search(r"FORT_SCOPED (?:launch|upload|download) ", result.stderr)
        return
    calls = result.stderr.split("CALL_BEGIN")[1:]
    assert len(calls) == 2
    manifest = json.loads((directory / "scope-manifest.json").read_text())
    resources = manifest["scopes"][0]["checked_execution_plan"]["resources"]
    identities = {item["resource"]: item["registration_identity"] for item in resources}
    ledgers = []
    for ordinal, call in enumerate(calls):
        trace = call.split("CALL_END", 1)[0]
        events = [dict(re.findall(r"(\w+)=([^ ]+)", line)) | {"kind": line.split()[1]}
                  for line in trace.splitlines() if line.startswith("FORT_SCOPED ") and "provenance_version=1" in line]
        launches = [index for index, event in enumerate(events) if event["kind"] == "launch"]
        failed = mode in {"resource", "alias"} and ordinal == 0
        # Prework rejection publishes the completed prefix, executes the CALL
        # once, and permits a new domain for the later consumer. Queued work
        # still executes once; execution and storage context IDs are distinct.
        assert len(launches) == 2, trace
        assert sum(item["kind"] == "close" for item in events) == (2 if failed else 1), trace
        assert len({item["domain"] for item in events}) == (2 if failed else 1), trace
        assert len({item["context"] for item in events if item["domain_role"] == "execution"}) == (2 if failed else 1), trace
        transfers = [item for item in events if item["kind"] in {"upload", "download"}]
        assert sorted((item["kind"], int(item["buffer"]), int(item["bytes"])) for item in transfers) == sorted([
            ("upload", identities["native_graph::a"], 4096),
            ("download", identities["native_graph::b"], 4096),
            ("download", identities["native_graph::c"], 4096),
            *([("upload", identities["native_graph::a"], 4096),
               ("upload", identities["native_graph::b"], 4096)] if failed else [])]), trace
        if not failed:
            assert not any(item["kind"] in {"upload", "download"}
                           for item in events[launches[0]+1:launches[1]]), trace
        ledgers.append(transfers)
    if mode == "alias":
        assert "OPAQUE_REAL_ALIAS_REJECT" in result.stderr
    (directory / (mode + "-ledger.json")).write_text(json.dumps(ledgers, indent=2) + "\n")
