"""Real copy ledgers across empty and active source-proved native wrappers."""

import json
import re

import pytest

from compiler.tests import test_same_family_native_call_cuda as base
from compiler.tests.test_hierarchical_child_regions_cuda import copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment

CALL = "call native_worker(right,left,n)"
WRAPPER = """subroutine native_worker(values,readonly,n)
integer,intent(in)::n
real(8),intent(inout)::values(:)
real(8),intent(in)::readonly(:)
call apply(values,readonly,n,records,scale1,scale2,scale3)
contains
""" + (base.NATIVE.replace("subroutine native_worker(values,readonly,n,f1,f2,f3)",
                          "subroutine apply(values,readonly,n,src,f1,f2,f3)")
       .replace("procedure(scale1)::f1,f2,f3", "type(record),allocatable,intent(in)::src(:)\nprocedure(scale1)::f1,f2,f3")
       .replace("records", "src")) + "end subroutine\n"


def patch_fixture(patch):
    text = base.source().replace(base.NATIVE, WRAPPER).replace(base.CALL, CALL)
    patch.setattr(base, "source", lambda: text)
    patch.setattr(base, "CALL", CALL)
    patch.setattr(base, "NATIVE", WRAPPER)


def test_callback_fixture_retains_checked_empty_path_before_cuda_execution(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        patch_fixture(patch)
        _source, _outputs, report = base.sealed_emission(tmp_path)
    returned = report["scopes"][0]["checked_execution_plan"]["storage_epoch"]["return_contract"]
    assert returned["native_call_preservation_policy_version"] == 3


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch_fixture(patch)
        result = base.binaries.__wrapped__(tmp_path_factory)
    directory = result[0]
    report = json.loads((directory / "scope-manifest.json").read_text())
    returned = report["scopes"][0]["checked_execution_plan"]["storage_epoch"]["return_contract"]
    assert returned["native_call_preservation_policy_version"] == 3
    assert sum("empty_call_guard" in route for member in returned["members"] for route in member["routes"]) == 1
    return result


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["traced", "untraced", "hidden"])
def test_zero_trip_calls_keep_resident_values_and_active_calls_publish(binaries, mode):
    directory, candidate, native, _fault, provenance = binaries
    env = environment(mode)
    expected = base.base.checked([str(native)], directory, mode + "-native", env)
    result = base.base.checked([str(candidate)], directory, mode + "-candidate", env)
    assert result.stdout == expected.stdout
    assert "SAME_FAMILY_NATIVE_CALLBACK_FIELDS_HALOS_AND_BOUNDS_OK" in result.stdout
    if mode == "untraced":
        assert "FORT_SCOPED " not in result.stderr
        return
    for name in ("COLD", "ZERO"):
        text, = re.findall("FAMILY_" + name + "_BEGIN\n(.*?)FAMILY_" + name + "_END", result.stderr, re.S)
        assert not any(e["kind"] in {"domain_create", "register", "launch"} for e in events_in(text))
    cases = re.findall(r"FAMILY_CASE_BEGIN (-?\d+) (\d+)\n(.*?)FAMILY_CASE_END \1 \2", result.stderr, re.S)
    assert len(cases) == 9
    ledger = []
    for record_mode, width, text in cases:
        events = events_in(text)
        launches = [i for i, event in enumerate(events) if event["kind"] == "launch"]
        ledger.append({"record_mode": int(record_mode), "n": int(width), "events": events})
        if mode == "hidden":
            assert not launches
            assert copy_bytes(events, "upload") == copy_bytes(events, "download") == 0
            continue
        assert [procedure(events[i], provenance) for i in launches] == ["hierarchical::producer", "hierarchical::consumer"]
        first, last = launches
        between = events[first + 1:last]
        if int(record_mode) <= 0:
            assert events[first]["domain"] == events[last]["domain"]
            assert not any(e["kind"] == "domain_retire" for e in between)
            assert copy_bytes(between, "download") == copy_bytes(between, "upload") == 0
        else:
            assert events[first]["domain"] != events[last]["domain"]
            assert sum(e["kind"] == "domain_retire" for e in between) == 1
            assert copy_bytes(between, "download") == copy_bytes(between, "upload") == 8 * int(width)
    (directory / (mode + "-empty-call-ledger.json")).write_text(json.dumps(ledger, indent=2) + "\n")


@pytest.mark.cuda
def test_active_call_publication_failure_remains_fatal_before_callback(binaries):
    base.test_post_submission_close_failure_stops_before_native_call_or_replay(binaries)
