"""Partial GPU definitions survive conservative original joined host work."""

import json
import os
import re
from collections import Counter

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as build_support
from compiler.tests.test_native_call_graph import SOURCE, generate
from compiler.tests.test_native_call_graph_cuda import DRIVER


def group_source(*, precise=False, child=False):
    text = SOURCE.replace("type(record),target::points", "type(record)::points")
    for axis in ("i", "j", "k"):
        text = text.replace(f"do {axis}=-1,6", f"do {axis}=0,5", 1)
    text = text.replace("call update(points,a)", """!$omp parallel do private(i)
! This original host read must not discard the GPU producer's replica.
do i=0,2
if(points(i)%aux>0.d0) c(i,1,1)=b(i,1,1)+points(i)%flux
enddo
!$omp end parallel do
visits=visits+1""").replace("c(i,j,k)=b(i,j,k)+a(i,j,k)", "a(i,j,k)=b(i,j,k)+c(i,j,k)")
    if precise:
        text = text.replace("b(i,j,k)=2*a(i,j,k)", "b(i,j,k)=2*a(i,j,k)\nc(i,j,k)=3*a(i,j,k)")
        text = text.replace("points(i)%aux>0.d0", "visits==0").replace("+points(i)%flux", "+0.5d0")
        text = text.replace("c(i,1,1)=b(i,1,1)+0.5d0", "c(i,1,1)=sum(b(0:2,1:1,1:1))+0.5d0")
    if child:
        head, tail = text.split("!$omp parallel do private(i)\n", 1)
        body, rest = tail.split("end subroutine\nsubroutine update", 1)
        body = re.sub(r"\b([abc])(?=\()", lambda match: {"a": "z", "b": "x", "c": "y"}[match[1]], body)
        text = (head + "call consume(b,c,a)\nend subroutine\nsubroutine consume(x,y,z)\n"
                "real(8),contiguous,intent(inout)::x(-1:,-1:,-1:),y(-1:,-1:,-1:),z(-1:,-1:,-1:)\n"
                "integer::i,j,k\n!$omp parallel do private(i)\n" + body
                + "end subroutine\nsubroutine update" + rest)
    return text


@pytest.fixture(scope="module", params=["whole-envelope", "precise-regions", "child-formals"])
def binaries(tmp_path_factory, request):
    precise = request.param == "precise-regions"

    def emit(directory):
        from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

        source, _, outputs, manifest = generate(directory, group_source(precise=precise,
            child=request.param == "child-formals"), capture_updates={
            "native_graph::b": {"storage": "stable", "initialized": "none",
                                "escapes": False, "allocation_changes": False}})
        assert manifest["scope_count"] == 1, manifest.get("graph_owner_attempt")
        validate_checked_execution_plans(manifest)
        return source, outputs, manifest

    driver = DRIVER.replace("points%aux=-119.d0", "points%aux=real(3-2*step,8)")
    if precise:
        driver = driver.replace("visits=0", "visits=step-1").replace("if(visits/=1)", "if(visits/=step)")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(build_support, "emit", emit)
        patch.setattr(build_support, "DRIVER", driver)
        return precise, build_support.epoch_binaries.__wrapped__(tmp_path_factory)


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True], ids=["enabled", "hidden"])
def test_original_group_keeps_partial_producer_and_readonly_replica(binaries, hidden):
    precise, (directory, candidate, native) = binaries
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        env["CUDA_VISIBLE_DEVICES"] = ""
    label = "hidden" if hidden else "enabled"
    build_support.checked([str(native)], directory, label + "-native", env)
    expected = (directory / "fields.bin").read_bytes()
    result = build_support.checked([str(candidate)], directory, label + "-candidate", env)
    assert (directory / "fields.bin").read_bytes() == expected
    manifest = json.loads((directory / "scope-manifest.json").read_text())
    resources = manifest["scopes"][0]["checked_execution_plan"]["resources"]
    identities = {item["resource"]: item["registration_identity"] for item in resources}
    calls = result.stderr.split("CALL_BEGIN")[1:]
    assert len(calls) == 2
    for call in calls:
        trace = call.split("CALL_END", 1)[0]
        events = [dict(re.findall(r"(\w+)=([^ ]+)", line)) | {"kind": line.split()[1]}
                  for line in trace.splitlines() if line.startswith("FORT_SCOPED ")
                  and "provenance_version=1" in line]
        launches = [i for i, event in enumerate(events) if event["kind"] == "launch"]
        if hidden:
            assert not launches
            assert not any(event["kind"] in {"upload", "download"} for event in events)
            continue
        assert len(launches) == 2
        assert trace.count("FORT_SCOPED close ") == 1, trace
        assert "uninitialized_read" not in trace
        totals = Counter()
        for event in events:
            if event["kind"] in {"upload", "download"}:
                totals[event["kind"], int(event["buffer"])] += int(event["bytes"])
        a, b, c = [identities["native_graph::" + name] for name in ("a", "b", "c")]
        expected_c_upload = 2392 if precise else 4096
        assert totals == {("upload", a): 1728, ("upload", b): 2368, ("upload", c): expected_c_upload,
                          ("download", b): 1728, ("download", a): 4096,
                          **({("download", c): 1728} if precise else {})}, trace
        assert sum(int(event["bytes"]) for event in events[launches[0] + 1:launches[1]]
                   if event["kind"] == "download") == (48 if precise else 1728)
