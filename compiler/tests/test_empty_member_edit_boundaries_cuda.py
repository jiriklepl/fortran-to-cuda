"""Genuine empty checked callees preserve fields and the live GPU domain."""

import json
import re
import shutil

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_empty_member_edit_boundaries import emit_package, empty_source
from compiler.tests.test_hierarchical_child_regions_cuda import DRIVER, copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment
from compiler.tests.test_native_retirement_scalar_formals_cuda import physical_gpu_events

EMPTY_DRIVER = (DRIVER.replace("do mode=0,1", "observed=123.5_8\ndo mode=0,1", 1)
                .replace("if(any(shape(left)", "if(observed/=123.5_8)error stop 'empty callee changed scalar'\nif(any(shape(left)"))


@pytest.fixture(scope="module", params=[(1, False), (6, True)], ids=["one", "six-nested"])
def binaries(tmp_path_factory, request):
    arity, nested = request.param

    def emit(directory):
        original, _builder, _owner, empty, outputs, manifest = emit_package(directory, arity=arity, nested=nested)
        assert empty.routine.execution is None
        assert not empty.resources
        scope, = manifest["scopes"]
        assert {name.split("#")[0] for name in scope["gpu_leaves"]} == {
            "hierarchical::producer", "hierarchical::consumer"}
        return original, outputs, manifest

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "emit", emit)
        patch.setattr(base, "DRIVER", EMPTY_DRIVER)
        directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    manifest = json.loads((directory / "scope-manifest.json").read_text())
    provenance = {row["id"]: row for row in manifest["runtime_provenance"]["records"]}
    return directory, candidate, native, provenance


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["traced", "untraced", "hidden"])
def test_empty_child_keeps_original_fields_and_does_not_cut_residency(binaries, mode):
    directory, candidate, native, provenance = binaries
    env = environment(mode)
    reference = base.checked([str(native)], directory, mode + "-native", env)
    result = base.checked([str(candidate)], directory, mode + "-candidate", env)
    assert result.stdout == reference.stdout
    assert "HIERARCHICAL_CHILD_FIELDS_BOUNDS_AND_HOST_READ_OK" in result.stdout
    if mode == "untraced":
        assert "FORT_SCOPED " not in result.stderr
        return
    zeros = re.findall(r"HIER_ZERO_BEGIN\n(.*?)HIER_ZERO_END", result.stderr, re.S)
    assert len(zeros) == 2
    for text in zeros:
        assert not any(event["kind"] in {"domain_create", "register", "launch"} for event in events_in(text))
        assert not physical_gpu_events(text)
    if mode == "hidden":
        assert not physical_gpu_events(result.stderr)
    cases = re.findall(r"HIER_CASE_BEGIN (\d+) (\d+)\n(.*?)HIER_CASE_END \1 \2", result.stderr, re.S)
    assert len(cases) == 6
    ledger = []
    for changed, width, text in cases:
        n = int(width)
        events = events_in(text)
        ledger.append({"change_flag": bool(int(changed)), "n": n, "events": events})
        (directory / (mode + "-ledger.json")).write_text(json.dumps(ledger, indent=2) + "\n")
        if mode == "hidden":
            assert not physical_gpu_events(text)
            assert copy_bytes(events, "upload") == copy_bytes(events, "download") == 0
            continue
        launches = [i for i, event in enumerate(events) if event["kind"] == "launch"]
        assert [procedure(events[i], provenance) for i in launches] == [
            "hierarchical::producer", "hierarchical::consumer"] * 2
        for stage in range(2):
            first, last = launches[2*stage:2*stage+2]
            between = events[first+1:last]
            assert events[first]["domain"] == events[last]["domain"]
            assert not any(event["kind"] == "domain_retire" for event in between)
            assert copy_bytes(between, "upload") == copy_bytes(between, "download") == 0
        assert copy_bytes(events, "upload") == 0
        assert copy_bytes(events, "download") == 4*8*n


@pytest.mark.parametrize(("arity", "nested"), [(1, False), (6, True)], ids=["one", "six-nested"])
def test_original_empty_callee_preserves_defined_scalars_fields_and_bounds(tmp_path, arity, nested):
    compiler = shutil.which("gfortran-15") or shutil.which("gfortran")
    assert compiler is not None
    original = tmp_path / "original.f90"
    original.write_text(empty_source(arity=arity, nested=nested))
    driver = tmp_path / "driver.f90"
    driver.write_text(EMPTY_DRIVER)
    native = tmp_path / "native"
    env = environment("hidden")
    base.checked([compiler, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(tmp_path),
                  str(original), str(driver), "-o", str(native)], tmp_path, "native-compile", env)
    result = base.checked([str(native)], tmp_path, "native-run", env)
    assert "HIERARCHICAL_CHILD_FIELDS_BOUNDS_AND_HOST_READ_OK" in result.stdout
    assert len(re.findall("HIER_CASE_BEGIN ", result.stderr)) == 6
