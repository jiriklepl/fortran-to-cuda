"""Inherited lexical calls keep exact storage identities through real kernels."""

import json
import re

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests import test_same_family_native_call_cuda as original_fixture
from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_hierarchical_child_regions_cuda import copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment
from compiler.tests.test_native_retirement_scalar_formals_cuda import physical_gpu_events


def source(native_call):
    text = original_fixture.source().replace(
        "real(8)::observed", "real(8)::observed\nreal(8)::weights(0:1)=[3.0_8,4.0_8]", 1)
    old = "call consumer(left,right,n)\nend subroutine\nsubroutine advance"
    new = """call internal_helper
contains
subroutine internal_helper
call leaf(n)
end subroutine
end subroutine
subroutine advance"""
    assert text.count(old) == 1
    text = text.replace(old, new)
    leaf = """subroutine leaf(n)
integer,intent(in)::n
integer::j
observed=0.0_8
do j=0,1
observed=observed+weights(j)
enddo
call consumer(left,right,n)
end subroutine
"""
    text = text.replace("end module", leaf + "end module")
    if not native_call:
        assert text.count(original_fixture.CALL) == 1
        text = text.replace(original_fixture.CALL, "continue")
    return text


def driver(native_call):
    text = original_fixture.DRIVER
    text = text.replace("call advance(n,1,.false.)", """weights=[3.0_8+record_mode,7.0_8+2*record_mode]
call advance(n,1,.false.)
if(observed/=10.0_8+3*record_mode)error stop 'descendant native read was omitted'
if(any(weights/=[3.0_8+record_mode,7.0_8+2*record_mode]))error stop 'readonly weights changed'""")
    text = text.replace("write(*,'(*(ES26.17E3,1X))')left,right",
                        "write(*,'(*(ES26.17E3,1X))')left,right,weights,observed")
    if not native_call:
        text = text.replace("if(record_mode==1)expected=expected+6.0_8", "continue")
    return text.replace("SAME_FAMILY_NATIVE_CALLBACK_FIELDS_HALOS_AND_BOUNDS_OK",
                        "INHERITED_LEXICAL_FIELDS_HALOS_BOUNDS_AND_NATIVE_READ_OK")


def sealed_emission(directory, native_call):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    original, builder = fixture(directory, source(native_call))
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    owner = builder._conditional_return_draft.validate().owner
    helper = owner.members["hierarchical::middle::internal_helper"]
    assert helper.parent is owner.members["hierarchical::middle"]
    assert helper.resource_slot_namespace is helper.parent
    assert helper.resources is helper.parent.resources
    assert helper.canonical_bindings is helper.parent.canonical_bindings
    scope, = report["scopes"]
    epoch = scope["checked_execution_plan"]["storage_epoch"]
    returned = epoch["return_contract"]
    assert returned["version"] == 4
    assert returned["root_selection"]["selected_node_ids"]
    assert {name.split("#")[0] for name in scope["gpu_leaves"]} == {
        "hierarchical::producer", "hierarchical::consumer"}
    weights, = [row for row in epoch["resources"] if row["resource"] == "hierarchical::weights"]
    assert weights["storage"] == "opaque host reservation"
    assert weights["device_replica_authorized"] is False
    assert weights["initialization_authorized"] is False
    assert set(builder.facts["captures"]) == {"hierarchical::left", "hierarchical::right"}
    validate_checked_execution_plans(report)
    package = directory / "source-package"
    for name, text in outputs.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest = json.loads((package / "scope-manifest.json").read_text())
    _scope_package_build({"artifact_root": str(package), "scope_manifest": manifest})
    assert builder.finish()
    return original, outputs, manifest


@pytest.fixture(scope="module", params=[False, True], ids=["readonly", "retirement"])
def binaries(tmp_path_factory, request):
    native_call = request.param

    def emit(directory):
        return sealed_emission(directory, native_call)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "emit", emit)
        patch.setattr(base, "DRIVER", driver(native_call))
        directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    manifest = json.loads((directory / "scope-manifest.json").read_text())
    provenance = {row["id"]: row for row in manifest["runtime_provenance"]["records"]}
    resources = {row["resource"]: row["registration_identity"]
                 for row in manifest["scopes"][0]["checked_execution_plan"]["storage_epoch"]["resources"]}
    return native_call, directory, candidate, native, provenance, resources


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["traced", "untraced", "hidden"])
def test_nested_original_calls_preserve_replicas_and_native_read(binaries, mode):
    native_call, directory, candidate, native, provenance, resources = binaries
    env = environment(mode)
    reference = base.checked([str(native)], directory, mode + "-native", env)
    result = base.checked([str(candidate)], directory, mode + "-candidate", env)
    assert result.stdout == reference.stdout
    assert "INHERITED_LEXICAL_FIELDS_HALOS_BOUNDS_AND_NATIVE_READ_OK" in result.stdout
    if mode == "untraced":
        assert "FORT_SCOPED " not in result.stderr
        return
    for marker in ["COLD", "ZERO"]:
        text, = re.findall("FAMILY_" + marker + "_BEGIN\n(.*?)FAMILY_" + marker + "_END", result.stderr, re.S)
        assert not any(event["kind"] in {"domain_create", "register"} for event in events_in(text))
        assert not physical_gpu_events(text)
    if mode == "hidden":
        assert not physical_gpu_events(result.stderr)
    cases = re.findall(r"FAMILY_CASE_BEGIN (-?\d+) (\d+)\n(.*?)FAMILY_CASE_END \1 \2", result.stderr, re.S)
    assert len(cases) == 9
    ledger = []
    for record_mode, width, text in cases:
        n = int(width)
        events = events_in(text)
        ledger.append({"record_mode": int(record_mode), "n": n, "events": events})
        (directory / (mode + "-ledger.json")).write_text(json.dumps(ledger, indent=2) + "\n")
        if mode == "hidden":
            assert not physical_gpu_events(text)
            assert copy_bytes(events, "upload") == copy_bytes(events, "download") == 0
            continue
        launches = [index for index, event in enumerate(events) if event["kind"] == "launch"]
        assert [procedure(events[index], provenance) for index in launches] == [
            "hierarchical::producer", "hierarchical::consumer"]
        first, last = launches
        between = events[first+1:last]
        if native_call:
            assert events[first]["domain"] != events[last]["domain"]
            assert sum(event["kind"] == "domain_retire" for event in between) == 1
            assert copy_bytes(between, "download") == copy_bytes(between, "upload") == 8*n
        else:
            assert events[first]["domain"] == events[last]["domain"]
            assert not any(event["kind"] == "domain_retire" for event in between)
            assert copy_bytes(between, "download") == copy_bytes(between, "upload") == 0
        assert copy_bytes(events, "upload") == (8*n if native_call else 0)
        assert copy_bytes(events, "download") == 2*8*n
        assert not any(event["kind"] in {"upload", "download", "allocate"}
                       and int(event.get("buffer", -1)) == resources["hierarchical::weights"] for event in events)


@pytest.mark.parametrize("native_call", [False, True], ids=["readonly", "retirement"])
def test_nested_device_fixture_source_and_driver_are_exact(tmp_path, native_call):
    # The strict serialized publication gate also runs without CUDA hardware.
    sealed_emission(tmp_path, native_call)
    for text in [source(native_call), driver(native_call)]:
        assert all(line == line.rstrip() and len(line) <= 132 for line in text.splitlines())
