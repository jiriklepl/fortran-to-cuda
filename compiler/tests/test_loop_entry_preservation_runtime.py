"""A complete loop invariant never restores coverage killed inside the loop."""
import json
import os
import re
import subprocess

import pytest

from compiler.tests.test_owner_local_call_groups_runtime import DRIVER, SOURCE, build_local_group_binaries

LOOP = SOURCE.replace("intent(out)", "intent(inout)").replace("integer::prefix_visits=0,suffix_visits=0",
                      "integer::prefix_visits=0,suffix_visits=0,cycles=2")
LOOP = LOOP.replace("logical,intent(in)::flag", "logical,intent(in)::flag\ninteger::stage")
LOOP = LOOP.replace("call unknown_prefix()", "prefix_visits=prefix_visits+1\ndo stage=1,cycles")
LOOP = LOOP.replace("endif\ncall unknown_suffix()", "endif\nenddo\ncall unknown_suffix()")

EXPECTED = DRIVER.replace("integer::n,i,repeat,repeats", "integer::n,i,repeat,repeats,stage")
EXPECTED = EXPECTED.replace("alias=trim(mode)=='alias'", "alias=.false.\nif(trim(mode)=='zero_loop')cycles=0")
EXPECTED = EXPECTED.replace("  do i=1,n\n", "  do stage=1,cycles\n  do i=1,n\n")
EXPECTED = EXPECTED.replace("  enddo\n endif\n write(error_unit", "  enddo\n  enddo\n endif\n write(error_unit")


@pytest.fixture(scope="module")
def loop_binaries(tmp_path_factory):
    result = build_local_group_binaries(tmp_path_factory, source=LOOP, driver_source=EXPECTED,
                                       initialized="whole", lifetime="invocation")
    _build, manifest, _candidate, _native = result
    epoch = manifest["scopes"][0]["checked_execution_plan"]["storage_epoch"]["selected_position_epoch"]
    assert epoch["entry_preservation"]["version"] == 5
    assert set(epoch["entry_preservation"]["preserved_resources"]) == {
        "argument::a", "argument::b", "argument::c"}
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    validate_checked_execution_plans(manifest)
    return result


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["resident", "late", "zero", "zero_loop", "unallocated", "hidden"])
def test_repeated_loop_matches_native_with_exact_cuts_and_no_readonly_download(loop_binaries, mode):
    build, manifest, candidate, native = loop_binaries
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
           "FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("FORT_SCOPE_TEST_FAIL_ALLOC_AFTER", None)
    if mode == "hidden":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif mode == "late":
        env["FORT_SCOPE_TEST_FAIL_ALLOC_AFTER"] = "2"
    outputs = []
    for role, binary in (("native", native), ("candidate", candidate)):
        command = [str(binary), mode]
        (build/f"{mode}-{role}.command.json").write_text(json.dumps({"argv": command,
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "FORT_SCOPE_TEST_FAIL_ALLOC_AFTER": env.get("FORT_SCOPE_TEST_FAIL_ALLOC_AFTER")})+"\n")
        result = subprocess.run(command, cwd=build, env=env, capture_output=True, text=True, timeout=30)
        for suffix, text in (("stdout", result.stdout), ("stderr", result.stderr)):
            (build/f"{mode}-{role}.{suffix}").write_text(text)
        (build/f"{mode}-{role}.completion.json").write_text(json.dumps({"returncode": result.returncode})+"\n")
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(result)
    reference, result = outputs
    assert reference.stdout == result.stdout
    copies = {kind: [dict(re.findall(r"(\w+)=([^ ]+)", line)) for line in result.stderr.splitlines()
                     if line.startswith("FORT_SCOPED " + kind + " ")] for kind in ("upload", "download")}
    launches = result.stderr.count("FORT_SCOPED launch ")
    resources = manifest["scopes"][0]["checked_execution_plan"]["storage_epoch"]["resources"]
    readonly = next(item["registration_identity"] for item in resources if item["resource"] == "argument::c")
    assert not any(int(item["buffer"]) == readonly for item in copies["download"])
    ledger = {"mode": mode, "launches": launches, "copies": copies,
              "domains": result.stderr.count('"event":"domain_create"'),
              "upload_bytes": sum(int(item["bytes"]) for item in copies["upload"]),
              "download_bytes": sum(int(item["bytes"]) for item in copies["download"])}
    (build/f"{mode}-ledger.json").write_text(json.dumps(ledger, indent=2)+"\n")
    if mode in {"hidden", "zero", "zero_loop", "unallocated"}:
        assert not launches, result.stderr
        assert not copies["upload"], result.stderr
        assert not copies["download"], result.stderr
        if mode in {"zero_loop", "unallocated"}:
            assert not re.search(r"FORT_SCOPED (initialize|allocate) ", result.stderr)
    elif mode == "resident":
        assert launches == 12, result.stderr  # Two calls, two iterations, three workers.
        assert ledger["domains"] == 4, result.stderr
    else:
        # Allocation rejection resets for each fresh iteration's domain. Each
        # traversal commits its first worker once; exact fields detect replay.
        assert launches == 2, result.stderr
