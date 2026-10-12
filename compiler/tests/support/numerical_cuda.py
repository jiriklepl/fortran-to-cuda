"""CUDA builds and strict field/traffic checks for numerical source fixtures.

Legacy actual_profile/run helpers remain at their existing test locations; this
focused module removes cross-imports among reached/combined numerical tests.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.numerical_contract import require_explicit_cuda_environment, require_numerical_build_contract
from compiler.tests.support.numerical_regions import ROOTS, emit_reached_numerical_fixture
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run


def _driver(precision):
    return f"""program verify
use iso_fortran_env,only:error_unit
use quartz_owner,only:advance,prefix_visits,tail_visits
use quartz_numerics,only:bridge_visits,middle_visits
implicit none
real({precision}),allocatable::a(:),b(:),c(:)
integer,parameter::uppers(3)=[0,7,4096]
integer::shape,repeat,n,tile,low,extent,i,unit,calls,nonempty
character(16)::path
call get_command_argument(1,path)
if(path/='eligible'.and.path/='negative'.and.path/='overflow'.and.path/='empty'.and.path/='late') &
error stop 'unknown path'
calls=0
nonempty=0
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(uppers)
if(path/='eligible'.and.shape/=2) cycle
n=uppers(shape)
if(path=='empty') n=-4
if(path=='overflow') n=10
extent=n+6
low=-10-shape
allocate(a(low:low+extent-1),b(low-3:low+extent-4),c(low-7:low+extent-8))
do repeat=1,2
tile=3
if(repeat==2) tile=7
! These nonpositive/overflowing proof inputs still have valid native execution.
! The negative outer DO is zero-work; INOUT b was initialized by the caller.
if(path=='negative') tile=-3
! A non-idempotent original GPU prefix precedes the same later failing guard.
if(path=='late') tile=-5
! -3+tile-1 fits INTEGER in the original sole positive-step tile.
if(path=='overflow') tile=2147483640
do i=lbound(a,1),ubound(a,1)
a(i)=real(i-low+2*repeat,{precision})*0.125_{precision}
enddo
b=-77._{precision}
c=-91._{precision}
calls=calls+1
if(n>=-3) nonempty=nonempty+1
write(error_unit,'(a,i0)') 'BEGIN_CALL ',calls
call advance(a,b,c,n,tile)
write(error_unit,'(a,i0)') 'END_CALL ',calls
if(prefix_visits/=calls.or.bridge_visits/=calls.or.tail_visits/=calls) error stop 'original prefix replayed'
if(middle_visits/=nonempty) error stop 'original native face update replayed'
if(lbound(a,1)/=low.or.ubound(a,1)/=low+extent-1) error stop 'input bounds changed'
if(lbound(b,1)/=low-3.or.ubound(b,1)/=low+extent-4) error stop 'producer bounds changed'
if(lbound(c,1)/=low-7.or.ubound(c,1)/=low+extent-8) error stop 'consumer bounds changed'
if(b(lbound(b,1))/=-77._{precision}.or.b(ubound(b,1))/=-77._{precision}) error stop 'producer halo changed'
if(c(lbound(c,1))/=-91._{precision}.or.c(ubound(c,1))/=-91._{precision}) error stop 'consumer halo changed'
write(unit)shape,repeat,n,tile,prefix_visits,bridge_visits,middle_visits,tail_visits
write(unit)lbound(a),ubound(a),lbound(b),ubound(b),lbound(c),ubound(c)
write(unit)a,b,c
enddo
deallocate(a,b,c)
enddo
close(unit)
end program
"""


def _transfers(trace):
    counts = {"upload": Counter(), "download": Counter()}
    for line in trace.splitlines():
        if not line.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download ")):
            continue
        match = re.match(r"FORT_SCOPED (upload|download) buffer=(\d+) generation=(\d+) bytes=(\d+) "
                         r"provenance_version=1 context=(\S+) buffer_handle=(\S+) ", line)
        assert match is not None, "transfer lacks public buffer/context provenance"
        direction, identity, generation, size, _context, _handle = match.groups()
        assert int(generation) == 1
        counts[direction][int(identity)] += int(size)
    return counts


def build_reached_numerical_binaries(tmp_path_factory, precision, resource_fallback, *,
                                    emit_fixture=emit_reached_numerical_fixture, prefix="reached_numerical_",
                                    driver_source=None, expected_gpu_leaves=3, expected_capture_states=None):
    """Build the same authenticated chain for source participation fixtures."""
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA, C++ and Fortran toolchain required")
    require_explicit_cuda_environment()
    directory = tmp_path_factory.mktemp(prefix + str(precision))
    architecture = actual_profile(directory, nvcc, host)["hardware"]["compute_capability"].replace(".", "")
    originals, outputs, report, owner = emit_fixture(
        directory, precision=precision, device_budget=0 if resource_fallback else None)
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == expected_gpu_leaves
    builder = getattr(owner, "builder", owner)
    expected_capture_states = ({ROOTS[1]: "whole", ROOTS[2]: "none"}
                               if expected_capture_states is None else expected_capture_states)
    for root, state in expected_capture_states.items():
        assert builder.facts["captures"][root]["initialized"] == state
    resources = {item["resource"]: item for item in scope["resource_bindings"]["resources"]}
    for name, text in outputs.items():
        artifact = directory / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    flags = [fortran, "-O3", "-cpp", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-ffp-contract=off"]
    order, unchanged_sources = {}, []
    for index, path in enumerate(originals):
        source_hash = sha256(path.read_bytes()).hexdigest()
        assert report["source_inputs"][str(path)] == source_hash
        if str(path) in report["sources"]:
            candidate_source = report["sources"][str(path)]["replacement"]
        else:
            assert path.parent == directory
            candidate_source = path.name
            unchanged_sources.append({"role": "original_source", "path": candidate_source, "language": "fortran",
                                      "source_input_sha256": source_hash})
        assert candidate_source not in order
        order[candidate_source] = index
    objects, commands = [], []
    for role in ("common_runtime", "shared_entry", "original_source"):
        sources = [item for item in report["build_sources"] if item["role"] == role]
        if role == "original_source":
            sources.extend(unchanged_sources)
            sources.sort(key=lambda item: order[item["path"]])
        for item in sources:
            artifact = directory / item["path"]
            target = artifact.with_suffix(".o")
            if item["language"] == "cuda":
                contract = item["numerical_contract"]
                require_numerical_build_contract(contract)
                compiler = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_" + architecture,
                            "-Xcompiler=-fopenmp", *contract["required_cuda_options"],
                            *("-Xcompiler=" + value for value in contract["required_host_options"])]
            else:
                compiler = flags
            command = [*compiler, "-I", str(directory), "-c", str(artifact), "-o", str(target)]
            run(command, cwd=directory)
            commands.append({"role": role, "path": item["path"], "language": item["language"], "argv": command,
                             "numerical_contract": item.get("numerical_contract"),
                             "source_input_sha256": item.get("source_input_sha256")})
            objects.append(str(target))
    driver = directory / "driver.f90"
    driver.write_text(_driver(precision) if driver_source is None else driver_source(precision))
    library = Path(nvcc).resolve().parents[1] / "lib64"
    candidate, native = directory / "candidate", directory / "native"
    for role, command in (
        ("candidate-link", [*flags, "-I", str(directory), str(driver), *objects,
                            "-L" + str(library), "-Wl,-rpath," + str(library),
                            "-lcudart", "-lstdc++", "-o", str(candidate)]),
        ("native-compile-link", [*flags, *(str(path) for path in originals), str(driver), "-o", str(native)]),
    ):
        run(command, cwd=directory)
        commands.append({"role": role, "argv": command})
    (directory / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (directory / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return directory, candidate, native, precision, resource_fallback, resources


def check_prepared_chain(binaries, disabled, path):
    """Check full native agreement, original visits and exact resident traffic."""
    directory, candidate, native, precision, resource_fallback, resources = binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    reference = run([str(native), path], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(candidate), path], cwd=directory, env=environment)
    actual = (directory / "fields.bin").read_bytes()
    label = path + ("-disabled" if disabled else "-cuda")
    (directory / (label + "-native.bin")).write_bytes(expected)
    (directory / (label + "-candidate.bin")).write_bytes(actual)
    (directory / (label + ".stderr")).write_text(result.stderr)
    assert actual == expected
    assert "array temporary" not in result.stderr.lower()
    assert "array temporary" not in reference.stderr.lower()
    numerical = path == "eligible" and not disabled and not resource_fallback
    numerical_prefix = path == "late" and not disabled and not resource_fallback
    launches_per_call = 2 if numerical else 1 if numerical_prefix else 0
    expected_calls = 6 if path == "eligible" else 2
    call_pattern = r"BEGIN_CALL (\d+)\n(.*?)END_CALL \1\n"
    call_traces = re.findall(call_pattern, result.stderr, re.DOTALL)
    assert [int(ordinal) for ordinal, _trace in call_traces] == list(range(1, expected_calls + 1))
    outside_calls = re.sub(call_pattern, "", result.stderr, flags=re.DOTALL)
    event_prefixes = ("FORT_SCOPED launch ", "FORT_SCOPED upload ", "FORT_SCOPED download ")
    assert not any(line.startswith(event_prefixes) for line in outside_calls.splitlines())
    for _ordinal, trace in call_traces:
        events = [line for line in trace.splitlines() if line.startswith(event_prefixes)]
        assert trace.count("FORT_SCOPED launch ") == launches_per_call
        if launches_per_call:
            event_contexts = [re.findall(r"\bcontext=(\S+)", line) for line in events]
            assert all(len(context) == 1 for context in event_contexts)
            assert len({context[0] for context in event_contexts}) == 1
        else:
            assert not events
    launches = result.stderr.count("FORT_SCOPED launch ")
    assert launches == expected_calls * launches_per_call
    transfers = _transfers(result.stderr)
    identities = {name: item["registration_identity"] for name, item in resources.items()}
    assert all(identity in identities.values() for counts in transfers.values() for identity in counts)
    uploads, downloads = transfers["upload"], transfers["download"]
    cells = 2 * (4 + 11 + 4100) if numerical else 2 * 11 if numerical_prefix else 0
    interior_bytes = cells * precision
    producer_upload_bytes = 6 * precision if numerical else interior_bytes if numerical_prefix else 0
    consumer_download_bytes = interior_bytes if numerical else 0
    assert uploads[identities["argument::a"]] == interior_bytes
    assert uploads[identities["argument::b"]] == producer_upload_bytes
    assert uploads[identities["argument::c"]] == 0
    assert sum(uploads.values()) == interior_bytes + producer_upload_bytes
    assert downloads[identities["argument::a"]] == 0
    assert downloads[identities["argument::b"]] == interior_bytes
    assert downloads[identities["argument::c"]] == consumer_download_bytes
    assert sum(downloads.values()) == interior_bytes + consumer_download_bytes
    for ordinal, trace in call_traces:
        per_call = _transfers(trace)
        call_cells = (4, 4, 11, 11, 4100, 4100)[int(ordinal) - 1] if numerical else 11 if numerical_prefix else 0
        call_bytes = call_cells * precision
        expected_up = {identities["argument::a"]: call_bytes,
                       identities["argument::b"]: precision if numerical else call_bytes if numerical_prefix else 0}
        expected_down = {identities["argument::b"]: call_bytes,
                         identities["argument::c"]: call_bytes if numerical else 0}
        assert dict(per_call["upload"]) == {identity: size for identity, size in expected_up.items() if size}
        assert dict(per_call["download"]) == {identity: size for identity, size in expected_down.items() if size}
    context_launches = Counter()
    for line in result.stderr.splitlines():
        if line.startswith("FORT_SCOPED launch "):
            context, = re.findall(r"\bcontext=(\S+)", line)
            context_launches[context] += 1
    assert Counter(context_launches.values()) == ({2: 6} if numerical else {1: 2} if numerical_prefix else {})
    receipt = {"schema_version": 1, "precision_bytes": precision, "path": path, "cuda_disabled": disabled,
               "resource_preexecution_fallback": resource_fallback,
               "complete_arrays_halos_bounds_and_original_visits_equal": True,
               "producer_input_sentinels_are_independently_defined": True,
               "consumer_sentinels_have_no_admitted_output_coverage": True,
               "prepared_source_helper_private_array_and_tile_guards_used": True,
               "native_face_update_retains_resident_producer_consumer_owner": numerical,
               "late_gpu_prefix_is_published_before_failed_later_guard": numerical_prefix,
               "non_idempotent_original_prefix_makes_replay_observable": path == "late",
               "all_cuda_events_are_inside_call_markers_and_one_context_per_call": True,
               "negative_original_step_has_zero_producer_work": path == "negative",
               "overflowing_proof_input_preserves_valid_original_execution": path == "overflow",
               "failed_preconditions_close_before_original_group_once": path in {"negative", "overflow", "late"},
               "closed_paths_do_not_reopen_or_replay": True, "independently_traced_calls": expected_calls,
               "numerical_launches": launches, "launches_per_context": dict(context_launches),
               "registration_identities": identities, "upload_bytes_by_identity": dict(uploads),
               "download_bytes_by_identity": dict(downloads), "h2d_bytes": sum(uploads.values()),
               "d2h_bytes": sum(downloads.values()), "compared_bytes": len(actual),
               "native_sha256": sha256(expected).hexdigest(), "candidate_sha256": sha256(actual).hexdigest(),
               "trace_sha256": sha256(result.stderr.encode()).hexdigest(), "performance_samples": 0}
    (directory / (label + "-receipt.json")).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
