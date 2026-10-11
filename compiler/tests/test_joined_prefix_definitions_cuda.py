"""Joined native interiors seed one GPU chain without undefined halo uploads."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.numerical_contract import (
    require_explicit_cuda_environment,
    require_numerical_build_contract,
)
from compiler.tests.test_joined_prefix_definitions import (
    emit_joined_prefix_fixture,
    joined_consumer_source,
    shifted_child_source,
)
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run


def _driver(precision, controls):
    initialization = ""
    if controls == "sentinel":
        initialization = f"seed=-77._{precision}\ndelta=-91._{precision}\nout=-105._{precision}"
    return f"""program verify
use coast_owner,only:advance,prefix_visits,tail_visits
implicit none
real({precision}),allocatable::a(:,:),seed(:,:),delta(:,:),out(:,:)
integer,parameter::ns(7)=[0,17,0,1,17,4096,17],ms(7)=[7,0,0,1,9,7,13],counts(2)=[1,3]
integer::shape,iteration,round_case,n,m,i,j,unit,lx,ly,calls
character(16)::path
call get_command_argument(1,path)
if(path/='eligible'.and.path/='empty'.and.path/='native_group') error stop 'unknown path'
calls=0
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(ns)
n=ns(shape)
m=ms(shape)
if(path=='empty'.and.n*m/=0) cycle
if(path=='eligible'.and.(n*m==0.or.m==13)) cycle
if(path=='native_group'.and.m/=13) cycle
lx=-11-shape
ly=-7-2*shape
allocate(a(lx:lx+n+1,ly:ly+m+1),seed(lx-4:lx+n-3,ly-3:ly+m-2))
allocate(delta(lx-6:lx+n-5,ly-5:ly+m-4),out(lx-8:lx+n-7,ly-7:ly+m-6))
do round_case=1,size(counts)
do iteration=1,2
do j=lbound(a,2),ubound(a,2)
do i=lbound(a,1),ubound(a,1)
a(i,j)=real(i-2*j+32*iteration+4*round_case,{precision})*0.125_{precision}
enddo
enddo
{initialization}
! OUT controls intentionally leave real output payload undefined before source
! definitions. The original native tail restores every halo before any read.
call advance(a,seed,delta,out,n,m,counts(round_case))
calls=calls+1
if(prefix_visits/=calls.or.tail_visits/=calls) error stop 'original prefix/tail replayed'
if(any(lbound(a)/=[lx,ly]).or.any(ubound(a)/=[lx+n+1,ly+m+1])) error stop 'input bounds changed'
if(any(lbound(seed)/=[lx-4,ly-3])) error stop 'seed bounds changed'
if(any(lbound(delta)/=[lx-6,ly-5])) error stop 'delta bounds changed'
if(any(lbound(out)/=[lx-8,ly-7])) error stop 'out bounds changed'
write(unit)shape,round_case,iteration,n,m,prefix_visits,tail_visits
write(unit)lbound(a),ubound(a),lbound(seed),ubound(seed)
write(unit)lbound(delta),ubound(delta),lbound(out),ubound(out)
write(unit)a,seed,delta,out
enddo
enddo
deallocate(a,seed,delta,out)
enddo
close(unit)
end program
"""


def _transfers(trace):
    transfers = {"upload": Counter(), "download": Counter()}
    for line in trace.splitlines():
        if not line.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download ")):
            continue
        match = re.match(r"FORT_SCOPED (upload|download) buffer=(\d+) generation=(\d+) bytes=(\d+) "
                         r"provenance_version=1 context=(\S+) buffer_handle=(\S+) ", line)
        assert match is not None, "transfer lacks public buffer/context provenance"
        direction, identity, generation, count, _context, _handle = match.groups()
        assert int(generation) == 1
        transfers[direction][int(identity)] += int(count)
    return transfers


@pytest.fixture(scope="module", params=(
    (4, False, "out"), (4, True, "out"), (8, False, "out"), (8, True, "out"),
    (4, False, "sentinel"), (4, True, "sentinel"), (8, False, "sentinel"), (8, True, "sentinel"),
))
def joined_prefix_binaries(tmp_path_factory, request):
    precision, resource_fallback, controls = request.param
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA, C++ and Fortran toolchain required")
    require_explicit_cuda_environment()
    directory = tmp_path_factory.mktemp("joined_prefix_" + str(precision) + "_" + controls)
    architecture = actual_profile(directory, nvcc, host)["hardware"]["compute_capability"].replace(".", "")
    child_source = joined_consumer_source()
    if controls == "sentinel":
        child_source = shifted_child_source(child_source)
    originals, outputs, report, owner = emit_joined_prefix_fixture(
        directory, precision=precision, device_budget=0 if resource_fallback else None, controls=controls,
        child_source=child_source)
    if isinstance(originals, Path):
        originals = (originals,)
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    for name in ("seed", "delta", "out"):
        assert owner.builder.facts["captures"]["argument::" + name]["initialized"] == "none"
    resources = {item["resource"]: item for item in scope["resource_bindings"]["resources"]}
    coverage = {item["resource"]: item for item in scope["reached_window"]["defined_coverage"]}
    assert coverage["argument::seed"]["on_activation"] == "sections"
    assert coverage["argument::seed"]["source_sections_on_activation"]["overwrites"]
    assert coverage["argument::out"]["on_activation"] == "none"
    for name, text in outputs.items():
        artifact = directory / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    flags = [fortran, "-O3", "-cpp", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-ffp-contract=off"]
    objects, commands = [], []
    order = {report["sources"][str(path)]["replacement"]: index for index, path in enumerate(originals)}
    for role in ("common_runtime", "shared_entry", "original_source"):
        sources = [item for item in report["build_sources"] if item["role"] == role]
        if role == "original_source":
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
                             "numerical_contract": item.get("numerical_contract")})
            objects.append(str(target))
    driver = directory / "driver.f90"
    driver.write_text(_driver(precision, controls))
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
    return directory, candidate, native, precision, resource_fallback, controls, resources


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("path", ["eligible", "empty", "native_group"])
def test_joined_native_interiors_do_not_upload_undefined_halos(joined_prefix_binaries, disabled, path):
    directory, candidate, native, precision, resource_fallback, controls, resources = joined_prefix_binaries
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
    launches = result.stderr.count("FORT_SCOPED launch ")
    numerical = path != "empty" and not disabled and not resource_fallback
    assert launches == ({"eligible": 24, "native_group": 4}.get(path, 0) if numerical else 0)
    transfers = _transfers(result.stderr)
    identities = {name: item["registration_identity"] for name, item in resources.items()}
    assert all(identity in identities.values() for counts in transfers.values() for identity in counts)
    # Each invocation uploads only the native-defined seed interior once.
    # The native-group path closes after its first GPU producer, before the
    # complete original team; subsequent enclosing iterations remain native.
    interior_cells = {"eligible": 1 + 17 * 9 + 4096 * 7, "native_group": 17 * 13}.get(path, 0)
    interior_bytes = 4 * interior_cells * precision if numerical else 0
    uploads, downloads = transfers["upload"], transfers["download"]
    assert uploads[identities["argument::seed"]] == interior_bytes
    assert uploads[identities["argument::delta"]] == 0
    assert uploads[identities["argument::out"]] == 0
    assert sum(uploads.values()) == interior_bytes
    assert downloads[identities["argument::seed"]] == 0
    assert downloads[identities["argument::delta"]] == interior_bytes
    output_bytes = interior_bytes if path == "eligible" else 0
    assert downloads[identities["argument::out"]] == output_bytes
    assert sum(downloads.values()) == interior_bytes + output_bytes
    context_launches = Counter()
    for line in result.stderr.splitlines():
        if line.startswith("FORT_SCOPED launch "):
            context, = re.findall(r"\bcontext=(\S+)", line)
            context_launches[context] += 1
    expected_contexts = {"eligible": {2: 12}, "native_group": {1: 4}}.get(path, {}) if numerical else {}
    assert Counter(context_launches.values()) == expected_contexts
    receipt = {"schema_version": 1, "precision_bytes": precision, "controls": controls, "path": path,
               "cuda_disabled": disabled, "resource_preexecution_fallback": resource_fallback,
               "complete_arrays_halos_bounds_and_original_visits_equal": True,
               "actual_undefined_halos_read_only_after_native_restoration": controls == "out",
               "sentinel_halos_preserved_with_no_admitted_initial_coverage": controls == "sentinel",
               "shifted_child_formals_preserve_original_logical_coordinates": controls == "sentinel",
               "initial_output_capture_facts": {name: "none" for name in ("seed", "delta", "out")},
               "source_joined_interior_coverage_used_for_first_registration": True,
               "native_join_after_gpu_producer_path": path == "native_group",
               "native_join_output_has_zero_cuda_transfers": path == "native_group",
               "independently_traced_path": True, "numerical_launches": launches,
               "launches_per_context": dict(context_launches), "registration_identities": identities,
               "upload_bytes_by_identity": dict(uploads), "download_bytes_by_identity": dict(downloads),
               "h2d_bytes": sum(uploads.values()), "d2h_bytes": sum(downloads.values()),
               "expected_seed_interior_bytes": interior_bytes, "zero_uninitialized_output_upload_bytes": True,
               "compared_bytes": len(actual), "native_sha256": sha256(expected).hexdigest(),
               "candidate_sha256": sha256(actual).hexdigest(),
               "trace_sha256": sha256(result.stderr.encode()).hexdigest(), "performance_samples": 0}
    (directory / (label + "-receipt.json")).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
