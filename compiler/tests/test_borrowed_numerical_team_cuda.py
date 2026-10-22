"""Reusable original teams retain one owner through GPU and native choices."""

import json
import os
import re
import shutil
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.numerical_contract import require_explicit_cuda_environment, require_numerical_build_contract
from compiler.tests.support.borrowed_numerical_team import (
    ENTRY,
    INITIALIZER,
    OWNER_SOURCE,
    SCALER,
    borrowed_team_launch_counts,
    emit_borrowed_team_fixture,
)
from compiler.tests.support.numerical_cuda import _transfers
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run

VOTE_PROBE = r"""#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
extern "C" int __real_fort_scope_numerical_environment_supported();
extern "C" int __wrap_fort_scope_numerical_environment_supported() {
    if (!std::getenv("FORT_TEST_BORROWED_TEAM_VETO") || omp_get_level() != 1)
        return __real_fort_scope_numerical_environment_supported();
    const int thread = omp_get_thread_num(), saved = std::fegetround();
    const bool veto = thread == omp_get_num_threads()-1;
    if (saved < 0 || (veto && std::fesetround(FE_UPWARD))) std::abort();
    const int supported = __real_fort_scope_numerical_environment_supported();
    if (veto && std::fesetround(saved)) std::abort();
    std::fprintf(stderr, "TEST_BORROWED_TEAM_VOTE thread=%d level=%d supported=%d\n",
                 thread, omp_get_level(), supported);
    return supported;
}
"""


def _driver(precision):
    return f"""program verify
use iso_fortran_env,only:error_unit
use amber_owner,only:advance,owner_visits
implicit none
real({precision}),allocatable::a(:,:,:),work(:,:,:),b(:,:,:),c(:,:,:),d(:,:,:)
real({precision})::gain
integer,parameter::sizes(3)=[2,7,31]
integer::shape,repeat,n,m,p,nx,ny,nz,low,i,j,k,unit,calls
character(16)::path
call get_command_argument(1,path)
if(path/='eligible'.and.path/='empty'.and.path/='partial') error stop 'unknown path'
calls=0
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
if(path=='empty'.and.shape/=2) cycle
n=sizes(shape)
m=3
p=2
nx=n+5
ny=m+4
nz=p+3
if(path=='empty') then
n=0
m=0
p=0
nx=0
ny=0
nz=0
endif
low=-11-shape
allocate(a(low:low+nx-1,low-1:low+ny-2,low-2:low+nz-3))
allocate(work(low-3:low+nx-4,low-4:low+ny-5,low-5:low+nz-6))
allocate(b(low-6:low+nx-7,low-7:low+ny-8,low-8:low+nz-9))
allocate(c(low-9:low+nx-10,low-10:low+ny-11,low-11:low+nz-12))
allocate(d(low-12:low+nx-13,low-13:low+ny-14,low-14:low+nz-15))
do repeat=1,2
do k=lbound(a,3),ubound(a,3)
do j=lbound(a,2),ubound(a,2)
do i=lbound(a,1),ubound(a,1)
a(i,j,k)=real(i+j+k+repeat,{precision})*0.125_{precision}
enddo
enddo
enddo
work=0.5_{precision}
b=-77.0_{precision}
c=-91.0_{precision}
d=-103.0_{precision}
gain=real(repeat+1,{precision})*0.125_{precision}
calls=calls+1
write(error_unit,'(a,i0)') 'BEGIN_CALL ',calls
call advance(a,work,b,c,d,n,m,p,gain,path=='partial')
write(error_unit,'(a,i0)') 'END_CALL ',calls
if(owner_visits/=calls) error stop 'original prefix replayed'
write(unit)shape,repeat,n,m,p,owner_visits
write(unit)lbound(a),ubound(a),lbound(work),ubound(work)
write(unit)lbound(b),ubound(b),lbound(c),ubound(c),lbound(d),ubound(d)
write(unit)a,work,b,c,d
enddo
deallocate(a,work,b,c,d)
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module", params=[4, 8], ids=["real4", "real8"])
def borrowed_team_binaries(tmp_path_factory, request):
    precision = request.param
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not all((fc, nvcc, host)):
        pytest.skip("native Fortran and CUDA toolchain required")
    require_explicit_cuda_environment()
    directory = tmp_path_factory.mktemp("borrowed_numerical_team_" + str(precision))
    source = OWNER_SOURCE.replace("n,m,p,gain)", "n,m,p,gain,reject_partial)", 1)
    source = source.replace("logical::original_mask",
                            "logical,intent(in)::reject_partial\nlogical::original_mask", 1)
    source = source.replace("call reset(b,0)",
        "if(reject_partial) call scale(work(:,:,0:0),gain)\ncall reset(b,0)", 1)
    originals, outputs, report, builder = emit_borrowed_team_fixture(
        directory / "original", precision=precision, owner_source=source)
    owner, = report["scopes"]
    boundary, = owner["boundaries"]
    assert "independent original owned storage: argument::work versus argument::a" in boundary["reason"]
    assert "native rectangular source-call mapping requires projected native hooks" in boundary["reason"]
    assert boundary["conditional_close"].lower() == "reject_partial"
    assert boundary["reopen"] is False
    emitted = outputs[report["sources"][str(originals[1])]["replacement"]].lower().replace(" ", "")
    assert emitted.count("callscale(work(:,:,0:0),gain)") == 1
    assert not owner["module_coordinators"]
    children = {child["procedure"]: child for child in report["borrowed_source_coordinators"]}
    assert set(children) == {INITIALIZER, SCALER}
    assert all(child["ownership"]["close_count"] == 0 for child in children.values())
    assert all(builder.facts["captures"]["argument::" + name]["initialized"] == "none" for name in ("b", "c", "d"))
    resources = {item["resource"]: item for item in owner["resource_bindings"]["resources"]}
    root_launches, helper_launches = borrowed_team_launch_counts(builder, report, rejected_actual="work(:,:,0:0)")
    calls = [call for call in report["resolved_calls"]
             if call["caller"] == ENTRY and call["procedure"] in {INITIALIZER, SCALER}]
    rejected, = [call for call in calls
                 if call["actual_arguments"][0].replace(" ", "").lower() == "work(:,:,0:0)"]
    assert rejected["procedure"] == SCALER
    assert (rejected["first_line"], rejected["last_line"]) == (boundary["first_line"], boundary["last_line"])
    assert len(calls) == 7
    whole_calls = [call for call in calls if call is not rejected]
    assert len(whole_calls) == 6
    for procedure in (INITIALIZER, SCALER):
        assert {call["resource_mapping"]["argument::values"] for call in whole_calls
                if call["procedure"] == procedure} == {"argument::b", "argument::c", "argument::d"}
    output, build, native_build = directory / "generated", directory / "build", directory / "native-build"
    output.mkdir()
    build.mkdir()
    native_build.mkdir()
    for name, text in outputs.items():
        artifact = output / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    (directory / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, expected in report["artifacts_sha256"].items():
        assert sha256((output / name).read_bytes()).hexdigest() == expected
    for path in originals:
        assert sha256(path.read_bytes()).hexdigest() == report["source_inputs"][str(path)]
    profile = actual_profile(directory, nvcc, host)
    (directory / "device-and-tool-identity.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    architecture = profile["hardware"]["compute_capability"].replace(".", "")
    flags = [fc, "-O3", "-std=f2018", "-fopenmp", "-ffp-contract=off", "-fcheck=all,array-temps"]
    commands, objects = [], []

    def execute(role, command, *, cwd=build):
        commands.append({"role": role, "argv": command, "cwd": str(cwd)})
        (directory / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result = run(command, cwd=cwd)
        (directory / f"command-{len(commands):02}.stdout").write_text(result.stdout)
        (directory / f"command-{len(commands):02}.stderr").write_text(result.stderr)
        return result

    original_order = {report["sources"][str(path)]["replacement"]: index for index, path in enumerate(originals)}
    for role in ("common_runtime", "shared_entry", "original_source"):
        sources = [item for item in report["build_sources"] if item["role"] == role]
        if role == "original_source":
            sources.sort(key=lambda item: original_order[item["path"]])
        for item in sources:
            source = output / item["path"]
            target = build / (str(len(objects)) + ".o")
            if item["language"] == "cuda":
                contract = item["numerical_contract"]
                require_numerical_build_contract(contract)
                compiler = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_" + architecture,
                            "-Xcompiler=-fopenmp", *contract["required_cuda_options"],
                            *("-Xcompiler=" + value for value in contract["required_host_options"])]
            else:
                compiler = flags
            execute(role, [*compiler, "-I", str(output), "-I", str(build), "-c", str(source), "-o", str(target)])
            objects.append(str(target))
    driver = directory / "driver.f90"
    driver.write_text(_driver(precision))
    library = Path(nvcc).resolve().parents[1] / "lib64"
    candidate, native, wrapped = build / "candidate", native_build / "native", build / "candidate-vote"
    link = [*flags, "-I", str(build), str(driver), *objects,
            "-L" + str(library), "-Wl,-rpath," + str(library), "-lcudart", "-lstdc++"]
    execute("candidate-link", [*link, "-o", str(candidate)])
    execute("native-compile-link", [*flags, *(str(path) for path in originals), str(driver), "-o", str(native)],
            cwd=native_build)
    probe, probe_object = directory / "vote-probe.cpp", build / "vote-probe.o"
    probe.write_text(VOTE_PROBE)
    execute("test-only-vote-compile", [host, "-O2", "-std=c++17", "-fopenmp", "-c", str(probe), "-o", str(probe_object)])
    execute("test-only-vote-link", [*link, str(probe_object),
        "-Wl,--wrap=fort_scope_numerical_environment_supported", "-o", str(wrapped)])
    return directory, candidate, native, wrapped, precision, resources, root_launches, helper_launches


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True], ids=["cuda", "disabled"])
@pytest.mark.parametrize("path", ["eligible", "empty"])
def test_reused_original_helpers_preserve_complete_fields_and_residency(borrowed_team_binaries, path, disabled):
    _check_chain(borrowed_team_binaries, path, disabled=disabled)


@pytest.mark.cuda
def test_actual_participant_veto_keeps_coherent_native_helpers_after_gpu_prefix(borrowed_team_binaries):
    _check_chain(borrowed_team_binaries, "eligible", veto=True)


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True], ids=["cuda", "disabled"])
def test_rejected_partial_helper_then_whole_calls_close_after_prefix_without_replay(borrowed_team_binaries, disabled):
    _check_chain(borrowed_team_binaries, "partial", disabled=disabled)


def _check_chain(binaries, path, *, disabled=False, veto=False):
    directory, candidate, native, wrapped, precision, resources, root_launches, helper_launches = binaries
    label = path + ("-veto" if veto else "-disabled" if disabled else "-cuda")
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    if veto:
        environment["FORT_TEST_BORROWED_TEAM_VETO"] = "1"
    fields, results = {}, {}
    for role, binary in (("native", native), ("candidate", wrapped if veto else candidate)):
        result = run([str(binary), path], cwd=binary.parent, env=environment)
        results[role] = result
        fields[role] = (binary.parent / "fields.bin").read_bytes()
        (directory / f"{label}-{role}.bin").write_bytes(fields[role])
        (directory / f"{label}-{role}.stderr").write_text(result.stderr)
        assert "array temporary" not in result.stderr.lower()
    assert fields["candidate"] == fields["native"]
    trace = results["candidate"].stderr
    pattern = r"BEGIN_CALL (\d+)\n(.*?)END_CALL \1\n"
    calls = re.findall(pattern, trace, re.DOTALL)
    assert [int(ordinal) for ordinal, _trace in calls] == list(range(1, 3 if path == "empty" else 7))
    events = ("FORT_SCOPED launch ", "FORT_SCOPED upload ", "FORT_SCOPED download ")
    assert not any(line.startswith(events) for line in re.sub(pattern, "", trace, flags=re.DOTALL).splitlines())
    ids = {root: item["registration_identity"] for root, item in resources.items()}
    numerical = not disabled and path != "empty"
    launches = (1 if path == "partial" else root_launches + (0 if veto else helper_launches)) if numerical else 0
    for ordinal, call_trace in calls:
        assert call_trace.count("FORT_SCOPED launch ") == launches, call_trace
        gpu_events = [line for line in call_trace.splitlines() if line.startswith(events)]
        if not numerical:
            assert not gpu_events
            continue
        contexts = [re.findall(r"\bcontext=(\S+)", line) for line in gpu_events]
        assert all(len(context) == 1 for context in contexts)
        assert len({context[0] for context in contexts}) == 1
        assert call_trace.count("FORT_SCOPED initialize ") == 1
        n = (2, 2, 7, 7, 31, 31)[int(ordinal) - 1]
        interior = n * 3 * 2 * precision
        whole = (n + 5) * 7 * 5 * precision
        up = {ids["argument::a"]: interior, ids["argument::work"]: interior}
        down = {ids["argument::work"]: interior}
        if path != "partial":
            for name in ("b", "c", "d"):
                if veto:
                    up[ids["argument::" + name]] = 2 * interior
                down[ids["argument::" + name]] = interior if veto else whole
        transfers = _transfers(call_trace)
        assert dict(transfers["upload"]) == up, call_trace
        assert dict(transfers["download"]) == down, call_trace
        if path == "partial":
            assert call_trace.count("FORT_SCOPED create ") == 1
            assert call_trace.count("FORT_SCOPED close ") == 1
            close = call_trace.index("FORT_SCOPED close ")
            assert call_trace.index("FORT_SCOPED launch ") < close
            assert call_trace.rindex("FORT_SCOPED download ") < close
            assert not any(line.startswith(events) for line in call_trace[close:].splitlines())
        if veto:
            votes = re.findall(r"TEST_BORROWED_TEAM_VOTE thread=(\d+) level=(\d+) supported=(\d+)", call_trace)
            assert Counter(votes) == Counter({("0", "1", "1"): 6, ("1", "1", "1"): 6,
                                            ("2", "1", "1"): 6, ("3", "1", "0"): 6})
    (directory / (label + "-receipt.json")).write_text(json.dumps({
        "schema_version": 1, "precision_bytes": precision, "path": path, "cuda_disabled": disabled,
        "test_only_actual_participant_veto": veto,
        "complete_fields_halos_bounds_and_original_visits_equal": True,
        "three_distinct_original_actual_mappings_reuse_two_team_workers": True,
        "original_out_entry_has_no_admitted_incoming_payload_definitions": True,
        "native_helpers_keep_same_context_after_non_idempotent_gpu_prefix": veto,
        "rejected_partial_helper_publishes_gpu_prefix_then_runs_original_suffix_once": path == "partial" and numerical,
        "all_cuda_events_are_inside_call_markers_and_one_context_per_call": True,
        "launches_per_call": launches, "native_sha256": sha256(fields["native"]).hexdigest(),
        "candidate_sha256": sha256(fields["candidate"]).hexdigest(),
        "trace_sha256": sha256(trace.encode()).hexdigest(), "performance_samples": 0,
    }, indent=2, sort_keys=True) + "\n")
