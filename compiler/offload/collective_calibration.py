"""Offline measurements of source-proven, generated persistent-team workers.

The fixture uses the public source-scope artifacts. It does not reproduce the
coordinator algorithm, measure an application, or tune a live invocation.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import statistics
from copy import deepcopy
from hashlib import sha256
from importlib.resources import files

from .profile import (
    SCOPED_TEAM_COST_NAMES,
    SCOPED_TEAM_PROTOCOL_ID,
    SCOPED_TEAM_RATE_NAMES,
    ProfileError,
    collective_protocol_identity,
    validate_profile,
)

REPETITIONS = 16
# Keep each source-proven team below its existing effect budget. Pool fixed
# complete team passes into each observation; no numerical call forks a team.
BATCH_PASSES = 8
OBSERVED_REPETITIONS = REPETITIONS * BATCH_PASSES
WARMUP_SAMPLES = 3
MEASURED_SAMPLES = 9
COMPUTE_STEPS = 64
AUTO_WORK_CALLS = 16
AUTO_OWNER_CALLS = 1
AUTO_OBSERVED_OWNER_CALLS = AUTO_OWNER_CALLS * BATCH_PASSES
FORTRAN_FLAGS = ("-std=f2018", "-O3", "-fopenmp")
PROTOCOL_FIT_SIZES = (4096, 16384)
PROTOCOL_HOLDOUT_SIZE = 8192


def _fit_protocol_costs(observations):
    """Fit fixed stages on equal samples from predetermined training shapes."""
    stages = {
        "owner_seconds": (2, 1, OBSERVED_REPETITIONS, OBSERVED_REPETITIONS),
        "descriptor_seconds": (2, 2, OBSERVED_REPETITIONS * 4, OBSERVED_REPETITIONS * 4 * 3),
        "entry_seconds": (2, 3, OBSERVED_REPETITIONS, OBSERVED_REPETITIONS),
        "cpu_worker_seconds": (1, 4, OBSERVED_REPETITIONS, OBSERVED_REPETITIONS),
        "gpu_worker_seconds": (2, 5, OBSERVED_REPETITIONS, OBSERVED_REPETITIONS),
        "native_call_seconds": (2, 6, OBSERVED_REPETITIONS, OBSERVED_REPETITIONS),
        "native_worker_seconds": (0, 7, OBSERVED_REPETITIONS, OBSERVED_REPETITIONS),
    }

    def samples(fixture, mode, size, stage, divisor, count):
        rows = observations.get((fixture, mode, size), ())
        selected = [row for row in rows if row["stage"] == stage]
        if (len(selected) != MEASURED_SAMPLES
                or sorted(row["sample"] for row in selected) != list(range(1, MEASURED_SAMPLES + 1))
                or any(row["calls"] != count for row in selected)):
            raise ValueError(f"collective production stage count mismatch: {fixture}: {mode}: {size}: {stage}")
        selected.sort(key=lambda row: row["sample"])
        return {"fixture": fixture, "mode": mode, "n": size, "stage": stage,
                "divisor": divisor, "expected_calls": count, "raw_samples": selected,
                "seconds": [row["seconds"] / divisor for row in selected]}

    records = []
    for name, (mode, stage, divisor, count) in stages.items():
        sizes = PROTOCOL_FIT_SIZES if mode == 2 else (1,) if mode == 1 else (0,)
        sources = [samples(fixture, mode, size, stage, divisor, count)
                   for fixture in ("memory", "compute") for size in sizes]
        pooled = {fixture: [value for source in sources if source["fixture"] == fixture
                            for value in source["seconds"]] for fixture in ("memory", "compute")}
        selected_fixture = max(pooled, key=lambda fixture: statistics.median(pooled[fixture]))
        values = pooled[selected_fixture]
        record = {"kind": "collective_cost", "name": name, "seconds": values,
                  "fit": {"method": "larger fixture median after equal pooling of fixed training shapes",
                          "selected_fixture": selected_fixture, "sizes": list(sizes), "sources": sources}}
        if name == "owner_seconds":
            guards = [samples(fixture, 3, 0, 1, OBSERVED_REPETITIONS, OBSERVED_REPETITIONS)
                      for fixture in ("memory", "compute")]
            guard = max(guards, key=lambda source: statistics.median(source["seconds"]))
            # Each training shape has the same fixed sample count. Repeat the
            # independently measured identity guard for every shape, in order.
            guard_values = guard["seconds"] * len(sizes)
            record["components"] = {
                "source_owner_seconds": values, "fortran_guard_seconds": guard_values,
                "guard_sources": guards, "selected_guard_fixture": guard["fixture"],
                "barrier": "outside exclusive helper stage; existing owner barrier charged once",
            }
            record["seconds"] = [a + b for a, b in zip(values, guard_values, strict=True)]
        records.append(record)
    return records


def fixture_source(kind, threads, precision, *, work_calls=1, owner_calls=REPETITIONS):
    """Fixed generic compute/memory cases plus an indirect native middle call."""
    if kind not in {"compute", "memory"}:
        raise ValueError("unknown collective calibration fixture")
    width = precision // 8
    if type(work_calls) is not int or not 1 <= work_calls <= AUTO_WORK_CALLS:
        raise ValueError("invalid fixed collective work-call count")
    if type(owner_calls) is not int or not 1 <= owner_calls <= REPETITIONS:
        raise ValueError("invalid fixed collective owner-call count")
    body = "b(i,1,1)=a(i,1,1)+0.25*c(i,1,1)"
    if kind == "compute":
        body = (
            """x1=a(i,1,1)
x2=c(i,1,1)
x3=a(i,1,1)*0.5
x4=c(i,1,1)*0.25
"""
            + """x1=x1*1.000001+0.000001
x2=x2*1.000002+0.000002
x3=x3*1.000003+0.000003
x4=x4*1.000004+0.000004
"""
            * COMPUTE_STEPS
            + "b(i,1,1)=x1+x2+x3+x4"
        )
    local = "integer::i\n" + (f"real({width})::x1,x2,x3,x4\n" if kind == "compute" else "")
    work_calls_source = "call work(a,b,c,n)\n" * work_calls
    return (
        f"""module team_operations
implicit none
contains
subroutine work(a,b,c,n)
real({width}),intent(in)::a(:,:,:),c(:,:,:)
real({width}),intent(inout)::b(:,:,:)
integer,intent(in)::n
{local}!$omp do
do i=1,n
{body}
end do
!$omp end do
end subroutine
subroutine native_lookup(a,c,permutation,n)
real({width}),intent(in)::a(:,:,:)
real({width}),intent(inout)::c(:,:,:)
integer,intent(in)::permutation(:),n
integer::i
!$omp do
do i=1,n
c(permutation(i),1,1)=a(permutation(i),1,1)
end do
!$omp end do
end subroutine
subroutine step(a,b,c,permutation,n)
real({width}),intent(in)::a(:,:,:)
real({width}),intent(inout)::b(:,:,:),c(:,:,:)
integer,intent(in)::permutation(:),n
{work_calls_source}\
call native_lookup(a,c,permutation,n)
end subroutine
subroutine original_identity(version,options)
use iso_fortran_env,only:compiler_version,compiler_options
character(*),intent(out)::version,options
version=compiler_version()
options=compiler_options()
end subroutine
subroutine observe_identity_guard(expected_version,expected_options)
use iso_fortran_env,only:compiler_version,compiler_options
use iso_c_binding,only:c_null_char
use fort_scoped_team_observer
character(*),intent(in)::expected_version,expected_options
integer::status
call fort_scope_team_observe_begin_v1(FORT_SCOPE_TEAM_OBSERVE_OWNER)
status=fort_scope_team_fortran_compatible_v1(compiler_version()//c_null_char, &
  compiler_options()//c_null_char,expected_version//c_null_char,expected_options//c_null_char)
call fort_scope_team_observe_end_v1(FORT_SCOPE_TEAM_OBSERVE_OWNER)
if(status==0) error stop 'offline Fortran identity guard'
end subroutine
end module
module team_callers
use team_operations,only:step
implicit none
contains
subroutine qualified(a,b,c,permutation,n)
real({width}),allocatable,intent(in)::a(:,:,:)
real({width}),allocatable,intent(inout)::b(:,:,:),c(:,:,:)
integer,allocatable,intent(in)::permutation(:)
integer,intent(in)::n
!$omp parallel default(none) shared(a,b,c,permutation,n) num_threads({threads})
"""
        + "call step(a,b,c,permutation,n)\n" * owner_calls
        + "!$omp end parallel\nend subroutine\nend module\n"
    )


def fixture_facts(path, threads):
    text = path.read_text().splitlines(keepends=True)
    team_first = next(i for i, line in enumerate(text, 1) if line.startswith("!$omp parallel"))
    team_last = next(i for i, line in enumerate(text, 1) if line.startswith("!$omp end parallel"))
    stable = {"storage": "stable", "initialized": "whole", "allocation_changes": False, "escapes": False}
    return {
        "schema_version": 2,
        "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
        "captures": {
            **{
                "argument::" + name: {**stable, "association": "shared_whole_storage", "descriptor_uniform": True}
                for name in ("a", "b", "c", "permutation")
            },
            "argument::n": {**stable, "association": "shared_immutable_control"},
        },
        "participation": {
            "kind": "omp_full_team",
            "dispatch": "qualified_companion",
            "entry": "team_operations::step",
            "host_threads": threads,
            "expected_omp_level": 1,
            "call_sites": [
                {
                    "source": str(path),
                    "caller": "team_callers::qualified",
                    "first_line": i,
                    "last_line": i,
                    "span_sha256": sha256(line.encode()).hexdigest(),
                    "team_first_line": team_first,
                    "team_last_line": team_last,
                    "uniform_guard": "unconditional",
                }
                for i, line in enumerate(text, 1)
                if line.startswith("call step(")
            ],
        },
    }


def _driver(public, kind, threads, precision):
    width = precision // 8
    # Numerical capture names are public ABI fields, not parsed emitted CUDA.
    arguments = {"a": "ha", "b": "hb", "c": "hc", "n": "n"}
    for name in public["argument_order"][2:]:
        if name not in arguments:
            if "_lb_" in name or "_lower_" in name:
                arguments[name] = "1"
            elif "_dim_" in name:
                arguments[name] = "n" if name.endswith("_1") else "1"
            else:
                raise ValueError("collective calibration cannot supply public ABI parameter: " + name)
    call = ", &\n  ".join(["context", "0_c_int", *[arguments[name] for name in public["argument_order"][2:]]])
    return f"""program observe_team
use iso_c_binding
use iso_fortran_env,only:compiler_version,compiler_options
use omp_lib
use team_operations,only:work,original_identity,observe_identity_guard
use team_callers,only:qualified
use fort_scoped_memory
use fort_scoped_team_observer
use {public["fortran_module"]},only:run_team
implicit none
real({width}),allocatable,target::a(:,:,:),b(:,:,:),c(:,:,:)
real({width}),allocatable::expected(:,:,:)
integer,allocatable::permutation(:)
integer(c_int64_t)::context,ha,hb,hc,calls
integer(c_size_t),target::extents(3)
integer(c_int64_t),target::lowers(3)
type(fort_scope_layout)::layout
integer::n,i,r,s,pass,mode,status,stage
real(c_double)::seconds
character(32)::arg
character(8192)::expected_version,expected_options
character(8192)::actual_version,actual_options
call get_command_argument(1,arg)
read(arg,*) n
call get_command_argument(2,arg)
read(arg,*) mode
call get_command_argument(3,expected_version)
call get_command_argument(4,expected_options)
call omp_set_dynamic(.false.)
call original_identity(actual_version,actual_options)
allocate(a(n,1,1),b(n,1,1),c(n,1,1),expected(n,1,1),permutation(n))
do i=1,n
a(i,1,1)=1.0+real(i,{width})/real(max(n,1),{width})
permutation(i)=i
enddo
c=a
b=0
expected=0
call work(a,expected,c,n)
if(mode==1) then
status=fort_scope_create(0_c_int,context)
if(status/=0) error stop 'CPU calibration context'
extents=[int(n,c_size_t),1_c_size_t,1_c_size_t]
lowers=1_c_int64_t
layout=fort_scope_layout(3,{"FORT_SCOPE_REAL64" if precision == 64 else "FORT_SCOPE_REAL32"}, &
  {width}_c_size_t,c_null_ptr,c_loc(extents),c_loc(lowers),1_c_int64_t)
if(n>0) layout%host=c_loc(a)
status=fort_scope_register(context,1_c_int64_t,1_c_int64_t,layout,1_c_int,ha)
if(status/=0) error stop 'CPU input registration'
if(n>0) layout%host=c_loc(b)
status=fort_scope_register(context,2_c_int64_t,1_c_int64_t,layout,1_c_int,hb)
if(status/=0) error stop 'CPU output registration'
if(n>0) layout%host=c_loc(c)
status=fort_scope_register(context,3_c_int64_t,1_c_int64_t,layout,1_c_int,hc)
if(status/=0) error stop 'CPU input registration'
endif
do s=1,{WARMUP_SAMPLES + MEASURED_SAMPLES}
status=fort_scope_team_observer_reset_v1({threads}_c_int32_t)
if(status/=0) error stop 'observer reset'
if(mode==2.or.mode==4) then
do pass=1,merge(1,{BATCH_PASSES},mode==4)
call qualified(a,b,c,permutation,n)
enddo
else
!$omp parallel default(none) shared(a,b,c,n,context,ha,hb,hc,mode,expected_version,expected_options) &
!$omp private(r,status) num_threads({threads})
do r=1,{OBSERVED_REPETITIONS}
if(mode==0) then
call fort_scope_team_observe_begin_v1(FORT_SCOPE_TEAM_OBSERVE_COMPUTE)
call work(a,b,c,n)
call fort_scope_team_observe_end_v1(FORT_SCOPE_TEAM_OBSERVE_COMPUTE)
else if(mode==3) then
call observe_identity_guard(trim(expected_version),trim(expected_options))
!$omp barrier
else
status=run_team( &
  {call})
if(status/=0) error stop 'generated CPU calibration'
endif
enddo
!$omp end parallel
endif
if(mode==4) exit
if(s<={WARMUP_SAMPLES}) cycle
do stage=1,8
status=fort_scope_team_observer_read_v1(int(stage,c_int32_t),{threads}_c_int32_t,seconds,calls)
if(status/=0) error stop 'observer read'
write(*,'(A,I0,A,I0,A,ES24.16,A,I0,A)') '{{"stage":',stage,',"sample":',s-{WARMUP_SAMPLES}, &
  ',"seconds":',seconds,',"calls":',calls,'}}'
enddo
enddo
if(mode/=3.and.(any(b/=b).or.any(abs(b-expected)>1024*epsilon(b)*max(1.0_{width},abs(expected))))) &
  error stop 'collective calibration complete field mismatch'
if(any(c/=a)) error stop 'collective calibration native update mismatch'
do i=1,n
if(permutation(i)/=i) error stop 'immutable calibration index mismatch'
enddo
if(mode==1) then
status=fort_scope_close(context)
if(status/=0) error stop 'CPU calibration close'
endif
write(*,'(A)') 'COMPILER_VERSION='//trim(actual_version)
write(*,'(A)') 'COMPILER_OPTIONS='//trim(actual_options)
end program
"""


def calibrate_collective(profile, args, directory, nvcc, host, *, run, tool):
    """Build public artifacts sequentially and fit explicit offline costs."""
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.scopes.source import form_source_scopes

    fortran = tool(getattr(args, "fortran", None), ("gfortran-15", "gfortran-14", "gfortran"))
    # Explicit flags replace defaults so calibration can match an application's
    # exact ordered semantic options; independent build paths remain excluded.
    flags = list(getattr(args, "fortran_flag", None) or FORTRAN_FLAGS)
    if "-fopenmp" not in flags:
        raise ValueError("collective Fortran flags must include -fopenmp")
    root = directory / "collective"
    root.mkdir(exist_ok=True)
    records, commands, observations, identity = [], [], {}, None
    environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE")
    for name in list(environment):
        if name in {"FORT_RUNTIME_TRACE", "FORT_PHASE_TIMING", "CUDA_LAUNCH_BLOCKING"} or name.startswith(
            "FORT_SCOPE_TEST_"
        ):
            environment.pop(name)
    for kind, size in (("memory", 1 << 20), ("compute", 1 << 15)):
        build = root / kind
        build.mkdir(exist_ok=True)
        original = build / "original.f90"
        original.write_text(fixture_source(kind, args.threads, args.precision))
        facts = fixture_facts(original, args.threads)
        (build / "facts.json").write_text(json.dumps(facts, indent=2) + "\n")
        outputs, manifest = form_source_scopes(
            [original],
            "team_operations::step",
            facts=facts,
            options=CompilerOptions(gpu_policy="sections", memory_model="scoped"),
            config=OffloadConfig("sections", None, args.threads, True),
        )
        if manifest["scope_count"] != 1 or manifest["scopes"][0]["gpu_leaves"] != ["team_operations::work"]:
            raise ValueError(
                "collective calibration fixture has no proved mixed source owner: " + str(manifest["boundaries"])
            )
        if manifest["runtime"]["runtime_id"] != profile["scoped"]["runtime_id"]:
            raise ValueError("common runtime changed after scoped calibration")
        (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for name, content in outputs.items():
            target = build / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        # The manifest's reusable numerical artifact is also a public API.
        entry = next(
            item for item in manifest["scopes"][0]["numerical_entries"] if item["procedure"] == "team_operations::work"
        )
        public = entry["public"]
        driver = build / "driver.f90"
        driver.write_text(_driver(public, kind, args.threads, args.precision))
        objects = []
        for role in ("common_runtime", "shared_entry", "original_source"):
            for item in manifest["build_sources"]:
                if item["role"] != role:
                    continue
                target = build / (item["path"].replace("/", "_") + ".o")
                if item["language"] == "cuda":
                    command = [
                        nvcc,
                        "-O3",
                        "-std=c++17",
                        "-arch=" + args.arch,
                        "-ccbin",
                        host,
                        "-Xcompiler=-fopenmp",
                        "-DFORT_SCOPE_CALIBRATION",
                        "-I",
                        str(build),
                        "-c",
                        str(build / item["path"]),
                        "-o",
                        str(target),
                    ]
                else:
                    command = [fortran, *flags, "-c", str(build / item["path"]), "-o", str(target)]
                run(command, build, build / (target.name + ".log"), timeout=180)
                commands.append(command)
                objects.append(str(target))
        binary = build / "observe"
        command = [
            fortran,
            *flags,
            str(driver),
            *objects,
            "-L/usr/local/cuda/lib64",
            "-Wl,-rpath,/usr/local/cuda/lib64",
            "-lcudart",
            "-lstdc++",
            "-o",
            str(binary),
        ]
        run(command, build, build / "link.log", timeout=180)
        commands.append(command)
        for mode in (0, 1, 2, 3):
            for n in (0, size) if mode == 0 else (0, 1, size) if mode == 1 else (*PROTOCOL_FIT_SIZES, PROTOCOL_HOLDOUT_SIZE) if mode == 2 else (0,):
                command = [str(binary), str(n), str(mode)]
                if mode == 3:
                    command += [identity["compiler_version"], normalize_fortran_options(identity["compiler_options"])]
                text = run(command, build, build / f"mode-{mode}-n-{n}.log", timeout=180, env=environment)
                commands.append(command)
                rows = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
                if len(rows) != 8 * MEASURED_SAMPLES:
                    raise ValueError("collective fixture did not produce all complete stage samples")
                observations[kind, mode, n] = rows
                current = {
                    key: next(line.split("=", 1)[1] for line in text.splitlines() if line.startswith(prefix))
                    for key, prefix in (
                        ("compiler_version", "COMPILER_VERSION="),
                        ("compiler_options", "COMPILER_OPTIONS="),
                    )
                }
                if identity is not None and current != identity:
                    raise ValueError("collective Fortran compiler identity changed between measurements")
                identity = current
        for mode, prefix in ((0, "native_"), (1, "")):
            values = [
                row["seconds"] / OBSERVED_REPETITIONS for row in observations[kind, mode, size] if row["stage"] == 7
            ]
            # Generated throughput includes its collective completion barrier.
            # Original throughput excludes the separately charged original
            # empty worksharing call, avoiding duplicate native coordination.
            baseline = (
                [row["seconds"] / OBSERVED_REPETITIONS for row in observations[kind, mode, 0] if row["stage"] == 7]
                if mode == 0
                else [0.0] * len(values)
            )
            name = prefix + ("cpu_bandwidth" if kind == "memory" else "cpu_flops")
            # The compiler's calibrated compute rate consumes its public work
            # units (assignments and operations), not only physical FLOPs.
            # Preserve both bases so the numerical/model distinction is clear.
            unit_work = public["planning"]["units"][0]["work_per_iteration"]
            if kind == "compute" and (type(unit_work) not in {float, int} or unit_work <= 0):
                raise ValueError("fixed collective compute work estimate is unavailable")
            work = size * (3 * (args.precision // 8) if kind == "memory" else unit_work)
            records.append(
                {
                    "kind": "collective_rate",
                    "name": name,
                    "work": work,
                    "seconds": values,
                    "baseline_seconds": baseline,
                    "fixture": kind,
                    "threads": args.threads,
                    "work_basis": "array bytes" if kind == "memory" else "public planning work units",
                    "physical_flops": size * (2 if kind == "memory" else COMPUTE_STEPS * 8 + 5),
                    "public_work_per_iteration": unit_work,
                }
            )
    records.insert(
        0,
        {
            "kind": "collective_identity",
            "threads": args.threads,
            "level": 1,
            "protocol_id": SCOPED_TEAM_PROTOCOL_ID,
            **identity,
        },
    )
    records.extend(_fit_protocol_costs(observations))
    # Validation uses measured exclusive protocol totals from a different
    # warmed shape; numerical/API work remains separately excluded.
    costs = {row["name"]: statistics.median(row["seconds"]) for row in records if row["kind"] == "collective_cost"}
    for kind in ("memory", "compute"):
        rows = observations[kind, 2, PROTOCOL_HOLDOUT_SIZE]
        guard = [row["seconds"] / OBSERVED_REPETITIONS for row in observations[kind, 3, 0] if row["stage"] == 1]
        sums = [
            sum(row["seconds"] for row in rows if row["sample"] == sample and row["stage"] in (1, 2, 3, 5, 6))
            / OBSERVED_REPETITIONS
            + guard[sample - 1]
            for sample in range(1, MEASURED_SAMPLES + 1)
        ]
        predicted = (
            costs["owner_seconds"]
            + 4 * costs["descriptor_seconds"]
            + costs["entry_seconds"]
            + costs["gpu_worker_seconds"]
            + costs["native_call_seconds"]
        )
        records.append(
            {
                "kind": "collective_validation",
                "fixture": kind,
                "n": PROTOCOL_HOLDOUT_SIZE,
                "seconds": sums,
                "predicted_seconds": predicted,
                "comparison_scope": "exclusive production coordination; numerical and runtime API work excluded",
            }
        )
    (root / "measurements.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    from compiler.emission.common.resources import read_scoped_runtime

    if read_scoped_runtime()[1]["runtime_id"] != profile["scoped"]["runtime_id"]:
        raise ValueError("common runtime changed during collective calibration")
    provenance = {
        "build_commands": commands,
        "fortran": fortran,
        "fortran_flags": flags,
        "artifacts": str(root),
        "fixed_sizes": {"memory": 1 << 20, "compute": 1 << 15},
        "protocol_fit_sizes": list(PROTOCOL_FIT_SIZES),
        "protocol_holdout_size": PROTOCOL_HOLDOUT_SIZE,
        "repetitions": REPETITIONS,
        "team_passes_per_sample": BATCH_PASSES,
        "observed_repetitions_per_sample": OBSERVED_REPETITIONS,
        "warmup_samples": WARMUP_SAMPLES,
        "measured_samples": MEASURED_SAMPLES,
        "sampling": "fixed batches and warmups; equal pooled training shapes, independent holdout; no application profiling, live tuning, or affinity changes",
        "validation_scope": "exclusive production coordination, not whole-owner execution prediction",
        "openmp_environment": {
            name: environment.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OMP_DYNAMIC",
                "OMP_PROC_BIND",
                "OMP_PLACES",
                "OMP_WAIT_POLICY",
                "GOMP_SPINCOUNT",
            )
        },
        "observer_header_sha256": sha256((root / "memory" / "scoped_team_observer.hpp").read_bytes()).hexdigest(),
        "helper_sources": {
            name: sha256(files("compiler.offload").joinpath(name).read_bytes()).hexdigest()
            for name in ("collective_calibration.py", "calibrate.py", "profile.py")
        },
    }
    initial = profile_with_collective_measurements(profile, records, calibration=provenance)
    records += _validate_auto_holdout(initial, args, root, nvcc, host, fortran, flags, environment, run, commands)
    (root / "measurements.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    return profile_with_collective_measurements(profile, records, calibration=provenance)


def _validate_auto_holdout(profile, args, root, nvcc, host, fortran, flags, environment, run, commands):
    """Validate the combined protocol on public AUTO artifacts, then freeze it."""
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.scopes.source import form_source_scopes

    build = root / "auto-holdout"
    build.mkdir(exist_ok=True)
    original = build / "original.f90"
    original.write_text(
        fixture_source(
            "compute", args.threads, args.precision, work_calls=AUTO_WORK_CALLS, owner_calls=AUTO_OWNER_CALLS
        )
    )
    outputs, manifest = form_source_scopes(
        [original],
        "team_operations::step",
        facts=fixture_facts(original, args.threads),
        options=CompilerOptions(gpu_policy="auto", memory_model="scoped"),
        config=OffloadConfig("auto", profile, args.threads, True),
    )
    if manifest["scope_count"] != 1 or not manifest["automatic_estimate_available"]:
        raise ValueError("public collective auto holdout unavailable: " + str(manifest["boundaries"]))
    (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, content in outputs.items():
        target = build / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    public = next(
        item["public"]
        for item in manifest["scopes"][0]["numerical_entries"]
        if item["procedure"] == "team_operations::work"
    )
    driver = build / "driver.f90"
    driver.write_text(_driver(public, "compute", args.threads, args.precision))
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] != role:
                continue
            target = build / (item["path"].replace("/", "_") + ".o")
            if role == "common_runtime":
                old = root / "compute" / item["path"]
                if not old.exists() or old.read_bytes() != (build / item["path"]).read_bytes():
                    raise ValueError("auto holdout common runtime artifact changed")
                target = root / "compute" / (item["path"].replace("/", "_") + ".o")
            else:
                command = (
                    [
                        nvcc,
                        "-O3",
                        "-std=c++17",
                        "-arch=" + args.arch,
                        "-ccbin",
                        host,
                        "-Xcompiler=-fopenmp",
                        "-DFORT_SCOPE_CALIBRATION",
                        "-I",
                        str(build),
                        "-c",
                        str(build / item["path"]),
                        "-o",
                        str(target),
                    ]
                    if item["language"] == "cuda"
                    else [
                        fortran,
                        *flags,
                        "-I",
                        str(root / "compute"),
                        "-c",
                        str(build / item["path"]),
                        "-o",
                        str(target),
                    ]
                )
                run(command, build, build / (target.name + ".log"), timeout=180)
                commands.append(command)
            objects.append(str(target))
    binary = build / "observe"
    command = [
        fortran,
        *flags,
        "-I",
        str(root / "compute"),
        str(driver),
        *objects,
        "-L/usr/local/cuda/lib64",
        "-Wl,-rpath,/usr/local/cuda/lib64",
        "-lcudart",
        "-lstdc++",
        "-o",
        str(binary),
    ]
    run(command, build, build / "link.log", timeout=180)
    commands.append(command)
    costs = profile["scoped"]["collective"]["costs"]
    records = []
    for n in (4096, 1 << 22):
        command = [str(binary), str(n), "2"]
        text = run(command, build, build / f"n-{n}.log", timeout=180, env=environment)
        commands.append(command)
        rows = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
        if len(rows) != 8 * MEASURED_SAMPLES:
            raise ValueError("public auto holdout produced incomplete observations")
        gpu = [row["calls"] for row in rows if row["stage"] == 5]
        cpu = [row["calls"] for row in rows if row["stage"] == 4]
        native = [row["calls"] for row in rows if row["stage"] == 6]
        entry = [row["calls"] for row in rows if row["stage"] == 3]
        if any(len(set(counts)) != 1 for counts in (gpu, cpu, native, entry)):
            raise ValueError("public auto holdout placement changed between samples")
        applied = gpu[0] > 0
        sums = [
            sum(row["seconds"] for row in rows if row["sample"] == sample and row["stage"] in (1, 2, 3, 4, 5, 6))
            / AUTO_OBSERVED_OWNER_CALLS
            for sample in range(1, MEASURED_SAMPLES + 1)
        ]
        predicted = costs["owner_seconds"] + 4 * costs["descriptor_seconds"]
        predicted += entry[0] / AUTO_OBSERVED_OWNER_CALLS * costs["entry_seconds"]
        predicted += gpu[0] / AUTO_OBSERVED_OWNER_CALLS * costs["gpu_worker_seconds"]
        predicted += cpu[0] / AUTO_OBSERVED_OWNER_CALLS * costs["cpu_worker_seconds"]
        predicted += native[0] / AUTO_OBSERVED_OWNER_CALLS * costs["native_call_seconds"]
        # Explain the automatic outcome in a separate diagnostic invocation.
        # Trace overhead never enters stage/rate samples or fitted costs.
        diagnostic = build / f"n-{n}-decisions.log"
        diagnostic_command = [str(binary), str(n), "4"]
        run(diagnostic_command, build, diagnostic, timeout=180, env={**environment, "FORT_RUNTIME_TRACE": "1"})
        commands.append(diagnostic_command)
        decisions = [
            json.loads(line.split("FORT_SCOPED evidence ", 1)[1])
            for line in diagnostic.read_text().splitlines()
            if line.startswith("FORT_SCOPED evidence ")
        ]
        decisions = [row for row in decisions if row.get("event") == "decision"]
        if not decisions or any(
            row.get("available") != 1 or (row.get("gpu_units", 0) > 0) != applied for row in decisions
        ):
            raise ValueError("public auto holdout has no consistent available decision evidence")
        records.append(
            {
                "kind": "collective_auto_holdout",
                "fixture": "compute",
                "elements": n,
                "seconds": sums,
                "predicted_seconds": predicted,
                "gpu_applied": applied,
                "gpu_workers": gpu[0],
                "cpu_workers": cpu[0],
                "native_calls": native[0],
                "entries": entry[0],
                "work_calls_per_owner": AUTO_WORK_CALLS,
                "owner_calls_per_sample": AUTO_OBSERVED_OWNER_CALLS,
                "decision_evidence": decisions,
                "diagnostic_log": str(diagnostic),
                "diagnostic_log_sha256": sha256(diagnostic.read_bytes()).hexdigest(),
                "diagnostic_samples_used_for_costs": False,
                "comparison_scope": "exclusive production coordination; numerical and runtime API work excluded",
                "dependency": "repeated independent writes of the same field, then a native indirect update of a separate input field",
                "field_agreement": True,
                "all_native_model": "original numerical fallback; generated protocol comparison inapplicable"
                if not applied
                else None,
            }
        )
    return records


def normalize_fortran_options(raw: str) -> str:
    """Remove only build-location/output options; preserve ordered semantics."""
    result, skip = [], False
    for token in shlex.split(raw):
        if skip:
            skip = False
        elif token in {"-I", "-J", "-o"}:
            skip = True
        elif token.startswith(("-openmp", "-offload", "-opt")) or not any(
            token.startswith(flag) and len(token) > len(flag) for flag in ("-I", "-J", "-o")
        ):
            result.append(token)
    if skip:
        raise ValueError("Fortran location/output flag is missing its argument")
    return "\x1f".join(result)


def profile_with_collective_measurements(profile, records, *, calibration):
    """Reject missing/noisy stages instead of inventing collective estimates."""
    result = deepcopy(profile)
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("collective observations must be JSON records")
    identity = [row for row in records if row.get("kind") == "collective_identity"]
    if len(identity) != 1:
        raise ValueError("one collective identity observation is required")
    identity = identity[0]
    if (
        type(identity.get("threads")) is not int
        or identity["threads"] != profile["cpu_threads"]
        or type(identity.get("level")) is not int
        or identity["level"] != 1
        or type(identity.get("protocol_id")) is not int
        or identity["protocol_id"] != SCOPED_TEAM_PROTOCOL_ID
    ):
        raise ValueError("collective protocol, level or thread budget mismatch")
    version, options = identity.get("compiler_version"), identity.get("compiler_options")
    if not isinstance(version, str) or not version or not isinstance(options, str) or not options:
        raise ValueError("collective Fortran compiler identity is missing")
    costs = {}
    for row in records:
        if row.get("kind") == "collective_identity":
            continue
        if row.get("kind") not in {
            "collective_rate",
            "collective_cost",
            "collective_validation",
            "collective_auto_holdout",
        }:
            raise ValueError("unknown collective measurement")
        samples = row.get("seconds")
        if (
            not isinstance(samples, list)
            or len(samples) < 3
            or any(type(value) not in {int, float} or not math.isfinite(value) or value <= 0 for value in samples)
        ):
            raise ValueError("collective observations require three finite positive samples")
        if row["kind"] == "collective_auto_holdout":
            if type(row.get("gpu_applied")) is not bool or row.get("field_agreement") is not True:
                raise ValueError("invalid public auto holdout placement or field agreement")
            if type(row.get("gpu_workers")) is not int or row["gpu_workers"] < 0:
                raise ValueError("invalid public auto holdout worker count")
            if row["gpu_applied"] != (row["gpu_workers"] > 0):
                raise ValueError("invalid public auto holdout worker placement")
            if not row["gpu_applied"]:
                continue
        if row["kind"] in {"collective_validation", "collective_auto_holdout"}:
            predicted = row.get("predicted_seconds")
            if (
                type(predicted) not in {float, int}
                or not math.isfinite(predicted)
                or predicted <= 0
                or abs(statistics.median(samples) - predicted) / predicted > 0.25
            ):
                raise ValueError("collective protocol model validation differs by more than 25%")
            continue
        name = row.get("name")
        allowed = SCOPED_TEAM_RATE_NAMES if row["kind"] == "collective_rate" else SCOPED_TEAM_COST_NAMES
        if name not in allowed or name in costs:
            raise ValueError("unknown or duplicate collective cost")
        duration = statistics.median(samples)
        if row["kind"] == "collective_rate":
            baseline = row.get("baseline_seconds", [0] * len(samples))
            if (
                not isinstance(baseline, list)
                or len(baseline) != len(samples)
                or any(type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in baseline)
            ):
                raise ValueError("invalid collective rate baseline")
            duration -= statistics.median(baseline)
            work = row.get("work")
            if duration <= 0 or type(work) not in {int, float} or not math.isfinite(work) or work <= 0:
                raise ValueError("collective useful-work duration is nonpositive after protocol subtraction")
            costs[name] = work / duration
        else:
            costs[name] = duration
    if set(costs) != set(SCOPED_TEAM_RATE_NAMES + SCOPED_TEAM_COST_NAMES):
        raise ValueError("collective calibration does not cover every required cost")
    if not any(row.get("kind") == "collective_validation" for row in records):
        raise ValueError("collective calibration requires measured protocol model validation")
    result["scoped"]["collective"] = {
        "schema_version": 1,
        "protocol_id": SCOPED_TEAM_PROTOCOL_ID,
        "cpu_threads": profile["cpu_threads"],
        "expected_omp_level": 1,
        "costs": costs,
        "protocol_sources": collective_protocol_identity(),
        "measurements": records,
        "fortran": {
            "compiler_version": version,
            "compiler_options": options,
            "semantic_options": normalize_fortran_options(options),
            "excluded_options": ["-I", "-J", "-o"],
        },
        "calibration": {
            **calibration,
            "application_profiled": False,
            "protocol": "production public source-scope and run_team artifacts; one persistent team per batch",
            "costs": "coordinator-exclusive nested stages; CPU throughput includes whole-team completion",
            "thread_budget": "all configured threads participate; no hybrid pump is reserved",
        },
    }
    try:
        return validate_profile(result)
    except ProfileError as error:
        raise ValueError(str(error)) from error
