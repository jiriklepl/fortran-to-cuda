"""Unused child inputs preserve residency; reached native reads close once."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.numerical_contract import require_explicit_cuda_environment, require_numerical_build_contract
from compiler.tests.support.numerical_cuda import _transfers
from compiler.tests.test_module_deferred_captures import module_deferred_facts, write_module_deferred_sources
from compiler.tests.test_scoped_batch_sources_cuda import PROBE


def _driver(precision):
    return f"""program verify
use iso_fortran_env,only:error_unit
use saffron_owner,only:advance,owner_visits
use saffron_gate,only:gate_visits
use saffron_sweep,only:sweep_visits,middle_visits
implicit none
real({precision}),allocatable::a(:),b(:),c(:),extra(:)
integer,parameter::uppers(3)=[0,7,31]
integer::shape,repeat,n,extent,low,i,unit,calls,nonempty
logical::flag
character(16)::path
call get_command_argument(1,path)
if(path/='unused'.and.path/='zero-unused'.and.path/='used'.and.path/='empty') error stop 'unknown path'
flag=path=='used'
calls=0
nonempty=0
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(uppers)
if(path=='empty'.and.shape/=2) cycle
n=uppers(shape)
if(path=='empty') n=-4
extent=n+6
low=-10-shape
allocate(a(low:low+extent-1),b(low-3:low+extent-4),c(low-7:low+extent-8))
if(path=='zero-unused'.or.path=='empty') then
allocate(extra(low-11:low-12))
else
allocate(extra(low-11:low+extent-12))
endif
do repeat=1,2
do i=lbound(a,1),ubound(a,1)
a(i)=real(i-low+2*repeat,{precision})*0.125_{precision}
enddo
do i=lbound(extra,1),ubound(extra,1)
extra(i)=real(i-lbound(extra,1)+repeat,{precision})*0.25_{precision}
enddo
b=-77._{precision}
c=-91._{precision}
calls=calls+1
if(n>=-3) nonempty=nonempty+1
write(error_unit,'(a,i0)') 'BEGIN_CALL ',calls
call advance(a,b,c,extra,n,flag)
write(error_unit,'(a,i0)') 'END_CALL ',calls
if(owner_visits/=calls.or.gate_visits/=calls.or.sweep_visits/=calls) error stop 'original call replayed'
if(middle_visits/=nonempty) error stop 'native point update replayed'
if(lbound(a,1)/=low.or.ubound(a,1)/=low+extent-1) error stop 'owner bounds changed'
if(lbound(b,1)/=low-3.or.ubound(b,1)/=low+extent-4) error stop 'producer bounds changed'
if(lbound(c,1)/=low-7.or.ubound(c,1)/=low+extent-8) error stop 'child bounds changed'
if(b(lbound(b,1))/=-77._{precision}.or.b(ubound(b,1))/=-77._{precision}) error stop 'producer halo changed'
if(c(lbound(c,1))/=-91._{precision}.or.c(ubound(c,1))/=-91._{precision}) error stop 'child halo changed'
write(unit)shape,repeat,n,owner_visits,gate_visits,sweep_visits,middle_visits
! Compare native empty descriptors rather than assuming their requested lower bound.
write(unit)lbound(a),ubound(a),lbound(b),ubound(b),lbound(c),ubound(c),lbound(extra),ubound(extra)
write(unit)a,b,c,extra
enddo
deallocate(a,b,c,extra)
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module", params=[4, 8], ids=["real4", "real8"])
def deferred_module_binaries(tmp_path_factory, request):
    precision = request.param
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not all((fc, nvcc, host)):
        pytest.skip("native Fortran and CUDA toolchain required")
    require_explicit_cuda_environment()
    directory = tmp_path_factory.mktemp("deferred_module_" + str(precision))
    paths = write_module_deferred_sources(directory / "original", precision=precision)
    facts = module_deferred_facts(paths)
    assert "argument::extra" not in facts["captures"]
    assert set(facts["captures"]) == {"argument::a", "argument::b", "argument::c"}
    facts_path = directory / "facts.json"
    facts_path.write_text(json.dumps(facts, indent=2) + "\n")
    output, build, native_build = directory / "generated", directory / "build", directory / "native-build"
    build.mkdir()
    native_build.mkdir()
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    environment.pop("FORT_PHASE_TIMING", None)
    environment.pop("FORT_RUNTIME_TRACE", None)
    commands = []

    def run(arguments, *, cwd=build):
        ordinal = len(commands)
        commands.append({"argv": arguments, "cwd": str(cwd)})
        (directory / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result = subprocess.run(arguments, cwd=cwd, env=environment, capture_output=True,
                                text=True, timeout=180, check=False)
        (directory / f"command-{ordinal:02}.stdout").write_text(result.stdout)
        (directory / f"command-{ordinal:02}.stderr").write_text(result.stderr)
        (directory / f"command-{ordinal:02}.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    response = run([sys.executable, "-m", "compiler", "--form-scopes", "--input", str(paths[-1]),
        *(argument for path in paths[:-1] for argument in ("--source-file", str(path))),
        "--kernel", "saffron_owner::advance", "--scope-facts", str(facts_path), "--host-threads", "4",
        "--scope-execution", "reached", "--gpu-policy", "sections", "--memory-model", "scoped",
        "--json", "--output-dir", str(output)])
    public = json.loads(response.stdout)
    assert public["supported"], public
    manifest = json.loads((output / "scope-manifest.json").read_text())
    assert manifest == public["scopes"]
    owner, = manifest["scopes"]
    assert len(owner["gpu_leaves"]) == 4
    assert not owner["boundaries"], owner["boundaries"]
    companions = {item["procedure"]: item for item in owner["module_coordinators"]}
    assert "saffron_gate::branch_inputs" in companions
    assert companions["saffron_gate::branch_inputs"]["boundaries"]
    sweep, = manifest["borrowed_source_coordinators"]
    assert sweep["procedure"] == "saffron_sweep::diffuse"
    assert sweep["ownership"]["close_count"] == 0
    assert len(sweep["gpu_leaves"]) == 2
    point, = (operation for operation in sweep["native_operations"]
              if operation["resource_effects"] == {"argument::result": ["read", "write"]})
    section, = point["sections"]["resources"]
    assert section["reads"][0]["axes"][0]["kind"] == "point"
    assert section["reads"][0]["axes"][0]["lower"] == {"kind": "literal", "value": -3}
    resources = {item["resource"]: item for item in owner["resource_bindings"]["resources"]}
    assert "argument::extra" not in resources
    for path in paths:
        assert manifest["source_inputs"][str(path)] == sha256(path.read_bytes()).hexdigest()
    for name, expected in manifest["artifacts_sha256"].items():
        assert sha256((output / name).read_bytes()).hexdigest() == expected
    probe, probe_binary = directory / "probe.cu", directory / "probe"
    probe.write_text(PROBE)
    run([nvcc, "-std=c++17", "-ccbin", host, str(probe), "-o", str(probe_binary)])
    hardware = json.loads(run([str(probe_binary)]).stdout)
    (directory / "device.json").write_text(json.dumps(hardware, indent=2) + "\n")
    architecture = hardware["compute_capability"].replace(".", "")
    flags = [fc, "-O3", "-std=f2018", "-fopenmp", "-ffp-contract=off", "-fcheck=all,array-temps"]
    objects, compiled = [], set()
    original_order = {manifest["sources"][str(path)]["replacement"]: index for index, path in enumerate(paths)}
    for role in ("common_runtime", "shared_entry", "original_source"):
        descriptors = [item for item in manifest["build_sources"] if item["role"] == role]
        if role == "original_source":
            descriptors.sort(key=lambda item: original_order[item["path"]])
        for descriptor in descriptors:
            source, target = output / descriptor["path"], build / f"artifact-{len(objects):02}.o"
            compiler = [*flags, "-J", str(build), "-I", str(build)]
            if descriptor["language"] == "cuda":
                contract = descriptor["numerical_contract"]
                require_numerical_build_contract(contract)
                compiler = [nvcc, "-O3", "-std=c++17", "-ccbin", host, "-Xcompiler=-fopenmp",
                            "-arch=sm_" + architecture, *contract["required_cuda_options"],
                            *("-Xcompiler=" + option for option in contract["required_host_options"])]
            run([*compiler, "-c", str(source), "-o", str(target)])
            objects.append(str(target))
            compiled.add(descriptor["path"])
    assert compiled == {item["path"] for item in manifest["build_sources"]}
    driver = directory / "driver.f90"
    driver.write_text(_driver(precision))
    library = Path(nvcc).resolve().parents[1] / "lib64"
    candidate, native = build / "candidate", native_build / "native"
    run([*flags, "-J", str(build), "-I", str(build), str(driver), *objects,
         "-L" + str(library), "-Wl,-rpath," + str(library), "-lcudart", "-lstdc++", "-o", str(candidate)])
    run([*flags, "-J", str(native_build), "-I", str(native_build), *(str(path) for path in paths),
         str(driver), "-o", str(native)], cwd=native_build)
    return directory, candidate, native, precision, resources, environment


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True], ids=["cuda", "disabled"])
@pytest.mark.parametrize("path", ["unused", "zero-unused", "used", "empty"])
def test_deferred_child_inputs_preserve_fields_and_reached_closure(deferred_module_binaries, path, disabled):
    directory, candidate, native, precision, resources, environment = deferred_module_binaries
    label = path + ("-disabled" if disabled else "-cuda")
    environment = {**environment, "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    runs, fields = {}, {}
    for role, binary in (("native", native), ("candidate", candidate)):
        result = subprocess.run([str(binary), path], cwd=binary.parent, env=environment,
                                capture_output=True, text=True, timeout=120, check=False)
        (directory / f"{label}-{role}.stdout").write_text(result.stdout)
        (directory / f"{label}-{role}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "array temporary" not in result.stderr.lower()
        fields[role] = (binary.parent / "fields.bin").read_bytes()
        (directory / f"{label}-{role}.bin").write_bytes(fields[role])
        runs[role] = result
    assert fields["candidate"] == fields["native"]
    trace = runs["candidate"].stderr
    pattern = r"BEGIN_CALL (\d+)\n(.*?)END_CALL \1\n"
    calls = re.findall(pattern, trace, re.DOTALL)
    assert [int(ordinal) for ordinal, _trace in calls] == list(range(1, 3 if path == "empty" else 7))
    events = ("FORT_SCOPED launch ", "FORT_SCOPED upload ", "FORT_SCOPED download ")
    assert not any(line.startswith(events) for line in re.sub(pattern, "", trace, flags=re.DOTALL).splitlines())
    ids = {root: item["registration_identity"] for root, item in resources.items()}
    retained = not disabled and path in {"unused", "zero-unused"}
    closed_prefix = not disabled and path == "used"
    launches = 4 if retained else 1 if closed_prefix else 0
    for ordinal, call_trace in calls:
        assert call_trace.count("FORT_SCOPED launch ") == launches, call_trace
        gpu_events = [line for line in call_trace.splitlines() if line.startswith(events)]
        if not launches:
            assert not gpu_events
            continue
        contexts = [re.findall(r"\bcontext=(\S+)", line) for line in gpu_events]
        assert all(len(context) == 1 for context in contexts)
        assert len({context[0] for context in contexts}) == 1
        assert call_trace.count("FORT_SCOPED initialize ") == 1, "closed ownership must not reopen"
        cells = (4, 4, 11, 11, 35, 35)[int(ordinal) - 1]
        transfers = _transfers(call_trace)
        expected_up = {ids["argument::a"]: cells * precision, ids["argument::b"]: cells * precision}
        expected_down = {ids["argument::b"]: cells * precision}
        if retained:
            expected_up[ids["argument::c"]] = precision
            expected_down.update({ids["argument::a"]: cells * precision, ids["argument::c"]: cells * precision})
        assert dict(transfers["upload"]) == expected_up, call_trace
        assert dict(transfers["download"]) == expected_down, call_trace
    (directory / (label + "-receipt.json")).write_text(json.dumps({
        "schema_version": 1, "precision_bytes": precision, "path": path, "cuda_disabled": disabled,
        "complete_native_fields_halos_bounds_and_original_visits_equal": True,
        "missing_input_facts": ["argument::extra"], "extra_allocated_zero_extent": path in {"zero-unused", "empty"},
        "retained_producer_child_consumer_context": retained,
        "true_branch_publishes_non_idempotent_prefix_and_runs_native_suffix_once": closed_prefix,
        "no_gpu_events_outside_original_call_markers": True, "launches_per_call": launches,
        "native_sha256": sha256(fields["native"]).hexdigest(),
        "candidate_sha256": sha256(fields["candidate"]).hexdigest(),
        "trace_sha256": sha256(trace.encode()).hexdigest(), "performance_samples": 0,
    }, indent=2, sort_keys=True) + "\n")
