"""A calibrated reached child builds independently and preserves native fields."""

import json
import os
import shlex
import shutil
from hashlib import sha256

import pytest

from compiler.numerical_contract import require_numerical_build_contract
from compiler.offload.collective_calibration import normalize_fortran_options
from compiler.offload.numerical_calibration import profile_from_compute_measurements
from compiler.tests.test_borrowed_source_coordinators import source
from compiler.tests.test_numerical_calibration_v2 import observations_v2
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import FACT, ROOT, generate, run

DRIVER = """program verify
use original,only:step
implicit none
real(8),allocatable::a(:),b(:),out(:)
integer,parameter::sizes(3)=[0,8,262144]
integer::shape,repeat,sign,n,i,unit
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
 allocate(a(-7:sizes(shape)-8),b(-7:sizes(shape)-8),out(-7:sizes(shape)-8))
 do repeat=1,2
  do sign=-1,1,2
   n=size(a)
   do i=lbound(a,1),ubound(a,1)
    a(i)=real(sign*(i+8),8)*0.25d0+repeat
   enddo
   b=-99.d0
   out=-101.d0
   call step(a,b,out,n)
   write(unit)n,a,b,out
  enddo
 enddo
 deallocate(a,b,out)
enddo
close(unit)
end program
"""


def synthetic_native_profile(directory, nvcc, host, flags):
    """Exercise AUTO with valid identities, never measured performance costs."""
    profile = actual_profile(directory, nvcc, host)
    probe = directory / "fortran-identity.f90"
    probe.write_text("""program identity
use iso_fortran_env,only:compiler_version,compiler_options
print '(a)',compiler_version()
print '(a)',compiler_options()
end program
""")
    binary = directory / "fortran-identity"
    object_file = directory / "fortran-identity.o"
    # The original module is compiled with -c. A compile-and-link probe can
    # report additional driver-injected link options that its object lacks.
    commands = []
    for role, command in (
        ("identity-compile", [*flags, "-c", str(probe), "-o", str(object_file)]),
        ("identity-link", [*flags, str(object_file), "-o", str(binary)]),
        ("identity-run", [str(binary)]),
    ):
        result = run(command, cwd=directory)
        commands.append({"role": role, "argv": command, "returncode": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
    version, options = commands[-1]["stdout"].strip().splitlines()
    observations = observations_v2()
    identity = observations[0]
    identity.update({name: profile["hardware"][name]
                     for name in ("gpu_name", "gpu_uuid", "compute_capability")})
    identity.update({name: profile["toolchain"][name]
                     for name in ("cuda_runtime_version", "driver_version")})
    identity["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    if len(identity["cpu_affinity"]) != profile["cpu_threads"]:
        pytest.skip("AUTO CUDA correctness requires the calibrated four-CPU placement")
    identity["fortran"] = {"compiler_version": version, "compiler_options": options,
                            "semantic_options": normalize_fortran_options(options)}
    (directory / "fortran-identity-evidence.json").write_text(json.dumps({
        "schema_version": 1, "compile_mode": "separate_object", "commands": commands,
        "fortran_flags": flags,
        "normalized_compile_flags": normalize_fortran_options(shlex.join(flags[1:])),
        "fortran": identity["fortran"],
        "artifacts_sha256": {path.name: sha256(path.read_bytes()).hexdigest()
                             for path in (probe, object_file, binary)},
        "synthetic_calibration_for_correctness_only": True,
    }, indent=2, sort_keys=True) + "\n")
    # Scale complete CPU observations, including holdouts, to favor offload.
    # The synthetic GPU observations and applicability ranges remain intact.
    for row in observations[1:]:
        if row["backend"] != "gpu":
            for sample in row["samples"]:
                sample["elapsed_seconds"] *= 1e8
                sample["wall_seconds"] *= 1e8
    return profile_from_compute_measurements(profile, observations,
        calibration={"synthetic_correctness_only": True})


@pytest.fixture(scope="module", params=("bounded", "reached", "reached_rollback"))
def calibrated_child_binary(tmp_path_factory, request):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA/C++/Fortran toolchain unavailable")
    directory = tmp_path_factory.mktemp("calibrated_borrowed_child")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    profile = directory / "profile.json"
    # Favorable synthetic correctness costs, bound to the actual hardware and
    # tool identities, exercise AUTO; these are never performance evidence.
    flags = [fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    calibration = synthetic_native_profile(directory, nvcc, host, flags)
    identity_evidence_path = directory / "fortran-identity-evidence.json"
    identity_evidence = json.loads(identity_evidence_path.read_text())
    assert calibration["numerical"]["identity"]["fortran"] == identity_evidence["fortran"]
    profile.write_text(json.dumps(calibration))
    text = source()
    if request.param == 'reached_rollback':
        text = text.replace('call child(a,b,out,n)', 'if(n<0) then\ncall child(a,b,b,n)\nendif\ncall child(a,b,out,n)')
    execution = 'bounded' if request.param == 'bounded' else 'reached'
    original, output, manifest = generate(directory / "case", text, mode="auto", profile=profile,
        checkout=checkout, scope_execution=execution, facts={"schema_version": 1, "participation": "serial",
                                 "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}})
    assert manifest["scopes"][0]["estimate_available"]
    if execution == "bounded":
        assert {item["procedure"] for item in manifest["borrowed_source_coordinators"]} == {"original::child"}
    else:
        # The reached graph keeps child as the original ModuleOwner body;
        # transform is its separately checked borrowed numerical coordinator.
        transform, = manifest["borrowed_source_coordinators"]
        assert transform["procedure"] == "original::transform"
        assert transform["role"] == "borrowed_reached_coordinator"
        scope, = manifest["scopes"]
        plan = scope["checked_execution_plan"]
        assert plan["entry"] == "original::step"
        leaves = {"original::producer#region1", "original::consumer#region1", "original::transform#region1"}
        assert set(plan["gpu_leaves"]) == set(scope["gpu_leaves"]) == leaves
        assert plan["numerical_invocations"] == 3
        assert leaves <= {item["procedure"] for item in manifest["numerical_decisions"] if item["supported"]}
        members = {item["procedure"]: item for item in plan["procedures"]}
        assert len(plan["procedures"]) == len(members) == 4
        assert set(members) == {"original::step", "original::child", "original::producer", "original::consumer"}
        modules = {item["procedure"]: item for item in scope["module_coordinators"]}
        assert len(scope["module_coordinators"]) == len(modules) == 3
        assert set(modules) == {"original::child", "original::producer", "original::consumer"}
        expected_calls = {
            "original::child": ("original::step", "call child(a,b,out,n)", ("a", "b", "out")),
            "original::producer": ("original::child", "call producer(a,b,n)", ("a", "b")),
            "original::consumer": ("original::child", "call consumer(a,b,out,n)", ("a", "b", "out")),
        }
        for name, (caller, call_text, resources) in expected_calls.items():
            module, member = modules[name], members[name]
            assert module["interface"] == "original_module_entry_v1"
            assert module["structured_summary_identity"] == member["structured_summary_identity"]
            assert member["source_sha256"] == sha256(original.read_bytes()).hexdigest()
            assert module["resource_mappings"] == {"argument::" + key: "argument::" + key for key in resources}
            assert module["calls"] == member["calls"]
            call, = member["calls"]
            assert call["caller"] == caller
            assert call["first_line"] == call["last_line"] == text.splitlines().index(call_text) + 1
        point, delegated = members["original::child"]["units"]
        native, = point["operations"]["native_operations"]
        assert point["selected_node_ids"] == native["selected_original_nodes"]
        assert point["first_line"] == point["last_line"] == text.splitlines().index("b(1)=b(1)+5.d0") + 1
        assert native["resource_effects"] == {"argument::b": ["read", "write"]}
        assert native["completion"]["available"]
        assert not native["completion"]["has_opaque_calls_in_closure"]
        assert not native["completion"]["has_openmp_in_closure"]
        assert native["sections"]["available"]
        section, = native["sections"]["resources"]
        assert (section["resource"], section["rank"], section["logical_lower_bounds"]) == ("argument::b", 1, [1])
        for effect in ("reads", "writes", "overwrites"):
            access, = section[effect]
            assert access["source_access"] == "b(1)"
            axis, = access["axes"]
            assert axis == {"kind": "point", "lower": {"kind": "literal", "value": 1},
                            "upper": {"kind": "literal", "value": 1}}
        node, = delegated["operations"]["structured_tree"]["nodes"]
        assert node["kind"] == "reached_source_coordinator"
        assert node["procedure"] == transform["procedure"]
        assert node["structured_summary_identity"] == transform["structured_summary_identity"]
        assert node["requirements"]["unique_canonical_resource_mappings"]
        for name in ("original::producer", "original::consumer"):
            unit, = members[name]["units"]
            segment, = unit["operations"]["planning_segments"]
            assert segment["query_available"]
            assert segment["estimate_available"]
            assert segment["gpu_leaves"] == [name + "#region1"]
        segment, = transform["planning_segments"]
        assert segment["query_available"]
        assert segment["estimate_available"]
        assert segment["gpu_leaves"] == ["original::transform#region1"]
        if request.param == "reached_rollback":
            boundary, = scope["boundaries"]
            assert "alias" in boundary["reason"]
            assert not boundary["reopen"]
            assert boundary["execution"] == "publish and close when reached; original continuation once"
            assert boundary["first_line"] == boundary["last_line"] == text.splitlines().index("call child(a,b,b,n)") + 1
        else:
            assert not scope["boundaries"]
    objects, commands = [], []
    headers = tuple((name, digest) for name, digest in sorted(manifest["artifacts_sha256"].items())
                    if name.endswith((".h", ".hpp", ".cuh")))
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] != role:
                continue
            artifact = output / item["path"]
            assert sha256(artifact.read_bytes()).hexdigest() == manifest["artifacts_sha256"][item["path"]]
            target = artifact.with_suffix(".o")
            if item["language"] == "cuda":
                contract = item["numerical_contract"]
                require_numerical_build_contract(contract)
                compiler = [nvcc, "-O2", "-std=c++17", "-ccbin", host,
                            "-arch=sm_" + calibration['hardware']['compute_capability'].replace('.', ''),
                            "-Xcompiler=-fopenmp", *contract["required_cuda_options"],
                            *("-Xcompiler=" + option for option in contract["required_host_options"])]
            else:
                compiler = flags
                if role == "original_source":
                    assert normalize_fortran_options(shlex.join(compiler[1:])) == (
                        identity_evidence["normalized_compile_flags"])
            command = [*compiler, "-I", str(output), "-c", str(artifact), "-o", str(target)]
            run(command, cwd=output)
            commands.append({"path": item["path"], "role": role, "argv": command,
                             "numerical_contract": item.get("numerical_contract")})
            objects.append(str(target))
    driver = output / "driver.f90"
    driver.write_text(DRIVER)
    target, native = output / "candidate", output / "native"
    cuda_lib = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(nvcc))), "lib64")
    command = [*flags, str(driver), *objects, "-L" + cuda_lib, "-Wl,-rpath," + cuda_lib,
               "-lcudart", "-lstdc++", "-o", str(target)]
    run(command, cwd=output)
    commands.append({"role": "candidate-link", "argv": command})
    command = [*flags, str(original), str(driver), "-o", str(native)]
    run(command, cwd=output)
    commands.append({"role": "native-compile-link", "argv": command})
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}
    run([str(native)], cwd=output, env=environment)
    expected = (output / "fields.bin").read_bytes()
    (output / "native-fields.bin").write_bytes(expected)
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "artifact-evidence.json").write_text(json.dumps({"headers": headers,
        "runtime_id": manifest["runtime"]["runtime_id"], "profile": str(profile),
        "fortran_identity_evidence": {"path": str(identity_evidence_path),
            "sha256": sha256(identity_evidence_path.read_bytes()).hexdigest()}}, indent=2))
    return target, output, expected


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_calibrated_auto_child_preserves_reached_native_fallback(calibrated_child_binary, disabled):
    target, output, expected = calibrated_child_binary
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = run([str(target)], cwd=output, env=environment)
    actual = (output / "fields.bin").read_bytes()
    assert actual == expected
    assert "array temporary" not in result.stderr.lower()
    if disabled:
        assert "FORT_SCOPED launch " not in result.stderr
    else:
        assert "FORT_SCOPED launch " in result.stderr
    label = "disabled" if disabled else "enabled"
    (output / (label + ".stderr")).write_text(result.stderr)
    (output / (label + "-receipt.json")).write_text(json.dumps({
        "schema_version": 1, "disabled": disabled,
        "native_fields_sha256": sha256(expected).hexdigest(),
        "candidate_fields_sha256": sha256(actual).hexdigest(),
        "runtime_trace_sha256": sha256(result.stderr.encode()).hexdigest(),
        "launches": result.stderr.count("FORT_SCOPED launch "),
        "complete_arrays_and_halos_equal": True,
        "independent_compiler_checkout": True,
        "synthetic_calibration_for_correctness_only": True,
        "performance_sample": False,
    }, indent=2, sort_keys=True) + "\n")
