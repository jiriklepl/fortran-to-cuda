"""Real descriptor forwarding, including absent allocation and unknown data."""
import json
import os
import re
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.tests.test_readonly_allocatable_forwarding import SOURCE, generate

DRIVER = """program exercise
use forwarding
implicit none
integer,parameter::n=19
real(8),allocatable::a(:),q(:),expected(:),before_q(:)
integer::i,mode
character(1)::argument
call get_command_argument(1,argument)
read(argument,*)mode
allocate(a(-7:n),expected(-7:n))
do i=lbound(a,1),ubound(a,1)
 a(i)=real(13*i-7,8)/16.0_8
enddo
expected=a
if(mode==1)then
 allocate(q(-3:n+4),before_q(-3:n+4))
 do i=lbound(q,1),ubound(q,1)
  q(i)=real(3*i+41,8)/32.0_8
 enddo
 before_q=q
endif
do i=1,n
 if(mode==1)then
  expected(i-8)=expected(i-8)+q(i-4)
 else
  expected(i-8)=expected(i-8)+2.0_8
 endif
enddo
call advance(a,q,n)
if(any(a/=expected))error stop 'full original fields, tail and no replay'
if(mode==1)then
 if(.not.allocated(q))error stop 'input allocation status lost'
 if(lbound(q,1)/=-3.or.ubound(q,1)/=n+4)error stop 'input allocated bounds lost'
 if(any(q/=before_q))error stop 'readonly input payload/halos changed'
else
 if(allocated(q))error stop 'unallocated input was allocated or inspected'
endif
print *, 'READONLY_ALLOCATABLE_DESCRIPTOR_FIELDS_OK',mode
end program
"""


def checked(command, build, stem, env=None):
    (build / (stem + ".command.json")).write_text(json.dumps({"argv": command,
        "cwd": str(build), "CUDA_VISIBLE_DEVICES": (env or {}).get("CUDA_VISIBLE_DEVICES")}) + "\n")
    result = subprocess.run(command, cwd=build, env=env, capture_output=True, text=True, timeout=90)
    (build / (stem + ".stdout")).write_text(result.stdout)
    (build / (stem + ".stderr")).write_text(result.stderr)
    (build / (stem + ".completion.json")).write_text(json.dumps({"returncode": result.returncode}) + "\n")
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def q_read_preconditions(value):
    if isinstance(value, dict):
        if value.get("proof_role") == "exact_original_reached_read_preconditions" and any(
                row["resource"] == "argument::q" for row in value["sections"]["resources"]):
            yield value
        for child in value.values():
            yield from q_read_preconditions(child)
    elif isinstance(value, list):
        for child in value:
            yield from q_read_preconditions(child)


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    nvcc, host, fortran = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA/GNU toolchains required")
    directory = tmp_path_factory.mktemp("readonly_allocatable_forwarding")
    result = {}
    for variant in ("whole", "none", "conditional"):
        initialized = "whole" if variant == "whole" else "none"
        root = directory / variant
        root.mkdir()
        # WHOLE here is an explicit fixture fact, justified by the complete
        # original driver fill. INTENT(IN) never supplies this fact.
        original_source = SOURCE if variant != "conditional" else SOURCE.replace(
            "a(i)=a(i)+q(lbound(q,1)+i-1)", "if(q(lbound(q,1)+i-1)>a(i))then\na(i)=a(i)+2.0_8\nendif")
        outputs, manifest = generate(root, source=original_source, initialized=initialized)
        assert manifest["scope_count"] == 1
        # Both branches retain actual numerical implementations. The negative
        # case must decline on data authority, not on missing GPU lowering.
        assert len(manifest["scopes"][0]["gpu_leaves"]) == 2
        reads = {proof["proof_identity"]: proof for proof in q_read_preconditions(manifest)}
        if variant == "conditional":
            assert not reads
        else:
            assert len(reads) == 1
            proof = next(iter(reads.values()))
            assert proof["prefix_definitions_authorized"] is False
            assert proof["capture_authorized"] is False
            q = next(row for row in proof["sections"]["resources"] if row["resource"] == "argument::q")
            assert q["original_allocation_lower_bounds"] is True
            assert not q["writes"]
            assert not q["overwrites"]
        build = root / "build"
        build.mkdir()
        for name, text in outputs.items():
            path = build / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (build / "scope-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        objects = []
        for role in ("common_runtime", "shared_entry", "original_source"):
            for index, artifact in enumerate(manifest["build_sources"]):
                if artifact["role"] != role:
                    continue
                source, target = build / artifact["path"], build / f"object-{index}.o"
                assert sha256(source.read_bytes()).hexdigest() == manifest["artifacts_sha256"][artifact["path"]]
                options = ([nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp",
                            *artifact["numerical_contract"]["required_cuda_options"], "-I", str(build)]
                           if artifact["language"] == "cuda" else
                           [fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build), "-I", str(build)])
                checked([*options, "-c", str(source), "-o", str(target)], build, f"compile-{index}")
                objects.append(str(target))
        driver = build / "driver.f90"
        driver.write_text(DRIVER if variant != "conditional" else DRIVER.replace(
            "expected(i-8)=expected(i-8)+q(i-4)",
            "if(q(i-4)>expected(i-8))expected(i-8)=expected(i-8)+2.0_8"))
        native_source = build / "original.f90"
        native_source.write_text(original_source)
        candidate, native = build / "candidate", build / "native"
        checked([fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-I", str(build), str(driver), *objects,
                 "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++",
                 "-o", str(candidate)], build, "link-candidate")
        checked([fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build),
                 str(native_source), str(driver), "-o", str(native)], build, "link-native")
        slot = next(resource["registration_identity"] for resource in manifest["scopes"][0]["checked_execution_plan"]["resources"]
                    if resource["resource"] == "argument::q")
        result[variant] = build, candidate, native, slot
    return result


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True], ids=["enabled", "hidden"])
@pytest.mark.parametrize(("allocated", "variant"), [(False, "none"), (True, "whole"), (True, "none"), (True, "conditional")],
                         ids=["unallocated", "allocated-defined", "allocated-reached-reads", "allocated-conditional-unknown"])
def test_original_allocation_presence_and_definition_guards_remain_independent(binaries, hidden, allocated, variant):
    build, candidate, native, slot = binaries[variant]
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
           "FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        env["CUDA_VISIBLE_DEVICES"] = ""
    mode = ("hidden" if hidden else "enabled") + ("-allocated" if allocated else "-unallocated")
    reference = checked([str(native), str(int(allocated))], build, mode + "-native", env)
    result = checked([str(candidate), str(int(allocated))], build, mode + "-candidate", env)
    assert result.stdout == reference.stdout
    assert "READONLY_ALLOCATABLE_DESCRIPTOR_FIELDS_OK" in result.stdout
    assert "array temporary" not in result.stderr.lower()
    lines = result.stderr.splitlines()
    launches = [line for line in lines if line.startswith("FORT_SCOPED launch ")]
    uploads = [line for line in lines if line.startswith("FORT_SCOPED upload ")]
    downloads = [line for line in lines if line.startswith("FORT_SCOPED download ")]
    if hidden or variant == "conditional":
        assert not launches, result.stderr
        assert not uploads, result.stderr
        assert not downloads, result.stderr
    else:
        assert len(launches) == 1, result.stderr
        assert sum(int(re.search(r"\bbytes=(\d+)", line)[1]) for line in uploads) == (2 if allocated else 1)*19*8
        assert sum(int(re.search(r"\bbytes=(\d+)", line)[1]) for line in downloads) == 19*8
    if not allocated:
        assert not any(re.search(rf"\bbuffer={slot}\b", line) for line in lines
                       if line.startswith(("FORT_SCOPED allocate ", "FORT_SCOPED upload ", "FORT_SCOPED download ")))
    elif variant == "none" and not hidden:
        # NONE records unknown coverage. The existing, separate reached-read
        # token can establish only the exact original unconditional read set;
        # neither INTENT nor allocation forwarding supplies this authority.
        assert any(re.search(rf"\bbuffer={slot}\b", line) for line in lines
                   if line.startswith("FORT_SCOPED admit_source_reads "))
