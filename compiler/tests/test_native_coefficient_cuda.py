"""Real CUDA prefix/native coefficient team/future GPU suffix coherence.

Hardware identities are real; policy costs are deliberately synthetic. These
checks establish correctness and authority, never a measured speedup.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.numerical_contract import require_numerical_build_contract
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_native_choice_cuda import policy_profile
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run


def straight_source(lower):
    parts = []
    bodies = [
        "v=a(i,j,k)\ndo q=1,32\nv=sqrt(v*v)+1.0_8\nenddo\nb(i,j,k)=v",
        "c(i,j,k)=b(i,j,k)*coefficient(stage)*dt\ncounter(i,j,k)=counter(i,j,k)+1",
        "v=a(i,j,k)+b(i,j,k)+c(i,j,k)\ndo q=1,32\nv=sqrt(v*v)+1.0_8\nenddo\nd(i,j,k)=v",
    ]
    for body in bodies:
        parts.append(
            "!$omp parallel do private(i,j,q,v) default(none) shared(a,b,c,d,counter,coefficient,n,stage,dt)\n"
            "do k=1,n\ndo j=1,n\ndo i=1,n\n" + body + "\nenddo\nenddo\nenddo\n!$omp end parallel do\n"
        )
    return (
        """module coefficient_pipeline
implicit none
contains
subroutine advance(a,b,c,d,counter,coefficient,n,stage,dt)
real(8),intent(in)::a(-2:,-2:,-2:),dt
real(8),intent(inout)::b(-2:,-2:,-2:),c(-2:,-2:,-2:),d(-2:,-2:,-2:)
integer,intent(inout)::counter(-2:,-2:,-2:)
integer,intent(in)::n,stage
real(8),intent(in)::coefficient("""
        + f"{lower}:{lower + 2})\ninteger::i,j,k,q\nreal(8)::v\n"
        + "".join(parts)
        + "end subroutine\nend module\n"
    )


def straight_driver(lower):
    return (
        """program verify
use coefficient_pipeline
implicit none
real(8),allocatable::a(:,:,:),b(:,:,:),c(:,:,:),d(:,:,:)
integer,allocatable::counter(:,:,:)
real(8)::coefficient("""
        + f"{lower}:{lower + 2})\n"
        + """integer,parameter::sizes(3)=[0,1,9]
integer::shape,n,repeat,i,j,k,unit,stage
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
coefficient=[2.0_8,4.0_8,8.0_8]
do shape=1,size(sizes)
n=sizes(shape)
allocate(a(-2:n+3,-2:n+3,-2:n+3),b(-2:n+3,-2:n+3,-2:n+3))
allocate(c(-2:n+3,-2:n+3,-2:n+3),d(-2:n+3,-2:n+3,-2:n+3),counter(-2:n+3,-2:n+3,-2:n+3))
do repeat=1,2
do k=lbound(a,3),ubound(a,3)
do j=lbound(a,2),ubound(a,2)
do i=lbound(a,1),ubound(a,1)
a(i,j,k)=real(i+j+k+9,8)*0.25_8+repeat
enddo
enddo
enddo
b=-99.0_8
c=-77.0_8
d=-55.0_8
counter=0
stage="""
        + f"{lower}+repeat-1\n"
        + """call advance(a,b,c,d,counter,coefficient,n,stage,0.125_8)
write(unit)n,repeat,lbound(a),ubound(a),lbound(counter),ubound(counter)
write(unit)a,b,c,d,counter,coefficient
enddo
deallocate(a,b,c,d,counter)
enddo
close(unit)
end program
"""
    )


def source(lower, *, tiled=False):
    text = straight_source(lower)
    if not tiled:
        return text
    return (
        text.replace("coefficient,n,stage,dt)", "coefficient,n,stage,dt,tile)")
        .replace("integer,intent(in)::n,stage", "integer,intent(in)::n,stage,tile")
        .replace("integer::i,j,k,q", "integer::i,j,k,q,bi,bj,bk")
        .replace("private(i,j,q,v)", "private(i,j,k,q,v,bi,bj)")
        .replace(
            "do k=1,n\ndo j=1,n\ndo i=1,n",
            "do bk=1,n,tile\ndo bj=1,n,tile\ndo bi=1,n,tile\ndo k=bk,min(bk+tile-1,n)\ndo j=bj,min(bj+tile-1,n)\ndo i=bi,min(bi+tile-1,n)",
        )
        .replace(
            "enddo\nenddo\nenddo\n!$omp end parallel do",
            "enddo\nenddo\nenddo\nenddo\nenddo\nenddo\n!$omp end parallel do",
        )
    )


def driver(lower, *, tiled=False):
    text = straight_driver(lower)
    if not tiled:
        return text
    return text.replace(
        "integer::shape,n,repeat,i,j,k,unit,stage", "integer::shape,n,repeat,i,j,k,unit,stage,tile"
    ).replace(
        "call advance(a,b,c,d,counter,coefficient,n,stage,0.125_8)",
        "tile=2\nif(repeat==2)tile=-1\ncall advance(a,b,c,d,counter,coefficient,n,stage,0.125_8,tile)",
    )


@pytest.fixture(
    scope="module",
    params=[(1, False), (-1, False), (1, True), (-1, True)],
    ids=["one_lower", "negative_lower", "tiled_one_lower", "tiled_negative_lower"],
)
def binaries(request, tmp_path_factory):
    lower, tiled = request.param
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA and Fortran toolchains required")
    root = tmp_path_factory.mktemp("native_coefficient_cuda")
    observed = actual_profile(root, nvcc, host)
    arch = observed["hardware"]["compute_capability"].replace(".", "")
    flags = [fortran, "-O2", "-cpp", "-std=f2018", "-fopenmp", "-ffp-contract=off", "-fcheck=all,array-temps"]
    cases = {}
    runtime_objects = {}
    for policy in ("cpu", "mixed"):
        case = root / policy
        case.mkdir()
        original = case / "original.f90"
        original.write_text(source(lower, tiled=tiled))
        profile = policy_profile(observed, policy)
        builder = ScopeBuilder(
            [original],
            "coefficient_pipeline::advance",
            facts={
                "schema_version": 1,
                "participation": "serial",
                "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
                "captures": {
                    "argument::" + name: dict(FACT) for name in ("a", "b", "c", "d", "counter", "coefficient")
                },
            },
            options=CompilerOptions(opt_level=0),
            config=OffloadConfig("auto", profile),
        )
        outputs, manifest = builder.run()
        (owner,) = manifest["scopes"]
        teams = [p["operations"]["combined_numerical_team"] for p in owner["planning_segments"]]
        assert len(teams) == 3
        assert all(t["original_cpu_alternative"]["available"] for t in teams)
        assert any(q["workers"] == 3 for q in owner["query_regions"])
        if tiled:
            assert all(
                t["original_cpu_alternative"]["physical_section_authority"]
                == "registered prepared single numerical worker; no aggregation"
                for t in teams
            )
        for name, text in outputs.items():
            path = case / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        objects = []
        for role in ("common_runtime", "shared_entry", "original_source"):
            for item in manifest["build_sources"]:
                if item["role"] != role:
                    continue
                artifact = case / item["path"]
                target = case / (item["path"].replace("/", "_") + ".o")
                key = item["path"], manifest["artifacts_sha256"][item["path"]]
                if role == "common_runtime" and key in runtime_objects and item["language"] == "cuda":
                    objects.append(runtime_objects[key])
                    continue
                if item["language"] == "cuda":
                    contract = item["numerical_contract"]
                    require_numerical_build_contract(contract)
                    compiler = [
                        nvcc,
                        "-O2",
                        "-std=c++17",
                        "-ccbin",
                        host,
                        "-arch=sm_" + arch,
                        "-Xcompiler=-fopenmp",
                        *contract["required_cuda_options"],
                        *("-Xcompiler=" + o for o in contract["required_host_options"]),
                    ]
                else:
                    compiler = flags
                run([*compiler, "-I", str(case), "-c", str(artifact), "-o", str(target)], cwd=case)
                objects.append(str(target))
                if role == "common_runtime":
                    runtime_objects[key] = str(target)
        caller = case / "driver.f90"
        caller.write_text(driver(lower, tiled=tiled))
        native, candidate = case / "native", case / "candidate"
        run([*flags, str(original), str(caller), "-o", str(native)], cwd=case)
        run(
            [
                *flags,
                str(caller),
                *objects,
                "-L/usr/local/cuda/lib64",
                "-Wl,-rpath,/usr/local/cuda/lib64",
                "-lcudart",
                "-lstdc++",
                "-o",
                str(candidate),
            ],
            cwd=case,
        )
        (case / "manifest.json").write_text(json.dumps(manifest, indent=2))
        cases[policy] = case, native, candidate, manifest
    return cases


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_coefficient_cpu_team_preserves_future_gpu_choices_and_read_mirrors(binaries, disabled):
    for policy, (case, native, candidate, manifest) in binaries.items():
        tiled = (
            "registered prepared single numerical worker"
            in manifest["scopes"][0]["planning_segments"][0]["operations"]["combined_numerical_team"][
                "original_cpu_alternative"
            ]["physical_section_authority"]
        )
        env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
        if disabled:
            env["CUDA_VISIBLE_DEVICES"] = ""
        run([str(native)], cwd=case, env=env)
        expected = (case / "fields.bin").read_bytes()
        result = run([str(candidate)], cwd=case, env=env)
        (case / ("disabled.stderr" if disabled else "candidate.stderr")).write_text(result.stderr)
        assert (case / "fields.bin").read_bytes() == expected
        assert "array temporary" not in result.stderr.lower()
        events = [
            json.loads(line.removeprefix("FORT_SCOPED evidence "))
            for line in result.stderr.splitlines()
            if line.startswith("FORT_SCOPED evidence ")
        ]
        decisions = [e for e in events if e["event"] == "decision"]
        assert decisions
        if tiled:
            native_handoffs = [
                line for line in result.stderr.splitlines() if line.startswith("FORT_SCOPED plan_native_begin ")
            ]
            assert len(native_handoffs) == (6 if disabled or policy == "cpu" else 4)
        if disabled or policy == "cpu":
            assert all(e["gpu_units"] == 0 for e in decisions)
            for action in ("initialize", "upload", "download", "allocate", "launch", "graph_submit"):
                assert "FORT_SCOPED " + action + " " not in result.stderr
        else:
            assert any(e["gpu_units"] == 2 and e["cpu_units"] == 1 for e in decisions)
            assert "FORT_SCOPED plan_native_begin " in result.stderr
            ids = {
                r["resource"]: r["registration_identity"]
                for r in manifest["scopes"][0]["resource_bindings"]["resources"]
            }
            uploads = [line for line in result.stderr.splitlines() if line.startswith("FORT_SCOPED upload ")]
            assert any(re.search(r"\bbuffer=" + str(ids["argument::a"]) + r"\b", line) for line in uploads)
            assert not any(re.search(r"\bbuffer=" + str(ids["argument::b"]) + r"\b", line) for line in uploads)
            assert not any(
                re.search(r"\bbuffer=" + str(ids["argument::coefficient"]) + r"\b", line) for line in uploads
            )
