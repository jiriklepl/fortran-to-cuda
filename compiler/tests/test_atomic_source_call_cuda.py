"""Distinct guarded SECTION calls use each original field and scalar argument."""

from __future__ import annotations

import json
import os
import re
import shutil
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_atomic_source_call_bindings import SOURCE as BASE
from compiler.tests.test_source_scopes import FACT, run

PREFIX = BASE.split("subroutine step", 1)[0]
SOURCE = (
    PREFIX
    + """subroutine step(a,b,p,q,na,nb,fa,fb,active)
real(8),intent(inout)::a(-2:),b(-2:)
real(8),intent(in)::p(-2:),q(-2:)
integer,intent(in)::na,nb
logical,intent(in)::fa,fb,active
integer::i
if(active)then
do i=-2,na+3
a(i)=2*p(i)+real(i,8)
enddo
do i=-2,nb+3
b(i)=3*q(i)-real(i,8)
enddo
call relay(a,p,na,fa)
call relay(b,q,nb,fb)
call relay(a,p,na,fa)
do i=-2,na+3
a(i)=a(i)+p(i)
enddo
do i=-2,nb+3
b(i)=b(i)+q(i)
enddo
endif
end subroutine
end module
"""
)
DRIVER = """program exercise
use atomic_calls
use iso_fortran_env,only:error_unit
implicit none
real(8),allocatable::a(:),b(:),p(:),q(:),ea(:),eb(:),ep(:),eq(:)
integer::which,na,nb,branch,i,pass
logical::fa,fb,active
do which=1,3
select case(which)
case(1)
na=3
nb=7
case(2)
na=7
nb=67
case default
na=67
nb=3
end select
allocate(a(-2:na+3),b(-2:nb+3),p(-2:na+3),q(-2:nb+3))
allocate(ea(-2:na+3),eb(-2:nb+3),ep(-2:na+3),eq(-2:nb+3))
do branch=1,4
fa=branch==1.or.branch==2
fb=branch==1.or.branch==3
active=branch/=4
do i=-2,na+3
p(i)=real(11*i*i-5*i+which,8)/64._8
a(i)=-991._8
ea(i)=a(i)
enddo
do i=-2,nb+3
q(i)=real(7*i*i+3*i+which,8)/32._8
b(i)=-993._8
eb(i)=b(i)
enddo
ep=p
eq=q
if(active)then
do i=-2,na+3
ea(i)=2*p(i)+real(i,8)
enddo
do i=-2,nb+3
eb(i)=3*q(i)-real(i,8)
enddo
if(fa)then
do pass=1,2
do i=na+1,na+3
ea(i)=p(i-na)
enddo
do i=-2,0
ea(i)=p(i+na)
enddo
enddo
endif
if(fb)then
do i=nb+1,nb+3
eb(i)=q(i-nb)
enddo
do i=-2,0
eb(i)=q(i+nb)
enddo
endif
ea=ea+p
eb=eb+q
endif
write(error_unit,'(A,3(1X,I0))')'ATOMIC_CALL_BEGIN',na,nb,branch
call step(a,b,p,q,na,nb,fa,fb,active)
write(error_unit,'(A,3(1X,I0))')'ATOMIC_CALL_END',na,nb,branch
if(any(a/=ea).or.any(b/=eb))error stop 'per-call fields or halo values changed'
if(any(p/=ep).or.any(q/=eq))error stop 'readonly source changed'
write(*,'(A,3(1X,I0))')'CASE',na,nb,branch
write(*,'(*(ES25.17E3,1X))')a,b,p,q
enddo
deallocate(a,b,p,q,ea,eb,ep,eq)
enddo
print *, 'ATOMIC_CALL_FIELDS_HALOS_READONLY_OK'
end program
"""


def emit(tmp_path):
    path = tmp_path / "original.f90"
    path.write_text(SOURCE)
    builder = ScopeBuilder(
        [path],
        "atomic_calls::step",
        options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy="sections", scope_execution="graph", host_threads=4),
        facts={
            "schema_version": 1,
            "participation": "serial",
            "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
            "captures": {"argument::" + n: dict(FACT) for n in ("a", "b", "p", "q", "na", "nb", "fa", "fb", "active")},
        },
    )
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report["graph_owner_attempt"]
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    validate_checked_execution_plans(json.loads(outputs["scope-manifest.json"]), capture_facts=builder.facts)
    return outputs, report, path


def test_repeated_section_call_chain_seals_original_scalars_and_distinct_fields(tmp_path):
    outputs, report, _path = emit(tmp_path)
    (task,) = report["scopes"][0]["source_device_tasks"]
    projection = task["execution_binding"]["call_projection"]
    assert task["version"] == 4
    assert len(projection["calls"]) == 4
    assert projection["field_roots"] == {
        "argument::a": ["argument::a", "argument::b"],
        "argument::p": ["argument::p", "argument::q"],
    }
    calls = [c for c in projection["calls"] if c["caller"] == "atomic_calls::step"]
    assert [[a["actual"].lower() for a in c["source_call"]["original_arguments"]] for c in calls] == [
        ["a", "p", "na", "fa"],
        ["b", "q", "nb", "fb"],
        ["a", "p", "na", "fa"],
    ]
    (tmp_path / "scope-manifest.json").write_text(outputs["scope-manifest.json"])


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    build = tmp_path_factory.mktemp("atomic-call-cuda")
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fc)):
        pytest.skip("CUDA and Fortran toolchains required")
    outputs, report, original = emit(build)
    for name, text in outputs.items():
        target = build / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(report["build_sources"]):
            if artifact["role"] != role:
                continue
            path, target = build / artifact["path"], build / f"object-{index}.o"
            assert sha256(path.read_bytes()).hexdigest() == report["artifacts_sha256"][artifact["path"]]
            if artifact["language"] == "cuda":
                command = [
                    nvcc,
                    "-std=c++17",
                    "-ccbin",
                    host,
                    "-arch=sm_86",
                    "-Xcompiler=-fopenmp",
                    *artifact["numerical_contract"]["required_cuda_options"],
                    "-I",
                    str(build),
                ]
            else:
                command = [fc, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build), "-I", str(build)]
            process = run([*command, "-c", str(path), "-o", str(target)], cwd=build)
            (build / f"compile-{index}.stdout").write_text(process.stdout)
            (build / f"compile-{index}.stderr").write_text(process.stderr)
            objects.append(str(target))
    driver = build / "driver.f90"
    driver.write_text(DRIVER)
    candidate, native = build / "candidate", build / "native"
    run(
        [
            fc,
            "-fopenmp",
            "-fcheck=all",
            "-ffp-contract=off",
            "-I",
            str(build),
            str(driver),
            *objects,
            "-L/usr/local/cuda/lib64",
            "-Wl,-rpath,/usr/local/cuda/lib64",
            "-lcudart",
            "-lstdc++",
            "-o",
            str(candidate),
        ],
        cwd=build,
    )
    run(
        [
            fc,
            "-fopenmp",
            "-fcheck=all",
            "-ffp-contract=off",
            "-J",
            str(build),
            str(original),
            str(driver),
            "-o",
            str(native),
        ],
        cwd=build,
    )
    return build, candidate, native


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["traced", "untraced", "hidden"])
def test_section_calls_use_each_reached_allocation_and_scalar(compiled, mode):
    build, candidate, native = compiled
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "FALSE"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.pop("FORT_RUNTIME_TRACE", None)
    if mode != "untraced":
        environment["FORT_RUNTIME_TRACE"] = "1"
    if mode == "hidden":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    reference = run([str(native)], cwd=build, env=environment)
    result = run([str(candidate)], cwd=build, env=environment)
    for role, process in (("native", reference), ("candidate", result)):
        (build / f"{mode}-{role}.stdout").write_text(process.stdout)
        (build / f"{mode}-{role}.stderr").write_text(process.stderr)
    assert result.stdout == reference.stdout
    assert "ATOMIC_CALL_FIELDS_HALOS_READONLY_OK" in result.stdout
    if mode == "untraced":
        return
    manifest = json.loads((build / "scope-manifest.json").read_text())
    slots = {
        row["resource"]: str(row["registration_identity"])
        for row in manifest["scopes"][0]["checked_execution_plan"]["storage_epoch"]["resources"]
    }
    cases = re.findall(r"ATOMIC_CALL_BEGIN (\d+) (\d+) (\d+)\n(.*?)ATOMIC_CALL_END \1 \2 \3", result.stderr, re.S)
    assert len(cases) == 12
    for na, nb, branch, events in cases:
        na, nb, branch = int(na), int(nb), int(branch)
        commits = [line for line in events.splitlines() if line.startswith("FORT_SCOPED external_commit ")]
        if mode == "hidden" or branch == 4:
            assert not commits
            assert not re.search(r"^FORT_SCOPED (?:upload|download|allocate) ", events, re.M)
        else:
            assert len(commits) == (3 if branch == 1 else 2 if branch == 2 else 1), events
            if branch == 1:
                lines = events.splitlines()
                positions = [i for i, line in enumerate(lines) if line.startswith("FORT_SCOPED external_commit ")]
                # Identical repeated call-site records need ordered positions,
                # not string splitting. First b/q use may upload q between the
                # first two commits; repeated a/p needs no payload transfer.
                repeated = "\n".join(lines[positions[1] + 1 : positions[2]])
                assert not re.search(r"^FORT_SCOPED (?:upload|download) ", repeated, re.M), repeated
                payload = [
                    (i, line.split()[1], dict(word.split("=", 1) for word in line.split()[2:] if "=" in word))
                    for i, line in enumerate(lines)
                    if line.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download "))
                ]
                expected = {
                    "upload": {slots["argument::p"]: 8 * (na + 6), slots["argument::q"]: 8 * (nb + 6)},
                    "download": {slots["argument::a"]: 8 * (na + 6), slots["argument::b"]: 8 * (nb + 6)},
                }
                for kind in ("upload", "download"):
                    rows = [(i, row) for i, event, row in payload if event == kind]
                    assert len(rows) == 2, events
                    assert {row["buffer"]: int(row["bytes"]) for _, row in rows} == expected[kind], events
                    if kind == "upload":
                        first_use = {slots["argument::p"]: positions[0], slots["argument::q"]: positions[1]}
                        assert all(i < first_use[row["buffer"]] for i, row in rows), events
                    else:
                        assert all(i > positions[-1] for i, _ in rows), events
                assert len({row["domain"] for _, _, row in payload}) == 1, events

                copies = {
                    kind: sum(
                        int(re.search(r"\bbytes=(\d+)\b", line)[1])
                        for line in events.splitlines()
                        if line.startswith("FORT_SCOPED " + kind + " ")
                    )
                    for kind in ("upload", "download")
                }
                assert copies == {"upload": 8 * (na + nb + 12), "download": 8 * (na + nb + 12)}, events
