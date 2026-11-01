"""Per-call source tasks retain distinct layouts and unchanged read replicas."""

import json
import math
import os
import re
import shutil
from hashlib import sha256

import pytest

from compiler.tests.test_source_scopes import run
from compiler.tests.test_source_task_call_bindings import multi_field_builder

SHAPES = (
    ((128, 64, 64), (129, 61, 65), (125, 66, 63)),
    ((1, 7, 5), (1, 6, 4), (1, 8, 6)),
    ((9, 1, 5), (8, 1, 4), (10, 1, 6)),
    ((9, 5, 1), (8, 4, 1), (10, 6, 1)),
    ((0, 5, 3), (0, 4, 6), (0, 7, 4)),
)


def driver_source():
    source = """program exercise
use caller, only:step
implicit none
integer::which,i,j,k,f,n(3,3),residue
real(8),allocatable::a(:,:,:),b(:,:,:),c(:,:,:)
integer,allocatable::ma(:,:,:),mb(:,:,:),mc(:,:,:)
character(512)::arg,out
call get_command_argument(1,arg)
read(arg,*)which
call get_command_argument(2,out)
select case(which)
"""
    for case, shapes in enumerate(SHAPES):
        values = ",".join(str(n) for shape in shapes for n in shape)
        source += f"case({case})\nn=reshape([{values}],[3,3])\n"
    source += "case default\nerror stop 'unknown case'\nend select\n"
    for f, name in enumerate("abc", 1):
        source += f"""allocate({name}(-2:n(1,{f})+3,-2:n(2,{f})+3,-2:n(3,{f})+3))
allocate(m{name}(-2:n(1,{f})+3,-2:n(2,{f})+3,-2:n(3,{f})+3))
do k=-2,n(3,{f})+3
 do j=-2,n(2,{f})+3
  do i=-2,n(1,{f})+3
   {name}(i,j,k)=real({f}*i*i*i*i+3*j*j*j*j+5*k*k*k*k+17*i*j+1000*{f},8)/7._8
   residue=modulo(i+2*j+3*k+{f},5)
   m{name}(i,j,k)=0
   if({f}>1)then
    if(residue==0)m{name}(i,j,k)=1
    if(residue==1)m{name}(i,j,k)=-1
   endif
  enddo
 enddo
enddo
"""
    return (
        source
        + """call step(a,b,c,ma,mb,mc,n(1,1),n(2,1),n(3,1),n(1,2),n(2,2),n(3,2),n(1,3),n(2,3),n(3,3))
open(10,file=trim(out),access='stream',form='unformatted',status='replace')
write(10)a,b,c,ma,mb,mc
close(10)
print *, 'MULTI_FIELD_ORIGINAL_CALLS_OK'
end program
"""
    )


def _number(row, name):
    return int(re.search(r"\b" + name + r"=(\d+)", row)[1])


def checked_multi_field_ledger(rows, registrations, shapes):
    """Independent original line geometry; no path/provider identity shortcuts."""

    def directions(shape):
        nx, ny, nz = shape
        return (2, 3) if nx == 1 else (1, 3) if ny == 1 else (3, 3) if nz == 1 else (1, 2, 3)

    commits = [i for i, row in enumerate(rows) if row.startswith("FORT_SCOPED external_commit ")]
    launches = [
        (i, 1 if row.startswith("FORT_SCOPED launch ") else _number(row, "bytes"))
        for i, row in enumerate(rows)
        if row.startswith(("FORT_SCOPED launch ", "FORT_SCOPED graph_submit "))
    ]
    copies = [
        (i, row) for i, row in enumerate(rows) if row.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download "))
    ]
    if not all(shapes[0]):
        assert not commits and not launches and not copies
        return {"zero_trip": True}
    sequence = [(field, axis) for field in (0, 0, 1, 2) for axis in directions(shapes[field])]
    # multi_field_builder(repeat=True) emits a,a,b,c in original source order.
    assert len(commits) == len(sequence)
    expected_launches = 0
    for field, axis in sequence:
        shape = shapes[field]
        lines = math.prod(shape[d] for d in range(3) if d != axis - 1)
        stride = 8 * (shape[axis - 1] + 6)
        batch = min(lines, max(1, 4194304 // stride))
        expected_launches += (lines + batch - 1) // batch
    assert sum(count for _, count in launches) == expected_launches
    mask_ids = [registrations["argument::m" + name] for name in "abc"]
    assert len(set(mask_ids)) == 3
    seen = [set(), set(), set()]
    identity = [set(), set(), set()]
    phases = []
    previous = -1
    for (field, axis), finish in zip(sequence, commits, strict=True):
        shape = shapes[field]
        others = math.prod(shape[d] for d in range(3) if d != axis - 1)
        expected = 0 if axis in seen[field] else 4 * others * (6 if seen[field] else shape[axis - 1] + 6)
        observed = [0, 0, 0]
        for i, row in copies:
            if not row.startswith("FORT_SCOPED upload ") or not previous < i < finish:
                continue
            buffer = _number(row, "buffer")
            if buffer in mask_ids:
                which = mask_ids.index(buffer)
                observed[which] += _number(row, "bytes")
                identity[which].add((_number(row, "buffer_handle"), _number(row, "generation")))
        wanted = [expected if f == field else 0 for f in range(3)]
        assert observed == wanted
        phases.append({"field": field, "axis": axis, "expected": wanted, "observed": observed})
        seen[field].add(axis)
        previous = finish
    assert all(len(values) == 1 for values in identity)
    assert len({next(iter(values))[0] for values in identity}) == 3
    assert not any(row.startswith("FORT_SCOPED download ") and _number(row, "buffer") in mask_ids for _, row in copies)
    assert all(i > commits[-1] for i, row in copies if row.startswith("FORT_SCOPED download "))
    # Equal original a invocations retain the full previously read line union.
    start, end = commits[len(directions(shapes[0])) - 1], commits[2 * len(directions(shapes[0])) - 1]
    assert not any(start < i < end for i, _row in copies)
    return {
        "zero_trip": False,
        "kernel_launches": expected_launches,
        "phases": phases,
        "mask_identities": [list(next(iter(values))) for values in identity],
        "repeated_call_payload_copies": 0,
    }


@pytest.mark.cuda
def test_three_field_original_calls_keep_canonical_layouts_and_read_replicas(tmp_path):
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    assert all((nvcc, host, fc)), "Controlled CUDA qualification requires the explicit toolchain"
    builder = multi_field_builder(tmp_path, repeat=True, separate_sizes=True)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report["boundaries"]
    manifest = json.loads(outputs["scope-manifest.json"])
    (scope,) = manifest["scopes"]
    assert len(scope["source_device_tasks"]) == 3
    build = tmp_path / "build"
    native_build = tmp_path / "native"
    build.mkdir()
    native_build.mkdir()
    for name, content in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["role"] != role:
                continue
            path = build / artifact["path"]
            assert sha256(path.read_bytes()).hexdigest() == manifest["artifacts_sha256"][artifact["path"]]
            output = build / f"object-{index}.o"
            command = (
                [
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
                if artifact["language"] == "cuda"
                else [
                    fc,
                    "-fopenmp",
                    "-fcheck=all",
                    "-ffp-contract=off",
                    "-ffree-line-length-none",
                    "-J",
                    str(build),
                    "-I",
                    str(build),
                ]
            )
            result = run([*command, "-c", str(path), "-o", str(output)], cwd=build)
            (build / f"compile-{index}.stderr").write_text(result.stderr)
            objects.append(str(output))
    driver = tmp_path / "driver.f90"
    driver.write_text(driver_source())
    native, candidate = native_build / "chain", build / "chain"
    flags = [fc, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-ffree-line-length-none"]
    run(
        [*flags, "-J", str(native_build), str(tmp_path / "source.f90"), str(driver), "-o", str(native)],
        cwd=native_build,
    )
    run(
        [
            *flags,
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
    env = {k: v for k, v in os.environ.items() if not k.startswith("FORT_") and k != "CUDA_VISIBLE_DEVICES"}
    env.update(OMP_NUM_THREADS="4", OMP_THREAD_LIMIT="4", OMP_DYNAMIC="FALSE", OMP_SCHEDULE="static")
    registrations = {row["resource"]: row["registration_identity"] for row in scope["resource_bindings"]["resources"]}
    results = []
    for case, shapes in enumerate(SHAPES):
        reference = build / f"case{case}-native.bin"
        native_result = run([str(native), str(case), str(reference)], cwd=native_build, env=env, timeout=60)
        expected = reference.read_bytes()
        assert len(expected) == sum(math.prod(n + 6 for n in shape) * 12 for shape in shapes)
        for mode in ("traced", "untraced", "hidden"):
            current_env = dict(env)
            if mode != "untraced":
                current_env["FORT_RUNTIME_TRACE"] = "1"
            if mode == "hidden":
                current_env["CUDA_VISIBLE_DEVICES"] = ""
            output = build / f"case{case}-{mode}.bin"
            actual = run([str(candidate), str(case), str(output)], cwd=build, env=current_env, timeout=60)
            (build / f"case{case}-{mode}.stderr").write_text(actual.stderr)
            assert actual.stdout == native_result.stdout
            assert output.read_bytes() == expected  # All fields, complete halos and unchanged masks.
            rows = [row for row in actual.stderr.splitlines() if row.startswith("FORT_SCOPED ")]
            ledger = checked_multi_field_ledger(rows, registrations, shapes) if mode == "traced" else None
            if mode == "hidden":
                assert not any(
                    row.startswith(("FORT_SCOPED launch ", "FORT_SCOPED graph_submit ", "FORT_SCOPED external_commit "))
                    for row in rows
                )
            results.append({"case": case, "mode": mode, "byte_exact": True, "ledger": ledger})
            (build / "qualification.json").write_text(json.dumps(results, indent=2) + "\n")
