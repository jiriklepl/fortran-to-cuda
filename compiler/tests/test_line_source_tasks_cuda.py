"""A generated GPU producer, three original line teams and GPU consumer."""

import json
import os
import re
import shutil
from hashlib import sha256

import numpy as np
import pytest

from compiler.tests.test_line_source_tasks import emit_chain
from compiler.tests.test_source_scopes import run


def _exercise_line_chain(tmp_path, *, repeat_first):
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    assert all((nvcc, host, fc)), "Controlled CUDA qualification requires its explicit toolchain"
    original, outputs, manifest = emit_chain(tmp_path, repeat_first=repeat_first)
    (scope,) = manifest["scopes"]
    line_count = 4 if repeat_first else 3
    assert len(scope["source_device_tasks"]) == line_count
    assert len(scope["planning_segments"]) == 2
    build = tmp_path / "build"
    build.mkdir()
    for name, content in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (build / "scope-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["role"] != role:
                continue
            path = build / artifact["path"]
            assert sha256(path.read_bytes()).hexdigest() == manifest["artifacts_sha256"][artifact["path"]]
            target = build / f"object-{index}.o"
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
            result = run([*command, "-c", str(path), "-o", str(target)], cwd=build)
            (build / f"compile-{index}.stdout").write_text(result.stdout)
            (build / f"compile-{index}.stderr").write_text(result.stderr)
            objects.append(str(target))
    driver = build / "driver.f90"
    driver.write_text("""program exercise
use lines
implicit none
integer::nx,ny,nz,i,j,k,mode,residue
real(8),allocatable::a(:,:,:),b(:,:,:)
integer,allocatable::mask(:,:,:)
character(512)::arg,out
call get_command_argument(1,arg)
read(arg,*)nx
call get_command_argument(2,arg)
read(arg,*)ny
call get_command_argument(3,arg)
read(arg,*)nz
call get_command_argument(4,arg)
read(arg,*)mode
call get_command_argument(5,out)
allocate(a(-2:nx+3,-2:ny+3,-2:nz+3),b(-2:nx+3,-2:ny+3,-2:nz+3),mask(-2:nx+3,-2:ny+3,-2:nz+3))
do k=-2,nz+3
 do j=-2,ny+3
  do i=-2,nx+3
   a(i,j,k)=real(i*i*i*i+3*j*j*j*j+5*k*k*k*k+17*i*j,8)/7._8
   b(i,j,k)=real(500-i+2*j-3*k,8)/32._8
   residue=modulo(i+2*j+3*k+mode,5)
   mask(i,j,k)=0
   if(mode/=0)then
    if(residue==0)mask(i,j,k)=1
    if(residue==1)mask(i,j,k)=-1
   endif
  enddo
 enddo
enddo
call advance(a,b,mask,nx,ny,nz)
open(10,file=trim(out),access='stream',form='unformatted',status='replace')
write(10)a,b,mask
close(10)
print *, 'LINE_CHAIN_FULL_FIELDS_HALOS_MASK_OK'
end program
""")
    candidate, native = build / "candidate", build / "native"
    command = [fc, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-I", str(build), str(driver)]
    run(
        [
            *command,
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
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("FORT_") and key != "CUDA_VISIBLE_DEVICES"
    }
    environment.update(OMP_NUM_THREADS="4", OMP_THREAD_LIMIT="4", OMP_DYNAMIC="FALSE", OMP_SCHEDULE="static")
    registrations = {row["resource"]: row["registration_identity"] for row in scope["resource_bindings"]["resources"]}
    mask_identity = registrations["argument::mask"]
    qualification = []
    for case, (shape, mask_mode) in enumerate([((0, 3, 5), 0), ((7, 5, 3), 0), ((9, 4, 6), 1), ((5, 7, 4), 2)]):
        length = int(np.prod([n + 6 for n in shape]))
        args = [*map(str, shape), str(mask_mode)]
        reference_path = build / f"case{case}-native.bin"
        reference = run([str(native), *args, str(reference_path)], cwd=build, env=environment, timeout=30)
        (build / f"case{case}-native.stdout").write_text(reference.stdout)
        (build / f"case{case}-native.stderr").write_text(reference.stderr)
        data = reference_path.read_bytes()
        fields = np.frombuffer(data, dtype=np.float64, count=2 * length)
        mask_bytes = data[16 * length :]
        for mode in ("traced", "untraced", "hidden"):
            env = dict(environment)
            if mode != "untraced":
                env["FORT_RUNTIME_TRACE"] = "1"
            if mode == "hidden":
                env["CUDA_VISIBLE_DEVICES"] = ""
            output = build / f"case{case}-{mode}.bin"
            result = run([str(candidate), *args, str(output)], cwd=build, env=env, timeout=30)
            (build / f"case{case}-{mode}.stdout").write_text(result.stdout)
            (build / f"case{case}-{mode}.stderr").write_text(result.stderr)
            actual_data = output.read_bytes()
            assert len(actual_data) == len(data)
            actual = np.frombuffer(actual_data, dtype=np.float64, count=2 * length)
            np.testing.assert_allclose(actual, fields, atol=1e-6, rtol=1e-5)
            assert actual_data[16 * length :] == mask_bytes
            extents = tuple(n + 6 for n in shape)
            halo = np.ones(extents, dtype=bool)
            halo[3 : shape[0] + 3, 3 : shape[1] + 3, 3 : shape[2] + 3] = False
            for field in (0, 1):
                actual_field = actual[field * length : (field + 1) * length].reshape(extents, order="F")
                original_field = fields[field * length : (field + 1) * length].reshape(extents, order="F")
                assert actual_field[halo].tobytes() == original_field[halo].tobytes()
            assert result.stdout == reference.stdout
            rows = [line for line in result.stderr.splitlines() if line.startswith("FORT_SCOPED ")]
            launches = [
                (i, 1 if row.startswith("FORT_SCOPED launch ") else int(re.search(r"\bbytes=(\d+)", row)[1]))
                for i, row in enumerate(rows)
                if row.startswith(("FORT_SCOPED launch ", "FORT_SCOPED graph_submit "))
            ]
            uploads = [(i, row) for i, row in enumerate(rows) if row.startswith("FORT_SCOPED upload ")]
            downloads = [(i, row) for i, row in enumerate(rows) if row.startswith("FORT_SCOPED download ")]
            commits = [row for row in rows if row.startswith("FORT_SCOPED external_commit ")]
            mask_ledger = None
            if mode == "traced" and all(shape):
                assert sum(count for _, count in launches) == 2 + line_count, result.stderr
                assert len(commits) == line_count, result.stderr
                assert downloads, result.stderr
                assert min(i for i, _ in downloads) > max(i for i, _ in launches), result.stderr
                assert not any(int(re.search(r"\bbuffer=(\d+)", row)[1]) == mask_identity for _, row in downloads)
                mask_ledger = checked_mask_upload_phases(rows, mask_identity, shape, repeat_first=repeat_first)

            else:
                assert not launches, result.stderr
                assert not commits, result.stderr
                assert not uploads, result.stderr
                assert not downloads, result.stderr
            qualification.append(
                {
                    "shape": list(shape),
                    "mask_mode": mask_mode,
                    "mode": mode,
                    "fields_pass": True,
                    "mask_byte_exact": True,
                    "all_bytes_equal": actual_data == data,
                    "max_absolute_error": float(np.max(np.abs(actual - fields))),
                    "logical_kernels": sum(count for _, count in launches),
                    "source_commits": len(commits),
                    "mask_ledger": mask_ledger,
                    "uploads": [row for _, row in uploads],
                    "downloads": [row for _, row in downloads],
                }
            )
    (build / "line-chain-qualification.json").write_text(json.dumps(qualification, indent=2) + "\n")


def checked_mask_upload_phases(rows, mask_identity, shape, *, repeat_first):
    """Independent geometry for three axis slabs sharing an unchanged interior.

    The first axis needs its complete x line; later axes extend that replica by
    their two three-cell halo slabs. Repeating x has exactly the same read set
    and cannot require any payload copy. Trace records contain physical byte
    counts, not boxes, so this checks exact source-ordered union increments.
    """
    nx, ny, nz = shape
    expected = [4 * (nx + 6) * ny * nz, 4 * 6 * nx * nz, 4 * 6 * nx * ny]
    if repeat_first:
        expected.insert(1, 0)
    commits = [i for i, row in enumerate(rows) if row.startswith("FORT_SCOPED external_commit ")]
    assert len(commits) == len(expected)
    uploads = [
        (i, row)
        for i, row in enumerate(rows)
        if row.startswith("FORT_SCOPED upload ") and int(re.search(r"\bbuffer=(\d+)", row)[1]) == mask_identity
    ]
    assert uploads
    assert max(i for i, _ in uploads) < commits[-1]
    observed, previous = [], -1
    for commit in commits:
        observed.append(sum(int(re.search(r"\bbytes=(\d+)", row)[1]) for i, row in uploads if previous < i < commit))
        previous = commit
    assert observed == expected, (observed, expected)
    # A new generation/handle would not establish retention of this unchanged
    # canonical allocation, even if its fresh copies happened to match a budget.
    identities = {
        (int(re.search(r"\bgeneration=(\d+)", row)[1]), int(re.search(r"\bbuffer_handle=(\d+)", row)[1]))
        for _, row in uploads
    }
    assert len(identities) == 1
    if repeat_first:
        assert not any(
            row.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download "))
            for row in rows[commits[0] + 1 : commits[1]]
        )
    return {
        "source_geometry": "x-line, optional repeated x-line, y halos, z halos",
        "expected_mask_upload_bytes": expected,
        "physical_mask_upload_bytes": observed,
        "mask_generation_handle": list(next(iter(identities))),
        "equal_axis_payload_copy_count": 0 if repeat_first else None,
    }


@pytest.mark.cuda
def test_generated_line_chain_preserves_fields_halos_and_resident_values(tmp_path):
    _exercise_line_chain(tmp_path, repeat_first=False)


@pytest.mark.cuda
def test_generated_repeated_line_chain_reuses_unchanged_interior_and_extends_replicas(tmp_path):
    _exercise_line_chain(tmp_path, repeat_first=True)
