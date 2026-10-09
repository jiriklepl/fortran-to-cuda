"""Compile public Fortran artifacts for source-preserving keyword mappings."""

import json
import shutil
from hashlib import sha256

import pytest

from compiler.tests.test_source_scopes import PROGRAM, generate, run


def keyword_program(nested):
    if not nested:
        return PROGRAM.replace("call producer(a,b,n)", "call producer(n=n,b=b,a=a)").replace(
            "call transform(b)", "call transform(b=b)"
        ).replace("call consumer(a,b,out,n)", "call consumer(out=out,n=n,a=a,b=b)")
    return PROGRAM[:PROGRAM.index("subroutine step(a,b,out,n)")] + """subroutine wrapper(input_field,work_field,result_field,extent)
real(8),intent(in)::input_field(:)
real(8),intent(out)::work_field(:),result_field(:)
integer,intent(in)::extent
call producer(b=work_field,n=extent,a=input_field)
call transform(b=work_field)
call consumer(out=result_field,b=work_field,n=extent,a=input_field)
end subroutine
subroutine step(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:),out(:)
integer,intent(in)::n
call wrapper(result_field=out,extent=n,input_field=a,work_field=b)
call wrapper(extent=n,work_field=b,result_field=out,input_field=a)
end subroutine
end module
"""


@pytest.mark.native
@pytest.mark.parametrize("nested", [False, True], ids=["reordered-native-middle", "nested-wrapper-keywords"])
def test_public_mixed_scope_keyword_artifacts_compile_as_fortran(tmp_path, nested):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    original, output, manifest = generate(tmp_path / "generated", keyword_program(nested), mode="sections")
    scope, = manifest["scopes"]
    assert scope["gpu_leaves"] == ["original::consumer", "original::producer"]
    assert scope["calls"] == (["original::wrapper", "original::wrapper"] if nested else
                              ["original::producer", "original::transform", "original::consumer"])
    native = next(call for call in manifest["resolved_calls"] if call["procedure"] == "original::transform")
    assert native["original_arguments"][0]["keyword"] == "b"
    assert manifest["sources"][str(original)]["sha256"] == sha256(original.read_bytes()).hexdigest()

    build = tmp_path / "fortran-build"
    build.mkdir()
    evidence = []
    # Respect the public dependency roles and source order within each role.
    # Original modules use the generated entry modules, which use the runtime.
    for role in ("common_runtime", "shared_entry", "original_source"):
        for source in manifest["build_sources"]:
            if source["role"] != role or source["language"] != "fortran":
                continue
            path = output / source["path"]
            target = build / (str(len(evidence)) + ".o")
            command = [fortran, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-c",
                       str(path), "-J", str(build), "-I", str(build), "-o", str(target)]
            result = run(command, cwd=build)
            evidence.append({"role": role, "path": source["path"], "sha256": sha256(path.read_bytes()).hexdigest(),
                             "command": command, "stdout": result.stdout, "stderr": result.stderr})
            assert target.is_file()
    assert {item["role"] for item in evidence} == {"common_runtime", "shared_entry", "original_source"}
    (build / "compile-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
