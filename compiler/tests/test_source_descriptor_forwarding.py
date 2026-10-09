"""Native children retain optional presence and allocation descriptors."""

import shutil

import pytest

from compiler.tests.test_source_scopes import PROGRAM, generate, run


def program(*, supplied, nested=False, allocatable=False):
    attributes = "allocatable" if allocatable else "optional"
    transform = f"""subroutine transform(b,extra)
real(8),intent(inout)::b(:)
real(8),{attributes},intent(in)::extra(:)
b=3*b
if ({'allocated' if allocatable else 'present'}(extra)) b=b+extra
end subroutine
"""
    original = PROGRAM[PROGRAM.index("subroutine transform(b)"):PROGRAM.index("subroutine consumer(a,b,out,n)")]
    source = PROGRAM.replace(original, transform)
    if nested:
        wrapper = """subroutine forward(b,extra)
real(8),intent(inout)::b(:)
real(8),optional,intent(in)::extra(:)
call transform(extra=extra,b=b)
end subroutine
"""
        source = source.replace("subroutine consumer(a,b,out,n)", wrapper + "subroutine consumer(a,b,out,n)")
    target = "forward" if nested else "transform"
    call = "call " + target + "(b=b" + (",extra=a)" if supplied else ")")
    source = source.replace("call transform(b)", call)
    if allocatable:
        before, entry = source.split("subroutine step(a,b,out,n)")
        entry = entry.replace("real(8),intent(in)::a(:)", "real(8),allocatable,intent(in)::a(:)", 1)
        source = before + "subroutine step(a,b,out,n)" + entry
    return source


@pytest.mark.parametrize("supplied", [False, True], ids=["omitted", "present"])
@pytest.mark.parametrize("nested", [False, True], ids=["direct", "native-forwarding"])
def test_native_optional_children_share_the_owner_without_optional_numerical_lowering(tmp_path, supplied, nested):
    _, output, manifest = generate(tmp_path, program(supplied=supplied, nested=nested))
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    scope, = manifest["scopes"]
    assert scope["gpu_leaves"] == ["original::consumer", "original::producer"]
    assert {resource["resource"] for resource in scope["resources"]} == {"argument::a", "argument::b", "argument::out"}
    target = "original::forward" if nested else "original::transform"
    native = next(item for item in manifest["numerical_decisions"] if item["procedure"] == target)
    assert not native["supported"]
    assert "descriptor-aware numerical variants" in native["reason"]
    assert all((output / source["path"]).is_file() for source in manifest["build_sources"])


def test_readonly_allocatable_native_child_gets_the_original_allocation_descriptor(tmp_path):
    original, output, manifest = generate(tmp_path, program(supplied=True, allocatable=True))
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    scope, = manifest["scopes"]
    assert scope["allocation_preflight"]["resources"] == ["argument::a"]
    edited = (output / manifest["sources"][str(original)]["replacement"]).read_text().lower()
    assert "allocatable, target, intent(in)" in edited
    assert "original::transform" not in scope["gpu_leaves"]


@pytest.mark.native
@pytest.mark.parametrize(("supplied", "nested", "allocatable"), [
    (False, False, False), (True, False, False), (False, True, False), (True, True, False), (True, False, True),
], ids=["omitted", "present", "forwarded-omitted", "forwarded-present", "readonly-allocation"])
def test_forwarded_native_descriptor_artifacts_compile_as_fortran(tmp_path, supplied, nested, allocatable):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    _, output, manifest = generate(tmp_path / "generated", program(supplied=supplied, nested=nested, allocatable=allocatable))
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    build = tmp_path / "build"
    build.mkdir()
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, source in enumerate(manifest["build_sources"]):
            if source["role"] == role and source["language"] == "fortran":
                run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-c",
                     str(output / source["path"]), "-J", str(build), "-I", str(build),
                     "-o", str(build / (str(index) + ".o"))], cwd=build)
