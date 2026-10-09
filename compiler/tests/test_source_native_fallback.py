"""Original hidden numerical state is used after new dummy association ends."""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

from compiler.tests.test_module_allocatable_scopes_runtime import ROOT, _run
from compiler.tests.test_runtime_allocation_bounds_cuda import _driver, _facts, _normalized, _package, _source


def generate(directory, *, structured=False, policy="sections"):
    directory.mkdir(parents=True, exist_ok=True)
    source, normalized, facts, package = [directory / name for name in
                                         ("original.f90", "normalized.f90", "captures.json", "numerical.json")]
    text = _source(1, False).replace("allocatable,target::field", "allocatable::field").replace("call touch(ni,nj)\n", "")
    if structured:
        text = text.replace("call consume(out,ni,nj)", "if(ni>0) then\ncall consume(out,ni,nj)\nendif")
    source.write_text(text)
    normalized.write_text(_normalized(1))
    facts.write_text(json.dumps(_facts(source, False)))
    package.write_text(json.dumps(_package(source, normalized, 1)))
    output = directory / "generated"
    response = _run([sys.executable, "-m", "compiler", "--form-scopes", "--scope-facts", str(facts),
                     "--numerical-sources", str(package), "--input", str(source), "--kernel", "step",
                     "--memory-model", "scoped", "--gpu-policy", policy, "--host-threads", "4",
                     "--json", "--output-dir", str(output)], ROOT)
    (directory / "public.json").write_text(response.stdout)
    public = json.loads(response.stdout)
    assert public["supported"], public
    scope, = public["scopes"]["scopes"]
    assert scope["native_fallback"]["position"] == "original caller after owning helper returns"
    return source, output, public["scopes"]


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("policy", ["sections", "auto"])
def test_hidden_numerical_fallback_is_returned_to_original_caller(tmp_path, structured, policy):
    source, output, manifest = generate(tmp_path, structured=structured, policy=policy)
    scope, = manifest["scopes"]
    text = (output / manifest["sources"][str(source)]["replacement"]).read_text()
    owner = text[text.index("subroutine " + scope["owner"]):text.index("end subroutine " + scope["owner"])]
    assert "logical, intent(out) :: fort_native_required" in owner
    assert owner.index("fort_native_required = .true.") < owner.index("fort_context = 0")
    assert owner.rstrip().endswith("fort_native_required = .false.")
    assert "call produce(" not in owner.lower()
    assert "call consume(" not in owner.lower()
    caller = text[text.index("subroutine step"):text.index("end subroutine", text.index("subroutine step"))]
    fallback = caller[caller.index("call " + scope["owner"]):]
    assert fallback.index("if (fort_native_") < fallback.index("call produce(")
    assert "logical :: fort_native_" in caller


@pytest.fixture(scope="module")
def fallback_binaries(tmp_path_factory):
    nvcc, host, fortran = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++"), shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA, Fortran and OpenMP toolchains are required")
    directory = tmp_path_factory.mktemp("source-native-fallback")
    flags = [fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    targets = {}
    for label, structured, policy in [("straight", False, "sections"), ("tree", True, "sections"), ("automatic", False, "auto")]:
        source, output, manifest = generate(directory / label, structured=structured, policy=policy)
        driver = output / "driver.f90"
        driver.write_text(_driver(1, False).replace(
            "expected(lo1+1)=expected(lo1+1)+sum(expected)+real(lbound(field,1,kind=8),8)", ""))
        native = output / "native"
        _run([*flags, str(source), str(driver), "-o", str(native)], output)
        reference = _run([str(native), "fields"], output, env={**os.environ, "OMP_NUM_THREADS": "4"}).stdout
        (output / "native.stdout").write_text(reference)
        objects = []
        for role in ("common_runtime", "shared_entry", "original_source"):
            for unit in manifest["build_sources"]:
                if unit["role"] != role:
                    continue
                target = output / (unit["path"].replace("/", "_") + ".o")
                command = ([nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp", "-I", str(output)]
                           if unit["language"] == "cuda" else flags)
                _run([*command, "-c", str(output / unit["path"]), "-o", str(target)], output)
                objects.append(str(target))
        shim = output / "preflight-failure.cpp"
        shim.write_text('''#include "scoped_runtime.h"
#include <cstdio>
#include <cstdlib>
extern "C" int __real_fort_scope_create(int,fort_scope_t*);
extern "C" int __wrap_fort_scope_create(int flags,fort_scope_t* context) {
  if (std::getenv("FIXTURE_FAIL_CREATE")) {
    *context=0; std::fprintf(stderr,"FIXTURE_CREATE_FAILURE\\n"); return FORT_SCOPE_RESOURCE;
  }
  return __real_fort_scope_create(flags,context);
}
''')
        shim_object = output / "preflight-failure.o"
        _run([host, "-std=c++17", "-I", str(output), "-c", str(shim), "-o", str(shim_object)], output)
        target = output / "verify"
        _run([*flags, str(driver), *objects, str(shim_object), "-Wl,--wrap=fort_scope_create",
              "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(target)], output)
        targets[label] = target, output, reference
    return targets


@pytest.mark.cuda
@pytest.mark.parametrize("label", ["straight", "tree", "automatic"])
@pytest.mark.parametrize("mode", ["gpu", "no-device", "create-failure"])
def test_hidden_numerical_bounds_and_fields_survive_caller_fallback(fallback_binaries, label, mode):
    target, output, reference = fallback_binaries[label]
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "FORT_RUNTIME_TRACE": "1"}
    if mode == "no-device":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    if mode == "create-failure":
        environment["FIXTURE_FAIL_CREATE"] = "1"
    result = _run([str(target), "fields"], output, env=environment)
    (output / (mode + ".stdout")).write_text(result.stdout)
    (output / (mode + ".stderr")).write_text(result.stderr)
    assert result.stdout == reference
    assert "FIELDS_OK" in result.stdout
    assert "array temporary" not in result.stderr.lower()
    if mode == "create-failure":
        # Missing-profile straight selection returns before context creation.
        assert result.stderr.count("FIXTURE_CREATE_FAILURE") == (0 if label == "automatic" else 8)
    if mode != "gpu" or label == "automatic":
        assert "FORT_SCOPED launch" not in result.stderr
        assert "FORT_SCOPED upload" not in result.stderr
    else:
        assert result.stderr.count("FORT_SCOPED launch") >= 8, result.stderr
