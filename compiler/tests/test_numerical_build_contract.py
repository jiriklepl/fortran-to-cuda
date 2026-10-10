"""Build requirements travel through public metadata and invalidate old costs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources
from compiler.emission.common.resources import read_scoped_runtime
from compiler.frontend import lower_source
from compiler.numerical_contract import (
    CUDA_OPTIONS,
    HOST_OPTIONS,
    cuda_compile_options,
    numerical_build_contract,
    require_explicit_cuda_environment,
    require_numerical_build_contract,
)
from compiler.offload.calibrate import _calibrate_scoped, calibrate, profile_from_measurements
from compiler.offload.collective_calibration import calibrate_collective
from compiler.offload.config import OffloadConfig
from compiler.offload.numerical_calibration import calibrate_numerical, calibrate_numerical_v2
from compiler.offload.profile import ProfileError, compiler_identity, validate_profile
from compiler.scopes.source import form_source_scopes
from compiler.tests.test_offload_profile import observations, profile
from compiler.tests.test_source_scopes import FACT, PROGRAM

SOURCE = """module renamed_arithmetic
implicit none
contains
subroutine advance(a,b,out,n)
real(8),intent(in)::a(:,:,:),b(:,:,:)
real(8),intent(out)::out(:,:,:)
integer,intent(in)::n
integer::i,j,k
do k=1,n
do j=1,n
do i=1,n
out(i,j,k)=a(i,j,k)*b(i,j,k)+a(i,j,k)
enddo
enddo
enddo
end subroutine
end module
"""


def test_contract_is_canonical_complete_and_returns_fresh_metadata():
    contract = numerical_build_contract()
    payload = {key: value for key, value in contract.items() if key != "identity"}
    assert contract["identity"] == sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert contract["required_cuda_options"] == list(CUDA_OPTIONS)
    assert contract["required_host_options"] == list(HOST_OPTIONS)
    assert cuda_compile_options() == (*CUDA_OPTIONS, "-Xcompiler=-ffp-contract=off")
    contract["required_cuda_options"].append("--use_fast_math")
    assert numerical_build_contract()["required_cuda_options"] == list(CUDA_OPTIONS)


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(schema_version=True),
    lambda value: value.update(identity="0" * 64),
    lambda value: value.update(id="permissive"),
    lambda value: value["required_cuda_options"].remove("--fmad=false"),
    lambda value: value["required_host_options"].append("-ffast-math"),
    lambda value: value.update(unreviewed_option=True),
])
def test_contract_rejects_altered_requirements_even_when_identity_is_retained(mutate):
    value = numerical_build_contract()
    mutate(value)
    with pytest.raises(ValueError, match="missing or incompatible"):
        require_numerical_build_contract(value)


def test_legacy_profile_remains_readable_but_cannot_authorize_changed_backend():
    legacy = profile()
    legacy.pop("numerical_contract")
    assert validate_profile(legacy) is legacy
    with pytest.raises(ProfileError, match="numerical build contract"):
        compiler_identity(legacy)
    automatic = OffloadConfig("auto", legacy)
    assert automatic.profile is None
    assert "numerical build contract" in automatic.profile_reason
    assert OffloadConfig("sections", legacy).policy == "sections"
    changed = deepcopy(profile())
    changed["numerical_contract"]["required_cuda_options"][0] = "--fmad=true"
    with pytest.raises(ProfileError, match="numerical build contract"):
        validate_profile(changed)


def test_reinterpreting_unlabelled_raw_measurements_does_not_invent_build_authority():
    old = profile_from_measurements(observations(), precision_bits=64, cpu_threads=4,
        cpu_name="CPU", nvcc_version="NVCC 13.4", host_cxx_version="GCC 14.4")
    assert "numerical_contract" not in old


@pytest.mark.parametrize("name", ["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"])
@pytest.mark.parametrize("call", ["base", "scoped", "numerical", "compute", "collective"])
def test_hidden_nvcc_flags_fail_before_any_build_or_tool_invocation(tmp_path, monkeypatch, name, call):
    monkeypatch.setenv(name, "--fmad=true")
    invoked = []
    def record_invocation(*args, **kwargs):
        invoked.append(args)

    args = SimpleNamespace()
    calls = {
        "base": lambda: calibrate(args),
        "scoped": lambda: _calibrate_scoped(None, args, tmp_path / "absent", "nvcc", "host"),
        "numerical": lambda: calibrate_numerical(
            None, args, tmp_path / "absent", "nvcc", "host", run=record_invocation),
        "compute": lambda: calibrate_numerical_v2(
            None, args, tmp_path / "absent", "nvcc", "host", run=record_invocation, tool=record_invocation),
        "collective": lambda: calibrate_collective(
            None, args, tmp_path / "absent", "nvcc", "host", run=record_invocation, tool=record_invocation),
    }
    with pytest.raises(ValueError, match=name):
        calls[call]()
    assert invoked == []
    assert list(tmp_path.iterdir()) == []


def test_empty_hidden_options_are_harmless_and_never_scrubbed(monkeypatch):
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "")
    require_explicit_cuda_environment({"NVCC_APPEND_FLAGS": ""})
    assert os.environ["NVCC_APPEND_FLAGS"] == ""


def test_library_and_scoped_runtime_publish_the_same_contract():
    function = lower_source(SOURCE, "advance", source_name="source.f90")
    function, plan = prepare_function(function, options=CompilerOptions())
    generated = generate_sources(function, plan)
    contract = numerical_build_contract()
    assert generated.numerical_contract == contract
    assert contract["identity"] in generated.cuda
    runtime, manifest = read_scoped_runtime()
    assert manifest["numerical_contract"] == contract
    assert contract["identity"] in runtime["scoped_runtime.cu"]
    for item in manifest["sources"]:
        if item["language"] == "cuda":
            assert item["numerical_contract"] == contract
        else:
            assert "numerical_contract" not in item


def test_source_scopes_publish_contract_on_every_cuda_artifact(tmp_path):
    path = tmp_path / "source.f90"
    # The public bounded owner requires a useful multi-operation scope.
    path.write_text(PROGRAM.replace("module original", "module renamed_arithmetic"))
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": FACT, "argument::b": {**FACT, "initialized": "none"},
                          "argument::out": {**FACT, "initialized": "none"}}}
    _, manifest = form_source_scopes([path], "renamed_arithmetic::step", facts=facts,
        options=CompilerOptions(), config=OffloadConfig("sections"))
    assert manifest["scope_count"], manifest["boundaries"]
    for item in manifest["build_sources"]:
        assert (item.get("numerical_contract") == numerical_build_contract()) == (item["language"] == "cuda")


def test_ordinary_json_build_metadata_uses_actual_output_names(tmp_path):
    path = tmp_path / "source.f90"
    path.write_text(SOURCE)
    result = subprocess.run([sys.executable, "-m", "compiler", "-i", str(path), "-k", "advance",
        "--json", "--cuda-output", "custom.cu", "--cpp-output", "custom.cpp",
        "--fortran-output", "custom.f90", "--output-dir", str(tmp_path / "generated")],
        text=True, capture_output=True, check=False, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["numerical_contract"] == numerical_build_contract()
    units = {item["path"]: item for item in report["build_sources"]}
    assert units["custom.cu"]["numerical_contract"] == numerical_build_contract()
    assert units["custom.cpp"]["numerical_contract"] == numerical_build_contract()
    assert "numerical_contract" not in units["custom.f90"]
