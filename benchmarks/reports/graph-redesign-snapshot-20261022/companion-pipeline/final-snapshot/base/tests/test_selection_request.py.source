"""Immutable request transport and a real compiler/build-origin handoff."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from elmm_pipeline import selection_request as requests
from elmm_pipeline.build_receipts import (
    CommandRecorder,
    file_identity,
    snapshot_inputs,
    summarize_build,
)
from elmm_pipeline.local_adapter import apply_scope_sources, transform_scopes

SOURCE = """module orbit_fields
implicit none
real(8)::old(-1:16),viscosity(-1:16),star(-1:16),out(-1:16)
contains
subroutine advance(n)
integer,intent(in)::n
call producer(old,viscosity,n)
call momentum(old,viscosity,star,out,n)
end subroutine
subroutine producer(a,v,n)
real(8),intent(in)::a(-1:)
real(8),intent(inout)::v(-1:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
v(i)=2.d0*a(i)
enddo
!$omp end parallel do
end subroutine
subroutine momentum(a,v,s,o,n)
real(8),intent(in)::a(-1:)
real(8),intent(inout)::v(-1:),s(-1:),o(-1:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
s(i)=v(i)+a(i)
enddo
!$omp end parallel do
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
o(i)=o(i)+s(i)
enddo
!$omp end parallel do
end subroutine
end module
"""


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return file_identity(path)


@pytest.fixture(scope="module")
def executed_origins(tmp_path_factory):
    """Compile real native/sections origins; no fabricated compiler receipts."""
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    host = shutil.which("g++-14")
    if not fortran or not Path(nvcc).is_file() or not host:
        pytest.skip("Fortran/CUDA build toolchains required")
    root = tmp_path_factory.mktemp("request-origins")
    compiler = Path(__file__).resolve().parents[2]
    if not (compiler / "compiler/__main__.py").is_file():
        pytest.skip("public compiler checkout required")
    immutable = root / "immutable"
    immutable.mkdir()
    source, driver = immutable / "orbit.f90", immutable / "main.f90"
    source.write_text(SOURCE)
    driver.write_text("program main\nuse orbit_fields\nold=1.d0\nviscosity=0.d0\nstar=0.d0\nout=0.d0\ncall advance(16)\nend program\n")
    diagnostic = immutable / "diagnostic.f90"
    diagnostic.write_text(driver.read_text() + "! reviewed full-storage diagnostic driver\n")
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(source): file_identity(source)["sha256"]},
             "captures": {"orbit_fields::" + name: {"storage": "stable", "initialized": "whole", "escapes": False,
                                                   "allocation_changes": False}
                          for name in ("old", "viscosity", "star", "out")}}
    facts_ref = save(root / "captures.json", facts)
    origins = {}
    for mode in ("native", "resident"):
        build_root, originals = root / mode, root / (mode + "-sources")
        build_root.mkdir()
        originals.mkdir()
        copied = {str(path): originals / path.name for path in (source, driver)}
        for original, destination in copied.items():
            shutil.copyfile(original, destination)
        generated = []
        if mode == "resident":
            metadata = transform_scopes(source, compiler, Path(sys.executable), build_root,
                entry="orbit_fields::advance", capture_facts=facts,
                gpu_options={"gpu_policy": "sections", "scope_execution": "reached"})
            assert metadata["scope_manifest"]["scope_count"] == 1
            apply_scope_sources(metadata, {str(source): copied[str(source)]})
            generated = [(build_root / item["path"], item["language"])
                         for item in sorted(metadata["scope_manifest"]["build_sources"],
                                            key=lambda row: row["role"] != "common_runtime")
                         if item["role"] != "original_source"]
            contracts = [item["numerical_contract"] for item in metadata["gpu_sources"] if "numerical_contract" in item]
            cuda_options = list(dict.fromkeys(flag for item in contracts for flag in item["required_cuda_options"]))
            host_options = list(dict.fromkeys(flag for item in contracts for flag in item["required_host_options"]))
        recorder = CommandRecorder(build_root / "commands")
        objects = {}
        # Build modules first, while preserving a single generated-object link block.
        for index, (path, language) in enumerate([*generated, *((path, "fortran") for path in copied.values())]):
            obj = build_root / (str(index) + ".o")
            if language == "fortran":
                argv = [fortran, "-O2", "-cpp", "-fopenmp", "-ffp-contract=off", "-std=f2018",
                        "-I" + str(build_root), "-J" + str(build_root), "-c", str(path), "-o", str(obj)]
            else:
                argv = [nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp",
                        *cuda_options, *("-Xcompiler=" + flag for flag in host_options),
                        "-I" + str(build_root), "-c", str(path), "-o", str(obj)]
            completed = recorder.run(argv, cwd=originals, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
            assert completed.returncode == 0, completed.stderr.decode()
            objects[path] = obj
        binary = build_root / "application"
        linked = [objects[path] for path in copied.values()] + [objects[path] for path, _language in generated]
        argv = [fortran, "-fopenmp", *map(str, linked)]
        if mode == "resident":
            argv += ["-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++"]
        completed = recorder.run([*argv, "-o", str(binary)], cwd=originals,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
        assert completed.returncode == 0, completed.stderr.decode()
        receipt = save(build_root / "receipt.json", summarize_build(binary, [recorder.directory],
                       snapshot_inputs([originals, build_root])))
        origins[mode] = {"receipt": receipt, "build_root": str(build_root),
                         "source_mapping": {str(copy): original for original, copy in copied.items()}}
        if mode == "resident":
            origins[mode]["manifest"] = file_identity(build_root / "scope-manifest.json")
    config = root / "configuration"
    config.mkdir()
    (config / "case.conf").write_text("steps=3\n")
    policy = {"schema_version": 2, "protocol": "full-float64-comparison-v2", "dtype": "<f8",
              "fields": {name: {"payload": name + ".bin", "metadata": name + ".shape"}
                         for name in ("old", "viscosity", "star", "out")},
              "metadata_format": "storage-shape-bounds-v1", "atol": 1e-6, "rtol": 1e-5,
              "finite_only": True, "exact_layouts": True,
              "outputs": {"format": "raw-float64-storage-v1", "files": {
                  name: {"payload": name + ".bin", "metadata": name + ".shape"} for name in ("final", "average")}},
              "driver": {"original_source": str(driver), "original_sha256": file_identity(driver)["sha256"],
                         "replacement_sha256": file_identity(diagnostic)["sha256"]}}
    request = {"schema_version": 1, "compiler": {"compiler_root": str(compiler), "python": sys.executable},
               "analysis": {"source": file_identity(source), "entry": "orbit_fields::advance", "capture_facts": facts_ref},
               "origins": origins, "validation_policy": save(root / "policy.json", policy),
               "diagnostic_source": file_identity(diagnostic), "acceptance": {"criterion": "paired_non_regression"},
               "gpu_options": {"host_threads": 4},
               "workloads": [{"configuration_directory": str(config), "files": ["case.conf"],
                              "argv": ["application"], "environment": {}, "completion": {"steps": 3}}]}
    return request


def test_real_executed_origins_issue_capture_contracts_through_public_compiler(tmp_path, executed_origins):
    assembled = requests.assemble(executed_origins, tmp_path / "request")
    assert assembled["placement_authority"] is False
    contracts = []
    for mode in ("native", "resident"):
        kwargs = requests.generation_inputs(assembled["session"], mode)
        assert kwargs["gpu_options"]["gpu_policy"] == "auto"
        assert kwargs["source_path"] == executed_origins["analysis"]["source"]["path"]
        metadata = transform_scopes(output_dir=tmp_path / mode, **kwargs)
        assert metadata["scope_manifest"]["scope_count"] == 1
        assert "application_build" in metadata
        scope, = metadata["scope_manifest"]["scopes"]
        exact = scope["exact_window"]
        assert exact["capture_variant"] == mode
        assert exact["contract"]["protocol"] == "checked-interval-process-v4"
        contracts.append(exact["contract"])
    assert contracts[0] == contracts[1]


def test_session_reuses_sources_and_accepts_only_matching_collector_transport(tmp_path, executed_origins):
    assembled = requests.assemble(executed_origins, tmp_path / "request")
    session = requests._read(assembled["session"])
    base = session["base"]
    evidence = {**deepcopy(base), "evidence": {"protocol": "checked-interval-process-v4", "campaigns": [
        {"contract": {"sources": session["source_inputs"]}}]}}
    evidence["validation_policy"]["identity"] = "compiler-added-policy-identity"
    evidence["build_contract"]["identity"] = "compiler-added-build-identity"
    reference = save(tmp_path / "production.json", evidence)
    production = requests.generation_inputs(assembled["session"], "production", production_scope_cost=reference)
    assert production["gpu_options"]["scope_cost"] == reference["path"]
    assert production["source_path"] == executed_origins["analysis"]["source"]["path"]
    for change in (lambda value: value["workloads"][0]["completion"].update(steps=99),
                   lambda value: value["acceptance"].update(criterion="strict_advantage"),
                   lambda value: value["evidence"]["campaigns"][0]["contract"].update(sources={})):
        changed = deepcopy(evidence)
        change(changed)
        with pytest.raises(ValueError, match="another"):
            requests.generation_inputs(assembled["session"], "production",
                                       production_scope_cost=save(tmp_path / "changed.json", changed))
    with pytest.raises(ValueError, match="cannot consume production"):
        requests.generation_inputs(assembled["session"], "native", production_scope_cost=reference)
    input_path = tmp_path / "input.json"
    save(input_path, executed_origins)
    command = [sys.executable, "-m", "elmm_pipeline.selection_request", "--request", str(input_path),
               "--output", str(tmp_path / "cli-request")]
    result = subprocess.run(command, text=True, capture_output=True, timeout=240, check=False)
    assert result.returncode == 0, result.stderr
    cli = json.loads(result.stdout)
    assert requests.generation_inputs(cli["session"], "resident")["source_path"] == production["source_path"]


def test_dependency_changes_invalidate_session_before_public_generation(tmp_path, executed_origins):
    request = deepcopy(executed_origins)
    dependency = tmp_path / "configured.inc"
    dependency.write_text("integer,parameter::n=16\n")
    source = request["analysis"]["source"]
    package = {"source_inputs": {source["path"]: source["sha256"]}, "entries": [],
               "dependencies": {str(dependency): file_identity(dependency)["sha256"]}}
    request["analysis"]["analysis_sources"] = save(tmp_path / "analysis.json", package)
    assembled = requests.assemble(request, tmp_path / "request")
    dependency.write_text("integer,parameter::n=17\n")
    with pytest.raises(ValueError, match="input changed"):
        requests.generation_inputs(assembled["session"], "resident")


def test_foreign_facts_and_replayed_output_are_rejected(tmp_path, executed_origins):
    request = deepcopy(executed_origins)
    request["analysis"]["capture_facts"] = save(tmp_path / "facts.json", {"sources": {}})
    with pytest.raises(ValueError, match="another analysis source set"):
        requests.assemble(request, tmp_path / "bad")
    requests.assemble(executed_origins, tmp_path / "existing")
    with pytest.raises(FileExistsError):
        requests.assemble(executed_origins, tmp_path / "existing")
