"""Independent source-scope integration consumes only public compiler artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from elmm_pipeline.local_adapter import (
    LocalAdapterError,
    apply_scope_sources,
    compose_scope_packages,
    transform_scopes,
)

PROGRAM="""module source_application
contains
subroutine produce(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=2*a(i)
enddo
end subroutine
subroutine native_scale(b)
real(8),intent(inout)::b(:)
b=3*b
end subroutine
subroutine consume(a,b,c,n)
real(8),intent(in)::a(:),b(:)
real(8),intent(out)::c(:)
integer,intent(in)::n
integer::i
do i=1,n
c(i)=a(i)+b(i)
enddo
end subroutine
subroutine step(a,b,c,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:),c(:)
integer,intent(in)::n
call produce(a,b,n)
call native_scale(b)
call consume(a,b,c,n)
end subroutine
end module
"""
FACT={"storage":"stable","escapes":False,"allocation_changes":False,"initialized":"whole"}
FACTS={"schema_version":1,"participation":"serial","captures":{
    "argument::a":FACT,"argument::b":{**FACT,"initialized":"none"},
    "argument::c":{**FACT,"initialized":"none"}}}


def checked(command, *, cwd, env=None):
    result=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True,timeout=180,check=False)
    assert result.returncode==0, result.stdout+result.stderr
    return result


@pytest.fixture(scope="module")
def independent(tmp_path_factory):
    directory=tmp_path_factory.mktemp("independent_scope_pipeline")
    compiler_source=Path(os.environ.get("ELMM_TEST_COMPILER_ROOT",Path(__file__).resolve().parents[2]))
    if not (compiler_source/"compiler/__main__.py").is_file():
        pytest.skip("Set ELMM_TEST_COMPILER_ROOT to a compiler source checkout")
    compiler=directory/"compiler-checkout"
    pipeline=directory/"pipeline-checkout"
    shutil.copytree(compiler_source/"compiler",compiler/"compiler",ignore=shutil.ignore_patterns("__pycache__",".*cache"))
    shutil.copytree(Path(__file__).resolve().parents[1]/"elmm_pipeline",pipeline/"elmm_pipeline",
                    ignore=shutil.ignore_patterns("__pycache__"))
    source=directory/"application-source.f90"
    source.write_text(PROGRAM)
    facts=directory/"facts.json"
    facts.write_text(json.dumps(FACTS))
    output=directory/"artifacts"
    script="""import json,sys
from pathlib import Path
from elmm_pipeline.local_adapter import transform_scopes
source,compiler,facts,output=map(Path,sys.argv[1:])
metadata=transform_scopes(source,compiler,Path(sys.executable),output,entry='step',
                          capture_facts=json.loads(facts.read_text()))
print(json.dumps(metadata))
"""
    environment={**os.environ,"PYTHONPATH":str(pipeline)}
    response=checked([sys.executable,"-c",script,str(source),str(compiler),str(facts),str(output)],
                     cwd=directory,env=environment)
    return directory,pipeline,compiler,source,output,json.loads(response.stdout)


def test_independent_checkouts_form_and_apply_public_scope(independent,tmp_path):
    directory,pipeline,_compiler,source,output,metadata=independent
    assert metadata["scope_manifest"]["scope_count"]==1
    assert metadata["scope_manifest"]["runtime"]["link_once"]
    assert len(metadata["gpu_sources"])==3 # two entries and one common runtime
    expected = [str(output / item["path"]) for item in metadata["scope_manifest"]["build_sources"]
                if item["language"] == "fortran" and item["role"] != "original_source"]
    assert metadata["extra_sources"] == expected
    assert len(expected) == len(set(expected))
    assert os.path.normpath(metadata["command"][0]) == os.path.normpath(sys.executable)
    assert "--form-scopes" in metadata["command"]
    assert "--summary-cache" not in metadata["command"]
    destination=tmp_path/"application-copy.f90"
    destination.write_bytes(source.read_bytes())
    mappings=tmp_path/"destinations.json"
    mappings.write_text(json.dumps({str(source):str(destination)}))
    script="""import json,sys
from pathlib import Path
from elmm_pipeline.local_adapter import apply_scope_sources
metadata=json.loads(Path(sys.argv[1]).read_text())
mappings={s:Path(d) for s,d in json.loads(Path(sys.argv[2]).read_text()).items()}
print(json.dumps([str(p) for p in apply_scope_sources(metadata,mappings)]))
"""
    result=checked([sys.executable,"-c",script,str(output/"scope-transformation.json"),str(mappings)],
                   cwd=directory,env={**os.environ,"PYTHONPATH":str(pipeline)})
    assert json.loads(result.stdout)==[str(destination)]
    assert metadata["scope_manifest"]["scopes"][0]["owner"] in destination.read_text()
    assert source.read_text()==PROGRAM


def test_independent_compiler_packages_compose_without_ir_or_source_merging(independent, tmp_path):
    _, _, compiler, _, _, _ = independent
    originals, destinations = [], {}
    for name in ("first", "second"):
        original = tmp_path / f"{name}_original.f90"
        original.write_text(PROGRAM.replace("source_application", f"{name}_application"))
        destination = tmp_path / f"{name}_application.f90"
        destination.write_bytes(original.read_bytes())
        originals.append(original)
        destinations[str(original)] = destination
    packages = [transform_scopes(source, compiler, Path(sys.executable), tmp_path / f"artifacts_{index}",
                                 entry=f"{name}_application::step", source_files=[originals[1-index]],
                                 capture_facts=FACTS)
                for index, (name, source) in enumerate(zip(("first", "second"), originals))]
    bundle = compose_scope_packages(packages, destinations)
    assert len(bundle["patched_sources"]) == 2
    assert all(package["scope_manifest"]["scopes"][0]["owner"] in destinations[str(source)].read_text()
               for package, source in zip(packages, originals))
    common = [item for item in bundle["build_artifacts"] if item["role"] == "common_runtime"]
    assert len(common) == 3 and all(item["packages"] == [0, 1] for item in common)
    assert len(bundle["gpu_sources"]) == 5


@pytest.mark.parametrize("execution", ["bounded", "reached"])
def test_automatic_missing_profile_composes_authenticated_native_package(independent, tmp_path, execution):
    _, _, compiler, source, _, _ = independent
    output = tmp_path / "native-artifacts"
    with pytest.raises(LocalAdapterError, match="Public source scopes require graph execution"):
        transform_scopes(source, compiler, Path(sys.executable), output, entry="step", capture_facts=FACTS,
                         gpu_options={"gpu_policy": "auto", "scope_execution": execution})
    assert not output.exists()
    metadata = transform_scopes(source, compiler, Path(sys.executable), output, entry="step",
                                capture_facts=FACTS, gpu_options={"gpu_policy": "auto", "scope_execution": "graph"})
    manifest = metadata["scope_manifest"]
    assert metadata["compiler_report"]["supported"] is True
    assert metadata["compiler_report"]["outputs"] == ["scope-manifest.json"]
    assert metadata["changed"] is False
    assert metadata["gpu_sources"] == metadata["extra_sources"] == []
    assert manifest["schema_version"] == 1
    assert manifest["scope_execution"] == "graph"
    assert manifest["scope_count"] == 0
    assert manifest["scopes"] == manifest["source_edits"] == manifest["build_sources"] == []
    assert manifest["sources"] == manifest["artifacts_sha256"] == {}
    assert manifest["runtime"] is None
    assert manifest["automatic_scope_available"] is manifest["automatic_estimate_available"] is False
    assert manifest["numerical_decisions"] == []
    assert manifest["native_effects"]["closure_materialization"] == "not_requested"
    assert manifest["source_inputs"] == {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    facts = {**FACTS, "sources": manifest["source_inputs"]}
    assert manifest["capture_facts_sha256"] == hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
    assert any("calibration" in boundary["reason"] for boundary in manifest["boundaries"])
    assert json.loads((output / "scope-manifest.json").read_text()) == manifest

    destination = tmp_path / "native-application.f90"
    destination.write_bytes(source.read_bytes())
    bundle = compose_scope_packages([metadata], {str(source): destination})
    assert bundle["runtime"] is None
    assert bundle["patched_sources"] == bundle["gpu_sources"] == bundle["extra_sources"] == []
    assert bundle["build_artifacts"] == []
    assert destination.read_text() == source.read_text() == PROGRAM


@pytest.mark.parametrize("transfers", ["pinned", "pipelined", "coalesced"])
def test_independent_adapter_transports_transfer_choice_and_compiler_reason(independent, tmp_path, transfers):
    _, _, compiler, source, _, _ = independent
    metadata = transform_scopes(source, compiler, Path(sys.executable), tmp_path / "generated", entry="step",
                                capture_facts=FACTS, gpu_options={"scope_transfers": transfers})
    command = metadata["command"]
    assert command[command.index("--scope-transfers") + 1] == transfers
    scope, = metadata["scope_manifest"]["scopes"]
    configuration = scope["transfer_configuration"]
    assert configuration["requested"] == transfers
    assert configuration["selected"] == {"pinned": "pinned", "pipelined": "direct", "coalesced": "runtime"}[transfers]
    assert configuration["reason"] == ("pipelined_not_available" if transfers == "pipelined" else None)
    if transfers == "coalesced":
        assert configuration["coalesced"]["fallback_reason"] == "coalesced_upload_ineligible"
        assert "exact direct" in configuration["coalesced"]["placement"]
    assert metadata["compiler_report"]["scopes"] == metadata["scope_manifest"]


def test_independent_adapter_transports_reached_execution_without_choosing_segments(independent,tmp_path):
    _, _, compiler, source, _, _ = independent
    with pytest.raises(LocalAdapterError, match="Public source scopes require graph execution"):
        transform_scopes(source,compiler,Path(sys.executable),tmp_path/'retired-reached',entry='step',
                         capture_facts=FACTS,gpu_options={'scope_execution':'reached'})
    assert not (tmp_path/'retired-reached').exists()
    metadata = transform_scopes(source,compiler,Path(sys.executable),tmp_path/'reached',entry='step',
                                capture_facts=FACTS,gpu_options={'scope_execution':'graph'})
    argv = metadata['command']
    assert argv[argv.index('--scope-execution')+1] == 'graph'
    manifest = metadata['scope_manifest']
    assert manifest['scope_execution'] == 'graph'
    assert manifest['scope_count'] == 1
    assert manifest['scopes'][0]['native_continuation'].startswith('original lexical scope')
    plan = manifest['scopes'][0]['checked_execution_plan']
    assert plan['schema_version'] == 1
    assert len(plan['execution_identity']) == 64
    assert plan['entry'] == manifest['entry'] == 'source_application::step'
    assert plan['sources'] == manifest['source_inputs']
    owner = manifest['scopes'][0]
    public_members = {plan['entry']: owner, **{item['procedure']: item for item in
                      [*owner['module_coordinators'], *owner['internal_coordinators']]}}
    assert set(public_members) == {item['procedure'] for item in plan['procedures']}
    assert any(item['units'] for item in plan['procedures'])
    for member in plan['procedures']:
        assert [unit['operations'] for unit in member['units']] == [
            unit['operations'] for unit in public_members[member['procedure']]['planning_segments']]
    assert metadata['compiler_report']['scopes'] == manifest


def test_independent_adapter_reuses_summary_cache_and_invalidates_changed_sources(independent, tmp_path):
    _, _, compiler, _, _, _ = independent
    source = tmp_path / "cached_application.f90"
    source.write_text(PROGRAM)
    cache = tmp_path / "new summary cache"

    def native_effects():
        # Graph admission deliberately avoids materializing a transitive
        # closure. Exercise disk reuse through the existing public complete
        # effects query, then keep graph transport and artifact checks below.
        response = checked([sys.executable, "-m", "compiler", "--analyze-effects", "--kernel", "step",
                            "--input", str(source), "--summary-cache", str(cache), "--json"],
                           cwd=compiler, env={**os.environ, "PYTHONPATH": str(compiler)})
        return json.loads(response.stdout)["effects"]

    cold_effects, warm_effects = native_effects(), native_effects()
    assert cold_effects["summary_cache"]["disk_writes"] > 0
    assert warm_effects["summary_cache"]["disk_hits"] > 0
    packages = [transform_scopes(source, compiler, Path(sys.executable), tmp_path / f"generated_{index}",
                                 entry="step", capture_facts=FACTS, summary_cache=cache)
                for index in range(2)]
    cold, warm = (package["scope_manifest"] for package in packages)
    for package in packages:
        command = package["command"]
        assert command[command.index("--summary-cache") + 1] == str(cache.resolve())
        assert package["compiler_report"]["scopes"] == package["scope_manifest"]
        assert package["scope_manifest"]["native_effects"]["summary_cache"]["disk_writes"] == 0
    assert cold["scope_count"] == warm["scope_count"] == 1
    assert cold["scopes"][0]["gpu_leaves"] == warm["scopes"][0]["gpu_leaves"]
    assert warm["native_effects"]["closure_materialization"] == "not_requested"
    kernels = [{unit["path"]: manifest["artifacts_sha256"][unit["path"]]
                for unit in manifest["build_sources"] if unit["language"] == "cuda"}
               for manifest in (cold, warm)]
    assert kernels[0] == kernels[1]
    records = set(cache.glob("*.json"))
    assert records
    changed = PROGRAM.replace("b(i)=2*a(i)", "b(i)=5*a(i)")
    source.write_text(changed)
    changed_effects = native_effects()
    assert changed_effects["summary_cache"]["disk_writes"] > 0
    refreshed = transform_scopes(source, compiler, Path(sys.executable), tmp_path / "changed",
                                 entry="step", capture_facts=FACTS, summary_cache=cache)["scope_manifest"]
    assert refreshed["scope_count"] == 1
    assert refreshed["source_inputs"] == {str(source): hashlib.sha256(changed.encode()).hexdigest()}
    assert refreshed["source_inputs"] != warm["source_inputs"]
    refreshed_kernels = {unit["path"]: refreshed["artifacts_sha256"][unit["path"]]
                         for unit in refreshed["build_sources"] if unit["language"] == "cuda"}
    assert refreshed_kernels != kernels[1]
    assert records < set(cache.glob("*.json"))
    assert source.read_text() == changed


def test_independent_adapter_accepts_original_descriptor_guards(independent, tmp_path):
    _, _, compiler, source, _, _ = independent
    expected = {"lower_bounds": [1], "extents": [8]}
    facts = {**FACTS, "captures": {**FACTS["captures"],
             "argument::a": {**FACTS["captures"]["argument::a"], "expected_descriptor": expected}}}
    metadata = transform_scopes(source, compiler, Path(sys.executable), tmp_path / "guarded",
                                entry="step", capture_facts=facts,
                                gpu_options={"scope_execution": "graph"})
    manifest = metadata["scope_manifest"]
    assert manifest["scope_count"] == 1
    assert manifest["capture_descriptor_expectations"]["schema_version"] == 1
    resource = next(row for row in manifest["scopes"][0]["resource_bindings"]["resources"]
                    if row["resource"] == "argument::a")
    assert resource["expected_descriptor"] == expected
    assert resource["descriptor_expectation_guard"]["definition_authority"] is False
    assert source.read_text() == PROGRAM
    destination = tmp_path / "guarded-application.f90"
    destination.write_text(PROGRAM)
    assert apply_scope_sources(metadata, {str(source): destination}) == [destination]
    assert source.read_text() == PROGRAM


def test_stale_application_source_is_rejected_before_any_edit(independent,tmp_path):
    *_,source,_output,metadata=independent
    destination=tmp_path/"application.f90"
    destination.write_text(PROGRAM+"! changed\n")
    before=destination.read_bytes()
    with pytest.raises(LocalAdapterError,match="Application source changed"):
        apply_scope_sources(metadata,{str(source):destination})
    assert destination.read_bytes()==before


def test_modified_replacement_is_rejected_before_any_edit(independent,tmp_path):
    *_,source,output,metadata=independent
    copied=tmp_path/"artifacts"
    shutil.copytree(output,copied)
    metadata={**metadata,"artifact_root":str(copied)}
    replacement=copied/metadata["scope_manifest"]["sources"][str(source)]["replacement"]
    replacement.write_text(replacement.read_text()+"! modified\n")
    destination=tmp_path/"application.f90"
    destination.write_text(PROGRAM)
    with pytest.raises(LocalAdapterError,match="Approved scope artifact changed"):
        apply_scope_sources(metadata,{str(source):destination})
    assert destination.read_text()==PROGRAM


@pytest.mark.parametrize("change",["abi","runtime_identity","build_role","escape"])
def test_incompatible_public_build_manifest_is_rejected(independent,tmp_path,change):
    _,_,compiler,source,_output,metadata=independent
    report=json.loads(json.dumps(metadata["compiler_report"]))
    manifest=report["scopes"]
    if change=="abi":
        manifest["runtime"]["abi_version"]=999
    elif change=="runtime_identity":
        manifest["runtime"]["runtime_id"]="0"*64
    elif change=="build_role":
        manifest["build_sources"][0]["role"]="unknown"
    else:
        manifest["artifacts_sha256"]["../outside.f90"]=hashlib.sha256(b"outside").hexdigest()
    with patch("elmm_pipeline.local_adapter._invoke",return_value=(report,[])),pytest.raises(LocalAdapterError):
        transform_scopes(source,compiler,Path(sys.executable),tmp_path/"output",entry="step",capture_facts=FACTS)


def test_pipeline_builds_and_links_public_common_runtime_once(independent,tmp_path):
    directory,pipeline,_compiler,source,output,metadata=independent
    fortran=shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc=shutil.which("nvcc")
    if not nvcc or not fortran:
        pytest.skip("CUDA and GNU Fortran toolchain required")
    destination=tmp_path/"application.f90"
    destination.write_text(PROGRAM)
    apply_scope_sources(metadata,{str(source):destination})
    script="""import json,sys
from pathlib import Path
from elmm_pipeline.gpu_toolchain import compile_gpu_sources
metadata=json.loads(Path(sys.argv[1]).read_text())
print(json.dumps(compile_gpu_sources(metadata['gpu_sources'],Path(sys.argv[2]),{'architecture':'86'})))
"""
    result=checked([sys.executable,"-c",script,str(output/"scope-transformation.json"),str(tmp_path)],
                   cwd=directory,env={**os.environ,"PYTHONPATH":str(pipeline)})
    linked=json.loads(result.stdout)
    assert len(linked["objects"])==3
    objects=[]
    # The common modules precede interfaces which USE them. SCons discovers
    # those dependencies in production; this manual linker fixture uses the
    # compiler's public roles instead of assuming an incidental artifact order.
    units = sorted((item for item in metadata["scope_manifest"]["build_sources"]
                    if item["language"] == "fortran" and item["role"] != "original_source"),
                   key=lambda item: item["role"] != "common_runtime")
    for index,item in enumerate(units):
        filename = str(output / item["path"])
        target=tmp_path/(str(index)+"_"+Path(filename).stem+".o")
        checked([fortran,"-std=f2018","-fopenmp","-c",filename,"-o",str(target)],cwd=tmp_path)
        objects.append(str(target))
    driver=tmp_path/"driver.f90"
    driver.write_text("""program caller
use source_application,only:step
use fort_scoped_memory,only:fort_scope_serial_caller
real(8)::a(-2:29),b(-2:29),c(-2:29)
if(fort_scope_serial_caller()/=1) error stop 'runtime was built without OpenMP caller detection'
a=4
call step(a,b,c,32)
if(any(b/=24) .or. any(c/=28)) error stop 'complete fields disagree'
print *, 'FIELDS_OK'
end program
""")
    target=tmp_path/"caller"
    arguments=[fortran,"-std=f2018","-fopenmp","-fcheck=all",str(destination),str(driver),
               *objects,*linked["objects"],*["-L"+p for p in linked["library_dirs"]],
               *["-Wl,-rpath,"+p for p in linked["rpaths"]],*["-l"+n for n in linked["libraries"]],
               "-o",str(target)]
    checked(arguments,cwd=tmp_path)
    result=checked([str(target)],cwd=tmp_path,
                   env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"4"})
    assert "FIELDS_OK" in result.stdout
