"""Collective source authority crosses independent public interfaces unchanged."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from elmm_pipeline.local_adapter import (
    LocalAdapterError,
    apply_scope_sources,
    remap_scope_facts,
    transform_scopes,
)

SOURCE = """module application
contains
subroutine produce(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i
!$omp do
do i=1,n
b(i)=2*a(i)
end do
!$omp end do
end subroutine
subroutine native_scale(b,n)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i
!$omp do
do i=1,n
b(i)=3*b(i)
end do
!$omp end do
end subroutine
subroutine consume(b,c,n)
real(8),intent(in)::b(:)
real(8),intent(inout)::c(:)
integer,intent(in)::n
integer::i
!$omp do
do i=1,n
c(i)=b(i)+1.d0
end do
!$omp end do
end subroutine
subroutine step(a,b,c,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:)
integer,intent(in)::n
call produce(a,b,n)
call native_scale(b,n)
call consume(b,c,n)
end subroutine
subroutine caller(a,b,c,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:)
integer,intent(in)::n
!$omp parallel default(none) shared(a,b,c,n) num_threads(4)
call step(a,b,c,n)
!$omp end parallel
end subroutine
end module
"""


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def facts_for(source):
    lines = source.read_text().splitlines(keepends=True)
    first = next(i for i, line in enumerate(lines, 1) if line.startswith("call step("))
    team_first, team_last = first - 1, first + 1
    stable = {"storage": "stable", "escapes": False, "allocation_changes": False, "initialized": "whole"}
    return {"schema_version": 2, "sources": {str(source): digest(source.read_text())},
            "captures": {**{"argument::" + name: {**stable, "association": "shared_whole_storage",
                                                  "descriptor_uniform": True} for name in ("a", "b", "c")},
                         "argument::n": {**stable, "association": "shared_immutable_control"}},
            "participation": {"kind": "omp_full_team", "dispatch": "qualified_companion",
                              "entry": "application::step", "expected_omp_level": 1, "host_threads": 4,
                              "call_sites": [{"source": str(source), "source_sha256": digest(source.read_text()),
                                              "caller": "application::caller", "first_line": first, "last_line": first,
                                              "team_first_line": team_first, "team_last_line": team_last,
                                              "span_sha256": digest(lines[first-1]),
                                              "team_span_sha256": digest("".join(lines[team_first-1:team_last])),
                                              "uniform_guard": "unconditional"}]}}


FAKE_COMPILER = r'''
import hashlib,json,sys
from pathlib import Path
def value(flag):
 return sys.argv[sys.argv.index(flag)+1]
def digest(text):
 return hashlib.sha256(text.encode()).hexdigest()
out=Path(value('--output-dir'));out.mkdir(parents=True,exist_ok=True)
source=Path(value('--input'))
facts=json.loads(Path(value('--scope-facts')).read_text())
collective=facts.get('schema_version')==2
assert ('--gpu-collective' in sys.argv)==collective
mode=Path(__file__).with_name('mode.json')
mode=json.loads(mode.read_text()) if mode.exists() else {}
analysis=json.loads(Path(value('--analysis-sources')).read_text()) if '--analysis-sources' in sys.argv else None
normalized=json.loads(Path(value('--numerical-sources')).read_text()) if '--numerical-sources' in sys.argv else {'entries':[]}
artifacts={'scoped_runtime.cu':'PUBLIC ARTIFACT: deliberately opaque to the adapter\n',
           'sources/replacement.f90':source.read_text()+'! public approved replacement\n'}
native=mode.get('native',False)
if native:artifacts={}
hashes={name:digest(text) for name,text in artifacts.items()}
runtime_hashes={'scoped_runtime.cu':hashes['scoped_runtime.cu']} if not native else {}
manifest={'schema_version':1,'scope_execution':value('--scope-execution'),'scope_count':0 if native else 1,
 'scopes':[] if native else [{'owner':'qualified_companion','path':str(source),'first_line':1,'last_line':1,'gpu_leaves':['compiler_decision']}],
 'source_inputs':facts['sources'],'analysis_sources':analysis,
 'numerical_sources':sorted([{key:item[key] for key in ('procedure','path','entry','sha256')} for item in normalized['entries']],key=lambda item:item['procedure']),
 'capture_facts_sha256':digest(json.dumps(facts,sort_keys=True)),
 'artifacts_sha256':hashes,'sources':{} if native else {str(source):{'sha256':facts['sources'][str(source)],'replacement':'sources/replacement.f90'}},
 'runtime':None if native else {'schema_version':1,'abi_version':1,'link_once':True,'source_sha256':runtime_hashes,
  'sources':[{'path':'scoped_runtime.cu','language':'cuda'}],
  'runtime_id':digest('\n'.join(f'{name}:{value}' for name,value in sorted(runtime_hashes.items())))},
 'build_sources':[] if native else [{'path':'scoped_runtime.cu','language':'cuda','role':'common_runtime'},
 {'path':'sources/replacement.f90','language':'fortran','role':'original_source'}]}
if 'descriptor_contract' in mode:
 manifest['capture_descriptor_expectations']=mode['descriptor_contract']
if collective:
 proof={**facts['participation'],'schema_version':2,'other_callers':'unchanged'}
 proof=json.loads(json.dumps(proof))
 if mode.get('mismatch'):proof['call_sites'][0][mode['mismatch']]='contradictory'
 if mode.get('missing_proof'):proof=None
 manifest['participation']=proof
for name,text in artifacts.items():
 path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
(out/'scope-manifest.json').write_text(json.dumps(manifest))
print(json.dumps({'supported':True,'scopes':manifest,'outputs':[*artifacts,'scope-manifest.json']}))
'''


@pytest.fixture
def fixture(tmp_path):
    compiler = tmp_path / "independent-compiler"
    (compiler / "compiler").mkdir(parents=True)
    (compiler / "compiler/__init__.py").touch()
    (compiler / "compiler/__main__.py").write_text(FAKE_COMPILER)
    source = tmp_path / "application.f90"
    source.write_text(SOURCE)
    return compiler, source, facts_for(source)


def test_explicit_full_team_facts_pass_only_through_public_cli(fixture, tmp_path):
    compiler, source, facts = fixture
    metadata = transform_scopes(source, compiler, Path(sys.executable), tmp_path / "output",
                                entry="application::step", capture_facts=facts,
                                gpu_options={"host_threads": 4, "gpu_policy": "sections"})
    assert "--gpu-collective" in metadata["command"]
    assert metadata["scope_manifest"]["participation"]["schema_version"] == 2
    assert metadata["scope_manifest"]["scopes"][0]["gpu_leaves"] == ["compiler_decision"]
    transported = json.loads((tmp_path / "output/scope-captures.json").read_text())
    assert transported == facts
    assert source.read_text() == SOURCE
    apply_scope_sources(metadata, {str(source): source})
    assert source.read_text() == SOURCE + "! public approved replacement\n"


def test_serial_facts_never_enable_collective_from_options(fixture, tmp_path):
    compiler, source, facts = fixture
    facts = {**facts, "schema_version": 1, "participation": "serial"}
    metadata = transform_scopes(source, compiler, Path(sys.executable), tmp_path / "serial",
                                entry="application::step", capture_facts=facts,
                                gpu_options={"gpu_collective": True})
    assert "--gpu-collective" not in metadata["command"]


DESCRIPTOR_CONTRACT = {
    "schema_version": 1, "enforcement": "original_descriptor_before_registration",
    "definition_authority": False,
    "supported_consumers": ["original_lexical", "original_module_companion"],
    "unsupported_consumers": ["synthetic_owner", "qualified_collective_owner"],
}


@pytest.mark.parametrize("change", ["missing", "boolean_version", "new_version", "enforcement",
                                    "definition_authority", "supported_consumers", "unsupported_consumers"])
def test_descriptor_expectations_require_public_enforcement_before_application_edit(fixture, tmp_path, change):
    compiler, source, facts = fixture
    facts = {**facts, "schema_version": 1, "participation": "serial"}
    facts["captures"]["argument::a"]["expected_descriptor"] = {"lower_bounds": [-3], "extents": [8]}
    contract = {**DESCRIPTOR_CONTRACT}
    if change == "boolean_version":
        contract["schema_version"] = True
    elif change == "new_version":
        contract["schema_version"] = 2
    elif change == "enforcement":
        contract["enforcement"] = "unchecked"
    elif change == "definition_authority":
        contract["definition_authority"] = True
    elif change == "supported_consumers":
        contract["supported_consumers"] = ["original_lexical"]
    elif change == "unsupported_consumers":
        contract["unsupported_consumers"] = []
    mode = {} if change == "missing" else {"descriptor_contract": contract}
    (compiler / "compiler/mode.json").write_text(json.dumps(mode))
    output = tmp_path / change
    with pytest.raises(LocalAdapterError, match="descriptor expectation enforcement"):
        transform_scopes(source, compiler, Path(sys.executable), output,
                         entry="application::step", capture_facts=facts)
    assert source.read_text() == SOURCE
    assert not (output / "scope-transformation.json").exists()


def test_descriptor_expectations_transport_only_with_confirmed_public_enforcement(fixture, tmp_path):
    compiler, source, facts = fixture
    facts = {**facts, "schema_version": 1, "participation": "serial"}
    facts["captures"]["argument::a"]["expected_descriptor"] = {"lower_bounds": [-3], "extents": [8]}
    (compiler / "compiler/mode.json").write_text(json.dumps({"descriptor_contract": DESCRIPTOR_CONTRACT}))
    output = tmp_path / "descriptor"
    metadata = transform_scopes(source, compiler, Path(sys.executable), output,
                                entry="application::step", capture_facts=facts)
    assert metadata["scope_manifest"]["capture_descriptor_expectations"] == DESCRIPTOR_CONTRACT
    assert json.loads((output / "scope-captures.json").read_text()) == facts
    assert source.read_text() == SOURCE
    assert apply_scope_sources(metadata, {str(source): source}) == [source]


def test_native_report_needs_no_descriptor_enforcement_capability(fixture, tmp_path):
    compiler, source, facts = fixture
    facts = {**facts, "schema_version": 1, "participation": "serial"}
    facts["captures"]["argument::a"]["expected_descriptor"] = {"lower_bounds": [-3], "extents": [8]}
    (compiler / "compiler/mode.json").write_text(json.dumps({"native": True}))
    metadata = transform_scopes(source, compiler, Path(sys.executable), tmp_path / "native-descriptor",
                                entry="application::step", capture_facts=facts)
    assert not metadata["changed"]
    assert metadata["scope_manifest"]["scope_count"] == 0
    assert apply_scope_sources(metadata, {str(source): source}) == []
    assert source.read_text() == SOURCE


@pytest.mark.parametrize("change", ["threads", "conditional", "missing_sites", "unknown_source", "bad_span", "unqualified", "bad_native_source"])
def test_partial_or_mismatched_collective_facts_do_not_invoke_compiler(fixture, tmp_path, change):
    compiler, source, facts = fixture
    facts = json.loads(json.dumps(facts))
    participation = facts["participation"]
    site = participation["call_sites"][0]
    if change == "threads":
        participation["host_threads"] = 3
    elif change == "conditional":
        site["uniform_guard"] = "if(flag)"
    elif change == "missing_sites":
        participation["call_sites"] = []
    elif change == "unknown_source":
        site["source"] = "prepared-analysis.f90"
    elif change == "bad_span":
        site["first_line"] = True
    elif change == "unqualified":
        participation["entry"] = "step"
    else:
        facts["native_participation"] = {"application::native_scale": {"source": "unknown.f90"}}
    with patch("elmm_pipeline.local_adapter._invoke", side_effect=AssertionError("proof is incomplete")), \
            pytest.raises(LocalAdapterError):
        transform_scopes(source, compiler, Path(sys.executable), tmp_path / change,
                         entry="application::step", capture_facts=facts)
    assert source.read_text() == SOURCE


@pytest.mark.parametrize("field", ["caller", "span_sha256", "team_span_sha256", "source_sha256"])
def test_mismatched_public_caller_proof_is_rejected_before_application_edit(fixture, tmp_path, field):
    compiler, source, facts = fixture
    (compiler / "compiler/mode.json").write_text(json.dumps({"mismatch": field}))
    with pytest.raises(LocalAdapterError, match="caller evidence"):
        transform_scopes(source, compiler, Path(sys.executable), tmp_path / field,
                         entry="application::step", capture_facts=facts)
    assert source.read_text() == SOURCE


def test_successful_all_native_collective_report_keeps_participation_proof(fixture, tmp_path):
    compiler, source, facts = fixture
    (compiler / "compiler/mode.json").write_text(json.dumps({"native": True}))
    metadata = transform_scopes(source, compiler, Path(sys.executable), tmp_path / "native",
                                entry="application::step", capture_facts=facts)
    assert metadata["scope_manifest"]["scope_count"] == 0
    assert metadata["scope_manifest"]["participation"]["call_sites"] == facts["participation"]["call_sites"]
    assert not metadata["changed"]
    assert metadata["gpu_sources"] == []
    assert apply_scope_sources(metadata, {str(source): source}) == []


def test_old_compiler_without_public_participation_proof_is_rejected(fixture, tmp_path):
    compiler, source, facts = fixture
    (compiler / "compiler/mode.json").write_text(json.dumps({"missing_proof": True}))
    with pytest.raises(LocalAdapterError, match="versioned participation proof"):
        transform_scopes(source, compiler, Path(sys.executable), tmp_path / "old",
                         entry="application::step", capture_facts=facts)


def test_build_copy_remaps_declared_paths_without_mutating_any_proof(fixture, tmp_path):
    _compiler, source, facts = fixture
    facts["native_participation"] = {"application::native_scale": {
        "source": source.name, "source_sha256": facts["sources"][str(source)],
        "kind": "existing_team_worksharing", "completion": "all_participants_before_effect_commit"}}
    facts["sources"] = {source.name: facts["sources"][str(source)]}
    facts["participation"]["call_sites"][0]["source"] = source.name
    before = json.dumps(facts, sort_keys=True)
    copied = tmp_path / "copied-application"
    mapped = remap_scope_facts(facts, source.parent, copied)
    expected = str(copied / source.name)
    assert list(mapped["sources"]) == [expected]
    assert mapped["participation"]["call_sites"][0]["source"] == expected
    assert mapped["native_participation"]["application::native_scale"]["source"] == expected
    assert mapped["participation"]["call_sites"][0]["span_sha256"] == facts["participation"]["call_sites"][0]["span_sha256"]
    assert mapped["native_participation"]["application::native_scale"]["kind"] == "existing_team_worksharing"
    assert json.dumps(facts, sort_keys=True) == before


def test_declared_caller_path_cannot_escape_original_application_tree(fixture, tmp_path):
    _compiler, source, facts = fixture
    facts["participation"]["call_sites"][0]["source"] = "../outside.f90"
    with pytest.raises(LocalAdapterError, match="escapes"):
        remap_scope_facts(facts, source.parent)


def test_independent_pipeline_checkout_consumes_public_compiler_json(fixture, tmp_path):
    compiler, source, facts = fixture
    pipeline = tmp_path / "independent-pipeline"
    shutil.copytree(Path(__file__).resolve().parents[1] / "elmm_pipeline", pipeline / "elmm_pipeline",
                    ignore=shutil.ignore_patterns("__pycache__"))
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts))
    script = """import json,sys
from pathlib import Path
from elmm_pipeline.local_adapter import transform_scopes
source,compiler,facts,output=map(Path,sys.argv[1:])
metadata=transform_scopes(source,compiler,Path(sys.executable),output,entry='application::step',
                          capture_facts=json.loads(facts.read_text()))
print(json.dumps(metadata))
"""
    response = subprocess.run([sys.executable, "-c", script, str(source), str(compiler), str(facts_path), str(tmp_path / "isolated")],
                              cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(pipeline)},
                              text=True, capture_output=True, timeout=30, check=False)
    assert response.returncode == 0, response.stdout + response.stderr
    metadata = json.loads(response.stdout)
    assert metadata["scope_manifest"]["participation"]["schema_version"] == 2
    assert "--gpu-collective" in metadata["command"]
    assert source.read_text() == SOURCE


def test_isolated_extraction_worker_keeps_original_caller_spans(fixture, tmp_path):
    compiler, source, facts = fixture
    pipeline = tmp_path / "isolated-pipeline"
    shutil.copytree(Path(__file__).resolve().parents[1] / "elmm_pipeline", pipeline / "elmm_pipeline",
                    ignore=shutil.ignore_patterns("__pycache__"))
    facts["sources"] = {source.name: facts["sources"][str(source)]}
    facts["participation"]["call_sites"][0]["source"] = source.name
    facts_path = tmp_path / "relative-facts.json"
    facts_path.write_text(json.dumps(facts))
    script = """import json,sys
from pathlib import Path
from unittest.mock import patch
from elmm_pipeline.adaptation import adapt_project
source,compiler,facts,output=map(Path,sys.argv[1:])
with patch('elmm_pipeline.gpu_toolchain.compile_gpu_sources',return_value={'public_build_inputs_only':True}):
 result=adapt_project('local',source.parent,Path(sys.executable),output,compiler_root=compiler,
                     target='gpu',memory_model='scoped',gpu_options={'gpu_policy':'sections','host_threads':4},
                     scope_options={'entry':'application::step','source':source.name,'capture_facts':facts})
print(json.dumps(result))
"""
    response = subprocess.run([sys.executable, "-c", script, str(source), str(compiler), str(facts_path), str(tmp_path / "extracted")],
                              cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(pipeline)},
                              text=True, capture_output=True, timeout=30, check=False)
    assert response.returncode == 0, response.stdout + response.stderr
    adapted = json.loads(response.stdout)
    manifest = adapted["scope_manifest"]
    proof = manifest["participation"]["call_sites"][0]
    assert proof["source"] == str(source)
    assert proof["first_line"] == facts["participation"]["call_sites"][0]["first_line"]
    assert proof["span_sha256"] == facts["participation"]["call_sites"][0]["span_sha256"]
    assert manifest["analysis_sources"]["source_inputs"] == {str(source): digest(SOURCE)}
    assert manifest["analysis_sources"]["entries"][0]["source"] == str(source)
    assert "--analysis-sources" in adapted["scope_transformation"]["command"]
    assert "--numerical-sources" in adapted["scope_transformation"]["command"]
    assert "--gpu-collective" in adapted["scope_transformation"]["command"]


def test_build_cache_copy_remaps_collective_caller_and_native_assertions(tmp_path):
    from elmm_pipeline.cli import prepare_build

    for name in ("elmm/src", "PoisFFT/src", "local-compiler/compiler"):
        path = tmp_path / ".cache/repos" / name
        path.mkdir(parents=True)
        (path / "application.f90").write_text(SOURCE)
    implementation = Path(__file__).resolve().parents[1] / "elmm_pipeline"
    shutil.copytree(implementation, tmp_path / "elmm_pipeline", ignore=shutil.ignore_patterns("__pycache__"))
    source = tmp_path / ".cache/repos/elmm/src/application.f90"
    facts = facts_for(source)
    facts["native_participation"] = {"application::native_scale": {
        "source": str(source), "source_sha256": digest(SOURCE), "kind": "existing_team_worksharing",
        "completion": "all_participants_before_effect_commit"}}
    facts_path = tmp_path / "build-facts.json"
    facts_path.write_text(json.dumps(facts))
    args = SimpleNamespace(build_arg=[], target="gpu", tools=["baseline", "local"], gpu_toolchain={},
                           gpu_policy="sections", memory_model="scoped", scope_entry="application::step",
                           scope_source="application.f90", scope_source_file=[], scope_facts=facts_path,
                           scope_effect_contracts=None, calibration_profile=None, source_pattern=["**/*.f90"],
                           select=[], exclude=[], marked_only=False, no_coverage=False, jobs=1, threads=4)
    seen = []

    def adapt(_tool, project_root, _python, _output, **kwargs):
        body = json.loads(kwargs["scope_options"]["capture_facts"].read_text())
        expected = str(project_root / source.name)
        assert body["sources"] == {expected: digest(SOURCE)}
        assert body["participation"]["call_sites"][0]["source"] == expected
        assert body["native_participation"]["application::native_scale"]["source"] == expected
        assert body["participation"]["call_sites"][0]["span_sha256"] == facts["participation"]["call_sites"][0]["span_sha256"]
        seen.append(body)
        return {"tool": "local", "regions": [], "native_success": True, "changed_files": []}

    def build(_elmm, _fft, destination, *_args, **_kwargs):
        destination.mkdir(parents=True)
        (destination / "ELMM").write_text("fake binary")
        (destination / "build-command.json").write_text("{}")
        return destination / "ELMM"

    setup = {"pins": {}, "packages": {"local": []}, "native": {}}
    with patch("elmm_pipeline.cli.ROOT", tmp_path), patch("elmm_pipeline.adaptation.adapt_project", side_effect=adapt), \
            patch("elmm_pipeline.cli.build", side_effect=build):
        first, _ = prepare_build("local", args, setup, {})
        second, _ = prepare_build("local", args, setup, {})
    assert first == second
    assert len(seen) == 1
    assert json.loads(facts_path.read_text()) == facts


@pytest.mark.skipif(os.environ.get("ELMM_COLLECTIVE_CLI_TESTS") != "1",
                    reason="Enable after the selected compiler collective source interface is frozen")
def test_independent_real_public_cli_preserves_configured_original_provenance(tmp_path):
    """Generate public artifacts only; no native or CUDA compilation/execution."""
    compiler_root = Path(os.environ.get("ELMM_TEST_COMPILER_ROOT", Path(__file__).resolve().parents[2]))
    compiler = tmp_path / "compiler-checkout"
    pipeline = tmp_path / "pipeline-checkout"
    shutil.copytree(compiler_root / "compiler", compiler / "compiler", ignore=shutil.ignore_patterns("__pycache__", ".*cache"))
    shutil.copytree(Path(__file__).resolve().parents[1] / "elmm_pipeline", pipeline / "elmm_pipeline",
                    ignore=shutil.ignore_patterns("__pycache__"))
    source = tmp_path / "application.f90"
    source.write_text(SOURCE.replace("contains", '#include "configured.inc"\ncontains', 1))
    (tmp_path / "configured.inc").write_text("#ifdef REAL64\ninteger,parameter::wp=8\n#else\ninteger,parameter::wp=4\n#endif\n")
    facts = facts_for(source)
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts))
    script = """import json,sys
from pathlib import Path
from unittest.mock import patch
from elmm_pipeline.adaptation import adapt_project
source,compiler,facts,output=map(Path,sys.argv[1:])
with patch('elmm_pipeline.gpu_toolchain.compile_gpu_sources',return_value={'native_build_deferred':True}):
 result=adapt_project('local',source.parent,Path(sys.executable),output,compiler_root=compiler,
                     target='gpu',memory_model='scoped',defines=['REAL64'],
                     gpu_options={'gpu_policy':'sections','host_threads':4},
                     scope_options={'entry':'application::step','source':source.name,'capture_facts':facts})
print(json.dumps(result))
"""
    response = subprocess.run([sys.executable, "-c", script, str(source), str(compiler), str(facts_path), str(tmp_path / "artifacts")],
                              cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(pipeline)},
                              text=True, capture_output=True, timeout=60, check=False)
    assert response.returncode == 0, response.stdout + response.stderr
    adapted = json.loads(response.stdout)
    manifest = adapted["scope_manifest"]
    assert manifest["schema_version"] == 1
    assert manifest["participation"]["schema_version"] == 2
    site = manifest["participation"]["call_sites"][0]
    assert site["source"] == str(source)
    assert site["caller"] == "application::caller"
    assert site["first_line"] == facts["participation"]["call_sites"][0]["first_line"]
    assert site["span_sha256"] == facts["participation"]["call_sites"][0]["span_sha256"]
    assert str(tmp_path / "configured.inc") in manifest["analysis_sources"]["dependencies"]
    assert "--gpu-collective" in adapted["scope_transformation"]["command"]
    assert "#include" in source.read_text()
    assert any(edit["source"] == str(source) for edit in manifest["source_edits"])
