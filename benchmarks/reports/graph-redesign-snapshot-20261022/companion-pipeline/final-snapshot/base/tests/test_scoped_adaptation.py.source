"""Opt-in normal adaptation and build cache use independent public interfaces."""
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
from test_local_scopes import FACTS, PROGRAM

from elmm_pipeline.adaptation import adapt_project
from elmm_pipeline.cli import prepare_build, scoped_configuration
from elmm_pipeline.gpu_validation import gpu_activity, native_policy_exercised


@pytest.fixture(scope="module")
def independent_scoped_pipeline(tmp_path_factory):
    directory = tmp_path_factory.mktemp("normal_scoped_pipeline")
    source_checkout = Path(os.environ.get("ELMM_TEST_COMPILER_ROOT", Path(__file__).resolve().parents[2]))
    if not (source_checkout / "compiler/__main__.py").is_file():
        pytest.skip("Set ELMM_TEST_COMPILER_ROOT to an explicit compiler checkout")
    compiler = directory / "compiler-checkout"
    pipeline = directory / "pipeline-checkout"
    ignored = shutil.ignore_patterns("__pycache__", ".*cache")
    shutil.copytree(source_checkout / "compiler", compiler / "compiler", ignore=ignored)
    shutil.copytree(Path(__file__).resolve().parents[1] / "elmm_pipeline", pipeline / "elmm_pipeline", ignore=ignored)
    return directory, pipeline, compiler


@pytest.mark.parametrize("policy", ["sections", "auto"])
def test_normal_adaptation_uses_public_source_scope_path(independent_scoped_pipeline, tmp_path, policy):
    directory, pipeline, compiler = independent_scoped_pipeline
    source = tmp_path / "application.f90"
    source.write_text(PROGRAM)
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({**FACTS, "sources": {"application.f90": hashlib.sha256(source.read_bytes()).hexdigest()}}))
    output = tmp_path / "artifacts"
    script = """import json,sys
from pathlib import Path
from unittest.mock import patch
from elmm_pipeline.adaptation import adapt_project
source,compiler,facts,output=map(Path,sys.argv[1:5])
with patch('elmm_pipeline.discovery.discover_project',side_effect=AssertionError('call discovery must not run')), \\
     patch('elmm_pipeline.gpu_toolchain.compile_gpu_sources',return_value={'public_manifest_check':True},
           side_effect=AssertionError('native fallback must not compile CUDA') if sys.argv[5]=='auto' else None):
    result=adapt_project('local',source.parent,Path(sys.executable),output,compiler_root=compiler,
                         target='gpu',memory_model='scoped',gpu_options={'gpu_policy':sys.argv[5]},
                         scope_options={'entry':'source_application::step','source':source.name,'capture_facts':facts})
print(json.dumps(result))
"""
    result = subprocess.run([sys.executable, "-c", script, str(source), str(compiler), str(facts), str(output), policy],
                            cwd=directory, env={**os.environ, "PYTHONPATH": str(pipeline)},
                            capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    adapter = json.loads(result.stdout)
    manifest = adapter["scope_manifest"]
    assert manifest["scope_count"] == (1 if policy == "sections" else 0)
    expected = [str(output / "compiler" / item["path"])
                for item in adapter["scope_manifest"]["build_sources"]
                if item["language"] == "fortran" and item["role"] != "original_source"]
    assert adapter["extra_sources"] == expected
    assert len(expected) == len(set(expected))
    assert adapter["coverage_enabled"] is False
    assert adapter["scope_extraction"]["numerical_extraction"]["entries"] == 2
    assert "--form-scopes" in adapter["scope_transformation"]["command"]
    activity = gpu_activity("", "local", adapter["regions"], scope_manifest=adapter["scope_manifest"])
    assert native_policy_exercised(activity, adapter["regions"]) is (policy == "auto")
    assert activity["verified"] is False
    if policy == "auto":
        assert adapter["native_success"] is True
        assert adapter["changed_files"] == adapter["regions"] == []
        assert adapter["gpu_sources"] == adapter["extra_sources"] == []
        assert adapter["gpu_compile_check"] is None
        assert manifest["runtime"] is None
        assert manifest["build_sources"] == manifest["source_edits"] == []
        assert source.read_text() == PROGRAM
        assert activity["offload_decisions"]["calls"] == 0
        assert "coverage unverified" in activity["native_selection_evidence"]
    else:
        assert manifest["runtime"]["link_once"] is True
        assert len(adapter["gpu_sources"]) == 3
        assert adapter["changed_files"] == ["application.f90"]
        assert "fort_scope_owner_" in source.read_text()


def test_no_approved_scope_is_a_successful_native_application(independent_scoped_pipeline, tmp_path):
    _directory, _pipeline, compiler = independent_scoped_pipeline
    source = tmp_path / "application.f90"
    source.write_text(PROGRAM)
    with patch("elmm_pipeline.gpu_toolchain.compile_gpu_sources", side_effect=AssertionError("no GPU sources")):
        metadata = adapt_project("local", tmp_path, Path(sys.executable), tmp_path / "output",
                                 compiler_root=compiler, target="gpu", memory_model="scoped",
                                 gpu_options={"gpu_policy": "sections"},
                                 scope_options={"entry": "step", "source": "application.f90",
                                                "capture_facts": {"schema_version": 1, "participation": "serial", "captures": {}}})
    assert metadata["native_success"]
    assert metadata["regions"] == []
    assert metadata["gpu_sources"] == []
    assert source.read_text() == PROGRAM
    assert metadata["rejected"]


@pytest.mark.parametrize("change", ["target", "tool", "filters", "missing_facts", "escape"])
def test_invalid_scoped_options_do_not_change_source(tmp_path, change):
    source = tmp_path / "application.f90"
    source.write_text(PROGRAM)
    options = {"entry": "step", "source": "application.f90", "capture_facts": FACTS}
    kwargs = {"target": "gpu", "memory_model": "scoped", "scope_options": options}
    tool = "local"
    if change == "target":
        kwargs["target"] = "cpu"
    elif change == "tool":
        tool = "unsupported"
    elif change == "filters":
        kwargs["include"] = ["*::step"]
    elif change == "missing_facts":
        del options["capture_facts"]
    else:
        outside = tmp_path.parent / "outside-scope.f90"
        outside.write_text(PROGRAM)
        options["source"] = "../outside-scope.f90"
    with pytest.raises(ValueError):
        adapt_project(tool, tmp_path, Path(sys.executable), tmp_path / "output", **kwargs)
    assert source.read_text() == PROGRAM


def test_scoped_build_cache_copies_facts_and_keeps_all_native_builds(tmp_path):
    for name in ("elmm/src", "PoisFFT/src", "local-compiler/compiler"):
        path = tmp_path / ".cache/repos" / name
        path.mkdir(parents=True)
        (path / "application.f90").write_text(PROGRAM)
    implementation = Path(__file__).resolve().parents[1] / "elmm_pipeline"
    shutil.copytree(implementation, tmp_path / "elmm_pipeline")
    source = tmp_path / ".cache/repos/elmm/src/application.f90"
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({**FACTS, "sources": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}}))
    args = SimpleNamespace(build_arg=[], target="gpu", tools=["baseline", "local"], gpu_toolchain={},
                           gpu_policy="auto", memory_model="scoped", scope_entry="step", scope_source="application.f90",
                           scope_source_file=[], scope_facts=facts, scope_effect_contracts=None,
                           calibration_profile=None, source_pattern=["**/*.f90"], select=[], exclude=[],
                           marked_only=False, no_coverage=False, jobs=1, threads=4)
    setup = {"pins": {}, "packages": {"local": []}, "native": {}}
    seen = []

    def adapt(_tool, project_root, _python, _output, **kwargs):
        assert kwargs["memory_model"] == "scoped"
        assert kwargs["gpu_options"]["scope_transfers"] == getattr(args, "scope_transfers", "direct")
        assert kwargs["gpu_options"]["scope_execution"] == getattr(args, "scope_execution", "bounded")
        assert kwargs["gpu_options"]["scope_measurement"] == getattr(args, "scope_measurement", None)
        copied = kwargs["scope_options"]["capture_facts"]
        assert copied != facts
        body = json.loads(copied.read_text())
        assert body["sources"] == {str(project_root / "application.f90"): hashlib.sha256(source.read_bytes()).hexdigest()}
        seen.append(copied)
        return {"tool": "local", "regions": [], "native_success": True, "changed_files": []}

    def build_(_elmm, _fft, destination, *_args, **_kwargs):
        destination.mkdir(parents=True)
        (destination / "ELMM").write_text("binary")
        (destination / "build-command.json").write_text("{}")
        return destination / "ELMM"

    with patch("elmm_pipeline.cli.ROOT", tmp_path), patch("elmm_pipeline.adaptation.adapt_project", side_effect=adapt), \
            patch("elmm_pipeline.cli.build", side_effect=build_):
        first, metadata = prepare_build("local", args, setup, {})
        cached, _ = prepare_build("local", args, setup, {})
        assert first == cached and len(seen) == 1
        facts.write_text(facts.read_text() + "\n")
        second, _ = prepare_build("local", args, setup, {})
        worker = tmp_path / "elmm_pipeline/tool_workers/scope_worker.py"
        worker.write_text(worker.read_text() + "\n# changed capture export\n")
        third, _ = prepare_build("local", args, setup, {})
        args.scope_transfers = "pinned"
        fourth, _ = prepare_build("local", args, setup, {})
        args.scope_execution = "reached"
        fifth, reached_metadata = prepare_build("local", args, setup, {})
        reached_cached, _ = prepare_build("local", args, setup, {})
        assert reached_cached == fifth
        args.gpu_policy = "sections"
        args.scope_measurement = "native"
        sixth, native_measurement = prepare_build("local", args, setup, {})
        args.scope_measurement = "resident"
        seventh, resident_measurement = prepare_build("local", args, setup, {})
    assert len(seen) == 7
    assert len({first.parent, second.parent, third.parent, fourth.parent, fifth.parent, sixth.parent, seventh.parent}) == 7
    assert metadata["identity"]["scope"]["execution"] == "bounded"
    assert reached_metadata["identity"]["scope"]["execution"] == "reached"
    assert metadata["identity"]["memory_model"] == "scoped"
    assert "scope_extraction.py" in metadata["identity"]["implementation"]
    assert native_measurement["identity"]["scope"]["measurement"] == "native"
    assert resident_measurement["identity"]["scope"]["measurement"] == "resident"
    assert "owner_measurement.py" in resident_measurement["identity"]["implementation"]


@pytest.mark.parametrize("variant", ["native", "resident"])
def test_measurement_configuration_requires_sections_reached(tmp_path, variant):
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(FACTS))
    args = SimpleNamespace(memory_model="scoped", scope_entry="step", scope_source="application.f90",
                           scope_facts=facts, target="gpu", tools=["baseline", "local"], gpu_policy="sections",
                           scope_execution="reached", scope_measurement=variant)
    assert scoped_configuration(args)["measurement"] == variant
    args.gpu_policy = "auto"
    with pytest.raises(ValueError, match="measurement requires"):
        scoped_configuration(args)
    args.gpu_policy, args.scope_execution = "sections", "bounded"
    with pytest.raises(ValueError, match="measurement requires"):
        scoped_configuration(args)
    args.scope_measurement = "unsupported"
    with pytest.raises(ValueError, match="Unknown scope measurement"):
        scoped_configuration(args)


@pytest.mark.parametrize("transfers", ["pinned", "coalesced"])
def test_scoped_configuration_requires_explicit_opt_in_and_supported_policy(tmp_path, transfers):
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(FACTS))
    args = SimpleNamespace(memory_model="call", scope_entry="step", scope_source="application.f90",
                           scope_facts=facts, target="gpu", tools=["baseline", "local"], gpu_policy="auto")
    with pytest.raises(ValueError, match="require --memory-model"):
        scoped_configuration(args)
    args.memory_model = "scoped"
    assert scoped_configuration(args)["capture_facts"] == facts
    args.scope_transfers = transfers
    assert scoped_configuration(args)["transfers"] == transfers
    args.scope_execution = "reached"
    assert scoped_configuration(args)["execution"] == "reached"
    args.memory_model = "call"
    args.scope_entry = args.scope_source = args.scope_facts = None
    with pytest.raises(ValueError, match="require --memory-model"):
        scoped_configuration(args)
    args.memory_model = "scoped"
    args.scope_entry, args.scope_source, args.scope_facts = "step", "application.f90", facts
    args.gpu_policy = "hybrid"
    with pytest.raises(ValueError, match="sections or auto"):
        scoped_configuration(args)


@pytest.mark.skipif(os.environ.get("ELMM_SCOPED_NATIVE_TESTS") != "1",
                    reason="Set ELMM_SCOPED_NATIVE_TESTS=1 for real CUDA/source-scope application builds")
@pytest.mark.parametrize("policy", ["sections", "auto"])
def test_real_normal_scopes_build_and_execute_full_fields(independent_scoped_pipeline, tmp_path, policy):
    """Exercise extraction, real CUDA checks, SCons dependencies and coherence.

    A native read/modify/write helper between GPU leaves must see the producer's
    current output, and its update must feed the GPU consumer. Two shapes and
    negative host bounds exercise complete fields and registration lifetimes.
    """
    directory, pipeline, compiler = independent_scoped_pipeline
    scons_python = Path(os.environ.get("ELMM_TEST_SCONS_PYTHON",
                                     Path(__file__).resolve().parents[1] / ".cache/envs/build/bin/python"))
    fortran = shutil.which("gfortran-15")
    cuda_host = shutil.which("g++-14")
    nvcc = shutil.which("nvcc")
    if not scons_python.is_file() or not fortran or not cuda_host or not nvcc:
        pytest.skip("Explicit SCons Python, GNU Fortran 15, g++-14 and nvcc are required")
    elmm, poisfft = tmp_path / "application", tmp_path / "PoisFFT"
    source_root = elmm / "src"
    source_root.mkdir(parents=True)
    (poisfft / "src").mkdir(parents=True)
    (poisfft / "src/poisfft.f90").touch()
    (tmp_path / "lib").mkdir()
    source = source_root / "application.f90"
    source.write_text(PROGRAM)
    (source_root / "SConstruct").write_text("""import os
env = Environment(ENV=os.environ, FORTRAN=os.environ['FC'], F90=os.environ['FC'], LINK=os.environ['FC'])
target_dir = ARGUMENTS['dir']
env.Append(FORTRANMODDIR=target_dir, F90PATH=[target_dir],
           F90FLAGS=['-O3', '-fopenmp', '-fcheck=all', '-std=f2018'], LINKFLAGS=['-fopenmp'])
objs = env.Object(target=target_dir + '/application', source='application.f90')
objs += env.Object(target=target_dir + '/main', source='main.f90')
objs = [obj for obj in objs if not str(obj).endswith(('.mod', '.smod'))]
t = env.Program(target=target_dir + '/ELMM', source=objs)
Default(t)
""")
    (source_root / "main.f90").write_text("""program check
use source_application,only:step
use omp_lib,only:omp_set_dynamic
implicit none
real(8),allocatable::a(:),b(:),c(:)
integer::trial,n,lower,i
call omp_set_dynamic(.false.)
do trial=1,2
 n=32*trial
 lower=1-3*trial
 allocate(a(lower:lower+n-1),b(lower:lower+n-1),c(lower:lower+n-1))
 do i=lower,lower+n-1
  a(i)=1+real(i,8)/4
 enddo
 b=-999
 c=-999
 call step(a,b,c,n)
 do i=lower,lower+n-1
  if(a(i)/=1+real(i,8)/4) error stop 'input changed'
 enddo
 if(any(b/=6*a)) error stop 'native helper/producer fields disagree'
 if(any(c/=7*a)) error stop 'GPU consumer missed native updates'
 deallocate(a,b,c)
enddo
print *, 'FULL_FIELDS_OK'
end program
""")
    facts = tmp_path / "captures.json"
    facts.write_text(json.dumps(FACTS))
    # The subprocess imports the independent pipeline, which in turn launches
    # its parser worker and explicit compiler checkout through public CLIs.
    script = """import json,os,sys
from pathlib import Path
from elmm_pipeline.adaptation import adapt_project
from elmm_pipeline.build import build
from elmm_pipeline.gpu_toolchain import discover_toolchain
root,compiler,scons,facts=map(Path,sys.argv[1:5])
policy=sys.argv[5]
options=discover_toolchain(languages=['cuda'],nvcc=sys.argv[6],cuda_host=sys.argv[7],arch='86')
options.update(gpu_policy=policy,host_threads=4)
if os.environ.get('ELMM_TEST_SCOPED_PROFILE'):
    options['calibration_profile']=os.environ['ELMM_TEST_SCOPED_PROFILE']
source=root/'application/src'
adapter=adapt_project('local',source,Path(sys.executable),root/'generated',compiler_root=compiler,
                      target='gpu',memory_model='scoped',defines=['DPREC'],gpu_options=options,
                      scope_options={'entry':'source_application::step','source':'application.f90',
                                     'capture_facts':facts})
binary=build(root/'application',root/'PoisFFT',root/'build',scons,root/'lib',1,4,[],
             extra_sources=[Path(path) for path in adapter['extra_sources']],
             patched_sources=[source/path for path in adapter['changed_files']],
             gpu_sources=adapter['gpu_sources'],gpu_options=options)
result={'binary':str(binary),'adapter':adapter,
        'build':json.loads((binary.parent/'build-command.json').read_text())}
(root/'integration.json').write_text(json.dumps(result,indent=2)+'\\n')
"""
    environment = {**os.environ, "PYTHONPATH": str(pipeline), "FC": fortran, "CXX": cuda_host,
                   "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}
    if os.environ.get("ELMM_TEST_SCOPED_PROFILE"):
        environment["ELMM_TEST_SCOPED_PROFILE"] = str(Path(os.environ["ELMM_TEST_SCOPED_PROFILE"]).resolve(strict=True))
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(compiler), str(scons_python),
                             str(facts), policy, nvcc, cuda_host], cwd=directory, env=environment,
                            capture_output=True, text=True, timeout=240, check=False)
    (tmp_path / "integration-build.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr + str(tmp_path / "integration-build.log")
    metadata = json.loads((tmp_path / "integration.json").read_text())
    adapter = metadata["adapter"]
    public_units = adapter["scope_manifest"]["build_sources"]
    if policy == "auto" and not os.environ.get("ELMM_TEST_SCOPED_PROFILE"):
        assert public_units == adapter["gpu_sources"] == adapter["extra_sources"] == []
        assert adapter["scope_manifest"]["scope_count"] == 0
        assert adapter["native_success"] is True
        assert "gpu" not in metadata["build"]
        assert metadata["build"]["extra_sources"] == metadata["build"]["patched_sources"] == []
        assert source.read_text() == PROGRAM
    else:
        assert sum(unit["role"] == "common_runtime" and unit["language"] == "cuda" for unit in public_units) == 1
        assert len(metadata["build"]["gpu"]["objects"]) == len(adapter["gpu_sources"])
        assert all("-arch=sm_86" in argv and argv[argv.index("-ccbin") + 1] == str(Path(cuda_host).resolve())
                   for argv in metadata["build"]["gpu"]["commands"])
    for name in ("FORT_RUNTIME_TRACE", "FORT_OFFLOAD_TRACE", "CUDA_LAUNCH_BLOCKING"):
        environment.pop(name, None)
    environment.update(FORT_RUNTIME_TRACE="1", FORT_OFFLOAD_TRACE="1")
    run = subprocess.run([metadata["binary"]], cwd=tmp_path, env=environment,
                         capture_output=True, text=True, timeout=60, check=False)
    (tmp_path / "integration-runtime.log").write_text(run.stdout + run.stderr)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "FULL_FIELDS_OK" in run.stdout
    activity = gpu_activity(run.stdout + run.stderr, "local", adapter["regions"],
                            scope_manifest=adapter["scope_manifest"])
    (tmp_path / "integration-activity.json").write_text(json.dumps(activity, indent=2) + "\n")
    if policy == "sections":
        assert activity["verified"]
        assert activity["kernel_launches"] == 4
        # A transfers once per scope. B downloads for the native helper and
        # uploads its update; C downloads at the host-visible scope boundary.
        assert activity["upload_bytes"] == activity["download_bytes"] == 2 * (32 + 64) * 8
        assert activity["scoped_runtime"]["events"]["upload"] == 4
        assert activity["scoped_runtime"]["events"]["download"] == 4
    else:
        assert native_policy_exercised(activity, adapter["regions"])
        assert activity["kernel_launches"] == activity["upload_bytes"] == activity["download_bytes"] == 0
        if os.environ.get("ELMM_TEST_SCOPED_PROFILE"):
            assert adapter["scope_manifest"]["automatic_estimate_available"] is True
            assert adapter["scope_manifest"]["scopes"][0]["estimate_available"] is True
            assert activity["offload_decisions"]["calls"] == 2
            assert activity["offload_decisions"]["native_reasons"] == {"native-scoped-scheduled": 2}
            decisions = activity["scoped_runtime"]["modeled_decisions"]
            assert len(decisions) == 2
            assert all(record["available"] and record["gpu_units"] == 0 for record in decisions)
            assert activity["scoped_runtime"]["modeled_totals"]["available_calls"] == 2
            endings = [record for record in activity["scoped_runtime"]["modeled_evidence"] if record["event"] == "end"]
            assert len(endings) == 2
            assert all(record["complete"] and not record["truncated"] for record in endings)
        else:
            assert activity["offload_decisions"]["calls"] == 0
