"""Evidence boundaries in the fresh three-way application comparison."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("reproduce_e2e", HERE / "reproduce_e2e.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_cli_defaults_to_pinned_sgs_control_and_preserves_explicit_override(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run_comparison", calls.append)
    runner.main(["--output-dir", str(tmp_path / "default")])
    assert calls[-1].committed_ref == "be58ab809296b97ea39d701828a2621adfc6b3dc"
    runner.main(["--committed-ref", "explicit-compatible-revision", "--output-dir", str(tmp_path / "override")])
    assert calls[-1].committed_ref == "explicit-compatible-revision"


def test_application_root_preparation_reaches_the_same_strict_source_guard(tmp_path, monkeypatch):
    compare = runner.run_comparison
    arguments, preparations, guarded_roots = [], [], []
    monkeypatch.setattr(runner, "run_comparison", arguments.append)
    explicit = tmp_path / "approved-checkouts"
    runner.main(["--output-dir", str(tmp_path / "default")])
    runner.main(["--app-root", str(explicit), "--output-dir", str(tmp_path / "explicit")])

    def prepare(upstream, destination):
        preparations.append((upstream, destination))
        return destination

    class SourceGuardReached(Exception):
        pass

    def verify(root, contract):
        guarded_roots.append(root)
        raise SourceGuardReached

    monkeypatch.setattr(runner, "prepare_application_basis", prepare)
    monkeypatch.setattr(runner, "verify_contract", verify)
    for args in arguments:
        with pytest.raises(SourceGuardReached):
            compare(args)
        assert not args.output_dir.exists()
    assert preparations == [(runner.PIPELINE / ".cache/repos", tmp_path / "default-application-basis")]
    assert guarded_roots == [tmp_path / "default-application-basis", explicit]


def test_rounds_rotate_all_three_modes_fairly():
    orders = [runner.rotated_order(number) for number in range(3)]
    assert all(set(order) == set(runner.MODES) for order in orders)
    assert {order[0] for order in orders} == set(runner.MODES)


def test_diagnostic_compile_and_relink_keep_the_public_240_second_cap(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / "elmm-pipeline"))
    import elmm_pipeline.build_receipts as receipts

    source = tmp_path / "original-main.f90"
    original = "program main\n  call Dynamics_Deallocate\n  call DeallocateGlobals\nend program\n"
    source.write_text(original)
    binary, obj = tmp_path / "ELMM", tmp_path / "main.o"
    binary.write_bytes(b"production executable")
    obj.write_bytes(b"production object")
    recipe = {"units": [{"source": str(source), "object": str(obj), "execution": {
        "argv": ["gfortran", "-c", str(source), "-o", str(obj)], "cwd": str(tmp_path), "environment": {}}}],
        "link": {"argv": ["gfortran", str(obj), "-o", str(binary)],
                 "cwd": str(tmp_path), "environment": {}}}
    (tmp_path / "build-receipts.json").write_text(json.dumps(recipe))
    calls = []

    def execute(_recorder, argv, **kwargs):
        calls.append((argv, kwargs["timeout"]))
        Path(argv[argv.index("-o") + 1]).write_bytes(b"private diagnostic artifact")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(receipts, "require_build_receipts", lambda _path: recipe)
    monkeypatch.setattr(receipts, "read_receipts", lambda _path: [])
    monkeypatch.setattr(receipts.CommandRecorder, "run", execute)
    result = runner.relink_diagnostic(binary, source, tmp_path / "diagnostic")
    assert len(calls) == 2 and all(timeout == 240 for _argv, timeout in calls)
    assert Path(result["binary"]).parent == tmp_path / "diagnostic"
    assert source.read_text() == original and binary.read_bytes() == b"production executable"


def test_diagnostic_and_warmup_cannot_improve_timing_summary():
    rows = [{"mode": mode, "warmup": False, "traced": False, "full_storage": False,
             "wall_seconds": seconds} for mode, seconds in zip(runner.MODES, (10., 8., 5.))]
    rows += [{"mode": mode, "warmup": True, "traced": True, "full_storage": True,
              "wall_seconds": .001} for mode in runner.MODES]
    summary = runner.summarize(rows)
    assert summary["reworked"]["speedup_over_native"] == 2.
    assert summary["reworked"]["wall_reduction_percent"] == 50.
    assert summary["reworked"]["speedup_over_committed"] == 1.6
    rows[0]["full_storage"] = True
    with pytest.raises(ValueError, match="production"):
        runner.summarize(rows)
    rows[0]["full_storage"] = False
    rows[0]["traced"] = True
    with pytest.raises(ValueError, match="production"):
        runner.summarize(rows)


def test_exact_committed_snapshot_ignores_modified_worktree(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / "elmm-pipeline"))
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    compiler = repository / "compiler"
    compiler.mkdir()
    (compiler / "__main__.py").write_text("committed = True\n")
    subprocess.run(["git", "-C", str(repository), "add", "compiler"], check=True)
    subprocess.run(["git", "-C", str(repository), "-c", "user.name=Benchmark Test", "-c",
                    "user.email=benchmark@example.invalid", "commit", "-qm", "baseline"], check=True)
    (compiler / "__main__.py").write_text("dirty = True\n")
    (compiler / "untracked.py").write_text("also_dirty = True\n")
    committed = tmp_path / "committed"
    receipt = runner.snapshot_compiler(repository, committed, revision="HEAD")
    assert (committed / "compiler/__main__.py").read_text() == "committed = True\n"
    assert not (committed / "compiler/untracked.py").exists()
    assert receipt["exact_committed_snapshot"]
    candidate = tmp_path / "candidate"
    runner.snapshot_compiler(repository, candidate)
    assert (candidate / "compiler/untracked.py").is_file()
    assert (candidate / "compiler/__main__.py").read_text() == "dirty = True\n"


def test_new_snapshots_exclude_scratch_without_redefining_historical_hashes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / "elmm-pipeline"))
    from elmm_pipeline.common import digest_tree
    source = tmp_path / "source"
    compiler = source / "compiler"
    (compiler / "_local").mkdir(parents=True)
    (compiler / ".ruff_cache").mkdir()
    (compiler / "__main__.py").write_text("source = True\n")
    scratch = compiler / "_local" / "report.json"
    scratch.write_text("old measurement\n")
    (compiler / ".ruff_cache" / "cache").write_text("cache\n")
    historical = digest_tree(compiler)
    equivalence = runner.compiler_code_equivalence(compiler)
    scratch.write_text("new measurement\n")
    assert digest_tree(compiler) != historical
    assert runner.compiler_code_equivalence(compiler) == equivalence
    snapshot = tmp_path / "snapshot"
    record = runner.snapshot_compiler(source, snapshot)
    assert not (snapshot / "compiler/_local").exists()
    assert not (snapshot / "compiler/.ruff_cache").exists()
    assert record["code_equivalence"] == equivalence
    assert record["compiler_tree_sha256"] == digest_tree(snapshot / "compiler")
    (compiler / "__main__.py").write_text("source = False\n")
    assert runner.compiler_code_equivalence(compiler)["sha256"] != equivalence["sha256"]


def test_generic_candidate_request_keeps_host_source_and_graph_options():
    control = {"source": "elmm/src/sgs.f90", "entry": "sgs::step", "source_sha256": {
        "elmm/src/sgs.f90": "hash", "elmm/src/types.f90": "hash"}, "capture_facts": {"proof": "retained"}}
    custom = {"schema_version": 1, "source": "elmm/src/dynamics.f90", "entry": "dynamics::step",
              "source_files": ["elmm/src/sgs.f90"], "capture_facts": {"proof": "caller"},
              "options": {"scope_execution": "graph", "gpu_policy": "sections"}}
    selected = runner.selection_request(control, custom)
    assert selected["source"] == custom["source"]
    assert selected["capture_facts"] == custom["capture_facts"]
    assert selected["options"]["scope_execution"] == "graph"
    assert runner.selection_request(control)["options"]["scope_execution"] == "bounded"
    with pytest.raises(ValueError, match="schema_version"):
        runner.selection_request(control, {"source": "elmm/src/unreviewed.f90"})


def test_multiple_candidate_roots_preserve_each_checked_request():
    contract = {"source": "elmm/src/subgrid.f90", "entry": "subgrid::sgs", "source_sha256": {},
                "capture_facts": {"checked": "sgs"}}
    graph = {"schema_version": 1, "source": "elmm/src/dynamics.f90", "entry": "dynamics::convection",
             "capture_facts": {"checked": "convection"}, "options": {"scope_execution": "graph"}}
    requests = runner.selection_requests(contract, [None, graph])
    assert [item["entry"] for item in requests] == ["subgrid::sgs", "dynamics::convection"]
    assert requests[0]["capture_facts"] == {"checked": "sgs"}
    assert requests[1]["capture_facts"] == {"checked": "convection"}
    assert runner.selection_requests(contract, graph) == [runner.selection_request(contract, graph)]
    assert runner.selection_requests(contract, scope_execution="graph")[0]["options"]["scope_execution"] == "graph"
    with pytest.raises(ValueError, match="empty"):
        runner.selection_requests(contract, [])
    with pytest.raises(ValueError, match="repeat"):
        runner.selection_requests(contract, [graph, graph])


def test_generation_transport_keeps_original_proof_sidecars_and_rejects_tampering(tmp_path):
    packages = []
    for number in range(2):
        directory = tmp_path / str(number)
        directory.mkdir()
        manifest = {"authority": {"source": str(number), "proof": ["immutable"] * 100}}
        (directory / "scope-manifest.json").write_text(json.dumps(manifest))
        packages.append({"artifact_root": str(directory), "scope_manifest": manifest})
    generation = {"reworked": {**packages[0], "scope_packages": packages}}
    transported = runner.generation_references(generation)
    assert "scope_manifest" not in transported["reworked"]
    assert all("scope_manifest" not in item for item in transported["reworked"]["scope_packages"])
    restored = runner.load_generation(transported)
    assert restored["reworked"]["scope_manifest"] == generation["reworked"]["scope_manifest"]
    assert [item["scope_manifest"] for item in restored["reworked"]["scope_packages"]] == [
        item["scope_manifest"] for item in packages]
    (tmp_path / "1/scope-manifest.json").write_text('{"authority":"changed"}')
    with pytest.raises(ValueError, match="sidecar"):
        runner.load_generation(transported)
    with pytest.raises(ValueError, match="sidecar"):
        runner.generation_references(transported)


def test_candidate_calibration_binds_one_external_profile_without_changing_controls(tmp_path):
    profile = tmp_path / 'hardware.json'
    document = {'schema_version':1,'cpu_threads':4,'precision_bits':64,
                'hardware':{'gpu_uuid':'GPU-test'},'toolchain':{'cuda_runtime_version':13040},
                'numerical_contract':{'identity':'separate-arithmetic'},
                'scoped':{'schema_version':1,'runtime_id':'a'*64}}
    profile.write_text(json.dumps(document))
    original = [{'entry':entry,'options':{'gpu_policy':'auto'}} for entry in ('first','second')]
    selected, identity = runner.candidate_calibration(original,profile,threads=4)
    assert all(row['options']['calibration_profile'] == str(profile.resolve()) for row in selected)
    assert all('calibration_profile' not in row['options'] for row in original)
    assert identity['sha256'] == runner.digest(profile) and identity['candidate_only']
    assert identity['scoped_runtime_id'] == 'a'*64
    other = tmp_path / 'other.json'
    other.write_text(profile.read_text())
    conflicting = [{**original[0],'options':{**original[0]['options'],'calibration_profile':str(other)}}]
    with pytest.raises(ValueError,match='conflicting'):
        runner.candidate_calibration(conflicting,profile,threads=4)
    with pytest.raises(ValueError,match='thread'):
        runner.candidate_calibration(original,profile,threads=2)
    profile.write_text(json.dumps(document,indent=2))
    _, changed = runner.candidate_calibration(original,profile,threads=4)
    assert changed['sha256'] != identity['sha256']


def test_calibration_reuse_checks_gpu_uuid_and_cpu_identity(monkeypatch):
    hardware = {'gpu_uuid':'GPU-test','gpu_name':'Test GPU','compute_capability':'8.6',
                'cpu_name':'Test CPU','machine':runner.os.uname().machine}
    identity = {'hardware':hardware,'sha256':'profile-hash','scoped_runtime_id':'runtime'}
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *_args,**_kwargs:'GPU-test, Test GPU, 8.6\n')
    original = Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda path,*args,**kwargs:'model name: Test CPU\n'
                        if str(path)=='/proc/cpuinfo' else original(path,*args,**kwargs))
    observed = runner.calibration_hardware_binding(identity,{'device':{'argv':['nvidia-smi']}})
    assert observed['cpu_name'] == 'Test CPU' and observed['devices'][0]['gpu_uuid'] == 'GPU-test'
    identity['hardware'] = {**hardware,'gpu_uuid':'different GPU'}
    with pytest.raises(ValueError,match='GPU UUID'):
        runner.calibration_hardware_binding(identity,{'device':{'argv':['nvidia-smi']}})
    identity['hardware'] = {**hardware,'cpu_name':'different CPU'}
    with pytest.raises(ValueError,match='CPU/machine'):
        runner.calibration_hardware_binding(identity,{'device':{'argv':['nvidia-smi']}})


def test_all_native_auto_needs_explicit_runtime_decisions_or_preexecution_declines():
    requests = [{'options':{'gpu_policy':'auto'}}]
    activity = {'runtime_cuda_submissions':0,'scoped_runtime':{'modeled_decisions':[
        {'gpu_units':0,'cpu_units':3,'reason':'native_scope_path_cheaper'}]}}
    selection = runner.native_auto_selection(activity,requests)
    assert selection and selection['gpu_execution_observed'] is False
    assert selection['offloading_performance_achievement'] is False
    assert runner.native_auto_selection(activity,[{'options':{'gpu_policy':'sections'}}]) is None
    assert runner.native_auto_selection({**activity,'runtime_cuda_submissions':1},requests) is None
    assert runner.native_auto_selection({'runtime_cuda_submissions':0,'scoped_runtime':{}},requests) is None
    decline = {'runtime_cuda_submissions':0,'scoped_runtime':{'modeled_evidence':[
        {'event':'definition_validation','status':7,'reason':'unavailable_definition'}]}}
    assert runner.native_auto_selection(decline,requests)['preexecution_declines']


def test_composed_sources_keep_component_proofs_and_authenticate_before_writes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / "elmm-pipeline"))
    import hashlib
    def sha(content):
        return hashlib.sha256(content.encode()).hexdigest()
    original = tmp_path / "application.f90"
    original.write_text("original\n")
    source, replacement = str(original), "originals/application.f90"
    unit = {"path": "runtime.cu", "language": "cuda", "role": "common_runtime"}
    packages = []
    for number, content in enumerate(("first\noriginal\n", "original\nsecond\n")):
        directory = tmp_path / str(number)
        (directory / "originals").mkdir(parents=True)
        (directory / replacement).write_text(content)
        (directory / "runtime.cu").write_text("shared runtime\n")
        component = {"schema_version": 1, "scopes": [], "entry": f"root{number}",
                     "source_inputs": {source: sha("original\n")},
                     "sources": {source: {"sha256": sha("original\n"), "replacement": replacement}},
                     "artifacts_sha256": {replacement: sha(content), "runtime.cu": sha("shared runtime\n")},
                     "build_sources": [unit], "runtime": {"identity": "same checked runtime"}}
        (directory / "scope-manifest.json").write_text(json.dumps(component))
        packages.append({"scope_manifest": component, "artifact_root": str(directory)})
    merged = tmp_path / "composed"
    (merged / "originals").mkdir(parents=True)
    merged_content = "first\noriginal\nsecond\n"
    (merged / replacement).write_text(merged_content)
    (merged / "runtime.cu").write_text("shared runtime\n")
    aggregate = {**packages[0]["scope_manifest"], "entry": ["root0", "root1"],
                 "component_manifests": [item["scope_manifest"] for item in packages],
                 "artifacts_sha256": {replacement: sha(merged_content), "runtime.cu": sha("shared runtime\n")}}
    (merged / "scope-manifest.json").write_text(json.dumps(aggregate))
    manifest, prepared = runner.verify_composed_sources(packages, merged, {source: original})
    assert manifest == aggregate and prepared == [(original, merged_content.encode())]
    assert original.read_text() == "original\n"
    (merged / "runtime.cu").write_text("tampered\n")
    with pytest.raises(ValueError, match="composed artifact changed"):
        runner.verify_composed_sources(packages, merged, {source: original})
    assert original.read_text() == "original\n"


def test_prepared_builds_authenticate_sources_and_never_import_timings(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / "elmm-pipeline"))
    import elmm_pipeline.build_receipts as receipts
    from elmm_pipeline.common import digest_tree
    main = tmp_path / 'main.f90'
    main.write_text('program main\n  call Dynamics_Deallocate\n  call DeallocateGlobals\nend program\n')
    original = {'repositories': {'elmm': {'tracked_sha256': {'src/main.f90': runner.digest(main)}}}}
    expected = {'contract_sha256':'contract', 'original_application':original, 'selection_requests':{}, 'compilers':{}}
    prepared = {'schema_version':1, 'kind':'prepared-elmm-comparison-builds', 'field_policy':runner.FIELD_POLICY,
                'threads':4, 'build_args':[], 'builds':{}, 'diagnostic_builds':{}, 'generation':{},
                'runs':[{'wall_seconds':.00001}], **expected}
    inventories = {}
    for mode in runner.MODES:
        directory = tmp_path / mode
        directory.mkdir()
        binary = directory / 'ELMM'
        binary.write_bytes(b'fresh compiled executable')
        receipt_path = directory / 'build-receipts.json'
        receipt_path.write_text('{}')
        source = directory / 'diagnostic.f90'
        source.write_text(runner.diagnostic_main(main.read_text()))
        prepared['builds'][mode] = {'binary':str(binary), 'sha256':runner.digest(binary),
            'build_receipt':str(receipt_path), 'build_receipt_sha256':runner.digest(receipt_path)}
        prepared['diagnostic_builds'][mode] = {'binary':str(binary), 'sha256':runner.digest(binary),
            'source':str(source), 'source_sha256':runner.digest(source), 'original_build_receipt':str(receipt_path),
            'original_build_receipt_sha256':runner.digest(receipt_path)}
        inventory = {'binary':str(binary), 'units':[{'source':str(main)}],
                     'source_inputs':{str(main):runner.digest(main)}, 'files_sha256':{str(main):runner.digest(main)}}
        inventories[str(receipt_path)] = inventory
        if mode == 'native':
            continue
        (directory / 'compiler').mkdir()
        (directory / 'compiler/__main__.py').write_text('checked compiler\n')
        expected['compilers'][mode] = {'root':str(directory), 'compiler_tree_sha256':digest_tree(directory / 'compiler'),
                                       'base_commit':'commit', 'exact_committed_snapshot':mode == 'committed'}
        kernel = directory / 'kernel.cu'
        kernel.write_text('checked CUDA source\n')
        manifest = {'artifacts_sha256':{'kernel.cu':runner.digest(kernel)}}
        (directory / 'scope-manifest.json').write_text(json.dumps(manifest))
        package = {'artifact_root':str(directory), 'scope_manifest':manifest}
        prepared['generation'][mode] = {**package, 'scope_packages':[package], 'gpu_sources':[{'source':str(kernel)}]}
        inventory['units'].append({'source':str(kernel)})
        inventory['source_inputs'][str(kernel)] = runner.digest(kernel)
    monkeypatch.setattr(receipts, 'require_build_receipts', lambda name: inventories[str(name)])
    monkeypatch.setattr(receipts, 'read_receipts', lambda _directory: [])
    selected, identities = runner.validate_prepared_builds(prepared, expected, threads=4, build_args=[])
    assert set(selected) == {'builds', 'diagnostic_builds', 'generation'}
    assert 'runs' not in selected and identities[str(main)] == runner.digest(main)
    with pytest.raises(ValueError, match='build configuration'):
        runner.validate_prepared_builds(prepared, expected, threads=8, build_args=[])
    with pytest.raises(ValueError, match='build configuration'):
        runner.validate_prepared_builds(prepared,{**expected,'calibration_profile':{'sha256':'changed profile'}},threads=4,build_args=[])
    (tmp_path / 'reworked/kernel.cu').write_text('stale CUDA source\n')
    with pytest.raises(ValueError, match='compiler artifact changed'):
        runner.validate_prepared_builds(prepared, expected, threads=4, build_args=[])


def test_natural_evidence_preserves_repeatability_failures_and_checks_vtk_bytes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / "elmm-pipeline"))
    import struct

    from elmm_pipeline import validation
    from elmm_pipeline.campaigns.halos import compare_fields, fields

    monkeypatch.setattr(validation, 'read_vtk', lambda _path: ((1, 1, 1), {'u':[1., 0., 0.], 'p':[0.]}))
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    (config_dir / 'main.conf').write_text('4\n')
    configuration = {'expected_max_steps':None, 'runtime_arguments':["output_dir='output/'"]}
    expected = {'original_application':{'pins':'pristine'}, 'contract_sha256':'contract',
                'builds':{}, 'diagnostic_builds':{}}
    for mode in ('native', 'committed'):
        expected['builds'][mode] = {'sha256':'production', 'build_receipt_sha256':'receipt'}
        expected['diagnostic_builds'][mode] = {'binary':'/fresh/' + mode + '/ELMM', 'sha256':'diagnostic'}
    source_report = tmp_path / 'natural-report.json'
    report = {'cases':{'channel':{'config_sha256':{'main.conf':runner.digest(config_dir / 'main.conf')}, 'runs':{}}}}
    selected = {'config_dir':str(config_dir), 'config_sha256':report['cases']['channel']['config_sha256'], 'runs':{}}
    reference_fields = None
    reference_vtk = None
    for mode, value in (('native', 1.), ('native-repeat', 1.001), ('committed', 1.002)):
        directory = tmp_path / mode
        (directory / 'output').mkdir(parents=True)
        (directory / 'output/out.vtk').write_bytes(b'original VTK bytes')
        for name in (*runner.REQUIRED_FIELDS, *runner.OPTIONAL_FIELDS):
            (directory / f'fields-{name}.bin').write_bytes(struct.pack('d', value))
            (directory / f'fields-{name}.shape').write_text('64 1 1 1\n')
        log = 'tstep: 1\nTime limit reached.\nTotal wall clock time for time steps: 0.1\nsaved.\n'
        (directory / 'run.log').write_text(log)
        observed = fields(directory, {**runner.FIELD_POLICY, 'field_count':20})
        vtk = runner.read_outputs(directory / 'output', validation.read_vtk)
        if mode == 'native':
            reference_fields, reference_vtk = observed, vtk
        comparison = compare_fields(observed, reference_fields, runner.FIELD_POLICY)
        vtk_comparison = runner.compare_outputs(reference_vtk, vtk, validation.compare_vtk)
        build_mode = 'native' if mode == 'native-repeat' else mode
        row = {'natural_duration':True, 'completed_original_configuration':True, 'binary_sha256':'diagnostic',
               'production_binary_sha256':'production', 'build_receipt_sha256':'receipt', 'field_count':20,
               'argv':[expected['diagnostic_builds'][build_mode]['binary'], *configuration['runtime_arguments']],
               'environment':{'OMP_NUM_THREADS':'4', 'OMP_DYNAMIC':'FALSE', 'OMP_SCHEDULE':'static',
                              'OMP_PROC_BIND':'false', 'CUDA_VISIBLE_DEVICES':'' if mode.startswith('native') else None},
               'timing':validation.validate_log(log), 'fields':comparison, 'vtk':vtk_comparison,
               'passed':all(field['passed'] for field in comparison.values()),
               'diagnostic_wall_seconds':.0001,
               'field_sha256':{path.name:runner.digest(path) for path in directory.glob('fields-*')}}
        (directory / 'run.json').write_text(json.dumps(row))
        report['cases']['channel']['runs'][mode] = row
        artifacts = [*directory.glob('fields-*'), directory / 'output/out.vtk', directory / 'run.log', directory / 'run.json']
        selected['runs'][mode] = {'directory':str(directory), 'run':row,
            'artifacts_sha256':{str(path.relative_to(directory)):runner.digest(path) for path in artifacts}}
    source_report.write_text(json.dumps(report))
    evidence = {'schema_version':1, 'kind':'fresh-session-natural-controls', 'threads':4,
                'field_policy':runner.FIELD_POLICY, 'original_application':expected['original_application'],
                'contract_sha256':'contract', 'source_report':{'path':str(source_report), 'sha256':runner.digest(source_report)},
                'cases':{'channel':selected}}
    reference, _fields, rows, identities = runner.validate_control_evidence(
        evidence, expected, case='channel', configuration=configuration, configuration_dir=config_dir, threads=4)
    assert reference['steps'] == 1 and [row['passed'] for row in rows] == [True, False, False]
    assert all('diagnostic_wall_seconds' not in row for row in rows)
    assert rows[1]['float64_storage_validation']['fields'] == report['cases']['channel']['runs']['native-repeat']['fields']
    assert str(tmp_path / 'committed/output/out.vtk') in identities
    (tmp_path / 'committed/output/out.vtk').write_bytes(b'changed VTK bytes')
    with pytest.raises(ValueError, match='natural control artifact changed'):
        runner.validate_control_evidence(evidence, expected, case='channel', configuration=configuration,
                                         configuration_dir=config_dir, threads=4)


def test_complete_fields_are_dumped_before_their_original_retirement():
    original = "program main\n  call Dynamics_Deallocate\n  call DeallocateGlobals\nend program\n"
    transformed = runner.diagnostic_main(original)
    for name in runner.REQUIRED_FIELDS:
        assert f"fields-{name}.bin" in transformed
        assert transformed.index(f"fields-{name}.bin") < transformed.index("call Dynamics_Deallocate")
    assert transformed.index("fields-U_avg.bin") < transformed.index("call DeallocateGlobals")
    assert "storage_size" in transformed and "lbound" in transformed and "ubound" in transformed
    assert "shape(" in transformed
    with pytest.raises(ValueError, match="lifetime boundaries"):
        runner.diagnostic_main("program unsupported\nend program\n")


def test_report_never_fabricates_counts_for_absent_trace():
    markdown = runner.markdown_report({"status": "failed", "gpu_evidence": {}})
    assert "committed | 0" not in markdown
    assert "inferred" in markdown


def test_custom_configuration_cannot_reuse_stale_output(tmp_path):
    configuration = tmp_path / 'configuration'
    (configuration / 'output').mkdir(parents=True)
    (configuration / 'main.conf').write_text('2\n')
    (configuration / 'output/out.vtk').write_text('stale reference\n')
    destination = runner.run_directory(configuration, tmp_path / 'run')
    assert (destination / 'main.conf').read_text() == '2\n'
    assert not (destination / 'output').exists()
    assert (configuration / 'output/out.vtk').is_file()


def test_cuda_api_activity_counts_observed_calls_without_device_time(tmp_path):
    import sqlite3
    database = tmp_path / 'trace.sqlite'
    with sqlite3.connect(database) as connection:
        connection.executescript('''
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES(1, 'cudaLaunchKernel'), (2, 'cudaMemcpyAsync');
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(nameId INTEGER, start INTEGER, end INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(1, 0, 100), (1, 200, 400), (2, 500, 550);
        ''')
    result = runner.cuda_api_activity(database)
    assert result['CUPTI_ACTIVITY_KIND_RUNTIME'] == [
        {'name':'cudaLaunchKernel', 'calls':2, 'cpu_api_seconds':3e-7},
        {'name':'cudaMemcpyAsync', 'calls':1, 'cpu_api_seconds':5e-8}]
    assert 'kernel_seconds' not in result
    assert 'CUPTI_ACTIVITY_KIND_DRIVER' not in result


def test_nsight_transport_records_graph_nodes_without_changing_application_args(tmp_path):
    profiler = tmp_path / 'nsys'
    profiler.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    profiler.chmod(0o755)
    wrapper, identity = runner.node_profile_transport(str(profiler),tmp_path/'transport')
    args = ['profile','--trace=cuda','application','output_dir=output/']
    observed = json.loads(subprocess.check_output([wrapper,*args],text=True))
    assert observed == ['profile','--cuda-graph-trace=node',*args[1:]]
    assert identity['profiler_sha256'] == runner.digest(profiler)
    assert identity['wrapper_sha256'] == runner.digest(wrapper)
    assert json.loads(subprocess.check_output([wrapper,'--version'],text=True)) == ['--version']


def test_performance_cap_preserves_natural_fourth_order_filter_and_wall_inputs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / 'elmm-pipeline'))
    source = tmp_path / 'channel'
    source.mkdir()
    (source / 'main.conf').write_text('! original fourth order\n4\n')
    (source / 'time_stepping.conf').write_text('time_stepping(\n end_time = 50,\n)\n')
    (source / 'filter.conf').write_text('filtertype=4\n')
    (source / 'wall.conf').write_text('wallmodeltype=2\n')
    originals = {path.name:path.read_bytes() for path in source.iterdir()}
    correctness, timing, timing_dir = runner.prepare_workloads(source, tmp_path / 'out', timing_steps=100)
    assert correctness['expected_max_steps'] is None
    assert correctness['overrides'] == {}
    assert timing['expected_max_steps'] == 100
    assert timing['overrides'] == {'max_number_of_time_steps':100}
    assert correctness['advection_method'] == timing['advection_method'] == 4
    assert timing_dir == tmp_path / 'out/timing-config'
    for name, original in originals.items():
        assert (tmp_path / 'out/config' / name).read_bytes() == original
        if name != 'time_stepping.conf':
            assert (timing_dir / name).read_bytes() == original
    assert {path.name:path.read_bytes() for path in source.iterdir()} == originals
    with pytest.raises(ValueError, match='performance-only'):
        runner.prepare_workloads(source, tmp_path / 'invalid', steps=2, timing_steps=100)


def test_completion_statistics_are_validated_and_retained(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / 'elmm-pipeline'))
    import elmm_pipeline.gpu_validation as validation
    evidence = {'schema_version':1,'context':3,'event':'completion_statistics','stats_version':1,
                'complete':True,'event_records':2,'event_queries':0,'event_waits':1,'stream_waits':0,
                'created_events':1,'reused_events':1,'independent_host_tasks':7}
    observed = []
    def activity(log, *_args, **_kwargs):
        observed.append(log)
        return {'verified':True}
    monkeypatch.setattr(validation, 'gpu_activity', activity)
    log = 'FORT_SCOPED launch bytes=0\nFORT_SCOPED evidence ' + json.dumps(evidence) + '\n'
    result = runner.observed_gpu_activity(log, scope_manifest={})
    assert result['completion_statistics'] == [evidence]
    assert 'FORT_SCOPED launch bytes=0' in observed[0]
    assert 'completion_statistics' not in observed[0]
    evidence['event_records'] = -1
    with pytest.raises(ValueError, match='runtime counters'):
        runner.observed_gpu_activity('FORT_SCOPED evidence '+json.dumps(evidence),scope_manifest={})


def test_generic_policy_alias_requires_authenticated_generated_call(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / 'elmm-pipeline'))
    kernel = tmp_path / 'worker.cu'
    kernel.write_text('offload::decision_trace("generic_label", "native-scoped-scheduled", 0, 1);\n')
    package = {'artifact_root': str(tmp_path), 'scope_manifest': {
        'schema_version': 1, 'scope_count': 0, 'scopes': [],
        'artifacts_sha256': {kernel.name: runner.digest(kernel)}}}
    log = 'FORT_OFFLOAD entry=generic_label mode=native-scoped-scheduled gpu_units=0 cpu_units=1 unit_kind=regions\n'
    result = runner.observed_gpu_activity(log, scope_packages=[package])
    assert result['offload_decisions']['modes']['native'] == 1
    assert result['runtime_cuda_submissions'] == 0
    assert result['policy_trace_aliases']['artifacts']['generic_label'][0]['sha256'] == runner.digest(kernel)
    with pytest.raises(ValueError, match='absent from the build manifest'):
        runner.observed_gpu_activity(log.replace('generic_label', 'unknown'), scope_packages=[package])
    with pytest.raises(ValueError, match='absent from the build manifest'):
        runner.observed_gpu_activity(log, scope_manifest={})
    kernel.write_text('offload::decision_trace("unverified", "native-scoped-scheduled", 0, 1);\n')
    with pytest.raises(ValueError, match='policy trace artifact changed'):
        runner.observed_gpu_activity(log, scope_packages=[package])


def test_runtime_activity_sidecar_preserves_full_evidence_and_checks_summary(tmp_path):
    decision = {'available': 1, 'gpu_units': 0, 'cpu_units': 2, 'reason': 'native_is_cheaper'}
    decline = {'event': 'definition_validation', 'status': 7, 'reason': 'missing_definition', 'context': 4}
    activity = {'verified': False, 'runtime_cuda_submissions': 0, 'upload_bytes': 0,
        'scoped_runtime': {'modeled_decisions': [decision, decision],
            'modeled_evidence': [decline, {'event': 'end', 'complete': False}],
            'modeled_totals': {'gpu_units': 0, 'cpu_units': 4}, 'source_provenance': {'available': True}},
        'native_auto_selection_evidence': {'policy': 'explicit generic AUTO',
            'modeled_native_decisions': [decision, decision], 'preexecution_declines': [decline],
            'gpu_execution_observed': False, 'offloading_performance_achievement': False}}
    original = json.loads(json.dumps(activity))
    transported = runner.runtime_activity_reference(activity, tmp_path / 'trace')
    assert activity == original
    assert 'modeled_evidence' not in transported['scoped_runtime']
    summary = transported['scoped_runtime']['modeled_decision_summary']
    assert summary['calls'] == 2 and summary['selections'][0]['occurrences'] == 2
    assert transported['scoped_runtime']['modeled_evidence_summary']['incomplete_end_records'] == 1
    assert transported['scoped_runtime']['source_provenance'] == activity['scoped_runtime']['source_provenance']
    assert transported['native_auto_selection_evidence']['modeled_native_decision_count'] == 2
    assert transported['native_auto_selection_evidence']['offloading_performance_achievement'] is False
    assert runner.load_runtime_activity(transported) == original
    assert runner.load_runtime_activity(original) == original
    with pytest.raises(ValueError, match='already exists'):
        runner.runtime_activity_reference(activity, tmp_path / 'trace')
    modified = json.loads(json.dumps(transported))
    modified['scoped_runtime']['modeled_decision_summary']['calls'] = 3
    with pytest.raises(ValueError, match='summary differs'):
        runner.load_runtime_activity(modified)
    Path(transported['runtime_activity_reference']['path']).write_text('{}\n')
    with pytest.raises(ValueError, match='sidecar changed'):
        runner.load_runtime_activity(transported)


def test_graph_and_command_queue_counters_keep_actual_submissions_separate(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / 'elmm-pipeline'))
    import elmm_pipeline.gpu_validation as validation
    monkeypatch.setattr(validation, 'gpu_activity', lambda *_args, **_kwargs: {'verified': False})
    graph = {'schema_version':1, 'context':2, 'event':'graph_statistics', 'stats_version':1, 'complete':True,
             'creations':1, 'instantiations':1, 'rebuilds':0, 'submissions':2, 'replays':1, 'parameter_updates':4,
             'logical_kernels':41, 'live_graphs':0, 'destructions':1}
    queue = {'schema_version':1, 'context':2, 'event':'command_queue_statistics', 'stats_version':1, 'complete':True,
             'queue_enabled':1, 'graphs_enabled':1, 'accepted_tasks':4, 'submissions':2, 'conflict_flushes':0,
             'explicit_flushes':2, 'pending_tasks':0, 'peak_pending_tasks':2, 'copied_argument_bytes':128,
             'direct_submissions':0, 'direct_kernels':0, 'graph_resource_fallbacks':0}
    position = ' provenance_version=1 context=2 owner=unknown procedure=unknown segment=unknown operation=unknown implementation=unknown boundary=unknown'
    submissions = '\n'.join('FORT_SCOPED graph_submit bytes='+str(count)+position for count in (20,21))
    stats = '\n'.join('FORT_SCOPED evidence '+json.dumps(item) for item in (graph, queue))
    result = runner.observed_gpu_activity(submissions+'\n'+stats,
                                          scope_packages=[{'scope_manifest': {}}])
    assert result['graph_statistics'] == [graph]
    assert result['command_queue_statistics'] == [queue]
    assert 'kernel_launches' not in result
    assert result['verified'] and result['runtime_cuda_submissions'] == 2
    assert sum(item['logical_kernels'] for item in result['graph_submission_evidence']) == 41
    with pytest.raises(ValueError, match='disagree'):
        runner.observed_gpu_activity(stats,scope_manifest={})
    with pytest.raises(ValueError, match='completed graph statistics'):
        runner.observed_gpu_activity(submissions,scope_manifest={})
    with pytest.raises(ValueError, match='source manifest'):
        runner.observed_gpu_activity((submissions+'\n'+stats).replace('owner=unknown','owner='+'f'*64),scope_manifest={})
    direct = {**queue,'graphs_enabled':0,'direct_submissions':2,'direct_kernels':2}
    direct_log = '\n'.join(['FORT_SCOPED launch bytes=0'+position]*2+['FORT_SCOPED evidence '+json.dumps(direct)])
    monkeypatch.setattr(validation, 'gpu_activity', lambda *_args, **_kwargs: {'verified': True,'kernel_launches':2})
    direct_result = runner.observed_gpu_activity(direct_log,scope_manifest={})
    assert direct_result['runtime_cuda_submissions'] == 2 and direct_result['graph_submission_evidence'] == []
    with pytest.raises(ValueError, match='direct kernels'):
        runner.observed_gpu_activity(direct_log.replace('"direct_kernels": 2','"direct_kernels": 20'),scope_manifest={})
    duplicate = json.dumps(graph).replace('"submissions": 2', '"submissions": 2, "submissions": 200')
    with pytest.raises(ValueError, match='duplicate'):
        runner.observed_gpu_activity('FORT_SCOPED evidence '+duplicate,scope_manifest={})
    queue['complete'] = False
    with pytest.raises(ValueError, match='complete runtime counters'):
        runner.observed_gpu_activity('FORT_SCOPED evidence '+json.dumps(queue),scope_manifest={})


def test_compact_carrier_evidence_stays_separate_and_rejects_incomplete_counts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / 'elmm-pipeline'))
    import elmm_pipeline.gpu_validation as validation
    logs = []
    monkeypatch.setattr(validation, 'gpu_activity', lambda log, *_args, **_kwargs:
                        logs.append(log) or {'verified': False})
    compact = {'schema_version': 1, 'event': 'compact_transfer', 'complete': True,
               'context': 3, 'identity': 4, 'direction': 'd2h', 'bytes': 8192,
               'operations': 9, 'kernels': 2, 'word_bytes': 8, 'scratch_capacity': 16384}
    def parse(value):
        return runner.observed_gpu_activity('FORT_SCOPED evidence '+json.dumps(value), scope_manifest={})
    activity = parse(compact)
    assert logs == ['']
    assert activity['runtime_cuda_numerical_kernels'] == 0
    assert activity['runtime_cuda_submissions'] == 0
    assert activity['runtime_cuda_carrier_kernels'] == 2
    assert activity['runtime_cuda_total_kernels'] == 2
    assert activity['runtime_cuda_total_submissions'] == 2
    assert not activity['verified']
    transported = runner.runtime_activity_reference(activity, tmp_path / 'compact')
    assert 'compact_transfer_events' not in transported
    assert transported['compact_transfer_summary']['directions']['d2h']['kernels'] == 2
    assert runner.load_runtime_activity(transported)['compact_transfer_events'] == [compact]
    for update in ({'complete': False}, {'schema_version': True}, {'bytes': -1},
                   {'operations': 0}, {'kernels': 1}, {'direction': 'gpu'},
                   {'word_bytes': 3}, {'scratch_capacity': 4096}, {'context': True}):
        with pytest.raises(ValueError, match='compact transfer'):
            parse({**compact, **update})


def test_residency_counters_preserve_dirty_eviction_traffic(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / 'elmm-pipeline'))
    import elmm_pipeline.gpu_validation as validation
    monkeypatch.setattr(validation, 'gpu_activity', lambda *_args, **_kwargs: {'verified': True})
    residency = {'schema_version':1, 'context':3, 'event':'residency_statistics', 'stats_version':1, 'complete':True,
                 'device_budget':1024, 'reserve_batches':4, 'requested_resources':7, 'reuse_hits':3,
                 'evictions':2, 'evicted_bytes':512, 'dirty_evictions':1, 'dirty_download_bytes':128, 'pressure_flushes':1}
    result = runner.observed_gpu_activity('FORT_SCOPED evidence '+json.dumps(residency),scope_manifest={})
    assert result['residency_statistics'] == [residency]
    assert result['residency_statistics'][0]['dirty_download_bytes'] == 128
    residency['device_budget'] = -1
    with pytest.raises(ValueError, match='runtime counters'):
        runner.observed_gpu_activity('FORT_SCOPED evidence '+json.dumps(residency),scope_manifest={})


def native_hint_record(**updates):
    return {'schema_version': 1, 'event': 'native_hint', 'phase': 'lookup',
            'reason': 'conservative_native_owner_hint', 'hit': False,
            'observation_eligible': True, 'learned': False,
            'estimates_available': False, 'key_words': 25, 'key_hash': 0, **updates}


def test_native_hint_metadata_preserves_order_and_never_counts_execution(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / 'elmm-pipeline'))
    import elmm_pipeline.gpu_validation as validation
    logs = []
    previous = {'verified': True, 'kernel_launches': 2, 'upload_bytes': 128,
                'download_bytes': 64, 'scoped_runtime': {'modeled_totals': {'calls': 3, 'cpu_units': 4, 'gpu_units': 2}}}
    monkeypatch.setattr(validation, 'gpu_activity', lambda log, *_args, **_kwargs:
                        logs.append(log) or json.loads(json.dumps(previous)))
    events = [native_hint_record(), native_hint_record(phase='finish', observation_eligible=False, learned=True),
              native_hint_record(hit=True, observation_eligible=False),
              native_hint_record(phase='finish', observation_eligible=False, learned=False)]
    original_log = 'existing application evidence\n'
    trace = original_log + '\n'.join('FORT_SCOPED native_hint ' + json.dumps(row) for row in events)
    activity = runner.observed_gpu_activity(trace, scope_manifest={})
    assert logs == [original_log + '\n' * 3]
    assert activity['native_hint_events'] == events
    assert activity['native_hint_counts'] == {'events': 4, 'lookups': 2, 'lookup_hits': 1,
                                             'observation_eligible': 1, 'finishes': 2, 'learned': 1}
    assert activity['scoped_runtime'] == previous['scoped_runtime']
    assert activity['runtime_cuda_numerical_kernels'] == activity['runtime_cuda_submissions'] == 2
    assert activity['runtime_cuda_carrier_kernels'] == 0
    assert activity['upload_bytes'] == 128 and activity['download_bytes'] == 64
    summary = runner.runtime_activity_reference(activity, tmp_path / 'hint')
    assert 'native_hint_events' not in summary
    assert summary['native_hint_counts'] == activity['native_hint_counts']
    assert 'Contextless' in summary['native_hint_semantics']
    assert runner.load_runtime_activity(summary) == activity
    changed = json.loads(json.dumps(summary))
    changed['native_hint_counts']['lookup_hits'] += 1
    with pytest.raises(ValueError, match='summary differs'):
        runner.load_runtime_activity(changed)


def test_native_hint_alone_has_no_cuda_or_placement_evidence(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / 'elmm-pipeline'))
    activity = runner.observed_gpu_activity('FORT_SCOPED native_hint ' + json.dumps(native_hint_record()),
                                          scope_manifest={})
    assert not activity['verified']
    assert activity['runtime_cuda_total_kernels'] == activity['runtime_cuda_total_submissions'] == 0
    assert activity['scoped_runtime']['modeled_totals']['calls'] == 0
    assert activity['native_hint_counts']['lookups'] == 1


@pytest.mark.parametrize('changes', [
    {'schema_version': True}, {'schema_version': 2}, {'event': 'decision'}, {'phase': 'choose'},
    {'reason': 'native_is_cheaper'}, {'context': 0}, {'owner': 'unknown'},
    {'hit': 1}, {'observation_eligible': 1}, {'learned': 0}, {'estimates_available': 0},
    {'estimates_available': True}, {'key_words': True}, {'key_words': 0}, {'key_words': 513},
    {'key_hash': True}, {'key_hash': -1}, {'key_hash': 2**64}, {'key_hash': 1.0},
    {'hit': True}, {'observation_eligible': False}, {'learned': True},
    {'phase': 'finish'}, {'phase': 'finish', 'observation_eligible': False, 'hit': True},
])
def test_native_hint_rejects_invalid_schema_and_inconsistent_phase_flags(monkeypatch, changes):
    monkeypatch.syspath_prepend(str(HERE.parent / 'elmm-pipeline'))
    with pytest.raises(ValueError, match='native hint'):
        runner.observed_gpu_activity('FORT_SCOPED native_hint ' + json.dumps(native_hint_record(**changes)),
                                     scope_manifest={})


def test_native_hint_rejects_duplicate_nonfinite_and_missing_fields(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE.parent / 'elmm-pipeline'))
    record = native_hint_record(key_words=512, key_hash=2**64-1)
    runner.observed_gpu_activity('FORT_SCOPED native_hint ' + json.dumps(record), scope_manifest={})
    for raw, message in [
        (json.dumps(record).replace('"key_words": 512', '"key_words": 512, "key_words": 1'), 'duplicate'),
        (json.dumps(record).replace('"key_hash": '+str(2**64-1), '"key_hash": NaN'), 'finite'),
        (json.dumps({**record, 'extra': {'value': float('inf')}}), 'finite'),
        (json.dumps({key: value for key, value in record.items() if key != 'phase'}), 'native hint'),
        ('[]', 'native hint'),
    ]:
        with pytest.raises(ValueError, match=message):
            runner.observed_gpu_activity('FORT_SCOPED native_hint ' + raw, scope_manifest={})


@pytest.mark.parametrize('raw', ['FORT_SCOPED native_hint', 'FORT_SCOPED native_hint{}',
                                 'FORT_SCOPED native_hint ', 'FORT_SCOPED native_hint {broken'])
def test_native_hint_rejects_malformed_payload_and_separator(monkeypatch, raw):
    monkeypatch.syspath_prepend(str(HERE.parent / 'elmm-pipeline'))
    with pytest.raises(ValueError):
        runner.observed_gpu_activity(raw, scope_manifest={})
