"""Compile a discovered, self-contained region using an explicit compiler snapshot."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from .checked_execution_plan import validate_checked_execution_plans
from .common import COMPILER_PHASE_TIMEOUT_SECONDS, python_environment
from .gpu_toolchain import validate_numerical_contract
from .owner_measurement import validate_owner_measurements


class LocalAdapterError(RuntimeError):
    pass


def _owner_measurements(manifest):
    try:
        return validate_owner_measurements(manifest)
    except ValueError as error:
        raise LocalAdapterError(str(error)) from error


def _checked_execution_plans(manifest, *, capture_facts=None):
    try:
        validate_checked_execution_plans(manifest, capture_facts=capture_facts)
    except ValueError as error:
        raise LocalAdapterError(str(error)) from error


def _numerical_build_fields(item):
    if not isinstance(item, dict):
        raise LocalAdapterError("Compiler numerical build metadata requires a source descriptor")
    if "exact_window_build_contract" in item:
        raise LocalAdapterError("Exact-window build contracts are unsupported by the application consumer; "
                                "use the independently authenticated fixture producer")
    if "numerical_contract" not in item:
        return {}
    if item.get("language") not in {"cuda", "c++"}:
        raise LocalAdapterError("Compiler numerical contract requires a CUDA or C++ build source")
    try:
        return {"numerical_contract": validate_numerical_contract(item["numerical_contract"])}
    except ValueError as error:
        raise LocalAdapterError(str(error)) from error


def cuda_build_fields(report, path):
    """Transport one ordinary compiler-approved CUDA build descriptor."""
    if "build_sources" not in report:
        return {}  # Historical compiler response; the build records legacy status.
    units = report["build_sources"]
    if not isinstance(units, list) or any(not isinstance(item, dict) for item in units):
        raise LocalAdapterError("Compiler build sources require public source descriptors")
    for item in units:
        _numerical_build_fields(item)
    selected = [item for item in units if item.get("path") == path and item.get("language") == "cuda"]
    if len(selected) != 1:
        raise LocalAdapterError("Compiler build sources must identify the requested CUDA artifact exactly once")
    return _numerical_build_fields(selected[0])


def _require_supported_scope_build(manifest):
    """Do not drop cost applicability contracts the consumer cannot enforce."""
    scopes = manifest.get("scopes", [])
    if not isinstance(scopes, list):
        raise LocalAdapterError("Compiler scope metadata requires a list of source scopes")
    from .application_build import selection_builds

    if "exact_window_build_contract" in manifest:
        raise LocalAdapterError("Exact-window cost evidence requires complete CUDA and Fortran build attestation; "
                                "the application consumer does not support this contract")
    try:
        selection_builds(manifest)
    except (KeyError, TypeError, ValueError) as error:
        raise LocalAdapterError(str(error)) from error


def _scope_numerical_contracts(manifest):
    """Required runtime metadata must match the actual common build source."""
    _require_supported_scope_build(manifest)
    units = manifest.get("build_sources", [])
    if not isinstance(units, list):
        raise LocalAdapterError("Compiler build sources require public source descriptors")
    for item in units:
        _numerical_build_fields(item)
    runtime = manifest.get("runtime") or {}
    if not isinstance(runtime, dict):
        raise LocalAdapterError("Compiler numerical runtime metadata requires a source descriptor")
    declarations = runtime.get("sources", [])
    if not isinstance(declarations, list):
        raise LocalAdapterError("Compiler numerical runtime metadata requires public source descriptors")
    for item in declarations:
        required = _numerical_build_fields(item)
        if required:
            selected = [unit for unit in units if unit.get("role") == "common_runtime"
                        and (unit.get("path"), unit.get("language")) == (item.get("path"), item.get("language"))]
            if len(selected) != 1 or _numerical_build_fields(selected[0]) != required:
                raise LocalAdapterError("Common runtime numerical build contract differs from its declared source")
    for unit in units:
        if unit.get("role") == "common_runtime" and _numerical_build_fields(unit):
            selected = [item for item in declarations
                        if (item.get("path"), item.get("language")) == (unit.get("path"), unit.get("language"))]
            if len(selected) != 1 or _numerical_build_fields(selected[0]) != _numerical_build_fields(unit):
                raise LocalAdapterError("Common runtime numerical build contract differs from its declared source")


def remap_scope_facts(facts, original_root, destination_root=None):
    """Move declared original-source identities without inventing proof facts.

    Collective spans always refer to original lines. Configured analysis text
    keeps its separate public line map and is never substituted here.
    """
    if not isinstance(facts, dict):
        raise LocalAdapterError("Scope capture facts must be a JSON object")
    original_root = Path(original_root).resolve(strict=True)
    destination_root = Path(destination_root or original_root).resolve()

    def source(name):
        if not isinstance(name, str):
            raise LocalAdapterError("Scope source identities must be strings")
        path = Path(name)
        path = (original_root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(original_root):
            raise LocalAdapterError("Scope capture source evidence escapes the selected application tree")
        return str(destination_root / path.relative_to(original_root))

    mapped = dict(facts)
    if "sources" in facts:
        if not isinstance(facts["sources"], dict):
            raise LocalAdapterError("Scope capture source evidence must be a JSON object")
        mapped["sources"] = {source(name): digest for name, digest in facts["sources"].items()}
    if facts.get("schema_version") == 2:
        participation = facts.get("participation")
        if not isinstance(participation, dict) or not isinstance(participation.get("call_sites"), list):
            raise LocalAdapterError("Collective participation requires explicit original call-site records")
        records = []
        for record in participation["call_sites"]:
            if not isinstance(record, dict) or "source" not in record:
                raise LocalAdapterError("Collective call-site records require original source identities")
            records.append({**record, "source": source(record["source"])})
        mapped["participation"] = {**participation, "call_sites": records}
        if "native_participation" in facts:
            native = facts["native_participation"]
            if not isinstance(native, dict) or any(not isinstance(record, dict) for record in native.values()):
                raise LocalAdapterError("Native participation assertions must be keyed procedure objects")
            mapped["native_participation"] = {
                procedure: {**record, **({"source": source(record["source"])} if "source" in record else {})}
                for procedure, record in native.items()}
    return mapped


def _collective_scope_contract(facts, entry, host_threads, hashes):
    """Check public transport fields; the compiler proves participation."""
    if facts.get("schema_version") != 2:
        return None
    participation = facts.get("participation")
    if (type(facts.get("schema_version")) is not int or not isinstance(participation, dict)
            or participation.get("kind") != "omp_full_team"
            or participation.get("dispatch") != "qualified_companion"
            or type(participation.get("expected_omp_level")) is not int
            or participation.get("expected_omp_level") != 1
            or type(host_threads) is not int or host_threads < 1
            or type(participation.get("host_threads")) is not int
            or participation.get("host_threads") != host_threads
            or not re.fullmatch(r"[a-z][a-z0-9_]*::[a-z][a-z0-9_]*", str(participation.get("entry", "")))
            or participation["entry"] != entry.lower()):
        raise LocalAdapterError("Collective scopes require a matching qualified full-team participation contract")
    records = participation.get("call_sites")
    if not isinstance(records, list) or not 1 <= len(records) <= 64:
        raise LocalAdapterError("Collective scopes require between 1 and 64 explicit original call sites")
    for record in records:
        if not isinstance(record, dict):
            raise LocalAdapterError("Collective call-site proof must be an object")
        source = record.get("source")
        if not isinstance(source, str) or source not in hashes or record.get("uniform_guard") != "unconditional":
            raise LocalAdapterError("Collective call sites require supplied original sources and unconditional participation")
        if "source_sha256" in record and record["source_sha256"] != hashes[source]:
            raise LocalAdapterError("Collective call-site source hash differs")
        spans = [record.get(key) for key in ("team_first_line", "first_line", "last_line", "team_last_line")]
        if any(type(value) is not int for value in spans) or not 1 <= spans[0] < spans[1] <= spans[2] < spans[3]:
            raise LocalAdapterError("Collective call sites require exact original call/team spans")
        for field in ("span_sha256", "team_span_sha256"):
            if (field == "span_sha256" or field in record) and not re.fullmatch(r"[0-9a-f]{64}", str(record.get(field, ""))):
                raise LocalAdapterError("Collective call-site hashes must describe exact original spans")
    native = facts.get("native_participation", {})
    if not isinstance(native, dict) or any(not isinstance(record, dict) for record in native.values()):
        raise LocalAdapterError("Native participation assertions must be keyed procedure objects")
    if any(not isinstance(record["source"], str) or record["source"] not in hashes
           for record in native.values() if "source" in record):
        raise LocalAdapterError("Native participation assertions require supplied original source identities")
    return participation


def _check_collective_scope_report(manifest, requested):
    """Require compiler-returned caller authority, without interpreting roles."""
    proof = manifest.get("participation")
    if (not isinstance(proof, dict) or type(proof.get("schema_version")) is not int or proof["schema_version"] != 2
            or type(proof.get("expected_omp_level")) is not int or type(proof.get("host_threads")) is not int
            or any(proof.get(key) != requested[key] for key in
                   ("kind", "dispatch", "entry", "expected_omp_level", "host_threads"))
            or not isinstance(proof.get("call_sites"), list)
            or len(proof["call_sites"]) != len(requested["call_sites"])):
        raise LocalAdapterError("Compiler collective scope report lacks matching versioned participation proof")
    fields = ("source", "first_line", "last_line", "team_first_line", "team_last_line", "span_sha256", "uniform_guard")
    if any(not isinstance(record, dict) or not isinstance(record.get("source"), str)
           or any(type(record.get(field)) is not int for field in
                  ("first_line", "last_line", "team_first_line", "team_last_line"))
           or any(not isinstance(record.get(field), str) for field in ("span_sha256", "uniform_guard"))
           for record in proof["call_sites"]):
        raise LocalAdapterError("Compiler collective scope report has invalid original call-site fields")
    records = {tuple(record.get(field) for field in fields): record
               for record in proof["call_sites"] if isinstance(record, dict)}
    if len(records) != len(proof["call_sites"]):
        raise LocalAdapterError("Compiler collective scope report repeats or omits a call site")
    for record in requested["call_sites"]:
        accepted = records.get(tuple(record[field] for field in fields))
        if accepted is None or any(accepted.get(field) != record[field] for field in
                                   ("source_sha256", "team_span_sha256", "caller") if field in record):
            raise LocalAdapterError("Compiler collective scope report differs from original caller evidence")


def _check_descriptor_expectation_contract(manifest, facts):
    """Require public enforcement metadata before using requested descriptor guards."""
    captures = facts.get("captures", {})
    requested = isinstance(captures, dict) and any(
        isinstance(fact, dict) and "expected_descriptor" in fact for fact in captures.values())
    if not requested or not manifest["scope_count"]:
        return
    contract = manifest.get("capture_descriptor_expectations")
    if (not isinstance(contract, dict) or type(contract.get("schema_version")) is not int
            or contract["schema_version"] != 1
            or contract.get("enforcement") != "original_descriptor_before_registration"
            or contract.get("definition_authority") is not False
            or contract.get("supported_consumers") != ["original_lexical", "original_module_companion"]
            or contract.get("unsupported_consumers") != ["synthetic_owner", "qualified_collective_owner"]):
        raise LocalAdapterError("Compiler source scopes lack versioned capture descriptor expectation enforcement")


def _invoke(source_path, compiler_root, python, output_dir, entry, arguments, log_name):
    """Use only the selected snapshot's public CLI and JSON response."""
    source_path = Path(source_path).resolve(strict=True)
    compiler_root = Path(compiler_root).resolve(strict=True)
    python = Path(python).absolute()
    output_dir = Path(output_dir).resolve()
    if not (compiler_root / "compiler/__main__.py").is_file():
        raise LocalAdapterError("Supply an explicit compiler snapshot containing compiler/__main__.py")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:::[A-Za-z][A-Za-z0-9_]*)?", entry):
        raise LocalAdapterError(f"Invalid generated entry name: {entry!r}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise LocalAdapterError(f"Configured compiler Python is not executable: {python}")
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = python_environment()
    environment["PYTHONPATH"] = str(compiler_root)
    command = [str(python), "-m", "compiler", "--input", str(source_path),
               "--opt-level", "1", "--fallback", "error", "--json", *arguments]
    try:
        result = subprocess.run(command, cwd=compiler_root, env=environment, text=True,
                                capture_output=True, check=False, timeout=COMPILER_PHASE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        stdout, stderr = (value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
                          for value in (error.stdout, error.stderr))
        (output_dir / log_name).write_text(stdout + stderr)
        (output_dir / (log_name + ".timeout.json")).write_text(json.dumps({
            "argv": command, "cwd": str(compiler_root), "timeout_seconds": COMPILER_PHASE_TIMEOUT_SECONDS,
            "timed_out": True, "returncode": None}, indent=2) + "\n")
        raise LocalAdapterError(f"Compiler {entry} exceeded the {COMPILER_PHASE_TIMEOUT_SECONDS}-second phase limit; "
                                "no artifacts accepted") from error
    (output_dir / log_name).write_text(result.stdout + result.stderr)
    try:
        report = json.loads(result.stdout)
    except (ValueError, TypeError) as error:
        raise LocalAdapterError(
            "Compiler did not return a JSON result; refresh the explicit compiler snapshot: "
            + (result.stdout + result.stderr)[-4000:]) from error
    if result.returncode:
        reason = report.get("reason") if isinstance(report, dict) else None
        raise LocalAdapterError(f"Compiler rejected {entry}: {reason or (result.stdout + result.stderr)[-4000:]}")
    return report, command


def check(source_path: Path, compiler_root: Path, python: Path, entry: str, output_dir: Path) -> dict:
    """Return the selected entry's authoritative, strict eligibility record.

    A rejected candidate returns ``supported=False`` without emitting sources.
    Invalid tool responses and missing or ambiguous entries raise an adapter error.
    """
    records, _ = _invoke(source_path, compiler_root, python, output_dir, entry,
                         ["--list-candidates"], "check.log")
    if not isinstance(records, list):
        raise LocalAdapterError("Compiler candidate report must be a JSON array")
    selector = "qualified_name" if "::" in entry else "name"
    matches = [record for record in records if isinstance(record, dict)
               and str(record.get(selector, "")).lower() == entry.lower()]
    if len(matches) != 1:
        raise LocalAdapterError(f"Compiler candidate report must identify exactly one entry: {entry}")
    report, = matches
    if type(report.get("supported")) is not bool:
        raise LocalAdapterError("Compiler candidate report is missing its supported decision")
    (Path(output_dir) / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def transform(source_path: Path, compiler_root: Path, python: Path,
              output_dir: Path, *, entry: str, context=None, gpu_options=None) -> dict:
    """Translate normalized captures using the compiler's numerical proofs.

    Discovery owns capture lifetimes, OpenMP context and array-bound normalization.
    The selected compiler owns numeric legality and ordered kernel generation.
    """
    source_path = Path(source_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    arguments = ["--kernel", entry, "--output-dir", str(output_dir),
                 "--cpp-output", "region.cpp", "--fortran-output", "interface.f90",
                 "--cuda-output", "region.cu", "--common-header", "runtime.hpp"]
    options = gpu_options or {}
    policy = options.get("gpu_policy", "always")
    if policy not in {"always", "sections", "auto", "chunked", "hybrid"}:
        raise LocalAdapterError("Unsupported local GPU policy: " + str(policy))
    if gpu_options is not None:
        arguments += ["--gpu-policy", policy, "--host-threads", str(options.get("host_threads", 4))]
        if options.get("calibration_profile"):
            arguments += ["--calibration-profile", str(Path(options["calibration_profile"]).resolve())]
        if policy != "always" and (context or {}).get("gpu_dispatch") == "collective_single":
            arguments.append("--gpu-collective")
    report, command = _invoke(source_path, compiler_root, python, output_dir, entry,
                              arguments, "compiler.log")
    expected = {"region.cpp", "region.cu", "interface.f90", "runtime.hpp"}
    if (not isinstance(report, dict) or report.get("supported") is not True
            or not isinstance(report.get("kernel"), str)
            or report.get("kernel", "").lower() != entry.lower()
            or type(report.get("parallel_regions")) is not int or report["parallel_regions"] < 0
            or not isinstance(report.get("outputs"), list)
            or not all(isinstance(name, str) for name in report["outputs"])
            or not expected.issubset(report["outputs"])):
        raise LocalAdapterError("Compiler generation report is missing its supported decision, regions or outputs")
    if policy != "always":
        offload = report.get("offload")
        if (not isinstance(offload, dict) or offload.get("policy") != policy
                or type(offload.get("collective_entry")) is not bool
                or not isinstance(offload.get("native_fallback_query"), str)
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", offload["native_fallback_query"])):
            raise LocalAdapterError("Compiler offload report is missing its policy/query dispatch contract")
        if offload["collective_entry"] and (context or {}).get("gpu_dispatch") != "collective_single":
            raise LocalAdapterError("Compiler requested collective dispatch outside a proved OpenMP team")
    for filename in expected:
        if not (output_dir / filename).is_file():
            raise LocalAdapterError(f"Compiler did not emit required file: {output_dir / filename}")
    (output_dir / "compiler-report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Enclosing native OpenMP loops already distribute these calls. Symbol
    # isolation belongs to the compiler; other emitted artifacts stay intact.
    if (context or {}).get("nested_parallel"):
        path = output_dir / "region.cpp"
        text = "\n".join(line for line in path.read_text().splitlines()
                          if not re.match(r"\s*#\s*pragma\s+omp\b", line)) + "\n"
        path.write_text(text)
    metadata = {
        "tool": "local", "target": "cpp", "entry": entry, "changed": True,
        "context": context or {}, "opt_level": 1,
        "extra_sources": [str(output_dir / "interface.f90"), str(output_dir / "region.cpp")],
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "command": command,
        "compiler_report": report,
        "policy": "compiler dependence analysis, scalar motion, loop fusion and locality scheduling",
    }
    (output_dir / "transformation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata



def transform_scopes(source_path: Path, compiler_root: Path, python: Path, output_dir: Path, *,
                     entry: str, source_files=(), capture_facts: dict, gpu_options=None,
                     effect_contracts: Path | None = None,
                     numerical_sources: Path | None = None,
                     analysis_sources: Path | None = None,
                     summary_cache: Path | None = None) -> dict:
    """Consume the current graph source protocol; historical transport is private."""
    options = dict(gpu_options or {})
    retired = {"scope_cost", "scope_measurement"} & options.keys()
    if retired:
        raise LocalAdapterError("Retired public source options: " + ", ".join(sorted(retired)))
    if options.get("scope_execution", "graph") != "graph":
        raise LocalAdapterError("Public source scopes require graph execution")
    options["scope_execution"] = "graph"
    return _transform_scopes_protocol(
        source_path, compiler_root, python, output_dir, entry=entry,
        source_files=source_files, capture_facts=capture_facts, gpu_options=options,
        effect_contracts=effect_contracts, numerical_sources=numerical_sources,
        analysis_sources=analysis_sources, summary_cache=summary_cache,
        legacy_protocol=False)


def _transform_scopes_protocol(source_path: Path, compiler_root: Path, python: Path, output_dir: Path, *,
                              entry: str, source_files=(), capture_facts: dict, gpu_options=None,
                              effect_contracts: Path | None = None,
                              numerical_sources: Path | None = None,
                              analysis_sources: Path | None = None,
                              summary_cache: Path | None = None,
                              legacy_protocol: bool) -> dict:
    """Request approved source scopes; consume only versioned public artifacts.

    Capture facts belong to extraction/capture safety. This adapter does not
    choose grouping, effects, GPU leaves, transfers, or placement.
    """
    source_path=Path(source_path).resolve(strict=True)
    output_dir=Path(output_dir).resolve()
    source_files=[Path(p).resolve(strict=True) for p in source_files]
    options=gpu_options or {}
    policy=options.get("gpu_policy","sections")
    if policy not in {"sections","auto"}:
        raise LocalAdapterError("Scoped source execution requires sections or auto")
    measurement = options.get("scope_measurement")
    if measurement not in (None, "native", "resident"):
        raise LocalAdapterError("Unknown scope measurement variant")
    if measurement is not None and (policy != "sections" or options.get("scope_execution") != "reached"):
        raise LocalAdapterError("Scope measurement requires sections policy and reached execution")
    facts=dict(capture_facts)
    sources=list(dict.fromkeys([source_path,*source_files]))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if "sources" in facts and facts["sources"] != hashes:
        raise LocalAdapterError("Scope capture facts belong to different source contents")
    facts["sources"]=hashes
    collective = _collective_scope_contract(facts, entry, options.get("host_threads", 4), hashes)
    if measurement is not None and collective is not None:
        raise LocalAdapterError("Scope measurement requires a noncollective original caller")
    output_dir.mkdir(parents=True,exist_ok=True)
    facts_path=output_dir/"scope-captures.json"
    facts_path.write_text(json.dumps(facts,indent=2)+"\n")
    arguments=["--kernel",entry,"--output-dir",str(output_dir),"--form-scopes",
               "--scope-facts",str(facts_path),"--memory-model","scoped","--gpu-policy",policy,
               "--host-threads",str(options.get("host_threads",4))]
    if options.get("scope_transfers", "direct") != "direct":
        arguments += ["--scope-transfers", options["scope_transfers"]]
    if options.get("scope_execution", "bounded") != "bounded":
        arguments += ["--scope-execution", options["scope_execution"]]
    if measurement is not None:
        arguments += ["--scope-measurement", measurement]
    cost_path = options.get("scope_cost")
    cost_identity = None
    if cost_path is not None:
        cost_path = Path(cost_path).resolve(strict=True)
        cost_identity = hashlib.sha256(cost_path.read_bytes()).hexdigest()
        if collective is not None or measurement is not None:
            raise LocalAdapterError("Scope cost requires a noncollective selection without diagnostic call measurement")
        arguments += ["--scope-cost", str(cost_path)]
    if collective is not None:
        arguments += ["--gpu-collective"]
    for source in source_files:
        arguments+=["--source-file",str(source)]
    if options.get("calibration_profile"):
        arguments+=["--calibration-profile",str(Path(options["calibration_profile"]).resolve(strict=True))]
    if effect_contracts:
        arguments+=["--effect-contracts",str(Path(effect_contracts).resolve(strict=True))]
    if summary_cache is not None:
        arguments+=["--summary-cache",str(Path(summary_cache).resolve())]
    analysis=None
    if analysis_sources:
        analysis_sources=Path(analysis_sources).resolve(strict=True)
        analysis=json.loads(analysis_sources.read_text())
        if analysis.get("source_inputs")!=hashes:
            raise LocalAdapterError("Configured analysis package belongs to different original sources")
        arguments+=["--analysis-sources",str(analysis_sources)]
    normalized=[]
    if numerical_sources:
        numerical_sources=Path(numerical_sources).resolve(strict=True)
        try:
            package=json.loads(numerical_sources.read_text())
            normalized=[{key:item[key] for key in ("procedure","path","entry","sha256")}
                        for item in package["entries"]]
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise LocalAdapterError("Invalid numerical source package") from error
        arguments+=["--numerical-sources",str(numerical_sources)]
    report,command=_invoke(source_path,compiler_root,python,output_dir,entry,arguments,"scope-compiler.log")
    if cost_path is not None and hashlib.sha256(cost_path.read_bytes()).hexdigest() != cost_identity:
        raise LocalAdapterError("Scope cost request changed during compilation")
    manifest=report.get("scopes") if isinstance(report,dict) else None
    if (not isinstance(manifest,dict) or report.get("supported") is not True
            or manifest.get("schema_version") != 1 or type(manifest.get("scope_count")) is not int
            or manifest["scope_count"] < 0 or not isinstance(manifest.get("scopes"),list)
            or len(manifest["scopes"]) != manifest["scope_count"]
            or manifest.get("source_inputs") != hashes
            or manifest.get("analysis_sources") != analysis
            or manifest.get("numerical_sources",[]) != sorted(normalized,key=lambda item:item["procedure"])
            or manifest.get("capture_facts_sha256") != hashlib.sha256(json.dumps(facts,sort_keys=True).encode()).hexdigest()
            or not isinstance(manifest.get("artifacts_sha256"),dict)
            or not isinstance(report.get("outputs"),list)):
        raise LocalAdapterError("Compiler source-scope report is missing versioned scope/capture/artifact facts")
    _checked_execution_plans(manifest, capture_facts=facts)
    if not legacy_protocol and manifest.get("scope_execution") != "graph":
        raise LocalAdapterError("Compiler returned a non-graph source protocol")
    if legacy_protocol and manifest.get("scope_execution") != options.get("scope_execution", "bounded"):
        raise LocalAdapterError("Historical compiler source protocol differs from its explicit request")
    if _owner_measurements(manifest) != measurement:
        raise LocalAdapterError("Compiler scope measurement variant differs from its request")
    _check_descriptor_expectation_contract(manifest, facts)
    if collective is not None:
        _check_collective_scope_report(manifest, collective)
    _scope_numerical_contracts(manifest)
    if manifest["scope_count"] and measurement != "native":
        runtime=manifest.get("runtime")
        if (not isinstance(runtime,dict) or runtime.get("abi_version") != 1
                or runtime.get("link_once") is not True or not isinstance(runtime.get("runtime_id"),str)
                or not isinstance(manifest.get("build_sources"),list)):
            raise LocalAdapterError("Compiler source scopes require a compatible common runtime build manifest")
        runtime_hashes=runtime.get("source_sha256")
        if (not isinstance(runtime_hashes,dict) or not runtime_hashes
                or any(manifest["artifacts_sha256"].get(name) != digest for name,digest in runtime_hashes.items())
                or runtime["runtime_id"] != hashlib.sha256("\n".join(
                    f"{name}:{digest}" for name,digest in sorted(runtime_hashes.items())).encode()).hexdigest()):
            raise LocalAdapterError("Compiler scope runtime identity differs from its build artifacts")
        for item in manifest["build_sources"]:
            if (not isinstance(item,dict) or item.get("path") not in manifest["artifacts_sha256"]
                    or item.get("language") not in {"cuda","fortran"}
                    or item.get("role") not in {"common_runtime","shared_entry","original_source"}):
                raise LocalAdapterError("Compiler scope build source is missing a public artifact/role")
            _numerical_build_fields(item)
    for name,digest in manifest["artifacts_sha256"].items():
        artifact=_scope_artifact(output_dir,name)
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
            raise LocalAdapterError("Compiler scope artifact hash differs: " + name)
    if set(report["outputs"]) != {*manifest["artifacts_sha256"],"scope-manifest.json"}:
        raise LocalAdapterError("Compiler source-scope outputs differ from its artifact manifest")
    recorded=json.loads((output_dir/"scope-manifest.json").read_text())
    if recorded != manifest:
        raise LocalAdapterError("Compiler's saved source-scope manifest differs from its response")
    for source in sources:
        if hashlib.sha256(source.read_bytes()).hexdigest() != hashes[str(source)]:
            raise LocalAdapterError("Source changed while its compiler scope artifacts were prepared")
    if analysis:
        for item in analysis["entries"]:
            if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()!=item["sha256"]:
                raise LocalAdapterError("Configured analysis source changed while artifacts were being prepared")
        for path,digest in analysis["dependencies"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                raise LocalAdapterError("Configured source dependency changed while artifacts were being prepared")
    for item in normalized:
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise LocalAdapterError("Normalized source changed while its compiler scope artifacts were prepared")
    metadata={"tool":"local","memory_model":"scoped","entry":entry,"changed":bool(manifest["scope_count"]),
              "command":command,"compiler_report":report,"scope_manifest":manifest,"artifact_root":str(output_dir)}
    if cost_path is not None:
        metadata["scope_cost"] = {"path": str(cost_path), "sha256": cost_identity}
    from .application_build import selection_builds

    if selection_builds(manifest):
        metadata["application_build"] = {"compiler_root": str(Path(compiler_root).resolve()),
                                         "python": str(Path(python).absolute()), "artifact_root": str(output_dir),
                                         "manifest": str(output_dir / "scope-manifest.json"),
                                         "manifest_sha256": hashlib.sha256((output_dir / "scope-manifest.json").read_bytes()).hexdigest()}
    if measurement is not None:
        metadata.update(measurement_only=True, scope_measurement=measurement)
    # The existing application build already scans GNU Fortran dependencies
    # and compiles separate CUDA objects. Supply its inputs from public roles.
    units=manifest.get("build_sources",[])
    metadata["extra_sources"]=[str(_scope_artifact(output_dir,item["path"])) for item in units
                               if item["language"]=="fortran" and item["role"]!="original_source"]
    metadata["gpu_sources"]=[{"source":str(_scope_artifact(output_dir,item["path"])),
                              "language":"cuda","tool":"local", **_numerical_build_fields(item)}
                             for item in units if item["language"]=="cuda"]
    (output_dir/"scope-transformation.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata


def _scope_artifact(root, name):
    """Validate a public artifact path without interpreting its source."""
    if not isinstance(name,str) or Path(name).is_absolute():
        raise LocalAdapterError("Scope artifacts require relative paths")
    try:
        path=(root/name).resolve(strict=True)
    except OSError as error:
        raise LocalAdapterError("Missing compiler scope artifact: " + name) from error
    if not path.is_relative_to(root) or not path.is_file():
        raise LocalAdapterError("Scope artifact escapes its output directory: " + name)
    return path


def _prepare_scope_sources(metadata, destinations):
    """Validate replacements without changing any application source."""
    manifest=metadata["scope_manifest"]
    _scope_numerical_contracts(manifest)
    _owner_measurements(manifest)
    root=Path(metadata["artifact_root"]).resolve(strict=True)
    prepared=[]
    dependencies=manifest.get("analysis_sources") or {}
    for source,digest in dependencies.get("dependencies",{}).items():
        dependency=Path(destinations.get(source,source)).resolve(strict=True)
        if hashlib.sha256(dependency.read_bytes()).hexdigest()!=digest:
            raise LocalAdapterError("Configured source dependency changed before applying approved scope: "+source)
    for source,digest in manifest["source_inputs"].items():
        path=Path(destinations.get(source,source)).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise LocalAdapterError("Application source changed before applying approved scope: "+source)
    for source,item in manifest["sources"].items():
        if source not in destinations:
            raise LocalAdapterError("No application-copy destination for scoped source: " + source)
        destination=Path(destinations[source]).resolve(strict=True)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
            raise LocalAdapterError("Application source changed before applying approved scope: " + source)
        name=item["replacement"]
        replacement=_scope_artifact(root,name)
        content=replacement.read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["artifacts_sha256"].get(name):
            raise LocalAdapterError("Approved source replacement changed: " + name)
        prepared.append((destination,content))
    return prepared


def apply_scope_sources(metadata: dict, destinations: dict[str, Path]) -> list[Path]:
    """Apply complete compiler-approved replacements to an application copy.

    Validate every source/artifact first, so a stale source cannot produce a
    partially applied transformation. Unchanged application sources stay under
    the build system's existing ownership.
    """
    _scope_package_build(metadata)
    prepared = _prepare_scope_sources(metadata, destinations)
    for destination, content in prepared:
        destination.write_bytes(content)
    if "application_build" in metadata:
        from .application_build import selection_builds

        records = selection_builds(metadata["scope_manifest"])
        originals = records[0]["contract"]["build_contract"]["original_inputs"]
        # An omitted destination means the unchanged exact original path; no
        # implicit basename/root relocation is performed.
        metadata["application_build"]["destinations"] = {
            source: str(Path(destinations.get(source, source)).resolve()) for source in originals}
    return [destination for destination, _ in prepared]


def _scope_package_build(metadata):
    """Revalidate saved public artifacts and return their declared build roles."""
    if not isinstance(metadata, dict) or not isinstance(metadata.get("scope_manifest"), dict):
        raise LocalAdapterError("Scope composition requires compiler package metadata")
    manifest = metadata["scope_manifest"]
    _require_supported_scope_build(manifest)
    _checked_execution_plans(manifest)
    measurement = _owner_measurements(manifest)
    try:
        root = Path(metadata["artifact_root"]).resolve(strict=True)
    except (KeyError, TypeError, OSError) as error:
        raise LocalAdapterError("Scope composition requires an existing artifact directory") from error
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or type(manifest.get("scope_count")) is not int or manifest["scope_count"] < 0
            or not isinstance(manifest.get("source_inputs"), dict) or not isinstance(manifest.get("sources"), dict)
            or not isinstance(manifest.get("artifacts_sha256"), dict)):
        raise LocalAdapterError("Scope composition requires a versioned artifact manifest")
    try:
        recorded = json.loads((root / "scope-manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise LocalAdapterError("Cannot read saved compiler scope manifest") from error
    if recorded != manifest:
        raise LocalAdapterError("Saved compiler scope manifest differs from its package")
    artifact_hashes = {}
    for name, digest in manifest["artifacts_sha256"].items():
        artifact = _scope_artifact(root, name)
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
            raise LocalAdapterError("Approved scope artifact changed: " + name)
        artifact_hashes[artifact.relative_to(root).as_posix()] = digest
    units = manifest.get("build_sources", [])
    if not isinstance(units, list):
        raise LocalAdapterError("Scope composition requires public build roles")
    runtime = manifest.get("runtime")
    if manifest.get("scope_count", 0) and measurement != "native":
        if (not isinstance(runtime, dict) or runtime.get("schema_version") != 1
                or runtime.get("abi_version") != 1 or runtime.get("link_once") is not True
                or not isinstance(runtime.get("source_sha256"), dict) or not runtime["source_sha256"]
                or not isinstance(runtime.get("sources"), list)):
            raise LocalAdapterError("Scope composition requires a compatible link-once runtime")
        hashes = runtime["source_sha256"]
        if any(not isinstance(name, str) or not isinstance(digest, str) for name, digest in hashes.items()):
            raise LocalAdapterError("Scope runtime identity requires source hashes")
        identity = hashlib.sha256("\n".join(f"{name}:{digest}" for name, digest in sorted(hashes.items())).encode()).hexdigest()
        if (runtime.get("runtime_id") != identity
                or any(manifest["artifacts_sha256"].get(name) != digest for name, digest in hashes.items())):
            raise LocalAdapterError("Scope runtime identity differs from its approved artifacts")
        if any(not isinstance(item, dict) or not isinstance(item.get("path"), str)
               or item["path"] not in hashes or item.get("language") not in {"cuda", "fortran"}
               for item in runtime["sources"]):
            raise LocalAdapterError("Scope runtime sources require public paths and languages")
        for item in runtime["sources"]:
            _numerical_build_fields(item)
        expected_runtime = {(item["path"], item["language"]) for item in runtime["sources"]}
        if len(expected_runtime) != len(runtime["sources"]):
            raise LocalAdapterError("Scope runtime source descriptors must not repeat")
    else:
        if runtime is not None:
            raise LocalAdapterError("Native scope package unexpectedly declares a runtime")
        expected_runtime = set()
    actual_runtime, build, replacements = set(), [], set()
    for item in units:
        if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
                or item["path"] not in manifest["artifacts_sha256"]
                or item.get("language") not in {"cuda", "fortran"}
                or item.get("role") not in {"common_runtime", "shared_entry", "original_source"}):
            raise LocalAdapterError("Scope build source is missing a public artifact/role")
        name, language, role = (item[key] for key in ("path", "language", "role"))
        numerical = _numerical_build_fields(item)
        source = _scope_artifact(root, name)
        if role == "common_runtime":
            actual_runtime.add((name, language))
            if runtime is None or name not in runtime["source_sha256"]:
                raise LocalAdapterError("Common runtime build source lacks runtime identity")
            declared = next((entry for entry in runtime["sources"]
                             if (entry["path"], entry["language"]) == (name, language)), None)
            if declared is None or _numerical_build_fields(declared) != numerical:
                raise LocalAdapterError("Common runtime numerical build contract differs from its declared source")
        if role == "original_source":
            replacements.add(source.relative_to(root).as_posix())
        build.append({"path": source.relative_to(root).as_posix(), "source": str(source),
                      "language": language, "role": role, "sha256": manifest["artifacts_sha256"][name],
                      **numerical})
    if actual_runtime != expected_runtime:
        raise LocalAdapterError("Common runtime build roles differ from its declared sources")
    # Complete original-source replacements compose by destination below;
    # their incidental artifact paths do not identify shared helper code.
    return runtime, build, {path: digest for path, digest in artifact_hashes.items() if path not in replacements}


def compose_scope_packages(packages, destinations: dict[str, Path]) -> dict:
    """Apply independent approved packages once and combine their build inputs.

    Every package must describe pristine sources. Validate all packages before
    writing any replacement; differing replacements of one destination are
    rejected. Deduplicate artifacts by public path, role and digest, and require
    exactly matching link-once runtime identities. Original manifests remain
    separate in the receipt; this function makes no compiler decisions.
    """
    packages = list(packages)
    from .application_build import selection_builds

    for package in packages:
        _require_supported_scope_build(package.get("scope_manifest", {}))
    if any(selection_builds(package.get("scope_manifest", {})) for package in packages):
        raise LocalAdapterError("Priced selection bundles require one compiler-issued package and its complete build attestation")
    replacements, artifacts, artifact_hashes, runtime = {}, {}, {}, None
    for index, metadata in enumerate(packages):
        package_runtime, build, package_hashes = _scope_package_build(metadata)
        if package_runtime is not None:
            if runtime is not None and package_runtime != runtime:
                raise LocalAdapterError("Scope packages require exactly matching runtime identities and artifacts")
            runtime = package_runtime
        # Shared entry bytes alone do not identify included helper headers.
        # Reject every colliding approved path before dropping any duplicate TU.
        for name, digest in package_hashes.items():
            if name in artifact_hashes and artifact_hashes[name] != digest:
                raise LocalAdapterError("Conflicting approved scope artifact: " + name)
            artifact_hashes[name] = digest
        for destination, content in _prepare_scope_sources(metadata, destinations):
            if destination in replacements and replacements[destination] != content:
                raise LocalAdapterError("Conflicting approved scope replacements: " + str(destination))
            replacements[destination] = content
        for item in build:
            if item["role"] == "original_source":
                continue
            name = item["path"]
            if name in artifacts:
                previous = artifacts[name]
                if any(previous[key] != item[key] for key in ("language", "role", "sha256")):
                    raise LocalAdapterError("Conflicting approved scope build artifact: " + name)
                if previous.get("numerical_contract") != item.get("numerical_contract"):
                    raise LocalAdapterError("Conflicting approved scope numerical build contract: " + name)
                previous["packages"].append(index)
            else:
                artifacts[name] = {**item, "packages": [index]}
    for destination, content in replacements.items():
        destination.write_bytes(content)
    return {"schema_version": 1, "tool": "local", "memory_model": "scoped",
            **({"measurement_only": True} if any(p["scope_manifest"].get("scope_measurement") for p in packages) else {}),
            "scope_packages": packages, "runtime": runtime, "build_artifacts": list(artifacts.values()),
            "patched_sources": [str(path) for path in replacements],
            "extra_sources": [item["source"] for item in artifacts.values() if item["language"] == "fortran"],
            "gpu_sources": [{"source": item["source"], "language": "cuda", "tool": "local",
                             **_numerical_build_fields(item)}
                            for item in artifacts.values() if item["language"] == "cuda"]}
