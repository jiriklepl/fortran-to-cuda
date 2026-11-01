"""Private diagnostic source-unit instrumentation of authenticated ELMM builds."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = "7a612e40b5790e2d112ea137c6a38a9ff0ef7d41"
MODES = {"native": "native", "be58": "committed", "baseline": "reworked"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify(bindings):
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError("baseline diagnostic input changed: " + path)


def freeze(prepared_path, output):
    """Preserve committed harness/source bytes before workspace redesign edits."""
    prepared_path, output = Path(prepared_path).resolve(), Path(output).resolve()
    prepared = read(prepared_path)
    output.mkdir(parents=True, exist_ok=False)
    revisions = {"main": BASELINE, "pipeline": "55ba91723749eb28269c471819e3045ee8ae507e"}
    for name, source, paths, destination in (
        ("main", REPO, ["compiler", "containers"], output / "checkout"),
        ("pipeline", REPO / "elmm-pipeline", ["elmm_pipeline"], output / "checkout/elmm-pipeline"),
    ):
        archive = subprocess.check_output(["git", "-C", str(source), "archive", revisions[name], *paths])
        with (output / (name + ".tar")).open("xb") as stream:
            stream.write(archive)
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(destination, filter="data")
    shutil.copyfile(prepared_path, output / "prepared-builds.json")
    bindings = {str(prepared_path): digest(prepared_path)}
    builds = {}
    for mode, original in MODES.items():
        built = prepared["builds"][original]
        receipt = read(built["build_receipt"])
        if digest(built["binary"]) != built["sha256"] or digest(built["build_receipt"]) != built["build_receipt_sha256"]:
            raise ValueError("production build changed before baseline freeze")
        shutil.copyfile(built["build_receipt"], output / (mode + "-build-receipts.json"))
        diagnostic = prepared["diagnostic_builds"][original]
        for path, expected in ((built["binary"], built["sha256"]),
                               (diagnostic["binary"], diagnostic["sha256"]),
                               (diagnostic["source"], diagnostic["source_sha256"])):
            bindings[path] = expected
        bindings.update(receipt["files_sha256"])
        builds[mode] = {"source_mode": original, "production": built, "fullfield": diagnostic,
                        "receipt_copy": str(output / (mode + "-build-receipts.json"))}
    verify(bindings)
    frozen = {str(path.relative_to(output)): digest(path) for path in output.rglob("*") if path.is_file()}
    write(output / "identity.json", {"schema_version": 1, "kind": "immutable baseline source/harness and existing artifact authority",
          "revisions": revisions, "prepared_sha256": digest(prepared_path), "builds": builds,
          "original_artifacts_sha256": bindings, "frozen_files_sha256": frozen,
          "diagnostic_only": True, "baseline_objects_or_sources_modified": False,
          "application_or_timing_executed": False})
    print(output / "identity.json", digest(output / "identity.json"), flush=True)


def prepare(baseline, output):
    baseline, output = Path(baseline).resolve(), Path(output).resolve()
    identity = read(baseline / "identity.json")
    verify(identity["original_artifacts_sha256"])
    verify({str(baseline / name): value for name, value in identity["frozen_files_sha256"].items()})
    output.mkdir(parents=True, exist_ok=False)
    timer = REPO / "containers/profiling/budget_timer.cpp"
    from profiling.cabi_signatures import wrappers

    commands, variants = [], {}
    for mode, selected in identity["builds"].items():
        target = output / mode
        target.mkdir()
        receipt = read(selected["receipt_copy"])
        replacements, units = {}, []
        cuda = [Path(unit["source"]) for unit in receipt["units"] if Path(unit["source"]).suffix == ".cu"]
        wrap_flags = []
        wrapper_sources = []
        if cuda:
            runtime = next(path for path in cuda if path.name == "scoped_runtime.cu")
            wrapper_source = target / "runtime-wrappers.cpp"
            wrapper_functions = wrappers(cuda, wrapper_source)
            wrapper_sources = [str(wrapper_source)]
            wrap_flags = ["-Wl,--wrap=" + row["name"] for row in wrapper_functions]
            write(target / "runtime-wrapper-functions.json", wrapper_functions)
            commands.append({"kind": "typed runtime diagnostic wrapper", "argv": ["/usr/bin/g++-14", "-std=c++17", "-O2", "-fno-instrument-functions", "-I" + str(runtime.parent), "-c", str(wrapper_source), "-o", str(target / "runtime-wrappers.o")], "cwd": str(REPO), "environment": {}, "timeout_seconds": 120, "mode": mode})
        for number, unit in enumerate(receipt["units"]):
            if Path(unit["source"]).suffix.lower() not in {".f90", ".f", ".f03"}:
                continue
            private = target / (f"unit-{number:03d}")
            private.mkdir()
            source, obj = private / Path(unit["source"]).name, private / Path(unit["object"]).name
            # The full-field main is the existing, already validated observer.
            original = Path(selected["fullfield"]["source"]) if Path(unit["source"]).name == "main.f90" and "/elmm/" in unit["source"] else Path(unit["source"])
            source.write_bytes(original.read_bytes())
            argv = list(unit["execution"]["argv"])
            argv[argv.index("-o") + 1] = str(obj)
            cwd = Path(unit["execution"]["cwd"])
            operands = [i for i, value in enumerate(argv) if not value.startswith("-")
                        and (cwd / value).resolve() == Path(unit["source"]).resolve()]
            if len(operands) != 1:
                raise ValueError("one exact source operand required")
            argv[operands[0]] = str(source)
            argv = ["-J" + str(private) if value.startswith("-J") else value for value in argv]
            argv.append("-I" + str(Path(unit["source"]).parent))
            argv.append("-finstrument-functions")
            command = {"kind": "Fortran source instrumentation", "argv": argv,
                       "cwd": unit["execution"]["cwd"], "environment": unit["execution"]["environment"],
                       "timeout_seconds": 240, "mode": mode, "source": str(source),
                       "original_source": str(original), "object": str(obj)}
            commands.append(command)
            units.append(command)
            replacements[unit["object"]] = str(obj)
        executable = target / "ELMM-budget"
        link = list(receipt["link"]["argv"])
        link[link.index("-o") + 1] = str(executable)
        link = [replacements.get(value, value) for value in link]
        link.append(str(output / "budget_timer.o"))
        if cuda:
            link.append(str(target / "runtime-wrappers.o"))
            link.extend(wrap_flags)
        link.extend(["-lstdc++", "-pthread"])
        commands.append({"kind": "private diagnostic link", "argv": link,
                         "cwd": receipt["link"]["cwd"], "environment": receipt["link"]["environment"],
                         "timeout_seconds": 120, "mode": mode})
        variants[mode] = {"binary": str(executable), "Fortran_units": len(units),
                          "unchanged_original_objects": len(receipt["units"]) - len(units),
                          "original_link_objects": len(receipt["units"]), "source_replacements": replacements,
                          "typed_wrappers": len(wrap_flags), "wrapper_sources": wrapper_sources}
    commands.insert(0, {"kind": "uninstrumented timer compile", "argv": ["/usr/bin/g++-14", "-std=c++17", "-O2", "-fno-instrument-functions", "-c", str(timer), "-o", str(output / "budget_timer.o")], "cwd": str(REPO), "environment": {}, "timeout_seconds": 120, "mode": "timer"})
    plan = {"schema_version": 1, "status": "prepared_no_compiles_or_apps", "diagnostic_only": True,
            "baseline": str(baseline), "baseline_identity_sha256": digest(baseline / "identity.json"),
            "protected_sha256": {**identity["original_artifacts_sha256"], str(timer): digest(timer),
                                 str(Path(__file__)): digest(__file__),
                                 str(REPO / "containers/profiling/cabi_signatures.py"): digest(REPO / "containers/profiling/cabi_signatures.py"),
                                 **{str(path): digest(path) for path in output.rglob("*") if path.is_file()}}, "variants": variants,
            "commands": commands, "phase_cap_seconds": 240,
            "source_semantics": "original bodies retained; only existing full20-field main observer and GCC function entry/exit callbacks",
            "production_objects_modified": False, "performance_samples": False,
            "limitations": ["Instrumented elapsed times are diagnostic only.",
                            "GNU wrappers cover undefined external calls; same-TU/inlined runtime work stays within its enclosing observed interval.",
                            "Private modules only; original module interfaces and numerical CUDA objects are reused."]}
    write(output / "build-plan.json", plan)
    print(output / "build-plan.json", digest(output / "build-plan.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    frozen = subs.add_parser("freeze")
    frozen.add_argument("--prepared-builds", type=Path, required=True)
    frozen.add_argument("--output", type=Path, required=True)
    planned = subs.add_parser("prepare")
    planned.add_argument("--baseline", type=Path, required=True)
    planned.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.prepared_builds, args.output)
    else:
        prepare(args.baseline, args.output)


if __name__ == "__main__":
    main()
