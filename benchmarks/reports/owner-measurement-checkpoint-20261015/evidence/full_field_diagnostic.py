"""Plan/relink and compare complete-field diagnostics; never launch ELMM.

Use each mode's own authenticated application build. Diagnostic executables
retain all numerical objects and add only the existing final-lifetime dumps.
Application execution and sequencing belong to the caller. No timing result
from these executables establishes performance or qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "benchmarks/reports/native-diagnostics-20261014/evidence"
DRIVER = {"dtype": "float64", "field_count": 20,
          "metadata_format": "storage-shape-bounds-v1", "atol": 1e-6, "rtol": 1e-5}
NAMES = {"U", "V", "W", "Pr", "Viscosity", "U2", "V2", "W2", "Ustar", "Vstar", "Wstar",
         "Temperature", "Moisture", "Scalar", "TDiff", "U_avg", "V_avg", "W_avg", "Pr_avg", "Scalar_avg"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def imports(pipeline):
    pipeline = Path(pipeline).resolve(strict=True)
    sys.path.insert(0, str(pipeline))
    from elmm_pipeline import build_receipts, validation
    from elmm_pipeline.campaigns import driver_link, halos
    for module in (build_receipts, driver_link, halos, validation):
        if not Path(module.__file__).resolve().is_relative_to(pipeline):
            raise ValueError("Imported validation/build helper outside requested pipeline")
    return build_receipts, driver_link, halos, validation


def check_files(files):
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError("Frozen diagnostic input changed: " + path)


def plan(args):
    receipts, linker, _halos, _validation = imports(args.pipeline_root)
    receipt_path = args.build_receipt.resolve(strict=True)
    receipt = receipts.require_build_receipts(receipt_path)
    binary = Path(receipt["binary"])
    main_units = [unit for unit in receipt["units"] if Path(unit["object"]) == binary.parent / "main.o"]
    if len(main_units) != 1:
        raise ValueError("Exactly one original main object required")
    main = main_units[0]
    archived = ARCHIVE / "dump-main.f90"
    source = archived.read_text()
    pattern = r"(?ms)^  block\n.*?^  end block\n"
    blocks = [block for block in re.findall(pattern, source) if "fields-" in block]
    original = re.sub(pattern, lambda match: "" if "fields-" in match[0] else match[0], source)
    if len(blocks) != 2 or original != Path(main["source"]).read_text():
        raise ValueError("Archived diagnostic main differs beyond its two dump blocks")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    diagnostic_source = output / "dump-main.f90"
    diagnostic_source.write_text(source)
    relink = linker.derive_driver_link(binary, fortran=main["execution"]["compiler"]["path"],
        source=diagnostic_source, object_path=output / "dump-main.o", output=output / "ELMM-fields")
    if (relink["provenance"]["original_compile_argv"] != main["execution"]["argv"]
            or relink["provenance"]["original_link_argv"] != receipt["link"]["argv"]):
        raise ValueError("Logged relink commands disagree with executed build inventory")
    paths = [Path(__file__), archived, diagnostic_source, receipt_path,
             Path(receipts.__file__), Path(linker.__file__),
             ARCHIVE / "channel-simple-full-fields.json"]
    modules = sorted(binary.parent.rglob("*.mod"))
    record = {"schema_version": 1, "mode": args.mode,
        "kind": "complete-field-diagnostic-relink-plan", "performance_measured": False,
        "qualification": False, "pipeline_root": str(args.pipeline_root.resolve()),
        "application_build_receipt": str(receipt_path), "application_build_identity": receipt["identity"],
        "application_binary": str(binary), "diagnostic_binary": str(output / "ELMM-fields"),
        "driver": DRIVER, "field_names": sorted(NAMES), "relink": relink,
        "compile_environment": main["execution"]["environment"],
        "link_environment": receipt["link"]["environment"],
        "module_inputs": {str(path): digest(path) for path in modules},
        "files_sha256": {str(path.resolve()): digest(path) for path in paths},
        "numerical_objects_changed": False,
        "tolerance_provenance": str(ARCHIVE / "channel-simple-full-fields.json")}
    write(output / "plan.json", record)
    print(output / "plan.json")


def build(args):
    plan_path = args.plan.resolve(strict=True)
    record = json.loads(plan_path.read_text())
    receipts, linker, _halos, _validation = imports(record["pipeline_root"])
    check_files(record["files_sha256"])
    check_files(record["module_inputs"])
    receipt = receipts.require_build_receipts(record["application_build_receipt"])
    if receipt["identity"] != record["application_build_identity"]:
        raise ValueError("Application build changed after diagnostic planning")
    relink = record["relink"]
    fresh = linker.derive_driver_link(record["application_binary"],
        fortran=relink["provenance"]["fortran_compiler"],
        source=plan_path.parent / "dump-main.f90",
        object_path=plan_path.parent / "dump-main.o", output=record["diagnostic_binary"])
    if fresh != relink:
        raise ValueError("Diagnostic relink changed after planning")
    output = plan_path.parent
    recorder = receipts.CommandRecorder(output / "command-receipts")
    report = {"status": "running", "plan_sha256": digest(plan_path), "performance_measured": False}
    write(output / "build.json", report)
    try:
        with (output / "build.log").open("x") as log:
            for key, environment_key in (("compile_argv", "compile_environment"), ("link_argv", "link_environment")):
                environment = dict(os.environ)
                for name, value in record[environment_key].items():
                    if value is None:
                        environment.pop(name, None)
                    else:
                        environment[name] = value
                result = recorder.run(relink[key], cwd=relink["cwd"], env=environment,
                    stdout=log, stderr=log, additional_inputs=record["module_inputs"])
                if result.returncode:
                    raise RuntimeError("Diagnostic relink failed; inspect build.log")
        receipts.read_receipts(output / "command-receipts")
        receipts.require_build_receipts(record["application_build_receipt"])
        check_files(record["files_sha256"])
        check_files(record["module_inputs"])
        report.update(status="passed", binary=record["diagnostic_binary"],
                      binary_sha256=digest(record["diagnostic_binary"]))
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write(output / "build.json", report)


def compare(args):
    _receipts, _linker, halos, validation = imports(args.pipeline_root)
    if args.output.exists():
        raise ValueError("Existing comparison evidence is immutable")
    spec = importlib.util.spec_from_file_location("diagnostic_vtk_outputs", ROOT / "containers/sgs_experiment.py")
    outputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(outputs)
    report = {"schema_version": 1, "status": "running", "performance_measured": False,
              "qualification": False, "driver": DRIVER, "runs": [], "vtk": {}}
    try:
        observed = []
        for role, directory in (("native", args.native), ("candidate", args.candidate)):
            log = directory / args.log_name
            timing = validation.validate_log(log.read_text(), expected_completed_steps=args.expected_steps)
            if not timing["natural_end"]:
                raise ValueError("Complete-case diagnostic requires natural completion")
            data = halos.fields(directory, DRIVER)
            if data.keys() != {"fields-" + name + ".bin" for name in NAMES}:
                raise ValueError("Complete-field names differ from the maintained 20-field inventory")
            observed.append(data)
            paths = [log, *directory.glob("fields-*.bin"), *directory.glob("fields-*.shape")]
            report["runs"].append({"mode": role, "directory": str(directory.resolve()), "completion": timing,
                "files_sha256": {str(path.resolve()): digest(path) for path in sorted(paths)}})
        if report["runs"][0]["completion"]["completed_steps"] != report["runs"][1]["completion"]["completed_steps"]:
            raise ValueError("Native and candidate complete different workloads")
        report["comparisons"] = halos.compare_fields(observed[1], observed[0], DRIVER)
        reference = outputs.read_outputs(args.reference_vtk, validation.read_vtk)
        report["vtk_reference"] = {str((args.reference_vtk / name).resolve()): digest(args.reference_vtk / name)
                                   for name in reference}
        for role, directory in (("native", args.native), ("candidate", args.candidate)):
            report["vtk"][role] = outputs.compare_outputs(reference,
                outputs.read_outputs(directory / "output", validation.read_vtk), validation.compare_vtk)
            report["vtk"][role]["files_sha256"] = {name: digest(directory / "output" / name) for name in reference}
        report["helpers_sha256"] = {str(Path(module.__file__).resolve()): digest(module.__file__)
                                    for module in (halos, validation, outputs)}
        passed = all(row["passed"] for row in report["comparisons"].values()) and all(
            row["passed"] for row in report["vtk"].values())
        report["status"] = "passed" if passed else "failed"
        if not passed:
            raise RuntimeError("Complete-field/VTK comparison failed at unchanged tolerances")
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        write(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan", help="validate inputs and write exact relink commands; no compiler execution")
    planning.add_argument("--mode", required=True)
    planning.add_argument("--build-receipt", type=Path, required=True)
    planning.add_argument("--output", type=Path, required=True)
    planning.add_argument("--pipeline-root", type=Path, default=ROOT / "elmm-pipeline")
    building = sub.add_parser("build", help="execute only the recorded compile/link commands")
    building.add_argument("--plan", type=Path, required=True)
    comparison = sub.add_parser("compare", help="read existing complete runs; never execute an application")
    for name in ("native", "candidate", "reference-vtk", "output"):
        comparison.add_argument("--" + name, type=Path, required=True)
    comparison.add_argument("--expected-steps", type=int)
    comparison.add_argument("--log-name", default="stdout.log")
    comparison.add_argument("--pipeline-root", type=Path, default=ROOT / "elmm-pipeline")
    args = parser.parse_args()
    {"plan": plan, "build": build, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
