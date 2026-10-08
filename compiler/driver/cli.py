"""Command-line orchestration and output publication."""

import argparse
import json
import sys
from pathlib import Path

from compiler.analysis import format_plan
from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.emission.common.resources import read_scoped_runtime
from compiler.frontend import analyze_source_effects, discover_file, lower_file
from compiler.ir import CompilationError, format_ir
from compiler.memory import format_memory, plan_memory
from compiler.offload.config import OffloadConfig


def _tile_sizes(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("tile sizes must be comma-separated positive integers") from error
    if not result or any(size <= 0 for size in result):
        raise argparse.ArgumentTypeError("tile sizes must be comma-separated positive integers")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m compiler",
        description="Generate CUDA / C++ / Fortran code from a Fortran stencil kernel.",
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="FILE",
        help="Input Fortran source file.",
    )
    parser.add_argument(
        "--kernel",
        "-k",
        metavar="NAME",
        help="Entry subroutine name or module::name (case-insensitive); required unless listing candidates.",
    )
    parser.add_argument(
        "--require-markers",
        action="store_true",
        help="Require the legacy '! kernels' file and '! kernel' routine markers.",
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="Check every module subroutine and report generation eligibility without writing outputs.",
    )
    parser.add_argument("--emit-scoped-runtime", action="store_true",
                        help="Emit the versioned common memory runtime and build manifest; no input required.")
    parser.add_argument("--analyze-effects", action="store_true",
                        help="Report bounded native procedure effects separately from GPU eligibility; write no outputs.")
    parser.add_argument("--source-file", action="append", default=[], metavar="FILE",
                        help="Additional source available to native effect analysis; repeat for independent modules.")
    parser.add_argument("--effect-contracts", metavar="FILE",
                        help="Explicit versioned generic contracts for opaque native calls in effect analysis.")
    parser.add_argument(
        "--json", action="store_true", help="Print candidate eligibility or generation results as JSON."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        metavar="DIR",
        help="Directory for all generated output files (default: current directory).",
    )
    parser.add_argument(
        "--cuda-output",
        default="generated_code.cu",
        metavar="FILE",
        help="CUDA output filename (default: generated_code.cu).",
    )
    parser.add_argument(
        "--cpp-output",
        default="generated_cpp_impl.cpp",
        metavar="FILE",
        help="C++ implementation output filename (default: generated_cpp_impl.cpp).",
    )
    parser.add_argument(
        "--fortran-output",
        default="generated_interface.f90",
        metavar="FILE",
        help="Fortran interface output filename (default: generated_interface.f90).",
    )
    parser.add_argument(
        "--common-header",
        default="common_functions.cuh",
        metavar="FILE",
        help="Common functions header output filename (default: common_functions.cuh).",
    )
    parser.add_argument(
        "--no-common-header",
        action="store_true",
        help="Skip writing the common_functions.cuh header.",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        choices=(0, 1),
        default=1,
        help="Optimization level: 0 retains source passes; 1 enables checked scalar motion, fusion, and addressing (default).",
    )
    parser.add_argument(
        "--schedule", choices=("source", "auto"), help="Axis policy (default: auto at level 1, source at 0)."
    )
    parser.add_argument(
        "--indexing",
        choices=("source", "auto"),
        help="Array addressing: source INTEGER arithmetic or proved wide indices (default: auto at level 1, source at 0).",
    )
    parser.add_argument(
        "--tile-sizes", type=_tile_sizes, default=(), metavar="N[,N...]", help="Spatial tiles, fastest axis first."
    )
    parser.add_argument(
        "--fallback",
        choices=("error", "host"),
        default="error",
        help="Unproved parallel regions: reject or execute on the host (default: error).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print normalized IR, transformations, schedules, addressing proofs, ordered regions, legality, and memory operations.",
    )
    parser.add_argument("--gpu-policy", choices=("always", "sections", "auto", "chunked", "hybrid"),
                        default="always", help="Opt-in ordinary-call transfer/execution policy.")
    parser.add_argument("--memory-model", choices=("call", "scoped"), default="call",
                        help="Emit shared-buffer numerical entry interfaces with scoped; requires sections or auto.")
    parser.add_argument("--calibration-profile", metavar="FILE", help="Explicit offline hardware calibration JSON.")
    parser.add_argument("--host-threads", type=int, default=4, help="Total host-thread budget, including GPU coordination.")
    parser.add_argument("--gpu-collective", action="store_true",
                        help="Opt-in entry is called by every thread of one existing OpenMP team.")
    args = parser.parse_args()
    if args.analyze_effects and (args.list_candidates or args.emit_scoped_runtime):
        parser.error("--analyze-effects cannot be combined with candidate listing or runtime export")
    if (args.source_file or args.effect_contracts) and not args.analyze_effects:
        parser.error("--source-file and --effect-contracts require --analyze-effects")
    if args.emit_scoped_runtime:
        if args.input or args.kernel or args.list_candidates:
            parser.error("--emit-scoped-runtime cannot be combined with input, kernel, or candidate listing")
    elif not args.input:
        parser.error("--input is required unless --emit-scoped-runtime is used")
    elif not args.list_candidates and not args.kernel:
        parser.error("--kernel is required unless --list-candidates is used")
    return args


def _offload_config(args):
    profile, reason = None, None
    if args.calibration_profile:
        from compiler.offload.profile import compiler_identity, load_profile
        try:
            profile = load_profile(args.calibration_profile, cpu_threads=args.host_threads)
            compiler_identity(profile)
        except (OSError, ValueError) as error:
            profile = None
            reason = str(error)
    elif args.gpu_policy in {"auto", "hybrid"}:
        reason = "no calibration profile supplied; automatic policy retains native execution"
    try:
        return OffloadConfig(args.gpu_policy, profile, args.host_threads, args.gpu_collective, reason)
    except ValueError as error:
        raise CompilationError(str(error)) from error


def _list_candidates(source_file: Path, args: argparse.Namespace, options: CompilerOptions) -> None:
    records = []
    for candidate in discover_file(source_file, require_markers=args.require_markers):
        reason = candidate.reason
        if candidate.lowerable:
            try:
                function = lower_file(source_file, candidate.qualified_name, require_markers=args.require_markers)
                function, plan = prepare_function(function, options=options)
                generate_sources(function, plan, common_header=args.common_header, offload_config=_offload_config(args),
                                 memory_model=args.memory_model)
            except CompilationError as error:
                reason = str(error)
        records.append(
            {
                "module": candidate.module,
                "name": candidate.name,
                "qualified_name": candidate.qualified_name,
                "path": candidate.location.path,
                "line": candidate.location.line,
                "annotated": candidate.annotated,
                "supported": reason is None,
                "reason": reason,
            }
        )
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        for record in records:
            status = "supported" if record["supported"] else f"rejected: {record['reason']}"
            print(f"{record['qualified_name']}: {status}")


def main() -> None:
    args = _parse_args()

    if args.emit_scoped_runtime:
        outputs, manifest = read_scoped_runtime()
        outputs["scoped-runtime.json"] = json.dumps(manifest, indent=2) + "\n"
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in outputs.items():
            (output_dir / name).write_text(content, encoding="utf-8")
        report = {"supported": True, "reason": None, "outputs": list(outputs), "runtime": manifest}
        print(json.dumps(report, indent=2) if args.json else f"Generated shared memory runtime in {output_dir}/")
        return

    source_file = Path(args.input).resolve()
    try:
        if not source_file.exists():
            raise CompilationError(f"input file not found: {source_file}")
        if args.analyze_effects:
            contracts = None
            if args.effect_contracts:
                try:
                    document = json.loads(Path(args.effect_contracts).read_text())
                    if not isinstance(document, dict) or document.get("schema_version") != 1 or not isinstance(document.get("procedures"), dict):
                        raise ValueError("expected schema_version 1 and procedures object")
                    contracts = document["procedures"]
                except (OSError, ValueError) as error:
                    raise CompilationError(f"invalid effect contracts: {error}") from error
            effects = analyze_source_effects([source_file, *args.source_file], args.kernel, contracts=contracts)
            report = {"kernel": args.kernel, "supported": True, "reason": None, "outputs": [], "effects": effects}
            print(json.dumps(report, indent=2) if args.json else json.dumps(effects, indent=2))
            return
        options = CompilerOptions(
            opt_level=args.opt_level,
            schedule=args.schedule,
            tile_sizes=args.tile_sizes,
            fallback=args.fallback,
            indexing=args.indexing,
            gpu_policy=args.gpu_policy,
            memory_model=args.memory_model,
        )
        if args.list_candidates:
            _list_candidates(source_file, args, options)
            return
        function = lower_file(source_file, args.kernel, require_markers=args.require_markers)
        function, plan = prepare_function(function, options=options)
        sources = generate_sources(function, plan, common_header=args.common_header, offload_config=_offload_config(args),
                                   memory_model=args.memory_model)
        if set(sources.artifacts) & {args.cuda_output, args.cpp_output, args.fortran_output, args.common_header}:
            raise CompilationError("ordinary output filename conflicts with a shared runtime/entry artifact")
        common_header = read_common_header()
    except CompilationError as error:
        if args.json and not args.list_candidates:
            print(
                json.dumps({"kernel": args.kernel, "supported": False, "reason": str(error), "outputs": []}, indent=2)
            )
        raise SystemExit(f"error: {error}") from None

    if args.verbose:
        stream = sys.stderr if args.json else sys.stdout
        print("Normalized IR:", file=stream)
        print(format_ir(function), file=stream)
        print("\nExecution plan and dependence checks:", file=stream)
        print(format_plan(plan), file=stream)
        print("\nMemory operations (explicit sessions):", file=stream)
        print(format_memory(plan_memory(plan, function.parameters)), file=stream)
        print("\nMemory operations (ordinary CUDA calls):", file=stream)
        print(format_memory(plan_memory(plan, function.parameters, acquisition_policy="pooled")), file=stream)

    # Publish only after every stage has validated and generated successfully.
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        args.cuda_output: sources.cuda,
        args.cpp_output: sources.cpp,
        args.fortran_output: sources.fortran,
    }
    if not args.no_common_header:
        outputs[args.common_header] = common_header
    outputs.update(sources.artifacts)
    for filename, code in outputs.items():
        (output_dir / filename).write_text(code, encoding="utf-8")

    if args.json:
        print(
            json.dumps(
                {
                    "kernel": args.kernel,
                    "supported": True,
                    "reason": None,
                    "parallel_regions": len(plan.regions),
                    "execution_plan": format_plan(plan),
                    "memory_plan": (format_memory(plan_memory(plan, function.parameters, acquisition_policy="pooled"))
                                    if sources.offload is None else
                                    "Ordinary-call policy " + sources.offload["policy"] +
                                    "; runtime footprints and choices are described by offload.analysis and FORT_OFFLOAD_TRACE."),
                    "outputs": list(outputs),
                    **({"offload": sources.offload} if sources.offload is not None else {}),
                    **({"scoped": sources.scoped} if sources.scoped is not None else {}),
                },
                indent=2,
            )
        )
    else:
        print(f"Generated files in {output_dir}/")
        for filename in outputs:
            print(f"  {filename}")


if __name__ == "__main__":
    main()
