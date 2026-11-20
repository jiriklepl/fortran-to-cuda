"""Command-line orchestration and output publication."""

import argparse
import json
import sys
from pathlib import Path

from compiler.analysis import format_plan
from compiler.driver.kernel_entry import (
    compile_kernel_entry,
    kernel_entry_outputs,
    kernel_entry_report,
    write_kernel_entry,
)
from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources
from compiler.emission.common.resources import read_scoped_runtime
from compiler.frontend import discover_file, lower_file
from compiler.ir import CompilationError, format_ir
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
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="Check every module subroutine and report generation eligibility without writing outputs.",
    )
    parser.add_argument("--emit-scoped-runtime", action="store_true",
                        help="Emit the versioned common memory runtime and build manifest; no input required.")
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
        metavar="FILE",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cpp-output",
        metavar="FILE",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fortran-output",
        metavar="FILE",
        help=argparse.SUPPRESS,
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
    parser.add_argument("--host-threads", type=int, default=4, help="Total host-thread budget, including GPU coordination.")
    args = parser.parse_args()
    if args.require_markers:
        parser.error("--require-markers was removed; omit it and select an entry with --kernel or --list-candidates")
    if any((args.cuda_output, args.cpp_output, args.fortran_output)):
        parser.error("scoped generation emits shared_entry.cu and shared_interface.f90; "
                     "ordinary --cuda-output/--cpp-output/--fortran-output options do not apply")
    if args.emit_scoped_runtime:
        if args.input or args.kernel or args.list_candidates:
            parser.error("--emit-scoped-runtime cannot be combined with input, kernel, or candidate listing")
    elif not args.input:
        parser.error("--input is required unless --emit-scoped-runtime is used")
    elif not args.list_candidates and not args.kernel:
        parser.error("--kernel is required unless --list-candidates is used")
    return args


def _offload_config(args):
    try:
        return OffloadConfig(args.host_threads, False, "direct")
    except ValueError as error:
        raise CompilationError(str(error)) from error


def _list_candidates(source_file: Path, args: argparse.Namespace, options: CompilerOptions) -> None:
    records = []
    for candidate in discover_file(source_file):
        reason = candidate.reason
        if candidate.lowerable:
            try:
                function = lower_file(source_file, candidate.qualified_name)
                function, plan = prepare_function(function, options=options)
                generate_sources(function, plan, common_header=args.common_header, offload_config=_offload_config(args))
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
        options = CompilerOptions(
            opt_level=args.opt_level,
            schedule=args.schedule,
            tile_sizes=args.tile_sizes,
            fallback=args.fallback,
            indexing=args.indexing,
        )
        if args.list_candidates:
            _list_candidates(source_file, args, options)
            return
        function, plan, sources = compile_kernel_entry(
            source_file, args.kernel, options=options, offload_config=_offload_config(args),
            common_header=args.common_header)
        outputs = kernel_entry_outputs(sources, common_header=None if args.no_common_header else args.common_header)
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
        print("\nMemory ownership: caller domain; shared entries use canonical handles and regional coherence.",
              file=stream)

    # Publish only after every stage has validated and generated successfully.
    output_dir = write_kernel_entry(outputs, args.output_dir)

    if args.json:
        print(json.dumps(kernel_entry_report(args.kernel, plan, sources, outputs), indent=2))
    else:
        print(f"Generated files in {output_dir}/")
        for filename in outputs:
            print(f"  {filename}")


if __name__ == "__main__":
    main()
