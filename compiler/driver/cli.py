"""Command-line orchestration and output publication."""

import argparse
import json
import sys
from pathlib import Path

from compiler.analysis import format_plan
from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import discover_file, lower_file
from compiler.ir import CompilationError, format_ir
from compiler.memory import format_memory, plan_memory


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
        required=True,
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
    args = parser.parse_args()
    if not args.list_candidates and not args.kernel:
        parser.error("--kernel is required unless --list-candidates is used")
    return args


def _list_candidates(source_file: Path, args: argparse.Namespace, options: CompilerOptions) -> None:
    records = []
    for candidate in discover_file(source_file, require_markers=args.require_markers):
        reason = candidate.reason
        if candidate.lowerable:
            try:
                function = lower_file(source_file, candidate.qualified_name, require_markers=args.require_markers)
                function, plan = prepare_function(function, options=options)
                generate_sources(function, plan, common_header=args.common_header)
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
        function = lower_file(source_file, args.kernel, require_markers=args.require_markers)
        function, plan = prepare_function(function, options=options)
        sources = generate_sources(function, plan, common_header=args.common_header)
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
                    "memory_plan": format_memory(plan_memory(plan, function.parameters, acquisition_policy="pooled")),
                    "outputs": list(outputs),
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
