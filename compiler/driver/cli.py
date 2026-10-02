"""Command-line orchestration and output publication."""

import argparse
from pathlib import Path

from compiler.analysis import format_plan
from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import lower_file
from compiler.ir import CompilationError, format_ir


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
        required=True,
        metavar="NAME",
        help="Entry kernel function name (case-insensitive).",
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
        help="Optimization level: 0 retains source passes; 1 enables checked scalar motion and fusion (default).",
    )
    parser.add_argument(
        "--schedule", choices=("source", "auto"), help="Axis policy (default: auto at level 1, source at 0)."
    )
    parser.add_argument(
        "--tile-sizes", type=_tile_sizes, default=(), metavar="N[,N...]", help="Spatial tiles, fastest axis first."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print normalized IR, transformations, selected schedules, ordered regions, and parallel-legality results.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    source_file = Path(args.input).resolve()
    if not source_file.exists():
        raise SystemExit(f"error: input file not found: {source_file}")

    try:
        options = CompilerOptions(opt_level=args.opt_level, schedule=args.schedule, tile_sizes=args.tile_sizes)
        function = lower_file(source_file, args.kernel)
        function, plan = prepare_function(function, options=options)
        sources = generate_sources(function, plan, common_header=args.common_header)
        common_header = read_common_header()
    except CompilationError as error:
        raise SystemExit(f"error: {error}") from None

    if args.verbose:
        print("Normalized IR:")
        print(format_ir(function))
        print("\nExecution plan and dependence checks:")
        print(format_plan(plan))

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

    print(f"Generated files in {output_dir}/")
    for filename in outputs:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
