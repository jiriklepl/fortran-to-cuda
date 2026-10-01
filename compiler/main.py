import argparse
from pathlib import Path

from compiler.analysis import build_execution_plan, format_plan
from compiler.emission import generate_sources
from compiler.frontend import lower_file
from compiler.ir import CompilationError, format_ir

_TEMPLATES_DIR = Path(__file__).resolve().parent / "cuda_generation" / "templates"


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
        "--verbose",
        "-v",
        action="store_true",
        help="Print normalized IR, ordered regions, and parallel-legality results.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    source_file = Path(args.input).resolve()
    if not source_file.exists():
        raise SystemExit(f"error: input file not found: {source_file}")

    try:
        function = lower_file(source_file, args.kernel)
        plan = build_execution_plan(function)
        sources = generate_sources(function, plan, common_header=args.common_header)
        common_header = (_TEMPLATES_DIR / "common_functions.cuh").read_text(encoding="utf-8")
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
