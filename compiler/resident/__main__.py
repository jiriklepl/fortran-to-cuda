"""Build an application described in TOML for the resident GPU path, with its native control.

    python -m compiler.resident benchmarks/elmm/elmm.toml --input elmm=PATH ... --output DIR \\
        --fortran gfortran [--variant sp] [--prepare-only] --cxx g++ --nvcc nvcc --architecture 86

See ``compiler.resident.application`` for the description.
"""
import argparse
import json
import sys
from pathlib import Path

from compiler.ir import CompilationError
from compiler.resident.application import build, load, prepare
from compiler.resident.build import compiler_path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m compiler.resident", description=__doc__.split("\n\n")[0])
    parser.add_argument("application", type=Path, help="The application's TOML description")
    parser.add_argument("--input", action="append", default=[], metavar="NAME=PATH",
                        help="A source tree the description names; repeat for each")
    parser.add_argument("--output", type=Path, required=True, help="A new directory for the build")
    parser.add_argument("--variant", help="A variant of the description; its first by default")
    # A string keeps "./g++" apart from "g++", which Path would make one name.
    path_help = "A path, or a name to look up on PATH"
    parser.add_argument("--fortran", required=True, help=path_help)
    parser.add_argument("--prepare-only", action="store_true", help="Write the resident sources, compile nothing")
    parser.add_argument("--build-prepared", action="store_true", help="Compile what an earlier run prepared")
    parser.add_argument("--cxx", help=path_help)
    parser.add_argument("--nvcc", help=path_help)
    parser.add_argument("--architecture", help="The CUDA architecture, such as 86")
    parser.add_argument("--jobs", type=int, default=2, help="Entries compiled and build steps run at a time")
    parser.add_argument("--fast-fp", action="store_true",
                        help="FMA contraction and fast CUDA division and square root instead of strict options")
    args = parser.parse_args(argv)
    try:
        if args.jobs < 1:
            raise ValueError("Jobs must be positive")
        if args.prepare_only and args.build_prepared:
            raise ValueError("Choose preparation or building prepared sources")
        if not args.prepare_only:
            if None in (args.cxx, args.nvcc, args.architecture):
                raise ValueError("Building requires --cxx, --nvcc and --architecture")
            # Found before preparing, which can take minutes, rather than when CMake is given them.
            for tool in (args.fortran, args.cxx, args.nvcc):
                compiler_path(tool)
        given = {}
        for item in args.input:
            name, separator, path = item.partition("=")
            if not separator or not name or not path:
                raise ValueError("An input is NAME=PATH: " + item)
            given[name] = Path(path)
        application = load(args.application)
        if args.build_prepared:
            prepared = json.loads((args.output / "prepared.json").read_text())
        else:
            prepared = prepare(application, given, args.output, variant=args.variant or next(iter(application.variants)),
                               fortran=args.fortran, jobs=args.jobs)
        if not args.prepare_only:
            build(application, prepared, given, args.output, fortran=args.fortran, cxx=args.cxx, nvcc=args.nvcc,
                  architecture=args.architecture, jobs=args.jobs, strict_fp=not args.fast_fp)
    except (CompilationError, ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
