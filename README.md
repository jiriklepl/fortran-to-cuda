# Fortran applications on the GPU, from their own source

A source-to-source compiler that moves the loop nests of a Fortran application
to a GPU and keeps the arrays they use on the device between them. The
application's sources stay unchanged. A description says three things:

- which procedures to compile;
- where the arrays may stay on the device;
- how to build the result.

Nothing runs on the device that the compiler did not prove safe; a loop nest it
cannot prove stays the original Fortran. Every build also produces the native
executable from the same sources, as the control a resident run is compared
with.

## Repository structure

```
compiler/          the compiler (Python): kernel entries, resident builds of applications, the runtime
benchmarks/        applications described for the resident build
├── cdu/           the advection kernel of ELMM with a driver that times it
└── elmm/          ELMM, a large-eddy simulation of the atmospheric boundary layer
containers/        the ELMM comparison (native and resident) on GPU nodes: Docker, Apptainer, Slurm
poisson-solver/    submodule: the device Poisson solver of ELMM's resident build
run_on_node.sh     submits an ELMM comparison to a Slurm node (see containers/README.md)
requirements.txt   Python dependencies
```

## Setup

```bash
git submodule update --init -- poisson-solver    # only for ELMM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install -r requirements.txt --group ./compiler/pyproject.toml:dev
```

The compiler needs Python 3.11 or newer. Preparing an application needs GNU
Fortran, which preprocesses the sources. Building it also needs:

- a C++ compiler;
- the CUDA toolkit (`nvcc`);
- CMake 3.24 or newer;
- Ninja.

## Adapting an application

`python -m compiler.resident` prepares and builds an application from its
description:

```bash
python -m compiler.resident benchmarks/cdu/cdu.toml --output out/cdu \
    --fortran gfortran --cxx g++ --nvcc nvcc --architecture 86
out/cdu/build/cdu-native    # the original program
out/cdu/build/cdu           # its loop nests on the GPU
```

A description is TOML and has these parts:

- its source trees, given as `--input NAME=PATH` and checked against their
  pinned revisions;
- the sources the compiler reads, and the procedures whose loop nests become
  entries;
- edits that connect the entries to the application's code, such as residency
  directives like `!$fort enter`, anchored on the application's own text;
- the CMake build of the native and the resident executables.

[`benchmarks/README.md`](benchmarks/README.md) walks through the two examples.
The format is documented in `compiler/resident/application.py`, and
[`compiler/README.md`](compiler/README.md) documents what the preparation does,
the directives, the supported Fortran and the compiler's other entry point
(one kernel for a caller that owns its residency domain).

## The ELMM comparison

`benchmarks/elmm/elmm.toml` describes ELMM's time step on the GPU: about 270
entries, with a device Poisson solver for the pressure. `./run_on_node.sh` and
`containers/` build ELMM from it on a GPU node, run the native and the resident
executables on the same case, compare their fields and record the times; see
[`containers/README.md`](containers/README.md).

## Tests

```bash
python -m pytest compiler/tests -n auto     # also builds and runs CUDA programs where nvcc is installed
python -m pytest containers/tests
(cd compiler && python -m ruff check .)
```
