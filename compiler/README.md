# Fortran-to-CUDA/C++ Compiler

A source-to-source compiler that reads an annotated Fortran stencil module and
generates three files:

- **CUDA** (`generated_code.cu`) — each do-loop nest becomes a `__global__` kernel
  with automatic host↔device memory management.
- **C++** (`generated_cpp_impl.cpp`) — a flat, single-function kernel callable
  directly from Fortran (no GPU required).
- **Fortran interface** (`generated_interface.f90`) — `iso_c_binding` wrapper that
  keeps every call site in the driver unchanged.

---

## Quick Start

```bash
pip install fparser

python -m compiler \
    --input  fortran-stencils/elmm_cdu.f90 \
    --kernel CDU \
    --output-dir out/
```

---

## CLI Reference

```bash
python -m compiler --input FILE --kernel NAME [options]
```

| Flag | Short | Default | Description |
| ---- | ----- | ------- | ----------- |
| `--input FILE`     | `-i` | _(required)_ | Fortran source file |
| `--kernel NAME`    | `-k` | _(required)_ | Entry kernel name (case-sensitive) |
| `--output-dir DIR` | `-o` | `.` (cwd)    | Output directory (created if absent) |
| `--no-common-header` | | off | Skip copying `common_functions.cuh` |
| `--verbose`        | `-v` | off | Print parsing and kernel-graph info |

Individual output filenames can be overridden with `--cuda-output`,
`--cpp-output`, `--fortran-output`, and `--common-header`.

---

## Input Format

The **first line** of the file must be exactly `! kernels`.  Each subroutine to
compile must be preceded by `! kernel`:

```fortran
! kernels
module MomentumAdvection
  ...
contains
  ! kernel
  subroutine CDU(...)
    ...
  end subroutine CDU
end module MomentumAdvection
```

The entry kernel (given via `--kernel`) may call other `! kernel`-annotated
subroutines; the compiler inlines them all.

### Supported

- 3-D `(:,:,:)` assumed-shape arrays, `real(knd)` and `integer` scalars
- `intent(in/out/inout)` on all arguments
- `do` loops up to 3 levels deep
- Arithmetic expressions and scalar pre-computations before loops
- Calls to other `! kernel` subroutines (inlined into the output)

### Not yet supported

- Multiple source files per invocation
- `if`/`select case` inside kernels, reductions
- Fortran intrinsics beyond arithmetic operators

---

## Development

Run these commands from the **repository root** with Python 3.10 or newer.
The `dev` dependency group includes the parser, pytest, and Ruff; the smoke tests
do not need CUDA, a GPU, or a native Fortran/C++ compiler.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group ./compiler/pyproject.toml:dev

python -m pytest compiler/tests
python -m ruff check compiler
python -m ruff format --check compiler
```

The two smoke cases in `tests/` generate CUDA, C++, and Fortran interface files
from tiny annotated Fortran modules. They only check that the CLI runs and writes
the expected artifacts into pytest's temporary directories. They provide a place
to add future tests; they do not compile or execute the generated code or attempt
comprehensive library coverage. Add future Fortran inputs under `tests/fixtures/`.

Pytest uses [importlib mode](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
and discovers the compiler tests using `pyproject.toml`.
[Ruff](https://docs.astral.sh/ruff/configuration/) provides linting, import sorting
checks, and formatting with an 120-column line length and double quotes. Its
configuration applies to `compiler/`. The commands above only report issues and
return a nonzero status when existing code does not conform, including the
unfinished `for_loops.py` stub. Existing source cleanup is deferred.
