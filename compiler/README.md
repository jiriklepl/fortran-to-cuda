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
