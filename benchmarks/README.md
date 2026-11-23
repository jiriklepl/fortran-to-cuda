# Benchmarks: applications described for the resident build

Each directory holds one application, adapted to the GPU by a description that
`python -m compiler.resident` reads. The description is TOML. It names:

- the source trees;
- the sources the compiler reads;
- the procedures whose loop nests become entries;
- the edits that connect the entries to the application's own code;
- the build.

The application's sources stay as they are. The format is documented in
`compiler/resident/application.py`, and "Resident build of an application" in
[`compiler/README.md`](../compiler/README.md) explains what the preparation
does.

| Directory | Application | Source trees it needs |
|---|---|---|
| `cdu/` | The zonal momentum advection kernel of ELMM (four OpenMP loop nests), with a driver that times it | none: its sources are here |
| `elmm/` | ELMM, a large-eddy simulation of the atmospheric boundary layer: 69 sources, about 270 entries | pinned ELMM, PoisFFT and VkFFT checkouts, and the `poisson-solver` submodule |

Every build writes two executables, both compiled with the compiler's numerical
options:

- the resident one, from the prepared sources;
- the native one, from the original sources, as its control.

The compiler's tests load every description here and prepare `cdu/`
(`compiler/tests/test_resident_application_description.py`).

## The small example

```bash
python -m compiler.resident benchmarks/cdu/cdu.toml --output out/cdu \
    --fortran gfortran --cxx g++ --nvcc nvcc --architecture 86
out/cdu/build/cdu-native
out/cdu/build/cdu
```

Without a GPU, `--prepare-only` (with only `--fortran`) writes the resident
sources to `out/cdu/resident/` and compiles nothing. The description selects
the kernel module's procedures and inserts three directives into the driver:

- `!$fort enter` before the warm-up;
- `!$fort leave` after the timed loop;
- `!$fort close` before the arrays are deallocated.

Both executables print the time per iteration and a checksum of the result,
which is the same for both.

## ELMM

`elmm/elmm.toml` pins the revisions of ELMM, PoisFFT and VkFFT in its
`[inputs]`. The build is given a checkout of each:

```bash
python -m compiler.resident benchmarks/elmm/elmm.toml \
    --input elmm=PATH/elmm --input PoisFFT=PATH/PoisFFT \
    --input VkFFT=PATH/VkFFT --input solver=poisson-solver \
    --output out/elmm --variant dp --jobs 6 \
    --fortran gfortran --cxx g++ --nvcc nvcc --architecture 86
```

The inputs are checked before anything is copied:

- VkFFT must be a clean checkout of its pinned revision.
- The other inputs must be checkouts of their pinned revision or of a
  descendant.

`--variant sp` builds single precision. `--fast-fp` replaces the strict
options: it allows FMA contraction and fast CUDA division and square root. The build writes four
executables:

- `ELMM`, the resident one, and `ELMM-native`, its control. Both report the
  wall-clock time of initialization, of the time-stepping loop and of
  finalization.
- `ELMM-full20` and `ELMM-native-full20`, which also dump every field and
  running average in full at the end.

[`containers/README.md`](../containers/README.md) runs this build on a GPU node
and compares the executables' fields and times.

What the description contains:

- **Sources and selections.** The sources are ELMM's serial build in the
  order of its SConstruct, followed by PoisFFT's. Ten selections name ELMM's
  time step:
  - the Runge-Kutta stages, the boundaries, advection, the filter and the
    pressure step;
  - the diagnostics, the outputs, the forces and the wall model.

  Each selection has a minimum number of nests, so lost coverage fails the
  build.
- **The domain.** The pressure solver keeps state of its own on the device.
  Its Fortran and CUDA sources are in `sources/`, and they are the only
  sources the description adds. The description names their module as the
  residency domain and lists the arrays it binds itself.
- **Edits.** They are anchored on ELMM's text and use Fortran files in
  `templates/`:
  - one `[[patch]]` sums the pressure reference plane serially, for both
    builds, so that a native run repeats itself;
  - two `[[rewrite]]` edits accumulate the mean friction velocity at the
    ground over module arrays, which an entry can bind
    (`templates/ground-ustar-flat.f90`);
  - the `[[edit]]` entries call the device Poisson solver, open and close the
    epoch, and decide when ELMM's configuration is the supported one
    (`templates/rk-supported.f90`).
- **The build.** A CMake prologue adds the solver, with VkFFT, to the
  build. Besides it come the system libraries both executables link and the
  one per-file option of ELMM's own build.
