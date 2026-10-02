# Loki and PSyclone: performance with an unchanged physics source

Both tools work on the existing CDU/CDV/CDW Fortran modules. The recipes in
[`tools/`](../tools/) leave the originals untouched, retain their module/procedure
interfaces, and use the tools' real transformations to produce optimized serial
Fortran or GPU Fortran/OpenACC. No GOcean/LFRic DSL rewrite is needed.

The criterion is **useful performance for little application-editing effort**,
not reproducing this repository's compiler internals. In particular, the
default recipes fuse the four pointwise passes into one loop nest and one GPU
kernel. The stencil expressions come from the originals; there is no separately
handwritten replacement formula. The runner also measures an unfused variant
and the existing handwritten OpenACC implementation to expose the value of the
extra transformations.

## Practical comparison

| Aspect | Loki recipe | PSyclone recipe | Local compiler |
| --- | --- | --- | --- |
| Original kernel source edits | 0 | 0 | Uses existing `! kernel` annotations |
| Existing Fortran caller/ABI changes | 0 | 0 | 0 |
| Tested GPU route | Generated Fortran/OpenACC → NVFortran | Generated Fortran/OpenACC → NVFortran | Generated CUDA → nvcc |
| Main optimization | Built-in inlining and pragma-directed loop fusion | Inlining, dependency-checked motion/fusion, inferred data clauses | Current ordered four-kernel implementation |
| Work required here | A reusable Python recipe; declare/check this stencil family's pointwise-output contract | A reusable Python recipe; accommodate unsupported `CONTIGUOUS` in the parser | Existing CLI works directly |
| Control | Editable generated Fortran; fusion switch, vector length, loop/data pragmas | Editable generated Fortran; fusion switch and PSyIR loop/data transformation options | Editable generated `.cu`; generated launch/runtime code |
| Keep data across calls | Two caller data directives | Two caller data directives | Current generated wrapper allocates/copies per call |

The tool modules contain only transformation recipes and compatibility handling.
The shared [`harness/`](../harness/) owns the CLI, input validation, provenance,
drivers and measurement; [`tests/`](../tests/) checks both layers. No changes
to either upstream tool, the local compiler or the original kernels are needed.
For a one-off port, the existing handwritten OpenACC variant is also an
important low-effort baseline: these libraries become more useful when the
optimization recipe must be reapplied as the scientific source changes.

Loki is a programmable compiler toolkit. Its generic fusion follows explicit
requests; this adapter checks equal iteration spaces and pointwise output
accesses before requesting fusion. It is not a general dependence prover.
PSyclone's generic transformations provide dependency validation for motion and
fusion, and its `ACCDataTrans` derives the three input uploads and output copy.
Neither comparison needs to match the local compiler's ISL approach.
See [Loki's scope](https://sites.ecmwf.int/docs/loki/main/index.html),
[Loki programming models](https://sites.ecmwf.int/docs/loki/main/programming_models.html),
and [PSyclone transformations](https://psyclone.readthedocs.io/en/stable/user_guide/transformations.html).

### CUDA editing versus directive control

The tested paths emit editable **Fortran/OpenACC**, with explicit loop mapping
and data regions. NVFortran then generates GPU machine code. They do not emit
a standalone `.cu` file in this comparison, but they do provide the requested
directive/script control.

Loki also has an actual CUDA C translation path. Its SCC transformations,
`FortranCTransformation(language='cuda')`, and ISO C wrappers generate editable
CUDA plus launch/interface code. The upstream
[pinned CUDA integration tests](https://github.com/ecmwf-ifs/loki/blob/be67f4ae348c15d88413111590f1b42a058e21c3/loki/transformations/transpile/tests/test_scc_cuda.py)
exercise it. That route requires adapting horizontal/vertical/block conventions;
it was not necessary for the lower-effort OpenACC result measured here.

PSyclone's supported kernel backends include Fortran/OpenCL; its partial SIR/Dawn
route requires additional CUDA/Fortran interfacing. This is not a ready-to-use
CUDA output mode for these benchmarks. See the
[backend documentation](https://psyclone.readthedocs.io/en/stable/developer_guide/psyir_frontends_backends.html#available-back-ends).

## Reproduce

The tested versions are Loki commit
`be67f4ae348c15d88413111590f1b42a058e21c3` (the commit shown by the requested main
documentation when inspected) and **PSyclone 3.3.1**. Separate environments avoid
conflicting parser dependencies. Exact runtime dependencies are recorded in
the two requirements files under `benchmarks/tools/`; Python 3.13 was used for
this comparison. Run the commands below from the repository root.

```bash
python3.13 -m venv /tmp/loki-env
/tmp/loki-env/bin/python -m pip install -r benchmarks/tools/requirements-loki.txt
python3.13 -m venv /tmp/psyclone-env
/tmp/psyclone-env/bin/python -m pip install -r benchmarks/tools/requirements-psyclone.txt

# A shared command selects the recipe; output remains available for editing.
/tmp/loki-env/bin/python -m benchmarks.harness.transform --tool loki \
  --input benchmarks/cases/CDU/Fortran/cdu.f90 --case CDU \
  --target openacc --output /tmp/cdu-loki.f90
/tmp/psyclone-env/bin/python -m benchmarks.harness.transform --tool psyclone \
  --input benchmarks/cases/CDU/Fortran/cdu.f90 --case CDU \
  --target openacc --output /tmp/cdu-psyclone.f90

# Edit the generated Fortran/directives if desired, then compile it normally:
/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran \
  -O3 -acc=gpu -gpu=cc86 -cpp -module /tmp -I/tmp \
  /tmp/cdu-loki.f90 benchmarks/cases/CDU/main.f90 -o /tmp/cdu-loki
ACC_DEVICE_TYPE=nvidia /tmp/cdu-loki

# CPU correctness on all cases; use this repository's environment for the runner.
.venv/bin/python -m benchmarks.harness.compare \
  --output /tmp/tools-cpu \
  --loki-python /tmp/loki-env/bin/python \
  --psyclone-python /tmp/psyclone-env/bin/python

# GPU correctness plus performance, including unfused and resident-data modes.
.venv/bin/python -m benchmarks.harness.compare \
  --output /tmp/tools-gpu \
  --loki-python /tmp/loki-env/bin/python \
  --psyclone-python /tmp/psyclone-env/bin/python \
  --gpu run --resident --include-unfused \
  --acc-fc /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran \
  --compute-capability 86 --cuda-host-cxx g++-14 \
  --timing-grids 64x64x64 256x128x64 --iterations 20 --warmup 5 --rounds 5
```

The runner additionally needs this repository's Python dependencies, GNU
Fortran and G++. The GPU comparison needs NVIDIA Fortran with OpenACC, `nvcc`
and a compatible CUDA host compiler. Adapt the compiler path and compute
capability for another installation. This machine already had NVIDIA HPC SDK
26.1 under `/opt/nvidia`, although `nvfortran` was absent from `PATH`; no SDK
installation was needed.

Use a new/empty output directory. `--cases CDU` or `--tools psyclone` narrows a
run. `--gpu compile` builds the requested GPU modes without executing them.
Default CPU-only runs require no GPU tools. `--target cpu` on the transformation
command produces fused serial Fortran, and `--no-fuse` retains four nests. Loki's
`--vector-length` controls its OpenACC vector length (128 by default).

Each generated source has a sibling JSON manifest recording its recipe,
dependency/version information and source hashes. The runner saves fresh local
compiler outputs, binaries, `report.json`, and full command/stdout/stderr records
in `commands.json`. Native compiler diagnostics and actual GPU launch/transfer
notifications are retained. There is no successful skip on requested generation,
compilation, runtime or numerical failures.

## What is adapted

Loki reads the original module with its fparser frontend, expands the four
helpers, moves independent scalar coefficients before the nests, and applies
its built-in `do_loop_fusion` with `collapse(3)`. It then emits OpenACC data and
parallel-loop pragmas using Loki IR nodes. Scalar stencil temporaries are private.
The GPU output has one fused nest by default, or four with `--no-fuse`.

PSyclone uses `InlineTrans`, proven lower-bound simplification for these
assumed-shape dummy arrays, `MoveTrans`, nested `LoopFuseTrans`, `ACCLoopTrans`,
`ACCParallelTrans`, and `ACCDataTrans`. In 3.3.1, `CONTIGUOUS` causes the array
type to be represented as unsupported; the adapter removes that attribute only
in a temporary parsing copy and restores it in generated declarations. It
simplifies integer subscript offsets from inlining, without reassociating the
floating-point stencil expressions. No dependency-check override is used for
loop fusion.

Both preserve binary64 data, one-cell input halos, and the `MomentumAdvection`
public interface. The recipes are deliberately limited to the three benchmark
modules; they are not a promise of automatic parallelization for arbitrary
Fortran code.

## Measurement policy

Correctness uses the original deterministic drivers at noncubic, singleton,
cubic, and partial-block shapes: `5x4x3`, `1x1x1`, `16x16x16`, `259x7x5`.
Every interior value must be finite and within absolute tolerance `1e-10` of
original GNU Fortran; the output length must match. Unwritten output halos have
no defined original value and are excluded. Resident-mode correctness repeats
the call twice within one data region and checks the returned result.

Before GPU execution a probe checks that NVIDIA devices exist and confirms
`acc_on_device(acc_device_nvidia)` **inside a device region**. Each OpenACC
implementation must also produce a runtime kernel-launch trace. This prevents
counting a CPU fallback or missing directives as GPU execution.

Performance uses normal `-O3` settings and default fused multiply-add behavior,
not bit-exact arithmetic flags. GNU Fortran is the common CPU backend for the
original, Loki and PSyclone sources. When GPU tools are requested, the runner
also compiles original and fused CPU sources with NVFortran, exposing the
effect of changing the native compiler. Fresh local C++/CUDA and existing
handwritten OpenACC are included.

Two distinct application contracts are reported:

- **Drop-in:** each call starts with current host inputs and returns the output
  to the host. Timings include data mapping/allocation, transfers, computation,
  synchronization and cleanup as performed by each runtime.
- **Resident batch:** the caller adds `!$acc data copyin(u,v,w) copyout(result)`
  and `!$acc end data` around repeated calls. The inputs stay unchanged and only
  the final output is needed, exactly as in these benchmark drivers. The clock
  surrounds the data region, so its one upload/download round is included.
  Nested regions reuse existing mappings. This is not a kernel-only timing and
  is not the same host-visibility contract as the drop-in mode.

Warmups occur before timing. All variants use identical initialized inputs and
iteration counts, and consume a result checksum after the measured work. The
OpenACC code is synchronous, and tracing is disabled during timing. Reports
retain individual rounds and checksum validation, with no fixed clocks or CPU
affinity. Device residency and fusion are allowed because they serve the user's
performance/effort objective; they are explicitly labeled so their gains are
not mistaken for a different compiler backend alone.

## Tests

```bash
.venv/bin/python -m pytest benchmarks/tests
/tmp/loki-env/bin/python -m unittest benchmarks.tests.test_loki -v
/tmp/psyclone-env/bin/python -m unittest benchmarks.tests.test_psyclone -v
```

The normal repository environment skips library-specific tests if the separate
dependencies are absent; the two isolated test commands execute those checks.
The comparison runner performs the actual CPU/GPU correctness and performance
validation, including the data residency contract.

