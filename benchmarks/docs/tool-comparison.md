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
| Main optimization | Built-in inlining and pragma-directed loop fusion | Inlining, dependency-checked motion/fusion, inferred data clauses | Checked fusion at level 1; level 0 retains four kernels |
| Work required here | A reusable Python recipe; declare/check this stencil family's pointwise-output contract | A reusable Python recipe; accommodate unsupported `CONTIGUOUS` in the parser | Existing CLI works directly |
| Control | Editable generated Fortran; fusion switch, vector length, loop/data pragmas | Editable generated Fortran; fusion switch and PSyIR loop/data transformation options | Editable generated `.cu`; generated launch/runtime code |
| Keep data across calls | Two caller data directives | Two caller data directives | Generated create/run/update/destroy workspace API; ordinary calls allocate/copy per call |

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
- **Resident batch:** OpenACC callers add `!$acc data copyin(u,v,w) copyout(result)`
  and `!$acc end data` around repeated calls. Local compiler callers create a
  workspace, run it repeatedly, retrieve the final output, and destroy it.
  Inputs stay unchanged and only the final output is needed. The timer includes
  data setup, final retrieval and destruction, amortized over the batch. Local
  CPU sessions include owned-buffer copies as well. This is not a kernel-only
  timing and is not the same host-visibility contract as the drop-in mode.

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

## Rerun with local fusion and sessions (2026-10-02)

The full comparison was rerun from 15:56 to 16:06 UTC on the same Ryzen 5 5600G
virtualized host and RTX 3060 Ti 8 GiB (driver 617.14). Native compiler and
external-tool versions are unchanged. An immutable copy of the working sources
was used; original benchmark source hashes match the earlier run. The current
compiler is measured at optimization levels 0 and 1, including owned CPU/CUDA
sessions. Thus the new `Local-C++` and `Local-CUDA` labels mean **level 1**;
the earlier report's local implementations predate these capabilities.

Both `64x64x64` and `256x128x64` were timed with five warmups, 20 measured calls
per round and five rounds, using one CPU thread, `-O3`, default FMA, and no
fast-math/LTO flags. All **288 correctness checks** passed, with maximum absolute
error **3.27e-13** against tolerance `1e-10`. The **168 GPU execution traces**
cover OpenACC and local CUDA; local CUDA traces confirm one kernel per level-1
call, four per level-0 call, and one set of allocations/transfers per resident
batch. All **144 timing configurations (720 rounds)** passed checksum validation.

The table gives milliseconds per call at `256x128x64`, mean ± sample standard
deviation. Resident rows include setup, final output retrieval and teardown,
amortized over 20 calls with unchanged inputs and only the final output required.

| Implementation | Call mode | CDU | CDV | CDW |
| --- | --- | ---: | ---: | ---: |
| Original Fortran, GNU | Ordinary call | 12.58 ± 0.40 | 11.87 ± 1.01 | 12.52 ± 0.63 |
| Original Fortran, NVFortran | Ordinary call | 11.47 ± 0.60 | 10.49 ± 0.39 | 11.06 ± 0.25 |
| Local C++, level 0 | Ordinary call | 18.35 ± 3.62 | 15.97 ± 0.22 | 16.89 ± 0.18 |
| Local C++, level 1 | Ordinary call | 10.30 ± 0.22 | 10.11 ± 0.15 | 10.30 ± 0.20 |
| Loki fused Fortran, GNU | Ordinary call | 6.08 ± 0.18 | 6.88 ± 0.34 | 6.11 ± 0.20 |
| PSyclone fused Fortran, GNU | Ordinary call | 5.87 ± 0.17 | 6.03 ± 0.08 | 6.42 ± 0.42 |
| Loki fused Fortran, NVFortran | Ordinary call | 4.84 ± 0.17 | 4.78 ± 0.31 | 4.32 ± 0.08 |
| PSyclone fused Fortran, NVFortran | Ordinary call | 4.57 ± 0.19 | 4.12 ± 0.11 | 4.72 ± 0.44 |
| Local C++, level 0 | Resident batch | 18.95 ± 1.25 | 17.45 ± 0.79 | 20.25 ± 1.02 |
| Local C++, level 1 | Resident batch | 13.23 ± 0.60 | 13.19 ± 0.70 | 12.72 ± 0.41 |
| Local CUDA, level 0 | Ordinary call | 13.92 ± 1.00 | 13.61 ± 1.22 | 13.59 ± 1.49 |
| Local CUDA, level 1 | Ordinary call | 14.55 ± 1.90 | 12.82 ± 0.87 | 13.45 ± 1.74 |
| Handwritten OpenACC | Ordinary call | 9.76 ± 0.51 | 10.20 ± 0.62 | 9.87 ± 0.39 |
| Loki unfused OpenACC | Ordinary call | 9.79 ± 1.15 | 9.78 ± 0.84 | 9.93 ± 0.61 |
| PSyclone unfused OpenACC | Ordinary call | 11.22 ± 0.86 | 9.60 ± 0.37 | 10.75 ± 0.73 |
| Loki fused OpenACC | Ordinary call | 9.46 ± 0.42 | 9.54 ± 0.60 | 9.61 ± 1.07 |
| PSyclone fused OpenACC | Ordinary call | 9.63 ± 1.06 | 9.57 ± 0.95 | 10.42 ± 1.38 |
| Local CUDA, level 0 | Resident batch | 1.57 ± 0.12 | 1.60 ± 0.28 | 1.50 ± 0.08 |
| Local CUDA, level 1 | Resident batch | 1.39 ± 0.08 | 1.36 ± 0.11 | 1.31 ± 0.10 |
| Handwritten OpenACC | Resident batch | 1.81 ± 0.09 | 1.69 ± 0.19 | 1.57 ± 0.10 |
| Loki unfused OpenACC | Resident batch | 1.60 ± 0.09 | 1.64 ± 0.19 | 1.61 ± 0.11 |
| PSyclone unfused OpenACC | Resident batch | 1.88 ± 0.08 | 1.67 ± 0.16 | 1.63 ± 0.15 |
| Loki fused OpenACC | Resident batch | 1.18 ± 0.07 | 1.20 ± 0.10 | 1.30 ± 0.16 |
| PSyclone fused OpenACC | Resident batch | 1.27 ± 0.12 | 1.30 ± 0.05 | 1.14 ± 0.11 |

Within this run, optimized local C++ is **1.58–1.78× faster than level 0** and
**1.17–1.22× faster than original GNU Fortran** at the larger grid. Fused Fortran
from Loki/PSyclone is still faster on CPU. Local CUDA ordinary-call costs remain
**12.82–14.55 ms**; residency reduces them to **1.31–1.39 ms per call**, a
**9.44–10.50×** reduction under the resident-batch contract. Local fusion improves
resident CUDA by another **1.13–1.18×** over level 0. The corresponding fused
OpenACC results are **1.14–1.30 ms**, with overlapping variability in some cases;
small differences should not be treated as a stable ranking. Owned CPU sessions
were slower than ordinary local CPU calls because their buffer setup/copies are
included. The smaller grid and every individual sample are in the archived data.

These results supersede the earlier statement that the local compiler lacks
fusion and residency. They do not establish performance on arbitrary Fortran or
other machines. Clocks and CPU affinity were not fixed, and shifts in unchanged
baselines show why historical deltas alone should not be attributed to compiler
changes; the level-0 comparison above is from this same run.

Archive: [report](../results/tool-comparison-rerun-20261002T155622Z/report.json),
[timing CSV](../results/tool-comparison-rerun-20261002T155622Z/timings.csv),
[provenance and hashes](../results/tool-comparison-rerun-20261002T155622Z/provenance.json),
[full command/output log, gzip](../results/tool-comparison-rerun-20261002T155622Z/commands.json.gz),
and [exact source snapshot](../results/tool-comparison-rerun-20261002T155622Z/source.tar.gz).
The earlier measurements below remain available for historical comparison.

## Earlier measurements (before local fusion and sessions)

Measured on 2026-10-02: AMD Ryzen 5 5600G in a virtualized Linux environment,
GeForce RTX 3060 Ti 8 GiB, GCC/GFortran/G++ 16.2, NVIDIA HPC SDK 26.1, CUDA 13.4
and G++ 14 for nvcc's host compilation. All **216 correctness runs** passed;
maximum absolute difference was **3.27e-13**, well below `1e-10`.
The 120 OpenACC execution traces confirm one device kernel per fused call,
four per unfused call, and the corresponding doubled counts for resident
correctness's two calls. Both libraries' isolated regression suites also pass
(five Loki and three PSyclone tests), as do ten runner/driver checks.

The table shows milliseconds per call on `256x128x64`, mean ± sample standard
deviation over five rounds of 20 measured iterations. Resident rows include
one upload/download round per batch, amortized over those 20 calls.

| Implementation | Data contract | CDU | CDV | CDW |
| --- | --- | ---: | ---: | ---: |
| Original Fortran, GNU | Host | 11.12 ± 0.35 | 12.39 ± 0.73 | 15.16 ± 1.74 |
| Original Fortran, NVFortran | Host | 9.64 ± 0.27 | 10.26 ± 0.79 | 12.73 ± 1.97 |
| Local generated C++ | Host | 17.71 ± 0.31 | 18.25 ± 0.15 | 24.97 ± 1.40 |
| Loki fused Fortran, GNU | Host | 5.57 ± 0.15 | 6.20 ± 0.27 | 6.22 ± 0.22 |
| PSyclone fused Fortran, GNU | Host | 5.51 ± 0.09 | 5.68 ± 0.11 | 6.12 ± 0.12 |
| Loki fused Fortran, NVFortran | Host | 4.16 ± 0.36 | 3.95 ± 0.09 | 4.07 ± 0.06 |
| PSyclone fused Fortran, NVFortran | Host | 4.09 ± 0.12 | 3.90 ± 0.13 | 4.34 ± 0.19 |
| Local CUDA | Drop-in | 13.36 ± 1.25 | 13.76 ± 1.68 | 14.08 ± 0.34 |
| Handwritten OpenACC | Drop-in | 8.91 ± 0.50 | 9.07 ± 1.05 | 10.69 ± 1.74 |
| Loki fused OpenACC | Drop-in | 8.50 ± 0.75 | 8.43 ± 0.75 | 9.60 ± 0.50 |
| PSyclone fused OpenACC | Drop-in | 9.07 ± 1.14 | 8.36 ± 0.68 | 8.51 ± 0.54 |
| Handwritten OpenACC | Resident batch | 1.59 ± 0.14 | 1.49 ± 0.09 | 1.64 ± 0.05 |
| Loki unfused OpenACC | Resident batch | 1.49 ± 0.11 | 1.50 ± 0.12 | 1.55 ± 0.10 |
| PSyclone unfused OpenACC | Resident batch | 1.52 ± 0.12 | 1.64 ± 0.18 | 1.56 ± 0.10 |
| Loki fused OpenACC | Resident batch | 1.12 ± 0.09 | 1.14 ± 0.13 | 1.25 ± 0.16 |
| PSyclone fused OpenACC | Resident batch | 1.14 ± 0.11 | 1.19 ± 0.11 | 1.18 ± 0.17 |

[Archived results](../results/tool-comparison-2026-10-02.json) include all timing samples and
checksums for both `64x64x64` and `256x128x64`, unfused drop-in results, exact
versions/source hashes, transformation manifests, and verified GPU traces.
The fused NVFortran CPU rows were measured in a supplementary pass with the
same sources, drivers and iteration counts; its 24 numerical checks and all
timing checksums were also checked against the GNU Fortran reference.
There is noticeable run-to-run variation, especially in the CDW baselines;
these measurements do not establish a small performance difference between
Loki and PSyclone.

For these benchmarks, **fusion and data lifetime matter more than choosing
between the two libraries**:

- With the existing host-visible call contract, fused serial Fortran is the
  fastest measured option at the larger grid. It is roughly twice as fast as
  the original GNU Fortran on the same GNU backend, without any GPU or caller
  changes. Compiling that same generated source with NVFortran improves it
  further to approximately **3.9–4.3 ms per call**, requiring no additional
  transformation work.
- Both generated OpenACC implementations improve on the local CUDA wrapper's
  end-to-end cost, but much of that is already available from native OpenACC:
  the handwritten baseline is close. Fusion's benefit is clearer once repeated
  transfers are removed.
- If the application can keep arrays resident, two caller directives reduce
  the measured batch cost substantially. Fusion then improves the resident
  result by roughly another 20–30% versus the unfused generated variants.
  This is a real application-lifetime optimization, not an intrinsic speedup
  from choosing a particular translator.

**Recommendation:** PSyclone's generic Fortran/OpenACC route is a good starting
point for this scope when checked transformations and automatic data clauses
are useful. Loki is equally viable, accepts the current declarations directly,
and is the more relevant option to explore if an editable CUDA C pipeline is
a future requirement. Its current main API is explicitly incubating, so keep
the commit pin. There is no measured performance reason here to prefer one
library strongly. For only these three stable kernels, the existing handwritten
OpenACC plus caller data residency is an even smaller integration if automatic
regeneration is unnecessary.
