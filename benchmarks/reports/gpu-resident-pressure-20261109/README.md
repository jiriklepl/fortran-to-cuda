# Resident pressure integration, 2026-11-09

The pressure producer/consumer implementation now executes on GPU, including
RHS construction, the local `poisson-solver` solve, Phi halos, velocity correction,
pressure update, the approved deterministic TOP/BOTTOM reference, and Pr faces.
**This partial hot-path integration is slower than the controls.** It remains a
private integration candidate; it is not promoted as the production strategy.

| Primary case | Native | be58 | GPU pressure candidate |
| --- | ---: | ---: | ---: |
| simple109 | 11.810 s | 9.899 s | 13.100 s |
| channel100 | 13.281 s | 11.389 s | 14.381 s |

These are one complete-process measurement per mode/case after one excluded
warmup: 12 total processes, rotated order, four threads on CPUs 0,2,4,6, unchanged
physics, pressure reference and outputs. All output checks pass. Maximum observed
external CPU activity was 0.387 core. No builds, tests
or profiles ran concurrently. This is a short implementation comparison, not a
median/distribution result or the seven-round research gate. Controls reuse
existing binaries with fresh measurements; fresh release-control builds remain
required. Solver plan creation and compilation, startup, output and teardown are
inside the clock. No persistent solver kernel cache is enabled. The candidate
loads `poisson-solver/install-sm86/lib/libpoisson-solver.so`; controls retain
approved CPU PoisFFT and its four verified wisdom plans.

The preceding three-round pilot stopped after process 31 when unrelated CPU
activity reached 0.523 core, exceeding its predeclared 0.5 limit. Every observation
is retained under `timing/`; no speedup claim is based on that invalidated pilot.

## Implemented

- `elmm_pipeline/gpu_hot_path.py` outlines original pressure stencils, expands the
  existing PURE divergence expression, generates 14 shared compiler entries and
  integrates them at the original control positions. Native unsupported branches
  remain available. No loop arithmetic is replaced by a speculative formula.
- `resident_pressure_state.f90` owns one persistent pressure domain and solver
  plan. Saved Phi/RHS retain storage and replicas. Numerical entries and the
  external solver use the existing coherence runtime. Borrowed fields are
  published together and unregistered before unmanaged host use/return.
- `gpu_pressure_tail.cu` implements the approved serial plane reference and
  original `Bound_Pr` formulas, preserving edges/corners. Its source adapter
  checks both original routine hashes, layout and supported reference/boundary
  modes. It uses an accounted eight-byte scratch lease, the runtime stream and
  a checked external access transaction; errors never replay submitted work.
- `compiler/offload/analysis.py` recognizes unconditional exact-point definitions
  within a kernel iteration. A subsequent read of that same point no longer
  demands an incoming replica. Conditional definitions and changed scalar index
  state remain conservative. This fixed the first integration run's legitimate
  uninitialized-RHS rejection without inventing initialized data or uploading RHS.
- Numerical extraction no longer stops after eight levels or 64 procedures.
  OpenMP PRIVATE clauses no longer create false incoming scalar-liveness demands.

## Milestone evidence and limits

The latest two-step native/candidate runs pass all 20 fields, bounds, halos and
all output files in both configurations at `atol=1e-6`, `rtol=1e-5`. The focused
compiler footprint suite passes 33 tests. Each candidate records seven GPU
solves, seven Phi-halo operations and seven pressure-tail transactions, and zero
CPU FFTW plans. RHS and Phi payloads stay resident between their producers and
consumers. Additional inputs, including Pr, can still upload at first use.

The complete two-step-plus-initialization ledger is 224,215,040 bytes uploaded
and 157,794,688 bytes downloaded in each case, versus 107,643,264 combined bytes
for the preceding CPU-Convection diagnostic. GPU pressure adds field transfers
while the surrounding RK work remains native. The graph can reorder independent
velocity and pressure work; publication now drains the consumer frontier together.
These facts identify remaining boundaries, not an exclusive wall-time attribution.

Velocity/mass-flux preparation, surrounding Convection/diffusion/RK updates,
allocation exchanges and timestep control are not yet one resident GPU path.
The next integration must join those stages and replace the intervening boundary
work. Do not spend another full timing campaign on this isolated pressure slice.
Natural 109/2458 full20 qualification, the maintained suites, seven-round timing,
and the 2×/20× research checkpoints remain open. Legacy removal beyond the
extraction caps remains to be implemented alongside the replacements.

## Reproduction artifacts

The immutable generated sources, numerical entries, native/CUDA build commands,
input hashes, binaries, all comparison samples and physical ledgers are under
`/tmp/fortran-resident-hot-path-20261109-01`. See `summary.json` for exact receipts.
`build_integration-03.py` links the latest private candidate; `run_milestone-03.py`
checks its short full20 results. `comparison/plan.json` declares the small timing
protocol. Earlier failed/intermediate artifacts remain present. The preceding
state source is preserved under `integration-build-02/source_snapshot/` with its
original verified hash. Source edits outside this work, including the dirty
local solver checkout and editor configuration, were preserved.
