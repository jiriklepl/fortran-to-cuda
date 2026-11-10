# Full public regeneration and completion audit — 2026-11-10

The public `resident-build` command now reproduces the selected GPU implementation from current compiler, ELMM/PoisFFT and local Poisson sources. No historical objects or module files are reused. This closes the regeneration gap from the last two compiler optimizations; it does not establish full release completion or another speedup.

The complete build regenerates 130 GPU entries and finishes the complete Ninja build in 570.3 seconds. Both ordinary-output and full-20-field native/GPU executables and the local provider are rebuilt. The 518 build-time source inputs are archived with their original identities so later cleanup cannot erase this evidence.

## Reproduction and application qualification

All 130 Fortran interfaces and 229 numerical CUDA kernel bodies match the selected implementation after consistent generated-name and diagnostic-path normalization. All 130 common headers match byte for byte. The comparison retains every numeric expression/literal and reports its limits: it does not equate host coordinator logic, trace/cache identities, or execution time.

Eight fresh processes cover native and GPU simple2, channel2, natural simple109 and channel100. All 20 fields, bounds and original outputs pass `atol=1e-6, rtol=1e-5` against the freshly built four-thread native controls. GPU fields also match the selected GPU baseline exactly. Short ledgers establish 31/28 required GPU procedures, no generated CPU numerical events and zero CPU FFTW plans; native controls retain their four expected CPU plans.

A fresh natural channel2458 GPU run passes the approved independent 1% relative L2 rule across 20 fields and 114 outputs. Worst L2 is 0.154896% (frame-y-49.vtk:p); the largest local error remains 1.467061% of reference peak (fields-Pr.bin). All 40 field/bounds files exactly match the selected GPU trajectory. The immutable earlier native long reference is reused with verified identities. Zero-reference, exact structure/inventory, and original time checks remain unchanged; earlier pointwise failures remain recorded.

## Calibration

After the build and application runs ended, runtime/collective/source and CUDA graph measurements ran sequentially with affinity 0,2,4,6 and four threads. No owned builds, tests, profiles or application timing campaigns overlapped these measurements. Runtime costs use the current arithmetic contract and compiler sources. Numerical compute families are not remeasured by this milestone; raw graph observations alone grant no placement authority.

The current runtime identity and base, transfer and physical-copy costs validate. All 32 graph cases pass, covering 217,600 logical kernels and 10,240 replays. The collective model fails one public AUTO holdout: 286.77 microseconds observed against 107.22 predicted, exceeding the unchanged 25% error limit. The source-submit model fails three independent holdouts: (bindings, launches) = (2,18), (6,2), (6,4). Both optional models remain unavailable, and their public consumers reject cost/quote requests. The measurements succeeded; complete automatic placement is not qualified. Raw observations and thresholds are preserved, with no retries or relaxed criteria. See [summary.json](summary.json) for identities, failures and qualification receipts.

## Remaining implementation

The [26-group requirement audit](requirements-audit.json) binds inspected source files and distinguishes implemented code, qualified application behavior and missing release evidence. Completion remains unproven. The next concrete work is:

1. Consolidate production code generation and placement. Even scoped generation currently emits the ordinary backend; public ordinary policies still include a four-unit interval heuristic. Migrate call boundaries to the shared coherence implementation, then remove unused alternatives and production dependencies on legacy configuration types.
2. Implement the planned bounded asynchronous output. Current frame handling synchronously observes read-only field replicas before original serialization. It preserves GPU replicas but has no two-slot snapshot queue, writer backpressure or final writer join. Account graph/snapshot storage together with fields, scratch and solver workspace.
3. Finish current solver/operator and maintained release-suite qualification, including independent FD4 residual checks. Primary FD2 agreement does not establish that gate.
4. Run the fresh four-control release comparison after implementation stabilizes. Keep the measured 20× gap visible and preserve complete-process timing and all outputs.

The pipeline README now removes instructions for retired `bounded`/`reached` source execution and owner-timing flags. This milestone otherwise preserves production numerical code.

There is no new timing campaign here. The selected baseline remains the [host-preparation milestone](../gpu-host-preparation-20261110/README.md): 3.033684/3.523741 seconds and 3.83×/3.73× four-thread native speedup. Its latest incremental gains met median acceptance but not the all-seven-pairs criterion. The 20× research objective remains open.

Full commands, source/binary identities, observations, audit and receipts are in [summary.json](summary.json) and the referenced `/tmp/fortran-current-public-20261110-01` artifacts.
