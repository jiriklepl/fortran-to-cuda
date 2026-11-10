# Reached wall/pressure arithmetic comparison, 2026-11-10

Identical-input comparisons find small CPU/GPU arithmetic differences in sampled wall and pressure operations, while replaying the GPU operation reproduces the application's output exactly. This follows the completed synchronization-only control and narrows the numerical investigation. It is diagnostic evidence, not a waiver of the strict long-run criterion, proof about every reached operation, or a new speed measurement. The selected cached GPU build remains unchanged at 3.60×/3.45× native.

## Wall operands

Temporary read-only observations capture calls 1 and 256 for each of four wall components and both planes during channel100. Captures include actual velocity allocations, previous wall stress, dimensions and scalar arguments, and the resulting GPU stress/flux. Sixteen replays execute the exact generated CPU and GPU function bodies with the controlled floating-point flags. Every GPU replay output matches the captured application output bit for bit.

Across 229,376 stress/flux outputs, 44 differ between CPU and GPU, by at most three floating-point steps (maximum absolute error 1.6653345369377348e-16). There are no nonfinite results or zero/nonzero disagreements. These samples do not establish a large single-call wall defect or prove the entire long trajectory harmless. The initial replay harness failed compilation because its extracted indexing namespace was unqualified; the failure is retained, and only the corrected harness ran.

## Pressure operands

The temporary pressure diagnostic preserves the GPU solution. At solves 1 and 256 it runs original PoisFFT on private copies of the same reached RHS and records both interiors. It checks bitwise preservation of the complete private RHS. Exact read-only transactions observe the application interiors; no CPU result is adopted. The CPU creates the four approved FFTW plans under the same four-thread/wisdom controls.

These actual channel operands use **FD2**, on a 128×64×64 grid with lengths 12×4×2, periodic x/y and staggered Neumann z. Do not mistake fourth-order advection for the pressure discretization. The independent real-space residual uses the recorded operator and removes the RHS null-space component. The implementation also expresses the composed staggered FD4 stencil, but these captures do not exercise it. Mode checks verify the FD2 residual operator on this grid.

| Reached solve | Maximum CPU/GPU difference | CPU relative L2 residual | GPU relative L2 residual |
| --- | ---: | ---: | ---: |
| 1 | 3.88578e-16 | 4.84942e-15 | 5.34662e-15 |
| 256 | 2.77556e-17 | 1.25766e-15 | 1.139e-15 |

Both differential samples pass the original pointwise criterion and the stricter existing provider differential criterion. Neither this nor the residual result claims bitwise FFT equivalence. Both operand-capture builds pass the short/channel100 application checks and preserve all 40 field/bounds files exactly against their GPU controls. Sampling adds no changed trajectory to those cases.

## Long-run criterion for review — not adopted

The approved `atol=1e-6`, `rtol=1e-5` pointwise long-run milestone still fails 15 fields and 48 outputs. The earlier native-only experiment also fails that milestone after flipping the lowest mantissa bit of each pressure result, without any GPU execution. Together with the identical-input evidence, this motivates an explicit review of long-trajectory acceptance; it does not automatically authorize changing it.

A separate, post hoc review evaluates a proposed **1% relative L2 error independently for each field and output component** at natural channel2458. It includes all 20 field dumps and all 114 outputs, splits vector and Retau components rather than mixing their scales, and requires exact bounds, field names, layouts, coordinates, labels, inventories and frame indices. Time columns keep the original pointwise check. A zero reference requires maximum absolute error at most 1e-6. Primary and operator checks would remain unchanged.

The existing long run satisfies that proposed norm limit. The largest relative L2 error is **0.154897%**, in the pressure slice `frame-y-49.vtk`; the largest among final full-storage fields is approximately **0.1006%**, in pressure. The largest local error is **1.4671% of the reference field's peak magnitude**, also pressure. This last figure is reported because a norm criterion can hide localized differences. These figures are descriptive: no production validator or accepted tolerance has changed. Adoption requires the user's explicit decision; subsequent qualification must use the declared rule, not silently relabel the original failure.

Evidence roots: `/tmp/fortran-wall-operands-20261110-01`, `/tmp/fortran-pressure-operands-20261110-01`, and `/tmp/fortran-long-norm-review-20261110-01`. The [machine-readable summary](summary.json) binds the receipts. Original source, cached performance candidate, physics and timing definitions are unchanged. Completion and full release qualification remain open.
