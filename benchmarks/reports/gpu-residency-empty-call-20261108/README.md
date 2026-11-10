# Empty native CALL residency checkpoint

The compiler now retains a coherent domain across an original native CALL when
source proofs establish that every payload loop is empty. The original CALL
still executes once. An allocated nonempty list, unavailable descriptor, unknown
effect or unsupported entry association keeps the existing publication path.
This is generic compiler support; it contains no ELMM procedure names.

The compiler checks the complete wrapper/internal-child inventory, exact
`ALLOCATED`/`SIZE` inquiries, descriptor associations and automatic iterators.
It issues no payload or initialization authority. The companion reader validates
the version-3 preservation policy and exact emitted guard. Runtime ABI and
coherence semantics are unchanged.

## Qualification

The frozen compiler and reader sources are identified in [evidence.json](evidence.json).
They precede the separate joined-descriptor follow-up in the working tree.

- 172 affected compiler/reader/native checks pass, plus the callback-source check.
- Four actual CUDA controls pass: traced, untraced and hidden-device execution,
  plus fatal publication failure before the active callback. Unallocated and
  empty lists retain one domain with zero inter-kernel payload transfers; active
  lists retain the original publication path. Full fields, bounds and halos match.
- Normal ELMM generation produces 147 units in 55 procedures; strict package
  validation, public build and four two-step GPU/hidden-GPU checks pass.
- Natural `channel-simple` (109 steps) and `channel` (2,458 steps) pass all 20
  fields, bounds, halos and all 6/114 outputs at `atol=1e-6`, `rtol=1e-5`.
  All fields are bitwise equal to the previous AUTO-copy candidate. Both runs
  use four threads, the approved deterministic pressure reference and CPU PoisFFT.
  Input hashes hold and all owned process groups have exited.

The first natural attempt was interrupted by signal 15 and remains failed. A
fresh second attempt completed both cases. External history cleanup removed an
unused diagnostic helper; the second harness retains its exact original bytes
under `/tmp`, without restoring the deleted repository file or report archive.

## Measured effect and remaining work

These are **initialization plus two steps**, not per-step totals:

| Case | H2D bytes | D2H bytes | DMA calls | Change from previous candidate |
|---|---:|---:|---:|---|
| channel-simple | 134,664,192 | 146,316,352 | 148 | None |
| channel | 184,158,208 | 146,316,352 | 730 | None |

In simple, publication moves from the empty `ResistanceForce` call to the
following diffusion boundary. Channel's largest cut remains `BoundUVW_do_BC`.
Numerical GPU coverage and allocations are unchanged. This checkpoint establishes
correct empty-call retention, **not application traffic savings or speedup**.

The next change connects existing complete joined-worksharing lifetime proofs
to selected descriptor capture. Its initial six tests and 298 affected checks
pass, but this application build does not contain it. Fresh generation and
execution must establish whether it removes the diffusion cut. Further work
must address boundary payload access, actual diffusion/pressure GPU coverage,
cross-stage residency and launch costs.

The maintained baseline campaign has 17,125 unique passing analysis checks and
15 GPU opt-ins still pending; its CUDA continuation is incomplete. Those results
do not qualify every test against this draft. Fresh seven-rotation complete-process
timings and the two-case 10%, 2× and 20× gates remain open.
