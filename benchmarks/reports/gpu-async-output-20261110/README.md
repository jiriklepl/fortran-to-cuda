# Bounded asynchronous ELMM output — 2026-11-10

Primary-channel VTK output now uses **two immutable snapshot slots and one writer**. A checked read-only external task captures the five fields on the runtime stream; an event orders DMA on a separate stream. Serialization waits for DMA, slot reuse waits for serialization, and finalization joins the writer before retiring the domain or deallocating fields. Frames from one VTK call share a snapshot. Existing frame contents, scheduling and serializers are preserved; unsupported output paths wait before coherent native access.

Both device slots are charged to the existing field/scratch/provider workspace budget. Pinned snapshots have an explicit 256 MiB cap. The additive snapshot interface is version 1; core runtime ABI 1 and entry ABI 3 are unchanged. This is not a second coherence engine. Complete graph/queue/opaque-driver and host-metadata accounting remains open.

The complete public build regenerated 130 entries and rebuilt native/GPU executables and the local Poisson provider in 562.80 seconds. A subsequent frame-grouping refinement regenerated the existing time/wall/workspace adapters and rebuilt the two changed Fortran units and their dependents in 30.01 seconds. Initial sources/binaries and final source identities are preserved.

Two CUDA milestone checks cover snapshot generations, overwritten live fields, replica preservation, ordered callbacks, two-slot backpressure, final join, workspace pressure and writer failure. Eight fresh native/GPU processes pass all 20 fields and original outputs at the strict tolerances for simple2, channel2, simple109 and channel100. Short profiles retain 31/28 required GPU procedures and zero CPU FFTW plans; native controls retain four plans.

Natural channel2458 passes the approved independent 1% relative L2 rule across 20 fields and 114 outputs, including all 102 VTK frames. Worst L2 remains **0.154896%**; largest local error remains **1.467061% of reference peak**. All field/bounds files and 113 output files match the previous GPU trajectory byte for byte. Retau alone differs at roundoff level (relative L2 6.37e-16) and passes strict comparison. The immutable approved native long reference is reused, with its identities checked.

The first qualification controller incorrectly expected an asynchronous frame during channel100. That case ends at physical time 0.57; its only frames are at initialization. All eight processes completed successfully, and their immutable results were revalidated after correcting this instrumentation assertion. The failed assertion remains recorded; tolerances are unchanged.

The clean comparison retained one excluded warmup and seven rotated complete-process measurements, including startup, output and writer join:

| Case | Fresh native median | Previous GPU median | Candidate median | Candidate/native speedup | Candidate wins vs previous GPU |
|---|---:|---:|---:|---:|---:|
| simple109 | 11.513239 s | 3.020651 s | 3.004167 s | **3.83×** | 3/7 |
| channel100 | 13.168486 s | 3.533880 s | 3.528540 s | **3.73×** | 4/7 |

The candidate's median differences are 0.55% and 0.15%. They do **not** satisfy the all-seven-pairs requirement for an incremental speedup claim. All 48 processes pass output validation and the predeclared external-load gate. These primary cases do not reach postinitial asynchronous frames. This is a three-control comparison; fresh be58/7a612e4 controls remain required for release.

A separate 200-step channel profile reaches two postinitial slices. They share **one 25,449,248-byte snapshot**, with five ordered device copies and one download on a separate stream. Two device slots and two pinned slots each total **50,898,496 bytes**. The 2.009 ms download overlaps 0.118 ms of kernel execution. This is measured overlap, not an end-to-end speedup claim. All four initial/postinitial profile frames match the qualified long run exactly.

The remaining primary performance work is in measured computation and runtime costs. In this diagnostic 200-step run, SGS alone consumes 0.886 s of kernel activity (58 registers/thread, no local-memory allocation); the three CD4 advection kernels consume about 1.062 s together. Output work cannot close the 20× gap in the primary cases. The current native medians imply total 20× budgets of 0.576/0.658 s.

Further kernel optimization, full memory accounting, residual legacy/test migration, current FD4/operator qualification and the maintained four-control release protocol remain open. The full goal and 20× research objective are not marked complete. All samples, paired results, source/binary receipts and commands are linked in [summary.json](summary.json). Artifacts: `/tmp/fortran-async-output-20261110-01`.
