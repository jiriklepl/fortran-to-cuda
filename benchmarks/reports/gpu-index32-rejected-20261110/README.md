# Rejected coordinate decoder pilot, 2026-11-10

The guarded 32-bit coordinate decoder is **rejected and removed**. It rebuilt
24 hot SGS, diffusion, advection and filter CUDA entries on the qualified
source-build basis. Both operands were checked before narrowing, with the wide
path retained. Source integer semantics and addressing were unchanged.

All 148 existing coordinate-decoder cases pass. Both two-step cases, natural
simple109 and channel100 pass all 20 fields, bounds, halos and outputs; field
and shape files are bit-identical to the retained GPU control. This pilot did
not improve application performance.

| Case | Control median | Pilot median | Change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| simple109 | 4.555951 s | 4.769966 s | +4.70% | 0/3 |
| channel100 | 5.019880 s | 5.007606 s | −0.24% | 2/3 |

One excluded warmup and three rotated complete-process measurements per mode
and case; four threads and unchanged output/startup. All 16 outputs pass, all
owned groups are reaped, and external CPU remains below the predeclared bound.
No samples were retried or trimmed. The predeclared gate required at least 2%
lower median and all three paired wins in both cases. No fresh native or
historical compiler comparison was part of this isolated optimization pilot.

Static inspection finds unchanged register/stack counts in 22/24 entries and
small register changes in the other two. The guarded versions contain 32–56
more static SASS instructions. This is not a runtime instruction count or proof
of a specific cause. Do not assume narrower indices help without measuring the
whole application.

All samples, exact generated sources, failed gate, application checks, build
commands and implementation archive remain under
`/tmp/fortran-index32-20261110-01`. The generic emitter is restored exactly to
its pre-pilot state. The selected source-built GPU candidate remains unchanged.

The fresh 20-step channel profile is separately recorded at
`/tmp/fortran-resident-source-build-20261109-02/profile`. Kernel activity totals
0.667 s (union 0.667 s, first-to-last device span 1.034 s): SGS 0.114 s, filter
0.112 s, advection 0.134 s, FFT 0.059 s, diffusion 0.042 s and pressure-plane
reference 0.023 s. CPU API spans overlap device activity and must not be added
to these durations. This diagnosis is not an untraced performance measurement.

## Host-selected compact graph kernels also rejected

A follow-up removes the per-point choice entirely: the checked positive product
selects a separate compact kernel on the host. Graph topology identity already
includes the function pointer, so wide-domain transitions rebuild correctly.
The compact bodies contain 11,904 static SASS instructions versus 14,208 for the
wide bodies across the 24 hot kernels. All 297 decoder cases and primary all20
application checks pass; the field dumps remain bit-identical to the retained
GPU implementation.

Despite fewer static instructions, this version is also rejected and removed:

| Case | Control median | Compact median | Change | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| simple109 | 4.444360 s | 4.737245 s | +6.59% | 0/3 |
| channel100 | 5.008623 s | 5.001489 s | −0.14% | 2/3 |

The same predeclared three-round protocol and 2%-plus-all-paired-wins gate were
used. All 16 timing outputs pass, external load stays below the bound, and all
owned groups are reaped. All samples and the exact rejected source patch are
in `/tmp/fortran-compact-graph-20261110-01`; `decision.json` records the rejection.
No timing result from either index experiment replaces the qualified source
build. Stop index-width work unless new profiling provides a stronger reason.

Pressure-reference load strategies were explored separately in
`/tmp/fortran-plane-reference-20261110-01`. They preserve the ordered sum on four
diagnostic shapes, but isolated timings are variable and no application gain
was established. No pressure-reference change is retained. Next work targets
redundant graph parameter updates measured in the source-build profile.

A separate post-comparison 20-step channel trace verifies actual compact-kernel
execution: 17 reached compact functions and 1,860 launches. Generation without
execution does not explain this negative result. The trace is preserved at
`/tmp/fortran-compact-graph-20261110-01/profile`; its durations are diagnostic,
not additional timing samples and not a reason to reverse the rejection.
