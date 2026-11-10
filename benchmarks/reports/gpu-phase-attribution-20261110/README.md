# Startup and output phase attribution — 2026-11-10

**Initialization is the main measured cost outside the timestep timers; final output is smaller.** Diagnostic phase markers divide native/GPU execution into initialization, RK work, output and teardown without inserting synchronization. A second build adds nested wrappers around device visibility and pressure/provider creation. Neither changes the selected production executable or numerical kernels.

All six first-pass processes and four narrowed GPU processes pass strict original-output comparison in simple109/channel100. These are diagnostics, not speed measurements or a new full-field qualification. The selected seven-round result remains **3.78×/3.70× native**, with the approved long-run L2 qualification unchanged.

| Untraced diagnostic | simple109 | channel100 |
|---|---:|---:|
| Native initial conditions | 0.070 s | 0.063 s |
| GPU initial conditions, first / narrowed diagnostic | 0.639 / 0.470 s | 0.517 / 0.444 s |
| Pressure adapter creation, narrowed diagnostic | 0.323 s | 0.298 s |
| Device visibility, **inside** adapter creation | 0.071 s | 0.070 s |
| Solver creation, **inside** adapter creation | 0.251 s | 0.227 s |
| GPU final output, first / narrowed diagnostic | 0.070 / 0.069 s | 0.052 / 0.052 s |

These are individual diagnostic observations with different instrumentation, not paired performance samples. Nested rows must not be added. The warm FFT cache is reused and its cubins are checked. CUDA context creation, plan construction, library identity checks and module loading remain part of complete-process timing.

Nsight increases the visibility query from about 70 ms to 309–378 ms in the narrowed diagnostic. Its startup durations therefore cannot be used directly as an untraced speedup forecast. Traced `cudaFree(nullptr)` initialization occupies roughly another 0.15 seconds within solver creation; it does not represent freeing an application field. The selected provider already retains solver plans and verifies cached kernels with full runtime-library identities.

Channel's apparent output cost also needs careful interpretation. In the narrowed trace, host `OutTStep` intervals total 1.187 seconds. GPU activity overlaps 0.917 seconds of that interval, almost entirely kernels. Main-thread CUDA API spans total 0.952 seconds, predominantly scalar-result copies; only 0.036 seconds of those API spans has no simultaneous GPU activity. These are overlapping observations, not additive exclusive costs. The output path waits for preceding queued work, so removing a host wait alone would not remove that work.

The prior selected seven-round logs showed 0.735/0.693 seconds outside the printed step timers, against total 20× budgets of 0.573/0.647 seconds. This diagnostic locates much of that residual in initialization. It does not establish a hardware ceiling, prove a lower bound, or change the workload/timing definition. Further performance work must address measured kernel/runtime execution and startup together; final-output optimization alone cannot explain the gap.

Next release work is the maintained compiler/pipeline milestone, followed by its concrete failures, complete memory accounting, current FD4/operator qualification and fresh four-control timing. Full completion and the 20× objective remain open. Source identities, diagnostic commands, complete phase/API overlap tables and output comparisons are recorded in [summary.json](summary.json).
