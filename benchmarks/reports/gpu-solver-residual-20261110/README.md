# Selected GPU solver: independent FD2/FD4 residual milestone

**All 18 differential/residual cases and three ABI/allocation-contract checks pass against the selected ELMM solver library.** This closes the missing independent FD4 residual check without changing the provider, application arithmetic or tolerances.

The maintained differential test now applies the real-space operator to both GPU and pinned native PoisFFT results. For FD4 it uses the seven-point stencil obtained by composing the fourth-order staggered gradient and divergence: coefficients `[1, -54, 783, -1460, 783, -54, 1] / (576 h²)`. It reconstructs periodic or staggered-Neumann extensions from the interior, so the third neighbor never reads beyond the public two-ghost Phi allocation. This is not the standard five-point collocated fourth-order Laplacian.

Manufactured solutions are built using that independent stencil and checked against recovered Phi. An explicit negative check requires FD2 residual validation to reject each manufactured FD4 solution. Existing tests still require unchanged RHS/halos, Phi ghost canaries, zero-mean gauge, nondefault-stream producer ordering, cross-stream plan reuse and stable execution workspace.

| Manufactured channel case | Native/GPU maximum difference | GPU relative residual |
|---|---:|---:|
| FD2, 128×64×64, lengths 12×4×2 | 9.99×10⁻¹⁶ | 6.93×10⁻¹⁴ |
| FD4, 128×64×64, lengths 12×4×2 | 8.88×10⁻¹⁶ | 9.00×10⁻¹⁴ |

The 18 cases cover odd/noncubic grids, periodic/periodic/Neumann channel boundaries, all-periodic and all-Neumann boundaries, incompatible constant RHS projected out of the null space, and manufactured odd-grid/channel solutions. Residual and differential thresholds are `2e-10`; the application thresholds remain unchanged. The three additional checks cover status-returning C entry points, C/Fortran layout and `c_ptr, value` bindings, rejection before CUDA work, budget exhaustion and allocation-free execution enforcement.

Only the test executable was rebuilt. It links the exact selected GPU provider and the six native PoisFFT objects from the qualified application build, with their identities verified. The first link attempt lacked unversioned FFTW development symlinks; it is preserved. The successful build uses the same four versioned FFTW libraries as the selected native executable.

This is a solver milestone, not a fresh complete application build, full runtime failure campaign, performance comparison or full-release claim. The selected application remains at its previously measured 3.78×/3.70× native speedup. Complete memory accounting, residual legacy/test migration, maintained release checks and fresh four-control timing remain open. The 20× research objective remains unmet.

Commands, sources, binary identities, all case results and the preserved failed link are recorded in [summary.json](summary.json).
