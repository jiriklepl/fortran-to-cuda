#ifndef ELMM_GPU_PRESSURE_H
#define ELMM_GPU_PRESSURE_H
#include "scoped_runtime.h"
#include "poisson-solver/device_api.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Explicit numerical adapter, not a source/lifetime proof or placement rule.
 * Compiler-issued canonical roots must have verified dense rank3 REAL64
 * layouts: Phi n+4 (ghost2), RHS n+2 (ghost1), and the exact original lower
 * bounds supplied here. Only local FD2/FD4 and periodic/NeumannStag pairs are
 * supported. Dimension extents must be at least2 for two-layer face copies. */
typedef struct elmm_gpu_pressure_config_v1 {
    uint32_t version, reserved;
    poisson_solver_config_v1 solver;
    int64_t phi_lower_bounds[3], rhs_lower_bounds[3];
} elmm_gpu_pressure_config_v1;
typedef struct elmm_gpu_pressure_handle_v1 elmm_gpu_pressure_handle_v1;
/* Caller owns the domain. Reserve cap -> provider create -> shrink live claim.
 * On failure a nonnull output is a cleanup-only handle if provider cleanup
 * itself failed; destroy it before closing/abandoning the domain. */
int elmm_gpu_pressure_create_v1(fort_scope_domain_t domain,
    const elmm_gpu_pressure_config_v1 *, elmm_gpu_pressure_handle_v1 **);
/* Exact physical effects: RHS active interior read; Phi active interior write
 * and overwrite. Phi halos and the complete RHS are preserved. */
int elmm_gpu_pressure_solve_v1(fort_scope_t, elmm_gpu_pressure_handle_v1 *,
    fort_buffer_t phi, fort_buffer_t rhs, fort_scope_external_report_v1 *);
/* Provider completion/destruction precedes workspace-lease release. */
int elmm_gpu_pressure_destroy_v1(elmm_gpu_pressure_handle_v1 **);
const char *elmm_gpu_pressure_error_v1(void);
#ifdef __cplusplus
}
#endif
#endif
