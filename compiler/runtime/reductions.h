#ifndef FORT_RESIDENT_REDUCTIONS_H
#define FORT_RESIDENT_REDUCTIONS_H
#include "scoped_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Dense canonical REAL64 allocation, including every element. Returns
 * max(initial, non-NaN inputs) and a separate NaN flag. initial must not be NaN.
 * Read-only: both replicas survive. Only the scalar result is downloaded.
 * Synchronous result; all device work uses the domain stream and budget.
 * The report distinguishes rejection from submission failure; never replay
 * source work following a started failure. Output scalars change on success. */
int fort_scope_max_real64_v1(fort_scope_t context, fort_buffer_t input,
    uint32_t rank, size_t elements, double initial, double *maximum,
    int32_t *any_nan, fort_scope_external_report_v1 *report);
#ifdef __cplusplus
}
#endif
#endif
