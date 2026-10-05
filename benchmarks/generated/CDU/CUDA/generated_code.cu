#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include "common_functions.cuh"

namespace generated_kernels {
using namespace indexing;
using namespace timing;

__global__ void kernel_region_0_device(
    double* __restrict__ fort_v0_u2,
    std::size_t fort_v0_u2_dim1,
    std::size_t fort_v0_u2_dim2,
    std::size_t fort_v0_u2_dim3,
    const double* __restrict__ fort_v1_u,
    std::size_t fort_v1_u_dim1,
    std::size_t fort_v1_u_dim2,
    std::size_t fort_v1_u_dim3,
    const double* __restrict__ fort_v2_v,
    std::size_t fort_v2_v_dim1,
    std::size_t fort_v2_v_dim2,
    std::size_t fort_v2_v_dim3,
    const double* __restrict__ fort_v3_w,
    std::size_t fort_v3_w_dim1,
    std::size_t fort_v3_w_dim2,
    std::size_t fort_v3_w_dim3,
    int fort_v7_unx,
    int fort_v8_uny,
    int fort_v9_unz,
    double fort_v10_zero,
    double fort_v11_half,
    double fort_v15_ax,
    double fort_v16_ay,
    double fort_v17_az,
    double fort_v21_ax,
    double fort_v22_ay,
    double fort_v23_az,
    int fort_internal_lower0,
    int fort_internal_stride0,
    std::size_t fort_internal_extent0,
    int fort_internal_lower1,
    int fort_internal_stride1,
    std::size_t fort_internal_extent1,
    int fort_internal_lower2,
    int fort_internal_stride2,
    std::size_t fort_internal_extent2,
    std::size_t fort_internal_total
) {
    const std::size_t fort_internal_grid_stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::size_t fort_internal_point = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (fort_internal_point < fort_internal_total) {
        std::size_t fort_internal_index = fort_internal_point;
        const std::size_t fort_internal_ordinal2 = fort_internal_index % fort_internal_extent2;
        fort_internal_index /= fort_internal_extent2;
        const std::size_t fort_internal_ordinal1 = fort_internal_index % fort_internal_extent1;
        fort_internal_index /= fort_internal_extent1;
        const std::size_t fort_internal_ordinal0 = fort_internal_index % fort_internal_extent0;
        fort_internal_index /= fort_internal_extent0;
        const int fort_v14_k = static_cast<int>(static_cast<long long>(fort_internal_lower0) + static_cast<long long>(fort_internal_ordinal0) * fort_internal_stride0);
        const int fort_v13_j = static_cast<int>(static_cast<long long>(fort_internal_lower1) + static_cast<long long>(fort_internal_ordinal1) * fort_internal_stride1);
        const int fort_v12_i = static_cast<int>(static_cast<long long>(fort_internal_lower2) + static_cast<long long>(fort_internal_ordinal2) * fort_internal_stride2);
        double fort_v24_vadv;
        double fort_v25_wadv;
        fort_v0_u2[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v0_u2_dim1, fort_v0_u2_dim2, fort_v0_u2_dim3)] = fort_v10_zero;
        fort_v0_u2[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v0_u2_dim1, fort_v0_u2_dim2, fort_v0_u2_dim3)] = (-(((((fort_v15_ax * (fort_v1_u[F_IDX((fort_v12_i + 1), fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] + fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)])) * (fort_v1_u[F_IDX((fort_v12_i + 1), fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] + fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)])) - ((fort_v15_ax * (fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] + fort_v1_u[F_IDX((fort_v12_i - 1), fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)])) * (fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] + fort_v1_u[F_IDX((fort_v12_i - 1), fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)]))) + (((fort_v16_ay * (fort_v1_u[F_IDX(fort_v12_i, (fort_v13_j + 1), fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] + fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)])) * (fort_v2_v[F_IDX((fort_v12_i + 1), fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] + fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)])) - ((fort_v16_ay * (fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] + fort_v1_u[F_IDX(fort_v12_i, (fort_v13_j - 1), fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)])) * (fort_v2_v[F_IDX((fort_v12_i + 1), (fort_v13_j - 1), fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] + fort_v2_v[F_IDX(fort_v12_i, (fort_v13_j - 1), fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)])))) + (((fort_v17_az * (fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, (fort_v14_k + 1), fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] + fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)])) * (fort_v3_w[F_IDX((fort_v12_i + 1), fort_v13_j, fort_v14_k, fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)] + fort_v3_w[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)])) - ((fort_v17_az * (fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] + fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, (fort_v14_k - 1), fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)])) * (fort_v3_w[F_IDX((fort_v12_i + 1), fort_v13_j, (fort_v14_k - 1), fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)] + fort_v3_w[F_IDX(fort_v12_i, fort_v13_j, (fort_v14_k - 1), fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)])))));
        fort_v24_vadv = (((fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] + fort_v2_v[F_IDX((fort_v12_i + 1), fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)]) + fort_v2_v[F_IDX(fort_v12_i, (fort_v13_j - 1), fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)]) + fort_v2_v[F_IDX((fort_v12_i + 1), (fort_v13_j - 1), fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)]);
        fort_v25_wadv = (((fort_v3_w[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)] + fort_v3_w[F_IDX((fort_v12_i + 1), fort_v13_j, fort_v14_k, fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)]) + fort_v3_w[F_IDX(fort_v12_i, fort_v13_j, (fort_v14_k - 1), fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)]) + fort_v3_w[F_IDX((fort_v12_i + 1), fort_v13_j, (fort_v14_k - 1), fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)]);
        fort_v0_u2[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v0_u2_dim1, fort_v0_u2_dim2, fort_v0_u2_dim3)] = (fort_v0_u2[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v0_u2_dim1, fort_v0_u2_dim2, fort_v0_u2_dim3)] - ((((fort_v21_ax * (fort_v1_u[F_IDX((fort_v12_i + 1), fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] - fort_v1_u[F_IDX((fort_v12_i - 1), fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)])) * fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)]) + ((fort_v22_ay * (fort_v1_u[F_IDX(fort_v12_i, (fort_v13_j + 1), fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] - fort_v1_u[F_IDX(fort_v12_i, (fort_v13_j - 1), fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)])) * fort_v24_vadv)) + ((fort_v23_az * (fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, (fort_v14_k + 1), fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] - fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, (fort_v14_k - 1), fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)])) * fort_v25_wadv)));
        fort_v0_u2[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v0_u2_dim1, fort_v0_u2_dim2, fort_v0_u2_dim3)] = (fort_v0_u2[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v0_u2_dim1, fort_v0_u2_dim2, fort_v0_u2_dim3)] * fort_v11_half);
        if (fort_internal_total - fort_internal_point <= fort_internal_grid_stride) break;
        fort_internal_point += fort_internal_grid_stride;
    }
}

struct fort_internal_cdu_workspace_state {
    storage::Buffer<double> fort_v0_u2;
    storage::Buffer<double> fort_v1_u;
    storage::Buffer<double> fort_v2_v;
    storage::Buffer<double> fort_v3_w;
    fort_internal_cdu_workspace_state(std::size_t fort_v0_u2_dim1, std::size_t fort_v0_u2_dim2, std::size_t fort_v0_u2_dim3, std::size_t fort_v1_u_dim1, std::size_t fort_v1_u_dim2, std::size_t fort_v1_u_dim3, std::size_t fort_v2_v_dim1, std::size_t fort_v2_v_dim2, std::size_t fort_v2_v_dim3, std::size_t fort_v3_w_dim1, std::size_t fort_v3_w_dim2, std::size_t fort_v3_w_dim3) : fort_v0_u2({fort_v0_u2_dim1, fort_v0_u2_dim2, fort_v0_u2_dim3}), fort_v1_u({fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3}), fort_v2_v({fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3}), fort_v3_w({fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3}) {}
};
static storage::Registry<fort_internal_cdu_workspace_state> fort_internal_cdu_workspace_registry;

extern "C" std::int64_t cpp_cdu_create(
    double* __restrict__ fort_v0_u2,
    std::size_t fort_v0_u2_dim1,
    std::size_t fort_v0_u2_dim2,
    std::size_t fort_v0_u2_dim3,
    const double* __restrict__ fort_v1_u,
    std::size_t fort_v1_u_dim1,
    std::size_t fort_v1_u_dim2,
    std::size_t fort_v1_u_dim3,
    const double* __restrict__ fort_v2_v,
    std::size_t fort_v2_v_dim1,
    std::size_t fort_v2_v_dim2,
    std::size_t fort_v2_v_dim3,
    const double* __restrict__ fort_v3_w,
    std::size_t fort_v3_w_dim1,
    std::size_t fort_v3_w_dim2,
    std::size_t fort_v3_w_dim3
) {
    const auto fort_internal_token = fort_internal_cdu_workspace_registry.create(fort_v0_u2_dim1, fort_v0_u2_dim2, fort_v0_u2_dim3, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3, fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3);
    auto& fort_internal_state = fort_internal_cdu_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v1_u.update_device(fort_v1_u, {fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3});
    fort_internal_state.fort_v2_v.update_device(fort_v2_v, {fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3});
    fort_internal_state.fort_v3_w.update_device(fort_v3_w, {fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3});
    return fort_internal_token;
}

extern "C" void cpp_cdu_run(
    std::int64_t fort_internal_token,
    double fort_v4_dxmin,
    double fort_v5_dymin,
    double fort_v6_dzmin,
    int fort_v7_unx,
    int fort_v8_uny,
    int fort_v9_unz
) {
    auto& fort_internal_state = fort_internal_cdu_workspace_registry.get(fort_internal_token);
    double* fort_v0_u2 = nullptr;
    double* fort_v0_u2_device = nullptr;
    const std::size_t fort_v0_u2_dim1 = fort_internal_state.fort_v0_u2.extent(0);
    const std::size_t fort_v0_u2_dim2 = fort_internal_state.fort_v0_u2.extent(1);
    const std::size_t fort_v0_u2_dim3 = fort_internal_state.fort_v0_u2.extent(2);
    double* fort_v1_u = nullptr;
    double* fort_v1_u_device = nullptr;
    const std::size_t fort_v1_u_dim1 = fort_internal_state.fort_v1_u.extent(0);
    const std::size_t fort_v1_u_dim2 = fort_internal_state.fort_v1_u.extent(1);
    const std::size_t fort_v1_u_dim3 = fort_internal_state.fort_v1_u.extent(2);
    double* fort_v2_v = nullptr;
    double* fort_v2_v_device = nullptr;
    const std::size_t fort_v2_v_dim1 = fort_internal_state.fort_v2_v.extent(0);
    const std::size_t fort_v2_v_dim2 = fort_internal_state.fort_v2_v.extent(1);
    const std::size_t fort_v2_v_dim3 = fort_internal_state.fort_v2_v.extent(2);
    double* fort_v3_w = nullptr;
    double* fort_v3_w_device = nullptr;
    const std::size_t fort_v3_w_dim1 = fort_internal_state.fort_v3_w.extent(0);
    const std::size_t fort_v3_w_dim2 = fort_internal_state.fort_v3_w.extent(1);
    const std::size_t fort_v3_w_dim3 = fort_internal_state.fort_v3_w.extent(2);
    double fort_v10_zero;
    double fort_v11_half;
    double fort_v15_ax;
    double fort_v16_ay;
    double fort_v17_az;
    double fort_v21_ax;
    double fort_v22_ay;
    double fort_v23_az;
    measure_kernel_executions([&]() {
        fort_v10_zero = 0.0;
        fort_v11_half = 0.5;
        fort_v15_ax = (0.25 / fort_v4_dxmin);
        fort_v16_ay = (0.25 / fort_v5_dymin);
        fort_v17_az = (0.25 / fort_v6_dzmin);
        fort_v21_ax = (0.5 / fort_v4_dxmin);
        fort_v22_ay = (0.125 / fort_v5_dymin);
        fort_v23_az = (0.125 / fort_v6_dzmin);
        fort_v0_u2_device = fort_internal_state.fort_v0_u2.device_data();
        fort_v1_u_device = fort_internal_state.fort_v1_u.device_data();
        fort_v2_v_device = fort_internal_state.fort_v2_v.device_data();
        fort_v3_w_device = fort_internal_state.fort_v3_w.device_data();
        {
            // Scheduled parallel region 0.
            const int fort_internal_lower0 = 2;
            const int fort_internal_upper0 = (fort_v9_unz + 1);
            const int fort_internal_stride0 = 1;
            if (fort_internal_stride0 == 0) {
                std::cerr << "/home/jirka/research/fortran-abomination/benchmarks/cases/CDU/Fortran/cdu.f90:18 (inlined through CDU at /home/jirka/research/fortran-abomination/benchmarks/cases/CDU/Fortran/cdu.f90:116 -> set): DO stride must be nonzero" << std::endl;
                std::abort();
            }
            const std::size_t fort_internal_extent0 = fort_internal_stride0 > 0 && fort_internal_upper0 >= fort_internal_lower0
                ? static_cast<std::size_t>((static_cast<long long>(fort_internal_upper0) - fort_internal_lower0) / fort_internal_stride0 + 1)
                : fort_internal_stride0 < 0 && fort_internal_lower0 >= fort_internal_upper0
                    ? static_cast<std::size_t>((static_cast<long long>(fort_internal_lower0) - fort_internal_upper0)
                        / -static_cast<long long>(fort_internal_stride0) + 1) : 0;
            const int fort_internal_lower1 = (fort_internal_extent0 > 0) ? (2) : 0;
            const int fort_internal_upper1 = (fort_internal_extent0 > 0) ? ((fort_v8_uny + 1)) : 0;
            const int fort_internal_stride1 = (fort_internal_extent0 > 0) ? (1) : 0;
            if ((fort_internal_extent0 > 0) && fort_internal_stride1 == 0) {
                std::cerr << "/home/jirka/research/fortran-abomination/benchmarks/cases/CDU/Fortran/cdu.f90:19 (inlined through CDU at /home/jirka/research/fortran-abomination/benchmarks/cases/CDU/Fortran/cdu.f90:116 -> set): DO stride must be nonzero" << std::endl;
                std::abort();
            }
            const std::size_t fort_internal_extent1 = fort_internal_stride1 > 0 && fort_internal_upper1 >= fort_internal_lower1
                ? static_cast<std::size_t>((static_cast<long long>(fort_internal_upper1) - fort_internal_lower1) / fort_internal_stride1 + 1)
                : fort_internal_stride1 < 0 && fort_internal_lower1 >= fort_internal_upper1
                    ? static_cast<std::size_t>((static_cast<long long>(fort_internal_lower1) - fort_internal_upper1)
                        / -static_cast<long long>(fort_internal_stride1) + 1) : 0;
            const int fort_internal_lower2 = (fort_internal_extent0 > 0 && fort_internal_extent1 > 0) ? (2) : 0;
            const int fort_internal_upper2 = (fort_internal_extent0 > 0 && fort_internal_extent1 > 0) ? ((fort_v7_unx + 1)) : 0;
            const int fort_internal_stride2 = (fort_internal_extent0 > 0 && fort_internal_extent1 > 0) ? (1) : 0;
            if ((fort_internal_extent0 > 0 && fort_internal_extent1 > 0) && fort_internal_stride2 == 0) {
                std::cerr << "/home/jirka/research/fortran-abomination/benchmarks/cases/CDU/Fortran/cdu.f90:20 (inlined through CDU at /home/jirka/research/fortran-abomination/benchmarks/cases/CDU/Fortran/cdu.f90:116 -> set): DO stride must be nonzero" << std::endl;
                std::abort();
            }
            const std::size_t fort_internal_extent2 = fort_internal_stride2 > 0 && fort_internal_upper2 >= fort_internal_lower2
                ? static_cast<std::size_t>((static_cast<long long>(fort_internal_upper2) - fort_internal_lower2) / fort_internal_stride2 + 1)
                : fort_internal_stride2 < 0 && fort_internal_lower2 >= fort_internal_upper2
                    ? static_cast<std::size_t>((static_cast<long long>(fort_internal_lower2) - fort_internal_upper2)
                        / -static_cast<long long>(fort_internal_stride2) + 1) : 0;
            std::size_t fort_internal_total = 0;
            if ((fort_internal_extent2) != 0 && (fort_internal_extent1) != 0 && (fort_internal_extent0) != 0) {
                fort_internal_total = 1;
                if (fort_internal_total > static_cast<std::size_t>(-1) / (fort_internal_extent2)) {
                    std::cerr << "Iteration size product overflows size_t" << std::endl;
                    std::abort();
                }
                fort_internal_total *= (fort_internal_extent2);
                if (fort_internal_total > static_cast<std::size_t>(-1) / (fort_internal_extent1)) {
                    std::cerr << "Iteration size product overflows size_t" << std::endl;
                    std::abort();
                }
                fort_internal_total *= (fort_internal_extent1);
                if (fort_internal_total > static_cast<std::size_t>(-1) / (fort_internal_extent0)) {
                    std::cerr << "Iteration size product overflows size_t" << std::endl;
                    std::abort();
                }
                fort_internal_total *= (fort_internal_extent0);
            }
            if (fort_internal_total > 0) {
                constexpr unsigned int fort_internal_threads = 256;
                const std::size_t fort_internal_needed_blocks = (fort_internal_total - 1) / fort_internal_threads + 1;
                const unsigned int fort_internal_blocks = static_cast<unsigned int>(
                    fort_internal_needed_blocks > 65535 ? 65535 : fort_internal_needed_blocks);
                kernel_region_0_device<<<fort_internal_blocks, fort_internal_threads>>>(
                    fort_v0_u2_device,
                    fort_v0_u2_dim1,
                    fort_v0_u2_dim2,
                    fort_v0_u2_dim3,
                    fort_v1_u_device,
                    fort_v1_u_dim1,
                    fort_v1_u_dim2,
                    fort_v1_u_dim3,
                    fort_v2_v_device,
                    fort_v2_v_dim1,
                    fort_v2_v_dim2,
                    fort_v2_v_dim3,
                    fort_v3_w_device,
                    fort_v3_w_dim1,
                    fort_v3_w_dim2,
                    fort_v3_w_dim3,
                    fort_v7_unx,
                    fort_v8_uny,
                    fort_v9_unz,
                    fort_v10_zero,
                    fort_v11_half,
                    fort_v15_ax,
                    fort_v16_ay,
                    fort_v17_az,
                    fort_v21_ax,
                    fort_v22_ay,
                    fort_v23_az,
                    fort_internal_lower0,
                    fort_internal_stride0,
                    fort_internal_extent0,
                    fort_internal_lower1,
                    fort_internal_stride1,
                    fort_internal_extent1,
                    fort_internal_lower2,
                    fort_internal_stride2,
                    fort_internal_extent2,
                    fort_internal_total
                );
                CUCH(cudaGetLastError());
                storage::trace("kernel");
            }
        }
        fort_internal_state.fort_v0_u2.device_written();
    });
}

extern "C" void cpp_cdu_workspace_validate(
    std::int64_t fort_internal_token
) {
    fort_internal_cdu_workspace_registry.get(fort_internal_token);
    storage::synchronize();
}

extern "C" void cpp_cdu_update_device_0(
    std::int64_t fort_internal_token,
    const double* fort_v0_u2,
    std::size_t fort_v0_u2_dim1,
    std::size_t fort_v0_u2_dim2,
    std::size_t fort_v0_u2_dim3
) {
    auto& fort_internal_state = fort_internal_cdu_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v0_u2.update_device(fort_v0_u2, {fort_v0_u2_dim1, fort_v0_u2_dim2, fort_v0_u2_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdu_update_device_1(
    std::int64_t fort_internal_token,
    const double* fort_v1_u,
    std::size_t fort_v1_u_dim1,
    std::size_t fort_v1_u_dim2,
    std::size_t fort_v1_u_dim3
) {
    auto& fort_internal_state = fort_internal_cdu_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v1_u.update_device(fort_v1_u, {fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdu_update_device_2(
    std::int64_t fort_internal_token,
    const double* fort_v2_v,
    std::size_t fort_v2_v_dim1,
    std::size_t fort_v2_v_dim2,
    std::size_t fort_v2_v_dim3
) {
    auto& fort_internal_state = fort_internal_cdu_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v2_v.update_device(fort_v2_v, {fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdu_update_device_3(
    std::int64_t fort_internal_token,
    const double* fort_v3_w,
    std::size_t fort_v3_w_dim1,
    std::size_t fort_v3_w_dim2,
    std::size_t fort_v3_w_dim3
) {
    auto& fort_internal_state = fort_internal_cdu_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v3_w.update_device(fort_v3_w, {fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdu_update_host_0(
    std::int64_t fort_internal_token,
    double* fort_v0_u2,
    std::size_t fort_v0_u2_dim1,
    std::size_t fort_v0_u2_dim2,
    std::size_t fort_v0_u2_dim3
) {
    auto& fort_internal_state = fort_internal_cdu_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v0_u2.update_host(fort_v0_u2, {fort_v0_u2_dim1, fort_v0_u2_dim2, fort_v0_u2_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdu_update_host_1(
    std::int64_t fort_internal_token,
    double* fort_v1_u,
    std::size_t fort_v1_u_dim1,
    std::size_t fort_v1_u_dim2,
    std::size_t fort_v1_u_dim3
) {
    auto& fort_internal_state = fort_internal_cdu_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v1_u.update_host(fort_v1_u, {fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdu_update_host_2(
    std::int64_t fort_internal_token,
    double* fort_v2_v,
    std::size_t fort_v2_v_dim1,
    std::size_t fort_v2_v_dim2,
    std::size_t fort_v2_v_dim3
) {
    auto& fort_internal_state = fort_internal_cdu_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v2_v.update_host(fort_v2_v, {fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdu_update_host_3(
    std::int64_t fort_internal_token,
    double* fort_v3_w,
    std::size_t fort_v3_w_dim1,
    std::size_t fort_v3_w_dim2,
    std::size_t fort_v3_w_dim3
) {
    auto& fort_internal_state = fort_internal_cdu_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v3_w.update_host(fort_v3_w, {fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdu_destroy(
    std::int64_t fort_internal_token
) {
    fort_internal_cdu_workspace_registry.destroy(fort_internal_token);
}

extern "C" void cpp_start_hot() { reset_timing_vectors(); }
extern "C" void cpp_finish_hot() { print_timing_summary(); }

extern "C" void cpp_CDU(
    double* __restrict__ fort_v0_u2,
    std::size_t fort_v0_u2_dim1,
    std::size_t fort_v0_u2_dim2,
    std::size_t fort_v0_u2_dim3,
    const double* __restrict__ fort_v1_u,
    std::size_t fort_v1_u_dim1,
    std::size_t fort_v1_u_dim2,
    std::size_t fort_v1_u_dim3,
    const double* __restrict__ fort_v2_v,
    std::size_t fort_v2_v_dim1,
    std::size_t fort_v2_v_dim2,
    std::size_t fort_v2_v_dim3,
    const double* __restrict__ fort_v3_w,
    std::size_t fort_v3_w_dim1,
    std::size_t fort_v3_w_dim2,
    std::size_t fort_v3_w_dim3,
    double fort_v4_dxmin,
    double fort_v5_dymin,
    double fort_v6_dzmin,
    int fort_v7_unx,
    int fort_v8_uny,
    int fort_v9_unz
) {
    const auto fort_internal_token = cpp_cdu_create(fort_v0_u2, fort_v0_u2_dim1, fort_v0_u2_dim2, fort_v0_u2_dim3, fort_v1_u, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3, fort_v2_v, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3, fort_v3_w, fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3);
    cpp_cdu_run(fort_internal_token, fort_v4_dxmin, fort_v5_dymin, fort_v6_dzmin, fort_v7_unx, fort_v8_uny, fort_v9_unz);
    storage::synchronize();
    auto& fort_internal_state = fort_internal_cdu_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v0_u2.update_host(fort_v0_u2, {fort_v0_u2_dim1, fort_v0_u2_dim2, fort_v0_u2_dim3});
    cpp_cdu_destroy(fort_internal_token);
}
}
