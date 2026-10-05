#include <cstdlib>
#include "common_functions.cuh"

namespace generated_kernels {
using namespace generated_kernels::indexing;

extern "C" void cpp_start_hot() {}
extern "C" void cpp_finish_hot() {}

extern "C" void cpp_CDV(
    double* __restrict__ fort_v0_v2,
    std::size_t fort_v0_v2_dim1,
    std::size_t fort_v0_v2_dim2,
    std::size_t fort_v0_v2_dim3,
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
    int fort_v7_vnx,
    int fort_v8_vny,
    int fort_v9_vnz
) {
    double fort_v10_zero;
    double fort_v11_half;
    double fort_v15_ax;
    double fort_v16_ay;
    double fort_v17_az;
    double fort_v21_ax;
    double fort_v22_ay;
    double fort_v23_az;
    fort_v10_zero = 0.0;
    fort_v11_half = 0.5;
    fort_v15_ax = (0.25 / fort_v4_dxmin);
    fort_v16_ay = (0.25 / fort_v5_dymin);
    fort_v17_az = (0.25 / fort_v6_dzmin);
    fort_v21_ax = (0.125 / fort_v4_dxmin);
    fort_v22_ay = (0.5 / fort_v5_dymin);
    fort_v23_az = (0.125 / fort_v6_dzmin);
    {
        // Verified parallel region 0.
        const int fort_internal_lower0 = 2;
        const int fort_internal_upper0 = (fort_v9_vnz + 1);
        const int fort_internal_stride0 = 1;
        if (fort_internal_stride0 == 0) {
            std::cerr << "/home/jirka/research/fortran-abomination/benchmarks/cases/CDV/Fortran/cvd.f90:18 (inlined through CDV at /home/jirka/research/fortran-abomination/benchmarks/cases/CDV/Fortran/cvd.f90:116 -> set): DO stride must be nonzero" << std::endl;
            std::abort();
        }
        const std::size_t fort_internal_extent0 = fort_internal_stride0 > 0 && fort_internal_upper0 >= fort_internal_lower0
            ? static_cast<std::size_t>((static_cast<long long>(fort_internal_upper0) - fort_internal_lower0) / fort_internal_stride0 + 1)
            : fort_internal_stride0 < 0 && fort_internal_lower0 >= fort_internal_upper0
                ? static_cast<std::size_t>((static_cast<long long>(fort_internal_lower0) - fort_internal_upper0)
                    / -static_cast<long long>(fort_internal_stride0) + 1) : 0;
        const int fort_internal_lower1 = (fort_internal_extent0 > 0) ? (2) : 0;
        const int fort_internal_upper1 = (fort_internal_extent0 > 0) ? ((fort_v8_vny + 1)) : 0;
        const int fort_internal_stride1 = (fort_internal_extent0 > 0) ? (1) : 0;
        if ((fort_internal_extent0 > 0) && fort_internal_stride1 == 0) {
            std::cerr << "/home/jirka/research/fortran-abomination/benchmarks/cases/CDV/Fortran/cvd.f90:19 (inlined through CDV at /home/jirka/research/fortran-abomination/benchmarks/cases/CDV/Fortran/cvd.f90:116 -> set): DO stride must be nonzero" << std::endl;
            std::abort();
        }
        const std::size_t fort_internal_extent1 = fort_internal_stride1 > 0 && fort_internal_upper1 >= fort_internal_lower1
            ? static_cast<std::size_t>((static_cast<long long>(fort_internal_upper1) - fort_internal_lower1) / fort_internal_stride1 + 1)
            : fort_internal_stride1 < 0 && fort_internal_lower1 >= fort_internal_upper1
                ? static_cast<std::size_t>((static_cast<long long>(fort_internal_lower1) - fort_internal_upper1)
                    / -static_cast<long long>(fort_internal_stride1) + 1) : 0;
        const int fort_internal_lower2 = (fort_internal_extent0 > 0 && fort_internal_extent1 > 0) ? (2) : 0;
        const int fort_internal_upper2 = (fort_internal_extent0 > 0 && fort_internal_extent1 > 0) ? ((fort_v7_vnx + 1)) : 0;
        const int fort_internal_stride2 = (fort_internal_extent0 > 0 && fort_internal_extent1 > 0) ? (1) : 0;
        if ((fort_internal_extent0 > 0 && fort_internal_extent1 > 0) && fort_internal_stride2 == 0) {
            std::cerr << "/home/jirka/research/fortran-abomination/benchmarks/cases/CDV/Fortran/cvd.f90:20 (inlined through CDV at /home/jirka/research/fortran-abomination/benchmarks/cases/CDV/Fortran/cvd.f90:116 -> set): DO stride must be nonzero" << std::endl;
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
        #pragma omp parallel for collapse(3)
        for (std::size_t fort_internal_ordinal0 = 0;
             fort_internal_ordinal0 < fort_internal_extent0; ++fort_internal_ordinal0) {
            for (std::size_t fort_internal_ordinal1 = 0;
                 fort_internal_ordinal1 < fort_internal_extent1; ++fort_internal_ordinal1) {
                for (std::size_t fort_internal_ordinal2 = 0;
                     fort_internal_ordinal2 < fort_internal_extent2; ++fort_internal_ordinal2) {
                    const int fort_v14_k = static_cast<int>(static_cast<long long>(fort_internal_lower0) + static_cast<long long>(fort_internal_ordinal0) * fort_internal_stride0);
                    const int fort_v13_j = static_cast<int>(static_cast<long long>(fort_internal_lower1) + static_cast<long long>(fort_internal_ordinal1) * fort_internal_stride1);
                    const int fort_v12_i = static_cast<int>(static_cast<long long>(fort_internal_lower2) + static_cast<long long>(fort_internal_ordinal2) * fort_internal_stride2);
                    double fort_v24_uadv;
                    double fort_v25_wadv;
                    fort_v0_v2[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v0_v2_dim1, fort_v0_v2_dim2, fort_v0_v2_dim3)] = fort_v10_zero;
                    fort_v0_v2[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v0_v2_dim1, fort_v0_v2_dim2, fort_v0_v2_dim3)] = (-(((((fort_v16_ay * (fort_v2_v[F_IDX(fort_v12_i, (fort_v13_j + 1), fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] + fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)])) * (fort_v2_v[F_IDX(fort_v12_i, (fort_v13_j + 1), fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] + fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)])) - ((fort_v16_ay * (fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] + fort_v2_v[F_IDX(fort_v12_i, (fort_v13_j - 1), fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)])) * (fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] + fort_v2_v[F_IDX(fort_v12_i, (fort_v13_j - 1), fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)]))) + (((fort_v15_ax * (fort_v2_v[F_IDX((fort_v12_i + 1), fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] + fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)])) * (fort_v1_u[F_IDX(fort_v12_i, (fort_v13_j + 1), fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] + fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)])) - ((fort_v15_ax * (fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] + fort_v2_v[F_IDX((fort_v12_i - 1), fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)])) * (fort_v1_u[F_IDX((fort_v12_i - 1), (fort_v13_j + 1), fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] + fort_v1_u[F_IDX((fort_v12_i - 1), fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)])))) + (((fort_v17_az * (fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, (fort_v14_k + 1), fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] + fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)])) * (fort_v3_w[F_IDX(fort_v12_i, (fort_v13_j + 1), fort_v14_k, fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)] + fort_v3_w[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)])) - ((fort_v17_az * (fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] + fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, (fort_v14_k - 1), fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)])) * (fort_v3_w[F_IDX(fort_v12_i, (fort_v13_j + 1), (fort_v14_k - 1), fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)] + fort_v3_w[F_IDX(fort_v12_i, fort_v13_j, (fort_v14_k - 1), fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)])))));
                    fort_v24_uadv = (((fort_v1_u[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)] + fort_v1_u[F_IDX(fort_v12_i, (fort_v13_j + 1), fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)]) + fort_v1_u[F_IDX((fort_v12_i - 1), fort_v13_j, fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)]) + fort_v1_u[F_IDX((fort_v12_i - 1), (fort_v13_j + 1), fort_v14_k, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3)]);
                    fort_v25_wadv = (((fort_v3_w[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)] + fort_v3_w[F_IDX(fort_v12_i, (fort_v13_j + 1), fort_v14_k, fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)]) + fort_v3_w[F_IDX(fort_v12_i, fort_v13_j, (fort_v14_k - 1), fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)]) + fort_v3_w[F_IDX(fort_v12_i, (fort_v13_j + 1), (fort_v14_k - 1), fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3)]);
                    fort_v0_v2[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v0_v2_dim1, fort_v0_v2_dim2, fort_v0_v2_dim3)] = (fort_v0_v2[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v0_v2_dim1, fort_v0_v2_dim2, fort_v0_v2_dim3)] - ((((fort_v21_ax * (fort_v2_v[F_IDX((fort_v12_i + 1), fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] - fort_v2_v[F_IDX((fort_v12_i - 1), fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)])) * fort_v24_uadv) + ((fort_v22_ay * (fort_v2_v[F_IDX(fort_v12_i, (fort_v13_j + 1), fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] - fort_v2_v[F_IDX(fort_v12_i, (fort_v13_j - 1), fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)])) * fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)])) + ((fort_v23_az * (fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, (fort_v14_k + 1), fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)] - fort_v2_v[F_IDX(fort_v12_i, fort_v13_j, (fort_v14_k - 1), fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3)])) * fort_v25_wadv)));
                    fort_v0_v2[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v0_v2_dim1, fort_v0_v2_dim2, fort_v0_v2_dim3)] = (fort_v0_v2[F_IDX(fort_v12_i, fort_v13_j, fort_v14_k, fort_v0_v2_dim1, fort_v0_v2_dim2, fort_v0_v2_dim3)] * fort_v11_half);
                }
            }
        }
    }
}
}

namespace generated_kernels {
struct fort_internal_cdv_workspace_state {
    storage::Buffer<double> fort_v0_v2;
    storage::Buffer<double> fort_v1_u;
    storage::Buffer<double> fort_v2_v;
    storage::Buffer<double> fort_v3_w;
    fort_internal_cdv_workspace_state(std::size_t fort_v0_v2_dim1, std::size_t fort_v0_v2_dim2, std::size_t fort_v0_v2_dim3, std::size_t fort_v1_u_dim1, std::size_t fort_v1_u_dim2, std::size_t fort_v1_u_dim3, std::size_t fort_v2_v_dim1, std::size_t fort_v2_v_dim2, std::size_t fort_v2_v_dim3, std::size_t fort_v3_w_dim1, std::size_t fort_v3_w_dim2, std::size_t fort_v3_w_dim3) : fort_v0_v2({fort_v0_v2_dim1, fort_v0_v2_dim2, fort_v0_v2_dim3}), fort_v1_u({fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3}), fort_v2_v({fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3}), fort_v3_w({fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3}) {}
};
static storage::Registry<fort_internal_cdv_workspace_state> fort_internal_cdv_workspace_registry;

extern "C" std::int64_t cpp_cdv_create(
    double* __restrict__ fort_v0_v2,
    std::size_t fort_v0_v2_dim1,
    std::size_t fort_v0_v2_dim2,
    std::size_t fort_v0_v2_dim3,
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
    const auto fort_internal_token = fort_internal_cdv_workspace_registry.create(fort_v0_v2_dim1, fort_v0_v2_dim2, fort_v0_v2_dim3, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3, fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3);
    auto& fort_internal_state = fort_internal_cdv_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v1_u.update_device(fort_v1_u, {fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3});
    fort_internal_state.fort_v2_v.update_device(fort_v2_v, {fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3});
    fort_internal_state.fort_v3_w.update_device(fort_v3_w, {fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3});
    return fort_internal_token;
}

extern "C" void cpp_cdv_run(
    std::int64_t fort_internal_token,
    double fort_v4_dxmin,
    double fort_v5_dymin,
    double fort_v6_dzmin,
    int fort_v7_vnx,
    int fort_v8_vny,
    int fort_v9_vnz
) {
    auto& fort_internal_state = fort_internal_cdv_workspace_registry.get(fort_internal_token);
    double* fort_v0_v2 = fort_internal_state.fort_v0_v2.host_data();
    const std::size_t fort_v0_v2_dim1 = fort_internal_state.fort_v0_v2.extent(0);
    const std::size_t fort_v0_v2_dim2 = fort_internal_state.fort_v0_v2.extent(1);
    const std::size_t fort_v0_v2_dim3 = fort_internal_state.fort_v0_v2.extent(2);
    double* fort_v1_u = fort_internal_state.fort_v1_u.host_data();
    const std::size_t fort_v1_u_dim1 = fort_internal_state.fort_v1_u.extent(0);
    const std::size_t fort_v1_u_dim2 = fort_internal_state.fort_v1_u.extent(1);
    const std::size_t fort_v1_u_dim3 = fort_internal_state.fort_v1_u.extent(2);
    double* fort_v2_v = fort_internal_state.fort_v2_v.host_data();
    const std::size_t fort_v2_v_dim1 = fort_internal_state.fort_v2_v.extent(0);
    const std::size_t fort_v2_v_dim2 = fort_internal_state.fort_v2_v.extent(1);
    const std::size_t fort_v2_v_dim3 = fort_internal_state.fort_v2_v.extent(2);
    double* fort_v3_w = fort_internal_state.fort_v3_w.host_data();
    const std::size_t fort_v3_w_dim1 = fort_internal_state.fort_v3_w.extent(0);
    const std::size_t fort_v3_w_dim2 = fort_internal_state.fort_v3_w.extent(1);
    const std::size_t fort_v3_w_dim3 = fort_internal_state.fort_v3_w.extent(2);
    cpp_CDV(fort_v0_v2, fort_v0_v2_dim1, fort_v0_v2_dim2, fort_v0_v2_dim3, fort_v1_u, fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3, fort_v2_v, fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3, fort_v3_w, fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3, fort_v4_dxmin, fort_v5_dymin, fort_v6_dzmin, fort_v7_vnx, fort_v8_vny, fort_v9_vnz);
    fort_internal_state.fort_v0_v2.host_written();
}

extern "C" void cpp_cdv_workspace_validate(
    std::int64_t fort_internal_token
) {
    fort_internal_cdv_workspace_registry.get(fort_internal_token);
    storage::synchronize();
}

extern "C" void cpp_cdv_update_device_0(
    std::int64_t fort_internal_token,
    const double* fort_v0_v2,
    std::size_t fort_v0_v2_dim1,
    std::size_t fort_v0_v2_dim2,
    std::size_t fort_v0_v2_dim3
) {
    auto& fort_internal_state = fort_internal_cdv_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v0_v2.update_device(fort_v0_v2, {fort_v0_v2_dim1, fort_v0_v2_dim2, fort_v0_v2_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdv_update_device_1(
    std::int64_t fort_internal_token,
    const double* fort_v1_u,
    std::size_t fort_v1_u_dim1,
    std::size_t fort_v1_u_dim2,
    std::size_t fort_v1_u_dim3
) {
    auto& fort_internal_state = fort_internal_cdv_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v1_u.update_device(fort_v1_u, {fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdv_update_device_2(
    std::int64_t fort_internal_token,
    const double* fort_v2_v,
    std::size_t fort_v2_v_dim1,
    std::size_t fort_v2_v_dim2,
    std::size_t fort_v2_v_dim3
) {
    auto& fort_internal_state = fort_internal_cdv_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v2_v.update_device(fort_v2_v, {fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdv_update_device_3(
    std::int64_t fort_internal_token,
    const double* fort_v3_w,
    std::size_t fort_v3_w_dim1,
    std::size_t fort_v3_w_dim2,
    std::size_t fort_v3_w_dim3
) {
    auto& fort_internal_state = fort_internal_cdv_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v3_w.update_device(fort_v3_w, {fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdv_update_host_0(
    std::int64_t fort_internal_token,
    double* fort_v0_v2,
    std::size_t fort_v0_v2_dim1,
    std::size_t fort_v0_v2_dim2,
    std::size_t fort_v0_v2_dim3
) {
    auto& fort_internal_state = fort_internal_cdv_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v0_v2.update_host(fort_v0_v2, {fort_v0_v2_dim1, fort_v0_v2_dim2, fort_v0_v2_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdv_update_host_1(
    std::int64_t fort_internal_token,
    double* fort_v1_u,
    std::size_t fort_v1_u_dim1,
    std::size_t fort_v1_u_dim2,
    std::size_t fort_v1_u_dim3
) {
    auto& fort_internal_state = fort_internal_cdv_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v1_u.update_host(fort_v1_u, {fort_v1_u_dim1, fort_v1_u_dim2, fort_v1_u_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdv_update_host_2(
    std::int64_t fort_internal_token,
    double* fort_v2_v,
    std::size_t fort_v2_v_dim1,
    std::size_t fort_v2_v_dim2,
    std::size_t fort_v2_v_dim3
) {
    auto& fort_internal_state = fort_internal_cdv_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v2_v.update_host(fort_v2_v, {fort_v2_v_dim1, fort_v2_v_dim2, fort_v2_v_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdv_update_host_3(
    std::int64_t fort_internal_token,
    double* fort_v3_w,
    std::size_t fort_v3_w_dim1,
    std::size_t fort_v3_w_dim2,
    std::size_t fort_v3_w_dim3
) {
    auto& fort_internal_state = fort_internal_cdv_workspace_registry.get(fort_internal_token);
    fort_internal_state.fort_v3_w.update_host(fort_v3_w, {fort_v3_w_dim1, fort_v3_w_dim2, fort_v3_w_dim3});
    storage::synchronize();
}

extern "C" void cpp_cdv_destroy(
    std::int64_t fort_internal_token
) {
    fort_internal_cdv_workspace_registry.destroy(fort_internal_token);
}

}
