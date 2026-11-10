static __global__ void kernel_region_0_device(
    const double* __restrict__ fort_v0_u,
    std::size_t fort_v0_u_dim1,
    std::size_t fort_v0_u_dim2,
    std::size_t fort_v0_u_dim3,
    const double* __restrict__ fort_v1_v,
    std::size_t fort_v1_v_dim1,
    std::size_t fort_v1_v_dim2,
    std::size_t fort_v1_v_dim3,
    const double* __restrict__ fort_v2_w,
    std::size_t fort_v2_w_dim1,
    std::size_t fort_v2_w_dim2,
    std::size_t fort_v2_w_dim3,
    double* __restrict__ fort_v3_cfl,
    std::size_t fort_v3_cfl_dim1,
    std::size_t fort_v3_cfl_dim2,
    std::size_t fort_v3_cfl_dim3,
    int fort_v4_nx,
    int fort_v5_ny,
    int fort_v6_nz,
    double fort_v7_dx,
    double fort_v8_dy,
    double fort_v9_dz,
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
        const std::size_t fort_internal_ordinal0 = fort_internal_index;
        const int fort_v12_k = static_cast<int>(static_cast<long long>(fort_internal_lower0) + static_cast<long long>(fort_internal_ordinal0) * fort_internal_stride0);
        const long long fort_internal_wide12 = static_cast<long long>(fort_internal_lower0) + static_cast<long long>(fort_internal_ordinal0) * fort_internal_stride0;
        const int fort_v11_j = static_cast<int>(static_cast<long long>(fort_internal_lower1) + static_cast<long long>(fort_internal_ordinal1) * fort_internal_stride1);
        const long long fort_internal_wide11 = static_cast<long long>(fort_internal_lower1) + static_cast<long long>(fort_internal_ordinal1) * fort_internal_stride1;
        const int fort_v10_i = static_cast<int>(static_cast<long long>(fort_internal_lower2) + static_cast<long long>(fort_internal_ordinal2) * fort_internal_stride2);
        const long long fort_internal_wide10 = static_cast<long long>(fort_internal_lower2) + static_cast<long long>(fort_internal_ordinal2) * fort_internal_stride2;
        double fort_v13_p;
        fort_v13_p = ::generated_kernels::numeric::divide(::generated_kernels::numeric::maximum(::fabs(fort_v0_u[F_IDX((fort_v10_i + 3), (fort_v11_j + 3), (fort_v12_k + 3), fort_v0_u_dim1, fort_v0_u_dim2, fort_v0_u_dim3)]), ::fabs(fort_v0_u[F_IDX((fort_v10_i + 2), (fort_v11_j + 3), (fort_v12_k + 3), fort_v0_u_dim1, fort_v0_u_dim2, fort_v0_u_dim3)])), fort_v7_dx);
        fort_v13_p = (fort_v13_p + ::generated_kernels::numeric::divide(::generated_kernels::numeric::maximum(::fabs(fort_v1_v[F_IDX((fort_v10_i + 3), (fort_v11_j + 3), (fort_v12_k + 3), fort_v1_v_dim1, fort_v1_v_dim2, fort_v1_v_dim3)]), ::fabs(fort_v1_v[F_IDX((fort_v10_i + 3), (fort_v11_j + 2), (fort_v12_k + 3), fort_v1_v_dim1, fort_v1_v_dim2, fort_v1_v_dim3)])), fort_v8_dy));
        fort_v13_p = (fort_v13_p + ::generated_kernels::numeric::divide(::generated_kernels::numeric::maximum(::fabs(fort_v2_w[F_IDX((fort_v10_i + 3), (fort_v11_j + 3), (fort_v12_k + 3), fort_v2_w_dim1, fort_v2_w_dim2, fort_v2_w_dim3)]), ::fabs(fort_v2_w[F_IDX((fort_v10_i + 3), (fort_v11_j + 3), (fort_v12_k + 2), fort_v2_w_dim1, fort_v2_w_dim2, fort_v2_w_dim3)])), fort_v9_dz));
        fort_v3_cfl[F_IDX(fort_internal_wide10, fort_internal_wide11, fort_internal_wide12, fort_v3_cfl_dim1, fort_v3_cfl_dim2, fort_v3_cfl_dim3)] = fort_v13_p;
        if (fort_internal_total - fort_internal_point <= fort_internal_grid_stride) break;
        fort_internal_point += fort_internal_grid_stride;
    }
}
