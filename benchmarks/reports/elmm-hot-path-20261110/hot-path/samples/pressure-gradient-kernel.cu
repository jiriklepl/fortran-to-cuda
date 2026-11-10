static __global__ void kernel_region_0_device(
    double* __restrict__ fort_v0_u,
    std::size_t fort_v0_u_dim1,
    std::size_t fort_v0_u_dim2,
    std::size_t fort_v0_u_dim3,
    int fort_v1_fort_region_lb_u_1,
    int fort_v2_fort_region_lb_u_2,
    int fort_v3_fort_region_lb_u_3,
    double fort_v4_a,
    double fort_v5_pr_gradient_x,
    int fort_v6_unx,
    int fort_v7_uny,
    int fort_v8_unz,
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
        const int fort_v11_k = static_cast<int>(static_cast<long long>(fort_internal_lower0) + static_cast<long long>(fort_internal_ordinal0) * fort_internal_stride0);
        const int fort_v10_j = static_cast<int>(static_cast<long long>(fort_internal_lower1) + static_cast<long long>(fort_internal_ordinal1) * fort_internal_stride1);
        const int fort_v9_i = static_cast<int>(static_cast<long long>(fort_internal_lower2) + static_cast<long long>(fort_internal_ordinal2) * fort_internal_stride2);
        fort_v0_u[F_IDX(((fort_v9_i - fort_v1_fort_region_lb_u_1) + 1), ((fort_v10_j - fort_v2_fort_region_lb_u_2) + 1), ((fort_v11_k - fort_v3_fort_region_lb_u_3) + 1), fort_v0_u_dim1, fort_v0_u_dim2, fort_v0_u_dim3)] = (fort_v0_u[F_IDX(((fort_v9_i - fort_v1_fort_region_lb_u_1) + 1), ((fort_v10_j - fort_v2_fort_region_lb_u_2) + 1), ((fort_v11_k - fort_v3_fort_region_lb_u_3) + 1), fort_v0_u_dim1, fort_v0_u_dim2, fort_v0_u_dim3)] + (fort_v4_a * fort_v5_pr_gradient_x));
        if (fort_internal_total - fort_internal_point <= fort_internal_grid_stride) break;
        fort_internal_point += fort_internal_grid_stride;
    }
}
