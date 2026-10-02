#ifndef COMMON_FUNCTIONS_CUH
#define COMMON_FUNCTIONS_CUH

#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#ifdef __CUDACC__
    #define CUDA_CALLABLE __host__ __device__
#else
    #define CUDA_CALLABLE
#endif

#define CUCH(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace generated_kernels::indexing {

template <size_t Step, size_t N>
struct StaticLoop {
    CUDA_CALLABLE static void iterate(const size_t* arr, size_t& linear_idx, size_t& stride) {
        size_t current_index = arr[Step] - 1;
        size_t current_dim_size = arr[Step + N];

        linear_idx += current_index * stride;
        stride *= current_dim_size;

        StaticLoop<Step + 1, N>::iterate(arr, linear_idx, stride);
    }
};

template <size_t N>
struct StaticLoop<N, N> {
    CUDA_CALLABLE static void iterate(const size_t* arr, size_t& linear_idx, size_t& stride) {
        // Do nothing. The loop is finished.
    }
};

template <typename... Args>
CUDA_CALLABLE size_t F_IDX(Args... args) {
    constexpr size_t total_args = sizeof...(Args);

    static_assert(total_args % 2 == 0, "IDX requires N indices followed by N dimensions.");
    static_assert(total_args > 0, "IDX requires at least 2 arguments.");

    constexpr size_t N = total_args / 2;

    const size_t arr[total_args] = { static_cast<size_t>(args)... };

    size_t linear_idx = 0;
    size_t stride = 1;

    StaticLoop<0, N>::iterate(arr, linear_idx, stride);

    return linear_idx;
}

}
// FORT_RUNTIME_UNITS

#endif // COMMON_FUNCTIONS_CUH
