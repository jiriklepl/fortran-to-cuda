#ifndef COMMON_FUNCTIONS_CUH
#define COMMON_FUNCTIONS_CUH

#include <cstddef>
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
// ── Timing infrastructure — CUDA only ───────────────────────────────────────────
// The functions below use cudaEvent_t and related CUDA runtime APIs.
// They are excluded entirely from plain C++ compilation.
#ifdef __CUDACC__
namespace generated_kernels::timing {

static std::vector<float> g_malloc_ms;
static std::vector<float> g_h2d_ms;
static std::vector<float> g_kernel_ms;
static std::vector<float> g_d2h_ms;
static std::vector<float> g_free_ms;

static std::vector<double> g_h2d_bytes;
static std::vector<double> g_d2h_bytes;

template <typename Fn>
float measure_cuda_event_ms(Fn&& fn) {
    cudaEvent_t start_evt = nullptr;
    cudaEvent_t stop_evt = nullptr;
    float elapsed_ms = 0.0f;

    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);

    // Serialize to reduce overlap when profiling individual phases.
    cudaDeviceSynchronize();
    cudaEventRecord(start_evt, 0);
    fn();
    cudaDeviceSynchronize();
    cudaEventRecord(stop_evt, 0);
    cudaEventSynchronize(stop_evt);

    cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt);
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    return elapsed_ms;
}

template <typename Fn>
void measure_alloc(Fn&& fn) {
    #ifndef MEASURE_CUDA_EXECUTION_TIME
    fn(); return;
    #endif

    float elapsed_ms = measure_cuda_event_ms(std::forward<Fn>(fn));
    g_malloc_ms.push_back(elapsed_ms);
}

template <typename Fn>
void measure_h2d(std::size_t bytes, Fn&& fn) {
    #ifndef MEASURE_CUDA_EXECUTION_TIME
    fn(); return;
    #endif

    float elapsed_ms = measure_cuda_event_ms(std::forward<Fn>(fn));
    g_h2d_ms.push_back(elapsed_ms);
    g_h2d_bytes.push_back(static_cast<double>(bytes));
}

template <typename Fn>
void measure_d2h(std::size_t bytes, Fn&& fn) {
    #ifndef MEASURE_CUDA_EXECUTION_TIME
    fn(); return;
    #endif

    float elapsed_ms = measure_cuda_event_ms(std::forward<Fn>(fn));
    g_d2h_ms.push_back(elapsed_ms);
    g_d2h_bytes.push_back(static_cast<double>(bytes));
}

template <typename Fn>
void measure_free(Fn&& fn) {
    #ifndef MEASURE_CUDA_EXECUTION_TIME
    fn(); return;
    #endif

    float elapsed_ms = measure_cuda_event_ms(std::forward<Fn>(fn));
    g_free_ms.push_back(elapsed_ms);
}

template <typename Fn>
void measure_kernel_executions(Fn&& fn) {
    #ifndef MEASURE_CUDA_EXECUTION_TIME
    fn(); return;
    #endif

    float elapsed_ms = measure_cuda_event_ms(std::forward<Fn>(fn));
    g_kernel_ms.push_back(elapsed_ms);
}

float sum_ms(const std::vector<float>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0f);
}

double sum_bytes(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
}

double throughput_bps(double total_bytes, double total_ms) {
    if (total_ms <= 0.0) {
        return 0.0;
    }
    return (total_bytes * 1000.0) / total_ms;
}

double throughput_gbps(double total_bytes, double total_ms) {
    return throughput_bps(total_bytes, total_ms) / 1.0e9;
}

void reset_timing_vectors() {
    #ifndef MEASURE_CUDA_EXECUTION_TIME
    return;
    #endif

    g_malloc_ms.clear();
    g_h2d_ms.clear();
    g_kernel_ms.clear();
    g_d2h_ms.clear();
    g_free_ms.clear();

    g_h2d_bytes.clear();
    g_d2h_bytes.clear();
}

void print_timing_summary() {
    #ifndef MEASURE_CUDA_EXECUTION_TIME
    return;
    #endif

    const std::size_t calls = g_kernel_ms.size();
    const double h2d_total_ms = static_cast<double>(sum_ms(g_h2d_ms));
    const double d2h_total_ms = static_cast<double>(sum_ms(g_d2h_ms));
    const double kernel_total_ms = static_cast<double>(sum_ms(g_kernel_ms));
    const double free_total_ms = static_cast<double>(sum_ms(g_free_ms));
    const double h2d_total_bytes = sum_bytes(g_h2d_bytes);
    const double d2h_total_bytes = sum_bytes(g_d2h_bytes);

    std::cout << "--- CUDA timing summary (accumulated) ---\n";
    std::cout << "calls:              " << calls << "\n";
    std::cout << "malloc_total_ms:    " << sum_ms(g_malloc_ms) << "\n";
    std::cout << "h2d_total_ms:       " << h2d_total_ms << " (" << throughput_gbps(h2d_total_bytes, h2d_total_ms) << " GBps)\n";
    std::cout << "kernel_total_ms:    " << kernel_total_ms << "\n";
    std::cout << "d2h_total_ms:       " << d2h_total_ms << " (" << throughput_gbps(d2h_total_bytes, d2h_total_ms) << " GBps)\n";
    std::cout << "free_total_ms:      " << free_total_ms << "\n";
}

}
#endif  // __CUDACC__

#endif // COMMON_FUNCTIONS_CUH
