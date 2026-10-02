// CUDA profiling is opt-in. Public calls serialize only while it is enabled.
#ifdef __CUDACC__
#include <atomic>
#include <mutex>

namespace generated_kernels::timing {

struct ProfilingState {
    std::atomic<bool> enabled{false};
    std::recursive_mutex mutex;
    std::size_t calls = 0;
    std::vector<float> malloc_ms, h2d_ms, kernel_ms, d2h_ms, free_ms;
    std::vector<double> h2d_bytes, d2h_bytes;
};

// Inline storage shares both the lock and counters across generated entries.
inline ProfilingState profiling;

class ProfiledCallGuard {
    std::unique_lock<std::recursive_mutex> lock_;

  public:
    ProfiledCallGuard() : lock_(profiling.mutex, std::defer_lock) {
        if (profiling.enabled.load(std::memory_order_acquire))
            lock_.lock();
    }
    bool enabled() const {
        return lock_.owns_lock() && profiling.enabled.load(std::memory_order_relaxed);
    }
};

inline void record_run() {
    ProfiledCallGuard guard;
    if (guard.enabled())
        ++profiling.calls;
}

template <typename Fn> float measure_cuda_event_ms(Fn &&fn) {
    std::lock_guard<std::recursive_mutex> lock(profiling.mutex);
    cudaEvent_t start_evt = nullptr;
    cudaEvent_t stop_evt = nullptr;
    float elapsed_ms = 0.0f;

    CUCH(cudaEventCreate(&start_evt));
    CUCH(cudaEventCreate(&stop_evt));

    // This barrier isolates a measured phase; ordinary unprofiled calls skip it.
    CUCH(cudaDeviceSynchronize());
    CUCH(cudaEventRecord(start_evt, 0));
    fn();
    CUCH(cudaEventRecord(stop_evt, 0));
    CUCH(cudaEventSynchronize(stop_evt));

    CUCH(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    CUCH(cudaEventDestroy(start_evt));
    CUCH(cudaEventDestroy(stop_evt));
    return elapsed_ms;
}

template <typename Fn> void measure_phase(std::vector<float> &samples, Fn &&fn) {
    ProfiledCallGuard guard;
    if (!guard.enabled()) {
        std::forward<Fn>(fn)();
        return;
    }
    samples.push_back(measure_cuda_event_ms(std::forward<Fn>(fn)));
}

template <typename Fn> void measure_alloc(Fn &&fn) {
    measure_phase(profiling.malloc_ms, std::forward<Fn>(fn));
}

template <typename Fn> void measure_h2d(std::size_t bytes, Fn &&fn) {
    ProfiledCallGuard guard;
    if (!guard.enabled()) {
        std::forward<Fn>(fn)();
        return;
    }
    profiling.h2d_ms.push_back(measure_cuda_event_ms(std::forward<Fn>(fn)));
    profiling.h2d_bytes.push_back(static_cast<double>(bytes));
}

template <typename Fn> void measure_d2h(std::size_t bytes, Fn &&fn) {
    ProfiledCallGuard guard;
    if (!guard.enabled()) {
        std::forward<Fn>(fn)();
        return;
    }
    profiling.d2h_ms.push_back(measure_cuda_event_ms(std::forward<Fn>(fn)));
    profiling.d2h_bytes.push_back(static_cast<double>(bytes));
}

template <typename Fn> void measure_free(Fn &&fn) {
    measure_phase(profiling.free_ms, std::forward<Fn>(fn));
}

template <typename Fn> void measure_kernel_executions(Fn &&fn) {
    measure_phase(profiling.kernel_ms, std::forward<Fn>(fn));
}

inline float sum_ms(const std::vector<float> &values) { return std::accumulate(values.begin(), values.end(), 0.0f); }

inline double sum_bytes(const std::vector<double> &values) { return std::accumulate(values.begin(), values.end(), 0.0); }

inline double throughput_bps(double total_bytes, double total_ms) {
    if (total_ms <= 0.0) {
        return 0.0;
    }
    return (total_bytes * 1000.0) / total_ms;
}

inline double throughput_gbps(double total_bytes, double total_ms) { return throughput_bps(total_bytes, total_ms) / 1.0e9; }

// Hooks delimit a quiescent batch: callers must not race start/finish with work.
inline void reset_timing_vectors() {
    std::lock_guard<std::recursive_mutex> lock(profiling.mutex);
    CUCH(cudaDeviceSynchronize());
    profiling.calls = 0;
    profiling.malloc_ms.clear();
    profiling.h2d_ms.clear();
    profiling.kernel_ms.clear();
    profiling.d2h_ms.clear();
    profiling.free_ms.clear();
    profiling.h2d_bytes.clear();
    profiling.d2h_bytes.clear();
    profiling.enabled.store(true, std::memory_order_release);
}

inline void print_timing_summary() {
    std::lock_guard<std::recursive_mutex> lock(profiling.mutex);
    if (!profiling.enabled.load(std::memory_order_relaxed))
        return;
    CUCH(cudaDeviceSynchronize());
    profiling.enabled.store(false, std::memory_order_release);

    const double h2d_total_ms = static_cast<double>(sum_ms(profiling.h2d_ms));
    const double d2h_total_ms = static_cast<double>(sum_ms(profiling.d2h_ms));
    const double kernel_total_ms = static_cast<double>(sum_ms(profiling.kernel_ms));
    const double free_total_ms = static_cast<double>(sum_ms(profiling.free_ms));
    const double h2d_total_bytes = sum_bytes(profiling.h2d_bytes);
    const double d2h_total_bytes = sum_bytes(profiling.d2h_bytes);

    std::cout << "--- CUDA timing summary (accumulated) ---\n";
    std::cout << "calls:              " << profiling.calls << "\n";
    std::cout << "kernel_launches:    " << profiling.kernel_ms.size() << "\n";
    std::cout << "malloc_total_ms:    " << sum_ms(profiling.malloc_ms) << "\n";
    std::cout << "h2d_total_ms:       " << h2d_total_ms << " (" << throughput_gbps(h2d_total_bytes, h2d_total_ms)
              << " GBps)\n";
    std::cout << "kernel_total_ms:    " << kernel_total_ms << "\n";
    std::cout << "d2h_total_ms:       " << d2h_total_ms << " (" << throughput_gbps(d2h_total_bytes, d2h_total_ms)
              << " GBps)\n";
    std::cout << "free_total_ms:      " << free_total_ms << "\n";
}

} // namespace generated_kernels::timing
#endif // __CUDACC__
