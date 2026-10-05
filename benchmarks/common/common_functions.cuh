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
// Scalar intrinsic folds evaluate each argument once without widening its type.
namespace generated_kernels::numeric {

template <typename T, typename... Rest>
CUDA_CALLABLE T minimum(T value, Rest... remaining) {
    ((value = value < remaining ? value : remaining), ...);
    return value;
}

template <typename T, typename... Rest>
CUDA_CALLABLE T maximum(T value, Rest... remaining) {
    ((value = value > remaining ? value : remaining), ...);
    return value;
}

} // namespace generated_kernels::numeric

// ── Timing infrastructure — CUDA only ───────────────────────────────────────────
// The functions below use cudaEvent_t and related CUDA runtime APIs.
// They are excluded entirely from plain C++ compilation.
#ifdef __CUDACC__
namespace generated_kernels::timing {

static bool g_profiling_enabled = false;
static std::vector<float> g_malloc_ms;
static std::vector<float> g_h2d_ms;
static std::vector<float> g_kernel_ms;
static std::vector<float> g_d2h_ms;
static std::vector<float> g_free_ms;

static std::vector<double> g_h2d_bytes;
static std::vector<double> g_d2h_bytes;

template <typename Fn> float measure_cuda_event_ms(Fn &&fn) {
    cudaEvent_t start_evt = nullptr;
    cudaEvent_t stop_evt = nullptr;
    float elapsed_ms = 0.0f;

    CUCH(cudaEventCreate(&start_evt));
    CUCH(cudaEventCreate(&stop_evt));

    // Serialize to reduce overlap when profiling individual phases.
    CUCH(cudaDeviceSynchronize());
    CUCH(cudaEventRecord(start_evt, 0));
    fn();
    CUCH(cudaDeviceSynchronize());
    CUCH(cudaEventRecord(stop_evt, 0));
    CUCH(cudaEventSynchronize(stop_evt));

    CUCH(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));
    CUCH(cudaEventDestroy(start_evt));
    CUCH(cudaEventDestroy(stop_evt));
    return elapsed_ms;
}

template <typename Fn> void measure_alloc(Fn &&fn) {
    if (!g_profiling_enabled) {
        fn();
        return;
    }

    float elapsed_ms = measure_cuda_event_ms(std::forward<Fn>(fn));
    g_malloc_ms.push_back(elapsed_ms);
}

template <typename Fn> void measure_h2d(std::size_t bytes, Fn &&fn) {
    if (!g_profiling_enabled) {
        fn();
        return;
    }

    float elapsed_ms = measure_cuda_event_ms(std::forward<Fn>(fn));
    g_h2d_ms.push_back(elapsed_ms);
    g_h2d_bytes.push_back(static_cast<double>(bytes));
}

template <typename Fn> void measure_d2h(std::size_t bytes, Fn &&fn) {
    if (!g_profiling_enabled) {
        fn();
        return;
    }

    float elapsed_ms = measure_cuda_event_ms(std::forward<Fn>(fn));
    g_d2h_ms.push_back(elapsed_ms);
    g_d2h_bytes.push_back(static_cast<double>(bytes));
}

template <typename Fn> void measure_free(Fn &&fn) {
    if (!g_profiling_enabled) {
        fn();
        return;
    }

    float elapsed_ms = measure_cuda_event_ms(std::forward<Fn>(fn));
    g_free_ms.push_back(elapsed_ms);
}

template <typename Fn> void measure_kernel_executions(Fn &&fn) {
    if (!g_profiling_enabled) {
        fn();
        return;
    }

    float elapsed_ms = measure_cuda_event_ms(std::forward<Fn>(fn));
    g_kernel_ms.push_back(elapsed_ms);
}

float sum_ms(const std::vector<float> &values) { return std::accumulate(values.begin(), values.end(), 0.0f); }

double sum_bytes(const std::vector<double> &values) { return std::accumulate(values.begin(), values.end(), 0.0); }

double throughput_bps(double total_bytes, double total_ms) {
    if (total_ms <= 0.0) {
        return 0.0;
    }
    return (total_bytes * 1000.0) / total_ms;
}

double throughput_gbps(double total_bytes, double total_ms) { return throughput_bps(total_bytes, total_ms) / 1.0e9; }

void reset_timing_vectors() {
    CUCH(cudaDeviceSynchronize());
    g_profiling_enabled = true;
    g_malloc_ms.clear();
    g_h2d_ms.clear();
    g_kernel_ms.clear();
    g_d2h_ms.clear();
    g_free_ms.clear();

    g_h2d_bytes.clear();
    g_d2h_bytes.clear();
}

void print_timing_summary() {
    CUCH(cudaDeviceSynchronize());
    if (!g_profiling_enabled)
        return;
    g_profiling_enabled = false;

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
    std::cout << "h2d_total_ms:       " << h2d_total_ms << " (" << throughput_gbps(h2d_total_bytes, h2d_total_ms)
              << " GBps)\n";
    std::cout << "kernel_total_ms:    " << kernel_total_ms << "\n";
    std::cout << "d2h_total_ms:       " << d2h_total_ms << " (" << throughput_gbps(d2h_total_bytes, d2h_total_ms)
              << " GBps)\n";
    std::cout << "free_total_ms:      " << free_total_ms << "\n";
}

} // namespace generated_kernels::timing
#endif // __CUDACC__

// Owned fixed-shape storage shared by ordinary wrappers and resident sessions.
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <unordered_map>

namespace generated_kernels::storage {

inline void trace(const char *operation, std::size_t bytes = 0) {
    static const bool enabled = []() {
        const char *value = std::getenv("FORT_RUNTIME_TRACE");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled)
        return;
    std::cerr << "FORT_RUNTIME " << operation;
    if (std::strcmp(operation, "alloc") == 0 || std::strcmp(operation, "upload") == 0 ||
        std::strcmp(operation, "download") == 0)
        std::cerr << " bytes=" << bytes;
    std::cerr << '\n';
}

[[noreturn]] inline void fail(const char *message) {
    std::cerr << "Compiler workspace error: " << message << std::endl;
    std::abort();
}

inline std::size_t product(std::initializer_list<std::size_t> dimensions) {
    // A zero extent makes an empty array even if preceding extents are large.
    if (std::find(dimensions.begin(), dimensions.end(), 0) != dimensions.end())
        return 0;
    std::size_t result = 1;
    for (std::size_t dimension : dimensions) {
        if (dimension > std::numeric_limits<std::size_t>::max() / result)
            fail("array extent product overflow");
        result *= dimension;
    }
    return result;
}

inline void synchronize() {
#ifdef __CUDACC__
    CUCH(cudaDeviceSynchronize());
#endif
}

#if defined(__CUDACC__) && defined(USE_PINNED_MEMORY)
class HostRegistration {
    void *pointer_;

  public:
    HostRegistration(const void *pointer, std::size_t bytes) : pointer_(const_cast<void *>(pointer)) {
        CUCH(cudaHostRegister(pointer_, bytes, cudaHostRegisterPortable));
    }
    ~HostRegistration() { CUCH(cudaHostUnregister(pointer_)); }
};
#else
class HostRegistration {
  public:
    HostRegistration(const void *, std::size_t) {}
};
#endif

template <typename T> class Buffer {
    std::vector<std::size_t> dimensions_;
    std::size_t count_;
    std::size_t bytes_;
    std::unique_ptr<T[]> host_;
    bool host_current_ = false;
#ifdef __CUDACC__
    T *device_ = nullptr;
    bool device_current_ = false;
#endif
  public:
    explicit Buffer(std::initializer_list<std::size_t> dimensions)
        : dimensions_(dimensions), count_(product(dimensions)), bytes_(0) {
        if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T))
            fail("array byte size overflow");
        bytes_ = count_ * sizeof(T);
#ifdef __CUDACC__
        timing::measure_alloc([&]() {
            if (bytes_) {
                CUCH(cudaMalloc(reinterpret_cast<void **>(&device_), bytes_));
                trace("alloc", bytes_);
            }
        });
#else
        host_.reset(new T[count_]);
#endif
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    ~Buffer() {
#ifdef __CUDACC__
        timing::measure_free([&]() {
            if (device_) {
                CUCH(cudaFree(device_));
                trace("free");
            }
        });
#endif
    }
    std::size_t extent(std::size_t axis) const { return dimensions_.at(axis); }
    void validate(std::initializer_list<std::size_t> dimensions) const {
        if (dimensions.size() != dimensions_.size() ||
            !std::equal(dimensions.begin(), dimensions.end(), dimensions_.begin()))
            fail("array shape does not match workspace");
    }
    T *host_data() {
        if (!host_)
            host_.reset(new T[count_]);
#ifdef __CUDACC__
        if (!host_current_ && device_current_) {
            timing::measure_d2h(bytes_, [&]() {
                if (bytes_) {
                    HostRegistration registration(host_.get(), bytes_);
                    CUCH(cudaMemcpy(host_.get(), device_, bytes_, cudaMemcpyDeviceToHost));
                    trace("download", bytes_);
                }
            });
            host_current_ = true;
        }
#endif
        return host_.get();
    }
    T *device_data() {
#ifdef __CUDACC__
        if (!device_current_ && host_current_) {
            timing::measure_h2d(bytes_, [&]() {
                if (bytes_) {
                    HostRegistration registration(host_.get(), bytes_);
                    CUCH(cudaMemcpy(device_, host_.get(), bytes_, cudaMemcpyHostToDevice));
                    trace("upload", bytes_);
                }
            });
            device_current_ = true;
        }
        return device_;
#else
        return host_data();
#endif
    }
    void host_written() {
        host_current_ = true;
#ifdef __CUDACC__
        device_current_ = false;
#endif
    }
    void device_written() {
#ifdef __CUDACC__
        device_current_ = true;
        host_current_ = false;
#else
        host_current_ = true;
#endif
    }
    void update_device(const T *source, std::initializer_list<std::size_t> dimensions) {
        validate(dimensions);
#ifdef __CUDACC__
        timing::measure_h2d(bytes_, [&]() {
            if (bytes_) {
                HostRegistration registration(source, bytes_);
                CUCH(cudaMemcpy(device_, source, bytes_, cudaMemcpyHostToDevice));
                trace("upload", bytes_);
            }
        });
        device_current_ = true;
        host_current_ = false;
#else
        if (bytes_)
            std::memcpy(host_data(), source, bytes_);
        host_current_ = true;
#endif
    }
    void update_host(T *destination, std::initializer_list<std::size_t> dimensions) {
        validate(dimensions);
#ifdef __CUDACC__
        if (device_current_) {
            timing::measure_d2h(bytes_, [&]() {
                if (bytes_) {
                    HostRegistration registration(destination, bytes_);
                    CUCH(cudaMemcpy(destination, device_, bytes_, cudaMemcpyDeviceToHost));
                    trace("download", bytes_);
                }
            });
        } else
#endif
        {
            if (bytes_)
                std::memcpy(destination, host_data(), bytes_);
        }
    }
};

template <typename State> class Registry {
    std::unordered_map<std::int64_t, std::unique_ptr<State>> entries_;
    std::int64_t next_ = 1;

  public:
    template <typename... Args> std::int64_t create(Args &&...arguments) {
        if (next_ == std::numeric_limits<std::int64_t>::max())
            fail("workspace token limit reached");
        const std::int64_t token = next_++;
        entries_.emplace(token, std::make_unique<State>(std::forward<Args>(arguments)...));
        return token;
    }
    State &get(std::int64_t token) {
        auto found = entries_.find(token);
        if (found == entries_.end())
            fail("invalid or stale workspace token");
        return *found->second;
    }
    void destroy(std::int64_t token) {
        if (!token)
            return;
        get(token);
        synchronize();
        entries_.erase(token);
    }
};
} // namespace generated_kernels::storage


#endif // COMMON_FUNCTIONS_CUH
