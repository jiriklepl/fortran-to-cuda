// Bounded, synchronous-on-return CPU/CUDA slab execution. Explicit sessions do
// not use this runtime. Two event-protected slots own all staging allocations.
#pragma once

#ifdef __CUDACC__
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include <omp.h>
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define FORT_HYBRID_NVTX 1
#endif

namespace generated_kernels::hybrid {

constexpr std::size_t pinned_limit = 64ULL * 1024 * 1024;

struct Budget {
    std::mutex mutex;
    std::condition_variable released;
    std::size_t reserved = 0;
};
inline Budget &budget_state() {
    static Budget value;
    return value;
}
class Reservation {
    std::size_t bytes_;
  public:
    explicit Reservation(std::size_t bytes) : bytes_(bytes) {
        if (bytes > pinned_limit) storage::fail("hybrid pinned budget exceeded");
        auto &state = budget_state();
        std::unique_lock<std::mutex> lock(state.mutex);
        state.released.wait(lock, [&]() { return state.reserved <= pinned_limit - bytes; });
        state.reserved += bytes;
    }
    ~Reservation() {
        auto &state = budget_state();
        { std::lock_guard<std::mutex> lock(state.mutex); state.reserved -= bytes_; }
        state.released.notify_all();
    }
};

inline bool tracing() {
    const char *value = std::getenv("FORT_RUNTIME_TRACE");
    return value && std::strcmp(value, "1") == 0;
}

struct Range {
    explicit Range(const char *name) {
#ifdef FORT_HYBRID_NVTX
        if (tracing()) nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~Range() {
#ifdef FORT_HYBRID_NVTX
        if (tracing()) nvtxRangePop();
#endif
    }
};
using NvtxRange = Range;

// The common expression emitter still passes logical SIZE values to F_IDX.
// Decode that logical offset separately from the compact storage's pitch.
template <typename T, unsigned Rank> struct View {
    T *data;
    std::size_t logical[Rank], origin[Rank], pitch[Rank];
    __device__ T &operator[](std::size_t index) const {
        std::size_t physical = 0;
#pragma unroll
        for (unsigned dimension = 0; dimension < Rank; ++dimension) {
            const std::size_t coordinate = index % logical[dimension];
            index /= logical[dimension];
            physical += (coordinate - origin[dimension]) * pitch[dimension];
        }
        return data[physical];
    }
};

struct Array {
    void *host;
    std::size_t item_bytes;
    std::vector<std::size_t> dimensions;
    unsigned axis;
    int coefficient;
    long long lower_offset, upper_offset;
    bool written;
};

struct Layout {
    std::vector<std::size_t> logical, origin, extent, pitch;
    std::size_t offset = 0, bytes = 0;
    std::size_t blocks = 0, block_bytes = 0, host_stride = 0, host_offset = 0;
};

inline bool multiply(std::size_t &value, std::size_t factor) {
    if (factor && value > std::numeric_limits<std::size_t>::max() / factor) return false;
    value *= factor;
    return true;
}

inline bool layouts(const std::vector<Array> &arrays, std::size_t begin, std::size_t count,
                    int lower, int stride, std::vector<Layout> &result, std::size_t &bytes) {
    result.clear();
    bytes = 0;
    if (!count) return true;
    const long long first = static_cast<long long>(lower) + static_cast<long long>(begin) * stride;
    const long long last = first + static_cast<long long>(count - 1) * stride;
    for (const auto &array : arrays) {
        if (array.axis >= array.dimensions.size()) return false;
        std::size_t logical_bytes = array.item_bytes;
        for (const auto dimension : array.dimensions)
            if (!dimension || !multiply(logical_bytes, dimension)) return false;
        const long double a = static_cast<long double>(array.coefficient) * first;
        const long double b = static_cast<long double>(array.coefficient) * last;
        const long double first_physical = std::min(a, b) + array.lower_offset - 1;
        const long double last_physical = std::max(a, b) + array.upper_offset - 1;
        if (first_physical < 0 || last_physical < first_physical ||
            last_physical > std::numeric_limits<long long>::max() ||
            last_physical >= static_cast<long double>(array.dimensions[array.axis]))
            return false;
        const auto start = static_cast<long long>(first_physical), stop = static_cast<long long>(last_physical);
        Layout layout;
        layout.logical = layout.extent = array.dimensions;
        layout.origin.assign(array.dimensions.size(), 0);
        layout.origin[array.axis] = static_cast<std::size_t>(start);
        layout.extent[array.axis] = static_cast<std::size_t>(stop - start + 1);
        std::size_t product = 1;
        for (const auto extent : layout.extent) {
            layout.pitch.push_back(product);
            if (!multiply(product, extent)) return false;
        }
        if (!multiply(product, array.item_bytes)) return false;
        layout.bytes = product;
        if (bytes > pinned_limit / 2 - std::min<std::size_t>(pinned_limit / 2, 63)) return false;
        layout.offset = (bytes + 63) & ~std::size_t(63);
        if (layout.bytes > pinned_limit / 2 - layout.offset) return false;
        bytes = layout.offset + layout.bytes;
        std::size_t prefix = array.item_bytes;
        for (unsigned axis = 0; axis < array.axis; ++axis)
            if (!multiply(prefix, array.dimensions[axis])) return false;
        layout.block_bytes = prefix;
        if (!multiply(layout.block_bytes, layout.extent[array.axis])) return false;
        layout.host_stride = prefix;
        if (!multiply(layout.host_stride, array.dimensions[array.axis])) return false;
        layout.host_offset = prefix;
        if (!multiply(layout.host_offset, layout.origin[array.axis])) return false;
        layout.blocks = 1;
        for (unsigned axis = array.axis + 1; axis < array.dimensions.size(); ++axis)
            if (!multiply(layout.blocks, array.dimensions[axis])) return false;
        result.push_back(std::move(layout));
    }
    return true;
}

inline void copy_blocks(void *packed, const Array &array, const Layout &layout, bool unpack) {
    auto *compact = static_cast<unsigned char *>(packed);
    auto *host = static_cast<unsigned char *>(array.host) + layout.host_offset;
    for (std::size_t block = 0; block < layout.blocks; ++block) {
        void *a = compact + block * layout.block_bytes;
        void *b = host + block * layout.host_stride;
        if (unpack) std::memcpy(b, a, layout.block_bytes);
        else std::memcpy(a, b, layout.block_bytes);
    }
}

struct Choice {
    std::size_t gpu_iterations = 0, chunk_iterations = 1, slot_bytes = 0;
    double estimated_seconds = 0;
    int device = -1, threads = 0;
    std::size_t total_iterations = 0;
};

// Finite candidates from offline rates only: no wall-clock feedback or tuning.
template <class Profile>
Choice select(const std::vector<Array> &arrays, std::size_t total, int lower, int stride,
              double flops, double traffic, unsigned kernels, const Profile &profile,
              bool compatible, int threads, bool force_chunked, unsigned copy_engines = 1) {
    Choice best;
    if (!total || arrays.empty() || (!compatible && !force_chunked)) return best;
    const double infinity = std::numeric_limits<double>::infinity();
    const double cpu = compatible ? std::max(flops / profile.cpu_flops, traffic / profile.cpu_bandwidth) : infinity;
    best.estimated_seconds = cpu;
    std::vector<Layout> layout;
    std::size_t one = 0, two = 0;
    if (!layouts(arrays, 0, 1, lower, stride, layout, one)) return best;
    if (total > 1 && !layouts(arrays, 0, 2, lower, stride, layout, two)) two = one * 2;
    const std::size_t per_iteration = total > 1 && two > one ? two - one : one;
    if (!per_iteration) return best;
    const double fractions[] = {0.25, 0.5, 0.75, 1.0};
    const std::size_t payloads[] = {256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024};
    double winning = infinity;
    for (const double fraction : fractions) {
        if ((force_chunked || threads <= 1) && fraction != 1.0) continue;
        const auto gpu = std::min(total, std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(total * fraction))));
        for (const auto payload : payloads) {
            if (!compatible && payload != 1024 * 1024) continue;
            const std::size_t chunk = std::min(gpu, std::max<std::size_t>(1, payload / per_iteration));
            std::size_t bytes = 0;
            if (!layouts(arrays, gpu - 1, 1, lower, stride, layout, bytes)) continue;
            if (!layouts(arrays, 0, chunk, lower, stride, layout, bytes) || !bytes) continue;
            double upload = 0, download = 0;
            unsigned downloaded_arrays = 0;
            for (std::size_t index = 0; index < arrays.size(); ++index) {
                upload += layout[index].bytes;
                if (arrays[index].written) { download += layout[index].bytes; ++downloaded_arrays; }
            }
            const double chunks = static_cast<double>((gpu - 1) / chunk + 1);
            const double portion = static_cast<double>(gpu) / total;
            double estimate = 0;
            if (compatible) {
                const double h2d = arrays.size() * profile.pinned_h2d_latency + upload / profile.pinned_h2d_bandwidth;
                const double d2h = downloaded_arrays * profile.pinned_d2h_latency + download / profile.pinned_d2h_bandwidth;
                const double device = kernels * profile.launch_seconds +
                    std::max(flops * portion / chunks / profile.gpu_flops,
                             traffic * portion / chunks / profile.gpu_bandwidth);
                const double pump = upload / profile.pack_bandwidth + download / profile.unpack_bandwidth +
                    profile.pump_seconds;
                const double copies = copy_engines > 1 ? std::max(h2d, d2h) : h2d + d2h;
                const double period = std::max({copies, device, pump, (h2d + d2h + device) / 2});
                const double gpu_seconds = h2d + device + d2h + pump + (chunks - 1) * period;
                const double host = gpu == total ? 0 : std::max(
                    flops * (1 - portion) / profile.cpu_worker_flops,
                    traffic * (1 - portion) / profile.cpu_worker_bandwidth);
                estimate = std::max(host, gpu_seconds);
            }
            if ((!compatible || std::isfinite(estimate)) && estimate < winning) {
                winning = estimate;
                best = {gpu, chunk, bytes, estimate};
            }
        }
    }
    if (!force_chunked && !(winning <= 0.8 * cpu)) return {};
    best.threads = threads;
    return best;
}

class Slot {
  public:
    unsigned char *host = nullptr, *device = nullptr;
    cudaStream_t stream{};
    cudaEvent_t complete{};
    bool pending = false;
    std::vector<Layout> layout;
    Slot() = default;
    bool allocate(std::size_t bytes) {
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&complete, cudaEventDisableTiming) != cudaSuccess ||
            cudaMallocHost(reinterpret_cast<void **>(&host), bytes) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&device), bytes) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        storage::trace("alloc", bytes);
        return true;
    }
    Slot(const Slot &) = delete;
    ~Slot() {
        if (pending) CUCH(cudaEventSynchronize(complete));
        if (device) { CUCH(cudaFree(device)); storage::trace("free"); }
        if (host) CUCH(cudaFreeHost(host));
        if (complete) CUCH(cudaEventDestroy(complete));
        if (stream) CUCH(cudaStreamDestroy(stream));
    }
    void finish(const std::vector<Array> &arrays) {
        if (!pending) return;
        CUCH(cudaEventSynchronize(complete));
        Range range("FORT hybrid unpack");
        for (std::size_t index = 0; index < arrays.size(); ++index)
            if (arrays[index].written) copy_blocks(host + layout[index].offset, arrays[index], layout[index], true);
        pending = false;
    }
    template <typename T, unsigned Rank> View<T, Rank> view(std::size_t index) const {
        View<T, Rank> result{};
        result.data = reinterpret_cast<T *>(device + layout[index].offset);
        for (unsigned axis = 0; axis < Rank; ++axis) {
            result.logical[axis] = layout[index].logical[axis];
            result.origin[axis] = layout[index].origin[axis];
            result.pitch[axis] = layout[index].pitch[axis];
        }
        return result;
    }
};

struct Slots {
    Reservation reservation;
    Slot slots[2];
    bool ready;
    explicit Slots(std::size_t bytes) : reservation(2 * bytes), ready(false) {
        ready = slots[0].allocate(bytes) && slots[1].allocate(bytes);
    }
};

template <class CPU, class GPU> struct Work {
    std::vector<Array> arrays;
    std::size_t total;
    int lower, stride;
    Choice choice;
    CPU cpu;
    GPU gpu;
    const char *entry, *mode;
    std::unique_ptr<Slots> resources;
    int previous_device = -1;
    void prepare(int threads) {
        if (choice.threads && choice.threads != threads) choice = {};
        if (threads == 1 && choice.gpu_iterations < total) choice = {};
        if (choice.gpu_iterations) {
            if (!choice.slot_bytes || choice.slot_bytes > pinned_limit / 2) choice = {};
            else if (cudaGetDevice(&previous_device) != cudaSuccess ||
                     (choice.device >= 0 && cudaSetDevice(choice.device) != cudaSuccess)) {
                cudaGetLastError();
                choice = {};
            } else {
                resources = std::make_unique<Slots>(choice.slot_bytes);
                if (!resources->ready) { resources.reset(); choice = {}; }
            }
        }
        // Allocation failure occurs before any CPU/GPU numerical work. It can
        // therefore fall back once, without replaying partially written arrays.
        offload::decision_trace(entry, choice.gpu_iterations ? mode : "native",
                                choice.gpu_iterations, total - choice.gpu_iterations, "slabs");
    }
    void finish() {
        resources.reset();
        if (previous_device >= 0) CUCH(cudaSetDevice(previous_device));
    }
    void pump() {
        Range range("FORT hybrid GPU pump");
        auto &slots = resources->slots;
        std::size_t slot_index = 0;
        for (std::size_t begin = 0; begin < choice.gpu_iterations; ++slot_index) {
            auto &slot = slots[slot_index % 2];
            slot.finish(arrays); // A slot cannot be repacked while copies/kernels still use it.
            const std::size_t count = std::min(choice.chunk_iterations, choice.gpu_iterations - begin);
            std::size_t bytes = 0;
            if (!layouts(arrays, begin, count, lower, stride, slot.layout, bytes) || bytes > choice.slot_bytes)
                storage::fail("hybrid slab no longer matches checked staging capacity");
            {
                Range pack("FORT hybrid pack");
                for (std::size_t index = 0; index < arrays.size(); ++index) {
                    const auto &layout = slot.layout[index];
                    copy_blocks(slot.host + layout.offset, arrays[index], layout, false);
                    CUCH(cudaMemcpyAsync(slot.device + layout.offset, slot.host + layout.offset, layout.bytes,
                                         cudaMemcpyHostToDevice, slot.stream));
                    storage::trace("upload", layout.bytes);
                }
            }
            gpu(slot, begin, count);
            for (std::size_t index = 0; index < arrays.size(); ++index) {
                if (!arrays[index].written) continue;
                const auto &layout = slot.layout[index];
                CUCH(cudaMemcpyAsync(slot.host + layout.offset, slot.device + layout.offset, layout.bytes,
                                     cudaMemcpyDeviceToHost, slot.stream));
                storage::trace("download", layout.bytes);
            }
            CUCH(cudaEventRecord(slot.complete, slot.stream));
            slot.pending = true;
            begin += count;
        }
        for (auto &slot : slots) slot.finish(arrays);
    }
    void worker(int thread, int threads) {
        if (choice.gpu_iterations && thread == 0) {
            pump();
            return;
        }
        const int first = choice.gpu_iterations ? 1 : 0;
        const int workers = threads - first;
        if (workers <= 0) return;
        const std::size_t count = total - choice.gpu_iterations;
        const std::size_t index = static_cast<std::size_t>(thread - first);
        const auto boundary = [&](std::size_t part) {
            return choice.gpu_iterations + (count / workers) * part + (count % workers) * part / workers;
        };
        const auto begin = boundary(index), end = boundary(index + 1);
        if (end > begin) {
            Range range("FORT hybrid CPU window");
            cpu(begin, end - begin);
        }
    }
};

template <class CPU, class GPU>
void execute(std::vector<Array> arrays, std::size_t total, int lower, int stride,
             Choice choice, int budget, bool collective, CPU cpu, GPU gpu,
             const char *entry = "unknown", const char *mode = "hybrid") {
    if (!total) return;
    using State = Work<CPU, GPU>;
    if (collective && omp_get_level() != 0) {
        State *state = nullptr;
#pragma omp single copyprivate(state)
        { state = new State{arrays, total, lower, stride, choice, cpu, gpu, entry, mode, {}}; }
        if (omp_get_thread_num() == 0) state->prepare(omp_get_num_threads());
#pragma omp barrier
        // Collective callers use the existing team; only its thread zero pumps.
        state->worker(omp_get_thread_num(), omp_get_num_threads());
#pragma omp barrier
        if (omp_get_thread_num() == 0) state->finish();
#pragma omp barrier
#pragma omp single
        { delete state; }
    } else if (omp_get_level() != 0) {
        // An unrecognized serial call within an existing team cannot recruit it.
        Range range("FORT hybrid CPU window");
        offload::decision_trace(entry, "native", 0, total, "slabs");
        cpu(0, total);
    } else {
        State state{std::move(arrays), total, lower, stride, choice, cpu, gpu, entry, mode, {}};
#pragma omp parallel num_threads(budget)
        {
            if (omp_get_thread_num() == 0) state.prepare(omp_get_num_threads());
#pragma omp barrier
            state.worker(omp_get_thread_num(), omp_get_num_threads());
#pragma omp barrier
            if (omp_get_thread_num() == 0) state.finish();
        }
    }
}
} // namespace generated_kernels::hybrid
#ifdef FORT_HYBRID_NVTX
#undef FORT_HYBRID_NVTX
#endif
#endif // __CUDACC__
