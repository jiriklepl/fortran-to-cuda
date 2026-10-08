// Ordinary-call section transfers and bounded, calibrated CPU/GPU selection.
#ifndef FORT_RUNTIME_OFFLOAD_HPP
#define FORT_RUNTIME_OFFLOAD_HPP
#include "section_copy.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#if defined(__CUDACC__) && __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define FORT_OFFLOAD_NVTX 1
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace generated_kernels::offload {
struct Profile {
    bool valid = false;
    int threads = 1, precision = 64;
    std::string uuid, cc, cpu_name, host_compiler, cuda_compiler;
    int cuda_runtime = 0, cuda_driver = 0;
    double launch_seconds = 0, h2d_latency = 0, h2d_bandwidth = 1;
    double d2h_latency = 0, d2h_bandwidth = 1;
    double pinned_h2d_latency = 0, pinned_h2d_bandwidth = 1;
    double pinned_d2h_latency = 0, pinned_d2h_bandwidth = 1;
    double pack_bandwidth = 1, unpack_bandwidth = 1;
    double cpu_flops = 1, cpu_bandwidth = 1, cpu_worker_flops = 0, cpu_worker_bandwidth = 0;
    double gpu_flops = 1, gpu_bandwidth = 1, pump_seconds = 0;
};
inline int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}
inline int team_size() {
#ifdef _OPENMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}
inline bool in_parallel() {
#ifdef _OPENMP
    return omp_get_level() != 0;
#else
    return false;
#endif
}
inline bool context_valid(int threads, bool collective) {
    return collective ? in_parallel() && team_size() == threads : !in_parallel();
}

inline bool compatible(const Profile &p, int threads, int precision) {
    if (!p.valid || p.threads != threads || p.precision != precision) return false;
#ifdef __CUDACC__
    static const std::string cpu_name = []() {
        std::ifstream input("/proc/cpuinfo");
        std::string line;
        while (std::getline(input, line)) {
            if (line.rfind("model name", 0) != 0) continue;
            const auto colon = line.find(':');
            if (colon == std::string::npos) continue;
            const auto first = line.find_first_not_of(" \t", colon + 1);
            const auto last = line.find_last_not_of(" \t\r\n");
            return first == std::string::npos ? std::string{} : line.substr(first, last-first+1);
        }
        return std::string{};
    }();
    const std::string host = std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
    const std::string cuda = std::to_string(__CUDACC_VER_MAJOR__) + "." + std::to_string(__CUDACC_VER_MINOR__) + "." + std::to_string(__CUDACC_VER_BUILD__);
    static const auto versions = []() {
        std::pair<int, int> value{0, 0};
        if (cudaRuntimeGetVersion(&value.first) != cudaSuccess || cudaDriverGetVersion(&value.second) != cudaSuccess)
            return std::pair<int, int>{0, 0};
        return value;
    }();
    if (cpu_name.empty() || cpu_name != p.cpu_name || host != p.host_compiler || cuda != p.cuda_compiler ||
        !versions.first || versions.first != p.cuda_runtime || versions.second != p.cuda_driver) return false;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    struct Identity { int device; std::string uuid, cc; };
    static thread_local Identity cached{-1, {}, {}};
    if (cached.device != device) {
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return false;
        static const char hex[] = "0123456789abcdef";
        std::string uuid;
        for (unsigned char byte : properties.uuid.bytes) {
            uuid += hex[byte >> 4]; uuid += hex[byte & 15];
        }
        cached = {device, uuid, std::to_string(properties.major) + "." + std::to_string(properties.minor)};
    }
    std::string uuid;
    const std::size_t start = p.uuid.rfind("GPU-", 0) == 0 ? 4 : 0;
    for (std::size_t i = start; i < p.uuid.size(); ++i) {
        char ch = p.uuid[i];
        if (ch >= 'A' && ch <= 'F') ch += 'a' - 'A';
        if (ch != '-') uuid += ch;
    }
    return uuid == cached.uuid && p.cc == cached.cc;
#else
    return false;
#endif
}

inline bool mul(std::size_t a, std::size_t b, std::size_t &result) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) return false;
    result = a * b; return true;
}
inline bool add(std::size_t a, std::size_t b, std::size_t &result) {
    if (a > std::numeric_limits<std::size_t>::max() - b) return false;
    result = a + b; return true;
}

// Total default-INTEGER arithmetic for the side-effect-free decision slice.
inline int query_integer(long long value, bool &valid) {
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        valid = false; return 0;
    }
    return static_cast<int>(value);
}
inline int query_size(std::size_t value, bool &valid) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) { valid=false; return 0; }
    return static_cast<int>(value);
}
inline int query_divide(int a, int b, bool &valid) {
    if (!b) { valid=false; return 0; }
    return query_integer(static_cast<long long>(a)/b, valid);
}
inline int query_load(const int *array, std::initializer_list<std::size_t> dimensions,
                      std::initializer_list<int> indices, bool &valid) {
    if (!valid) return 0;
    if (!array || dimensions.size()!=indices.size()) { valid=false; return 0; }
    std::size_t offset=0, pitch=1;
    auto index=indices.begin();
    for (const auto extent:dimensions) {
        const int coordinate=*index++;
        std::size_t term=0;
        if (coordinate<1 || static_cast<std::size_t>(coordinate)>extent ||
            !mul(pitch,static_cast<std::size_t>(coordinate-1),term) || !add(offset,term,offset) ||
            !mul(pitch,extent,pitch)) { valid=false; return 0; }
    }
    return array[offset];
}
inline long long index(long double value, bool &valid) {
    if (!std::isfinite(value) || value < static_cast<long double>(std::numeric_limits<long long>::min()) ||
        value > static_cast<long double>(std::numeric_limits<long long>::max()) || std::trunc(value) != value) {
        valid = false; return 0;
    }
    return static_cast<long long>(value);
}
struct Box {
    std::vector<std::size_t> lower, upper; // zero-based inclusive physical coordinates
    bool operator==(const Box &other) const { return lower == other.lower && upper == other.upper; }
};
struct Array {
    void *host = nullptr;
    std::size_t element_bytes = 0, bytes = 0;
    std::vector<std::size_t> dimensions;
};
struct Footprint { std::vector<Box> upload, download; };
struct Unit {
    std::vector<Footprint> arrays;
    std::size_t iterations = 0;
    double flops = 0, memory_bytes = 0;
    std::size_t source_region = 0;
};
struct Choice { std::size_t begin, end; bool gpu; };
struct Plan {
    bool valid = true;
    std::vector<Choice> choices;
    bool has_gpu() const {
        for (const auto &choice : choices) if (choice.gpu) return true;
        return false;
    }
};
struct Data {
    bool valid = true;
    int device = -1;
    std::vector<Array> arrays;
    std::vector<Unit> units;
    Plan plan;
    // Structured preflight proves the value ABI safe on its selected path.
    bool guarded_inputs_checked = false;
};
inline Plan native_plan(const Data &data) {
    Plan result;
    result.valid = data.valid;
    for (std::size_t i=0; i<data.units.size(); ++i) result.choices.push_back({i, i+1, false});
    return result;
}
inline bool contains(const Box &a, const Box &b) {
    for (std::size_t k = 0; k < a.lower.size(); ++k)
        if (a.lower[k] > b.lower[k] || a.upper[k] < b.upper[k]) return false;
    return true;
}
inline void append_box(std::vector<Box> &boxes, const Box &box) {
    for (const auto &old : boxes) if (contains(old, box)) return;
    boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [&](const Box &old) { return contains(box, old); }), boxes.end());
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        auto merged = box;
        int changed = -1;
        bool possible = true;
        for (std::size_t axis = 0; axis < box.lower.size(); ++axis) {
            if (box.lower[axis] == boxes[i].lower[axis] && box.upper[axis] == boxes[i].upper[axis]) continue;
            if (changed != -1 ||
                (box.lower[axis] > boxes[i].upper[axis] && box.lower[axis] - boxes[i].upper[axis] > 1) ||
                (boxes[i].lower[axis] > box.upper[axis] && boxes[i].lower[axis] - box.upper[axis] > 1)) {
                possible = false; break;
            }
            changed = static_cast<int>(axis);
            merged.lower[axis] = std::min(box.lower[axis], boxes[i].lower[axis]);
            merged.upper[axis] = std::max(box.upper[axis], boxes[i].upper[axis]);
        }
        if (possible) {
            boxes.erase(boxes.begin() + i);
            append_box(boxes, merged);
            return;
        }
    }
    boxes.push_back(box);
}
inline std::size_t box_bytes(const Array &a, const Box &b, bool &valid) {
    std::size_t result = a.element_bytes;
    if (b.lower.size() != a.dimensions.size() || b.upper.size() != a.dimensions.size()) {
        valid = false; return 0;
    }
    for (std::size_t axis = 0; axis < b.lower.size(); ++axis) {
        if (b.lower[axis] > b.upper[axis] || b.upper[axis] >= a.dimensions[axis] ||
            !mul(result, b.upper[axis] - b.lower[axis] + 1, result)) { valid = false; return 0; }
    }
    return result;
}
inline std::size_t copy_operations(const Array &a, const Box &b, bool &valid) {
    box_bytes(a, b, valid);
    if (!valid) return 0;
    auto upper = b.upper;
    for (auto &value : upper) ++value; // box_bytes checked value < extent.
    const fort_physical::CopyPlan plan(a.element_bytes, a.dimensions, b.lower, upper);
    valid = valid && plan.valid;
    return valid ? plan.copies : 0;
}
// Exact unions are optional: bound planning work and retain the original
// rectangles if fragmentation is not worthwhile or arithmetic is uncertain.
inline constexpr std::size_t section_union_limit = 32;
inline bool overlaps(const Box &a, const Box &b) {
    for (std::size_t k = 0; k < a.lower.size(); ++k)
        if (a.lower[k] > b.upper[k] || b.lower[k] > a.upper[k]) return false;
    return true;
}
inline bool disjoint_union(const std::vector<Box> &input, std::vector<Box> &output) {
    if (input.size() > section_union_limit) return false;
    const auto rank = input.empty() ? 0 : input.front().lower.size();
    for (const auto &box : input) {
        if (!rank || box.lower.size() != rank || box.upper.size() != rank) return false;
        for (std::size_t axis = 0; axis < rank; ++axis)
            if (box.lower[axis] > box.upper[axis]) return false;
    }
    std::vector<Box> result;
    std::size_t work = 0;
    for (const auto &box : input) {
        std::vector<Box> pending{box};
        for (const auto &old : result) {
            std::vector<Box> next;
            for (const auto &piece : pending) {
                if (++work > section_union_limit * section_union_limit) return false;
                if (!overlaps(piece, old)) next.push_back(piece);
                else {
                    auto middle = piece;
                    // Peel slow axes first to preserve wide contiguous rows.
                    for (std::size_t axis = rank; axis-- > 0;) {
                        const auto lo = std::max(piece.lower[axis], old.lower[axis]);
                        const auto hi = std::min(piece.upper[axis], old.upper[axis]);
                        if (middle.lower[axis] < lo) {
                            auto slab = middle; slab.upper[axis] = lo - 1;
                            next.push_back(std::move(slab)); middle.lower[axis] = lo;
                        }
                        if (middle.upper[axis] > hi) {
                            auto slab = middle; slab.lower[axis] = hi + 1;
                            next.push_back(std::move(slab)); middle.upper[axis] = hi;
                        }
                        if (next.size() > section_union_limit) return false;
                    }
                }
                if (next.size() > section_union_limit) return false;
            }
            pending = std::move(next);
            if (pending.empty()) break;
        }
        for (const auto &piece : pending) {
            append_box(result, piece);
            if (result.size() > section_union_limit) return false;
        }
    }
    output = std::move(result);
    return true;
}
inline bool transfer_metrics(const Array &array, const std::vector<Box> &boxes,
                             std::size_t &bytes, std::size_t &copies) {
    bool valid = true;
    std::size_t total_bytes = 0, total_copies = 0;
    for (const auto &box : boxes) {
        const auto size = box_bytes(array, box, valid);
        const auto count = copy_operations(array, box, valid);
        if (!valid || !add(total_bytes, size, total_bytes) ||
            !add(total_copies, count, total_copies)) return false;
    }
    bytes = total_bytes; copies = total_copies;
    return true;
}
inline bool deduplicate_boxes(const Array &array, std::vector<Box> &boxes,
                              double latency = -1, double bandwidth = 0) {
    if (boxes.size() < 2 || boxes.size() > section_union_limit) return false;
    std::size_t old_bytes = 0, old_copies = 0, new_bytes = 0, new_copies = 0;
    if (!transfer_metrics(array, boxes, old_bytes, old_copies)) return false;
    bool overlap = false;
    for (std::size_t i = 0; i < boxes.size() && !overlap; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (overlaps(boxes[i], boxes[j])) { overlap = true; break; }
    if (!overlap) return false;
    std::vector<Box> candidate;
    if (!disjoint_union(boxes, candidate) ||
        !transfer_metrics(array, candidate, new_bytes, new_copies) ||
        new_bytes >= old_bytes) return false;
    if (new_copies > old_copies &&
        (!(latency >= 0) || !(bandwidth > 0) || !std::isfinite(latency) || !std::isfinite(bandwidth) ||
         static_cast<double>(old_bytes - new_bytes) / bandwidth <=
             static_cast<double>(new_copies - old_copies) * latency)) return false;
    boxes = std::move(candidate);
    return true;
}
inline std::vector<Footprint> interval(const Data &data, std::size_t begin, std::size_t end,
                                       const Profile &profile = Profile{}) {
    std::vector<Footprint> result(data.arrays.size());
    for (std::size_t u = begin; u < end; ++u)
        for (std::size_t a = 0; a < result.size(); ++a) {
            for (const auto &box : data.units[u].arrays[a].upload) append_box(result[a].upload, box);
            for (const auto &box : data.units[u].arrays[a].download) append_box(result[a].download, box);
        }
    for (std::size_t a = 0; a < result.size(); ++a) {
        deduplicate_boxes(data.arrays[a], result[a].upload,
                          profile.valid ? profile.h2d_latency : -1, profile.h2d_bandwidth);
        deduplicate_boxes(data.arrays[a], result[a].download,
                          profile.valid ? profile.d2h_latency : -1, profile.d2h_bandwidth);
    }
    return result;
}
inline double cpu_seconds(const Unit &unit, const Profile &p) {
    return std::max(unit.flops / p.cpu_flops, unit.memory_bytes / p.cpu_bandwidth);
}
inline double gpu_seconds(const Data &data, std::size_t begin, std::size_t end, const Profile &p) {
    bool valid = true;
    double flops = 0, memory = 0, result = 0;
    for (std::size_t u = begin; u < end; ++u) {
        flops += data.units[u].flops; memory += data.units[u].memory_bytes;
        if (data.units[u].iterations) result += p.launch_seconds;
    }
    auto footprints = interval(data, begin, end, p);
    for (std::size_t a = 0; a < data.arrays.size(); ++a) {
        for (const auto &b : footprints[a].upload)
            result += p.h2d_latency * copy_operations(data.arrays[a], b, valid) + box_bytes(data.arrays[a], b, valid) / p.h2d_bandwidth;
        for (const auto &b : footprints[a].download)
            result += p.d2h_latency * copy_operations(data.arrays[a], b, valid) + box_bytes(data.arrays[a], b, valid) / p.d2h_bandwidth;
    }
    result += std::max(flops / p.gpu_flops, memory / p.gpu_bandwidth);
    return valid && std::isfinite(result) ? result : std::numeric_limits<double>::infinity();
}
struct DecisionRange {
    inline static thread_local unsigned depth = 0;
    bool enabled = false;
    DecisionRange() {
#ifdef FORT_OFFLOAD_NVTX
        const char *value = std::getenv("FORT_RUNTIME_TRACE");
        enabled = depth == 0 && value && std::strcmp(value, "1") == 0;
        if (enabled) nvtxRangePushA("FORT offload decision");
#endif
        ++depth;
    }
    ~DecisionRange() {
        --depth;
#ifdef FORT_OFFLOAD_NVTX
        if (enabled) nvtxRangePop();
#endif
    }
};
inline Plan select(const Data &data, const Profile &p, bool automatic) {
    DecisionRange range;
    Plan result;
    result.valid = data.valid;
    const auto count = data.units.size();
    if (!count || !data.valid) return result;
    // The numerical value ABI may evaluate a scalar protected by an empty
    // earlier nest. Preserve the original guarded native entry in this case.
    bool empty = false, active = false;
    for (const auto &unit : data.units) {
        empty |= unit.iterations == 0;
        active |= unit.iterations != 0;
    }
    if (empty && active && !data.guarded_inputs_checked) return native_plan(data);
    if (automatic && !p.valid) return native_plan(data);
    if (!automatic) {
        if (active) result.choices.push_back({0, count, true});
        else for (std::size_t i=0; i<count; ++i) result.choices.push_back({i, i+1, false});
        return result;
    }
    std::vector<double> costs(count + 1, std::numeric_limits<double>::infinity());
    std::vector<Choice> first(count);
    costs[count] = 0;
    for (std::size_t i = count; i-- > 0;) {
        const double cpu = cpu_seconds(data.units[i], p);
        costs[i] = cpu + costs[i + 1]; first[i] = {i, i + 1, false};
        double native = 0;
        for (std::size_t j = i + 1; j <= count; ++j) {
            native += cpu_seconds(data.units[j - 1], p);
            if (j - i > 4 && !(i == 0 && j == count)) continue;
            const double gpu = gpu_seconds(data, i, j, p);
            if (gpu < .8 * native && gpu + costs[j] < costs[i]) {
                costs[i] = gpu + costs[j]; first[i] = {i, j, true};
            }
        }
    }
    for (std::size_t i = 0; i < count; i = first[i].end) result.choices.push_back(first[i]);
    return result;
}
inline void decision_trace_single(const char *entry, const char *mode, std::size_t gpu_units, std::size_t cpu_units,
                           const char *unit_kind = "regions") {
    const char *enabled = std::getenv("FORT_OFFLOAD_TRACE");
    if (enabled && std::strcmp(enabled, "0"))
        std::fprintf(stderr, "FORT_OFFLOAD entry=%s mode=%s gpu_units=%zu cpu_units=%zu unit_kind=%s\n", entry, mode, gpu_units, cpu_units, unit_kind);
}
inline void decision_trace(const char *entry, const char *mode, std::size_t gpu_units, std::size_t cpu_units,
                           const char *unit_kind = "regions") {
    if (thread_id() == 0) decision_trace_single(entry, mode, gpu_units, cpu_units, unit_kind);
}
inline void plan_trace(const char *entry, const Data &data, const Profile &p, const Plan &plan) {
    const char *enabled = std::getenv("FORT_OFFLOAD_TRACE");
    if (!enabled || !std::strcmp(enabled, "0")) return;
    if (data.guarded_inputs_checked) {
        std::fprintf(stderr, "FORT_OFFLOAD_ACTIVE entry=%s regions=", entry);
        for (std::size_t i=0; i<data.units.size(); ++i)
            std::fprintf(stderr, "%s%zu", i ? "," : "", data.units[i].source_region);
        std::fprintf(stderr, "\n");
    }
    for (const auto &choice : plan.choices) {
        std::size_t upload=0, download=0, launches=0;
        bool valid=true;
        double work=0, native=0;
        for (std::size_t u=choice.begin; u<choice.end; ++u) {
            work += data.units[u].flops;
            native += cpu_seconds(data.units[u], p);
            if (choice.gpu && data.units[u].iterations) ++launches;
        }
        if (choice.gpu) {
            auto footprints = interval(data, choice.begin, choice.end, p);
            for (std::size_t a=0; a<data.arrays.size(); ++a) {
                for (const auto &b : footprints[a].upload) {
                    const auto bytes = box_bytes(data.arrays[a], b, valid);
                    if (!add(upload, bytes, upload)) valid=false;
                }
                for (const auto &b : footprints[a].download) {
                    const auto bytes = box_bytes(data.arrays[a], b, valid);
                    if (!add(download, bytes, download)) valid=false;
                }
            }
        }
        const double estimate = p.valid ? (choice.gpu ? gpu_seconds(data,choice.begin,choice.end,p) : native) : -1;
        std::fprintf(stderr, "FORT_OFFLOAD_INTERVAL entry=%s begin=%zu end=%zu mode=%s work=%.0f launches=%zu upload_bytes=%zu download_bytes=%zu estimate_seconds=%.9g volume_valid=%d\n",
                     entry,choice.begin,choice.end,choice.gpu ? "gpu" : "native",work,launches,upload,download,estimate,valid);
    }
}

#ifdef __CUDACC__
class DeviceScope {
    int previous_ = -1;
    bool changed_ = false;
  public:
    explicit DeviceScope(int device) {
        CUCH(cudaGetDevice(&previous_));
        changed_ = device >= 0 && device != previous_;
        if (changed_) CUCH(cudaSetDevice(device));
    }
    ~DeviceScope() { if (changed_) CUCH(cudaSetDevice(previous_)); }
};
// Device allocation is full-shaped here; only the proved sections are copied.
// This storage has no whole-array coherence flags and never escapes the call.
class Allocation {
    storage::allocation_detail::DeviceAllocation allocation_;
  public:
    explicit Allocation(std::size_t bytes) : allocation_(bytes, storage::AllocationPolicy::pooled) {}
    void *get() const { return allocation_.get(); }
    void release_completed() { allocation_.release_completed(); }
};
inline void copy_box(const Array &a, const Box &box, void *device, bool upload) {
    bool valid = true;
    const auto bytes = box_bytes(a, box, valid);
    if (!valid) storage::fail("invalid ordinary-call transfer footprint");
    if (!bytes) return;
    auto upper = box.upper;
    for (auto &value : upper) ++value;
    const fort_physical::CopyPlan plan(a.element_bytes, a.dimensions, box.lower, upper);
    if (!plan.valid) storage::fail("invalid physical transfer layout");
    auto *host = static_cast<char *>(a.host);
    auto *gpu = static_cast<char *>(device);
    const auto direction = upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    plan.visit([&](const fort_physical::CopyOperation &op) {
        void *to = upload ? static_cast<void *>(gpu+op.offset) : static_cast<void *>(host+op.offset);
        const void *from = upload ? static_cast<void *>(host+op.offset) : static_cast<void *>(gpu+op.offset);
        if (op.depth == 1) {
            if (op.height == 1 || op.pitch == op.width) CUCH(cudaMemcpy(to, from, op.width*op.height, direction));
            else CUCH(cudaMemcpy2D(to, op.pitch, from, op.pitch, op.width, op.height, direction));
        } else {
            cudaMemcpy3DParms parameters{};
            parameters.srcPtr = make_cudaPitchedPtr(const_cast<void *>(from), op.pitch, op.pitch, op.physical_height);
            parameters.dstPtr = make_cudaPitchedPtr(to, op.pitch, op.pitch, op.physical_height);
            parameters.extent = make_cudaExtent(op.width, op.height, op.depth);
            parameters.kind = direction;
            CUCH(cudaMemcpy3D(&parameters));
        }
        return true;
    });
    storage::trace(upload ? "upload" : "download", bytes);
}
#endif
} // namespace generated_kernels::offload
#endif
