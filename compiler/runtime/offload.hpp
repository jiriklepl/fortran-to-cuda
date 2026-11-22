// Checked section-index arithmetic and placement trace lines of generated entries.
#ifndef FORT_RUNTIME_OFFLOAD_HPP
#define FORT_RUNTIME_OFFLOAD_HPP
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace generated_kernels::offload {
inline int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
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
        value > static_cast<long double>(std::numeric_limits<long long>::max())) {
        valid = false; return 0;
    }
    // A low-precision long double can round LLONG_MAX up to 2^63. Exclude
    // that endpoint before conversion, which would otherwise be undefined.
    if constexpr (std::numeric_limits<long double>::digits < std::numeric_limits<long long>::digits) {
        if (value >= -static_cast<long double>(std::numeric_limits<long long>::min())) {
            valid = false; return 0;
        }
    }
    // Conversion already truncates. Its round trip checks integrality without
    // a separate trunc operation. Keep finite/range rejection before the cast:
    // rejected NaNs and just-out-of-range fractions must not gain exceptions.
    const auto result = static_cast<long long>(value);
    if (static_cast<long double>(result) != value) { valid = false; return 0; }
    return result;
}
// Whether every checked operation of a section's bounds stayed in range.
struct Data {
    bool valid = true;
};
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
} // namespace generated_kernels::offload
#endif
