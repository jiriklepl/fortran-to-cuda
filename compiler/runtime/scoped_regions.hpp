/* Bounded physical region algebra shared by execution and cost simulation.
 * Inputs are validated physical rectangles with exclusive upper coordinates. */
#ifndef FORT_SCOPED_REGIONS_HPP
#define FORT_SCOPED_REGIONS_HPP
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
namespace fort_scoped { namespace coherence {
constexpr size_t rectangle_limit = 128, intersection_limit = 8192;
// Fortran arrays have at most 15 dimensions.
constexpr size_t rank_limit = 15;
struct Fragmented {};
// One corner of a rectangle. The values live inline, so building, copying and
// splitting boxes never allocates.
class Coordinates {
public:
    Coordinates() = default;
    Coordinates(std::initializer_list<size_t> values) { for (const auto value : values) push_back(value); }
    explicit Coordinates(size_t count, size_t value = 0) { resize(count, value); }
    Coordinates(const std::vector<size_t> &values) { for (const auto value : values) push_back(value); }
    size_t size() const { return count_; }
    bool empty() const { return !count_; }
    size_t &operator[](size_t k) { return values_[k]; }
    const size_t &operator[](size_t k) const { return values_[k]; }
    size_t *data() { return values_; }
    const size_t *data() const { return values_; }
    size_t *begin() { return values_; }
    size_t *end() { return values_ + count_; }
    const size_t *begin() const { return values_; }
    const size_t *end() const { return values_ + count_; }
    void push_back(size_t value) {
        if (count_ == rank_limit) throw std::length_error("array rank exceeds the section coordinate limit");
        values_[count_++] = value;
    }
    void resize(size_t count, size_t value = 0) {
        if (count > rank_limit) throw std::length_error("array rank exceeds the section coordinate limit");
        for (size_t k=count_; k<count; ++k) values_[k] = value;
        count_ = count;
    }
    friend bool operator==(const Coordinates &a, const Coordinates &b) {
        return a.count_ == b.count_ && std::equal(a.begin(), a.end(), b.begin());
    }
    friend bool operator!=(const Coordinates &a, const Coordinates &b) { return !(a == b); }
private:
    size_t values_[rank_limit] = {};
    size_t count_ = 0;
};
struct Box { Coordinates lo, hi; };
using Region = std::vector<Box>;
struct Budget {
    size_t checks = 0;
    void check() { if (++checks > intersection_limit) throw Fragmented{}; }
};
inline bool empty(const Box &b) {
    for (size_t k=0; k<b.lo.size(); ++k) if (b.lo[k] == b.hi[k]) return true;
    return false;
}
inline bool contains(const Box &a, const Box &b) {
    for (size_t k=0; k<a.lo.size(); ++k) if (a.lo[k] > b.lo[k] || a.hi[k] < b.hi[k]) return false;
    return true;
}
inline bool overlaps(const Box &a, const Box &b) {
    for (size_t k=0; k<a.lo.size(); ++k) if (a.lo[k] >= b.hi[k] || b.lo[k] >= a.hi[k]) return false;
    return true;
}
inline std::optional<Box> intersection(const Box &a, const Box &b, Budget &budget) {
    budget.check();
    Box result = a;
    for (size_t k=0; k<a.lo.size(); ++k) {
        result.lo[k] = std::max(a.lo[k], b.lo[k]);
        result.hi[k] = std::min(a.hi[k], b.hi[k]);
        if (result.lo[k] >= result.hi[k]) return {};
    }
    return result;
}
inline void append(Region &region, Box box) {
    if (empty(box)) return;
    if (region.size() == rectangle_limit) throw Fragmented{};
    region.push_back(std::move(box));
}
inline void coalesce(Region &region) {
    // Inputs already passed the rectangle budget. Each successful pass removes
    // one box, so the deterministic scan is bounded by that same small limit.
    // Merge only exact neighbours: a bounding box across a hole would change
    // definedness and could copy stale or uninitialized cells.
    for (;;) {
        bool merged = false;
        for (size_t i=0; i<region.size() && !merged; ++i) {
            for (size_t j=i+1; j<region.size(); ++j) {
                auto &a = region[i];
                const auto &b = region[j];
                size_t axis = a.lo.size();
                bool adjacent = true;
                for (size_t k=0; k<a.lo.size(); ++k) {
                    if (a.lo[k] == b.lo[k] && a.hi[k] == b.hi[k]) continue;
                    if (axis != a.lo.size() || (a.hi[k] != b.lo[k] && b.hi[k] != a.lo[k])) {
                        adjacent = false;
                        break;
                    }
                    axis = k;
                }
                if (!adjacent || axis == a.lo.size()) continue;
                // Exclusive endpoints require no addition, including SIZE_MAX.
                a.lo[axis] = std::min(a.lo[axis], b.lo[axis]);
                a.hi[axis] = std::max(a.hi[axis], b.hi[axis]);
                region.erase(region.begin()+j);
                merged = true;
                break;
            }
        }
        if (!merged) return;
    }
}
inline Region subtract(const Box &box, const Box &cut, Budget &budget) {
    auto common = intersection(box, cut, budget);
    if (!common) return {box};
    Region result;
    Box middle = box;
    for (size_t k=0; k<box.lo.size(); ++k) {
        if (middle.lo[k] < common->lo[k]) {
            Box part = middle; part.hi[k] = common->lo[k]; append(result, std::move(part));
            middle.lo[k] = common->lo[k];
        }
        if (common->hi[k] < middle.hi[k]) {
            Box part = middle; part.lo[k] = common->hi[k]; append(result, std::move(part));
            middle.hi[k] = common->hi[k];
        }
    }
    return result;
}
inline Region difference(const Region &a, const Region &b, Budget &budget) {
    Region result;
    for (const auto &box : a) {
        // An already-current exact section needs no splitting work. Avoid
        // consuming the intersection budget while searching unrelated cuts
        // before a containing rectangle, as unite already does for coverage.
        if (std::any_of(b.begin(),b.end(),[&](const Box &cut) { return contains(cut,box); })) continue;
        Region pending{box};
        for (const auto &cut : b) {
            Region next;
            for (const auto &part : pending)
                for (auto &piece : subtract(part, cut, budget)) append(next, std::move(piece));
            pending = std::move(next);
            if (pending.empty()) break;
        }
        for (auto &part : pending) append(result, std::move(part));
    }
    coalesce(result);
    return result;
}
inline Region unite(Region a, const Region &b, Budget &budget) {
    for (const auto &box : b) {
        if (std::any_of(a.begin(), a.end(), [&](const Box &old) { return contains(old, box); })) continue;
        a.erase(std::remove_if(a.begin(), a.end(), [&](const Box &old) { return contains(box, old); }), a.end());
        for (auto &piece : difference({box}, a, budget)) append(a, std::move(piece));
    }
    coalesce(a);
    return a;
}
inline bool overlaps(const Region &a, const Region &b) {
    for (const auto &left : a) for (const auto &right : b) if (overlaps(left, right)) return true;
    return false;
}
// Every rectangle of `parts` lies inside one rectangle of `whole`. This is
// sufficient for containment, and exactly the case the algebra skips.
inline bool covers(const Region &whole, const Region &parts) {
    for (const auto &part : parts)
        if (!std::any_of(whole.begin(), whole.end(), [&](const Box &box) { return contains(box, part); })) return false;
    return true;
}
struct Effects { Region reads, writes, overwrites; };
// Defined device work on sections that are already initialized and current on
// the device, and that no host mirror overlaps, changes no coherence state:
// preparation would return its three regions, normalized. Resident fields are
// in this state between host accesses, so recognize it without allocating.
template<class State> inline bool device_effects_keep_state(State &b, const Effects &e) {
    // Stay well inside the pairwise budget that preparation would spend.
    if (b.host_current.size()*e.writes.size() > 64) return false;
    if (!covers(b.initialized, e.reads) || !covers(b.device_current, e.reads) ||
        !covers(b.initialized, e.writes) || !covers(b.device_current, e.writes) ||
        overlaps(b.host_current, e.writes)) return false;
    coalesce(b.initialized); coalesce(b.device_current); coalesce(b.host_current);
    return true;
}
template<class State> inline Region physical_values(const State &b) {
    // Merge complementary mirror partitions before applying the union budget.
    Region physical = b.host_current;
    physical.insert(physical.end(), b.device_current.begin(), b.device_current.end());
    coalesce(physical);
    Budget budget;
    return unite(b.initialized, physical, budget);
}
template<class State> inline Region tracked_accesses(const State &b, const Region &requested) {
    Budget budget;
    const auto unknown = difference(requested, physical_values(b), budget);
    return difference(requested, unknown, budget);
}
struct AdmittedReads { Region initialized, host_current; };
// The compiler establishes these exact upward reads at their original source
// point. Already tracked values keep their current mirror; only previously
// unknown values acquire a host source. Compute both sets before committing.
template<class State> inline AdmittedReads admit_source_reads(const State &b, const Region &reads) {
    Budget budget;
    const auto missing = difference(reads, b.initialized, budget);
    if (missing.empty()) return {b.initialized,b.host_current};
    auto initialized = unite(b.initialized, missing, budget);
    // A semantic definition kill does not erase a newer physical device
    // value. Only values with no tracked physical source acquire a host one.
    const auto physical = unite(b.host_current,b.device_current,budget);
    const auto host_sources = difference(missing,physical,budget);
    auto host = unite(b.host_current, host_sources, budget);
    return {std::move(initialized), std::move(host)};
}
struct Prepared { bool device; Region initialized, current, opposite; };
template<class State> inline Prepared prepare(State &b, const Effects &e, bool device) {
    Budget budget;
    auto initialized = unite(b.initialized, e.writes, budget);
    auto current = unite(device ? b.device_current : b.host_current, e.writes, budget);
    auto opposite = difference(device ? b.host_current : b.device_current, e.writes, budget);
    return {device, std::move(initialized), std::move(current), std::move(opposite)};
}
// Original CPU execution may touch only part of its declared envelope. Known
// values in possible read/preservation regions have already reached the host.
// Unknown holes stay unknown: only proved overwrites add definitions/current
// values. Possible writes invalidate overlapping device replicas regardless.
template<class State> inline Prepared prepare_original_host(const State &b, const Effects &e) {
    Budget budget;
    auto initialized = unite(b.initialized, e.overwrites, budget);
    auto current = unite(b.host_current, e.overwrites, budget);
    auto opposite = difference(b.device_current, e.writes, budget);
    return {false, std::move(initialized), std::move(current), std::move(opposite)};
}
}} // namespace fort_scoped::coherence
#endif
