/* Bounded physical region algebra shared by execution and cost simulation.
 * Inputs are validated physical rectangles with exclusive upper coordinates. */
#ifndef FORT_SCOPED_REGIONS_HPP
#define FORT_SCOPED_REGIONS_HPP
#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>
namespace fort_scoped { namespace coherence {
constexpr size_t rectangle_limit = 32, intersection_limit = 1024;
struct Fragmented {};
struct Box { std::vector<size_t> lo, hi; };
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
    return result;
}
inline Region unite(Region a, const Region &b, Budget &budget) {
    for (const auto &box : b) {
        if (std::any_of(a.begin(), a.end(), [&](const Box &old) { return contains(old, box); })) continue;
        a.erase(std::remove_if(a.begin(), a.end(), [&](const Box &old) { return contains(box, old); }), a.end());
        for (auto &piece : difference({box}, a, budget)) append(a, std::move(piece));
    }
    return a;
}
struct Effects { Region reads, writes, overwrites; };
struct Prepared { bool device; Region initialized, current, opposite; };
template<class State> inline Prepared prepare(State &b, const Effects &e, bool device) {
    Budget budget;
    auto initialized = unite(b.initialized, e.writes, budget);
    auto current = unite(device ? b.device_current : b.host_current, e.writes, budget);
    auto opposite = difference(device ? b.host_current : b.device_current, e.writes, budget);
    return {device, std::move(initialized), std::move(current), std::move(opposite)};
}
}} // namespace fort_scoped::coherence
#endif
