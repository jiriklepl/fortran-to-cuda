// One bounded metadata record shared by the existing staging pool and planner.
// Credit represents submitted heuristic observations, never completion or data authority.
#ifndef FORT_COPY_STARTUP_HPP
#define FORT_COPY_STARTUP_HPP
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
namespace fort_physical { namespace compact {
constexpr std::size_t capacities[]={256ULL*1024,1024ULL*1024,4ULL*1024*1024,16ULL*1024*1024};
namespace startup {
struct Key {
    std::array<std::uint64_t,155> words{};
    bool operator==(const Key &other) const {return words==other.words;}
    bool operator!=(const Key &other) const {return !(*this==other);}
};
struct State {
    Key model{};
    std::array<double,4> credit{},cap{};
    std::uint64_t revision=0,generation=0;
    bool bound=false,disabled=false;
};
inline bool same(const State &a,const State &b) {
    return a.revision==b.revision && a.generation==b.generation && a.bound==b.bound &&
        a.disabled==b.disabled && a.model==b.model && a.credit==b.credit && a.cap==b.cap;
}
inline void touch(State &s) {
    if(s.revision==std::numeric_limits<std::uint64_t>::max()) {s.disabled=true;s.credit={};}
    else ++s.revision;
}
inline void ownership_changed(State &s) {
    s.credit={};
    if(s.generation==std::numeric_limits<std::uint64_t>::max())s.disabled=true;
    else ++s.generation;
    touch(s);
}
struct Offer {
    unsigned bucket=4;
    std::size_t capacity=0;
    double observation=0,premium=0;
    bool seed=false,selected=false;
};
// Only the caller that observed every direct batch submission succeed may book.
// Concurrent observations under the same ownership generation may accumulate;
// a model/device or ownership change makes an old observation inapplicable.
inline bool observe(State &s,const State &before,const Offer &d) {
    if(s.disabled || !s.bound || !before.bound || s.model!=before.model || s.generation!=before.generation ||
       d.selected || d.bucket>=4 || !(std::isfinite(d.observation)&&d.observation>=0) || d.observation<=0)return false;
    const double room=std::max(0.,s.cap[d.bucket]-s.credit[d.bucket]);
    s.credit[d.bucket]=std::min(s.cap[d.bucket],s.credit[d.bucket]+std::min(d.observation,room));
    touch(s);return true;
}
// Called inside the existing pool reservation lock before any CUDA allocation.
// Exact revision, key and generation prevent two contenders spending one credit.
inline bool reserve(State &s,const State &before,const Offer &d) {
    if(!same(s,before) || !s.bound || s.disabled || !d.seed || !d.selected || d.bucket>=4 ||
       d.capacity!=capacities[d.bucket] ||
       !(std::isfinite(d.premium)&&d.premium>=0) || s.credit[d.bucket]<d.premium)return false;
    ownership_changed(s);return true;
}
}}}
#endif
