/* Bounded metadata helpers. Native hints choose only unchanged original CPU
 * execution; they are not placement/definition proofs or retained data. */
#ifndef FORT_SCOPED_HOST_METADATA_HPP
#define FORT_SCOPED_HOST_METADATA_HPP
#include "scoped_runtime.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fort_scoped { namespace metadata {
template<class Append,class Remaining>
inline void string(Append &append,Remaining &remaining,const char *value) {
    if (!value) { append(0); return; }
    const size_t available=remaining();
    if (!available) throw std::length_error("memo environment budget");
    size_t length=0;
    while (length<available-1 && value[length]) ++length;
    if (value[length]) throw std::length_error("memo environment budget");
    append(length+1);
    for(size_t i=0;i<length;++i)append(static_cast<unsigned char>(value[i]));
}
// Serialization/order is shared with the existing placement candidate memo.
// Do not add native-hint-only fields here: that would change memo identities.
template<class Append,class Remaining>
inline void execution_environment(Append &append,Remaining &remaining) {
#ifdef __linux__
    cpu_set_t affinity;CPU_ZERO(&affinity);
    if(sched_getaffinity(0,sizeof(affinity),&affinity))
        throw std::runtime_error("memo CPU affinity unavailable");
    append(CPU_COUNT(&affinity));
    for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&affinity))append(cpu);
#else
    throw std::runtime_error("memo CPU affinity unavailable");
#endif
#ifdef _OPENMP
    omp_sched_t schedule;int chunk=0;omp_get_schedule(&schedule,&chunk);
    for(int value:{omp_get_level(),omp_get_active_level(),omp_get_max_active_levels(),
        omp_get_num_threads(),omp_get_max_threads(),omp_get_dynamic(),
        omp_get_thread_limit(),int(omp_get_proc_bind()),int(schedule),chunk})
        append(static_cast<uint64_t>(value));
#if _OPENMP >= 201307
    const int places=omp_get_num_places();
    if(places<0)throw std::runtime_error("memo OpenMP places unavailable");
    append(places);append(static_cast<uint64_t>(omp_get_place_num()));
    for(int place=0;place<places;++place) {
        const int count=omp_get_place_num_procs(place);
        if(count<0)throw std::runtime_error("memo OpenMP place processors unavailable");
        append(count);
        if(size_t(count)>remaining())throw std::length_error("memo environment budget");
        std::vector<int> processors(count);
        if(count)omp_get_place_proc_ids(place,processors.data());
        for(int cpu:processors)append(static_cast<uint64_t>(cpu));
    }
    const int partition_count=omp_get_partition_num_places();
    if(partition_count<0)throw std::runtime_error("memo OpenMP partition unavailable");
    append(partition_count);
    if(size_t(partition_count)>remaining())throw std::length_error("memo environment budget");
    std::vector<int> partition(partition_count);
    if(partition_count)omp_get_partition_place_nums(partition.data());
    for(int place:partition)append(static_cast<uint64_t>(place));
#else
    throw std::runtime_error("memo OpenMP place metadata unavailable");
#endif
#else
    append(0);
#endif
    for(const char *name:{"OMP_WAIT_POLICY","GOMP_SPINCOUNT","OMP_PLACES","GOMP_CPU_AFFINITY","OMP_PROC_BIND"})
        string(append,remaining,std::getenv(name));
}
}} // fort_scoped::metadata

namespace fort_scoped { namespace native_hint {
constexpr size_t word_limit=FORT_SCOPE_NATIVE_HINT_WORD_LIMIT, entry_limit=64;
constexpr uint32_t hit_limit=64;
class Generation {
    std::atomic<uint64_t> value_;
public:
    explicit Generation(uint64_t value=0) noexcept : value_(value) {}
    uint64_t value() const noexcept { return value_.load(std::memory_order_acquire); }
    void advance() noexcept {
        auto old=value_.load(std::memory_order_relaxed);
        while(old!=std::numeric_limits<uint64_t>::max() &&
              !value_.compare_exchange_weak(old,old+1,std::memory_order_release,std::memory_order_relaxed)) {}
    }
};
struct Key {
    std::array<uint64_t,word_limit> words;
    size_t count=0;
    uint64_t hash=1469598103934665603ULL;
    Key() noexcept = default;
    Key(const Key &other) noexcept { *this=other; }
    Key &operator=(const Key &other) noexcept {
        if(this==&other)return *this;
        count=other.count;hash=other.hash;
        std::copy_n(other.words.begin(),count,words.begin());return *this;
    }
    void integer(uint64_t value) {
        if(count==word_limit)throw std::length_error("native hint metadata budget");
        words[count++]=value;
        for(unsigned i=0;i<8;++i){hash^=(value>>(8*i))&255;hash*=1099511628211ULL;}
    }
    bool same(const Key &other) const noexcept {
        return hash==other.hash && count==other.count &&
            std::equal(words.begin(),words.begin()+count,other.words.begin());
    }
};
// Caller owns the registry mutex. Cache entries contain only copied words and
// bounded LRU/TTL counters; generations bracket observations, never old hits.
class Cache {
    struct Entry {bool valid=false;Key key;uint32_t hits=0;uint64_t age=0;};
    std::array<Entry,entry_limit> entries_;
    uint64_t clock_=0;
    uint64_t tick() noexcept {
        if(clock_==std::numeric_limits<uint64_t>::max()){
            for(auto &entry:entries_)entry.age=0;
            clock_=0;
        }
        return ++clock_;
    }
public:
    bool find(const Key &key) noexcept {
        for(auto &entry:entries_)if(entry.valid && entry.key.same(key)) {
            if(entry.hits==hit_limit){entry.valid=false;return false;}
            ++entry.hits;entry.age=tick();return true;
        }
        return false;
    }
    bool insert(const Key &key) noexcept {
        Entry *victim=nullptr;
        for(auto &entry:entries_)if(entry.valid && entry.key.same(key))return false;
        // Even the 64th hit cannot renew the budget. A later lookup first
        // expires this entry and takes the ordinary miss/reprobe path.
        if(!victim)for(auto &entry:entries_)if(!entry.valid){victim=&entry;break;}
        if(!victim)for(auto &entry:entries_)if(!victim || entry.age<victim->age)victim=&entry;
        victim->key=key;victim->valid=true;victim->hits=0;victim->age=tick();return true;
    }
    size_t size() const noexcept {
        return size_t(std::count_if(entries_.begin(),entries_.end(),[](const auto &e){return e.valid;}));
    }
};
}} // fort_scoped::native_hint
#endif
