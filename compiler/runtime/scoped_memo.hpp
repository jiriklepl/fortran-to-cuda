/* Optional bounded metadata-only candidate memo. No runtime pointers, mirrors, CUDA
 * resources, proof tokens, reports or prior search counters enter an entry.
 * Caller hardware/profile/FP guards remain mandatory at every installation. */
#ifndef FORT_SCOPED_MEMO_HPP
#define FORT_SCOPED_MEMO_HPP
#include "scoped_planning.hpp"
#include <cstring>
#include <list>
#include <mutex>
#include <stdexcept>

namespace fort_scoped { namespace memo {
constexpr size_t entry_limit=64, byte_limit=8*1024*1024;
struct Key {
    std::vector<uint64_t> words;
    uint64_t hash=1469598103934665603ULL;
    void integer(uint64_t value) {
        if (words.size()>=byte_limit/sizeof(uint64_t)) throw std::length_error("memo metadata budget");
        words.push_back(value);
        for (unsigned i=0;i<8;++i) { hash^=(value>>(8*i))&255; hash*=1099511628211ULL; }
    }
    void real(double value) {
        if (!std::isfinite(value)) throw std::domain_error("unknown memo model input");
        uint64_t bits=0; static_assert(sizeof(bits)==sizeof(value));
        std::memcpy(&bits,&value,sizeof(bits)); integer(bits);
    }
    void coordinates(const std::vector<size_t> &values) {
        integer(values.size());for(auto value:values)integer(value);
    }
    void region(const coherence::Region &parts) {
        integer(parts.size());for(const auto &part:parts){coordinates(part.lo);coordinates(part.hi);}
    }
    void resource(const planning::Resource &r) {
        integer(r.element_bytes);integer(r.bytes);coordinates(r.extents);
        region(r.initialized);region(r.host_current);region(r.device_current);
        integer(r.allocated);region(r.pending_host_reads);integer(r.last_device_use);
    }
};
inline bool eligible(const planning::Inputs &in) {
    return in.graph_mode && in.definitions_validated && !in.device_ready && !in.pending &&
        std::all_of(in.resources.begin(),in.resources.end(),[](const auto &r){
            return !r.allocated && r.device_current.empty() && r.pending_host_reads.empty() && !r.last_device_use;
        });
}
inline std::optional<Key> make_key(const planning::Inputs &in,const fort_scope_plan_costs &c,
                                    const std::vector<uint64_t> &environment) noexcept {
    if (!eligible(in))return std::nullopt;
    try {
        planning::detail::validate(in,c);
        Key key;key.integer(1); // serialization version
        key.integer(planning::detail::frontier_limit);key.integer(planning::detail::candidate_limit);
        key.integer(coherence::rectangle_limit);key.integer(coherence::intersection_limit);
        key.integer(environment.size());for(auto item:environment)key.integer(item);
        key.integer(in.device_budget);key.integer(in.device_ready);key.integer(in.driver_initialized);
        key.integer(in.pending);key.integer(in.graph_mode);key.real(in.graph_minimum_relative_saving);
        key.integer(in.query_construction_operations);key.integer(in.asynchronous_release);
        key.integer(in.continuation);key.integer(in.charge_create);key.integer(in.registrations_incurred);
        key.integer(in.transfer_mode);
        key.integer(in.direct_transfer_fit_applicable);
        key.integer(c.version);key.integer(c.valid);key.integer(c.max_allocation_bytes);
        for(double x:{c.cpu_flops,c.cpu_bandwidth,c.gpu_flops,c.gpu_bandwidth,c.h2d_latency,c.h2d_bandwidth,
                c.d2h_latency,c.d2h_bandwidth,c.create_seconds,c.register_seconds,c.host_access_seconds,
                c.device_access_seconds,c.gpu_setup_seconds,c.cold_driver_startup_seconds,c.allocation_seconds,
                c.release_seconds,c.wait_seconds,c.launch_enqueue_seconds,c.planning_operation_seconds,
                c.h2d_fit_relative_error,c.d2h_fit_relative_error})key.real(x);
        key.integer(bool(in.team_costs));
        if(in.team_costs){const auto &t=*in.team_costs;
            key.integer(t.version);key.integer(t.valid);key.integer(t.cpu_threads);key.integer(t.expected_omp_level);
            key.integer(t.protocol_id);
            for(double x:{t.native_cpu_flops,t.native_cpu_bandwidth,t.owner_seconds,t.descriptor_seconds,
                t.entry_seconds,t.cpu_worker_seconds,t.gpu_worker_seconds,t.native_call_seconds,
                t.native_worker_seconds})key.real(x);
        }
        key.integer(bool(in.transfer_costs));
        if(in.transfer_costs){const auto &t=*in.transfer_costs;
            key.integer(t.version);key.integer(t.valid);key.integer(t.async_engine_count);key.integer(t.max_slot_bytes);
            for(double x:t.staging_cold_seconds)key.real(x);
            for(double x:t.staging_reuse_seconds)key.real(x);
            for(double x:{t.event_record_seconds,t.event_wait_seconds,t.ready_event_seconds,
                t.preparation_operation_seconds,t.pack_bytes_per_second,t.unpack_bytes_per_second,
                t.pack_row_seconds,t.unpack_row_seconds,t.pinned_h2d_latency,t.pinned_h2d_bandwidth,
                t.pinned_d2h_latency,t.pinned_d2h_bandwidth})key.real(x);
        }
        std::unordered_map<fort_buffer_t,const planning::Resource *> resources;
        for(const auto &r:in.resources)resources.emplace(r.handle,&r);
        std::unordered_map<fort_buffer_t,size_t> incidence;
        std::vector<const planning::Resource *> used;
        for(const auto &op:in.operations)for(const auto &b:op.bindings)
            if(incidence.emplace(b.buffer,incidence.size()).second)used.push_back(resources.at(b.buffer));
        key.integer(used.size());for(const auto *r:used)key.resource(*r);
        // Publication order affects synchronous waits and pinned staging
        // capacity reuse. Preserve used-resource incidence and unused
        // positions; unused geometry itself remains an unordered multiset.
        key.integer(in.resources.size());
        for(const auto &r:in.resources) {
            const auto found=incidence.find(r.handle);
            key.integer(found==incidence.end() ? 0 : found->second+1);
        }
        // Unused fresh host-only resources cannot gain device allocations.
        // Their exact metadata still affects lifecycle/validation and is kept
        // as a multiset, independent of unordered registry iteration order.
        std::vector<std::vector<uint64_t>> unused;
        size_t unused_words=0;
        for(const auto &r:in.resources)if(!incidence.count(r.handle)){
            Key part;part.resource(r);
            if(part.words.size()+key.words.size()+unused_words+1>byte_limit/sizeof(uint64_t))
                throw std::length_error("memo metadata budget");
            unused_words+=part.words.size()+1;unused.push_back(std::move(part.words));
        }
        std::sort(unused.begin(),unused.end());key.integer(unused.size());
        for(const auto &part:unused){key.integer(part.size());for(auto word:part)key.integer(word);}
        key.integer(in.operations.size());
        for(const auto &op:in.operations){
            key.integer(op.kind);key.integer(op.unit);key.real(op.flops);key.real(op.memory_bytes);
            key.integer(op.gpu_available);key.real(op.cpu_numerical_seconds);key.real(op.gpu_numerical_seconds);
            key.integer(bool(op.compute_costs));
            if(op.compute_costs){const auto &v=*op.compute_costs;key.integer(v.version);key.integer(v.known_backends);
                key.real(v.native_fortran_seconds);key.real(v.generated_cpu_seconds);key.real(v.gpu_seconds);}
            key.integer(op.bindings.size());for(const auto &b:op.bindings){
                key.integer(incidence.at(b.buffer));key.region(b.effects.reads);
                key.region(b.effects.writes);key.region(b.effects.overwrites);
            }
        }
        return key;
    }catch(...){return std::nullopt;} // Optional allocation/representation failure uses normal planning.
}
class Cache {
    struct Entry {Key key;std::vector<bool> choices;size_t bytes;};
    std::list<Entry> entries_;
    size_t bytes_=0;
    std::mutex mutex_;
public:
    std::optional<std::vector<bool>> find(const Key &key) noexcept {
        try {
            std::lock_guard<std::mutex> guard(mutex_);
            for(auto it=entries_.begin();it!=entries_.end();++it)
                if(it->key.hash==key.hash && it->key.words==key.words){
                    auto choices=it->choices;entries_.splice(entries_.begin(),entries_,it);return choices;
                }
        }catch(...){ }return std::nullopt;
    }
    void insert(Key key,const std::vector<bool> &choices) noexcept {
        try {
            const size_t bytes=sizeof(Entry)+2*sizeof(void*)+key.words.capacity()*sizeof(uint64_t)+
                choices.capacity()/8+(choices.capacity()%8!=0);
            if(bytes>byte_limit)return;
            std::lock_guard<std::mutex> guard(mutex_);
            for(auto it=entries_.begin();it!=entries_.end();++it)
                if(it->key.hash==key.hash && it->key.words==key.words)return;
            entries_.push_front({std::move(key),choices,bytes});bytes_+=bytes;
            while(entries_.size()>entry_limit || bytes_>byte_limit){bytes_-=entries_.back().bytes;entries_.pop_back();}
        }catch(...){ } // Never replace a legal planning path with a cache failure.
    }
    size_t size(){std::lock_guard<std::mutex> guard(mutex_);return entries_.size();}
    size_t bytes(){std::lock_guard<std::mutex> guard(mutex_);return bytes_;}
};
}} // fort_scoped::memo
#endif
