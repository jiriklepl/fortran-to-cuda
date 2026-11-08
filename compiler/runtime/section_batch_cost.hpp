/* One complete physical-batch model shared by planning and admission.
 * Empirical placement costs confer no data, lifetime or execution authority. */
#ifndef FORT_SECTION_BATCH_COST_HPP
#define FORT_SECTION_BATCH_COST_HPP
#include "scoped_runtime.h"
#include "section_batch.hpp"
#include <cmath>
#include "copy_startup.hpp"
#include <cstring>
namespace fort_physical { namespace compact {
inline bool finite_nonnegative(double x){return std::isfinite(x)&&x>=0;}
inline bool valid_rate(const fort_scope_copy_rate_v3 &r) {
    double sum=0;
    for(auto x:{r.batch_seconds,r.submission_seconds,r.byte_seconds,r.row_seconds,r.descriptor_seconds}) {
        if(!finite_nonnegative(x))return false;
        sum+=x;
    }
    return std::isfinite(sum)&&sum>0 && finite_nonnegative(r.median_relative_error) &&
        finite_nonnegative(r.p90_relative_error)&&finite_nonnegative(r.maximum_relative_error)&&
        r.median_relative_error<=r.p90_relative_error&&r.p90_relative_error<=r.maximum_relative_error&&
        r.minimum_payload_bytes&&r.minimum_payload_bytes<=r.maximum_payload_bytes&&r.maximum_payload_bytes<=capacities[3]&&
        r.minimum_rows&&r.minimum_rows<=r.maximum_rows&&r.minimum_row_bytes&&r.minimum_row_bytes<=r.maximum_row_bytes&&
        r.maximum_row_bytes<=r.maximum_payload_bytes&&r.minimum_pitch_bytes&&r.minimum_pitch_bytes<=r.maximum_pitch_bytes&&
        r.maximum_span_bytes>=r.maximum_payload_bytes&&r.minimum_operations&&r.minimum_operations<=r.maximum_operations&&
        r.maximum_operations<=operation_limit&&r.minimum_descriptor_work&&r.minimum_descriptor_work<=r.maximum_descriptor_work&&
        r.minimum_kernel_chunks&&r.minimum_kernel_chunks<=r.maximum_kernel_chunks&&
        r.maximum_kernel_chunks<=(operation_limit+descriptor_limit-1)/descriptor_limit&&
        r.word_bytes_mask&&!(r.word_bytes_mask&~7u)&&r.layout_combinations_mask&&!(r.layout_combinations_mask&~254u);
}
inline bool finite_supported_prediction(const fort_scope_copy_rate_v3 &r,bool direct) {
    const auto submissions=direct?r.maximum_operations:r.maximum_kernel_chunks;
    const auto upper=r.batch_seconds+double(submissions)*r.submission_seconds+
        double(r.maximum_payload_bytes)*r.byte_seconds+double(r.maximum_rows)*r.row_seconds+
        double(r.maximum_descriptor_work)*r.descriptor_seconds;
    return std::isfinite(upper)&&upper>0;
}
inline bool valid_costs(const fort_scope_copy_costs_v3 &c) {
    if(c.version!=FORT_SCOPE_COPY_COST_ABI_VERSION||c.valid!=1||c.minimum_relative_saving!=0.5)return false;
    for(unsigned d=0;d<2;++d)
        if(!valid_rate(c.direct[d])||!valid_rate(c.compact_cold[d])||!valid_rate(c.compact_reuse[d])||
           !finite_supported_prediction(c.direct[d],true)||!finite_supported_prediction(c.compact_cold[d],false)||
           !finite_supported_prediction(c.compact_reuse[d],false)||
           c.compact_cold[d].minimum_operations<1||c.compact_reuse[d].minimum_operations<1)return false;
    return true;
}
inline unsigned layout_mask(const Plan &p) {
    unsigned mask=0;
    for(const auto &chunk:p.chunks)for(std::size_t i=0;i<chunk.count;++i) {
        const auto &d=chunk.ops[i];mask|=d.depth>1?4u:d.height>1&&d.pitch!=d.width?2u:1u;
    }
    return mask;
}
inline bool supports(const fort_scope_copy_rate_v3 &r,const Plan &p,std::size_t operations) {
    const auto word_bit=p.word_bytes==1?1u:p.word_bytes==4?2u:p.word_bytes==8?4u:0u;
    const auto layout=layout_mask(p);
    return word_bit&&(r.word_bytes_mask&word_bit)&&layout&&(r.layout_combinations_mask&(1u<<layout))&&
        p.bytes>=r.minimum_payload_bytes&&p.bytes<=r.maximum_payload_bytes&&p.rows>=r.minimum_rows&&p.rows<=r.maximum_rows&&
        p.minimum_row_bytes>=r.minimum_row_bytes&&p.maximum_row_bytes<=r.maximum_row_bytes&&
        p.minimum_pitch>=r.minimum_pitch_bytes&&p.maximum_pitch<=r.maximum_pitch_bytes&&
        p.last_offset-p.first_offset<=r.maximum_span_bytes&&operations>=r.minimum_operations&&operations<=r.maximum_operations&&
        p.work>=r.minimum_descriptor_work&&p.work<=r.maximum_descriptor_work&&
        p.chunks.size()>=r.minimum_kernel_chunks&&p.chunks.size()<=r.maximum_kernel_chunks;
}
inline double predict(const fort_scope_copy_rate_v3 &r,const Plan &p,std::size_t submissions) {
    return r.batch_seconds+double(submissions)*r.submission_seconds+double(p.bytes)*r.byte_seconds+
        double(p.rows)*r.row_seconds+double(p.work)*r.descriptor_seconds;
}
struct PoolState {std::size_t capacity=0,available_device_bytes=0;bool reused=false,cold_available=true;};
struct Pricing {fort_scope_copy_costs_v3 copy{};fort_scope_plan_costs base{};};
struct Price {
    bool available=false,compact_available=false,selected=false;
    std::size_t capacity=0,operations=0,chunks=0,workspace_bytes=0;
    double direct_seconds=0,compact_seconds=0,compact_path_seconds=0;
    double direct_p90_relative_error=0,compact_p90_relative_error=0;
};
inline Price price(const Plan &p,const fort_scope_copy_costs_v3 &c,const fort_scope_plan_costs &base,
                   bool upload,bool pending,const PoolState &pool) {
    Price out;if(!valid_costs(c)||!p.bytes||p.chunks.empty())return out;
    for(const auto &chunk:p.chunks)out.operations=add(out.operations,chunk.count);
    const auto direction=upload?0:1;const auto &direct=c.direct[direction];
    if(!supports(direct,p,out.operations))return out;
    out.direct_seconds=predict(direct,p,out.operations);out.direct_p90_relative_error=direct.p90_relative_error;
    if(!std::isfinite(out.direct_seconds)||out.direct_seconds<=0)return Price{};
    out.available=true;
    if(!out.operations||(!pool.reused&&!pool.cold_available))return out;
    unsigned index=0;while(index<3&&capacities[index]<p.bytes)++index;
    out.capacity=pool.reused?pool.capacity:capacities[index];
    index=0;while(index<4&&capacities[index]!=out.capacity)++index;
    if(index==4||out.capacity<p.bytes||out.capacity>pool.available_device_bytes/2)return out;
    out.workspace_bytes=2*out.capacity;out.chunks=p.chunks.size();
    const auto &compact=pool.reused?c.compact_reuse[direction]:c.compact_cold[direction];
    if(!supports(compact,p,out.operations)||!finite_nonnegative(base.wait_seconds))return out;
    out.compact_path_seconds=predict(compact,p,p.chunks.size());
    out.compact_seconds=out.compact_path_seconds+(pending?base.wait_seconds:0);
    out.compact_p90_relative_error=compact.p90_relative_error;
    if(!std::isfinite(out.compact_seconds)||out.compact_seconds<=0)return out;
    out.compact_available=true;
    out.selected=out.compact_seconds<out.direct_seconds*(1-c.minimum_relative_saving);
    return out;
}
// Bounded rent-or-buy metadata. Observed successful API submission is a pricing
// heuristic only: it does not certify completion, readability or future work.
// The same pure transition serves speculative planning and the locked pool.
namespace startup {
inline Key key(const Pricing &p,int device) {
    Key result;std::size_t i=0;
    auto integer=[&](std::uint64_t v){result.words[i++]=v;};
    auto real=[&](double v){std::uint64_t bits=0;std::memcpy(&bits,&v,sizeof(bits));integer(bits);};
    integer(static_cast<std::uint64_t>(device));integer(p.copy.version);integer(p.copy.valid);
    real(p.copy.minimum_relative_saving);real(p.base.wait_seconds);
    for(unsigned d=0;d<2;++d)for(const auto *r:{&p.copy.direct[d],&p.copy.compact_cold[d],&p.copy.compact_reuse[d]}) {
        for(double v:{r->batch_seconds,r->submission_seconds,r->byte_seconds,r->row_seconds,r->descriptor_seconds,
                r->median_relative_error,r->p90_relative_error,r->maximum_relative_error})real(v);
        for(auto v:{r->minimum_payload_bytes,r->maximum_payload_bytes,r->minimum_rows,r->maximum_rows,
                r->minimum_row_bytes,r->maximum_row_bytes,r->minimum_pitch_bytes,r->maximum_pitch_bytes,
                r->maximum_span_bytes,r->minimum_operations,r->maximum_operations,r->minimum_descriptor_work,
                r->maximum_descriptor_work,r->minimum_kernel_chunks,r->maximum_kernel_chunks})integer(v);
        integer(r->word_bytes_mask);integer(r->layout_combinations_mask);
    }
    return result;
}
inline void bind(State &s,const Pricing &p,int device) {
    const auto identity=key(p,device);
    if(s.bound && s.model==identity)return;
    ownership_changed(s); // Includes model ownership: A→B→A cannot accept an old observation.
    s.cap={};s.model=identity;s.bound=true;
    if(s.disabled || !valid_costs(p.copy) || !finite_nonnegative(p.base.wait_seconds))return;
    // A finite envelope upper bound caps metadata, not predicted speedup.
    // The artificial corner is never accepted as a measured geometry.
    for(unsigned bucket=0;bucket<4;++bucket)for(const auto &r:p.copy.compact_cold) {
        const auto bytes=std::min(capacities[bucket],r.maximum_payload_bytes);
        if(bytes<r.minimum_payload_bytes)continue;
        const double upper=r.batch_seconds+double(r.maximum_kernel_chunks)*r.submission_seconds+
            double(bytes)*r.byte_seconds+double(r.maximum_rows)*r.row_seconds+
            double(r.maximum_descriptor_work)*r.descriptor_seconds;
        if(std::isfinite(upper)&&upper>0)s.cap[bucket]=std::max(s.cap[bucket],upper);
    }
}
struct Decision : Offer { Price price{}; };
inline Decision choose(const Plan &plan,const Pricing &p,bool upload,bool pending,
                       const PoolState &pool,const State &s) {
    Decision out;out.price=compact::price(plan,p.copy,p.base,upload,pending,pool);
    out.selected=out.price.selected;out.capacity=out.price.capacity;
    if(s.disabled || !s.bound || s.model!=key(p,static_cast<int>(s.model.words[0])) || !out.price.compact_available || pool.reused || !pool.cold_available)return out;
    while(out.bucket>0 && capacities[out.bucket-1]>=out.price.capacity)--out.bucket;
    if(out.bucket==4 || capacities[out.bucket]!=out.price.capacity || s.cap[out.bucket]<=0)return out;
    const PoolState warm{out.price.capacity,pool.available_device_bytes,true,false};
    const auto reuse=compact::price(plan,p.copy,p.base,upload,pending,warm);
    if(!reuse.compact_available || !reuse.selected)return out;
    const auto saving=out.price.direct_seconds*(1-p.copy.minimum_relative_saving)-reuse.compact_seconds;
    if(!std::isfinite(saving)||saving<=0)return out;
    out.observation=std::min(saving,s.cap[out.bucket]);
    out.premium=std::max(0.,out.price.compact_seconds-reuse.compact_seconds);
    if(!out.price.selected && std::isfinite(out.premium) && s.credit[out.bucket]>=out.premium) {
        out.seed=true;out.selected=out.price.selected=true;
    }
    return out;
}

}
}}
#endif
