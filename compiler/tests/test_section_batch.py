"""Exact physical batching and calibrated admission grant no coherence credit."""
from pathlib import Path
import shutil
import subprocess

import pytest

SOURCE = r'''
#include "section_batch_cost.hpp"
#include "scoped_planning.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
using namespace fort_physical;
using namespace fort_physical::compact;
void roundtrip(const std::vector<CopyOperation> &ops,size_t bytes,unsigned words,size_t chunks) {
    const auto p=make_plan(ops,bytes,16*1024*1024);
    assert(p.word_bytes==words && p.chunks.size()==chunks);
    std::vector<unsigned char> original(bytes),expected(bytes,0xee),actual(bytes,0xee),packed(p.bytes),readback(p.bytes);
    for(size_t i=0;i<bytes;++i)original[i]=(17*i+31)%251;
    size_t prefix=0;
    for(const auto &op:ops) {
        for(size_t z=0;z<op.depth;++z)for(size_t y=0;y<op.height;++y) {
            const auto offset=op.offset+z*op.pitch*op.physical_height+y*op.pitch;
            std::memcpy(expected.data()+offset,original.data()+offset,op.width);
            std::memcpy(packed.data()+prefix,original.data()+offset,op.width);prefix+=op.width;
        }
    }
    assert(prefix==p.bytes);
    // Interpret every lowered kernel address; compare against independent
    // original z/y row copies, including all bytes outside the requested set.
    for(const auto &chunk:p.chunks)for(size_t n=0;n<chunk.count;++n) {
        const auto &d=chunk.ops[n];const auto width=d.width/words;
        for(size_t i=0;i<width*d.height*d.depth;++i) {
            const auto row=i/width;
            const auto offset=d.offset+(row/d.height)*d.pitch*d.physical_height+(row%d.height)*d.pitch+(i%width)*words;
            std::memcpy(actual.data()+offset,packed.data()+d.packed_offset+i*words,words);
            std::memcpy(readback.data()+d.packed_offset+i*words,original.data()+offset,words);
        }
    }
    assert(actual==expected && readback==packed);
}
fort_scope_copy_costs_v3 costs() {
    fort_scope_copy_costs_v3 c{};c.version=FORT_SCOPE_COPY_COST_ABI_VERSION;c.valid=1;c.minimum_relative_saving=.5;
    for(unsigned d=0;d<2;++d)for(auto *r:{&c.direct[d],&c.compact_cold[d],&c.compact_reuse[d]}) {
        const bool direct=r==&c.direct[d];
        r->submission_seconds=direct?1:0;r->batch_seconds=direct?0:.001;r->byte_seconds=1e-9;
        r->minimum_payload_bytes=r->minimum_rows=r->minimum_row_bytes=1;
        r->maximum_payload_bytes=r->maximum_rows=r->maximum_row_bytes=16u<<20;
        r->minimum_pitch_bytes=1;r->maximum_pitch_bytes=16u<<20;r->maximum_span_bytes=1u<<30;
        r->minimum_operations=direct?1:2;r->maximum_operations=256;
        r->minimum_descriptor_work=1;r->maximum_descriptor_work=1u<<30;
        r->minimum_kernel_chunks=1;r->maximum_kernel_chunks=6;
        r->word_bytes_mask=7;r->layout_combinations_mask=254;
    }
    return c;
}
int main(int argc,char **argv) try {
    assert(argc==2);const std::string mode=argv[1];
    if(mode=="word64")roundtrip({{0,16,1,1,16,1},{64,16,1,1,16,1}},96,8,1);
    else if(mode=="word32")roundtrip({{4,12,1,1,28,1},{68,12,1,1,28,1}},96,4,1);
    else if(mode=="byte")roundtrip({{1,3,1,1,11,1},{17,3,1,1,11,1}},32,1,1);
    else if(mode=="pitched")roundtrip({{0,8,3,2,32,5},{8,8,3,2,32,5}},256,8,1);
    else if(mode=="chunks") {
        std::vector<CopyOperation> ops;for(size_t i=0;i<97;++i)ops.push_back({16*i,8,1,1,8,1});
        roundtrip(ops,97*16,8,3);
    } else if(mode=="empty") {
        const auto p=make_plan({},0,0);assert(!p.bytes && p.chunks.empty());
    } else if(mode=="reject") {
        const auto maximum=std::numeric_limits<size_t>::max();
        const std::vector<std::vector<CopyOperation>> invalid={
            {{0,8,2,1,16,2},{20,8,1,1,8,1}},{{24,16,1,1,16,1}},
            {{0,0,1,1,8,1}},{{0,9,1,1,8,1}},{{0,8,3,2,8,2}},
            {{maximum-3,8,1,1,8,1}},{{0,8,maximum,1,8,1}},
            std::vector<CopyOperation>(257,{0,8,1,1,8,1})};
        for(const auto &ops:invalid) {bool rejected=false;try {make_plan(ops,32,32);}catch(const Ineligible &) {rejected=true;}assert(rejected);}
        bool rejected=false;try {make_plan({{0,8,1,1,8,1}},8,7);}catch(const Ineligible &) {rejected=true;}assert(rejected);
    } else {
        auto c=costs();fort_scope_plan_costs base{};base.valid=1;
        const auto p=make_plan({{0,8,1,1,8,1},{32,8,1,1,8,1}},64,64);
        PoolState pool{0,32*1024*1024,false};
        if(mode=="model" || mode=="model_budget" || mode=="model_direct_price" || mode=="model_unknown" || mode=="model_oneop" || mode=="model_pool" || mode=="model_idle") {
            namespace planning=fort_scoped::planning;
            planning::Inputs input;input.transfer_mode=FORT_SCOPE_TRANSFERS_AUTO;
            input.device_budget=mode=="model_budget" ? 512 : 1024*1024;
            if(mode=="model_pool")input.copy_pool_available_bytes=0;
            if(mode=="model_idle")input.copy_pool_cold_available=false;
            input.copy_costs=Pricing{c,base};
            if(mode=="model_direct_price")input.copy_costs->copy.compact_cold[0].batch_seconds=3;
            planning::Resource resource;resource.handle=1;resource.element_bytes=8;resource.bytes=512;
            resource.extents={64};resource.allocated=true;resource.initialized=resource.host_current={{{0},{64}}};
            planning::detail::State state;state.resources={resource};state.allocated=512;
            base.h2d_latency=.01;base.h2d_bandwidth=1e9;
            fort_scoped::coherence::Region edges={{{0},{3}},{{61},{64}}};
            if(mode=="model_unknown") {
                input.copy_costs->copy.direct[0].minimum_payload_bytes=64;
                bool unknown=false;try{planning::detail::ensure(state,state.resources[0],edges,true,input,base);}
                catch(const planning::detail::Unavailable &e){unknown=std::string(e.reason)=="physical_copy_estimate_unavailable";}
                assert(unknown && !state.count.uploads);return 0;
            }
            if(mode=="model_oneop")edges.resize(1);
            planning::detail::ensure(state,state.resources[0],edges,true,input,base);
            assert(state.count.upload_bytes==(mode=="model_oneop"?24u:48u));
            if(mode=="model") {
                assert(state.count.uploads==1 && !state.pending && state.upload_readers.empty());
                assert(state.compact_capacity==256*1024 && state.count.peak_device_bytes==512+512*1024);
            } else {const auto count=mode=="model_oneop"?1u:2u;assert(state.count.uploads==count && state.pending && state.upload_readers.size()==count);}
            if(mode=="model_oneop")assert(state.time>1 && state.time<1.001);
            if(mode=="model_pool" || mode=="model_idle")assert(state.time>2 && state.time<2.001);
            if(mode=="model_direct_price")assert(state.time>2 && state.time<2.001);
            const auto before=state.count.uploads;
            planning::detail::ensure(state,state.resources[0],edges,true,input,base);
            planning::detail::ensure(state,state.resources[0],edges,false,input,base);
            assert(state.count.uploads==before && !state.count.downloads);
        } else if(mode=="price") {
            const auto q=price(p,c,base,true,false,pool);
            assert(q.available && q.selected && q.operations==2 && q.chunks==1 && q.workspace_bytes==512*1024);
        } else if(mode=="cold_reuse") {
            pool.cold_available=false;
            assert(price(p,c,base,true,false,pool).available && !price(p,c,base,true,false,pool).compact_available);
            pool.cold_available=true;
            c.compact_cold[0].batch_seconds=3;
            assert(!price(p,c,base,true,false,pool).selected);
            pool.capacity=256*1024;pool.reused=true;pool.cold_available=false;
            assert(price(p,c,base,true,false,pool).selected);
        } else if(mode=="single_operation") {
            const auto single=make_plan({{8,24,13,7,176,17}},20944,16*1024*1024);
            roundtrip({{8,24,13,7,176,17}},20944,8,1);
            // Multi-operation measurements cannot price a new one-op shape.
            assert(!price(single,c,base,false,false,pool).compact_available);
            for(auto *r:{&c.compact_cold[1],&c.compact_reuse[1]})r->minimum_operations=1;
            const auto q=price(single,c,base,false,false,pool);
            assert(q.available && q.selected && q.operations==1 && q.chunks==1 && q.workspace_bytes==512*1024);
            pool.available_device_bytes=256*1024;
            assert(!price(single,c,base,false,false,pool).compact_available);
            c.version=2;assert(!valid_costs(c));
        } else if(mode=="coverage") {
            c.compact_cold[0].minimum_payload_bytes=17;assert(price(p,c,base,true,false,pool).available && !price(p,c,base,true,false,pool).compact_available);
            c=costs();c.compact_cold[0].maximum_pitch_bytes=7;assert(price(p,c,base,true,false,pool).available && !price(p,c,base,true,false,pool).compact_available);
            c=costs();const auto single=make_plan({{0,8,1,1,8,1}},8,8);
            assert(price(single,c,base,true,false,pool).available && !price(single,c,base,true,false,pool).selected);
        } else if(mode=="rate_coverage") {
            // A complete-path fit cannot extend its observed geometry envelope.
            c.direct[0].maximum_payload_bytes=c.direct[0].maximum_row_bytes=7;assert(valid_costs(c));
            assert(!price(p,c,base,true,false,pool).available);
            c=costs();c.direct[0].minimum_pitch_bytes=9;
            assert(!price(p,c,base,true,false,pool).available);
            c=costs();c.compact_cold[0].minimum_rows=3;
            assert(price(p,c,base,true,false,pool).available && !price(p,c,base,true,false,pool).compact_available);
            c=costs();c.compact_cold[0].minimum_operations=3;
            assert(price(p,c,base,true,false,pool).available && !price(p,c,base,true,false,pool).compact_available);
            c=costs();c.compact_cold[1].maximum_operations=2;
            assert(price(p,c,base,true,false,pool).selected);
        } else if(mode=="invalid_costs") {
            c.compact_cold[0].row_seconds=-1;assert(!valid_costs(c));
            assert(!price(p,c,base,true,false,pool).available);
            c=costs();c.direct[0].batch_seconds=std::numeric_limits<double>::infinity();assert(!valid_costs(c));
            c=costs();c.version=1;assert(!valid_costs(c));
            c=costs();c.direct[0].byte_seconds=std::numeric_limits<double>::max();assert(!valid_costs(c));
        } else if(mode=="budget") {
            pool.available_device_bytes=256*1024;
            assert(price(p,c,base,true,false,pool).available && !price(p,c,base,true,false,pool).selected);
            pool.available_device_bytes=512*1024;pool.capacity=1024*1024;pool.reused=true;
            assert(price(p,c,base,true,false,pool).available && !price(p,c,base,true,false,pool).selected);
            pool.available_device_bytes=8*1024*1024;pool.capacity=4*1024*1024;
            const auto oversized=price(p,c,base,true,false,pool);
            assert(oversized.available && oversized.selected && oversized.capacity==4*1024*1024);
            assert(oversized.workspace_bytes==8*1024*1024);
        } else if(mode=="dispersion") {
            const auto expected=price(p,c,base,true,false,pool);
            c.compact_cold[0].median_relative_error=100;
            c.compact_cold[0].p90_relative_error=500;
            c.compact_cold[0].maximum_relative_error=10000;
            auto q=price(p,c,base,true,false,pool);
            assert(q.available && q.selected==expected.selected && q.compact_seconds==expected.compact_seconds);
            assert(q.compact_p90_relative_error==500);
            c.direct[0].median_relative_error=1;c.direct[0].p90_relative_error=2;c.direct[0].maximum_relative_error=4;
            q=price(p,c,base,true,false,pool);
            assert(q.available && q.selected==expected.selected && q.direct_seconds==expected.direct_seconds);
            assert(q.direct_p90_relative_error==2); // Diagnostic errors never gate profitability.
        } else if(mode=="margin") {
            auto &d=c.direct[0];auto &k=c.compact_cold[0];
            d.batch_seconds=2;d.submission_seconds=d.byte_seconds=0;
            k.batch_seconds=1;k.byte_seconds=0;
            auto q=price(p,c,base,true,false,pool);
            assert(q.available && q.compact_available && q.direct_seconds==2 && q.compact_seconds==1 && !q.selected);
            k.batch_seconds=std::nextafter(1.,0.);assert(price(p,c,base,true,false,pool).selected);
            k.batch_seconds=std::nextafter(1.,2.);assert(!price(p,c,base,true,false,pool).selected);
            c.minimum_relative_saving=.49;assert(!valid_costs(c));
        } else if(mode=="complete_terms") {
            auto &d=c.direct[0];d.batch_seconds=1;d.submission_seconds=2;d.byte_seconds=3;
            d.row_seconds=4;d.descriptor_seconds=5;
            c.compact_cold[0]=d;c.compact_cold[0].minimum_operations=2;
            const auto q=price(p,c,base,true,false,pool);
            assert(q.available && q.compact_available);
            assert(q.direct_seconds==1+2*2+3*p.bytes+4*p.rows+5*p.work);
            assert(q.compact_seconds==1+2*p.chunks.size()+3*p.bytes+4*p.rows+5*p.work);
        } else if(mode=="support_envelope") {
            for(unsigned test=0;test<17;++test) {
                auto limited=c;auto &r=limited.direct[0];
                switch(test) {
                    case 0:r.minimum_payload_bytes=17;break;
                    case 1:r.maximum_payload_bytes=r.maximum_row_bytes=15;break;
                    case 2:r.minimum_rows=3;break;
                    case 3:r.maximum_rows=1;break;
                    case 4:r.minimum_row_bytes=9;break;
                    case 5:r.maximum_row_bytes=7;break;
                    case 6:r.minimum_pitch_bytes=9;break;
                    case 7:r.maximum_pitch_bytes=7;break;
                    case 8:r.maximum_payload_bytes=r.maximum_row_bytes=16;r.maximum_span_bytes=39;break;
                    case 9:r.minimum_operations=3;break;
                    case 10:r.maximum_operations=1;break;
                    case 11:r.minimum_descriptor_work=p.work+1;break;
                    case 12:r.maximum_descriptor_work=p.work-1;break;
                    case 13:r.minimum_kernel_chunks=2;break;
                    case 14:r.word_bytes_mask=3;break;
                    case 15:r.layout_combinations_mask=4;break;
                    case 16:r.maximum_kernel_chunks=2;break;
                }
                assert(valid_costs(limited));
                if(test==16) {
                    std::vector<CopyOperation> ops;for(size_t i=0;i<97;++i)ops.push_back({16*i,8,1,1,8,1});
                    const auto chunks=make_plan(ops,97*16,97*16);
                    assert(!price(chunks,limited,base,true,false,pool).available);
                } else assert(!price(p,limited,base,true,false,pool).available);
            }
            c.compact_cold[0].word_bytes_mask=3;
            assert(price(p,c,base,true,false,pool).available && !price(p,c,base,true,false,pool).compact_available);
            c=costs();c.compact_cold[0].layout_combinations_mask=4;
            assert(price(p,c,base,true,false,pool).available && !price(p,c,base,true,false,pool).compact_available);
        } else if(mode=="model_wait_accounting" || mode=="upload_reader_wait" || mode=="complete_compact_wait") {
            namespace planning=fort_scoped::planning;
            planning::Inputs input;input.transfer_mode=FORT_SCOPE_TRANSFERS_AUTO;
            input.device_budget=mode=="complete_compact_wait"?1024*1024:512;
            base.wait_seconds=7;base.cpu_flops=base.cpu_bandwidth=base.gpu_flops=base.gpu_bandwidth=1;
            base.h2d_bandwidth=base.d2h_bandwidth=1e9;
            if(mode=="complete_compact_wait")c.direct[0].submission_seconds=100;
            input.copy_costs=Pricing{c,base};
            planning::Resource resource;resource.handle=1;resource.element_bytes=8;resource.bytes=512;
            resource.extents={64};resource.allocated=true;resource.initialized=resource.host_current={{{0},{64}}};
            planning::detail::State state;state.resources={resource};state.index={{1,0}};state.allocated=512;
            fort_scoped::coherence::Region edges={{{0},{3}},{{61},{64}}};
            if(mode=="complete_compact_wait")planning::detail::queued_work(state);
            planning::detail::ensure(state,state.resources[0],edges,true,input,base);
            assert(state.count.upload_bytes==48 && !state.count.downloads);
            // Legacy contiguous-fit dispersion is not a multiplier for a V2 path.
            assert(state.h2d_seconds==0 && state.d2h_seconds==0);
            if(mode=="complete_compact_wait") {
                const auto actual=make_plan({{0,24,1,1,24,1},{488,24,1,1,24,1}},512,512);
                const auto expected=price(actual,c,base,true,true,{0,1024*1024-512,false});
                assert(expected.selected && expected.compact_seconds==expected.compact_path_seconds+7);
                assert(state.time==expected.compact_seconds && state.count.waits==1);
                assert(state.count.uploads==1 && !state.pending && !state.completion_already_priced);
                assert(state.upload_readers.empty());
            } else {
                assert(state.pending && state.completion_already_priced && state.upload_readers.size()==2);
                assert(state.count.uploads==2 && state.count.waits==0);
                const auto before=state.time;
                if(mode=="upload_reader_wait") {
                    planning::Operation native;native.kind=FORT_SCOPE_PLAN_NATIVE;native.flops=1;
                    fort_scoped::coherence::Effects writes;writes.writes=writes.overwrites=edges;
                    native.bindings={{1,writes}};uint64_t work=0;
                    planning::detail::execute(state,native,false,input,base,work);
                    assert(state.time==before+1 && state.count.waits==1 && state.upload_readers.empty());
                    assert(state.pending && !state.completion_already_priced && work==1);
                } else {
                    planning::detail::wait(state,base);
                    assert(state.time==before && state.count.waits==1 && !state.pending);
                    assert(!state.completion_already_priced && state.upload_readers.empty());
                    planning::detail::wait(state,base);assert(state.time==before && state.count.waits==1);
                    // An actual modeled new GPU operation cannot reuse the old completion credit.
                    state.completion_already_priced=true;
                    planning::Operation gpu;gpu.kind=FORT_SCOPE_PLAN_WORKER;gpu.unit=17;gpu.flops=1;gpu.gpu_available=true;
                    uint64_t work=0;planning::detail::execute(state,gpu,true,input,base,work);
                    assert(state.pending && !state.completion_already_priced && state.count.launches==1);
                    planning::detail::wait(state,base);
                    assert(state.time==before+1+7 && state.count.waits==2 && !state.pending);
                    // Device allocation is another independently queued boundary.
                    state.completion_already_priced=true;state.resources[0].allocated=false;state.allocated=0;
                    input.device_budget=1024;base.max_allocation_bytes=512;
                    planning::detail::allocate(state,state.resources[0],input,base);
                    assert(state.pending && !state.completion_already_priced && state.count.allocations==1);
                }
            }
        } else if(mode=="v2_direct_ignores_legacy_price") {
            namespace planning=fort_scoped::planning;
            for(bool overflow:{false,true}) {
                planning::Inputs input;input.transfer_mode=FORT_SCOPE_TRANSFERS_AUTO;input.device_budget=512;
                base.h2d_bandwidth=overflow?1e9:0;
                base.h2d_latency=overflow?std::numeric_limits<double>::max():0;
                input.copy_costs=Pricing{c,base};
                auto fresh=[] {
                    planning::Resource r;r.handle=1;r.element_bytes=8;r.bytes=512;r.extents={64};r.allocated=true;
                    r.initialized=r.host_current={{{0},{64}}};planning::detail::State state;
                    state.resources={r};state.index={{1,0}};state.allocated=512;return state;
                };
                const fort_scoped::coherence::Region edges={{{0},{3}},{{61},{64}}};
                auto state=fresh();planning::detail::ensure(state,state.resources[0],edges,true,input,base);
                assert(state.count.uploads==2 && state.count.upload_bytes==48 && state.pending);
                assert(state.completion_already_priced && state.upload_readers.size()==2);
                assert(state.time==2+48e-9 && state.h2d_seconds==0 && state.d2h_seconds==0);
                // Absent V2 retains the old direct-price validation and rejection.
                input.copy_costs.reset();auto legacy=fresh();bool rejected=false;
                try {planning::detail::ensure(legacy,legacy.resources[0],edges,true,input,base);}
                catch(const planning::detail::Unavailable &failure) {
                    rejected=std::string(failure.reason)==(overflow?"arithmetic_overflow":"invalid_cost_estimate");
                }
                assert(rejected);
            }
        } else assert(false);
    }
} catch(const fort_scoped::planning::detail::Unavailable &failure) {
    std::fprintf(stderr,"unavailable: %s\n",failure.reason);return 1;
}
'''


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    directory = tmp_path_factory.mktemp("section_batch")
    source = directory / "check.cpp"
    source.write_text(SOURCE)
    binary = directory / "check"
    compiler = shutil.which("g++")
    assert compiler is not None
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    result = subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                             "-I" + str(runtime), str(source), "-o", str(binary)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.parametrize("mode", ["word64", "word32", "byte", "pitched", "chunks", "empty", "reject",
                                 "price", "cold_reuse", "single_operation", "coverage", "rate_coverage", "invalid_costs", "budget", "dispersion",
                                 "margin", "complete_terms", "support_envelope",
                                 "model_wait_accounting", "upload_reader_wait", "complete_compact_wait", "v2_direct_ignores_legacy_price",
                                 "model", "model_budget", "model_direct_price", "model_unknown", "model_oneop", "model_pool", "model_idle"])
def test_exact_batch_and_bounded_offline_selection(executable, mode):
    result = subprocess.run([str(executable), mode], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
