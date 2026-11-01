"""Rent-or-buy is bounded pricing metadata, never completion/data authority."""
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.tests.test_staging_pool import MOCK

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"

SOURCE = r'''
#include "section_batch_cost.hpp"
#include "scoped_planning.hpp"
#include "scoped_memo.hpp"
#include "staging.hpp"
#include <cassert>
#include <atomic>
#include <string>
#include <thread>
using namespace fort_physical;
using namespace fort_physical::compact;
Pricing prices() {
    Pricing p{};p.copy.version=2;p.copy.valid=1;p.copy.minimum_relative_saving=.5;
    p.base.version=2;p.base.valid=1;p.base.max_allocation_bytes=1u<<30;
    p.base.cpu_flops=p.base.gpu_flops=p.base.cpu_bandwidth=p.base.gpu_bandwidth=1;
    p.base.h2d_bandwidth=p.base.d2h_bandwidth=1;
    p.base.create_seconds=p.base.register_seconds=p.base.host_access_seconds=p.base.device_access_seconds=
        p.base.gpu_setup_seconds=p.base.cold_driver_startup_seconds=p.base.allocation_seconds=p.base.release_seconds=
        p.base.wait_seconds=p.base.launch_enqueue_seconds=p.base.planning_operation_seconds=1e-100;
    for(unsigned d=0;d<2;++d)for(auto *r:{&p.copy.direct[d],&p.copy.compact_cold[d],&p.copy.compact_reuse[d]}) {
        const bool direct=r==&p.copy.direct[d];
        r->batch_seconds=direct?10:r==&p.copy.compact_cold[d]?9:1;
        r->minimum_payload_bytes=r->minimum_rows=r->minimum_row_bytes=r->minimum_pitch_bytes=1;
        r->maximum_payload_bytes=r->maximum_rows=r->maximum_row_bytes=r->maximum_pitch_bytes=16u<<20;
        r->maximum_span_bytes=1u<<30;r->minimum_operations=direct?1:2;r->maximum_operations=256;
        r->minimum_descriptor_work=1;r->maximum_descriptor_work=1u<<30;
        r->minimum_kernel_chunks=1;r->maximum_kernel_chunks=6;
        r->word_bytes_mask=7;r->layout_combinations_mask=254;
    }
    assert(valid_costs(p.copy));return p;
}
Plan geometry() {return make_plan({{0,8,1,1,8,1},{32,8,1,1,8,1}},64,64);}
startup::State earned(const Pricing &p,const Plan &plan,const PoolState &pool) {
    startup::State s;startup::bind(s,p,0);
    for(int i=0;i<2;++i) {
        const auto before=s;const auto q=startup::choose(plan,p,true,false,pool,s);
        assert(!q.selected && !q.seed && q.observation==4 && q.premium==8);
        assert(startup::observe(s,before,q));
    }
    assert(s.credit[0]==8);return s;
}
int main(int argc,char **argv) {
    assert(argc==2);const std::string mode=argv[1];auto p=prices();const auto plan=geometry();
    const PoolState pool{0,32u<<20,false,true};
    if(mode=="earn") {
        startup::State s;startup::bind(s,p,0);const auto revision=s.revision;
        auto q=startup::choose(plan,p,true,false,pool,s);assert(!q.selected && s.credit[0]==0);
        auto before=s;assert(startup::observe(s,before,q));assert(s.credit[0]==4 && s.revision>revision);
        q=startup::choose(plan,p,true,false,pool,s);assert(!q.seed && !q.selected);
        before=s;startup::observe(s,before,q);q=startup::choose(plan,p,true,false,pool,s);
        assert(q.seed && q.selected && q.price.compact_path_seconds==9 && q.price.compact_seconds==9);
        before=s;assert(startup::reserve(s,before,q));assert(s.credit[0]==0 && s.generation>before.generation);
        assert(!startup::reserve(s,before,q));
    } else if(mode=="saturation") {
        startup::State s;startup::bind(s,p,0);auto before=s;const auto q=startup::choose(plan,p,true,false,pool,s);
        for(int i=0;i<10000;++i)assert(startup::observe(s,before,q));
        assert(s.credit[0]==9 && s.cap[0]==9 && std::isfinite(s.credit[0]));
        for(unsigned i=1;i<4;++i)assert(s.credit[i]==0 && s.cap[i]==9);
        const auto after=s;startup::bind(s,p,0);assert(startup::same(s,after));
    } else if(mode=="keys") {
        auto s=earned(p,plan,pool);const auto original=s.model;
        startup::bind(s,p,1);assert(s.model!=original && s.credit[0]==0);
        s=earned(p,plan,pool);auto changed=p;changed.base.wait_seconds=.25;
        startup::bind(s,changed,0);assert(s.model!=original && s.credit[0]==0);
        s=earned(p,plan,pool);changed=p;changed.copy.direct[1].maximum_relative_error=1;
        startup::bind(s,changed,0);assert(s.model!=original && s.credit[0]==0);
        assert(!startup::choose(plan,p,true,false,pool,s).seed);
        s=earned(p,plan,pool);const auto earlier=s;
        auto offer=startup::choose(plan,p,true,false,pool,s);offer.selected=offer.seed=false;
        startup::bind(s,changed,0);startup::bind(s,p,0);
        assert(s.model==earlier.model && s.generation!=earlier.generation);
        assert(!startup::observe(s,earlier,offer)); // No model-identity ABA.
    } else if(mode=="eligibility") {
        auto s=earned(p,plan,pool);
        for(const auto bad:{PoolState{0,0,false,true},PoolState{0,32u<<20,false,false}}) {
            const auto q=startup::choose(plan,p,true,false,bad,s);assert(!q.seed && !q.observation);
        }
        const auto one=make_plan({{0,8,1,1,8,1}},64,64);
        assert(!startup::choose(one,p,true,false,pool,s).observation);
        auto unsupported=p;unsupported.copy.compact_reuse[0].minimum_payload_bytes=17;
        startup::bind(s,unsupported,0);assert(!startup::choose(plan,unsupported,true,false,pool,s).observation);
        startup::bind(s,p,0);const auto warm=startup::choose(plan,p,true,false,{4u<<20,8u<<20,true,false},s);
        assert(warm.selected && !warm.seed && warm.price.workspace_bytes==(8u<<20));
    } else if(mode=="stale") {
        auto s=earned(p,plan,pool);const auto before=s;auto q=startup::choose(plan,p,true,false,pool,s);
        startup::ownership_changed(s);assert(!startup::reserve(s,before,q) && !startup::observe(s,before,q));
        s=earned(p,plan,pool);const auto same_generation=s.generation;
        startup::touch(s);assert(s.credit[0]==8 && s.generation==same_generation);
        assert(!startup::reserve(s,before,q));
    } else if(mode=="overflow") {
        auto s=earned(p,plan,pool);s.revision=std::numeric_limits<uint64_t>::max();startup::touch(s);
        assert(s.disabled && !s.credit[0] && !startup::choose(plan,p,true,false,pool,s).seed);
        s=earned(p,plan,pool);s.generation=std::numeric_limits<uint64_t>::max();startup::ownership_changed(s);
        assert(s.disabled && !s.credit[0]);
    } else if(mode=="planner") {
        namespace planning=fort_scoped::planning;
        planning::Inputs input;input.copy_costs=p;input.transfer_mode=FORT_SCOPE_TRANSFERS_AUTO;
        input.device_budget=1u<<20;input.copy_device=0;
        planning::Resource b;b.handle=1;b.element_bytes=8;b.bytes=64;b.extents={8};b.allocated=true;
        planning::detail::State s;s.allocated=64;fort_scope_plan_costs base{};
        const fort_scoped::coherence::Region parts={{{0},{1}},{{4},{5}}};
        for(int i=0;i<3;++i)planning::detail::copy_region(s,b,parts,true,input,base,nullptr,0,"test");
        assert(s.time==29 && s.count.uploads==5 && s.count.upload_bytes==48 && s.compact_capacity==capacities[0]);
        assert(!s.copy_startup.credit[0] && !s.pending);
        planning::detail::copy_region(s,b,parts,true,input,base,nullptr,0,"test");
        assert(s.time==30 && s.count.uploads==6);
        // Each speculative alternative owns an independent state; input is inert.
        assert(!input.copy_startup.bound);
    } else if(mode=="pending") {
        p.base.wait_seconds=2;auto s=earned(p,plan,pool);
        auto q=startup::choose(plan,p,true,true,pool,s);
        assert(q.seed && q.premium==8 && q.price.compact_seconds==11 && q.price.compact_path_seconds==9);
        startup::State zero;startup::bind(zero,p,0);q=startup::choose(plan,p,true,true,pool,zero);
        assert(q.observation==2 && !q.seed);
    } else if(mode=="memo") {
        namespace planning=fort_scoped::planning;
        planning::Inputs input;input.graph_mode=true;input.definitions_validated=true;input.copy_costs=p;
        const auto base=p.base;
        auto key=fort_scoped::memo::make_key(input,base,{});assert(key);
        for(int i=0;i<8;++i) {
            auto x=input;
            switch(i) {case 0:x.copy_device=1;break;case 1:x.copy_startup.credit[0]=1;break;
            case 2:x.copy_startup.revision++;break;case 3:x.copy_startup.generation++;break;
            case 4:x.copy_startup.model.words[18]++;break;case 5:x.copy_startup.cap[3]=1;break;
            case 6:x.copy_startup.disabled=true;break;case 7:x.copy_completion_priced=true;break;}
            auto changed=fort_scoped::memo::make_key(x,base,{});assert(changed && changed->words!=key->words);
        }
    } else {
        using namespace fort_staging;auto &b=budget_state();b.startup=earned(p,plan,pool);
        const auto before=b.startup;const auto decision=startup::choose(plan,p,true,false,pool,before);
        assert(decision.seed);size_t freed=0;
        if(mode=="pool_race") {
            std::atomic<int> ready{0};std::atomic<bool> go{false};Result results[2];
            fake::probe=[&]{assert(b.mutex.try_lock());b.mutex.unlock();};
            auto run=[&](int i){++ready;while(!go.load())std::this_thread::yield();
                results[i]=acquire(decision.capacity,Role::Compact,false,device_limit,true,&before,&decision);};
            std::thread a(run,0),c(run,1);while(ready.load()!=2)std::this_thread::yield();go=true;a.join();c.join();
            fake::probe={};const auto winner=bool(results[0].slots)?0:1;
            assert(results[winner].startup_paid && !results[1-winner].slots && results[1-winner].stale_startup);
            assert(!b.startup.credit[0] && b.startup.generation>before.generation);
            assert(release(std::move(results[winner].slots),freed)==0);
        } else if(mode=="pool_budget") {
            auto r=acquire(decision.capacity,Role::Compact,false,1,true,&before,&decision);
            assert(r.exhausted && !r.slots && startup::same(b.startup,before));
        } else if(mode=="pool_allocation") {
            for(int failure=1;failure<=8;++failure) {
                b.startup=earned(p,plan,pool);const auto current=b.startup;
                fake::fail_at=failure;fake::allocations=0;
                auto r=acquire(decision.capacity,Role::Compact,false,device_limit,true,&current,&decision);
                assert(r.startup_paid && !r.slots && r.status==cudaErrorMemoryAllocation);
                assert(!b.startup.credit[0] && !b.reserved && !b.device_reserved && fake::memory.empty());
            }
            fake::fail_at=0;
        } else if(mode=="pool_changed") {
            auto other=acquire(256,Role::Staging,false);assert(other.slots);
            assert(release(std::move(other.slots),freed)==0);
            auto r=acquire(decision.capacity,Role::Compact,false,device_limit,true,&before,&decision);
            assert(r.exhausted && !r.slots && r.stale_startup);
            assert(b.idle && b.idle->role==Role::Staging && b.idle->capacity==256);
        } else assert(false);
        assert(trim_cache(freed)==0 && fake::memory.empty() && !b.reserved && !b.device_reserved);
    }
}
'''


@pytest.fixture(scope="module")
def startup_program(tmp_path_factory):
    compiler = shutil.which("g++")
    assert compiler, "required host C++ compiler unavailable"
    folder = tmp_path_factory.mktemp("copy_startup")
    mock = MOCK.replace("#include <algorithm>", "#include <algorithm>\n#include <atomic>\n#include <functional>")
    mock = mock.replace("events_waited=0,queries=0;", "events_waited=0; inline std::atomic<int> queries{0};")
    mock = mock.replace("inline bool fail() {", "inline std::function<void()> probe;\ninline bool fail() { if(probe)probe();")
    (folder / "cuda_runtime.h").write_text(mock)
    (folder / "main.cpp").write_text(SOURCE)
    command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
               "-D__CUDACC__", "-DFORT_SCOPE_CPU_TEST", "-I"+str(folder), "-I"+str(RUNTIME),
               str(folder / "main.cpp"), "-o", str(folder / "run")]
    result = subprocess.run(command, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    return folder / "run"


@pytest.mark.parametrize("mode", ["earn", "saturation", "keys", "eligibility", "stale", "overflow",
                                  "planner", "pending", "memo", "pool_race", "pool_budget",
                                  "pool_allocation", "pool_changed"])
def test_bounded_pricing_transition_and_real_pool_reservation(startup_program, mode):
    result = subprocess.run([str(startup_program), mode], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr


RUNTIME_SOURCE = r'''
#define FORT_SCOPE_CPU_TEST
#include "scoped_runtime.cu"
#include <cassert>
int main(int argc,char **argv) {
    assert(argc==2);const std::string mode=argv[1];fort_scope_t handle=0;
    assert(!fort_scope_create(0,&handle));assert(!fort_scope_set_transfers(handle,FORT_SCOPE_TRANSFERS_AUTO));
    auto p=prices();assert(!fort_scope_set_copy_costs_v2(handle,&p.copy,&p.base,1));
    double host[64];for(auto &x:host)x=-77;
    const size_t extent=64;const int64_t lower=-17;
    fort_scope_layout layout{1,2,8,host,&extent,&lower,1};fort_buffer_t buffer_handle=0;
    assert(!fort_scope_register(handle,1,1,&layout,1,&buffer_handle));
    size_t lo[2]={0,61},hi[2]={3,64};fort_scope_section sections[2]={{lo,hi},{lo+1,hi+1}};
    fort_scope_access reads{},writes{};reads.reads=sections;reads.read_count=2;
    writes.writes=writes.overwrites=sections;writes.write_count=writes.overwrite_count=2;
    auto copy=[&](int round) {
        assert(!fort_scope_host_begin(handle,buffer_handle,&writes));
        for(int i: {0,1,2,61,62,63})host[i]=100*round+i;
        assert(!fort_scope_host_end(handle,buffer_handle));
        void *device=nullptr;const auto status=fort_scope_device_begin(handle,buffer_handle,&reads,&device);
        if(status)return status;
        for(int i: {0,1,2,61,62,63})assert(static_cast<double*>(device)[i]==host[i]);
        return fort_scope_device_end(handle,buffer_handle);
    };
    const auto c=lookup(handle);auto &state=c->residency->compact_reference_startup;
    if(mode=="completion_metadata") {
        auto &b=*c->buffers.at(buffer_handle);
        const Region parts={Box{{0},{3}},Box{{61},{64}}};
        c->copy_costs->base.wait_seconds=2;
        copy_region(*c,b,parts,true);
        assert(c->pending && c->residency->copy_completion_priced && state.credit[0]==4);
        copy_region(*c,b,parts,true);
        assert(c->pending && c->residency->copy_completion_priced && state.credit[0]==8);
        mark_context_pending(*c);assert(!c->residency->copy_completion_priced);
        wait(*c);assert(!c->pending && !c->residency->copy_completion_priced);
        assert(!fort_scope_close(handle));return 0;
    }
    if(mode=="partial_failure") {
        setenv("FORT_SCOPE_TEST_FAIL_DIRECT_BATCH_AFTER","1",1);
        assert(copy(1)==FORT_SCOPE_EXECUTION);unsetenv("FORT_SCOPE_TEST_FAIL_DIRECT_BATCH_AFTER");
        assert(c->poisoned && !state.credit[0] && c->stats.uploads==1 && !c->residency->compact_reference_capacity);
        void *device=nullptr;assert(fort_scope_device_begin(handle,buffer_handle,&reads,&device)==FORT_SCOPE_EXECUTION);
        assert(c->stats.uploads==1);assert(!fort_scope_abandon(handle));return 0;
    }
    assert(!copy(1) && state.credit[0]==4 && c->stats.uploads==2);
    assert(!copy(2) && state.credit[0]==8 && c->stats.uploads==4);
    if(mode=="allocation_miss") {
        setenv("FORT_SCOPE_TEST_FAIL_COMPACT_ALLOC","1",1);assert(!copy(3));
        unsetenv("FORT_SCOPE_TEST_FAIL_COMPACT_ALLOC");
        assert(state.credit[0]==8 && c->stats.uploads==6 && !c->residency->compact_reference_capacity);
        assert(!copy(4) && c->stats.uploads==7);
    } else if(mode=="seed_poison") {
        setenv("FORT_SCOPE_TEST_FAIL_COMPACT_SUBMIT","1",1);
        assert(copy(3)==FORT_SCOPE_EXECUTION);unsetenv("FORT_SCOPE_TEST_FAIL_COMPACT_SUBMIT");
        assert(c->poisoned && !state.credit[0]);assert(!fort_scope_abandon(handle));return 0;
    } else {
        assert(mode=="sequence");assert(!copy(3) && c->stats.uploads==5);
        assert(!copy(4) && c->stats.uploads==6);
    }
    assert(state.credit[0]==0 && c->residency->compact_reference_capacity==256u*1024);
    for(int i=3;i<61;++i)assert(host[i]==-77);
    assert(!fort_scope_close(handle));
}
'''


@pytest.fixture(scope="module")
def startup_runtime_program(tmp_path_factory):
    folder = tmp_path_factory.mktemp("startup_runtime")
    pricing = SOURCE[SOURCE.index("Pricing prices()"):SOURCE.index("Plan geometry()")]
    source = RUNTIME_SOURCE.replace('#include <cassert>', '#include <cassert>\nusing namespace fort_physical::compact;\n'+pricing)
    (folder / "main.cpp").write_text(source)
    command = [shutil.which("g++"), "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
               "-I"+str(RUNTIME), str(folder / "main.cpp"), "-o", str(folder / "run")]
    result = subprocess.run(command, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    return folder / "run"


@pytest.mark.parametrize("mode", ["sequence", "partial_failure", "allocation_miss", "seed_poison", "completion_metadata"])
def test_reference_submission_credit_and_failure_boundaries(startup_runtime_program, mode):
    result = subprocess.run([str(startup_runtime_program), mode], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
