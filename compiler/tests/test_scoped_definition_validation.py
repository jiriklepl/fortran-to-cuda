"""Logical definition authority is independent of unused physical transitions."""

import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'
DRIVER = r'''
#include "scoped_planning.hpp"
#include <cassert>
#include <iostream>
#include <random>
using namespace fort_scoped::planning;
// Frozen old implementation is an independent differential reference.
DefinitionValidation legacy(const Inputs &input) {
 DefinitionValidation r;
 try {
  detail::validate_metadata(input,false,false);
  struct Logical {Region initialized,host_current,device_current;};
  std::vector<Logical> states;std::unordered_map<fort_buffer_t,size_t> index;
  for(const auto &b:input.resources){index.emplace(b.handle,states.size());states.push_back({b.initialized,{},{}});}
  for(size_t i=0;i<input.operations.size();++i){r.operation=i;r.buffer=0;const auto &op=input.operations[i];
   if(op.kind==FORT_SCOPE_PLAN_TEAM_ENTRY || op.kind==FORT_SCOPE_PLAN_TEAM_NATIVE_CALL)continue;
   if(op.kind==FORT_SCOPE_PLAN_FORGET || op.kind==FORT_SCOPE_PLAN_DISCARD){
    for(const auto &b:op.bindings){auto &defined=states[index.at(b.buffer)].initialized;
     if(op.kind==FORT_SCOPE_PLAN_FORGET)defined.clear();
     else{fort_scoped::coherence::Budget budget;defined=fort_scoped::coherence::difference(defined,b.effects.writes,budget);}}
    continue;
   }
   std::vector<fort_scoped::coherence::Prepared> prepared;
   for(const auto &b:op.bindings){r.buffer=b.buffer;auto &state=states[index.at(b.buffer)];
    fort_scoped::coherence::Budget budget;
    auto preserve=fort_scoped::coherence::difference(b.effects.writes,b.effects.overwrites,budget);
    auto required=fort_scoped::coherence::unite(b.effects.reads,preserve,budget);
    if(!fort_scoped::coherence::difference(required,state.initialized,budget).empty()){
     r.status=FORT_SCOPE_UNINITIALIZED;r.reason="uninitialized_read";return r;}
    prepared.push_back(fort_scoped::coherence::prepare(state,b.effects,false));
   }
   for(size_t j=0;j<op.bindings.size();++j)states[index.at(op.bindings[j].buffer)].initialized=std::move(prepared[j].initialized);
  }
  r.status=FORT_SCOPE_OK;r.reason="definition_plan_valid";r.operation=std::numeric_limits<size_t>::max();r.buffer=0;
 }catch(const detail::Unavailable &e){r.reason=e.reason;}
  catch(const fort_scoped::coherence::Fragmented &){r.reason="region_fragmentation_unavailable";}
 return r;
}
Region bits(unsigned mask){Region r;for(size_t i=0;i<16;){
 if(!(mask&(1u<<i))){++i;continue;}size_t first=i;while(i<16&&(mask&(1u<<i)))++i;
 r.push_back(Box{{first},{i}});}return r;}
Resource resource(fort_buffer_t handle,unsigned initialized){
 Resource b;b.handle=handle;b.element_bytes=8;b.bytes=128;b.extents={16};
 b.initialized=b.host_current=bits(initialized);return b;
}
void random_graphs(){
 std::mt19937 rng(64131);
 for(int trial=0;trial<800;++trial){
  Inputs in;in.graph_mode=true;std::vector<unsigned> defined;
  for(size_t i=0,n=1+rng()%4;i<n;++i){defined.push_back(rng()&65535);in.resources.push_back(resource(10+i,defined.back()));}
  for(size_t i=0,n=1+rng()%18;i<n;++i){Operation op;const auto kind=rng()%8;
   op.kind=kind==0?FORT_SCOPE_PLAN_FORGET:kind==1?FORT_SCOPE_PLAN_DISCARD:
    kind==2?FORT_SCOPE_PLAN_TEAM_ENTRY:kind==3?FORT_SCOPE_PLAN_TEAM_NATIVE_CALL:
    kind==4?FORT_SCOPE_PLAN_NATIVE:FORT_SCOPE_PLAN_WORKER;
   if(op.kind==FORT_SCOPE_PLAN_WORKER)op.unit=100+i;
   if(op.kind!=FORT_SCOPE_PLAN_TEAM_ENTRY && op.kind!=FORT_SCOPE_PLAN_TEAM_NATIVE_CALL){
    const size_t index=rng()%defined.size();unsigned reads=rng()&65535,writes=rng()&65535,overwrites=(rng()&writes);
    if(op.kind==FORT_SCOPE_PLAN_FORGET)reads=writes=overwrites=0;
    if(op.kind==FORT_SCOPE_PLAN_DISCARD)reads=overwrites=0;
    op.bindings.push_back({10+index,{bits(reads),bits(writes),bits(overwrites)}});
   }
   in.operations.push_back(op);
  }
  DefinitionValidation oracle;oracle.status=FORT_SCOPE_OK;oracle.operation=std::numeric_limits<size_t>::max();oracle.buffer=0;
  for(size_t i=0;i<in.operations.size();++i){const auto &op=in.operations[i];
   for(const auto &binding:op.bindings){const auto index=binding.buffer-10;
    auto mask=[](const Region &r){unsigned m=0;for(const auto &b:r)for(size_t k=b.lo[0];k<b.hi[0];++k)m|=1u<<k;return m;};
    const unsigned reads=mask(binding.effects.reads),writes=mask(binding.effects.writes),overwrites=mask(binding.effects.overwrites);
    if(op.kind==FORT_SCOPE_PLAN_FORGET)defined[index]=0;
    else if(op.kind==FORT_SCOPE_PLAN_DISCARD)defined[index]&=~writes;
    else if((reads|(writes&~overwrites))&~defined[index]){oracle.status=FORT_SCOPE_UNINITIALIZED;oracle.operation=i;oracle.buffer=binding.buffer;break;}
    else defined[index]|=writes;
   }
   if(oracle.status!=FORT_SCOPE_OK)break;
  }
  const auto before=in.resources;auto old=legacy(in),now=validate_definitions(in);
  assert(now.status==oracle.status && now.operation==oracle.operation && now.buffer==oracle.buffer);
  assert(old.status==now.status && old.operation==now.operation && old.buffer==now.buffer);
  for(size_t i=0;i<before.size();++i){assert(before[i].initialized.size()==in.resources[i].initialized.size());
   for(size_t j=0;j<before[i].initialized.size();++j)assert(before[i].initialized[j].lo==in.resources[i].initialized[j].lo && before[i].initialized[j].hi==in.resources[i].initialized[j].hi);}
 }
}
Inputs preservation(){Inputs in;in.graph_mode=true;in.resources={resource(1,0)};
 Operation producer;producer.kind=FORT_SCOPE_PLAN_WORKER;producer.unit=10;
 producer.bindings={{1,{bits(0),bits(255),bits(15)}}};
 Operation consumer;consumer.kind=FORT_SCOPE_PLAN_WORKER;consumer.unit=11;consumer.bindings={{1,{bits(255),{}, {}}}};
 in.operations={producer,consumer};return in;
}
void guards(){auto in=preservation();assert(validate_definitions(in).status==FORT_SCOPE_UNINITIALIZED);
 in.resources[0].initialized=bits(240);assert(validate_definitions(in).status==FORT_SCOPE_OK);
 in.operations[0].bindings[0].effects.reads=bits(1);assert(validate_definitions(in).status==FORT_SCOPE_UNINITIALIZED);
 in.operations[0].bindings[0].effects.reads.clear();
 Operation kill;kill.kind=FORT_SCOPE_PLAN_FORGET;kill.bindings={{1,{}}};in.operations.insert(in.operations.begin()+1,kill);
 assert(validate_definitions(in).status==FORT_SCOPE_UNINITIALIZED);
 in.operations[1].kind=FORT_SCOPE_PLAN_DISCARD;in.operations[1].bindings[0].effects.writes=bits(128);
 assert(validate_definitions(in).status==FORT_SCOPE_UNINITIALIZED);
 in.operations[1].bindings[0].effects.writes.clear();assert(validate_definitions(in).status==FORT_SCOPE_OK);
 in.operations[0].bindings[0].effects.overwrites=bits(65535);assert(validate_definitions(in).status==FORT_SCOPE_BOUNDARY);
 in=preservation();in.operations[0].bindings[0].buffer=999;assert(validate_definitions(in).status==FORT_SCOPE_BOUNDARY);
 in=preservation();in.resources[0].extents={15};assert(validate_definitions(in).status==FORT_SCOPE_BOUNDARY);
 in=preservation();in.operations[0].bindings[0].effects.writes={Box{{0},{17}}};assert(validate_definitions(in).status==FORT_SCOPE_BOUNDARY);
}
void fragmentation(){
 Inputs in;in.graph_mode=true;Resource b;b.handle=1;b.element_bytes=8;b.extents={11,11};b.bytes=968;
 const Region full={Box{{0,0},{11,11}}};b.initialized=b.host_current=full;in.resources={b};Region bars;
 for(size_t k=1;k<11;k+=2)bars.push_back(Box{{k,0},{k+1,11}});
 for(size_t k=1;k<11;k+=2)bars.push_back(Box{{0,k},{11,k+1}});
 Operation op;op.kind=FORT_SCOPE_PLAN_WORKER;op.unit=100;op.flops=100;op.memory_bytes=968;
 op.bindings={{1,{{},bars,bars}}};in.operations={op};
 assert(legacy(in).status==FORT_SCOPE_BOUNDARY);
 assert(validate_definitions(in).status==FORT_SCOPE_OK);
 fort_scope_plan_costs costs{};costs.version=FORT_SCOPE_PLANNING_ABI_VERSION;costs.valid=1;costs.max_allocation_bytes=4096;
 costs.cpu_flops=costs.cpu_bandwidth=costs.gpu_flops=costs.gpu_bandwidth=1e9;
 costs.h2d_bandwidth=costs.d2h_bandwidth=1e9;uint64_t work=0;
 auto physical=detail::simulate(in,costs,{false},work);assert(!physical.count.uploads && !physical.count.downloads);
 in.resources[0].initialized.clear();assert(validate_definitions(in).status==FORT_SCOPE_BOUNDARY);
}
int main(int argc,char**argv){assert(argc==2);const std::string mode=argv[1];
 if(mode=="random")random_graphs();else if(mode=="guards")guards();else if(mode=="fragmentation")fragmentation();else return 2;
 std::cout<<mode<<" definition authority verified\n";
}
'''


@pytest.fixture(scope='module')
def definition_binary(tmp_path_factory):
    compiler = shutil.which('g++-14') or shutil.which('g++')
    if compiler is None:
        pytest.skip('C++ compiler required')
    directory = tmp_path_factory.mktemp('logical-definitions')
    source = directory / 'definitions.cpp'
    source.write_text(DRIVER)
    target = directory / 'run'
    result = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                             '-I', str(RUNTIME), str(source), '-o', str(target)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return target


@pytest.mark.parametrize('mode', ['random', 'guards', 'fragmentation'])
def test_definition_authority_without_physical_copies(definition_binary, mode):
    result = subprocess.run([str(definition_binary), mode], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
