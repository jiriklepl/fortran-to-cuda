"""Native-owner hints retain metadata only and never authorize GPU/data effects."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.tests.test_scoped_runtime import RUNTIME

CLIENT = r'''
#include "scoped_runtime.h"
#include "scoped_host_metadata.hpp"
#include <cassert>
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <new>
#include <sys/mman.h>
#include <unistd.h>
thread_local bool reject_allocation=false;
void *operator new(std::size_t bytes){
 if(reject_allocation)throw std::bad_alloc();
 if(void *p=std::malloc(bytes?bytes:1))return p;
 throw std::bad_alloc();
}
void operator delete(void *p) noexcept {std::free(p);}
void operator delete(void *p,std::size_t) noexcept {std::free(p);}
#include <omp.h>
using Observation=fort_scope_native_hint_observation_v1;
uint64_t words[3]={0x18cafe,11,8};
void check(int status){if(status){std::fprintf(stderr,"%s\n",fort_scope_error());std::abort();}}
Observation begin(int &hit,const uint64_t *key=words,size_t count=3){
 Observation out{};hit=99;check(fort_scope_native_hint_begin_v1(key,count,&out,&hit));return out;
}
void finish(Observation &out){check(fort_scope_native_hint_finish_v1(&out));assert(!out.eligible);}
void learn(){int hit;auto out=begin(hit);assert(!hit&&out.eligible);finish(out);}
void cached(){int hit;auto out=begin(hit);assert(hit&&!out.eligible);}
void uncached(){int hit;auto out=begin(hit);assert(!hit&&out.eligible);}
struct Scope {
 fort_scope_t context{};fort_buffer_t buffer{};double data[8]{};
 size_t extents[1]={8};int64_t lower[1]={-2};
 explicit Scope(bool initialized=true){
  check(fort_scope_create(0,&context));
  fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,data,extents,lower,1};
  check(fort_scope_register(context,1,1,&layout,initialized,&buffer));
 }
 void close(){check(fort_scope_close(context));context=0;}
 ~Scope(){if(context)close();}
};
fort_scope_access read_access{FORT_SCOPE_READ_ALL,0,nullptr,0,nullptr,0,nullptr};
void gpu(){Scope scope;void *device=nullptr;check(fort_scope_device_begin(scope.context,scope.buffer,&read_access,&device));
 check(fort_scope_device_end(scope.context,scope.buffer));scope.close();}
void preview(){
 Scope scope;check(fort_scope_plan_reset(scope.context));
 fort_scope_plan_binding binding{scope.buffer,read_access};
 check(fort_scope_plan_add(scope.context,FORT_SCOPE_PLAN_WORKER,71,&binding,1,1e10,64,1));
 fort_scope_plan_costs costs{};costs.version=FORT_SCOPE_PLANNING_ABI_VERSION;costs.valid=1;
 costs.max_allocation_bytes=1<<20;costs.cpu_flops=costs.cpu_bandwidth=1e8;
 costs.gpu_flops=costs.gpu_bandwidth=costs.h2d_bandwidth=costs.d2h_bandwidth=1e12;
 costs.create_seconds=costs.register_seconds=costs.host_access_seconds=costs.device_access_seconds=1e-8;
 costs.gpu_setup_seconds=costs.cold_driver_startup_seconds=costs.allocation_seconds=costs.release_seconds=1e-6;
 costs.wait_seconds=costs.launch_enqueue_seconds=1e-6;costs.planning_operation_seconds=1e-10;
 fort_scope_plan_decision decision{};check(fort_scope_plan_select(scope.context,&costs,-1,&decision));
 assert(decision.available&&decision.gpu_units==1);scope.close();
}
void pure_cache(){
 using namespace fort_scoped::native_hint;
 Cache cache;Key key;key.integer(17);assert(cache.insert(key));
 for(unsigned i=0;i<32;++i)assert(cache.find(key));
 assert(!cache.insert(key)); // A duplicate observation must not renew the hit budget.
 for(unsigned i=0;i<32;++i)assert(cache.find(key));
 assert(!cache.insert(key)); // Late duplicate completion cannot avoid the forced miss.
 assert(!cache.find(key));cache.insert(key);assert(cache.find(key));
 Key collision;collision.integer(18);collision.hash=key.hash;
 assert(!cache.find(collision)); // Full equality, never just the hash.
 Cache bounded;
 for(unsigned i=0;i<entry_limit;++i){Key k;k.integer(i);bounded.insert(k);}
 assert(bounded.size()==entry_limit);
 Key first;first.integer(0);assert(bounded.find(first));
 Key next;next.integer(entry_limit);bounded.insert(next);
 Key second;second.integer(1);assert(!bounded.find(second));assert(bounded.find(first));
 Generation generation(UINT64_MAX-1);generation.advance();generation.advance();
 assert(generation.value()==UINT64_MAX);
}
// Independent reference for the existing memo environment serialization.
void environment(){
 std::vector<uint64_t> old,shared;
 cpu_set_t affinity;CPU_ZERO(&affinity);assert(!sched_getaffinity(0,sizeof(affinity),&affinity));
 old.push_back(CPU_COUNT(&affinity));
 for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&affinity))old.push_back(cpu);
 omp_sched_t schedule;int chunk;omp_get_schedule(&schedule,&chunk);
 for(int x:{omp_get_level(),omp_get_active_level(),omp_get_max_active_levels(),omp_get_num_threads(),
  omp_get_max_threads(),omp_get_dynamic(),omp_get_thread_limit(),int(omp_get_proc_bind()),int(schedule),chunk})old.push_back(uint64_t(x));
 old.push_back(omp_get_num_places());old.push_back(uint64_t(omp_get_place_num()));
 for(int place=0;place<omp_get_num_places();++place){int n=omp_get_place_num_procs(place);old.push_back(n);
  std::vector<int> ids(n);if(n)omp_get_place_proc_ids(place,ids.data());for(int id:ids)old.push_back(id);}
 int n=omp_get_partition_num_places();old.push_back(n);std::vector<int> ids(n);
 if(n)omp_get_partition_place_nums(ids.data());
 for(int id:ids)old.push_back(id);
 for(const char *name:{"OMP_WAIT_POLICY","GOMP_SPINCOUNT","OMP_PLACES","GOMP_CPU_AFFINITY","OMP_PROC_BIND"}){
  const char *value=std::getenv(name);old.push_back(value?std::strlen(value)+1:0);
  if(value)for(;*value;++value)old.push_back(static_cast<unsigned char>(*value));}
 auto append=[&](uint64_t value){shared.push_back(value);};auto remaining=[&]{return size_t(1000000)-shared.size();};
 fort_scoped::metadata::execution_environment(append,remaining);assert(old==shared);
}
int main(int argc,char **argv){
 assert(argc==2);const std::string mode=argv[1];
 if(mode=="cache")pure_cache();
 else if(mode=="environment")environment();
 else if(mode=="data"){
  int hit;auto out=begin(hit);assert(!hit&&out.eligible);
  Scope scope;fort_scope_access rw{FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL,0,nullptr,0,nullptr,0,nullptr};
  check(fort_scope_host_begin(scope.context,scope.buffer,&rw));
  for(auto &x:scope.data)x+=3;
  check(fort_scope_host_end(scope.context,scope.buffer));scope.close();finish(out);
  auto again=begin(hit);assert(hit&&!again.eligible);
  for(auto &x:scope.data)x=7*x+1;
  for(double x:scope.data)assert(x==22);
 }
 else if(mode=="duplicate"){
  int hit;auto first=begin(hit);auto second=begin(hit);assert(first.eligible&&second.eligible);
  finish(first);finish(second);cached();
 }
 else if(mode=="owned_key"){
  int hit;auto out=begin(hit);assert(out.eligible);words[2]=99;finish(out);
  uint64_t original[3]={0x18cafe,11,8};auto again=begin(hit,original);assert(hit&&!again.eligible);uncached();
 }
 else if(mode=="ttl"){
  learn();for(unsigned i=0;i<64;++i)cached();uncached();uncached(); // No tail observation silently renews it.
 }
 else if(mode=="live_begin"){
  Scope scope;int hit;auto out=begin(hit);assert(!hit&&!out.eligible);scope.close();learn();cached();
 }
 else if(mode=="live_finish"){
  int hit;auto out=begin(hit);Scope scope;finish(out);scope.close();uncached();
 }
 else if(mode=="expected7"){
  int hit;auto out=begin(hit);Scope scope(false);
  assert(fort_scope_host_begin(scope.context,scope.buffer,&read_access)==7);scope.close();finish(out);cached();
 }
 else if(mode=="unexpected"){
  int hit;auto out=begin(hit);assert(fort_scope_report_error(FORT_SCOPE_BOUNDARY,"declined") ==5);
  finish(out);uncached();
 }
 else if(mode=="foreign_error"){
  int hit;auto out=begin(hit);std::thread thread([]{assert(fort_scope_report_error(FORT_SCOPE_STATE,"foreign")==8);});
  thread.join();finish(out);uncached();
 }
 else if(mode=="gpu"||mode=="foreign_gpu"||mode=="preview"){
  int hit;auto out=begin(hit);
  if(mode=="foreign_gpu"){std::thread thread(gpu);thread.join();}else if(mode=="preview")preview();else gpu();
  finish(out);uncached();learn();gpu();cached(); // GPU between invocations does not invalidate an old hint.
 }
 else if(mode=="direct_undefined"){
  int hit;auto out=begin(hit);Scope scope(false);void *pointer=nullptr;
  assert(fort_scope_device_begin(scope.context,scope.buffer,&read_access,&pointer)==7);
  scope.close();finish(out);uncached(); // Status7 alone is exempt; a GPU attempt is not.
 }
 else if(mode=="queued_undefined"){
  int hit;auto out=begin(hit);Scope scope(false);void *device=nullptr;
  check(fort_scope_device_reserve_v1(scope.context,scope.buffer,&device));
  fort_scope_plan_binding binding{scope.buffer,read_access};
  fort_scope_graph_task_v1 task{};task.kernel.version=FORT_SCOPE_GRAPH_ABI_VERSION;
  task.kernel.function=reinterpret_cast<void *>(uintptr_t(1));
  task.kernel.grid[0]=task.kernel.grid[1]=task.kernel.grid[2]=1;
  task.kernel.block[0]=task.kernel.block[1]=task.kernel.block[2]=1;
  task.bindings=&binding;task.binding_count=1;
  assert(fort_scope_graph_enqueue_v1(scope.context,23,&task,nullptr)==7);
  scope.close();finish(out);uncached();
 }
 else if(mode=="allocation"){
  assert(omp_get_num_places()>0);learn();cached();
  reject_allocation=true;int hit;auto out=begin(hit);reject_allocation=false;
  assert(!hit&&!out.eligible);cached(); // Optional allocation failure keeps old cache usable.
 }
 else if(mode=="invalid_abi"){
  const long page=sysconf(_SC_PAGESIZE);assert(page>0);
  auto *mapping=static_cast<char *>(mmap(nullptr,2*page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
  assert(mapping!=MAP_FAILED&&!mprotect(mapping+page,page,PROT_NONE));
  auto *prefix=reinterpret_cast<uint32_t *>(mapping+page-sizeof(uint32_t));*prefix=0;
  check(fort_scope_native_hint_finish_v1(reinterpret_cast<Observation *>(prefix)));
  *prefix=77;assert(fort_scope_native_hint_finish_v1(reinterpret_cast<Observation *>(prefix))==1);
  assert(!munmap(mapping,2*page));
 }
 else if(mode=="bounds"){
  int hit;auto zero=begin(hit,words,0);assert(!hit&&!zero.eligible);
  auto missing=begin(hit,nullptr,3);assert(!hit&&!missing.eligible);
  uint64_t large[512]{};auto full=begin(hit,large,512);assert(!hit&&!full.eligible);
  auto extra=begin(hit,large,513);assert(!hit&&!extra.eligible);
  learn();cached();
 }
 else if(mode=="fp"){
  std::fenv_t saved;assert(!std::fegetenv(&saved));std::feraiseexcept(FE_INEXACT);
  const int flags=std::fetestexcept(FE_ALL_EXCEPT);learn();cached();assert(std::fetestexcept(FE_ALL_EXCEPT)==flags);
  assert(!std::fesetround(FE_UPWARD));int hit;auto out=begin(hit);assert(!hit&&!out.eligible);
  assert(std::fegetround()==FE_UPWARD);std::fesetenv(&saved);
  feenableexcept(FE_INVALID);out=begin(hit);assert(!hit&&!out.eligible&&fegetexcept()==FE_INVALID);std::fesetenv(&saved);
 }
 else if(mode=="nonserial"){
  int hit=-1;Observation out{};
  #pragma omp parallel num_threads(2) shared(hit,out)
  {
   #pragma omp single
   {out=begin(hit);}
  }
  assert(!hit&&!out.eligible);learn();cached();
 }
 else if(mode=="env_change"){
  learn();cached();const char *old=std::getenv("OMP_WAIT_POLICY");const std::string saved=old?old:"";
  setenv("OMP_WAIT_POLICY",old&&saved=="ACTIVE"?"PASSIVE":"ACTIVE",1);uncached();
  if(old)setenv("OMP_WAIT_POLICY",saved.c_str(),1);else unsetenv("OMP_WAIT_POLICY");cached();
 }
 else if(mode=="affinity"){
  learn();cached();cpu_set_t saved;assert(!sched_getaffinity(0,sizeof(saved),&saved));
  if(CPU_COUNT(&saved)>1){cpu_set_t one;CPU_ZERO(&one);for(int i=0;i<CPU_SETSIZE;++i)if(CPU_ISSET(i,&saved)){CPU_SET(i,&one);break;}
   assert(!sched_setaffinity(0,sizeof(one),&one));uncached();assert(!sched_setaffinity(0,sizeof(saved),&saved));cached();}
 }
 else std::abort();
 std::puts("native metadata hint checked");
}
'''


@pytest.fixture(scope="module")
def native_hint_binary(tmp_path_factory):
    compiler = shutil.which("g++-14") or shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    directory = tmp_path_factory.mktemp("native-owner-hint")
    source = directory / "client.cpp"
    source.write_text(CLIENT)
    binary = directory / "client"
    result = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fopenmp", "-pthread",
         "-DFORT_SCOPE_CPU_TEST", "-I", str(RUNTIME), "-x", "c++", str(RUNTIME / "scoped_runtime.cu"),
         str(source), "-o", str(binary)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("mode", [
    "cache", "environment", "data", "owned_key", "ttl", "live_begin", "live_finish", "expected7",
    "unexpected", "foreign_error", "gpu", "foreign_gpu", "preview", "direct_undefined", "bounds", "fp",
    "nonserial", "env_change", "affinity", "queued_undefined", "allocation", "invalid_abi", "duplicate",
])
def test_native_owner_hint_metadata_and_generation_contract(native_hint_binary, mode):
    result = subprocess.run([str(native_hint_binary), mode], capture_output=True, text=True,
                            env={**os.environ, "FORT_RUNTIME_TRACE": "1", "OMP_NUM_THREADS": "4",
                                 **({"OMP_PLACES": "cores"} if mode == "allocation" else {})}, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native metadata hint checked" in result.stdout
    events = [json.loads(line.removeprefix("FORT_SCOPED native_hint "))
              for line in result.stderr.splitlines() if line.startswith("FORT_SCOPED native_hint ")]
    assert all(row["event"] == "native_hint" and not row["estimates_available"] and "context" not in row
               for row in events)

    if mode == "duplicate":
        assert [row["learned"] for row in events if row["phase"] == "finish"] == [True, False]


@pytest.fixture(scope="module")
def cuda_native_hint_binary(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit required")
    directory = tmp_path_factory.mktemp("native-hint-cuda")
    source = directory / "client.cpp"
    source.write_text(CLIENT)
    objects = []
    for name, path in (("runtime", RUNTIME / "scoped_runtime.cu"), ("client", source)):
        target = directory / (name + ".o")
        # This client is host C++ only. Compile its allocation-failure override
        # with the host compiler so nvcc does not infer device operator-new.
        command = ([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler", "-fopenmp,-pthread"]
                   if name == "runtime" else [host, "-std=c++17", "-fopenmp", "-pthread"])
        result = subprocess.run([*command, "-I", str(RUNTIME), "-c", str(path), "-o", str(target)],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(target))
    target = directory / "run"
    result = subprocess.run([nvcc, "-ccbin", host, "-Xcompiler", "-fopenmp,-pthread",
                             *objects, "-o", str(target)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return Path(target)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["data", "gpu", "foreign_gpu", "preview", "direct_undefined", "queued_undefined"])
def test_actual_cuda_native_hint_never_authorizes_device_or_data_effects(cuda_native_hint_binary, mode):
    result = subprocess.run([str(cuda_native_hint_binary), mode], capture_output=True, text=True,
                            env={**os.environ, "FORT_RUNTIME_TRACE": "1", "OMP_NUM_THREADS": "4"}, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native metadata hint checked" in result.stdout


@pytest.fixture(scope="module")
def saturated_native_hint_binary(tmp_path_factory):
    """Reach saturation without adding a test/debug hook to the public ABI."""
    host = shutil.which("g++-14") or shutil.which("g++")
    if not host:
        pytest.skip("g++ required")
    directory = tmp_path_factory.mktemp("native-hint-saturation")
    source = directory / "client.cpp"
    source.write_text(r'''
#include "scoped_runtime.cu"
#include <cassert>
int main(int argc,char **argv) {
 assert(argc==2);using Generation=fort_scoped::native_hint::Generation;
 uint64_t old_key=11,new_key=22;int hit=0;fort_scope_native_hint_observation_v1 old{},pending{},check{};
 assert(!fort_scope_native_hint_begin_v1(&old_key,1,&old,&hit)&&!hit&&old.eligible);
 assert(!fort_scope_native_hint_finish_v1(&old));
 assert(!fort_scope_native_hint_begin_v1(&old_key,1,&old,&hit)&&hit);
 assert(!fort_scope_native_hint_begin_v1(&new_key,1,&pending,&hit)&&!hit&&pending.eligible);
 // Single-thread internal seam; no public mutation/diagnostic API is added.
 auto &generation=argv[1][0]=='g'?native_hint_gpu_generation:native_hint_error_generation;
 generation.~Generation();new(&generation)Generation(UINT64_MAX);generation.advance();
 assert(!fort_scope_native_hint_finish_v1(&pending)&&!pending.eligible);
 assert(!fort_scope_native_hint_begin_v1(&old_key,1,&check,&hit)&&!hit&&!check.eligible);
 assert(!fort_scope_native_hint_begin_v1(&new_key,1,&check,&hit)&&!hit&&!check.eligible);
}
''')
    binary = directory / "run"
    result = subprocess.run([host, "-std=c++17", "-fopenmp", "-pthread", "-DFORT_SCOPE_CPU_TEST",
                             "-I", str(RUNTIME), str(source), "-o", str(binary)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("generation", ["gpu", "error"])
def test_generation_saturation_disables_learning_and_old_hits(saturated_native_hint_binary, generation):
    result = subprocess.run([str(saturated_native_hint_binary), generation], capture_output=True, text=True,
                            env={**os.environ, "OMP_NUM_THREADS": "4"}, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
