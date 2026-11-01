"""Generate application-independent source/scratch and original-team probes.

These generators publish raw observations only. They neither extend a hardware
profile nor approve a fit, GPU placement, source authority or complete-owner
estimate. Official observations require the caller's quiet lease, exact source
and toolchain bindings, hardware checks, output retention and bounded process.
"""

from __future__ import annotations


def source_scratch_benchmark(*, precision=64, structural_submit=False):
    """Actual public runtime ABI; no handwritten estimate or application kernel.

    Host submission intervals exclude completion cleanup. Cold/reuse/release
    scratch measurements remain distinct, and requested bytes are explicit.
    """
    if type(precision) is not int or precision not in (32, 64):
        raise ValueError("source protocol probe requires REAL32 or REAL64")
    source = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
namespace {
using clock_type=std::chrono::steady_clock;
using real=FORT_PROBE_REAL;
constexpr unsigned samples=5;
void require(bool value,const char* message){if(!value){std::cerr<<message<<'\n';std::exit(2);}}
void scope(int status){if(status){std::cerr<<fort_scope_error()<<'\n';std::exit(2);}}
void cuda(cudaError_t status){if(status!=cudaSuccess){std::cerr<<cudaGetErrorString(status)<<'\n';std::exit(2);}}
struct owner {
 fort_scope_domain_t domain=0;fort_scope_t context=0;
 void open(int device,std::size_t budget){
  scope(fort_scope_domain_create_v1(device,&domain));scope(fort_scope_create_in_domain_v1(domain,&context));
  scope(fort_scope_set_device_budget(context,budget));
  int previous=0;void* stream=nullptr;scope(fort_scope_gpu_enter(context,&previous,&stream));scope(fort_scope_gpu_leave(context,previous));
 }
 void close(){scope(fort_scope_close(context));scope(fort_scope_domain_close_v1(domain));context=0;domain=0;}
};
template<class Before,class Operation,class After>
std::vector<double> measure(unsigned rounds,Before before,Operation operation,After after){
 before();operation();after();
 std::vector<double> values;
 for(unsigned sample=0;sample<samples;++sample){double total=0;
  for(unsigned round=0;round<rounds;++round){before();const auto start=clock_type::now();operation();
   total+=std::chrono::duration<double>(clock_type::now()-start).count();after();}
  values.push_back(total/rounds);
 }
 return values;
}
void row(const char* name,std::size_t bytes,unsigned bindings,unsigned launches,unsigned rounds,const std::vector<double>& values){
 std::cout<<std::setprecision(17)<<"{\"kind\":\"source_protocol_observation\",\"name\":\""<<name
 <<"\",\"bytes\":"<<bytes<<",\"bindings\":"<<bindings<<",\"launches\":"<<launches
 <<",\"rounds\":"<<rounds<<",\"seconds\":[";
 for(unsigned i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}
 std::cout<<"]}\n"<<std::flush;
}
void identity(int device,int threads){
 cudaDeviceProp property{};cuda(cudaGetDeviceProperties(&property,device));
 int runtime=0,driver=0;cuda(cudaRuntimeGetVersion(&runtime));cuda(cudaDriverGetVersion(&driver));
 std::ostringstream uuid;uuid<<"GPU-"<<std::hex<<std::setfill('0');
 for(unsigned i=0;i<16;++i){if(i==4||i==6||i==8||i==10)uuid<<'-';uuid<<std::setw(2)<<unsigned(static_cast<unsigned char>(property.uuid.bytes[i]));}
 omp_sched_t schedule{};int chunk=0;omp_get_schedule(&schedule,&chunk);
 std::cout<<"{\"kind\":\"source_protocol_identity\",\"version\":1,\"gpu_uuid\":\""<<uuid.str()
 <<"\",\"device\":"<<device<<",\"cpu_threads\":"<<threads<<",\"precision\":"<<sizeof(real)*CHAR_BIT
 <<",\"cuda_runtime_version\":"<<runtime<<",\"driver_version\":"<<driver
 <<",\"schedule_kind\":"<<static_cast<int>(schedule)<<",\"schedule_chunk\":"<<chunk<<"}\n"<<std::flush;
}
__global__ void empty_kernel(){}
struct submission{std::uint64_t launches;};
int worker(void* opaque,const fort_scope_external_view_v1*,std::size_t,void* stream,int* started,std::uint64_t* accepted){
 *started=0;*accepted=0;const auto count=static_cast<submission*>(opaque)->launches;
 for(std::uint64_t i=0;i<count;++i){*started=1;empty_kernel<<<1,1,0,static_cast<cudaStream_t>(stream)>>>();
  if(cudaGetLastError()!=cudaSuccess)return FORT_SCOPE_EXECUTION;++*accepted;}
 return FORT_SCOPE_OK;
}
void scratch(int device,std::size_t bytes,std::size_t budget){
 owner current;fort_scope_scratch_lease_v1 lease{};
 auto acquire=[&]{scope(fort_scope_scratch_acquire_v1(current.context,bytes,&lease));
  require(lease.version==1&&lease.bytes==bytes&&lease.capacity>=bytes,"invalid scratch descriptor");};
 auto release=[&]{scope(fort_scope_scratch_release_v1(current.context,lease.token));};
 row("scratch_cold_acquire_host",bytes,0,0,8,measure(8,[&]{current.open(device,budget);},acquire,[&]{release();current.close();}));
 current.open(device,budget);acquire();release();
 row("scratch_reuse_acquire_host",bytes,0,0,128,measure(128,[]{},acquire,release));
 row("scratch_release_host",bytes,0,0,128,measure(128,acquire,release,[]{}));
 fort_scope_scratch_stats_v1 stats{};scope(fort_scope_scratch_stats_get_v1(current.context,&stats));
 require(!stats.active&&!stats.active_bytes&&stats.acquisitions==stats.releases,"scratch lease imbalance");
 fort_scope_stats physical{};scope(fort_scope_stats_get(current.context,&physical));
 require(!physical.uploads&&!physical.downloads&&!physical.launches,"scratch probe performed source work");
 current.close();
}
void grow_scratch(int device,std::size_t old_bytes,std::size_t bytes,std::size_t budget){
 owner current;fort_scope_scratch_lease_v1 lease{};
 auto before=[&]{current.open(device,budget);
  scope(fort_scope_scratch_acquire_v1(current.context,old_bytes,&lease));
  scope(fort_scope_scratch_release_v1(current.context,lease.token));scope(fort_scope_wait(current.context));};
 auto operation=[&]{scope(fort_scope_scratch_acquire_v1(current.context,bytes,&lease));
  require(lease.version==1&&lease.bytes==bytes&&lease.capacity>=bytes,"invalid grown scratch descriptor");};
 auto after=[&]{scope(fort_scope_scratch_release_v1(current.context,lease.token));
  fort_scope_scratch_stats_v1 stats{};scope(fort_scope_scratch_stats_get_v1(current.context,&stats));
  require(stats.allocations==2&&stats.grows==1&&stats.acquisitions==2&&stats.releases==2&&!stats.active,"scratch growth did not replace one completed arena");
  current.close();};
 row("scratch_idle_grow_host",bytes,0,0,8,measure(8,before,operation,after));
}
void submit(int device,unsigned count,unsigned launches,std::size_t budget){
 owner current;current.open(device,budget);
 std::vector<std::vector<real>> arrays(count,std::vector<real>(128,1.0));
 std::vector<fort_scope_external_binding_v1> bindings;
 const std::size_t extents[]={4,4,8};const std::int64_t lower[]={-2,0,4};
 const fort_scope_access access{FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL,0,nullptr,0,nullptr,0,nullptr};
 for(unsigned i=0;i<count;++i){fort_buffer_t buffer=0;
  const fort_scope_layout layout{3,FORT_PROBE_TYPE,sizeof(real),arrays[i].data(),extents,lower,1};
  scope(fort_scope_register(current.context,i+1,1,&layout,1,&buffer));
  void* pointer=nullptr;scope(fort_scope_device_begin(current.context,buffer,&access,&pointer));scope(fort_scope_device_end(current.context,buffer));
  bindings.push_back({buffer,FORT_PROBE_TYPE,3,sizeof(real),access});
 }
 scope(fort_scope_wait(current.context));
 fort_scope_stats before{},after{};scope(fort_scope_stats_get(current.context,&before));
 submission state{launches};const fort_scope_external_call_v1 call{1,0,bindings.data(),bindings.size(),0,0};
 auto operation=[&]{fort_scope_external_report_v1 report{};
  scope(fort_scope_source_execute_v1(current.context,&call,launches,worker,&state,&report));
  require(report.version==1&&report.status==0&&report.started==1&&report.committed==1,"incomplete source submission");};
 auto finish=[&]{scope(fort_scope_wait(current.context));};
 constexpr unsigned rounds=32;
 row("source_submit_host_enqueue_included",0,count,launches,rounds,measure(rounds,[]{},operation,finish));
 scope(fort_scope_stats_get(current.context,&after));
 require(after.launches-before.launches==std::uint64_t(launches)*(1+samples*rounds),"accepted source launch count differs");
 require(after.uploads==before.uploads&&after.downloads==before.downloads,"source submission unexpectedly transferred");
 for(unsigned i=0;i<count;++i){scope(fort_scope_host_begin(current.context,bindings[i].buffer,&access));
  for(real value:arrays[i])require(value==1.0,"empty source callback changed numerical fields");
  scope(fort_scope_host_end(current.context,bindings[i].buffer));}
 current.close();
}
void native_source(int device,unsigned count,bool resident,bool writer,std::size_t budget){
 owner current;
 if(resident)current.open(device,budget);
 else{scope(fort_scope_domain_create_v1(device,&current.domain));
  scope(fort_scope_create_in_domain_v1(current.domain,&current.context));
  scope(fort_scope_set_device_budget(current.context,budget));}
 std::vector<std::vector<real>> arrays(count,std::vector<real>(128,1.0));
 std::vector<fort_scope_external_binding_v1> bindings;
 const std::size_t extents[]={4,4,8};const std::int64_t lower[]={-2,0,4};
 const fort_scope_access access{unsigned(FORT_SCOPE_READ_ALL|(writer?FORT_SCOPE_WRITE_ALL:0)),0,nullptr,0,nullptr,0,nullptr};
 const fort_scope_access resident_read{FORT_SCOPE_READ_ALL,0,nullptr,0,nullptr,0,nullptr};
 for(unsigned i=0;i<count;++i){fort_buffer_t buffer=0;
  const fort_scope_layout layout{3,FORT_PROBE_TYPE,sizeof(real),arrays[i].data(),extents,lower,1};
  scope(fort_scope_register(current.context,i+1,1,&layout,1,&buffer));
  if(resident){void* pointer=nullptr;
   scope(fort_scope_device_begin(current.context,buffer,&resident_read,&pointer));scope(fort_scope_device_end(current.context,buffer));}
  bindings.push_back({buffer,FORT_PROBE_TYPE,3,sizeof(real),access});
 }
 if(resident)scope(fort_scope_wait(current.context));
 fort_scope_stats before{},after{};scope(fort_scope_stats_get(current.context,&before));
 const fort_scope_external_call_v1 call{1,0,bindings.data(),bindings.size(),0,0};
 auto operation=[&]{std::uint64_t token=0;
  scope(fort_scope_source_host_prepare_v1(current.context,&call,&token));require(token!=0,"missing CPU source token");
  scope(fort_scope_source_host_start_v1(current.context,token));
  // Only management is measured. Original body/team costs are independent.
  scope(fort_scope_source_host_complete_v1(current.context,token));};
 const char* name=resident?(writer?"source_native_resident_readwrite_host":"source_native_resident_read_host"):
                          (writer?"source_native_metadata_readwrite_host":"source_native_metadata_read_host");
 row(name,0,count,0,128,measure(128,[]{},operation,[]{}));
 scope(fort_scope_stats_get(current.context,&after));
 require(before.uploads==after.uploads&&before.downloads==after.downloads&&before.launches==after.launches,
         "native management unexpectedly copied or launched");
 for(const auto& values:arrays)for(real value:values)require(value==1.0,"native management changed fields");
 current.close();
}
}
int main(int argc,char**argv){
 if(argc!=4)return 2;const int threads=std::atoi(argv[1]),max_mib=std::atoi(argv[2]),device=std::atoi(argv[3]);
 require(threads>=1&&threads<=256&&max_mib>=8&&max_mib<=1024&&device>=0,"invalid bounded probe arguments");
 omp_set_dynamic(0);int actual=0;
 #pragma omp parallel num_threads(threads)
 { #pragma omp master
  actual=omp_get_num_threads();
 }
 require(actual==threads,"requested original CPU width unavailable");cuda(cudaSetDevice(device));cuda(cudaFree(nullptr));
 identity(device,actual);
 const std::size_t budget=std::size_t(max_mib)*1024*1024;
 std::vector<std::size_t> sizes{4096,65536,1048576,4194304,16777216,budget};
 std::sort(sizes.begin(),sizes.end());sizes.erase(std::unique(sizes.begin(),sizes.end()),sizes.end());
 std::size_t previous=0;
 for(std::size_t bytes:sizes)if(bytes<=budget){scratch(device,bytes,budget);
  if(previous)grow_scratch(device,previous,bytes,budget);previous=bytes;}
 for(unsigned count:{1u,2u,4u,6u,8u})for(unsigned launches:{1u,2u,4u,8u,16u,32u,64u,128u,256u})submit(device,count,launches,budget);
 for(unsigned count:{1u,2u,4u,6u,8u})for(bool resident:{false,true})for(bool writer:{false,true})
  native_source(device,count,resident,writer,budget);
 return 0;
}
'''.replace(" { #pragma omp master", " {\n #pragma omp master").replace(
        "FORT_PROBE_REAL", "double" if precision == 64 else "float").replace(
        "FORT_PROBE_TYPE", "FORT_SCOPE_REAL64" if precision == 64 else "FORT_SCOPE_REAL32").replace(
        "#include <cstdint>", "#include <cstdint>\n#include <climits>")
    if structural_submit:
        from compiler.offload.source_submit_model import ARGUMENT_WORDS, BINDINGS, LAUNCHES

        source = source.replace("__global__ void empty_kernel(){}", "__global__ void empty_kernel(" +
            ",".join(f"std::uint64_t unused{i}" for i in range(ARGUMENT_WORDS)) + "){}")
        source = source.replace("empty_kernel<<<1,1,0,static_cast<cudaStream_t>(stream)>>>()",
            "empty_kernel<<<1,256,0,static_cast<cudaStream_t>(stream)>>>(" +
            ",".join("std::uint64_t(0)" for _ in range(ARGUMENT_WORDS)) + ")")
        source = source.replace('<<",\\\"schedule_kind\\\":',
            f'<<",\\\"kernel_argument_words\\\":{ARGUMENT_WORDS},\\\"kernel_argument_bytes\\\":{8*ARGUMENT_WORDS},\\\"schedule_kind\\\":')
        source = source.replace("for(unsigned count:{1u,2u,4u,6u,8u})for(unsigned launches:{1u,2u,4u,8u,16u,32u,64u,128u,256u})",
            "for(unsigned count:{" + ",".join(f"{value}u" for value in BINDINGS) + "})for(unsigned launches:{" +
            ",".join(f"{value}u" for value in LAUNCHES) + "})")
    return source


def original_team_benchmark(*, units=(1, 2, 3, 4, 8), rank=3, collapse=None):
    """Original fork/join + exact worksharing barriers, with observable receipts.

    This fixed protocol deliberately makes no PRIVATE-array allocation or body
    throughput claim. No transformed persistent-team protocol is substituted.
    """
    if not units or len(set(units)) != len(units) or any(type(value) is not int or not 1 <= value <= 32 for value in units):
        raise ValueError("original team probe requires distinct bounded positive unit counts")
    collapse = rank if collapse is None else collapse
    if type(rank) is not int or type(collapse) is not int or not 1 <= collapse <= rank <= 3:
        raise ValueError("original team probe requires an exact rank/collapse between one and three")
    coordinates = ("i", "j", "k")[:rank]
    cells = f"min(threads,8)**{rank}"
    ordinal = "+".join(f"({coordinate}-1)*extent**{axis}" for axis, coordinate in enumerate(coordinates))+"+1"
    workers = []
    for count in units:
        body = []
        for _ in range(count):
            body += [f"!$omp do schedule(runtime) collapse({collapse})",
                     *[f"do {coordinate}=1,extent" for coordinate in reversed(coordinates)],
                     f"cell={ordinal}", "hits(cell)=hits(cell)+1_c_int64_t", *["enddo"]*rank, "!$omp end do"]
        workers += [f"subroutine team_{count}(threads,hits,actual)", "integer,intent(in)::threads",
                    f"integer(c_int64_t),intent(inout)::hits({cells})", "integer,intent(out)::actual",
                    "integer::"+",".join((*coordinates, "cell", "extent")), "extent=min(threads,8)",
                    "!$omp parallel num_threads(threads) private("+",".join((*coordinates, "cell"))+") shared(hits,actual)",
                    "!$omp master", "actual=omp_get_num_threads()", "!$omp end master", *body,
                    "!$omp end parallel", f"end subroutine team_{count}"]
    observations = []
    for count in units:
        observations += [f"call observe({count},team_{count})"]
    return "\n".join([
        "program original_team_protocol", "use iso_c_binding", "use omp_lib", "implicit none",
        "integer::threads,rounds,status", "character(64)::argument", "integer(kind=omp_sched_kind)::schedule_kind",
        "integer::schedule_chunk", "call get_command_argument(1,argument)", "read(argument,*,iostat=status)threads",
        "if(status/=0)stop 2", "call get_command_argument(2,argument)", "read(argument,*,iostat=status)rounds",
        "if(status/=0)stop 2", "if(threads<1.or.threads>256.or.rounds<1.or.rounds>100000)stop 2",
        "call omp_set_dynamic(.false.)", "call omp_get_schedule(schedule_kind,schedule_chunk)", *observations,
        "contains", *workers,
        "subroutine observe(units,team)", "integer,intent(in)::units", "interface",
        "subroutine team(threads,hits,actual)", "use iso_c_binding", "integer,intent(in)::threads",
        f"integer(c_int64_t),intent(inout)::hits({cells})", "integer,intent(out)::actual", "end subroutine", "end interface",
        f"integer(c_int64_t)::hits({cells})", "integer::sample,round,actual", "real(c_double)::seconds(5),start,total",
        "hits=0", "call team(threads,hits,actual)", "if(actual/=threads.or.any(hits/=units))stop 3",
        "do sample=1,5", "total=0", "do round=1,rounds", "hits=0", "start=omp_get_wtime()",
        "call team(threads,hits,actual)", "total=total+(omp_get_wtime()-start)",
        "if(actual/=threads.or.any(hits/=units))stop 3", "enddo", "seconds(sample)=total/rounds", "enddo",
        "write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0,a)',advance='no') &",
        f"'{{\"kind\":\"original_team_observation\",\"rank\":{rank},\"collapse\":{collapse},\"units\":',units, &",
        "',\"cpu_threads\":',threads,',\"rounds\":',rounds, &",
        "',\"schedule_kind\":',int(schedule_kind),',\"schedule_chunk\":',schedule_chunk,',\"seconds\":['",
        "do sample=1,5", "if(sample>1)write(*,'(a)',advance='no')','",
        "write(*,'(es24.16)',advance='no')seconds(sample)", "enddo", "write(*,'(a)')']}'",
        "end subroutine observe", "end program original_team_protocol", ""])
