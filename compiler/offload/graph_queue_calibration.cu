// Measure the production queue/cache protocol, independently of ELMM kernels.
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define SCOPE(call) do { int status=(call); if(status) { std::cerr << #call << ": " << fort_scope_error() << '\n'; std::exit(2); } } while(0)
#define CUDA(call) do { auto status=(call); if(status!=cudaSuccess) { std::cerr << #call << ": " << cudaGetErrorString(status) << '\n'; std::exit(3); } } while(0)
// FORT_GRAPH_PROBE_KERNELS

using Clock=std::chrono::steady_clock;
constexpr int samples=5,repetitions=64;
struct Arguments {
    double *device;
    std::array<double,63> scalars{};
    std::array<void *,64> pointers{};
    Arguments(double *output):device(output) {
        pointers[0]=&device;
        for(size_t i=0;i<scalars.size();++i) pointers[i+1]=&scalars[i];
    }
};
void require(bool condition,const char *message) {
    if(!condition) { std::cerr << message << '\n'; std::exit(4); }
}
fort_scope_graph_stats_v1 graph_stats(fort_scope_t context) {
    fort_scope_graph_stats_v1 result{};
    SCOPE(fort_scope_graph_stats_get_v1(context,&result)); return result;
}
void values(const std::vector<double> &data) {
    std::cout << '[';
    for(size_t i=0;i<data.size();++i) std::cout << (i?",":"") << data[i];
    std::cout << ']';
}
void measure(int device,const void *kernel,int arguments,int tasks,bool changed) {
    fort_scope_t context{};fort_buffer_t buffer{};double host=0,*output=nullptr;
    const size_t extent=1;const int64_t lower=-3;
    fort_scope_layout layout{1,FORT_SCOPE_REAL64,sizeof(double),&host,&extent,&lower,1};
    SCOPE(fort_scope_create(device,&context));
    SCOPE(fort_scope_task_graph_enable_v1(context));
    SCOPE(fort_scope_register(context,1,1,&layout,0,&buffer));
    SCOPE(fort_scope_device_reserve_v1(context,buffer,reinterpret_cast<void **>(&output)));
    fort_scope_plan_binding binding{buffer,{FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL,0,nullptr,0,nullptr,0,nullptr}};
    std::vector<Arguments> payloads;payloads.reserve(tasks);
    for(int i=0;i<tasks;++i) payloads.emplace_back(output);
    std::array<size_t,64> sizes{};sizes.fill(sizeof(double));sizes[0]=sizeof(output);
    double sequence=0,expected=0;
    auto prepare=[&]() {
        if(changed) ++sequence;
        for(int i=0;i<tasks;++i)
            for(int j=0;j<arguments-1;++j) payloads[i].scalars[j]=sequence+17+i+j;
        expected=payloads.back().scalars[0]+payloads.back().scalars[arguments-2];
    };
    auto submit=[&]() {
        for(int i=0;i<tasks;++i) {
            fort_scope_graph_task_v1 task{{1,0,kernel,{1,1,1},{32,1,1},0,payloads[i].pointers.data(),size_t(arguments)},&binding,1};
            SCOPE(fort_scope_graph_enqueue_v1(context,0x1000+unsigned(i),&task,sizes.data()));
        }
        SCOPE(fort_scope_graph_flush_v1(context));
    };
    // Exclude direct warmup, graph creation and first replay from steady costs.
    for(int i=0;i<3;++i) { prepare();submit();SCOPE(fort_scope_wait(context)); }
    const auto before=graph_stats(context);
    require(before.creations==1 && before.replays==1,"queue did not reach steady graph replay");
    std::vector<double> host_seconds,complete_seconds;
    for(int sample=0;sample<samples;++sample) {
        double host_time=0,complete_time=0;
        for(int round=0;round<repetitions;++round) {
            prepare();
            const auto start=Clock::now();submit();const auto accepted=Clock::now();
            SCOPE(fort_scope_wait(context));const auto completed=Clock::now();
            host_time+=std::chrono::duration<double>(accepted-start).count();
            complete_time+=std::chrono::duration<double>(completed-start).count();
        }
        host_seconds.push_back(host_time/repetitions);complete_seconds.push_back(complete_time/repetitions);
    }
    const auto after=graph_stats(context);
    const uint64_t groups=samples*repetitions,kernels=groups*tasks;
    require(after.replays-before.replays==groups && after.logical_kernels-before.logical_kernels==kernels,
            "measured work did not execute as graph replay");
    require(after.parameter_updates-before.parameter_updates==(changed?kernels:0),
            "graph parameter updates differ from the declared mutation pattern");
    fort_scope_stats traffic{};SCOPE(fort_scope_stats_get(context,&traffic));
    require(!traffic.upload_bytes && !traffic.download_bytes,"calibration included unexpected field transfers");
    SCOPE(fort_scope_observe_v1(context,&buffer,1));require(host==expected,"queued numerical result differs");
    SCOPE(fort_scope_close(context));
    std::cout << "{\"kind\":\"graph_queue_observation\",\"arguments\":" << arguments
              << ",\"argument_bytes\":" << sizes[0]+(arguments-1)*sizeof(double)
              << ",\"tasks\":" << tasks << ",\"changed\":" << (changed?"true":"false")
              << ",\"repetitions\":" << repetitions << ",\"samples\":" << samples
              << ",\"graph_replays\":" << groups << ",\"logical_kernels\":" << kernels
              << ",\"parameter_updates\":" << after.parameter_updates-before.parameter_updates
              << ",\"numerical_result_verified\":true,\"host_seconds\":";
    values(host_seconds);std::cout << ",\"complete_seconds\":";values(complete_seconds);std::cout << "}\n" << std::flush;
}
int main(int argc,char **argv) {
    require(argc==3,"expected DEVICE THREADS");const int device=std::atoi(argv[1]),threads=std::atoi(argv[2]);
    require(threads>0 && omp_get_max_threads()==threads && !omp_get_dynamic(),"OpenMP thread budget differs");
    CUDA(cudaSetDevice(device));cudaDeviceProp properties{};CUDA(cudaGetDeviceProperties(&properties,device));
    int driver=0,runtime=0;CUDA(cudaDriverGetVersion(&driver));CUDA(cudaRuntimeGetVersion(&runtime));
    std::cout << std::setprecision(17) << "{\"kind\":\"graph_queue_identity\",\"device\":" << device
              << ",\"cpu_threads\":" << threads << ",\"driver_version\":" << driver
              << ",\"runtime_version\":" << runtime << ",\"gpu_uuid\":\"GPU-";
    for(int i=0;i<16;++i) {
        if(i==4 || i==6 || i==8 || i==10) std::cout << '-';
        char hex[3];std::snprintf(hex,sizeof(hex),"%02x",static_cast<unsigned char>(properties.uuid.bytes[i]));std::cout << hex;
    }
    std::cout << "\"}\n";
    // FORT_GRAPH_PROBE_RUNS
}
