// Evidence only: original registry, production CUDA workers, no AUTO decision.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <vector>
#include "gpu_recipes.hpp"

using real=execution_real;
using clock_type=std::chrono::steady_clock;
static void check(cudaError_t error) {
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
static std::string quoted(const char* text) {
    std::string value="\"";
    for(const unsigned char c:std::string(text)) {
        if(c=='"'||c=='\\') {value+='\\';value+=char(c);}
        else if(c=='\n')value+="\\n";
        else if(c=='\r')value+="\\r";
        else if(c=='\t')value+="\\t";
        else if(c<32)throw std::runtime_error("nonprintable identity");
        else value+=char(c);
    }
    return value+'"';
}
static bool training(const gpu_recipe& r,int n) {
    return std::find(r.fits,r.fits+r.fit_count,n)!=r.fits+r.fit_count;
}
struct buffers {
    static constexpr real sentinel=real(-131.25);
    const gpu_recipe& recipe;
    int n;
    bool fit;
    std::vector<real>a,b,out;
    real *da=nullptr,*db=nullptr,*dout=nullptr;
    cudaStream_t stream=nullptr;
    buffers(const gpu_recipe& r,int items):recipe(r),n(items),fit(training(r,items)),
            a(items+2,sentinel),b(items+2,sentinel),out(items+2,sentinel) {
        const auto *av=fit?r.training_a:r.holdout_a,*bv=fit?r.training_b:r.holdout_b;
        const auto ac=fit?r.training_a_count:r.holdout_a_count,bc=fit?r.training_b_count:r.holdout_b_count;
        for(int i=0;i<n;++i) {a[i+1]=av[i%ac];b[i+1]=bv[i%bc];}
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cudaMalloc(&da,a.size()*sizeof(real)));
        check(cudaMalloc(&db,b.size()*sizeof(real)));
        check(cudaMalloc(&dout,out.size()*sizeof(real)));
        check(cudaMemcpyAsync(da,a.data(),a.size()*sizeof(real),cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(db,b.data(),b.size()*sizeof(real),cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(dout,out.data(),out.size()*sizeof(real),cudaMemcpyHostToDevice,stream));
        check(cudaStreamSynchronize(stream));
    }
    ~buffers() {cudaFree(da);cudaFree(db);cudaFree(dout);if(stream)cudaStreamDestroy(stream);}
    void execute(std::uint64_t& launches) {
        check(static_cast<cudaError_t>(recipe.launch(n,da+1,db+1,dout+1,stream,&launches)));
    }
    bool agrees() {
        std::vector<real>aa(a.size()),bb(b.size());
        check(cudaMemcpyAsync(aa.data(),da,aa.size()*sizeof(real),cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(bb.data(),db,bb.size()*sizeof(real),cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(out.data(),dout,out.size()*sizeof(real),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        if(std::memcmp(aa.data(),a.data(),a.size()*sizeof(real))||std::memcmp(bb.data(),b.data(),b.size()*sizeof(real))||
           out.front()!=sentinel||out.back()!=sentinel)return false;
        const auto *reference=fit?recipe.training_reference:recipe.holdout_reference;
        const auto count=fit?recipe.training_reference_count:recipe.holdout_reference_count;
        constexpr double tolerance=sizeof(real)==8?2e-12:2e-5;
        for(int i=0;i<n;++i) {
            const real actual=out[i+1],expected=reference[i%count];
            if(!std::isfinite(actual)||!std::isfinite(expected))return false;
            if(actual==expected) {if(actual==real(0)&&std::signbit(actual)!=std::signbit(expected))return false;}
            else if(std::abs(double(actual)-double(expected))>tolerance*(1.+std::abs(double(expected))))return false;
        }
        return true;
    }
};
static void coordinate(std::ostream& out,const char* kind,const gpu_recipe& r,int n,bool agrees) {
    out<<"{\"kind\":"<<quoted(kind)<<",\"recipe\":"<<quoted(r.name)<<",\"recipe_id\":"<<quoted(r.identity)
             <<",\"backend\":\"gpu\",\"items\":"<<n<<",\"role\":"<<quoted(training(r,n)?"fit":"holdout")
             <<",\"traffic_bytes\":"<<std::size_t(n)*sizeof(real)*3
             <<",\"working_set_bytes\":"<<std::size_t(n)*sizeof(real)*3
             <<",\"agreement_passed\":"<<(agrees?"true":"false");
}
struct sample {std::uint64_t repetitions=0,launches=0;double event_seconds=0,enqueue_seconds=0,wall_seconds=0;};
static sample measure(buffers& data) {
    cudaEvent_t begin=nullptr,end=nullptr;
    check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    sample result;
    const auto start=clock_type::now();
    try {
        check(cudaEventRecord(begin,data.stream));
        do {
            for(int index=0;index<16;++index) {
                const auto enqueue=clock_type::now();
                data.execute(result.launches);
                result.enqueue_seconds+=std::chrono::duration<double>(clock_type::now()-enqueue).count();
                ++result.repetitions;
            }
            check(cudaStreamSynchronize(data.stream));
        } while(std::chrono::duration<double>(clock_type::now()-start).count()<.2);
        check(cudaEventRecord(end,data.stream));check(cudaEventSynchronize(end));
        float milliseconds=0;check(cudaEventElapsedTime(&milliseconds,begin,end));
        result.event_seconds=milliseconds*.001;
        result.wall_seconds=std::chrono::duration<double>(clock_type::now()-start).count();
    } catch(...) {cudaEventDestroy(begin);cudaEventDestroy(end);throw;}
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));return result;
}
struct close_file {void operator()(std::FILE* file)const noexcept {std::fclose(file);}};
static int run(int argc,char** argv) {
    if(argc!=3||std::size(gpu_recipes)!=17)return 2;
    char* end=nullptr;const long selected=std::strtol(argv[1],&end,10);
    const std::string mode=argv[2];
    if(!end||end==argv[1]||*end||selected<0||selected>=65536||
       (mode!="--identity"&&mode!="--preflight"&&mode!="--measure"))return 2;
    check(cudaSetDevice(int(selected)));
    cudaDeviceProp properties{};check(cudaGetDeviceProperties(&properties,int(selected)));
    int runtime=0,driver=0;check(cudaRuntimeGetVersion(&runtime));check(cudaDriverGetVersion(&driver));
    std::ostringstream uuid;uuid<<"GPU-"<<std::hex<<std::setfill('0');
    for(int i=0;i<16;++i) {if(i==4||i==6||i==8||i==10)uuid<<'-';
        uuid<<std::setw(2)<<unsigned(static_cast<unsigned char>(properties.uuid.bytes[i]));}
    std::cout<<std::setprecision(17)<<"{\"kind\":\"numerical_gpu_execution_identity_v1\","
        <<"\"protocol_id\":\"original-source-production-cuda-evidence-v1\",\"backend_id\":\"production-cuda-kernels-v1\","
        <<"\"generator_id\":"<<quoted(gpu_generator)<<",\"source_generator_id\":"<<quoted(gpu_source_generator)
        <<",\"registry_id\":"<<quoted(gpu_registry)<<",\"precision_bits\":"<<sizeof(real)*8
        <<",\"device_ordinal\":"<<selected<<",\"gpu_name\":"<<quoted(properties.name)<<",\"gpu_uuid\":"<<quoted(uuid.str().c_str())
        <<",\"compute_capability\":\""<<properties.major<<'.'<<properties.minor<<"\",\"cuda_runtime_version\":"<<runtime
        <<",\"driver_version\":"<<driver<<",\"sampling_cpu_affinity\":[";
    cpu_set_t affinity;CPU_ZERO(&affinity);if(sched_getaffinity(0,sizeof(affinity),&affinity))return 2;
    bool first=true;for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&affinity)) {
        if(!first)std::cout<<',';first=false;std::cout<<cpu;
    }
    std::cout<<"]}\n"<<std::flush;
    if(mode=="--identity")return 0;
    for(const auto& recipe:gpu_recipes) {
        for(const char* control:{"tiny","cell"}) {
            const std::vector<int> sizes=std::string(control)=="tiny"?std::vector<int>{0,1,8,17}:
                std::vector<int>(recipe.sizes,recipe.sizes+recipe.size_count);
            for(int n:sizes) {
                buffers data(recipe,n);std::uint64_t launches=0;data.execute(launches);
                const bool agrees=data.agrees()&&launches==std::uint64_t(n>0);
                coordinate(std::cout,"numerical_gpu_execution_preflight_v1",recipe,n,agrees);
                std::cout<<",\"control\":"<<quoted(control)<<",\"launches\":"<<launches<<"}\n"<<std::flush;
                if(!agrees)return 3;
            }
        }
    }
    if(mode=="--preflight")return 0;
    // Exclusive, flushed per batch. An interrupted attempt is never retried.
    std::unique_ptr<std::FILE,close_file> raw(std::fopen("gpu-raw-samples.jsonl","wx"));
    if(!raw)return 4;
    unsigned order=0;
    for(int batch=0;batch<7;++batch)for(const auto& recipe:gpu_recipes)for(std::size_t index=0;index<recipe.size_count;++index) {
        const int n=recipe.sizes[index];buffers data(recipe,n);
        std::uint64_t warm_launches=0;data.execute(warm_launches);check(cudaStreamSynchronize(data.stream));
        const sample measured=measure(data);
        const bool agrees=data.agrees()&&measured.launches==measured.repetitions;
        std::ostringstream line;line<<std::setprecision(17);
        coordinate(line,"numerical_gpu_execution_batch_v1",recipe,n,agrees);
        line<<",\"batch\":"<<batch<<",\"global_order\":"<<order++<<",\"repetitions\":"<<measured.repetitions
            <<",\"launches\":"<<measured.launches<<",\"resident_event_seconds\":"<<measured.event_seconds
            <<",\"enqueue_seconds\":"<<measured.enqueue_seconds<<",\"wall_seconds\":"<<measured.wall_seconds<<"}";
        const auto text=line.str();
        if(std::fprintf(raw.get(),"%s\n",text.c_str())<0||std::fflush(raw.get()))return 4;
        std::cout<<text<<'\n'<<std::flush;if(!agrees)return 3;
    }
    return order==679?0:4;
}
int main(int argc,char** argv) {
    try {return run(argc,argv);}catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
