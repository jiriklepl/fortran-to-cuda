// CPU-only source numerical protocol. No application code or online tuning.
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
#include <numeric>
#include <omp.h>
#include <sched.h>
#include <set>
#include <string>
#include <vector>
#include "numerical_execution_recipes.hpp"

extern "C" void fort_numerical_execution_fortran_identity_v1(char*, char*, int);
extern "C" void fort_cpu_protocol_proof_begin(int) __attribute__((weak));
extern "C" void fort_cpu_protocol_proof_end() __attribute__((weak));
using real=execution_real;
using clock_type=std::chrono::steady_clock;
constexpr std::array<const char*,3> backends{"native_serial","native_fork_join","generated_cpu"};
constexpr std::array<int,4> smoke_sizes{0,1,8,17};

static std::string quoted(const char* text) {
    std::string result="\"";
    for (const unsigned char c:std::string(text)) {
        if(c=='"'||c=='\\'){result+='\\';result+=char(c);}
        else if(c=='\n')result+="\\n";
        else if(c=='\r')result+="\\r";
        else if(c=='\t')result+="\\t";
        else if(c<32)std::exit(2);
        else result+=char(c);
    }
    return result+'"';
}
static execution_worker worker(const execution_recipe& recipe,int backend) {
    return backend==0?recipe.native_serial:backend==1?recipe.native_fork_join:recipe.generated_cpu;
}
static bool training(const execution_recipe& r,int n) {
    return std::find(r.fit_sizes,r.fit_sizes+r.fit_count,n)!=r.fit_sizes+r.fit_count;
}
struct buffers {
    static constexpr real sentinel=real(-131.25);
    std::vector<real>a,b,output;
    const execution_recipe& recipe;
    int n;
    bool fit;
    buffers(const execution_recipe&r,int count):a(count+2),b(count+2),output(count+2),recipe(r),n(count),fit(training(r,count)) {
        a.front()=a.back()=b.front()=b.back()=sentinel;
        const auto*av=fit?r.training_a:r.holdout_a;
        const auto*bv=fit?r.training_b:r.holdout_b;
        const auto ac=fit?r.training_a_count:r.holdout_a_count;
        const auto bc=fit?r.training_b_count:r.holdout_b_count;
        for(int i=0;i<n;++i){a[i+1]=av[i%ac];b[i+1]=bv[i%bc];}
        reset();
    }
    void reset(){std::fill(output.begin(),output.end(),sentinel);}
    void execute(int backend,int threads){worker(recipe,backend)(n,threads,a.data()+1,b.data()+1,output.data()+1);}
    bool agrees()const {
        if(a.front()!=sentinel||a.back()!=sentinel||b.front()!=sentinel||b.back()!=sentinel||
           output.front()!=sentinel||output.back()!=sentinel)return false;
        const auto*av=fit?recipe.training_a:recipe.holdout_a;
        const auto*bv=fit?recipe.training_b:recipe.holdout_b;
        const auto*ref=fit?recipe.training_reference:recipe.holdout_reference;
        const auto ac=fit?recipe.training_a_count:recipe.holdout_a_count;
        const auto bc=fit?recipe.training_b_count:recipe.holdout_b_count;
        const auto rc=fit?recipe.training_reference_count:recipe.holdout_reference_count;
        constexpr double tolerance=sizeof(real)==8?2e-12:2e-5;
        for(int i=0;i<n;++i){
            const real expected=ref[i%rc],actual=output[i+1];
            if(a[i+1]!=av[i%ac]||b[i+1]!=bv[i%bc]||!std::isfinite(actual)||!std::isfinite(expected))return false;
            if(actual==expected){if(actual==real(0)&&std::signbit(actual)!=std::signbit(expected))return false;}
            else if(std::abs(double(actual)-double(expected))>tolerance*(1.+std::abs(double(expected))))return false;
        }
        return true;
    }
};
struct sample{unsigned long long repetitions{};double elapsed{};unsigned global_order{};};
static sample measure(buffers&data,int backend,int threads,unsigned order){
    sample result;result.global_order=order;
    const auto start=clock_type::now();
    do{data.execute(backend,threads);++result.repetitions;
       result.elapsed=std::chrono::duration<double>(clock_type::now()-start).count();}while(result.elapsed<.2);
    return result;
}
static void print_samples(const std::array<sample,7>&rows){
    std::cout<<'[';
    for(std::size_t i=0;i<rows.size();++i){if(i)std::cout<<',';
        std::cout<<"{\"batch\":"<<i<<",\"repetitions\":"<<rows[i].repetitions<<",\"elapsed_seconds\":"
                 <<rows[i].elapsed<<",\"wall_seconds\":"<<rows[i].elapsed<<",\"global_order\":"<<rows[i].global_order<<'}';}
    std::cout<<']';
}
static bool save_raw(std::FILE*file,const char*kind,const execution_recipe&r,int n,int backend,int batch,
                     const sample&s,bool agrees){
    const auto written=std::fprintf(file,
        "{\"kind\":%s,\"recipe\":%s,\"recipe_id\":%s,\"family\":%s,\"backend\":%s,\"items\":%d,"
        "\"role\":%s,\"batch\":%d,\"repetitions\":%llu,\"elapsed_seconds\":%.17g,\"wall_seconds\":%.17g,"
        "\"global_order\":%u,\"traffic_bytes\":%zu,\"working_set_bytes\":%zu,\"agreement_passed\":%s}\n",
        quoted(kind).c_str(),quoted(r.name).c_str(),quoted(r.identity).c_str(),quoted(r.family).c_str(),
        quoted(backends[backend]).c_str(),n,quoted(training(r,n)?"fit":"holdout").c_str(),batch,
        s.repetitions,s.elapsed,s.elapsed,s.global_order,std::size_t(n)*sizeof(real)*3,std::size_t(n)*sizeof(real)*3,agrees?"true":"false");
    return written>0&&std::fflush(file)==0;
}
static bool registry_valid(){
    if(std::size(execution_recipes)!=17||(sizeof(real)!=4&&sizeof(real)!=8))return false;
    std::set<std::string>names,ids;std::size_t cells=0;
    for(const auto&r:execution_recipes){
        if(!r.name||!r.identity||!r.role||!r.family||!r.native_serial||!r.native_fork_join||!r.generated_cpu||
           !r.sizes||!r.fit_sizes||!r.training_a||!r.training_b||!r.training_reference||
           !r.holdout_a||!r.holdout_b||!r.holdout_reference)return false;
        const std::string id(r.identity),role(r.role);
        if(!names.insert(r.name).second||!ids.insert(id).second||id.size()!=64||
           id.find_first_not_of("0123456789abcdef")!=std::string::npos||
           (role!="coefficient"&&role!="memory"&&role!="holdout"))return false;
        if(r.size_count!=(role=="memory"?17:5)||r.fit_count!=(role=="memory"?8:role=="coefficient"?3:0))return false;
        for(std::size_t i=0;i<r.size_count;++i)if(r.sizes[i]<1||r.sizes[i]>1119744||(i&&r.sizes[i]<=r.sizes[i-1]))return false;
        for(std::size_t i=0;i<r.fit_count;++i)
            if(std::find(r.sizes,r.sizes+r.size_count,r.fit_sizes[i])==r.sizes+r.size_count)return false;
        for(auto count:{r.training_a_count,r.training_b_count,r.training_reference_count,r.holdout_a_count,r.holdout_b_count,r.holdout_reference_count})
            if(count<1||count>256)return false;
        if(r.training_reference_count%r.training_a_count||r.training_reference_count%r.training_b_count||
           r.holdout_reference_count%r.holdout_a_count||r.holdout_reference_count%r.holdout_b_count)return false;
        cells+=r.size_count;
    }
    return cells==97&&std::string(execution_recipes[1].name)=="memory";
}
static void cost_record(const char*kind,const execution_recipe&r,int n,int backend,const std::array<sample,7>&rows,bool agrees){
    std::cout<<"{\"kind\":"<<quoted(kind)<<",\"recipe\":"<<quoted(r.name)<<",\"recipe_id\":"<<quoted(r.identity)
             <<",\"family\":"<<quoted(r.family)
             <<",\"backend\":"<<quoted(backends[backend])<<",\"items\":"<<n
             <<",\"role\":"<<quoted(training(r,n)?"fit":"holdout")<<",\"agreement_passed\":"<<(agrees?"true":"false")
             <<",\"traffic_bytes\":"<<std::size_t(n)*sizeof(real)*3<<",\"working_set_bytes\":"<<std::size_t(n)*sizeof(real)*3<<",\"samples\":";
    print_samples(rows);std::cout<<"}\n"<<std::flush;
}
struct close_file{void operator()(std::FILE*f)const noexcept{std::fclose(f);}};
int main(int argc,char**argv){
    if(argc<2||argc>3||!registry_valid())return 2;
    char*end=nullptr;const long parsed=std::strtol(argv[1],&end,10);
    const bool identity_only=argc==3&&std::string(argv[2])=="--identity";
    const bool smoke=argc==3&&std::string(argv[2])=="--smoke";
    const bool proof=argc==3&&std::string(argv[2])=="--proof";
    if(!end||*end||parsed!=4||(argc==3&&!identity_only&&!smoke&&!proof))return 2;
    const int threads=int(parsed);cpu_set_t affinity;CPU_ZERO(&affinity);
    if(sched_getaffinity(0,sizeof(affinity),&affinity)||CPU_COUNT(&affinity)!=threads||omp_get_dynamic()||
       omp_get_proc_bind()!=omp_proc_bind_false||omp_get_level()!=0||omp_get_thread_limit()<threads)return 2;
    const char*wait=std::getenv("OMP_WAIT_POLICY"),*spin=std::getenv("GOMP_SPINCOUNT");
    for(const auto*v:{wait,spin})if(v){if(std::strlen(v)>128)return 2;for(const unsigned char c:std::string(v))if(c<32||c>126)return 2;}
    int actual_team=0;std::array<int,4>visits{};
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        actual_team=omp_get_num_threads();
        const int id=omp_get_thread_num();if(id>=0&&id<threads)++visits[id];
    }
    if(actual_team!=threads||std::any_of(visits.begin(),visits.end(),[](int n){return n!=1;}))return 2;
    char version[8192]{},options[8192]{};fort_numerical_execution_fortran_identity_v1(version,options,8192);
    std::cout<<std::setprecision(17)<<"{\"kind\":\"numerical_execution_identity_v1\",\"protocol_id\":"<<quoted(execution_protocol_id)
             <<",\"backend_id\":"<<quoted(execution_backend_id)<<",\"generator_id\":"<<quoted(execution_generator_id)
             <<",\"registry_id\":"<<quoted(execution_registry_id)<<",\"precision_bits\":"<<sizeof(real)*8
             <<",\"cpu_threads\":"<<threads<<",\"cpu_affinity\":[";
    bool first=true;for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&affinity)){if(!first)std::cout<<',';first=false;std::cout<<cpu;}
    std::cout<<"],\"actual_team_threads\":"<<actual_team<<",\"thread_limit\":"<<omp_get_thread_limit()
             <<",\"omp_dynamic\":false,\"omp_proc_bind\":\"false\",\"omp_wait_policy\":"<<(wait?quoted(wait):"null")
             <<",\"gomp_spincount\":"<<(spin?quoted(spin):"null")<<",\"fortran\":{\"compiler_version\":"<<quoted(version)
             <<",\"compiler_options\":"<<quoted(options)<<"}}\n"<<std::flush;
    if(identity_only)return 0;
    const auto&control=execution_recipes[1];
    if(proof){
        if(!fort_cpu_protocol_proof_begin||!fort_cpu_protocol_proof_end)return 2;
        for(int n:{0,1,8})for(int backend:{1,2}){
            buffers data(control,n);fort_cpu_protocol_proof_begin(threads);data.execute(backend,threads);
            const bool agrees=data.agrees();
            std::cout<<"{\"kind\":\"numerical_execution_team_proof_v1\",\"backend\":"<<quoted(backends[backend])
                     <<",\"items\":"<<n<<",\"recipe\":\"memory\",\"recipe_id\":"<<quoted(control.identity)
                     <<",\"family\":\"memory\",\"role\":\"holdout\",\"traffic_bytes\":"<<std::size_t(n)*sizeof(real)*3
                     <<",\"working_set_bytes\":"<<std::size_t(n)*sizeof(real)*3<<",\"agreement_passed\":"<<(agrees?"true":"false");
            fort_cpu_protocol_proof_end();std::cout<<"}\n"<<std::flush;if(!agrees)return 3;
        }
        return 0;
    }
    // Every recipe/size/backend and every tiny source control agrees BEFORE
    // clocks or raw output. Allocations/preparation/checks are never timed.
    for(const auto&r:execution_recipes){
        std::vector<int>sizes(smoke_sizes.begin(),smoke_sizes.end());
        if(!smoke)sizes.insert(sizes.end(),r.sizes,r.sizes+r.size_count);
        for(int n:sizes){buffers data(r,n);for(int backend=0;backend!=3;++backend){
            data.reset();data.execute(backend,threads);const bool agrees=data.agrees();
            if(smoke)std::cout<<"{\"kind\":\"numerical_execution_smoke_v1\",\"recipe\":"<<quoted(r.name)
                             <<",\"backend\":"<<quoted(backends[backend])<<",\"items\":"<<n
                             <<",\"agreement_passed\":"<<(agrees?"true":"false")<<"}\n"<<std::flush;
            if(!agrees){std::cerr<<"numerical execution correctness failed: "<<r.name<<','<<backends[backend]<<','<<n<<'\n';return 3;}
        }}
    }
    if(smoke)return 0;
    std::unique_ptr<std::FILE,close_file>raw(std::fopen("numerical-execution-raw-samples.jsonl","wx"));
    if(!raw)return 4;
    std::array<std::array<std::array<sample,7>,2>,3>startup{};
    std::array<std::array<std::array<sample,7>,3>,97>samples{};
    std::array<std::array<bool,3>,97>agreement;for(auto&row:agreement)row.fill(true);
    unsigned order=0;
    // One global protocol. No cell receives its seven rounds consecutively.
    for(int batch=0;batch!=7;++batch){
        for(int index=0;index!=3;++index){const int n=std::array<int,3>{0,1,8}[index];buffers data(control,n);
            for(int turn=0;turn!=2;++turn){const int backend=1+(batch+turn)%2;
                auto&s=startup[index][backend-1][batch];s=measure(data,backend,threads,order++);const bool agrees=data.agrees();
                if(!save_raw(raw.get(),"numerical_execution_startup_batch_v1",control,n,backend,batch,s,agrees)||!agrees)return 4;
            }}
        std::size_t cell=0;
        for(const auto&r:execution_recipes)for(std::size_t ni=0;ni<r.size_count;++ni){const int n=r.sizes[ni];buffers data(r,n);
            for(int turn=0;turn!=3;++turn){const int backend=(batch+turn)%3;
                auto&s=samples[cell][backend][batch];s=measure(data,backend,threads,order++);const bool agrees=data.agrees();
                agreement[cell][backend]=agreement[cell][backend]&&agrees;
                if(!save_raw(raw.get(),"numerical_execution_batch_v1",r,n,backend,batch,s,agrees)||!agrees)return 4;
            }++cell;
        }
    }
    for(int index=0;index!=3;++index)for(int backend=1;backend!=3;++backend)
        cost_record("numerical_execution_startup_v1",control,std::array<int,3>{0,1,8}[index],backend,startup[index][backend-1],true);
    std::size_t cell=0;
    for(const auto&r:execution_recipes)for(std::size_t ni=0;ni<r.size_count;++ni){
        for(int backend=0;backend!=3;++backend)cost_record("numerical_execution_cost_v1",r,r.sizes[ni],backend,samples[cell][backend],agreement[cell][backend]);
        ++cell;
    }
    return 0;
}
