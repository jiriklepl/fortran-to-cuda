// The original production worker/object and measurement cadence are unchanged.
// Rename only its executable main; reuse buffers, complete checks and measure.
#define main parent_gpu_evidence_main
#include "parent_driver.cu"
#undef main

static int launch_run(int argc,char** argv) {
    if(argc!=3)return 2;
    const std::string mode=argv[2];
    if(mode!="--identity"&&mode!="--preflight"&&mode!="--measure")return 2;
    char identity_mode[]="--identity";
    char* identity_argv[]={argv[0],argv[1],identity_mode};
    const int status=run(3,identity_argv);
    if(status||mode=="--identity")return status;
    const auto found=std::find_if(std::begin(gpu_recipes),std::end(gpu_recipes),
        [](const gpu_recipe& recipe){return std::string(recipe.name)=="memory";});
    if(found==std::end(gpu_recipes))return 2;
    const auto& recipe=*found;
    for(int n:{0,1,8,17}) {
        buffers data(recipe,n);std::uint64_t launches=0;data.execute(launches);
        const bool agrees=data.agrees()&&launches==std::uint64_t(n>0);
        coordinate(std::cout,"numerical_gpu_launch_preflight_v1",recipe,n,agrees);
        std::cout<<",\"launches\":"<<launches<<"}\n"<<std::flush;
        if(!agrees)return 3;
    }
    if(mode=="--preflight")return 0;
    std::unique_ptr<std::FILE,close_file> raw(std::fopen("launch-raw-samples.jsonl","wx"));
    if(!raw)return 4;
    unsigned order=0;
    for(int batch=0;batch<7;++batch)for(int n:{1,8,17}) {
        buffers data(recipe,n);std::uint64_t warm=0;data.execute(warm);
        check(cudaStreamSynchronize(data.stream));
        const sample measured=measure(data);
        const bool agrees=data.agrees()&&measured.launches==measured.repetitions;
        std::ostringstream line;line<<std::setprecision(17);
        coordinate(line,"numerical_gpu_launch_batch_v1",recipe,n,agrees);
        line<<",\"batch\":"<<batch<<",\"global_order\":"<<order++
            <<",\"repetitions\":"<<measured.repetitions<<",\"launches\":"<<measured.launches
            <<",\"resident_event_seconds\":"<<measured.event_seconds
            <<",\"enqueue_seconds\":"<<measured.enqueue_seconds<<",\"wall_seconds\":"<<measured.wall_seconds<<"}";
        const auto text=line.str();
        if(std::fprintf(raw.get(),"%s\n",text.c_str())<0||std::fflush(raw.get()))return 4;
        std::cout<<text<<'\n'<<std::flush;
        if(!agrees)return 3;
    }
    return order==21?0:4;
}
int main(int argc,char** argv) {
    try{return launch_run(argc,argv);}
    catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
