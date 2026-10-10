// Standalone numerical families; no application source, inputs or timings.
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#ifdef CALIBRATION_V2
#include <array>
#include <sched.h>
#endif

#ifndef CALIBRATION_PRECISION
#define CALIBRATION_PRECISION 64
#endif
using real = std::conditional_t<CALIBRATION_PRECISION == 64, double, float>;
using clock_type = std::chrono::steady_clock;
#ifndef CALIBRATION_V2
static constexpr int samples = 5;
static constexpr const char *backend_id = "standalone-cuda-openmp-cxx17-v1";
#endif

static void check(cudaError_t value, const char *operation) {
    if (value != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(value) << '\n';
        std::exit(2);
    }
}
#define CUDA(call) check((call), #call)

static std::string quoted(const std::string &value) {
    std::ostringstream result;
    result << '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') result << '\\' << ch;
        else if (ch == '\n') result << "\\n";
        else if (ch < 32) result << '?';
        else result << ch;
    }
    result << '"';
    return result.str();
}

#ifndef CALIBRATION_V2
template<int Operation> __host__ __device__ static real primitive(real x) {
    if constexpr (Operation == 0) return real(0.25) + real(0.125) * sqrt(real(1) + real(0.25) * x * x);
    else if constexpr (Operation == 1) return real(0.25) + real(0.125) * acos(real(0.125) * x);
    else return real(0.25) + real(0.125) * cos(x);
}

template<int Family> __host__ __device__ static real work(std::size_t item) {
    real x = real(0.2) + real(item % 31) * real(0.01);
    if constexpr (Family == 0) {
        // All arguments stay away from singularities and exceptional values.
        // The recurrence prevents dead-code elimination and keeps the exact
        // dependency structure part of this family's implementation identity.
        for (int step = 0; step < 16; ++step) {
            const real angle = acos(real(0.25) * x);
            x = real(0.125) * (cos(angle) + sin(real(0.5) * x)
                               + sqrt(real(1) + x * x) + log(real(1) + real(0.25) * x * x));
        }
        return x;
    } else if constexpr (Family == 1) {
        real a[3][3], b[3][3], c[3][3];
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) {
                a[row][col] = x + real(row + col + 1) * real(0.01);
                b[row][col] = real(row == col ? 0.5 : 0.125);
            }
        for (int step = 0; step < 16; ++step) {
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col) {
                    real dot = 0;
                    for (int k = 0; k < 3; ++k) dot += a[row][k] * b[k][col];
                    c[row][col] = real(0.125) + dot;
                }
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col) a[row][col] = c[row][col];
        }
        return a[0][0] + a[1][1] + a[2][2];
    } else if constexpr (Family < 5) {
        for (int step = 0; step < 16; ++step) x = primitive<Family - 2>(x);
        return x;
    } else {
        // Independent expression mixes validate primitive predictions; they
        // never modify a primitive coefficient. Different proportions test
        // both square-root-heavy and inverse-trigonometric-heavy work.
        constexpr int roots = Family == 5 ? 8 : 2;
        constexpr int angles = Family == 5 ? 1 : 3;
        constexpr int cosines = Family == 5 ? 3 : 1;
        for (int step = 0; step < 16; ++step) {
            for (int op = 0; op < roots; ++op) x = primitive<0>(x);
            for (int op = 0; op < angles; ++op) x = primitive<1>(x);
            for (int op = 0; op < cosines; ++op) x = primitive<2>(x);
        }
        return x;
    }
}

template<int Family> __global__ void gpu_worker(real *output, std::size_t size) {
    for (std::size_t item = blockIdx.x * blockDim.x + threadIdx.x;
         item < size; item += blockDim.x * gridDim.x) output[item] = work<Family>(item);
}
template<int Family> static __attribute__((noinline)) void cpu_worker(real *output, std::size_t size, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t item = 0; item < size; ++item) output[item] = work<Family>(item);
}

static void record(const char *family, const char *device, std::size_t size,
                   const char *role, const std::vector<double> &durations) {
    std::cout << std::setprecision(17) << "{\"kind\":\"numerical_cost\",\"family\":" << quoted(family)
              << ",\"device\":" << quoted(device) << ",\"items\":" << size << ",\"role\":" << quoted(role)
              << ",\"agreement_passed\":true,\"seconds\":[";
    for (std::size_t index = 0; index < durations.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << durations[index];
    }
    std::cout << "]}\n" << std::flush;
}

template<int Family> static void measure(const char *family, int threads, real *device_output,
                                         cudaStream_t stream, cudaEvent_t begin, cudaEvent_t end) {
    // Fit and holdout roles are fixed before any observation exists. The two
    // holdouts interleave training sizes but never enter the fit.
    for (std::size_t size : {16384, 32768, 65536, 131072, 262144}) {
        const char *role = size == 32768 || size == 131072 ? "holdout" : "fit";
        std::vector<real> host(size), gpu(size);
        cpu_worker<Family>(host.data(), size, threads);
        gpu_worker<Family><<<std::min<std::size_t>((size + 255) / 256, 1024), 256, 0, stream>>>(device_output, size);
        CUDA(cudaGetLastError());
        CUDA(cudaStreamSynchronize(stream));
        CUDA(cudaMemcpy(gpu.data(), device_output, size * sizeof(real), cudaMemcpyDeviceToHost));
        const double tolerance = CALIBRATION_PRECISION == 64 ? 1e-10 : 2e-5;
        for (std::size_t item = 0; item < size; ++item) {
            if (!std::isfinite(double(host[item])) || !std::isfinite(double(gpu[item]))
                    || std::abs(double(host[item]) - double(gpu[item])) > tolerance * (1 + std::abs(double(host[item])))) {
                std::cerr << family << " CPU/GPU disagreement at " << item << '\n';
                std::exit(2);
            }
        }
        std::vector<double> cpu_durations, gpu_durations;
        for (int sample = 0; sample < samples; ++sample) {
            const auto started = clock_type::now();
            cpu_worker<Family>(host.data(), size, threads);
            cpu_durations.push_back(std::chrono::duration<double>(clock_type::now() - started).count());
            CUDA(cudaEventRecord(begin, stream));
            gpu_worker<Family><<<std::min<std::size_t>((size + 255) / 256, 1024), 256, 0, stream>>>(device_output, size);
            CUDA(cudaGetLastError());
            CUDA(cudaEventRecord(end, stream));
            CUDA(cudaEventSynchronize(end));
            float milliseconds = 0;
            CUDA(cudaEventElapsedTime(&milliseconds, begin, end));
            gpu_durations.push_back(milliseconds * 0.001);
        }
        record(family, "cpu", size, role, cpu_durations);
        record(family, "gpu", size, role, gpu_durations);
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: numerical-calibration THREADS DEVICE\n";
        return 2;
    }
    const int threads = std::stoi(argv[1]), device = std::stoi(argv[2]);
    if (threads < 1 || device < 0) return 2;
    omp_set_dynamic(0);
    int actual_threads = 0;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    if (threads != actual_threads) {
        std::cerr << "requested OpenMP thread budget unavailable\n";
        return 2;
    }
    CUDA(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CUDA(cudaGetDeviceProperties(&properties, device));
    int runtime = 0, driver = 0;
    CUDA(cudaRuntimeGetVersion(&runtime));
    CUDA(cudaDriverGetVersion(&driver));
    std::ostringstream uuid;
    uuid << "GPU-" << std::hex << std::setfill('0');
    for (int index = 0; index < 16; ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) uuid << '-';
        uuid << std::setw(2) << unsigned(static_cast<unsigned char>(properties.uuid.bytes[index]));
    }
    std::cout << "{\"kind\":\"numerical_identity\",\"backend_id\":" << quoted(backend_id)
              << ",\"gpu_name\":" << quoted(properties.name) << ",\"gpu_uuid\":" << quoted(uuid.str())
              << ",\"compute_capability\":\"" << properties.major << '.' << properties.minor
              << "\",\"cuda_runtime_version\":" << runtime << ",\"driver_version\":" << driver
              << ",\"cpu_threads\":" << actual_threads << ",\"precision_bits\":" << CALIBRATION_PRECISION << "}\n" << std::flush;
    real *output = nullptr;
    CUDA(cudaMalloc(&output, 262144 * sizeof(real)));
    cudaStream_t stream;
    cudaEvent_t begin, end;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA(cudaEventCreate(&begin));
    CUDA(cudaEventCreate(&end));
    measure<0>("transcendental_chain_v1", threads, output, stream, begin, end);
    measure<1>("private_matrix_v1", threads, output, stream, begin, end);
    measure<2>("primitive_sqrt_v1", threads, output, stream, begin, end);
    measure<3>("primitive_acos_v1", threads, output, stream, begin, end);
    measure<4>("primitive_cos_v1", threads, output, stream, begin, end);
    measure<5>("angle_mix_v1", threads, output, stream, begin, end);
    measure<6>("angle_mix_skew_v1", threads, output, stream, begin, end);
    CUDA(cudaEventDestroy(begin));
    CUDA(cudaEventDestroy(end));
    CUDA(cudaStreamDestroy(stream));
    CUDA(cudaFree(output));
    return 0;
}
#else
// V2 uses the actual native Fortran object plus the same scalar C++/CUDA
// expressions. The two native participation modes have distinct evidence.
extern "C" void fort_numerical_native_v2(int, std::size_t, int, int,
                                          const real *, const real *, real *);
extern "C" void fort_numerical_fortran_identity_v2(char *, char *, int);
static constexpr int compute_samples = 7;
static constexpr double minimum_batch_seconds = 0.2;
static constexpr const char *compute_backend = "fortran-cxx17-cuda-roofline-v2";
static constexpr const char *families_v2[] = {"arithmetic_v2", "memory_v2", "primitive_sqrt_v2",
    "primitive_acos_v2", "primitive_cos_v2", "scalar_mix_v2", "scalar_mix_skew_v2",
    "private_mix_v2", "private_mix_skew_v2", "primitive_divide_v2", "ordinary_mix_v2"};
static constexpr const char *backends_v2[] = {"native_serial", "native_fork_join", "generated_cpu", "gpu"};

template<int Operation> __host__ __device__ static real primitive_v2(real x) {
    if constexpr (Operation == 0) return real(.25) + real(.125) * sqrt(real(1) + real(.25) * x * x);
    else if constexpr (Operation == 1) return real(.25) + real(.125) * acos(real(.125) * x);
    else return real(.25) + real(.125) * cos(x);
}
template<int Family> __host__ __device__ static real work_v2(real x, real other) {
    if constexpr (Family == 0) {
        real x1=x, x2=x*real(.5), x3=x*real(.25), x4=x*real(.125);
        for (int step=0; step<64; ++step) {
            x1=x1*real(1.000001)+real(.000001); x2=x2*real(1.000002)+real(.000002);
            x3=x3*real(1.000003)+real(.000003); x4=x4*real(1.000004)+real(.000004);
        }
        return x1+x2+x3+x4;
    } else if constexpr (Family == 1) return x+real(.25)*other;
    else if constexpr (Family < 5) {
        for (int step=0; step<16; ++step) x=primitive_v2<Family-2>(x);
        return x;
    } else if constexpr (Family == 9) {
        real x1=x,x2=x*real(.5),x3=x*real(.25),x4=x*real(.125);
        for(int step=0;step<16;++step) {
            x1=(x1+real(.125))/(other+x2*real(.25)+real(1));
            x2=(x2+real(.25))/(other+x3*real(.125)+real(1));
            x3=(x3+real(.5))/(other+x4*real(.0625)+real(1));
            x4=(x4+real(.75))/(other+x1*real(.03125)+real(1));
        }
        return x1+x2+x3+x4;
    } else if constexpr (Family == 10) {
        for(int step=0;step<64;++step) {
            x=x*real(1.000001)+real(.000001);
            x=-fabs(x);
            x=real(-.1)<x ? real(-.1) : x;
            x=x<real(-.9) ? real(-.9) : x;
            x=(x-real(.1))/(real(-1)-other);
        }
        return x;
    } else {
        real private_value=0;
        if constexpr (Family >= 7) {
            constexpr int width=Family==7 ? 4 : 2;
            real a[width][width],b[width][width],c[width][width],d[width][width];
            for(int row=0;row<width;++row) for(int col=0;col<width;++col) {
                a[row][col]=x+real(row+col+2)*real(.01);
                b[row][col]=real(.02)*x+real(row==col ? 1 : 0);
            }
            for(int row=0;row<width;++row) for(int col=0;col<width;++col) {
                real total=0;
                for(int k=0;k<width;++k) total+=a[row][k]*b[k][col];
                c[row][col]=total; d[row][col]=c[row][col]+a[row][col];
            }
            real diagonal=d[0][0];for(int k=1;k<width;++k) diagonal+=d[k][k];
            private_value=real(.125)+real(.01)*diagonal;
            x=private_value;
        }
        constexpr int roots = Family == 5 || Family == 7 ? 6 : 2;
        constexpr int angles = Family == 5 || Family == 7 ? 2 : 4;
        constexpr int cosines = Family == 5 || Family == 7 ? 5 : 1;
        for(int step=0;step<16;++step) {
            for(int op=0;op<roots;++op) x=primitive_v2<0>(x);
            for(int op=0;op<angles;++op) x=primitive_v2<1>(x);
            for(int op=0;op<cosines;++op) x=primitive_v2<2>(x);
            x=-fabs(x);
            x=real(-.1)<x ? real(-.1) : x;
            x=x<real(-.9) ? real(-.9) : x;
            x=(x-real(.1))/(real(-1)-other);
        }
        return x+real(.001)*private_value;
    }
}
template<int Family> __global__ void gpu_v2(const real *a,const real *b,real *out,std::size_t n) {
    for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)
        out[i]=work_v2<Family>(a[i],b[i]);
}
template<int Family> static __attribute__((noinline)) void cpu_v2(const real *a,const real *b,real *out,std::size_t n,int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(std::size_t i=0;i<n;++i) out[i]=work_v2<Family>(a[i],b[i]);
}
struct batch_v2 { unsigned long long repetitions=0; double elapsed_seconds=0,wall_seconds=0; };
template<int Family> static void measure_v2(int threads,const std::vector<real>& a,const std::vector<real>& b,
        real *da,real *db,real *dout,cudaStream_t stream,cudaEvent_t begin,cudaEvent_t end,bool smoke) {
    const std::vector<std::size_t> sizes=smoke ? std::vector<std::size_t>{16,257} :
        Family == 1 ? std::vector<std::size_t>{65536,80265,98304,120397,131072,147456,180596,221184,
            270894,331776,406341,497664,524288,609511,746496,914267,1119744} :
        std::vector<std::size_t>{65536,131072,262144,524288,1048576};
    for(std::size_t n : sizes) {
        const bool holdout=Family == 1 ? !(n==65536 || n==98304 || n==147456 || n==221184 ||
            n==331776 || n==497664 || n==746496 || n==1119744) : n==131072 || n==524288;
        std::vector<real> reference(n),host(n),gpu(n);
        fort_numerical_native_v2(Family,n,threads,0,a.data(),b.data(),reference.data());
        auto launch=[&] { gpu_v2<Family><<<std::min<std::size_t>((n+255)/256,4096),256,0,stream>>>(da,db,dout,n); };
        bool agreement[4]={true,true,true,true};
        const double tolerance=CALIBRATION_PRECISION==64 ? 1e-10 : 2e-5;
        for(int backend=1;backend<4;++backend) {
            if(backend==1) fort_numerical_native_v2(Family,n,threads,1,a.data(),b.data(),host.data());
            if(backend==2) cpu_v2<Family>(a.data(),b.data(),host.data(),n,threads);
            if(backend==3) {
                launch(); CUDA(cudaGetLastError()); CUDA(cudaStreamSynchronize(stream));
                CUDA(cudaMemcpy(gpu.data(),dout,n*sizeof(real),cudaMemcpyDeviceToHost)); host=gpu;
            }
            for(std::size_t i=0;i<n;++i) if(!std::isfinite(double(reference[i])) || !std::isfinite(double(host[i])) ||
                    std::abs(double(reference[i])-double(host[i]))>tolerance*(1+std::abs(double(reference[i])))) {
                agreement[backend]=false; break;
            }
        }
        if(smoke) {
            const bool passed=agreement[0]&&agreement[1]&&agreement[2]&&agreement[3];
            std::cout<<"{\"kind\":\"compute_smoke_v2\",\"family\":"<<quoted(families_v2[Family])
                <<",\"items\":"<<n<<",\"agreement_passed\":"<<(passed?"true":"false")<<"}\n"<<std::flush;
            if(!passed) std::exit(3);
            continue;
        }
        std::array<std::array<batch_v2,compute_samples>,4> observations;
        for(int sample=0;sample<compute_samples;++sample) {
            // Deterministic rotation interleaves all backends. Never retry a
            // completed batch or change its role in response to observations.
            for(int order=0;order<4;++order) {
                const int backend=(sample+order)%4;
                auto &row=observations[backend][sample];
                const auto started=clock_type::now();
                if(backend==3) {
                    CUDA(cudaEventRecord(begin,stream));
                    do {
                        for(int k=0;k<16;++k) {launch();++row.repetitions;}
                        CUDA(cudaGetLastError()); CUDA(cudaStreamSynchronize(stream));
                        row.wall_seconds=std::chrono::duration<double>(clock_type::now()-started).count();
                    } while(row.wall_seconds<minimum_batch_seconds);
                    CUDA(cudaEventRecord(end,stream)); CUDA(cudaEventSynchronize(end));
                    float milliseconds=0; CUDA(cudaEventElapsedTime(&milliseconds,begin,end));
                    row.elapsed_seconds=milliseconds*.001;
                    row.wall_seconds=std::chrono::duration<double>(clock_type::now()-started).count();
                } else {
                    do {
                        if(backend<2) fort_numerical_native_v2(Family,n,threads,backend,a.data(),b.data(),host.data());
                        else cpu_v2<Family>(a.data(),b.data(),host.data(),n,threads);
                        ++row.repetitions;
                        row.wall_seconds=std::chrono::duration<double>(clock_type::now()-started).count();
                    } while(row.wall_seconds<minimum_batch_seconds);
                    row.elapsed_seconds=row.wall_seconds;
                }
            }
        }
        for(int backend=0;backend<4;++backend) {
            std::cout<<std::setprecision(17)<<"{\"kind\":\"compute_cost_v2\",\"family\":"<<quoted(families_v2[Family])
              <<",\"backend\":"<<quoted(backends_v2[backend])<<",\"items\":"<<n
              <<",\"role\":"<<quoted(holdout ? "holdout" : "fit")<<",\"agreement_passed\":"<<(agreement[backend]?"true":"false")
              ;
            if constexpr (Family == 1)
                std::cout<<",\"working_set_bytes\":"<<3*n*sizeof(real)<<",\"traffic_bytes\":"<<3*n*sizeof(real);
            std::cout<<",\"samples\":[";
            for(int sample=0;sample<compute_samples;++sample) {
                if(sample) std::cout<<',';
                const auto &row=observations[backend][sample];
                std::cout<<"{\"batch\":"<<sample<<",\"repetitions\":"<<row.repetitions
                  <<",\"elapsed_seconds\":"<<row.elapsed_seconds<<",\"wall_seconds\":"<<row.wall_seconds<<'}';
            }
            std::cout<<"]}\n"<<std::flush;
        }
    }
}
int main(int argc,char **argv) {
    if(argc!=3 && !(argc==4 && std::string(argv[3])=="--smoke")) {
        std::cerr<<"usage: numerical-calibration-v2 THREADS DEVICE [--smoke]\n";return 2;
    }
    const bool smoke=argc==4;
    int threads=std::stoi(argv[1]),device=std::stoi(argv[2]); if(threads<1||device<0) return 2;
    omp_set_dynamic(0); int actual=0;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        actual=omp_get_num_threads();
    }
    if(actual!=threads) {std::cerr<<"requested thread budget unavailable\n";return 2;}
    cpu_set_t affinity; CPU_ZERO(&affinity);
    if(sched_getaffinity(0,sizeof(affinity),&affinity)!=0 || CPU_COUNT(&affinity)!=threads) {
        std::cerr<<"fixed CPU affinity must match thread budget\n";return 2;
    }
    CUDA(cudaSetDevice(device)); cudaDeviceProp p{}; CUDA(cudaGetDeviceProperties(&p,device));
    int runtime=0,driver=0; CUDA(cudaRuntimeGetVersion(&runtime));CUDA(cudaDriverGetVersion(&driver));
    std::ostringstream uuid;uuid<<"GPU-"<<std::hex<<std::setfill('0');
    for(int i=0;i<16;++i) {if(i==4||i==6||i==8||i==10)uuid<<'-';uuid<<std::setw(2)<<unsigned(static_cast<unsigned char>(p.uuid.bytes[i]));}
    char version[8192]={},options[8192]={};fort_numerical_fortran_identity_v2(version,options,8192);
    std::cout<<"{\"kind\":\"compute_identity_v2\",\"backend_id\":"<<quoted(compute_backend)
      <<",\"gpu_name\":"<<quoted(p.name)<<",\"gpu_uuid\":"<<quoted(uuid.str())
      <<",\"compute_capability\":\""<<p.major<<'.'<<p.minor<<"\",\"cuda_runtime_version\":"<<runtime
      <<",\"driver_version\":"<<driver<<",\"cpu_threads\":"<<actual<<",\"precision_bits\":"<<CALIBRATION_PRECISION
      <<",\"fortran\":{\"compiler_version\":"<<quoted(version)<<",\"compiler_options\":"<<quoted(options)<<"},\"cpu_affinity\":[";
    bool first=true;for(int i=0;i<CPU_SETSIZE;++i)if(CPU_ISSET(i,&affinity)){if(!first)std::cout<<',';first=false;std::cout<<i;}
    std::cout<<"]}\n"<<std::flush;
    constexpr std::size_t max_n=1119744;
    std::vector<real>a(max_n),b(max_n);
    for(std::size_t i=0;i<max_n;++i){a[i]=real(.2)+real(i%31)*real(.01);b[i]=real(.125)+real(i%19)*real(.005);}
    real *da=nullptr,*db=nullptr,*out=nullptr;
    CUDA(cudaMalloc(&da,max_n*sizeof(real)));CUDA(cudaMalloc(&db,max_n*sizeof(real)));CUDA(cudaMalloc(&out,max_n*sizeof(real)));
    CUDA(cudaMemcpy(da,a.data(),max_n*sizeof(real),cudaMemcpyHostToDevice));CUDA(cudaMemcpy(db,b.data(),max_n*sizeof(real),cudaMemcpyHostToDevice));
    cudaStream_t stream;cudaEvent_t begin,end;
    CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));CUDA(cudaEventCreate(&begin));CUDA(cudaEventCreate(&end));
    measure_v2<0>(threads,a,b,da,db,out,stream,begin,end,smoke);measure_v2<1>(threads,a,b,da,db,out,stream,begin,end,smoke);
    measure_v2<2>(threads,a,b,da,db,out,stream,begin,end,smoke);measure_v2<3>(threads,a,b,da,db,out,stream,begin,end,smoke);
    measure_v2<4>(threads,a,b,da,db,out,stream,begin,end,smoke);measure_v2<5>(threads,a,b,da,db,out,stream,begin,end,smoke);
    measure_v2<6>(threads,a,b,da,db,out,stream,begin,end,smoke);measure_v2<7>(threads,a,b,da,db,out,stream,begin,end,smoke);
    measure_v2<8>(threads,a,b,da,db,out,stream,begin,end,smoke);
    measure_v2<9>(threads,a,b,da,db,out,stream,begin,end,smoke);
    measure_v2<10>(threads,a,b,da,db,out,stream,begin,end,smoke);
    CUDA(cudaEventDestroy(begin));CUDA(cudaEventDestroy(end));CUDA(cudaStreamDestroy(stream));
    CUDA(cudaFree(da));CUDA(cudaFree(db));CUDA(cudaFree(out));return 0;
}
#endif
