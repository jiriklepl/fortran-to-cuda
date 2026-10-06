// Standalone offline costs: no application inputs, profiling, or tuning.
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifndef CALIBRATION_PRECISION
#define CALIBRATION_PRECISION 64
#endif
using real = std::conditional_t<CALIBRATION_PRECISION == 64, double, float>;
using clock_type = std::chrono::steady_clock;
static volatile double observed = 0;
static constexpr int compute_steps = 256;
static constexpr int samples = 5;

static void check(cudaError_t result, const char* expression) {
    if (result != cudaSuccess) {
        std::cerr << expression << ": " << cudaGetErrorString(result) << "\n";
        std::exit(2);
    }
}
#define CUDA(call) check((call), #call)

static double seconds(clock_type::time_point before) {
    return std::chrono::duration<double>(clock_type::now() - before).count();
}

static std::string quoted(const std::string& value) {
    std::ostringstream text;
    text << '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') text << '\\' << ch;
        else if (ch == '\n') text << "\\n";
        else if (ch < 32) text << "?";
        else text << ch;
    }
    text << '"';
    return text.str();
}

static void record(const std::string& fields, const std::vector<double>& durations) {
    std::cout << std::setprecision(17) << "{" << fields << ",\"seconds\":[";
    for (std::size_t i = 0; i < durations.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << durations[i];
    }
    std::cout << "]}\n" << std::flush;
}

__global__ void empty_kernel() {}

__global__ void memory_kernel(real* out, const real* a, const real* b, std::size_t n) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) out[i] = a[i] + real(0.25) * b[i];
}

__global__ void compute_kernel(real* out, std::size_t n) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        real a = real(0.1) + real(i % 31) * real(0.001);
        real b = real(0.2), c = real(0.3), d = real(0.4);
        for (int j = 0; j < compute_steps; ++j) {
            a = a * real(1.000001) + real(0.00001);
            b = b * real(1.000002) + real(0.00002);
            c = c * real(1.000003) + real(0.00003);
            d = d * real(1.000004) + real(0.00004);
        }
        out[i] = a + b + c + d;
    }
}

static __attribute__((noinline)) void cpu_compute(real* out, std::size_t n, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        real a = real(0.1) + real(i % 31) * real(0.001);
        real b = real(0.2), c = real(0.3), d = real(0.4);
        for (int j = 0; j < compute_steps; ++j) {
            a = a * real(1.000001) + real(0.00001);
            b = b * real(1.000002) + real(0.00002);
            c = c * real(1.000003) + real(0.00003);
            d = d * real(1.000004) + real(0.00004);
        }
        out[i] = a + b + c + d;
    }
}

static __attribute__((noinline)) void cpu_memory(real* out, const real* a, const real* b,
                                                std::size_t n, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + real(0.25) * b[i];
}

// One pump thread gathers/scatters a representative regular strided footprint.
static __attribute__((noinline)) void pack(real* packed, const real* array, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) packed[i] = array[4*i];
}
static __attribute__((noinline)) void unpack(real* array, const real* packed, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) array[4*i] = packed[i];
}

template<class Function> static std::vector<double> host_times(Function&& operation, real* result,
                                                              std::size_t probe) {
    std::vector<double> durations;
    operation();
    for (int iteration = 0; iteration < samples; ++iteration) {
        auto before = clock_type::now();
        operation();
        durations.push_back(seconds(before));
        observed = result[probe];
    }
    return durations;
}

template<class Function> static std::vector<double> device_times(Function&& operation, cudaStream_t stream) {
    cudaEvent_t begin, end;
    CUDA(cudaEventCreate(&begin));
    CUDA(cudaEventCreate(&end));
    operation();
    CUDA(cudaGetLastError());
    CUDA(cudaStreamSynchronize(stream));
    std::vector<double> durations;
    for (int iteration = 0; iteration < samples; ++iteration) {
        CUDA(cudaEventRecord(begin, stream));
        operation();
        CUDA(cudaGetLastError());
        CUDA(cudaEventRecord(end, stream));
        CUDA(cudaEventSynchronize(end));
        float milliseconds = 0;
        CUDA(cudaEventElapsedTime(&milliseconds, begin, end));
        durations.push_back(milliseconds * 0.001);
    }
    CUDA(cudaEventDestroy(begin));
    CUDA(cudaEventDestroy(end));
    return durations;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: calibration THREADS MAX_MIB DEVICE\n";
        return 2;
    }
    const int threads = std::stoi(argv[1]);
    const int max_mib = std::stoi(argv[2]);
    const int device = std::stoi(argv[3]);
    if (threads < 1 || max_mib < 8 || max_mib > 1024 || device < 0) return 2;
    omp_set_dynamic(0);
    int actual_threads = 0;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }
    if (actual_threads != threads) {
        std::cerr << "requested OpenMP thread budget is unavailable\n";
        return 2;
    }
    CUDA(cudaSetDevice(device));
    cudaDeviceProp properties;
    CUDA(cudaGetDeviceProperties(&properties, device));
    int runtime_version = 0, driver_version = 0;
    CUDA(cudaRuntimeGetVersion(&runtime_version));
    CUDA(cudaDriverGetVersion(&driver_version));
    std::ostringstream uuid;
    uuid << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) uuid << '-';
        uuid << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(properties.uuid.bytes[i]));
    }
    std::cout << "{\"kind\":\"device\",\"gpu_name\":" << quoted(properties.name)
              << ",\"gpu_uuid\":" << quoted(uuid.str()) << ",\"compute_capability\":\""
              << properties.major << '.' << properties.minor << "\",\"device_ordinal\":" << device
              << ",\"cuda_runtime_version\":" << runtime_version << ",\"driver_version\":" << driver_version
              << ",\"async_engine_count\":" << properties.asyncEngineCount
              << ",\"cpu_threads\":" << actual_threads << "}\n" << std::flush;

    const std::size_t bytes = std::size_t(max_mib) * 1024 * 1024;
    const std::size_t n = bytes / sizeof(real);
    const std::size_t compute_n = std::min<std::size_t>(n, 1 << 18);
    std::vector<real> a(n, real(0.5)), b(n, real(0.75)), out(n, real(0));
    real *pinned = nullptr, *device_a = nullptr, *device_b = nullptr, *device_out = nullptr;
    CUDA(cudaMallocHost(&pinned, bytes));
    std::fill(pinned, pinned+n, real(0.25));
    CUDA(cudaMalloc(&device_a, bytes));
    CUDA(cudaMalloc(&device_b, bytes));
    CUDA(cudaMalloc(&device_out, bytes));
    cudaStream_t stream;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

    std::vector<std::size_t> transfer_sizes = {4096, 65536, 1024*1024, bytes/4, bytes};
    std::sort(transfer_sizes.begin(), transfer_sizes.end());
    transfer_sizes.erase(std::unique(transfer_sizes.begin(), transfer_sizes.end()), transfer_sizes.end());
    for (bool locked : {false, true}) {
        for (bool upload : {true, false}) {
            for (std::size_t count : transfer_sizes) {
                real* host = locked ? pinned : a.data();
                auto copy = [&]() {
                    CUDA(cudaMemcpyAsync(upload ? device_a : host, upload ? host : device_a, count,
                                         upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, stream));
                    CUDA(cudaStreamSynchronize(stream));
                };
                copy();
                std::vector<double> durations;
                const int repeats = count < 1024*1024 ? 40 : 5;
                for (int iteration = 0; iteration < samples; ++iteration) {
                    auto before = clock_type::now();
                    for (int repeat = 0; repeat < repeats; ++repeat) copy();
                    durations.push_back(seconds(before) / repeats);
                }
                record("\"kind\":\"transfer\",\"direction\":\"" + std::string(upload ? "h2d" : "d2h")
                       + "\",\"memory\":\"" + (locked ? "pinned" : "pageable")
                       + "\",\"bytes\":" + std::to_string(count), durations);
            }
        }
    }

    std::vector<double> launch, pump;
    for (int iteration = 0; iteration < samples; ++iteration) {
        auto before = clock_type::now();
        for (int repeat = 0; repeat < 100; ++repeat) {
            empty_kernel<<<1, 1, 0, stream>>>();
            CUDA(cudaGetLastError());
            CUDA(cudaStreamSynchronize(stream));
        }
        launch.push_back(seconds(before) / 100);
        double total_enqueue = 0;
        for (int repeat = 0; repeat < 100; ++repeat) {
            before = clock_type::now();
            CUDA(cudaMemcpyAsync(device_a, pinned, 4096, cudaMemcpyHostToDevice, stream));
            empty_kernel<<<1, 1, 0, stream>>>();
            CUDA(cudaGetLastError());
            CUDA(cudaMemcpyAsync(pinned, device_a, 4096, cudaMemcpyDeviceToHost, stream));
            total_enqueue += seconds(before);
            CUDA(cudaStreamSynchronize(stream));
        }
        pump.push_back(total_enqueue / 100);
    }
    record("\"kind\":\"latency\",\"name\":\"launch_latency_seconds\"", launch);
    record("\"kind\":\"latency\",\"name\":\"pump_latency_seconds\"", pump);

    auto rate = [&](const std::string& name, double work, const std::vector<double>& durations) {
        std::ostringstream fields;
        fields << std::setprecision(17) << "\"kind\":\"rate\",\"name\":" << quoted(name) << ",\"work\":" << work;
        record(fields.str(), durations);
    };
    const double compute_work = double(compute_n) * (8.0 * compute_steps + 3.0);
    for (int budget : {threads, threads-1}) {
        if (budget < 1) continue;
        const std::string prefix = budget == threads ? "cpu_" : "cpu_worker_";
        rate(prefix + "flops_per_second", compute_work,
             host_times([&]() { cpu_compute(out.data(), compute_n, budget); }, out.data(), compute_n/2));
        rate(prefix + "memory_bytes_per_second", 3.0*bytes,
             host_times([&]() { cpu_memory(out.data(), a.data(), b.data(), n, budget); }, out.data(), n/2));
    }
    rate("pack_bytes_per_second", double(bytes)/4,
         host_times([&]() { pack(pinned, a.data(), n/4); }, pinned, n/8));
    rate("unpack_bytes_per_second", double(bytes)/4,
         host_times([&]() { unpack(out.data(), pinned, n/4); }, out.data(), n/2));
    rate("gpu_flops_per_second", compute_work,
         device_times([&]() { compute_kernel<<<1024, 256, 0, stream>>>(device_out, compute_n); }, stream));
    rate("gpu_memory_bytes_per_second", 3.0*bytes,
         device_times([&]() { memory_kernel<<<4096, 256, 0, stream>>>(device_out, device_a, device_b, n); }, stream));
    CUDA(cudaMemcpy(out.data(), device_out, sizeof(real), cudaMemcpyDeviceToHost));
    if (!std::isfinite(out[0]) || !std::isfinite(observed)) return 2;
    CUDA(cudaStreamDestroy(stream));
    CUDA(cudaFree(device_a));
    CUDA(cudaFree(device_b));
    CUDA(cudaFree(device_out));
    CUDA(cudaFreeHost(pinned));
    return 0;
}
