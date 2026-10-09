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

#ifndef CALIBRATION_PRECISION
#define CALIBRATION_PRECISION 64
#endif
using real = std::conditional_t<CALIBRATION_PRECISION == 64, double, float>;
using clock_type = std::chrono::steady_clock;
static constexpr int samples = 5;
static constexpr const char *backend_id = "standalone-cuda-openmp-cxx17-v1";

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
