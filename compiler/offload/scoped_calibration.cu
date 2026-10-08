// Generic offline costs for the public common-runtime API, with no application.
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
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

static void cuda_check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
        std::exit(2);
    }
}
static void scope_check(int status, const char *operation) {
    if (status != FORT_SCOPE_OK) {
        std::cerr << operation << ": " << fort_scope_error() << '\n';
        std::exit(2);
    }
}
#define CUDA(call) cuda_check((call), #call)
#define SCOPE(call) scope_check((call), #call)
static double elapsed(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}
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
static void record(const std::string &fields, const std::vector<double> &durations) {
    std::cout << std::setprecision(17) << '{' << fields << ",\"seconds\":[";
    for (size_t i = 0; i < durations.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << durations[i];
    }
    std::cout << "]}\n" << std::flush;
}
template<class Before, class Operation, class After>
static std::vector<double> measure(int rounds, Before setup, Operation operation, After cleanup) {
    setup(); operation(); cleanup(); // An explicit warmup does not enter samples.
    std::vector<double> durations;
    for (int sample = 0; sample < samples; ++sample) {
        double total = 0;
        for (int round = 0; round < rounds; ++round) {
            setup();
            auto start = clock_type::now();
            operation();
            total += elapsed(start);
            cleanup();
        }
        durations.push_back(total / rounds);
    }
    return durations;
}
template<class Operation> static std::vector<double> measure(int rounds, Operation operation) {
    return measure(rounds, [] {}, operation, [] {});
}
static void cost(const char *name, const std::vector<double> &durations) {
    record("\"kind\":\"scoped_cost\",\"name\":" + quoted(name), durations);
}
__global__ void empty_kernel() {}

static void identity(int device, int threads) {
    cudaDeviceProp properties{};
    CUDA(cudaGetDeviceProperties(&properties, device));
    int runtime = 0, driver = 0;
    CUDA(cudaRuntimeGetVersion(&runtime));
    CUDA(cudaDriverGetVersion(&driver));
    std::ostringstream uuid;
    uuid << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) uuid << '-';
        uuid << std::setw(2) << unsigned(static_cast<unsigned char>(properties.uuid.bytes[i]));
    }
    std::cout << "{\"kind\":\"device\",\"gpu_name\":" << quoted(properties.name)
              << ",\"gpu_uuid\":" << quoted(uuid.str()) << ",\"compute_capability\":\""
              << properties.major << '.' << properties.minor << "\",\"cpu_threads\":" << threads
              << ",\"cuda_runtime_version\":" << runtime << ",\"driver_version\":" << driver << "}\n";
}

static void measure_planning(fort_scope_t context, fort_buffer_t buffer) {
    fort_scope_plan_costs costs{};
    costs.version = FORT_SCOPE_PLANNING_ABI_VERSION;
    costs.valid = 1;
    costs.max_allocation_bytes = size_t(1024) * 1024 * 1024;
    // Synthetic coefficients make several alternatives viable. They are not
    // emitted as measured hardware costs: only actual planner durations are.
    costs.cpu_flops = costs.cpu_bandwidth = 1e9;
    costs.gpu_flops = costs.gpu_bandwidth = 1e11;
    costs.h2d_bandwidth = costs.d2h_bandwidth = 1e10;
    costs.h2d_latency = costs.d2h_latency = 5e-6;
    costs.create_seconds = costs.register_seconds = costs.host_access_seconds = 1e-6;
    costs.device_access_seconds = costs.gpu_setup_seconds = costs.allocation_seconds = 1e-6;
    costs.cold_driver_startup_seconds = 0.1;
    costs.release_seconds = costs.wait_seconds = costs.launch_enqueue_seconds = 1e-6;
    costs.planning_operation_seconds = 1e-9;
    fort_scope_layout layout{};
    SCOPE(fort_scope_layout_get(context, buffer, &layout));
    size_t lower[8][3]{}, upper[8][3]{};
    fort_scope_section rectangles[8]{};
    for (int footprint = 0; footprint < 3; ++footprint) {
        const size_t rectangle_count = footprint == 1 ? 2 : footprint == 2 ? 8 : 0;
        for (size_t index = 0; index < rectangle_count; ++index) {
            std::copy(layout.extents, layout.extents + 3, upper[index]);
            std::fill(lower[index], lower[index] + 3, 0);
            if (footprint == 1) {
                lower[index][0] = index ? layout.extents[0] - 1 : 0;
                upper[index][0] = lower[index][0] + 1;
            } else {
                lower[index][2] = 2 * index;
                upper[index][2] = lower[index][2] + 1;
            }
            rectangles[index] = {lower[index], upper[index]};
        }
        const fort_scope_access read_write{
            footprint == 0 ? uint32_t(FORT_SCOPE_READ_ALL | FORT_SCOPE_WRITE_ALL) : 0,
            rectangle_count, rectangle_count ? rectangles : nullptr,
            rectangle_count, rectangle_count ? rectangles : nullptr, 0, nullptr};
        const fort_scope_plan_binding binding{buffer, read_write};
        for (int units : {1, 4, 16}) {
            fort_scope_plan_decision decision{};
            uint64_t work = 0;
            auto durations = measure(32, [&] {
                SCOPE(fort_scope_plan_reset(context));
                for (int unit = 0; unit < units; ++unit) {
                    const auto kind = unit % 4 == 3 ? FORT_SCOPE_PLAN_NATIVE : FORT_SCOPE_PLAN_WORKER;
                    SCOPE(fort_scope_plan_add(context, kind, uint64_t(unit + 1), &binding, 1,
                                             1e9, 1e8, kind == FORT_SCOPE_PLAN_WORKER));
                }
                SCOPE(fort_scope_plan_validate(context));
                SCOPE(fort_scope_plan_select(context, &costs, -1, &decision));
                if (!decision.simulated_operations || (work && work != decision.simulated_operations)) {
                    std::cerr << "planner must report a stable positive simulation work count\n";
                    std::exit(2);
                }
                work = decision.simulated_operations;
            });
            const char *footprint_name = footprint == 0 ? "full" : footprint == 1 ? "opposite_faces" : "eight_rectangles";
            record("\"kind\":\"scoped_planning\",\"footprint\":" + quoted(footprint_name)
                   + ",\"units\":" + std::to_string(units)
                   + ",\"work\":" + std::to_string(work)
                   + ",\"candidates\":" + std::to_string(decision.candidates), durations);
        }
    }
    SCOPE(fort_scope_plan_reset(context));
}

int main(int argc, char **argv) {
    if (argc == 3 && std::string(argv[1]) == "--cold-startup") {
        const int device = std::stoi(argv[2]);
        if (device < 0) return 2;
        const auto start = clock_type::now();
        CUDA(cudaSetDevice(device));
        CUDA(cudaFree(nullptr));
        std::cout << std::setprecision(17) << "{\"kind\":\"cold_driver_startup\",\"seconds\":"
                  << elapsed(start) << "}\n";
        return 0;
    }
    if (argc != 4) {
        std::cerr << "usage: scoped-calibration THREADS MAX_MIB DEVICE | --cold-startup DEVICE\n";
        return 2;
    }
    const int threads = std::stoi(argv[1]), max_mib = std::stoi(argv[2]), device = std::stoi(argv[3]);
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
    CUDA(cudaFree(nullptr));
    identity(device, actual_threads);
    const size_t max_bytes = size_t(max_mib) * 1024 * 1024;
    std::vector<real> host(max_bytes / sizeof(real), real(1));
    const size_t extents[] = {16, 16, host.size() / 256};
    const int64_t lower_bounds[] = {-7, -3, -5};
    const fort_scope_layout layout{3, CALIBRATION_PRECISION == 64 ? FORT_SCOPE_REAL64 : FORT_SCOPE_REAL32,
                                   sizeof(real), host.data(), extents, lower_bounds, 1};
    fort_scope_t temporary = 0;
    cost("create_seconds", measure(512, [] {}, [&] { SCOPE(fort_scope_create(device, &temporary)); },
                                   [&] { SCOPE(fort_scope_close(temporary)); }));
    fort_scope_t context = 0;
    SCOPE(fort_scope_create(device, &context));
    fort_buffer_t buffer = 0;
    cost("register_seconds", measure(512, [] {},
         [&] { SCOPE(fort_scope_register(context, 1, 1, &layout, 1, &buffer)); },
         [&] { SCOPE(fort_scope_unregister(context, buffer)); }));
    SCOPE(fort_scope_register(context, 1, 1, &layout, 1, &buffer));
    const size_t lo[] = {1, 1, 1}, hi[] = {4, 4, 4};
    const fort_scope_section rectangle{lo, hi};
    const fort_scope_access read_rectangle{0, 1, &rectangle, 0, nullptr, 0, nullptr};
    cost("host_access_seconds", measure(512, [&] {
        SCOPE(fort_scope_host_begin(context, buffer, &read_rectangle));
        SCOPE(fort_scope_host_end(context, buffer));
    }));
    int previous = 0;
    void *stream_pointer = nullptr;
    // A newly initialized source scope also owns mandatory stream/pool
    // destruction. Measure that full warm lifecycle, including close.
    cost("gpu_setup_seconds", measure(32, [&] {
        SCOPE(fort_scope_create(device, &temporary));
        SCOPE(fort_scope_gpu_enter(temporary, &previous, &stream_pointer));
        SCOPE(fort_scope_gpu_leave(temporary, previous));
        SCOPE(fort_scope_close(temporary));
    }));
    SCOPE(fort_scope_gpu_enter(context, &previous, &stream_pointer));
    auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    void *device_pointer = nullptr;
    const fort_scope_access read_all{FORT_SCOPE_READ_ALL, 0, nullptr, 0, nullptr, 0, nullptr};
    SCOPE(fort_scope_device_begin(context, buffer, &read_all, &device_pointer));
    SCOPE(fort_scope_device_end(context, buffer));
    SCOPE(fort_scope_wait(context));
    cost("device_access_seconds", measure(128, [] {}, [&] {
        SCOPE(fort_scope_device_begin(context, buffer, &read_rectangle, &device_pointer));
        SCOPE(fort_scope_device_end(context, buffer));
    }, [&] { SCOPE(fort_scope_wait(context)); }));
    cost("launch_enqueue_seconds", measure(128, [] {}, [&] {
        empty_kernel<<<1, 1, 0, stream>>>();
        CUDA(cudaGetLastError());
        SCOPE(fort_scope_note_launch(context));
    }, [&] { SCOPE(fort_scope_wait(context)); }));
    cost("wait_seconds", measure(128, [&] {
        empty_kernel<<<1, 1, 0, stream>>>();
        CUDA(cudaGetLastError());
        SCOPE(fort_scope_note_launch(context));
    }, [&] { SCOPE(fort_scope_wait(context)); }, [] {}));
    // Definitions remain empty, so allocation/release samples contain no copies.
    const fort_scope_access no_access{};
    for (size_t bytes : {size_t(4096), size_t(65536), size_t(1048576), max_bytes}) {
        const size_t count = bytes / sizeof(real);
        const int64_t origin = -11;
        const fort_scope_layout allocation_layout{1, layout.type, sizeof(real), host.data(), &count, &origin, 1};
        // The existing full host capture is unregistered before borrowing the
        // same host allocation through a differently sized allocation fixture.
        SCOPE(fort_scope_unregister(context, buffer));
        auto register_empty = [&] { SCOPE(fort_scope_register(context, 2, 1, &allocation_layout, 0, &buffer)); };
        auto allocate = [&] {
            SCOPE(fort_scope_device_begin(context, buffer, &no_access, &device_pointer));
            SCOPE(fort_scope_device_end(context, buffer));
            SCOPE(fort_scope_wait(context));
        };
        auto release = [&] { SCOPE(fort_scope_unregister(context, buffer)); SCOPE(fort_scope_wait(context)); };
        record("\"kind\":\"scoped_allocation\",\"name\":\"allocation_seconds\",\"bytes\":" + std::to_string(bytes),
               measure(16, register_empty, allocate, release));
        record("\"kind\":\"scoped_allocation\",\"name\":\"release_seconds\",\"bytes\":" + std::to_string(bytes),
               measure(16, [&] { register_empty(); allocate(); }, release, [] {}));
        SCOPE(fort_scope_register(context, 1, 1, &layout, 1, &buffer));
    }
    measure_planning(context, buffer);
    SCOPE(fort_scope_gpu_leave(context, previous));
    SCOPE(fort_scope_close(context));
    return 0;
}
