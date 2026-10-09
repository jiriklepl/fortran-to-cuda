// Generic offline costs for the public common-runtime API, with no application.
#include "scoped_runtime.h"
#include "staging.hpp"
#include "section_copy.hpp"
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
static void transfer_cost(const char *name, const std::vector<double> &durations,
                          const std::string &extra = "") {
    record("\"kind\":\"scoped_transfer\",\"name\":" + quoted(name) + extra, durations);
}
__global__ void empty_kernel() {}

static void staging_check(const fort_staging::Result &result) {
    CUDA(result.status);
    if (result.exhausted || !result.slots) {
        std::cerr << "offline staging resources unavailable\n";
        std::exit(2);
    }
}
static void release_staging(std::unique_ptr<fort_staging::Slots> slots) {
    size_t freed = 0;
    CUDA(fort_staging::release(std::move(slots), freed));
}
// Use the same physical copy plan and tiled row order as scoped transfers.
// No logical bounds or application-specific indexing enters this benchmark.
static __attribute__((noinline)) void copy_rows(const fort_physical::CopyPlan &plan,
                                                unsigned char *packed, unsigned char *original,
                                                size_t capacity, bool unpack) {
    size_t compact_offset = 0;
    const bool ok = plan.visit([&](const fort_physical::CopyOperation &operation) {
        return fort_physical::visit_tiles(operation, capacity, [&](const fort_physical::CopyOperation &tile) {
            for (size_t z = 0; z < tile.depth; ++z) {
                for (size_t y = 0; y < tile.height; ++y) {
                    auto *root = original + tile.offset + z * tile.pitch * tile.physical_height + y * tile.pitch;
                    if (unpack) std::memcpy(root, packed + compact_offset, tile.width);
                    else std::memcpy(packed + compact_offset, root, tile.width);
                    compact_offset += tile.width;
                }
            }
            return true;
        });
    });
    if (!ok || compact_offset != plan.bytes) {
        std::cerr << "invalid offline physical packing plan\n";
        std::exit(2);
    }
}

static void measure_staging() {
    const size_t payloads[] = {256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024};
    for (const size_t payload : payloads) {
        auto acquire_release = [&] {
            auto acquired = fort_staging::acquire(payload, fort_staging::Role::Staging, false);
            staging_check(acquired);
            release_staging(std::move(acquired.slots));
        };
        auto trim = [] {
            size_t freed = 0;
            CUDA(fort_staging::trim_cache(freed));
        };
        const auto bytes = ",\"bytes\":" + std::to_string(payload);
        transfer_cost("staging_cold_seconds", measure(4, trim, acquire_release, [] {}), bytes);
        // The cold sample left an exact-capacity complete pair cached.
        transfer_cost("staging_reuse_seconds", measure(128, acquire_release), bytes);
        auto acquired = fort_staging::acquire(payload, fort_staging::Role::Staging, false);
        staging_check(acquired);
        auto &slot = acquired.slots->slots[0];
        std::vector<unsigned char> original(2 * payload, 7);
        std::memset(slot.host, 11, payload);
        for (const bool thin : {false, true}) {
            // One-byte physical faces cover mixed logical/integer captures as
            // well as the real precision selected for compute calibration.
            const size_t element_bytes = thin ? 1 : sizeof(real);
            const size_t rows = thin ? payload : 1;
            const std::vector<size_t> extents = thin ? std::vector<size_t>{2, rows}
                                                    : std::vector<size_t>{payload / sizeof(real)};
            const std::vector<size_t> lower(extents.size(), 0);
            const std::vector<size_t> upper = thin ? std::vector<size_t>{1, rows} : extents;
            const fort_physical::CopyPlan plan(element_bytes, extents, lower, upper);
            const auto geometry = bytes + ",\"geometry\":" + quoted(thin ? "thin_rows" : "contiguous")
                                  + ",\"rows\":" + std::to_string(rows);
            transfer_cost("pack_bytes_per_second", measure(4, [&] {
                copy_rows(plan, slot.host, original.data(), payload, false);
            }), geometry);
            transfer_cost("unpack_bytes_per_second", measure(4, [&] {
                copy_rows(plan, slot.host, original.data(), payload, true);
            }), geometry);
        }
        release_staging(std::move(acquired.slots));
    }
    auto acquired = fort_staging::acquire(payloads[0], fort_staging::Role::Staging, false);
    staging_check(acquired);
    auto &first = acquired.slots->slots[0];
    auto &second = acquired.slots->slots[1];
    transfer_cost("event_record_seconds", measure(128, [] {}, [&] {
        CUDA(cudaEventRecord(first.complete, first.stream));
    }, [&] { CUDA(cudaEventSynchronize(first.complete)); }));
    transfer_cost("event_wait_seconds", measure(128, [&] {
        CUDA(cudaEventRecord(first.complete, first.stream));
        CUDA(cudaStreamSynchronize(first.stream));
    }, [&] { CUDA(cudaEventSynchronize(first.complete)); }, [] {}));
    transfer_cost("ready_event_seconds", measure(128, [] {}, [&] {
        CUDA(cudaEventRecord(first.complete, first.stream));
        CUDA(cudaStreamWaitEvent(second.stream, first.complete, 0));
    }, [&] { CUDA(cudaStreamSynchronize(second.stream)); }));
    release_staging(std::move(acquired.slots));
    size_t freed = 0;
    CUDA(fort_staging::trim_cache(freed));
}

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

static void measure_batch_preparation(int device, const fort_scope_layout &layout) {
    fort_scope_t context = 0;
    fort_buffer_t buffer = 0;
    SCOPE(fort_scope_create(device, &context));
    SCOPE(fort_scope_set_transfers(context, FORT_SCOPE_TRANSFERS_PIPELINED));
    SCOPE(fort_scope_register(context, 1, 1, &layout, 1, &buffer));
    fort_scope_plan_costs base{};
    base.version = FORT_SCOPE_PLANNING_ABI_VERSION; base.valid = 1;
    base.max_allocation_bytes = size_t(1024) * 1024 * 1024;
    // As in the existing planner benchmark, coefficients below only exercise
    // viable alternatives. Only measured duration/work enters the profile.
    base.cpu_flops = base.cpu_bandwidth = 1e9;
    base.gpu_flops = base.gpu_bandwidth = 1e11;
    base.h2d_bandwidth = base.d2h_bandwidth = 1e10;
    base.h2d_latency = base.d2h_latency = 5e-6;
    base.create_seconds = base.register_seconds = base.host_access_seconds = 1e-6;
    base.device_access_seconds = base.gpu_setup_seconds = base.allocation_seconds = 1e-6;
    base.cold_driver_startup_seconds = 0.1;
    base.release_seconds = base.wait_seconds = base.launch_enqueue_seconds = 1e-6;
    base.planning_operation_seconds = 1e-9;
    fort_scope_batch_costs staging{};
    staging.version = FORT_SCOPE_BATCH_ABI_VERSION; staging.valid = 1;
    staging.async_engine_count = 1; staging.max_slot_bytes = 16 * 1024 * 1024;
    std::fill_n(staging.staging_cold_seconds, FORT_SCOPE_BATCH_CAPACITIES, 1e-6);
    std::fill_n(staging.staging_reuse_seconds, FORT_SCOPE_BATCH_CAPACITIES, 1e-7);
    staging.event_record_seconds = staging.event_wait_seconds = staging.ready_event_seconds = 1e-6;
    staging.preparation_operation_seconds = 1e-9;
    staging.pack_bytes_per_second = staging.unpack_bytes_per_second = 1e10;
    staging.pack_row_seconds = staging.unpack_row_seconds = 1e-9;
    staging.pinned_h2d_latency = staging.pinned_d2h_latency = 1e-6;
    staging.pinned_h2d_bandwidth = staging.pinned_d2h_bandwidth = 1e10;
    const size_t rank = layout.rank;
    std::vector<size_t> lower(rank, 0), upper(layout.extents, layout.extents + rank);
    // A one-plane ordinal retains the original full physical array layout.
    const auto iterations = layout.extents[rank - 1];
    upper[rank - 1] = 1;
    const fort_scope_section plane{lower.data(), upper.data()};
    const fort_scope_access ordinal_access{0, 1, &plane, 1, &plane, 1, &plane};
    const fort_scope_batch_binding binding{buffer, uint32_t(rank - 1), 1, ordinal_access};
    const fort_scope_access whole_access{FORT_SCOPE_READ_ALL | FORT_SCOPE_WRITE_ALL | FORT_SCOPE_OVERWRITE_ALL,
                                       0, nullptr, 0, nullptr, 0, nullptr};
    const fort_scope_plan_binding query_binding{buffer, whole_access};
    for (int count : {1, 4, 16}) {
        std::vector<fort_scope_batch_unit> units;
        SCOPE(fort_scope_plan_reset_mode(context, FORT_SCOPE_PLAN_CONTINUE));
        for (int unit = 0; unit < count; ++unit) {
            const auto id = uint64_t(unit + 1);
            SCOPE(fort_scope_plan_add(context, FORT_SCOPE_PLAN_WORKER, id, &query_binding, 1, 1e9, 1e8, 1));
            units.push_back({FORT_SCOPE_PLAN_WORKER, id, &binding, 1, 1e9, 1e8});
        }
        SCOPE(fort_scope_plan_validate(context));
        fort_scope_plan_decision decision{};
        SCOPE(fort_scope_plan_select(context, &base, 1, &decision));
        if (decision.gpu_units != unsigned(count)) {
            std::cerr << "offline batch preview must have an approved all-GPU chain\n";
            std::exit(2);
        }
        const fort_scope_batch batch{FORT_SCOPE_BATCH_ABI_VERSION, 1, iterations,
                                     units.data(), units.size(), nullptr, 0};
        fort_scope_batch_report report{};
        uint64_t work = 0;
        auto durations = measure(32, [&] {
            SCOPE(fort_scope_batch_execute_v1(context, &batch, &base, &staging, -1, nullptr, nullptr, &report));
            if (!report.preparation_operations || (work && work != report.preparation_operations) || report.applied) {
                std::cerr << "offline preview must report stable positive work and execute no callbacks\n";
                std::exit(2);
            }
            work = report.preparation_operations;
        });
        transfer_cost("preparation_operation_seconds", durations,
                      ",\"units\":" + std::to_string(count) + ",\"work\":" + std::to_string(work));
        // Preview must not consume the installed query. Retire its synthetic
        // metadata schedule outside the timed observations before reset/close.
        for (int unit = 0; unit < count; ++unit) {
            int gpu = 0;
            SCOPE(fort_scope_plan_next(context, uint64_t(unit + 1), &query_binding, 1, &gpu));
            if (!gpu) {
                std::cerr << "offline batch preview schedule changed\n";
                std::exit(2);
            }
        }
    }
    SCOPE(fort_scope_close(context));
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
    measure_staging();
    SCOPE(fort_scope_gpu_leave(context, previous));
    SCOPE(fort_scope_close(context));
    measure_batch_preparation(device, layout);
    return 0;
}
