// Generic offline costs for the public common-runtime API, with no application.
#include "scoped_runtime.h"
#include "staging.hpp"
#include "section_copy.hpp"
#include "section_batch_cost.hpp"
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
// Match the compiler's graph owner lifecycle, including its caller-owned
// domain. Private-context timings omit this metadata and cannot price it.
static int resident_create(int device, fort_scope_t *out) {
    fort_scope_domain_t domain = 0;
    int status = fort_scope_domain_create_v1(device, &domain);
    if (status) return status;
    status = fort_scope_create_in_domain_v1(domain, out);
    if (status) (void)fort_scope_domain_close_v1(domain);
    return status;
}
static int resident_close(fort_scope_t context) {
    fort_scope_domain_t domain = 0;
    int status = fort_scope_domain_get_v1(context, &domain);
    if (status) return status;
    status = fort_scope_close(context);
    if (status) return status;
    return fort_scope_domain_close_v1(domain);
}
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

// The same ordinal CopyPlans are rebuilt inside both complete-path clocks.
// Views share one owned allocation; no application resource is registered.
struct PhysicalView {
    size_t base;
    std::vector<size_t> extents,lower,upper;
};
struct PhysicalGeometry {
    unsigned word,expected_layout;
    size_t physical;
    std::vector<PhysicalView> views;
};
struct PreparedPhysical {
    std::vector<fort_physical::CopyOperation> operations;
    fort_physical::compact::Plan plan;
};
static PreparedPhysical prepare_physical(const PhysicalGeometry &g) {
    PreparedPhysical out;
    for(const auto &v:g.views) {
        const fort_physical::CopyPlan logical(g.word,v.extents,v.lower,v.upper);
        if(!logical.valid || !logical.visit([&](fort_physical::CopyOperation o) {
            o.offset=fort_physical::compact::add(v.base,o.offset);out.operations.push_back(o);return true;
        }))throw std::runtime_error("invalid physical calibration ordinal view");
    }
    out.plan=fort_physical::compact::make_plan(out.operations,g.physical,fort_physical::compact::capacities[3]);
    if(out.plan.word_bytes!=g.word || fort_physical::compact::layout_mask(out.plan)!=g.expected_layout)
        throw std::runtime_error("physical calibration word/layout differs");
    return out;
}
static std::vector<PhysicalGeometry> physical_geometries() {
    struct Shape {size_t count,width,height,depth;};
    const Shape shapes[3][10]={
        {{2,1,1,1},{4,128,1,1},{49,64,1,1},{2,131072,1,1},{},{},{1,262144,1,1},{1,524288,1,1},{1,1048576,1,1},{1,2097152,1,1}},
        {{2,1,8,1},{4,8,64,1},{49,128,32,1},{2,512,256,1},{},{}},
        {{2,1,8,4},{4,8,16,16},{49,64,4,4},{2,256,64,64},{16,1,64,64},{64,1,8,8}}};
    std::vector<PhysicalGeometry> result;
    for(const unsigned word:{1u,4u,8u})for(unsigned kind=0;kind<3;++kind)for(const auto &s:shapes[kind]) {
        if(!s.count || (s.count==1 && word!=8))continue;
        const size_t pitch=s.width+(kind?64:0),height=s.height+(kind==2?2:0);
        const size_t stride=(pitch*height*s.depth+64)*word;
        PhysicalGeometry g{word,1u<<kind,word+s.count*stride,{}};
        for(size_t i=0;i<s.count;++i)
            g.views.push_back({word+i*stride,{pitch,height,s.depth},{0,0,0},{s.width,s.height,s.depth}});
        result.push_back(std::move(g));
    }
    for(const unsigned word:{1u,4u,8u})for(const unsigned layout:{6u,7u})for(const bool large:{false,true}) {
        const size_t groups=large?4:1,block=large?64:16,h=large?16:4,d=h;
        const std::vector<size_t> extents{large?65u:17u,large?67u:19u,block*groups+1};
        PhysicalGeometry g{word,layout,extents[0]*extents[1]*extents[2]*word,{}};
        for(size_t group=0;group<groups;++group) {
            const size_t z=group*block;
            g.views.push_back({0,extents,{1,1,z+1},{2,1+h,z+1+d}});
            g.views.push_back({0,extents,{3,1,z+d+2},{3+h,1+h,z+d+3}});
            if(layout==7)g.views.push_back({0,extents,{1,h+2,z+d+4},{1+2*h,h+3,z+d+5}});
        }
        result.push_back(std::move(g));
    }
    // Thin disjoint columns have overlapping physical address spans. They
    // exercise exact row-interval validation work, independently of payload.
    for(const unsigned word:{1u,4u,8u}) {
        const std::vector<size_t> extents{67,34,18};
        PhysicalGeometry g{word,4,extents[0]*extents[1]*extents[2]*word,{}};
        for(size_t column=0;column<32;++column)
            g.views.push_back({0,extents,{1+2*column,1,1},{2+2*column,33,17}});
        result.push_back(std::move(g));
    }
    return result;
}
static void describe_physical_copies() {
    size_t records=0,batches=0,kernels=0,dmas=0;
    const auto geometries=physical_geometries();
    for(const auto &g:geometries) {
        const auto p=prepare_physical(g);const auto n=p.operations.size();
        const size_t compact=n>1?2:0;
        records+=2*(1+compact);batches+=16*(1+compact);
        kernels+=16*compact*p.plan.chunks.size();dmas+=16*(n+compact);
        std::cout<<"{\"kind\":\"scoped_copy_geometry\",\"version\":2,\"word_bytes\":"<<g.word
            <<",\"physical_bytes\":"<<g.physical<<",\"payload_bytes\":"<<p.plan.bytes<<",\"rows\":"<<p.plan.rows
            <<",\"descriptor_work\":"<<p.plan.work<<",\"kernel_chunks\":"<<p.plan.chunks.size()
            <<",\"layout_mask\":"<<g.expected_layout<<",\"operations\":[";
        for(size_t i=0;i<n;++i) {const auto &o=p.operations[i];if(i)std::cout<<',';
            std::cout<<'['<<o.offset<<','<<o.width<<','<<o.height<<','<<o.depth<<','<<o.pitch<<','<<o.physical_height<<']';}
        std::cout<<"]}\n";
    }
    std::cout<<"{\"kind\":\"scoped_copy_protocol\",\"version\":2,\"geometries\":"<<geometries.size()
        <<",\"records\":"<<records<<",\"verified_batches\":"<<batches<<",\"copy_kernel_launches\":"<<kernels
        <<",\"transfer_dma_calls\":"<<dmas<<",\"excluded_warmups_per_path\":1,\"retained_rotated_rounds\":7}\n";
}
// One excluded warmup and seven rotated direct/cold/reuse rounds per geometry
// and direction. Initialization, verification, cache trimming and reuse priming
// are outside the path clock. Plan construction, actual pool acquisition/release,
// pack/unpack, submissions and completion are inside it for the selected path.
static void measure_physical_copies() {
    namespace compact=fort_physical::compact;
    auto trim=[] {size_t freed=0;CUDA(fort_staging::trim_cache(freed));};
    cudaStream_t direct_stream=nullptr;CUDA(cudaStreamCreateWithFlags(&direct_stream,cudaStreamNonBlocking));
    for(const auto &g:physical_geometries()) {
        const auto reference=prepare_physical(g);const auto &ops=reference.operations;const auto &plan=reference.plan;
        const size_t physical=g.physical;
        std::ostringstream geometry;
        geometry<<",\"operations\":[";
        for(size_t i=0;i<ops.size();++i) {const auto &o=ops[i];if(i)geometry<<',';geometry<<'['<<o.offset<<','<<o.width<<','<<o.height<<','<<o.depth<<','<<o.pitch<<','<<o.physical_height<<']';}
        geometry<<"],\"physical_bytes\":"<<physical<<",\"payload_bytes\":"<<plan.bytes<<",\"rows\":"<<plan.rows
            <<",\"descriptor_work\":"<<plan.work<<",\"word_bytes\":"<<g.word<<",\"kernel_chunks\":"<<plan.chunks.size()
            <<",\"layout_mask\":"<<g.expected_layout<<",\"pending\":false";
        std::vector<unsigned char> storage(physical+32,0x5a),source(physical),expected(physical,0xa6),got(physical);
        auto *host=storage.data()+16;
        for(size_t i=0;i<physical;++i)source[i]=static_cast<unsigned char>((i*17+31)%251);
        for(const auto &o:ops)for(size_t z=0;z<o.depth;++z)for(size_t y=0;y<o.height;++y) {
            const auto offset=o.offset+z*o.pitch*o.physical_height+y*o.pitch;
            std::memcpy(expected.data()+offset,source.data()+offset,o.width);
        }
        unsigned char *device=nullptr;CUDA(cudaMalloc(reinterpret_cast<void**>(&device),physical));
        size_t requested=0;for(const auto size:compact::capacities)if(size>=plan.bytes) {requested=size;break;}
        if(!requested)throw std::runtime_error("physical calibration capacity unavailable");
        const unsigned methods=ops.size()>1?3:1;
        for(const bool upload:{true,false}) {
            std::vector<double> durations[3];
            for(int round=-1;round<7;++round)for(unsigned ordinal=0;ordinal<methods;++ordinal) {
                const unsigned method=(ordinal+unsigned(round+1))%methods; // direct, cold, reuse
                trim();
                if(method==2) {auto prime=fort_staging::acquire(requested,fort_staging::Role::Compact,false,std::numeric_limits<size_t>::max(),true);staging_check(prime);release_staging(std::move(prime.slots));}
                std::memcpy(host,source.data(),physical);
                if(!upload)std::memset(host,0xa6,physical);
                if(upload)CUDA(cudaMemset(device,0xa6,physical));
                else CUDA(cudaMemcpy(device,source.data(),physical,cudaMemcpyHostToDevice));
                CUDA(cudaDeviceSynchronize());
                const auto start=clock_type::now();
                [&] {
                    const auto prepared=prepare_physical(g); // Owned vectors + exact CopyPlan checks, timed on every path.
                    if(method==0) {
                        for(const auto &o:prepared.operations) {
                            auto *to=(upload?device:host)+o.offset;auto *from=(upload?host:device)+o.offset;
                            const auto direction=upload?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost;
                            if(o.depth==1) {
                                if(o.height==1 || o.pitch==o.width)CUDA(cudaMemcpyAsync(to,from,o.width*o.height,direction,direct_stream));
                                else CUDA(cudaMemcpy2DAsync(to,o.pitch,from,o.pitch,o.width,o.height,direction,direct_stream));
                            } else {
                                cudaMemcpy3DParms p{};p.srcPtr=make_cudaPitchedPtr(from,o.pitch,o.pitch,o.physical_height);
                                p.dstPtr=make_cudaPitchedPtr(to,o.pitch,o.pitch,o.physical_height);
                                p.extent=make_cudaExtent(o.width,o.height,o.depth);p.kind=direction;CUDA(cudaMemcpy3DAsync(&p,direct_stream));
                            }
                        }
                        CUDA(cudaStreamSynchronize(direct_stream));return;
                    }
                    auto lease=fort_staging::acquire(requested,fort_staging::Role::Compact,false,std::numeric_limits<size_t>::max(),true);staging_check(lease);
                    if(lease.reused!=(method==2) || lease.slots->capacity!=requested)throw std::runtime_error("isolated compact pool state changed");
                    auto &slot=lease.slots->slots[0];
                    auto row_copy=[&](bool unpack) {
                        size_t at=0;
                        for(const auto &o:prepared.operations)for(size_t z=0;z<o.depth;++z)for(size_t y=0;y<o.height;++y) {
                            auto *p=host+o.offset+z*o.pitch*o.physical_height+y*o.pitch;
                            if(unpack)std::memcpy(p,slot.host+at,o.width);else std::memcpy(slot.host+at,p,o.width);at+=o.width;
                        }
                        if(at!=prepared.plan.bytes)throw std::runtime_error("physical calibration packing differs");
                    };
                    if(upload)row_copy(false);
                    auto kernels=[&](bool scatter) {
                        for(const auto &chunk:prepared.plan.chunks) {
                            if(g.word==8) {if(scatter)compact::launch<uint64_t,true>(device,slot.device,chunk,slot.stream);else compact::launch<uint64_t,false>(device,slot.device,chunk,slot.stream);}
                            else if(g.word==4) {if(scatter)compact::launch<uint32_t,true>(device,slot.device,chunk,slot.stream);else compact::launch<uint32_t,false>(device,slot.device,chunk,slot.stream);}
                            else {if(scatter)compact::launch<unsigned char,true>(device,slot.device,chunk,slot.stream);else compact::launch<unsigned char,false>(device,slot.device,chunk,slot.stream);}
                            CUDA(cudaGetLastError());
                        }
                    };
                    slot.pending=true;
                    if(!upload)kernels(false);
                    CUDA(cudaMemcpyAsync(upload?slot.device:slot.host,upload?slot.host:slot.device,prepared.plan.bytes,
                                         upload?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost,slot.stream));
                    if(upload)kernels(true);
                    CUDA(cudaEventRecord(slot.complete,slot.stream));CUDA(cudaEventSynchronize(slot.complete));slot.pending=false;
                    if(!upload)row_copy(true);
                    release_staging(std::move(lease.slots));
                }();
                const auto seconds=elapsed(start);
                CUDA(cudaMemcpy(got.data(),device,physical,cudaMemcpyDeviceToHost));
                if(got!=(upload?expected:source) || std::memcmp(host,upload?source.data():expected.data(),physical))
                    throw std::runtime_error("physical calibration complete payload/canary verification failed");
                for(size_t i=0;i<16;++i)if(storage[i]!=0x5a || storage[physical+16+i]!=0x5a)
                    throw std::runtime_error("physical calibration host guard overwritten");
                if(round>=0)durations[method].push_back(seconds);
            }
            for(unsigned method=0;method<methods;++method) {
                const auto fields="\"kind\":\"scoped_copy\",\"version\":2,\"name\":"+quoted(method?"compact":"direct")+geometry.str()+
                    ",\"direction\":"+quoted(upload?"h2d":"d2h")+",\"pool_state\":"+quoted(method==0?"none":method==1?"cold":"reuse")+
                    ",\"capacity_bytes\":"+std::to_string(method?requested:0);
                record(fields,durations[method]);
            }
        }
        CUDA(cudaFree(device));
    }
    trim();CUDA(cudaStreamDestroy(direct_stream));
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
    SCOPE(resident_create(device, &context));
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
    SCOPE(resident_close(context));
}

int main(int argc, char **argv) {
    if(argc==2 && std::string(argv[1])=="--describe-physical-copies") {describe_physical_copies();return 0;}
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
    cost("create_seconds", measure(512, [] {}, [&] { SCOPE(resident_create(device, &temporary)); },
                                   [&] { SCOPE(resident_close(temporary)); }));
    fort_scope_t context = 0;
    SCOPE(resident_create(device, &context));
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
        SCOPE(resident_create(device, &temporary));
        SCOPE(fort_scope_gpu_enter(temporary, &previous, &stream_pointer));
        SCOPE(fort_scope_gpu_leave(temporary, previous));
        SCOPE(resident_close(temporary));
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
    SCOPE(resident_close(context));
    measure_batch_preparation(device, layout);
    measure_physical_copies();
    return 0;
}
