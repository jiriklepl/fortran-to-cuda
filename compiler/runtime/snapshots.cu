#include "snapshots.h"
#include <cuda_runtime.h>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>

namespace {
enum State { free_slot, acquired, queued, writing };
struct Slot {
    void *device=nullptr, *host=nullptr;
    cudaEvent_t captured=nullptr, ready=nullptr;
    State state=free_slot;
    fort_snapshot_writer_v1 writer=nullptr;
    void *user=nullptr;
    fort_scope_t context=0;
};
struct Queue {
    fort_scope_domain_t domain=0;
    uint64_t lease=0, submitted=0, completed=0, backpressure=0;
    size_t bytes=0;
    int device=0, error=0;
    bool stop=false;
    cudaStream_t dma=nullptr;
    std::array<Slot,2> slots;
    std::array<unsigned,2> order{};
    unsigned head=0, count=0;
    std::mutex mutex;
    std::condition_variable changed;
    std::thread worker;
};
struct Device {
    int previous=0;
    bool restore=false;
    explicit Device(int ordinal) {
        if(cudaGetDevice(&previous)!=cudaSuccess || cudaSetDevice(ordinal)!=cudaSuccess)
            throw FORT_SCOPE_EXECUTION;
        restore=previous!=ordinal;
    }
    ~Device() { if(restore) cudaSetDevice(previous); }
};
void checked(cudaError_t status) { if(status!=cudaSuccess) throw FORT_SCOPE_EXECUTION; }
void consume(Queue &q) noexcept {
    for(;;) {
        unsigned index;
        {
            std::unique_lock<std::mutex> lock(q.mutex);
            q.changed.wait(lock,[&]{ return q.stop || q.count; });
            if(!q.count && q.stop) return;
            index=q.order[q.head];q.head=(q.head+1)%2;--q.count;
            q.slots[index].state=writing;
        }
        auto &s=q.slots[index];
        int error=0;
        try {
            Device device(q.device);
            checked(cudaEventSynchronize(s.ready));
            if(s.writer(s.user,s.host,q.bytes)!=FORT_SCOPE_OK) error=FORT_SCOPE_EXECUTION;
        } catch(...) { error=FORT_SCOPE_EXECUTION; }
        if(error) fort_scope_execution_error(s.context,"asynchronous snapshot DMA/writer failed; replay prohibited");
        {
            std::lock_guard<std::mutex> lock(q.mutex);
            if(error) q.error=error;
            ++q.completed;s.state=free_slot;
        }
        q.changed.notify_all();
    }
}
int release(Queue &q) noexcept {
    int error=0;
    try {
        Device device(q.device);
        // Partial submission can fail before recording a completion event.
        // Fatal cleanup must still retire every attempted capture before free.
        if(q.error && cudaDeviceSynchronize()!=cudaSuccess) error=FORT_SCOPE_EXECUTION;
        if(q.dma && cudaStreamSynchronize(q.dma)!=cudaSuccess) error=FORT_SCOPE_EXECUTION;
        for(auto &s:q.slots) {
            // A rejected submission may have enqueued capture without DMA.
            if(s.captured && cudaEventSynchronize(s.captured)!=cudaSuccess) error=FORT_SCOPE_EXECUTION;
            if(s.device && cudaFree(s.device)!=cudaSuccess) error=FORT_SCOPE_EXECUTION;
            if(s.host && cudaFreeHost(s.host)!=cudaSuccess) error=FORT_SCOPE_EXECUTION;
            if(s.captured) cudaEventDestroy(s.captured);
            if(s.ready) cudaEventDestroy(s.ready);
        }
        if(q.dma) cudaStreamDestroy(q.dma);
    } catch(...) { error=FORT_SCOPE_EXECUTION; }
    if(q.lease) {
        const auto status=fort_scope_domain_workspace_release_v1(q.domain,q.lease);
        if(status) error=status;
    }
    return error;
}
struct Capture { Queue &queue; Slot &slot; const size_t *bytes; size_t count; };
int capture(void *user,const fort_scope_external_view_v1 *views,size_t count,void *stream,int *started) noexcept {
    *started=0;
    auto &work=*static_cast<Capture *>(user);
    if(count!=work.count) return FORT_SCOPE_ARGUMENT;
    for(size_t i=0;i<count;++i) {
        const auto &v=views[i];size_t bytes=v.element_bytes;
        if(!v.device || !bytes || v.bytes!=work.bytes[i]) return FORT_SCOPE_ARGUMENT;
        for(unsigned a=0;a<v.rank;++a) {
            if(v.strides[a]!=bytes || !v.extents[a] || bytes>SIZE_MAX/v.extents[a]) return FORT_SCOPE_ARGUMENT;
            bytes*=v.extents[a];
        }
        if(bytes!=v.bytes) return FORT_SCOPE_ARGUMENT;
    }
    *started=1;
    size_t offset=0;
    auto execution=static_cast<cudaStream_t>(stream);
    for(size_t i=0;i<count;++i) {
        if(cudaMemcpyAsync(static_cast<char *>(work.slot.device)+offset,views[i].device,work.bytes[i],
                           cudaMemcpyDeviceToDevice,execution)!=cudaSuccess) return FORT_SCOPE_EXECUTION;
        offset+=work.bytes[i];
    }
    return cudaEventRecord(work.slot.captured,execution)==cudaSuccess ? FORT_SCOPE_OK : FORT_SCOPE_EXECUTION;
}
}

extern "C" int fort_snapshot_create_v1(fort_scope_domain_t domain,size_t bytes,size_t pinned,void **out) {
    if(out) *out=nullptr;
    if(!out || !domain || !bytes || bytes>SIZE_MAX/2) return FORT_SCOPE_ARGUMENT;
    if(2*bytes>pinned) return FORT_SCOPE_RESOURCE;
    std::unique_ptr<Queue> q;
    try {
        q=std::make_unique<Queue>();q->domain=domain;q->bytes=bytes;
        int status=fort_scope_domain_device_get_v1(domain,&q->device);if(status) return status;
        status=fort_scope_domain_workspace_reserve_v1(domain,2*bytes,&q->lease);if(status) return status;
        Device device(q->device);
        checked(cudaStreamCreateWithFlags(&q->dma,cudaStreamNonBlocking));
        for(auto &s:q->slots) {
            checked(cudaMalloc(&s.device,bytes));
            checked(cudaMallocHost(&s.host,bytes));
            checked(cudaEventCreateWithFlags(&s.captured,cudaEventDisableTiming));
            checked(cudaEventCreateWithFlags(&s.ready,cudaEventDisableTiming));
        }
        q->worker=std::thread(consume,std::ref(*q));
        *out=q.release();return FORT_SCOPE_OK;
    } catch(...) {
        if(q) release(*q);
        return FORT_SCOPE_RESOURCE;
    }
}
extern "C" int fort_snapshot_acquire_v1(void *opaque,uint32_t *out) {
    if(out) *out=0;
    if(!opaque || !out) return FORT_SCOPE_ARGUMENT;
    auto &q=*static_cast<Queue *>(opaque);
    std::unique_lock<std::mutex> lock(q.mutex);
    if(q.error) return q.error;
    for(const auto &s:q.slots) if(s.state==acquired) return FORT_SCOPE_STATE;
    auto available=[&]{return q.error || q.slots[0].state==free_slot || q.slots[1].state==free_slot;};
    if(!available()) ++q.backpressure;
    q.changed.wait(lock,available);
    if(q.error) return q.error;
    for(unsigned i=0;i<2;++i) if(q.slots[i].state==free_slot) {
        q.slots[i].state=acquired;*out=i+1;return FORT_SCOPE_OK;
    }
    return FORT_SCOPE_STATE;
}
extern "C" int fort_snapshot_cancel_v1(void *opaque,uint32_t slot) {
    if(!opaque || !slot || slot>2) return FORT_SCOPE_ARGUMENT;
    auto &q=*static_cast<Queue *>(opaque);
    std::lock_guard<std::mutex> lock(q.mutex);
    if(q.slots[slot-1].state!=acquired) return FORT_SCOPE_STATE;
    q.slots[slot-1].state=free_slot;q.changed.notify_all();return FORT_SCOPE_OK;
}
extern "C" int fort_snapshot_submit_v1(void *opaque,uint32_t slot,fort_scope_t context,
    const fort_scope_external_binding_v1 *bindings,const size_t *bytes,size_t count,
    fort_snapshot_writer_v1 writer,void *user) {
    if(!opaque || !slot || slot>2 || !bindings || !bytes || !count || !writer) return FORT_SCOPE_ARGUMENT;
    auto &q=*static_cast<Queue *>(opaque);auto &s=q.slots[slot-1];
    {
        std::lock_guard<std::mutex> lock(q.mutex);
        if(q.error) return q.error;
        if(s.state!=acquired) return FORT_SCOPE_STATE;
    }
    fort_scope_domain_t domain=0;
    int status=fort_scope_domain_get_v1(context,&domain);if(status) return status;
    if(domain!=q.domain) return FORT_SCOPE_ARGUMENT;
    size_t total=0;
    for(size_t i=0;i<count;++i) {
        if(!bytes[i] || bytes[i]>q.bytes-total || bindings[i].access.flags!=FORT_SCOPE_READ_ALL ||
           bindings[i].access.read_count || bindings[i].access.write_count || bindings[i].access.overwrite_count)
            return FORT_SCOPE_ARGUMENT;
        total+=bytes[i];
    }
    if(total!=q.bytes) return FORT_SCOPE_ARGUMENT;
    fort_scope_external_call_v1 call{1,0,bindings,count,q.lease,2*q.bytes};
    fort_scope_external_report_v1 report{};Capture work{q,s,bytes,count};
    status=fort_scope_external_execute_v1(context,&call,capture,&work,&report);
    if(status) {
        if(report.started) {
            // A poisoned context cannot be waited through its normal API.
            // release() performs fatal device synchronization before freeing
            // either slot; no callback or reuse follows this partial capture.
            std::lock_guard<std::mutex> lock(q.mutex);q.error=FORT_SCOPE_EXECUTION;
        }
        return status;
    }
    try {
        Device device(q.device);
        checked(cudaStreamWaitEvent(q.dma,s.captured,0));
        checked(cudaMemcpyAsync(s.host,s.device,q.bytes,cudaMemcpyDeviceToHost,q.dma));
        checked(cudaEventRecord(s.ready,q.dma));
    } catch(...) {
        fort_scope_execution_error(context,"snapshot DMA submission failed; replay prohibited");
        std::lock_guard<std::mutex> lock(q.mutex);q.error=FORT_SCOPE_EXECUTION;
        return FORT_SCOPE_EXECUTION;
    }
    {
        std::lock_guard<std::mutex> lock(q.mutex);
        s.writer=writer;s.user=user;s.context=context;s.state=queued;
        q.order[(q.head+q.count)%2]=slot-1;++q.count;++q.submitted;
    }
    q.changed.notify_all();return FORT_SCOPE_OK;
}
extern "C" int fort_snapshot_wait_v1(void *opaque) {
    if(!opaque) return FORT_SCOPE_ARGUMENT;
    auto &q=*static_cast<Queue *>(opaque);
    std::unique_lock<std::mutex> lock(q.mutex);
    q.changed.wait(lock,[&]{return q.completed==q.submitted;});
    return q.error;
}
extern "C" int fort_snapshot_destroy_v1(void *opaque) {
    if(!opaque) return FORT_SCOPE_ARGUMENT;
    std::unique_ptr<Queue> q(static_cast<Queue *>(opaque));
    int status=fort_snapshot_wait_v1(opaque);
    {
        std::lock_guard<std::mutex> lock(q->mutex);q->stop=true;
    }
    q->changed.notify_all();q->worker.join();
    const char *trace=std::getenv("FORT_RUNTIME_TRACE");
    if(trace && trace[0] && trace[0]!='0')
        std::fprintf(stderr,"FORT_SNAPSHOTS submitted=%llu completed=%llu backpressure=%llu slots=2 device_bytes=%zu pinned_bytes=%zu capture_bytes=%llu download_bytes=%llu status=%d\n",
            (unsigned long long)q->submitted,(unsigned long long)q->completed,(unsigned long long)q->backpressure,
            2*q->bytes,2*q->bytes,(unsigned long long)(q->submitted*q->bytes),
            (unsigned long long)(q->submitted*q->bytes),status);
    const int cleanup=release(*q);return status?status:cleanup;
}
