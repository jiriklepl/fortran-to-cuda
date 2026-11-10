/* Context-owned CUDA graph executables. No registry, implicit stream or
 * process-lifetime CUDA destructor: close/abandon explicitly drain this cache.
 * CUDA kernel-node APIs copy values from kernelParams during add/set calls:
 * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html */
#ifndef FORT_SCOPED_GRAPHS_HPP
#define FORT_SCOPED_GRAPHS_HPP
#include "scoped_runtime.h"
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>
namespace fort_scoped::graphs {
class ArgumentValues {
    struct alignas(std::max_align_t) Word { unsigned char bytes[alignof(std::max_align_t)]; };
    std::vector<size_t> sizes_,offsets_;
    std::vector<Word> storage_;
public:
    // The previous per-argument allocations had malloc's alignment. Keep that
    // guarantee while owning one payload allocation, including stable padding.
    bool capture(const fort_scope_kernel_v1 &kernel,const size_t *sizes) {
        constexpr size_t alignment=alignof(Word),word=sizeof(Word);
        constexpr size_t maximum=std::numeric_limits<size_t>::max();
        sizes_.clear(); offsets_.clear(); storage_.clear();
        sizes_.reserve(kernel.argument_count); offsets_.reserve(kernel.argument_count);
        size_t bytes=0;
        for (size_t i=0;i<kernel.argument_count;++i) {
            if (!sizes[i] || sizes[i]>maximum-(alignment-1)) return false;
            const size_t padded=((sizes[i]+alignment-1)/alignment)*alignment;
            if (bytes>maximum-padded) return false;
            sizes_.push_back(sizes[i]); offsets_.push_back(bytes); bytes+=padded;
        }
        if (bytes>maximum-(word-1)) return false;
        storage_.resize((bytes+word-1)/word);
        if (storage_.empty()) return true;
        std::memset(storage_.data(),0,storage_.size()*word);
        for (size_t i=0;i<kernel.argument_count;++i)
            std::memcpy(argument(i),kernel.arguments[i],sizes[i]);
        return true;
    }
    void *argument(size_t index) {
        return reinterpret_cast<unsigned char *>(storage_.data())+offsets_[index];
    }
    bool operator==(const ArgumentValues &other) const {
        return sizes_==other.sizes_ && storage_.size()==other.storage_.size() &&
            (storage_.empty() || !std::memcmp(storage_.data(),other.storage_.data(),storage_.size()*sizeof(Word)));
    }
};
inline const char *invalid(const fort_scope_kernel_v1 *kernels, size_t count) {
    if (count && !kernels) return "missing kernel group descriptors";
    for (size_t i=0; i<count; ++i) {
        const auto &k = kernels[i];
        if (k.version != FORT_SCOPE_GRAPH_ABI_VERSION || k.reserved || !k.function)
            return "invalid kernel descriptor version or function";
        for (size_t axis=0; axis<3; ++axis)
            if (!k.grid[axis] || !k.block[axis]) return "kernel dimensions must be positive";
        if (k.shared_bytes > std::numeric_limits<unsigned int>::max()) return "kernel shared-memory size overflow";
        if (k.argument_count && !k.arguments) return "missing kernel argument array";
        for (size_t argument=0; argument<k.argument_count; ++argument)
            if (!k.arguments[argument]) return "missing kernel argument value";
    }
    return nullptr;
}
} // namespace fort_scoped::graphs
#ifndef FORT_SCOPE_CPU_TEST
#include <cuda_runtime.h>
#include <cstdlib>
#include <unordered_map>
#include <utility>
#include <vector>
namespace fort_scoped::graphs {
struct Topology {
    const void *function;
    size_t arguments;
};
// Only queued calls provide complete owned argument bytes. Raw graph calls
// must still update every node: pointer equality cannot establish value parity.
struct Parameters {
    bool known = false;
    unsigned int grid[3]{},block[3]{};
    size_t shared = 0;
    ArgumentValues values;
    bool matches(const fort_scope_kernel_v1 &kernel,const ArgumentValues *arguments) const {
        if (!known || !arguments || shared!=kernel.shared_bytes || !(values==*arguments)) return false;
        for (size_t i=0;i<3;++i)
            if (grid[i]!=kernel.grid[i] || block[i]!=kernel.block[i]) return false;
        return true;
    }
    Parameters() = default;
    Parameters(const fort_scope_kernel_v1 &kernel,const ArgumentValues *arguments) {
        if (!arguments) return;
        values=*arguments;
        for (size_t i=0;i<3;++i) { grid[i]=kernel.grid[i]; block[i]=kernel.block[i]; }
        shared=kernel.shared_bytes; known=true;
    }
};
struct Entry {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    std::vector<cudaGraphNode_t> nodes;
    std::vector<Topology> topology;
    std::vector<Parameters> parameters;
    bool replaces_graph = false;
};
inline cudaKernelNodeParams parameters(const fort_scope_kernel_v1 &kernel) {
    cudaKernelNodeParams p{};
    p.func = const_cast<void *>(kernel.function);
    p.gridDim = dim3(kernel.grid[0],kernel.grid[1],kernel.grid[2]);
    p.blockDim = dim3(kernel.block[0],kernel.block[1],kernel.block[2]);
    p.sharedMemBytes = static_cast<unsigned int>(kernel.shared_bytes);
    p.kernelParams = const_cast<void **>(kernel.arguments);
    return p;
}
inline bool matches(const Entry &entry, const fort_scope_kernel_v1 *kernels, size_t count) {
    if (entry.topology.size() != count) return false;
    for (size_t i=0; i<count; ++i)
        if (entry.topology[i].function != kernels[i].function ||
            entry.topology[i].arguments != kernels[i].argument_count) return false;
    return true;
}
inline cudaError_t destroy(Entry &entry, fort_scope_graph_stats_v1 &stats) noexcept {
    cudaError_t status = cudaSuccess;
    if (entry.executable) {
        status = cudaGraphExecDestroy(entry.executable);
        entry.executable = nullptr;
    }
    if (entry.graph) {
        const auto next = cudaGraphDestroy(entry.graph);
        if (next == cudaSuccess) ++stats.destructions;
        if (status == cudaSuccess) status = next;
        entry.graph = nullptr;
    }
    return status;
}
class Cache {
    std::unordered_map<uint64_t, Entry> entries_;
public:
    // One-shot groups execute directly. Retain only ordered function/argument
    // count metadata on first reach; never retain caller argument pointers.
    // A second matching reach can amortize construction through later replay.
    cudaError_t observe(uint64_t key, const fort_scope_kernel_v1 *kernels, size_t count,
                        fort_scope_graph_stats_v1 &stats, bool &warmup) {
        warmup = false;
        auto found = entries_.find(key);
        if (found != entries_.end() && matches(found->second,kernels,count)) return cudaSuccess;
        Entry replacement;
        replacement.topology.reserve(count);
        for (size_t i=0; i<count; ++i)
            replacement.topology.push_back({kernels[i].function,kernels[i].argument_count});
        if (found == entries_.end()) entries_.emplace(key,std::move(replacement));
        else {
            const bool had_graph = found->second.graph != nullptr;
            replacement.replaces_graph = had_graph || found->second.replaces_graph;
            const auto status = destroy(found->second,stats);
            if (status != cudaSuccess) return status;
            if (had_graph) --stats.live_graphs;
            found->second = std::move(replacement);
        }
        warmup = true;
        return cudaSuccess;
    }
    // A failure before launch_attempted can permit native continuation if its
    // CUDA status is explicitly recoverable. Once launch is attempted there
    // is no retry path, including failures reported by later APIs.
    cudaError_t submit(uint64_t key, const fort_scope_kernel_v1 *kernels, size_t count,
                       cudaStream_t stream, fort_scope_graph_stats_v1 &stats, bool &launch_attempted,
                       const ArgumentValues *const *values=nullptr) {
        launch_attempted = false;
        auto found = entries_.find(key);
        const bool replay = found != entries_.end() && found->second.executable &&
                            matches(found->second,kernels,count);
        if (!replay) {
#ifdef FORT_SCOPE_TEST_FAULTS
            if (std::getenv("FORT_SCOPE_TEST_FAIL_GRAPH_CREATE")) return cudaErrorMemoryAllocation;
#endif
            Entry replacement;
            replacement.nodes.resize(count);
            replacement.topology.reserve(count);
            replacement.parameters.reserve(count);
            // Finish all copying before acquiring CUDA graph resources.
            for (size_t i=0; i<count; ++i) {
                replacement.topology.push_back({kernels[i].function,kernels[i].argument_count});
                replacement.parameters.emplace_back(kernels[i],values?values[i]:nullptr);
            }
            auto status = cudaGraphCreate(&replacement.graph,0);
            if (status != cudaSuccess) return status;
            ++stats.creations;
            for (size_t i=0; i<count; ++i) {
                auto p = parameters(kernels[i]);
                status = cudaGraphAddKernelNode(&replacement.nodes[i],replacement.graph,
                    i ? &replacement.nodes[i-1] : nullptr,i ? 1 : 0,&p);
                if (status != cudaSuccess) { (void)destroy(replacement,stats); return status; }
            }
            status = cudaGraphInstantiate(&replacement.executable,replacement.graph,nullptr,nullptr,0);
            if (status != cudaSuccess) { (void)destroy(replacement,stats); return status; }
            ++stats.instantiations;
            const bool had_graph = found != entries_.end() && found->second.graph;
            const bool rebuild = had_graph || (found != entries_.end() && found->second.replaces_graph);
            if (found == entries_.end()) {
                try { found = entries_.emplace(key,std::move(replacement)).first; }
                catch (...) { (void)destroy(replacement,stats); throw; }
            } else {
                status = destroy(found->second,stats);
                if (status != cudaSuccess) { (void)destroy(replacement,stats); return status; }
                found->second = std::move(replacement);
                if (rebuild) ++stats.rebuilds;
            }
            if (!had_graph) ++stats.live_graphs;
        } else {
#ifdef FORT_SCOPE_TEST_FAULTS
            if (std::getenv("FORT_SCOPE_TEST_FAIL_GRAPH_UPDATE")) return cudaErrorNotSupported;
#endif
            for (size_t i=0; i<count; ++i) {
                const auto *arguments=values?values[i]:nullptr;
                auto &previous=found->second.parameters[i];
                if (previous.matches(kernels[i],arguments)) continue;
                Parameters next(kernels[i],arguments);
                auto p = parameters(kernels[i]);
                const auto status = cudaGraphExecKernelNodeSetParams(found->second.executable,found->second.nodes[i],&p);
                if (status != cudaSuccess) return status;
                // A failed update must never record uninstalled arguments.
                previous=std::move(next);
                ++stats.parameter_updates;
            }
        }
        launch_attempted = true;
#ifdef FORT_SCOPE_TEST_FAULTS
        if (std::getenv("FORT_SCOPE_TEST_FAIL_GRAPH_LAUNCH")) return cudaErrorLaunchFailure;
#endif
        const auto status = cudaGraphLaunch(found->second.executable,stream);
        if (status != cudaSuccess) return status;
        ++stats.submissions;
        if (replay) ++stats.replays;
        stats.logical_kernels += count;
        return cudaSuccess;
    }
    cudaError_t clear(fort_scope_graph_stats_v1 &stats) noexcept {
        cudaError_t status = cudaSuccess;
        for (auto &entry : entries_) {
            const auto next = destroy(entry.second,stats);
            if (status == cudaSuccess) status = next;
        }
        entries_.clear();
        stats.live_graphs = 0;
        return status;
    }
};
} // namespace fort_scoped::graphs
#endif
#endif
