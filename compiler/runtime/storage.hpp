// Owned fixed-shape storage shared by ordinary wrappers and resident sessions.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>

#include "allocation.hpp"

namespace generated_kernels::storage {

inline void trace(const char *operation, std::size_t bytes = 0) {
    static const bool enabled = []() {
        const char *value = std::getenv("FORT_RUNTIME_TRACE");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (!enabled)
        return;
    // Use one stdio operation per record, sharing stderr's stream lock with
    // placement diagnostics emitted concurrently by the coordinating thread.
    if (std::strcmp(operation, "alloc") == 0 || std::strcmp(operation, "upload") == 0 ||
        std::strcmp(operation, "download") == 0 || std::strcmp(operation, "pool_create") == 0 ||
        std::strcmp(operation, "pool_alloc") == 0 || std::strcmp(operation, "scratch_reuse") == 0)
        std::fprintf(stderr, "FORT_RUNTIME %s bytes=%zu\n", operation, bytes);
    else
        std::fprintf(stderr, "FORT_RUNTIME %s\n", operation);
}

[[noreturn]] inline void fail(const char *message) {
    std::cerr << "Compiler workspace error: " << message << std::endl;
    std::abort();
}

inline std::size_t product(std::initializer_list<std::size_t> dimensions) {
    // A zero extent makes an empty array even if preceding extents are large.
    if (std::find(dimensions.begin(), dimensions.end(), 0) != dimensions.end())
        return 0;
    std::size_t result = 1;
    for (std::size_t dimension : dimensions) {
        if (dimension > std::numeric_limits<std::size_t>::max() / result)
            fail("array extent product overflow");
        result *= dimension;
    }
    return result;
}

inline void synchronize() {
#ifdef __CUDACC__
    CUCH(cudaDeviceSynchronize());
#endif
}

#if defined(__CUDACC__) && defined(USE_PINNED_MEMORY)
class HostRegistration {
    void *pointer_;

  public:
    HostRegistration(const void *pointer, std::size_t bytes) : pointer_(const_cast<void *>(pointer)) {
        CUCH(cudaHostRegister(pointer_, bytes, cudaHostRegisterPortable));
    }
    ~HostRegistration() { CUCH(cudaHostUnregister(pointer_)); }
};
#else
class HostRegistration {
  public:
    HostRegistration(const void *, std::size_t) {}
};
#endif

template <typename T> class Buffer {
    std::vector<std::size_t> dimensions_;
    std::size_t count_;
    std::size_t bytes_;
    std::unique_ptr<T[]> host_;
    bool host_current_ = false;
#ifdef __CUDACC__
    std::optional<allocation_detail::DeviceAllocation> allocation_;
    T *device_ = nullptr;
    bool device_current_ = false;
#endif
  public:
    explicit Buffer(std::initializer_list<std::size_t> dimensions,
                    AllocationPolicy policy = AllocationPolicy::dedicated)
        : dimensions_(dimensions), count_(product(dimensions)), bytes_(0) {
        if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T))
            fail("array byte size overflow");
        bytes_ = count_ * sizeof(T);
#ifdef __CUDACC__
        allocation_.emplace(bytes_, policy);
        device_ = static_cast<T *>(allocation_->get());
#else
        host_.reset(new T[count_]);
#endif
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void release_completed() {
#ifdef __CUDACC__
        allocation_->release_completed();
        device_ = nullptr;
#endif
    }
    std::size_t extent(std::size_t axis) const { return dimensions_.at(axis); }
    void validate(std::initializer_list<std::size_t> dimensions) const {
        if (dimensions.size() != dimensions_.size() ||
            !std::equal(dimensions.begin(), dimensions.end(), dimensions_.begin()))
            fail("array shape does not match workspace");
    }
    T *host_data() {
        if (!host_)
            host_.reset(new T[count_]);
#ifdef __CUDACC__
        if (!host_current_ && device_current_) {
            timing::measure_d2h(bytes_, [&]() {
                if (bytes_) {
                    HostRegistration registration(host_.get(), bytes_);
                    CUCH(cudaMemcpy(host_.get(), device_, bytes_, cudaMemcpyDeviceToHost));
                    trace("download", bytes_);
                }
            });
            host_current_ = true;
        }
#endif
        return host_.get();
    }
    T *device_data() {
#ifdef __CUDACC__
        if (!device_current_ && host_current_) {
            timing::measure_h2d(bytes_, [&]() {
                if (bytes_) {
                    HostRegistration registration(host_.get(), bytes_);
                    CUCH(cudaMemcpy(device_, host_.get(), bytes_, cudaMemcpyHostToDevice));
                    trace("upload", bytes_);
                }
            });
            device_current_ = true;
        }
        return device_;
#else
        return host_data();
#endif
    }
    void host_written() {
        host_current_ = true;
#ifdef __CUDACC__
        device_current_ = false;
#endif
    }
    void device_written() {
#ifdef __CUDACC__
        device_current_ = true;
        host_current_ = false;
#else
        host_current_ = true;
#endif
    }
    void update_device(const T *source, std::initializer_list<std::size_t> dimensions) {
        validate(dimensions);
#ifdef __CUDACC__
        timing::measure_h2d(bytes_, [&]() {
            if (bytes_) {
                HostRegistration registration(source, bytes_);
                CUCH(cudaMemcpy(device_, source, bytes_, cudaMemcpyHostToDevice));
                trace("upload", bytes_);
            }
        });
        device_current_ = true;
        host_current_ = false;
#else
        if (bytes_)
            std::memcpy(host_data(), source, bytes_);
        host_current_ = true;
#endif
    }
    void update_host(T *destination, std::initializer_list<std::size_t> dimensions) {
        validate(dimensions);
#ifdef __CUDACC__
        if (device_current_) {
            timing::measure_d2h(bytes_, [&]() {
                if (bytes_) {
                    HostRegistration registration(destination, bytes_);
                    CUCH(cudaMemcpy(destination, device_, bytes_, cudaMemcpyDeviceToHost));
                    trace("download", bytes_);
                }
            });
        } else
#endif
        {
            if (bytes_)
                std::memcpy(destination, host_data(), bytes_);
        }
    }
};

template <typename T> using BufferSlot = std::optional<Buffer<T>>;

template <typename State> class Registry {
    std::unordered_map<std::int64_t, std::unique_ptr<State>> entries_;
    std::int64_t next_ = 1;

  public:
    template <typename... Args> std::int64_t create(Args &&...arguments) {
        if (next_ == std::numeric_limits<std::int64_t>::max())
            fail("workspace token limit reached");
        const std::int64_t token = next_++;
        entries_.emplace(token, std::make_unique<State>(std::forward<Args>(arguments)...));
        return token;
    }
    State &get(std::int64_t token) {
        auto found = entries_.find(token);
        if (found == entries_.end())
            fail("invalid or stale workspace token");
        return *found->second;
    }
    void destroy(std::int64_t token) {
        if (!token)
            return;
        get(token);
        entries_.erase(token);
    }
};
} // namespace generated_kernels::storage
