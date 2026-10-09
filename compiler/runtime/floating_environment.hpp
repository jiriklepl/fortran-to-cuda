// Runtime setup and cost arithmetic must not change the caller's IEEE state.
#ifndef FORT_RUNTIME_FLOATING_ENVIRONMENT_HPP
#define FORT_RUNTIME_FLOATING_ENVIRONMENT_HPP
#include <cfenv>

namespace fort_runtime {
inline bool numerical_environment_supported() noexcept {
#if defined(__GLIBC__)
    return std::fegetround() == FE_TONEAREST && fegetexcept() == 0;
#else
    // A platform without a trap-mask inquiry cannot establish this contract.
    return false;
#endif
}
class HostFloatingEnvironment {
    std::fenv_t saved_{};
    bool valid_ = false;
public:
    HostFloatingEnvironment() noexcept : valid_(std::feholdexcept(&saved_) == 0) {}
    HostFloatingEnvironment(const HostFloatingEnvironment &) = delete;
    HostFloatingEnvironment &operator=(const HostFloatingEnvironment &) = delete;
    ~HostFloatingEnvironment() { if (valid_) std::fesetenv(&saved_); }
    bool valid() const noexcept { return valid_; }
};
}
#endif
