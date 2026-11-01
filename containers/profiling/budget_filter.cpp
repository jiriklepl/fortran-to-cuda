// Coarse source/function address selection only; no application decision changes.
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void fort_budget_anchor() noexcept;
extern "C" void __real___cyg_profile_func_enter(void *, void *) noexcept;
extern "C" void __real___cyg_profile_func_exit(void *, void *) noexcept;

namespace {
struct Selection {
    std::vector<uintptr_t> addresses;
    Selection() {
        const char *name = std::getenv("FORT_BUDGET_FILTER");
        FILE *input = name ? std::fopen(name, "r") : nullptr;
        unsigned long long base, address;
        if (!input || std::fscanf(input, "%llx", &base) != 1) {
            std::fputs("missing authenticated budget function selection\n", stderr);
            std::abort();
        }
        const uintptr_t bias = uintptr_t(&fort_budget_anchor) - uintptr_t(base);
        while (std::fscanf(input, "%llx", &address) == 1)
            addresses.push_back(uintptr_t(address) + bias);
        const bool valid = std::feof(input) && !addresses.empty();
        std::fclose(input);
        if (!valid) std::abort();
        std::sort(addresses.begin(), addresses.end());
    }
};
bool selected(void *function) {
    static const auto *selection = new Selection;
    return std::binary_search(selection->addresses.begin(), selection->addresses.end(), uintptr_t(function));
}
}

extern "C" void __wrap___cyg_profile_func_enter(void *function, void *caller) noexcept {
    const int saved = errno;
    if (selected(function)) __real___cyg_profile_func_enter(function, caller);
    errno = saved;
}
extern "C" void __wrap___cyg_profile_func_exit(void *function, void *caller) noexcept {
    const int saved = errno;
    if (selected(function)) __real___cyg_profile_func_exit(function, caller);
    errno = saved;
}
