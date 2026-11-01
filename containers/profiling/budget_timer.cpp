// Diagnostic only. Compile without -finstrument-functions; preserve errno and FP state.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <mutex>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {
struct Event {
    uintptr_t function, caller;
    uint64_t parent, start, stop, cpu_start, cpu_stop;
};
struct Thread {
    long tid;
    std::vector<Event> events;
    std::vector<size_t> stack;
    uint64_t errors = 0;
};
struct Registry { std::mutex mutex; std::vector<Thread *> threads; };
Registry &registry() { static auto *value = new Registry; return *value; }
thread_local Thread *local = nullptr;
long main_tid = 0;
uint64_t clock_errors = 0;
uint64_t clock_ns(clockid_t clock) noexcept {
    timespec t{};
    if (clock_gettime(clock, &t)) {
        __atomic_fetch_add(&clock_errors, 1, __ATOMIC_RELAXED);
        return 0;
    }
    return uint64_t(t.tv_sec) * 1000000000ULL + uint64_t(t.tv_nsec);
}
Thread *thread() {
    if (!local) {
        local = new Thread{syscall(SYS_gettid), {}, {}, 0};
        local->events.reserve(65536);
        local->stack.reserve(128);
        std::lock_guard<std::mutex> lock(registry().mutex);
        registry().threads.push_back(local);
    }
    return local;
}
__attribute__((constructor)) void initialize() { main_tid = syscall(SYS_gettid); }
}

extern "C" void __cyg_profile_func_enter(void *function, void *caller) noexcept {
    const int saved = errno;
    Thread *t = thread();
    const uint64_t parent = t->stack.empty() ? UINT64_MAX : t->stack.back();
    t->events.push_back({uintptr_t(function), uintptr_t(caller), parent,
                        clock_ns(CLOCK_MONOTONIC), 0, clock_ns(CLOCK_THREAD_CPUTIME_ID), 0});
    t->stack.push_back(t->events.size() - 1);
    errno = saved;
}

extern "C" void __cyg_profile_func_exit(void *function, void *) noexcept {
    const int saved = errno;
    Thread *t = thread();
    if (t->stack.empty() || t->events[t->stack.back()].function != uintptr_t(function)) {
        ++t->errors;
    } else {
        auto &e = t->events[t->stack.back()];
        e.stop = clock_ns(CLOCK_MONOTONIC);
        e.cpu_stop = clock_ns(CLOCK_THREAD_CPUTIME_ID);
        t->stack.pop_back();
    }
    errno = saved;
}

extern "C" void fort_budget_anchor() noexcept {}

__attribute__((destructor)) static void dump() {
    const char *path = std::getenv("FORT_BUDGET_OUTPUT");
    if (!path || !*path) return;
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        std::perror("fresh diagnostic budget output");
        return;
    }
    FILE *out = fdopen(fd, "w");
    if (!out) { close(fd); return; }
    std::fprintf(out, "# schema=1 main_tid=%ld anchor=%llx clock_errors=%llu\n",
                 main_tid, (unsigned long long)uintptr_t(&fort_budget_anchor),
                 (unsigned long long)clock_errors);
    std::fprintf(out, "tid,event,parent,function,caller,start_ns,stop_ns,cpu_start_ns,cpu_stop_ns\n");
    for (auto *t : registry().threads) {
        std::fprintf(out, "# thread=%ld open_frames=%zu nesting_errors=%llu\n", t->tid,
                     t->stack.size(), (unsigned long long)t->errors);
        for (size_t i = 0; i < t->events.size(); ++i) {
            const auto &e = t->events[i];
            std::fprintf(out, "%ld,%zu,%llu,%llx,%llx,%llu,%llu,%llu,%llu\n", t->tid, i,
                (unsigned long long)e.parent, (unsigned long long)e.function,
                (unsigned long long)e.caller, (unsigned long long)e.start,
                (unsigned long long)e.stop, (unsigned long long)e.cpu_start,
                (unsigned long long)e.cpu_stop);
        }
    }
    if (std::fclose(out)) std::perror("diagnostic budget write");
}
