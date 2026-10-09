// Offline-only observations of the exact emitted persistent-team protocol.
#ifndef FORT_SCOPED_TEAM_OBSERVER_HPP
#define FORT_SCOPED_TEAM_OBSERVER_HPP
#include <stdint.h>

enum fort_scope_team_observer_stage {
    FORT_SCOPE_TEAM_OBSERVE_API = 0,
    FORT_SCOPE_TEAM_OBSERVE_OWNER = 1,
    FORT_SCOPE_TEAM_OBSERVE_DESCRIPTOR = 2,
    FORT_SCOPE_TEAM_OBSERVE_ENTRY = 3,
    FORT_SCOPE_TEAM_OBSERVE_CPU_WORKER = 4,
    FORT_SCOPE_TEAM_OBSERVE_GPU_WORKER = 5,
    FORT_SCOPE_TEAM_OBSERVE_NATIVE_CALL = 6,
    FORT_SCOPE_TEAM_OBSERVE_COMPUTE = 7,
    FORT_SCOPE_TEAM_OBSERVE_NATIVE_WORKER = 8,
    FORT_SCOPE_TEAM_OBSERVE_STAGES = 9
};
#ifdef __cplusplus
extern "C" {
#endif
int fort_scope_team_observer_enabled_v1(void);
/* Pure compatibility query: no CUDA, synchronization or numerical work. */
int fort_scope_team_fortran_compatible_v1(const char *actual_version, const char *actual_options,
                                         const char *expected_version, const char *expected_semantic_options);
void fort_scope_team_observe_begin_v1(uint32_t stage);
void fort_scope_team_observe_end_v1(uint32_t stage);
void fort_scope_team_observe_end_as_v1(uint32_t stage, uint32_t recorded_stage);
/* Call only after all participants have completed every observed region.
 * Protocol durations are coordinator-exclusive totals. COMPUTE returns the
 * maximum complete numerical span over participants. Peer barrier waits must
 * not reintroduce a coordinator API/compute interval into protocol costs.
 * A malformed/unbalanced observation fails the whole offline measurement. */
int fort_scope_team_observer_reset_v1(uint32_t threads);
int fort_scope_team_observer_read_v1(uint32_t stage, uint32_t threads,
                                    double *seconds, uint64_t *calls);
#ifdef __cplusplus
}
namespace fort_scoped {
class TeamObservation {
#ifdef FORT_SCOPE_CALIBRATION
    uint32_t stage_, recorded_;
public:
    explicit TeamObservation(uint32_t stage) : stage_(stage), recorded_(stage) {
        fort_scope_team_observe_begin_v1(stage_);
    }
    void record_as(uint32_t stage) { recorded_ = stage; }
    ~TeamObservation() { fort_scope_team_observe_end_as_v1(stage_, recorded_); }
#else
public:
    explicit TeamObservation(uint32_t) {}
    void record_as(uint32_t) {}
#endif
    TeamObservation(const TeamObservation &) = delete;
    TeamObservation &operator=(const TeamObservation &) = delete;
};
}
#endif

#ifdef FORT_SCOPE_TEAM_OBSERVER_IMPLEMENTATION
#include <string>
#include <vector>
#include <cctype>
extern "C" int fort_scope_team_fortran_compatible_v1(const char *version, const char *options,
                                                     const char *expected, const char *semantic) {
    try {
    if (!version || !options || !expected || !semantic || std::string(version) != expected) return 0;
    // Match Python shlex for shell-quoted compiler_options strings. Preserve
    // every semantic token and its order; remove only -I/-J/-o plus operands.
    std::vector<std::string> tokens;
    std::string token;
    char quote = 0; bool escape = false, active = false;
    for (const char *p=options; ; ++p) {
        const char c=*p;
        if (!c) { if (quote || escape) return 0; if (active) tokens.push_back(token); break; }
        if (escape) { token += c; escape=false; active=true; continue; }
        if (c=='\\' && quote!='\'') {
            if (quote=='"' && p[1]!='"' && p[1]!='\\') { token+=c; active=true; continue; }
            escape=true; active=true; continue;
        }
        if (quote) { if (c==quote) quote=0; else token+=c; active=true; continue; }
        if (c=='\'' || c=='"') { quote=c; active=true; continue; }
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (active) { tokens.push_back(token); token.clear(); active=false; }
        } else { token+=c; active=true; }
    }
    std::string normalized;
    bool skip=false;
    for (const auto &t: tokens) {
        if (skip) { skip=false; continue; }
        if (t=="-I" || t=="-J" || t=="-o") { skip=true; continue; }
        const bool semantic_o=t.compare(0,7,"-openmp")==0 || t.compare(0,8,"-offload")==0 || t.compare(0,4,"-opt")==0;
        if (t.size()>2 && (t.compare(0,2,"-I")==0 || t.compare(0,2,"-J")==0 ||
                          (t.compare(0,2,"-o")==0 && !semantic_o))) continue;
        if (!normalized.empty()) normalized+='\x1f';
        normalized+=t;
    }
    return !skip && normalized==semantic;
    } catch (...) { return 0; } // A pure compatibility failure selects native.
}
#ifdef FORT_SCOPE_CALIBRATION
#include <array>
#include <chrono>
#include <atomic>
#include <omp.h>
namespace fort_scope_team_observer_detail {
using Clock = std::chrono::steady_clock;
constexpr uint32_t capacity = 256, depth_limit = 32;
struct Frame { uint32_t stage; Clock::time_point start; double children = 0; };
struct Slot {
    std::array<Frame, depth_limit> stack{};
    std::array<double, FORT_SCOPE_TEAM_OBSERVE_STAGES> seconds{};
    std::array<uint64_t, FORT_SCOPE_TEAM_OBSERVE_STAGES> calls{};
    uint32_t depth = 0;
    bool failed = false;
};
static std::array<Slot, capacity> slots;
static std::atomic<bool> invalid_thread{false};
static Slot &slot() {
    const auto tid = uint32_t(omp_get_thread_num());
    if (tid < capacity) return slots[tid];
    invalid_thread.store(true, std::memory_order_relaxed);
    static thread_local Slot overflow;
    return overflow;
}
}
extern "C" int fort_scope_team_observer_enabled_v1(void) { return 1; }
extern "C" void fort_scope_team_observe_begin_v1(uint32_t stage) {
    using namespace fort_scope_team_observer_detail;
    auto &s = slot();
    if (stage >= FORT_SCOPE_TEAM_OBSERVE_STAGES || s.depth == depth_limit) { s.failed = true; return; }
    s.stack[s.depth++] = Frame{stage, Clock::now(), 0};
}
extern "C" void fort_scope_team_observe_end_as_v1(uint32_t stage, uint32_t recorded) {
    using namespace fort_scope_team_observer_detail;
    const auto end = Clock::now();
    auto &s = slot();
    if (!s.depth || s.stack[s.depth-1].stage != stage || recorded >= FORT_SCOPE_TEAM_OBSERVE_STAGES) {
        s.failed = true; return;
    }
    const auto frame = s.stack[--s.depth];
    const double elapsed = std::chrono::duration<double>(end-frame.start).count();
    const double exclusive = elapsed-frame.children;
    if (!(exclusive >= 0)) { s.failed = true; return; }
    s.seconds[recorded] += exclusive;
    ++s.calls[recorded];
    if (s.depth) s.stack[s.depth-1].children += elapsed;
}
extern "C" void fort_scope_team_observe_end_v1(uint32_t stage) {
    fort_scope_team_observe_end_as_v1(stage, stage);
}
extern "C" int fort_scope_team_observer_reset_v1(uint32_t threads) {
    using namespace fort_scope_team_observer_detail;
    if (!threads || threads > capacity) return 1;
    for (uint32_t i=0; i<threads; ++i) if (slots[i].depth) return 1;
    for (uint32_t i=0; i<threads; ++i) slots[i] = Slot{};
    invalid_thread.store(false, std::memory_order_relaxed);
    return 0;
}
extern "C" int fort_scope_team_observer_read_v1(uint32_t stage, uint32_t threads,
                                              double *seconds, uint64_t *calls) {
    using namespace fort_scope_team_observer_detail;
    if (!seconds || !calls || !threads || threads > capacity || stage >= FORT_SCOPE_TEAM_OBSERVE_STAGES ||
        invalid_thread.load(std::memory_order_relaxed)) return 1;
    *seconds = 0; *calls = 0;
    for (uint32_t i=0; i<threads; ++i) {
        if (slots[i].depth || slots[i].failed) return 1;
        if (stage != FORT_SCOPE_TEAM_OBSERVE_COMPUTE && i != 0) continue;
        if (slots[i].seconds[stage] > *seconds) *seconds = slots[i].seconds[stage];
        if (slots[i].calls[stage] > *calls) *calls = slots[i].calls[stage];
    }
    return 0;
}
#else
extern "C" int fort_scope_team_observer_enabled_v1(void) { return 0; }
extern "C" void fort_scope_team_observe_begin_v1(uint32_t) {}
extern "C" void fort_scope_team_observe_end_v1(uint32_t) {}
extern "C" void fort_scope_team_observe_end_as_v1(uint32_t, uint32_t) {}
extern "C" int fort_scope_team_observer_reset_v1(uint32_t) { return 1; }
extern "C" int fort_scope_team_observer_read_v1(uint32_t, uint32_t, double *, uint64_t *) { return 1; }
#endif
#endif
#endif
