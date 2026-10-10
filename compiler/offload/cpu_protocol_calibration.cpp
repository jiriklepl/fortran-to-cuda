// Empty fork/join startup. The timed executable has no GOMP instrumentation.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <omp.h>
#include <sched.h>
#include <string>
#include <type_traits>
#include <vector>

#ifndef CALIBRATION_PRECISION
#define CALIBRATION_PRECISION 64
#endif
using real = std::conditional_t<CALIBRATION_PRECISION == 64, double, float>;
using clock_type = std::chrono::steady_clock;
#include "cpu_protocol_generated.hpp"
extern "C" void fort_numerical_native_v2(int, std::size_t, int, int,
                                         const real*, const real*, real*);
extern "C" void fort_numerical_fortran_identity_v2(char*, char*, int);
extern "C" void fort_cpu_protocol_proof_begin(int) __attribute__((weak));
extern "C" void fort_cpu_protocol_proof_end() __attribute__((weak));

static std::string quoted(const char* text) {
    std::string result = "\"";
    for (const unsigned char c : std::string(text)) {
        if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
        else if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else if (c == '\t') result += "\\t";
        else if (c < 32) std::exit(2);
        else result += char(c);
    }
    return result + '"';
}

// The generated control uses the production renderer and cyclic tid/team
// partition. Original native Fortran serial/fork-join controls are unchanged.
static void run(int backend, std::size_t n, int threads,
                const real* a, const real* b, real* out) {
    if (backend == 0) fort_numerical_native_v2(1, n, threads, 1, a, b, out);
    else generated_cpu(a, b, out, n, threads);
}

struct buffers {
    static constexpr real sentinel = real(-131.25);
    std::array<real, 10> a{}, b{}, output{};
    buffers() {
        for (std::size_t i = 0; i != a.size(); ++i) {
            a[i] = real(.25) + real(i) / real(16);
            b[i] = real(.125) + real(i) / real(32);
        }
        output.fill(sentinel);
    }
    bool agrees(std::size_t n) const {
        for (std::size_t i = 0; i != a.size(); ++i) {
            if (a[i] != real(.25) + real(i) / real(16) ||
                b[i] != real(.125) + real(i) / real(32)) return false;
            const real expected = i > 0 && i <= n ? a[i] + real(.25) * b[i] : sentinel;
            if (output[i] != expected) return false;
        }
        return true;
    }
    void execute(int backend, std::size_t n, int threads) {
        run(backend, n, threads, a.data() + 1, b.data() + 1, output.data() + 1);
    }
};
struct sample { unsigned long long repetitions{}; double elapsed{}; };
struct close_file { void operator()(std::FILE* file) const noexcept { std::fclose(file); } };
static sample measure(int backend, std::size_t n, int threads, buffers& data) {
    const auto start = clock_type::now();
    sample result;
    do {
        data.execute(backend, n, threads);
        ++result.repetitions;
        result.elapsed = std::chrono::duration<double>(clock_type::now() - start).count();
    } while (result.elapsed < .2);
    return result;
}
static constexpr const char* backends[] = {"native_fork_join", "generated_cpu"};
static constexpr const char* memory_backends[] = {"native_serial", "native_fork_join", "generated_cpu"};
static constexpr std::array<std::size_t, 8> memory_fit{65536,98304,147456,221184,331776,497664,746496,1119744};
static constexpr std::array<std::size_t, 17> memory_sizes{65536,80265,98304,120397,131072,147456,
    180596,221184,270894,331776,406341,497664,524288,609511,746496,914267,1119744};
static bool is_memory_fit(std::size_t n) {
    return std::find(memory_fit.begin(), memory_fit.end(), n) != memory_fit.end();
}
struct memory_buffers {
    std::size_t n;
    std::vector<real> a, b, output;
    explicit memory_buffers(std::size_t items): n(items), a(n + 2), b(n + 2), output(n + 2, buffers::sentinel) {
        a.front() = a.back() = b.front() = b.back() = buffers::sentinel;
        for (std::size_t i = 0; i != n; ++i) {
            a[i + 1] = real(.25) + real(i % 31) / real(32);
            b[i + 1] = real(.125) + real(i % 19) / real(64);
        }
    }
    void execute(int backend, int threads) {
        if (backend < 2) fort_numerical_native_v2(1, n, threads, backend, a.data() + 1, b.data() + 1, output.data() + 1);
        else generated_cpu(a.data() + 1, b.data() + 1, output.data() + 1, n, threads);
    }
    bool agrees() const {
        if (a.front() != buffers::sentinel || a.back() != buffers::sentinel ||
            b.front() != buffers::sentinel || b.back() != buffers::sentinel ||
            output.front() != buffers::sentinel || output.back() != buffers::sentinel) return false;
        for (std::size_t i = 0; i != n; ++i) {
            if (a[i + 1] != real(.25) + real(i % 31) / real(32) ||
                b[i + 1] != real(.125) + real(i % 19) / real(64) ||
                output[i + 1] != a[i + 1] + real(.25) * b[i + 1]) return false;
        }
        return true;
    }
};
static sample measure_memory(memory_buffers& data, int backend, int threads) {
    sample result;
    const auto start = clock_type::now();
    do {
        data.execute(backend, threads);
        ++result.repetitions;
        result.elapsed = std::chrono::duration<double>(clock_type::now() - start).count();
    } while (result.elapsed < .2);
    return result;
}
static bool save_raw(std::FILE* file, const char* kind, const char* backend, std::size_t n,
                     int batch, const sample& value, bool agreement) {
    return std::fprintf(file,
        "{\"kind\":%s,\"backend\":%s,\"items\":%zu,\"batch\":%d,\"repetitions\":%llu,"
        "\"elapsed_seconds\":%.17g,\"wall_seconds\":%.17g,\"agreement_passed\":%s}\n",
        quoted(kind).c_str(), quoted(backend).c_str(), n, batch, value.repetitions,
        value.elapsed, value.elapsed, agreement ? "true" : "false") > 0 && std::fflush(file) == 0;
}
static void print_samples(const std::array<sample, 7>& rows) {
    std::cout << '[';
    for (std::size_t i = 0; i != rows.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"batch\":" << i << ",\"repetitions\":" << rows[i].repetitions
                  << ",\"elapsed_seconds\":" << rows[i].elapsed
                  << ",\"wall_seconds\":" << rows[i].elapsed << '}';
    }
    std::cout << ']';
}
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) return 2;
    char* end = nullptr;
    const long parsed_threads = std::strtol(argv[1], &end, 10);
    const bool identity = argc == 3 && std::string(argv[2]) == "--identity";
    const bool proof = argc == 3 && std::string(argv[2]) == "--proof";
    const bool smoke = argc == 3 && std::string(argv[2]) == "--smoke";
    if (!end || *end || parsed_threads < 1 || parsed_threads > 1024 ||
        (argc == 3 && !identity && !proof && !smoke)) return 2;
    const int threads = int(parsed_threads);
    cpu_set_t affinity; CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) || CPU_COUNT(&affinity) != threads ||
        omp_get_dynamic() || omp_get_proc_bind() != omp_proc_bind_false) return 2;
    const char* wait_policy = std::getenv("OMP_WAIT_POLICY");
    const char* spin_count = std::getenv("GOMP_SPINCOUNT");
    for (const auto* value : {wait_policy, spin_count}) if (value) {
        if (std::strlen(value) > 128) return 2;
        for (const unsigned char c : std::string(value)) if (c < 32 || c > 126) return 2;
    }
    const int thread_limit = omp_get_thread_limit();
    if (thread_limit < threads) { std::cerr << "OpenMP thread limit is below requested budget\n"; return 2; }
    std::array<int, 1024> team_visits{};
    int actual_team = 0;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        actual_team = omp_get_num_threads();
        const int id = omp_get_thread_num();
        if (id >= 0 && id < threads) ++team_visits[id];
    }
    if (actual_team != threads) { std::cerr << "actual OpenMP team does not match requested budget\n"; return 2; }
    for (int id = 0; id != threads; ++id) if (team_visits[id] != 1) return 2;
    if (proof && (!fort_cpu_protocol_proof_begin || !fort_cpu_protocol_proof_end)) {
        std::cerr << "GNU GOMP_parallel proof backend unavailable\n"; return 2;
    }
    char version[8192]{}, options[8192]{};
    fort_numerical_fortran_identity_v2(version, options, 8192);
    std::cout << std::setprecision(17)
              << "{\"kind\":\"cpu_protocol_identity_v1\",\"cpu_threads\":" << threads
              << ",\"precision_bits\":" << CALIBRATION_PRECISION << ",\"cpu_affinity\":[";
    bool first = true;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &affinity)) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << cpu;
    }
    std::cout << "],\"omp_dynamic\":false,\"omp_proc_bind\":\"false\",\"actual_team_threads\":" << actual_team
              << ",\"thread_limit\":" << thread_limit << ",\"omp_wait_policy\":"
              << (wait_policy ? quoted(wait_policy) : "null") << ",\"gomp_spincount\":"
              << (spin_count ? quoted(spin_count) : "null") << ',' <<
                 "\"proof_backend\":\"GNU GOMP_parallel wrap v1\",\"fortran\":{"
                 "\"compiler_version\":" << quoted(version) << ",\"compiler_options\":"
              << quoted(options) << "}}\n" << std::flush;
    if (identity) return 0;
    std::unique_ptr<std::FILE, close_file> raw;
    if (!proof && !smoke) {
        raw.reset(std::fopen("cpu-protocol-raw-samples.jsonl", "wx"));
        if (!raw) { std::cerr << "cannot create fresh CPU protocol raw observations\n"; return 4; }
    }
    for (const auto n : {std::size_t(0), std::size_t(1), std::size_t(8)}) {
        std::array<buffers, 2> data;
        for (int backend = 0; backend != 2; ++backend) {
            if (proof) fort_cpu_protocol_proof_begin(threads);
            data[backend].execute(backend, n, threads);
            if (proof || smoke) {
                std::cout << "{\"kind\":\"cpu_team_proof_v1\",\"backend\":" << quoted(backends[backend])
                          << ",\"items\":" << n << ",\"agreement_passed\":"
                          << (data[backend].agrees(n) ? "true" : "false");
                if (proof) fort_cpu_protocol_proof_end();
                std::cout << "}\n" << std::flush;
            }
        }
        if (proof || smoke) continue;
        std::array<std::array<sample, 7>, 2> rows;
        for (int batch = 0; batch != 7; ++batch) {
            for (int order = 0; order != 2; ++order) {
                const int backend = (batch + order) % 2;
                rows[backend][batch] = measure(backend, n, threads, data[backend]);
                if (!save_raw(raw.get(), "cpu_startup_batch_v1", backends[backend], n, batch,
                              rows[backend][batch], data[backend].agrees(n))) return 4;
            }
        }
        for (int backend = 0; backend != 2; ++backend) {
            std::cout << "{\"kind\":\"cpu_startup_cost_v1\",\"backend\":" << quoted(backends[backend])
                      << ",\"items\":" << n << ",\"agreement_passed\":"
                      << (data[backend].agrees(n) ? "true" : "false") << ",\"samples\":";
            print_samples(rows[backend]); std::cout << "}\n" << std::flush;
        }
    }
    if (proof || smoke) return 0;
    // Fresh three-array observations share the proven execution identity. All
    // cells agree before timing; seven rounds interleave the complete lattice.
    for (const auto n : memory_sizes) {
        memory_buffers data(n);
        for (int backend = 0; backend != 3; ++backend) {
            std::fill(data.output.begin(), data.output.end(), buffers::sentinel);
            data.execute(backend, threads);
            if (!data.agrees()) { std::cerr << "fresh CPU memory correctness failed\n"; return 3; }
        }
    }
    std::array<std::array<std::array<sample, 7>, 3>, 17> memory_samples{};
    std::array<std::array<bool, 3>, 17> memory_agreement;
    for (auto& row : memory_agreement) row.fill(true);
    for (int batch = 0; batch != 7; ++batch) {
        for (std::size_t index = 0; index != memory_sizes.size(); ++index) {
            memory_buffers data(memory_sizes[index]);
            for (int order = 0; order != 3; ++order) {
                const int backend = (batch + order) % 3;
                auto& observation = memory_samples[index][backend][batch];
                observation = measure_memory(data, backend, threads);
                const bool agreement = data.agrees();
                memory_agreement[index][backend] = memory_agreement[index][backend] && agreement;
                if (!save_raw(raw.get(), "cpu_memory_batch_v1", memory_backends[backend], data.n,
                              batch, observation, agreement)) return 4;
            }
        }
    }
    for (std::size_t index = 0; index != memory_sizes.size(); ++index) for (int backend = 0; backend != 3; ++backend) {
        const auto n = memory_sizes[index];
        std::cout << "{\"kind\":\"cpu_memory_cost_v1\",\"backend\":" << quoted(memory_backends[backend])
                  << ",\"items\":" << n << ",\"role\":" << quoted(is_memory_fit(n) ? "fit" : "holdout")
                  << ",\"traffic_bytes\":" << n * 3 * sizeof(real) << ",\"working_set_bytes\":" << n * 3 * sizeof(real)
                  << ",\"agreement_passed\":" << (memory_agreement[index][backend] ? "true" : "false") << ",\"samples\":";
        print_samples(memory_samples[index][backend]); std::cout << "}\n" << std::flush;
    }
    return 0;
}
