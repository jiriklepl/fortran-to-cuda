// Validate frozen native costs under a runtime static schedule; never fit them.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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
using worker = void (*)(int, std::size_t, int, int, const real*, const real*, real*);
extern "C" void fort_numerical_native_v2(int, std::size_t, int, int, const real*, const real*, real*);
extern "C" void fort_numerical_native_runtime_v1(int, std::size_t, int, int, const real*, const real*, real*);
extern "C" void fort_numerical_fortran_identity_v2(char*, char*, int);
extern "C" void fort_numerical_runtime_fortran_identity_v1(char*, char*, int);

static std::string quoted(const char* text) {
    std::string result = "\"";
    for (const unsigned char c : std::string(text)) {
        if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
        else if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else if (c == '\t') result += "\\t";
        else if (c < 32) { std::cerr << "unsupported identity control byte\n"; std::exit(2); }
        else result += char(c);
    }
    return result + '"';
}
struct sample { unsigned long repetitions{}; double elapsed{}; };
static sample measure(worker run, int family, std::size_t n, int threads,
                      const std::vector<real>& a, const std::vector<real>& b,
                      std::vector<real>& output) {
    const auto started = clock_type::now();
    sample result;
    do {
        run(family, n, threads, 1, a.data(), b.data(), output.data());
        ++result.repetitions;
        result.elapsed = std::chrono::duration<double>(clock_type::now() - started).count();
    } while (result.elapsed < 0.2);
    return result;
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
static constexpr const char* families[] = {
    "arithmetic_v2", "memory_v2", "primitive_sqrt_v2", "primitive_acos_v2",
    "primitive_cos_v2", "scalar_mix_v2", "scalar_mix_skew_v2", "private_mix_v2",
    "private_mix_skew_v2", "primitive_divide_v2", "ordinary_mix_v2"};
static bool agrees(const std::vector<real>& a, const std::vector<real>& b, std::size_t n) {
    for (std::size_t i = 0; i != n; ++i) {
        if (std::isnan(a[i]) && std::isnan(b[i])) continue;
        if (a[i] != b[i]) return false;
    }
    return true;
}
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) return 2;
    const int threads = std::atoi(argv[1]);
    const bool smoke = argc == 3 && std::string(argv[2]) == "--smoke";
    const bool identity_only = argc == 3 && std::string(argv[2]) == "--identity";
    if (threads < 1 || (argc == 3 && !smoke && !identity_only)) return 2;
    cpu_set_t affinity; CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) || CPU_COUNT(&affinity) != threads ||
        omp_get_dynamic() || omp_get_proc_bind() != omp_proc_bind_false) return 2;
    omp_sched_t schedule; int chunk = -1;
    omp_get_schedule(&schedule, &chunk);
    const unsigned schedule_kind = static_cast<unsigned>(schedule) & 0x7fffffffU;
    if (schedule_kind != static_cast<unsigned>(omp_sched_static) || chunk != 0) return 2;
    int actual = 0;
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp single
        actual = omp_get_num_threads();
    }
    if (actual != threads) return 2;
    std::cout << std::setprecision(17);
    std::cout << "{\"kind\":\"schedule_identity_v1\",\"cpu_threads\":" << actual
              << ",\"precision_bits\":" << CALIBRATION_PRECISION << ",\"cpu_affinity\":[";
    bool first = true;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &affinity)) {
        if (!first) std::cout << ','; first = false; std::cout << cpu;
    }
    std::cout << "],\"runtime_schedule\":{\"kind\":\"static\",\"chunk\":0},"
                 "\"omp_dynamic\":false,\"omp_proc_bind\":\"false\",\"fortran\":{";
    for (int mode = 0; mode != 2; ++mode) {
        char version[8192]{}, options[8192]{};
        if (mode) fort_numerical_runtime_fortran_identity_v1(version, options, 8192);
        else fort_numerical_fortran_identity_v2(version, options, 8192);
        if (mode) std::cout << ',';
        std::cout << (mode ? "\"runtime\":" : "\"static\":")
                  << "{\"compiler_version\":" << quoted(version)
                  << ",\"compiler_options\":" << quoted(options) << '}';
    }
    std::cout << "}}\n" << std::flush;
    if (identity_only) return 0;
    constexpr std::size_t maximum_n = 1119744;
    std::vector<real> a(maximum_n), b(maximum_n), expected(maximum_n), output(maximum_n);
    for (std::size_t i = 0; i != maximum_n; ++i) {
        a[i] = real(.2) + real(i % 31) * real(.01);
        b[i] = real(.125) + real(i % 19) * real(.005);
    }
    for (int family = 0; family != 11; ++family) {
        const std::vector<std::size_t> sizes = smoke ? std::vector<std::size_t>{16, 257} :
            family == 1 ? std::vector<std::size_t>{65536,80265,98304,120397,131072,147456,
                180596,221184,270894,331776,406341,497664,524288,609511,746496,914267,1119744} :
                std::vector<std::size_t>{65536,131072,262144,524288,1048576};
        for (const auto n : sizes) {
            fort_numerical_native_v2(family, n, threads, 1, a.data(), b.data(), expected.data());
            fort_numerical_native_runtime_v1(family, n, threads, 1, a.data(), b.data(), output.data());
            const bool agreement = agrees(expected, output, n);
            if (smoke) {
                std::cout << "{\"kind\":\"schedule_smoke_v1\",\"family\":" << quoted(families[family])
                          << ",\"items\":" << n << ",\"agreement_passed\":"
                          << (agreement ? "true" : "false") << "}\n" << std::flush;
                if (!agreement) return 3;
                continue;
            }
            std::array<sample, 7> static_rows, runtime_rows;
            for (int batch = 0; batch != 7; ++batch) {
                // Predetermined alternation; neither timing controls ordering.
                if (batch % 2 == 0) {
                    static_rows[batch] = measure(fort_numerical_native_v2, family, n, threads, a, b, expected);
                    runtime_rows[batch] = measure(fort_numerical_native_runtime_v1, family, n, threads, a, b, output);
                } else {
                    runtime_rows[batch] = measure(fort_numerical_native_runtime_v1, family, n, threads, a, b, output);
                    static_rows[batch] = measure(fort_numerical_native_v2, family, n, threads, a, b, expected);
                }
            }
            std::cout << "{\"kind\":\"schedule_cost_v1\",\"family\":" << quoted(families[family])
                      << ",\"items\":" << n << ",\"agreement_passed\":"
                      << ((agreement && agrees(expected, output, n)) ? "true" : "false") << ",\"samples\":";
            print_samples(runtime_rows); std::cout << ",\"static_samples\":"; print_samples(static_rows);
            std::cout << "}\n" << std::flush;
        }
    }
    return 0;
}
