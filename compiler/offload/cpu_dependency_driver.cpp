// Predeclared CPU work/span observations. No application or online tuning.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <omp.h>
#include <sched.h>
#include <set>
#include <string>
#include <vector>

#include "cpu_dependency_recipes.hpp"

extern "C" void fort_cpu_dependency_fortran_identity_v1(char*, char*, int);
using clock_type = std::chrono::steady_clock;
using real = dependency_real;
constexpr std::array<const char*, 3> backends{"native_serial", "native_fork_join", "generated_cpu"};
constexpr std::array<int, 4> smoke_sizes{0, 1, 8, 17};

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
static bool fit_size(int n) {
    return std::find(std::begin(dependency_fit_sizes), std::end(dependency_fit_sizes), n) !=
           std::end(dependency_fit_sizes);
}
static bool training(const dependency_recipe& recipe, int n) {
    return std::string(recipe.role) == "basis" && fit_size(n);
}
static dependency_worker worker(const dependency_recipe& recipe, int backend) {
    return backend == 0 ? recipe.native_serial : backend == 1 ? recipe.native_fork_join : recipe.generated_cpu;
}
static bool reference_agrees(real actual, real expected) {
    if (std::isnan(expected)) return std::isnan(actual);
    if (actual == expected) return actual != real(0) || std::signbit(actual) == std::signbit(expected);
    if (!std::isfinite(actual) || !std::isfinite(expected)) return false;
    constexpr double tolerance = sizeof(real) == sizeof(double) ? 2e-12 : 2e-5;
    return std::abs(double(actual) - double(expected)) <= tolerance * (1.0 + std::abs(double(expected)));
}
struct buffers {
    static constexpr real sentinel = real(-131.25);
    std::vector<real> a, d, output;
    const dependency_recipe& recipe;
    int n;
    bool fit;
    buffers(const dependency_recipe& selected, int items): a(items + 2), d(items + 2), output(items + 2),
        recipe(selected), n(items), fit(training(selected, items)) {
        a.front() = a.back() = d.front() = d.back() = sentinel;
        const auto* lattice_a = fit ? recipe.training_a : recipe.holdout_a;
        const auto* lattice_d = fit ? recipe.training_d : recipe.holdout_d;
        const auto count_a = fit ? recipe.training_a_count : recipe.holdout_a_count;
        const auto count_d = fit ? recipe.training_d_count : recipe.holdout_d_count;
        for (int i = 0; i != n; ++i) { a[i + 1] = lattice_a[i % count_a]; d[i + 1] = lattice_d[i % count_d]; }
        reset();
    }
    void reset() { std::fill(output.begin(), output.end(), sentinel); }
    void execute(int backend, int threads) {
        worker(recipe, backend)(n, threads, a.data() + 1, d.data() + 1, output.data() + 1);
    }
    bool agrees() const {
        if (a.front() != sentinel || a.back() != sentinel || d.front() != sentinel || d.back() != sentinel ||
            output.front() != sentinel || output.back() != sentinel) return false;
        const auto* lattice_a = fit ? recipe.training_a : recipe.holdout_a;
        const auto* lattice_d = fit ? recipe.training_d : recipe.holdout_d;
        const auto* reference = fit ? recipe.training_reference : recipe.holdout_reference;
        const auto count_a = fit ? recipe.training_a_count : recipe.holdout_a_count;
        const auto count_d = fit ? recipe.training_d_count : recipe.holdout_d_count;
        const auto count_ref = fit ? recipe.training_reference_count : recipe.holdout_reference_count;
        for (int i = 0; i != n; ++i) {
            if (a[i + 1] != lattice_a[i % count_a] || d[i + 1] != lattice_d[i % count_d] ||
                !reference_agrees(output[i + 1], reference[i % count_ref])) return false;
        }
        return true;
    }
};
struct sample { unsigned long long repetitions{}; double elapsed{}; };
struct close_file { void operator()(std::FILE* file) const noexcept { std::fclose(file); } };
static sample measure(buffers& data, int backend, int threads) {
    sample result;
    const auto start = clock_type::now();
    do {
        data.execute(backend, threads);
        ++result.repetitions;
        result.elapsed = std::chrono::duration<double>(clock_type::now() - start).count();
    } while (result.elapsed < .2);
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
static bool save_raw_sample(std::FILE* file, const dependency_recipe& recipe, int n, int backend,
                            int batch, const sample& value, bool agreement) {
    const int written = std::fprintf(file,
        "{\"kind\":\"cpu_dependency_batch_v1\",\"recipe\":%s,\"recipe_identity\":%s,"
        "\"backend\":%s,\"items\":%d,\"role\":%s,\"batch\":%d,\"repetitions\":%llu,"
        "\"elapsed_seconds\":%.17g,\"wall_seconds\":%.17g,\"agreement_passed\":%s}\n",
        quoted(recipe.name).c_str(), quoted(recipe.identity).c_str(), quoted(backends[backend]).c_str(),
        n, quoted(training(recipe, n) ? "fit" : "holdout").c_str(), batch, value.repetitions,
        value.elapsed, value.elapsed, agreement ? "true" : "false");
    return written > 0 && std::fflush(file) == 0;
}
static bool registry_valid() {
    if (std::size(dependency_recipes) != 26 || std::size(dependency_fit_sizes) != 3 ||
        std::size(dependency_holdout_sizes) != 2 || (sizeof(real) != 4 && sizeof(real) != 8)) return false;
    std::set<std::string> names, identities;
    for (const auto& recipe : dependency_recipes) {
        if (!recipe.name || !recipe.identity || !recipe.role || !recipe.native_serial ||
            !recipe.native_fork_join || !recipe.generated_cpu || !recipe.training_a || !recipe.training_d ||
            !recipe.training_reference || !recipe.holdout_a || !recipe.holdout_d || !recipe.holdout_reference)
            return false;
        const std::string identity(recipe.identity), role(recipe.role);
        if (!names.insert(recipe.name).second || !identities.insert(identity).second || identity.size() != 64 ||
            identity.find_first_not_of("0123456789abcdef") != std::string::npos ||
            (role != "basis" && role != "structural_holdout" && role != "domain_holdout")) return false;
        for (const auto count : {recipe.training_a_count, recipe.training_d_count, recipe.training_reference_count,
                                 recipe.holdout_a_count, recipe.holdout_d_count, recipe.holdout_reference_count})
            if (count < 1 || count > 256) return false;
        if (recipe.training_reference_count % recipe.training_a_count ||
            recipe.training_reference_count % recipe.training_d_count ||
            recipe.holdout_reference_count % recipe.holdout_a_count ||
            recipe.holdout_reference_count % recipe.holdout_d_count) return false;
    }
    std::set<int> sizes;
    for (int n : dependency_fit_sizes) if (n < 1 || n > 1048576 || !sizes.insert(n).second) return false;
    for (int n : dependency_holdout_sizes) if (n < 1 || n > 1048576 || !sizes.insert(n).second) return false;
    return sizes == std::set<int>{65536, 131072, 262144, 524288, 1048576};
}
int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) return 2;
    char* end = nullptr;
    const long parsed_threads = std::strtol(argv[1], &end, 10);
    const bool identity_only = argc == 3 && std::string(argv[2]) == "--identity";
    const bool smoke = argc == 3 && std::string(argv[2]) == "--smoke";
    if (!end || *end || parsed_threads < 1 || parsed_threads > 1024 ||
        (argc == 3 && !identity_only && !smoke) || !registry_valid()) return 2;
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
    char version[8192]{}, options[8192]{};
    fort_cpu_dependency_fortran_identity_v1(version, options, 8192);
    std::cout << std::setprecision(17)
              << "{\"kind\":\"cpu_dependency_identity_v1\",\"cpu_threads\":" << threads
              << ",\"precision_bits\":" << sizeof(real) * 8 << ",\"cpu_affinity\":[";
    bool first = true;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &affinity)) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << cpu;
    }
    std::cout << "],\"omp_dynamic\":false,\"omp_proc_bind\":\"false\",\"actual_team_threads\":" << actual_team
              << ",\"thread_limit\":" << thread_limit << ",\"omp_wait_policy\":"
              << (wait_policy ? quoted(wait_policy) : "null") << ",\"gomp_spincount\":"
              << (spin_count ? quoted(spin_count) : "null") << ",\"registry_generator_id\":"
              << quoted(dependency_generator_identity) << ",\"fortran\":{"
                 "\"compiler_version\":" << quoted(version) << ",\"compiler_options\":"
              << quoted(options) << "}}\n" << std::flush;
    if (identity_only) return 0;
    std::vector<int> sizes = smoke ? std::vector<int>(smoke_sizes.begin(), smoke_sizes.end()) : std::vector<int>{};
    if (!smoke) {
        sizes.insert(sizes.end(), std::begin(dependency_fit_sizes), std::end(dependency_fit_sizes));
        sizes.insert(sizes.end(), std::begin(dependency_holdout_sizes), std::end(dependency_holdout_sizes));
        std::sort(sizes.begin(), sizes.end());
    }
    // Validate every predeclared recipe/size/backend before any timing starts.
    for (const auto& recipe : dependency_recipes) for (int n : sizes) {
        buffers data(recipe, n);
        for (int backend = 0; backend != 3; ++backend) {
            data.reset(); data.execute(backend, threads);
            const bool agreement = data.agrees();
            if (smoke) std::cout << "{\"kind\":\"cpu_dependency_smoke_v1\",\"recipe\":" << quoted(recipe.name)
                                << ",\"recipe_identity\":" << quoted(recipe.identity)
                                << ",\"backend\":" << quoted(backends[backend]) << ",\"items\":" << n
                                << ",\"agreement_passed\":" << (agreement ? "true" : "false") << "}\n" << std::flush;
            if (!agreement) {
                std::cerr << "dependency correctness failed for " << recipe.name << "," << backends[backend]
                          << ",items=" << n << '\n'; return 3;
            }
        }
    }
    if (smoke) return 0;
    // Exclusive raw output retains every completed batch after interruption.
    // Identity and smoke create no numerical measurement artifacts.
    std::unique_ptr<std::FILE, close_file> raw(std::fopen("cpu-dependency-raw-samples.jsonl", "wx"));
    if (!raw) { std::cerr << "cannot create fresh dependency raw observations\n"; return 4; }
    constexpr std::size_t cell_count = 26 * 5;
    std::array<std::array<std::array<sample, 7>, 3>, cell_count> samples{};
    std::array<std::array<bool, 3>, cell_count> agreements;
    for (auto& agreement : agreements) agreement.fill(true);
    // The seven rounds cover the complete fixed lattice, not one cell at a
    // time. Within each round backend order rotates without observing costs.
    for (int batch = 0; batch != 7; ++batch) {
        std::size_t cell = 0;
        for (const auto& recipe : dependency_recipes) for (int n : sizes) {
            buffers data(recipe, n);
            for (int order = 0; order != 3; ++order) {
                const int backend = (batch + order) % 3;
                auto& observation = samples[cell][backend][batch];
                observation = measure(data, backend, threads);
                const bool agreement = data.agrees();
                agreements[cell][backend] = agreements[cell][backend] && agreement;
                if (!save_raw_sample(raw.get(), recipe, n, backend, batch, observation, agreement)) {
                    std::cerr << "cannot preserve dependency raw observation\n"; return 4;
                }
            }
            ++cell;
        }
    }
    std::size_t cell = 0;
    for (const auto& recipe : dependency_recipes) for (int n : sizes) {
        for (int backend = 0; backend != 3; ++backend) {
            std::cout << "{\"kind\":\"cpu_dependency_cost_v1\",\"recipe\":" << quoted(recipe.name)
                      << ",\"recipe_identity\":" << quoted(recipe.identity) << ",\"backend\":" << quoted(backends[backend])
                      << ",\"items\":" << n << ",\"role\":" << quoted(training(recipe, n) ? "fit" : "holdout")
                      << ",\"agreement_passed\":" << (agreements[cell][backend] ? "true" : "false") << ",\"samples\":";
            print_samples(samples[cell][backend]); std::cout << "}\n" << std::flush;
        }
        ++cell;
    }
    return 0;
}
