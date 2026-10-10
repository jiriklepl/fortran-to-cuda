"""Bounded untimed driver checks with a small independent numerical registry."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.offload.cpu_protocol_calibration import GENERATED_CONTROL_HEADER, generated_cpu_control_source

DRIVER = Path(__file__).parents[1] / "offload" / "cpu_dependency_driver.cpp"
PROTOCOL_DRIVER = Path(__file__).parents[1] / "offload" / "cpu_protocol_calibration.cpp"


def toy_registry(precision):
    rows = []
    for i in range(26):
        role = ("basis", "structural_holdout", "domain_holdout")[i % 3]
        rows.append('{"recipe_' + str(i) + '","' + f"{i:064x}" + '","' + role +
                    '","ordinary",1,serial,parallel,generated,' +
                    'a,1,d,1,reference,1,a,1,d,1,reference,1}')
    return """#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
using dependency_real=""" + ("double" if precision == 64 else "float") + """;
using dependency_worker=void(*)(int,int,const dependency_real*,const dependency_real*,dependency_real*);
inline void serial(int n,int,const dependency_real*a,const dependency_real*d,dependency_real*out){
if(std::getenv("TOY_REQUIRE_NO_EXECUTION"))std::exit(42);
for(int i=0;i<n;++i)out[i]=a[i]+dependency_real(.25)*d[i];
}
inline void parallel(int n,int threads,const dependency_real*a,const dependency_real*d,dependency_real*out){
if(std::getenv("TOY_REQUIRE_NO_EXECUTION"))std::exit(42);
#pragma omp parallel for num_threads(threads) schedule(static)
for(int i=0;i<n;++i)out[i]=a[i]+dependency_real(.25)*d[i];
}
inline void generated(int n,int threads,const dependency_real*a,const dependency_real*d,dependency_real*out){
parallel(n,threads,a,d,out);
#ifdef TOY_BAD_HALO
out[-1]=dependency_real(1);
#endif
}
struct dependency_recipe{
const char*name;const char*identity;const char*role;const char*coefficient_family;int width;
dependency_worker native_serial,native_fork_join,generated_cpu;
const dependency_real*training_a;std::size_t training_a_count;
const dependency_real*training_d;std::size_t training_d_count;
const dependency_real*training_reference;std::size_t training_reference_count;
const dependency_real*holdout_a;std::size_t holdout_a_count;
const dependency_real*holdout_d;std::size_t holdout_d_count;
const dependency_real*holdout_reference;std::size_t holdout_reference_count;
};
inline constexpr dependency_real a[]={.5},d[]={.25},reference[]={.5625};
inline constexpr int dependency_fit_sizes[]={65536,262144,1048576};
inline constexpr int dependency_holdout_sizes[]={131072,524288};
inline constexpr const char*dependency_generator_identity=""" + '"' + "f" * 64 + '";' + """
extern "C" inline void fort_cpu_dependency_fortran_identity_v1(char*version,char*options,int capacity){
std::strncpy(version,"Toy \\\"native\\\" compiler",capacity-1);
std::strncpy(options,"-O3 -fopenmp",capacity-1);
}
inline constexpr dependency_recipe dependency_recipes[]={
""" + ",\n".join(rows) + "\n};\n"


@pytest.fixture(scope="module", params=(32, 64))
def compiled_driver(request, tmp_path_factory):
    cxx = shutil.which("g++")
    if not cxx or not hasattr(os, "sched_getaffinity") or not shutil.which("taskset"):
        pytest.skip("GNU OpenMP/Linux CPU fixture backend unavailable")
    target = tmp_path_factory.mktemp("dependency-driver-" + str(request.param))
    (target / "cpu_dependency_recipes.hpp").write_text(toy_registry(request.param))
    binary = target / "driver"
    result = subprocess.run([cxx, "-O0", "-std=c++17", "-fopenmp", "-I", str(target), str(DRIVER),
                             "-o", str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return binary, request.param


@pytest.fixture(scope="module", params=(32, 64))
def compiled_protocol_driver(request, tmp_path_factory):
    """Exercise protocol guards without compiling/calibrating numerical fixtures."""
    cxx = shutil.which("g++")
    if not cxx or not hasattr(os, "sched_getaffinity") or not shutil.which("taskset"):
        pytest.skip("GNU OpenMP/Linux CPU fixture backend unavailable")
    target = tmp_path_factory.mktemp("protocol-driver-" + str(request.param))
    (target / GENERATED_CONTROL_HEADER).write_text(generated_cpu_control_source(request.param))
    stub = target / "native-stub.cpp"
    stub.write_text("""#include <cstddef>
#include <cstdlib>
#include <cstring>
using real=""" + ("double" if request.param == 64 else "float") + """;
extern "C" void fort_numerical_native_v2(int family,std::size_t n,int threads,int parallel,
const real*a,const real*b,real*out){
if(std::getenv("TOY_REQUIRE_NO_EXECUTION") || family != 1)std::exit(42);
if(parallel){
#pragma omp parallel for num_threads(threads) schedule(static)
for(std::size_t i=0;i<n;++i)out[i]=a[i]+real(.25)*b[i];
}else for(std::size_t i=0;i<n;++i)out[i]=a[i]+real(.25)*b[i];
}
extern "C" void fort_numerical_fortran_identity_v2(char*version,char*options,int capacity){
std::strncpy(version,"Toy native compiler",capacity-1);
std::strncpy(options,"-O3 -fopenmp",capacity-1);
}
""")
    binary = target / "driver"
    result = subprocess.run([cxx, "-O0", "-std=c++17", "-fopenmp", "-I", str(target),
                             "-DCALIBRATION_PRECISION=" + str(request.param), str(PROTOCOL_DRIVER), str(stub),
                             "-o", str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return binary, request.param


def invoke(binary, *arguments, extra_env=None):
    affinity = sorted(os.sched_getaffinity(0))[:2]
    environment = {key: value for key, value in os.environ.items() if key not in {
        "OMP_PLACES", "GOMP_CPU_AFFINITY", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT"}}
    environment.update(OMP_DYNAMIC="FALSE", OMP_PROC_BIND="false", OMP_THREAD_LIMIT="2147483647")
    environment.update(extra_env or {})
    return subprocess.run(["taskset", "-c", ",".join(map(str, affinity)), str(binary), *map(str, arguments)],
                          capture_output=True, text=True, timeout=10, env=environment, cwd=binary.parent)


def test_identity_does_not_invoke_numerical_workers(compiled_driver):
    binary, precision = compiled_driver
    result = invoke(binary, min(2, len(os.sched_getaffinity(0))), "--identity",
                    extra_env={"TOY_REQUIRE_NO_EXECUTION": "1"})
    assert result.returncode == 0, result.stderr
    identity, = [json.loads(line) for line in result.stdout.splitlines()]
    assert identity["kind"] == "cpu_dependency_identity_v1"
    assert identity["precision_bits"] == precision
    assert identity["registry_generator_id"] == "f" * 64
    assert identity["fortran"]["compiler_version"] == 'Toy "native" compiler'
    assert identity["omp_dynamic"] is False
    assert identity["actual_team_threads"] == identity["cpu_threads"]
    assert identity["thread_limit"] >= identity["cpu_threads"]
    assert identity["omp_wait_policy"] is None
    assert identity["gomp_spincount"] is None
    assert not (binary.parent / "cpu-dependency-raw-samples.jsonl").exists()


def test_protocol_identity_and_empty_controls_are_untimed(compiled_protocol_driver):
    binary, precision = compiled_protocol_driver
    threads = min(2, len(os.sched_getaffinity(0)))
    identity_result = invoke(binary, threads, "--identity", extra_env={"TOY_REQUIRE_NO_EXECUTION": "1"})
    assert identity_result.returncode == 0, identity_result.stderr
    identity, = [json.loads(line) for line in identity_result.stdout.splitlines()]
    assert identity["kind"] == "cpu_protocol_identity_v1"
    assert identity["precision_bits"] == precision
    assert identity["actual_team_threads"] == threads
    assert identity["thread_limit"] >= threads
    assert identity["omp_wait_policy"] is None
    assert identity["gomp_spincount"] is None
    smoke_result = invoke(binary, threads, "--smoke")
    assert smoke_result.returncode == 0, smoke_result.stderr
    rows = [json.loads(line) for line in smoke_result.stdout.splitlines()]
    assert len(rows) == 7
    assert {row["items"] for row in rows[1:]} == {0, 1, 8}
    assert all(row["agreement_passed"] is True for row in rows[1:])
    assert not any("samples" in row for row in rows)
    assert not (binary.parent / "cpu-protocol-raw-samples.jsonl").exists()


def assert_actual_thread_limit_rejects(compiled):
    binary, _ = compiled
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("requires two available CPUs")
    result = invoke(binary, 2, "--identity", extra_env={"OMP_THREAD_LIMIT": "1", "TOY_REQUIRE_NO_EXECUTION": "1"})
    assert result.returncode == 2
    assert not result.stdout
    assert "thread limit" in result.stderr


def test_dependency_actual_thread_limit_rejects_before_numerical_work(compiled_driver):
    assert_actual_thread_limit_rejects(compiled_driver)


def test_protocol_actual_thread_limit_rejects_before_numerical_work(compiled_protocol_driver):
    assert_actual_thread_limit_rejects(compiled_protocol_driver)


def assert_wait_and_spin_recorded(compiled):
    binary, _ = compiled
    threads = min(2, len(os.sched_getaffinity(0)))
    result = invoke(binary, threads, "--identity", extra_env={"OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "10000"})
    assert result.returncode == 0, result.stderr
    identity, = [json.loads(line) for line in result.stdout.splitlines()]
    assert identity["omp_wait_policy"] == "PASSIVE"
    assert identity["gomp_spincount"] == "10000"


def test_dependency_wait_and_spin_environment_is_not_overridden(compiled_driver):
    assert_wait_and_spin_recorded(compiled_driver)


def test_protocol_wait_and_spin_environment_is_not_overridden(compiled_protocol_driver):
    assert_wait_and_spin_recorded(compiled_protocol_driver)


def assert_unbounded_environment_rejects(compiled, field, value):
    binary, _ = compiled
    threads = min(2, len(os.sched_getaffinity(0)))
    result = invoke(binary, threads, "--identity", extra_env={field: value, "TOY_REQUIRE_NO_EXECUTION": "1"})
    assert result.returncode == 2
    assert not result.stdout


@pytest.mark.parametrize(("field", "value"), [("OMP_WAIT_POLICY", "x" * 129), ("GOMP_SPINCOUNT", "1\n2")])
def test_dependency_unbounded_environment_rejects_before_identity(compiled_driver, field, value):
    assert_unbounded_environment_rejects(compiled_driver, field, value)


@pytest.mark.parametrize(("field", "value"), [("OMP_WAIT_POLICY", "x" * 129), ("GOMP_SPINCOUNT", "1\n2")])
def test_protocol_unbounded_environment_rejects_before_identity(compiled_protocol_driver, field, value):
    assert_unbounded_environment_rejects(compiled_protocol_driver, field, value)


def test_smoke_checks_every_recipe_role_and_empty_domain_without_timings(compiled_driver):
    binary, _ = compiled_driver
    result = invoke(binary, min(2, len(os.sched_getaffinity(0))), "--smoke")
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 1 + 26 * 4 * 3
    smoke = rows[1:]
    assert all(row["kind"] == "cpu_dependency_smoke_v1" for row in smoke)
    assert all(row["agreement_passed"] is True for row in smoke)
    assert {row["items"] for row in smoke} == {0, 1, 8, 17}
    assert {row["backend"] for row in smoke} == {"native_serial", "native_fork_join", "generated_cpu"}
    assert not any("samples" in row for row in rows)
    assert not (binary.parent / "cpu-dependency-raw-samples.jsonl").exists()


@pytest.mark.parametrize("arguments", [(0, "--smoke"), ("bad", "--smoke"), (2, "--unknown"), (1025, "--identity")])
def test_invalid_cli_never_starts_timed_protocol(compiled_driver, arguments):
    binary, _ = compiled_driver
    result = invoke(binary, *arguments)
    assert result.returncode == 2
    assert not result.stdout


def test_mismatched_thread_budget_is_rejected_before_numerical_work(compiled_driver):
    binary, _ = compiled_driver
    result = invoke(binary, 3, "--identity")
    assert result.returncode == 2
    assert not result.stdout


def test_corrupt_halo_stops_untimed_smoke_before_sampling(tmp_path):
    cxx = shutil.which("g++")
    if not cxx or not hasattr(os, "sched_getaffinity") or not shutil.which("taskset"):
        pytest.skip("GNU OpenMP/Linux CPU fixture backend unavailable")
    (tmp_path / "cpu_dependency_recipes.hpp").write_text(toy_registry(64))
    binary = tmp_path / "bad-driver"
    built = subprocess.run([cxx, "-O0", "-std=c++17", "-fopenmp", "-DTOY_BAD_HALO=1", "-I", str(tmp_path),
                            str(DRIVER), "-o", str(binary)], capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stderr
    result = invoke(binary, min(2, len(os.sched_getaffinity(0))), "--smoke")
    assert result.returncode == 3
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[-1]["backend"] == "generated_cpu"
    assert rows[-1]["agreement_passed"] is False
    assert not any("samples" in row for row in rows)
