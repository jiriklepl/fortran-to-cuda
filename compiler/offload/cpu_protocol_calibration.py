"""Independent CPU execution startup and bounded memory-cost correction.

This optional section preserves numerical-v2 observations and interpretation.
It measures the original fork/join protocol with an empty numerical domain;
the n=1/8 controls and untimed GNU OpenMP wrapper are independent checks.
Arithmetic regression intercepts are never treated as execution startup.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import abi_arguments
from compiler.emission.common.c_family import cpp_type, indent
from compiler.emission.common.resources import read_common_header
from compiler.emission.cuda.offload import _cpu_worker
from compiler.frontend import lower_source
from compiler.ir import ParallelRegion

from .analysis import Unit
from .calibrate import CalibrationError, _run, _tool
from .collective_calibration import normalize_fortran_options
from .numerical_calibration import (
    COMPUTE_CLASSES,
    COMPUTE_COEFFICIENT_FAMILIES,
    COMPUTE_HOLDOUT_SIZES,
    COMPUTE_RECIPES,
    COMPUTE_SIZES,
    MAX_HOLDOUT_RELATIVE_ERROR,
    MEMORY_FIT_SIZES,
    MEMORY_SIZES,
    NumericalCalibrationError,
    _compute_samples,
    memory_compute_seconds,
    native_fixture_source,
    validate_numerical_profile,
)
from .schedule_calibrate import measurement_environment

CPU_PROTOCOL_ID = "empty-fork-join-cyclic-memory-v2"
CPU_BACKENDS = ("native_serial", "native_fork_join", "generated_cpu")
MEASURED_BACKENDS = CPU_BACKENDS[1:]
STARTUP_SIZES = (0, 1, 8)
HOST_FLAGS = ("-O3", "-std=c++17", "-fopenmp")
FIXED_COVERAGE = "one original fork/join; outside max(compute,variable_memory)"
MEMORY_ACCESS_CLASS = "pointwise_three_array_v1"
OPENMP_ENVIRONMENT_LIMIT = 128
GENERATED_CONTROL_HEADER = "cpu_protocol_generated.hpp"
WORKER_RENDERER_DEPENDENCIES = (
    "emission/cuda/offload.py", "emission/common/loops.py", "emission/common/c_family.py",
    "emission/common/abi.py", "emission/common/symbols.py", "emission/common/schedules.py",
    "emission/c/declarations.py",
)


@lru_cache(maxsize=2)
def generated_cpu_control_source(precision):
    """Use the production numerical worker and cyclic team dispatch unchanged."""
    if type(precision) is not int or precision not in {32, 64}:
        raise NumericalCalibrationError("CPU protocol precision must be 32 or 64")
    kind = precision // 8
    source = f"""module cpu_protocol_original_control
implicit none
contains
subroutine advance(a,b,output,n)
real({kind}),intent(in)::a(:),b(:)
real({kind}),intent(out)::output(:)
integer,intent(in)::n
integer::i
do i=1,n
output(i)=a(i)+0.25_{kind}*b(i)
enddo
end subroutine
end module
"""
    function, plan = prepare_function(lower_source(source, "cpu_protocol_original_control::advance",
        source_name="cpu-protocol-original-control.f90"), options=CompilerOptions(opt_level=0))
    if len(plan.steps) != 1 or not isinstance(plan.steps[0], ParallelRegion):
        raise NumericalCalibrationError("CPU protocol control must have one independent source region")
    region = plan.steps[0]
    abi = abi_arguments(function.parameters)
    signature = ", ".join(cpp_declaration(argument) if argument.symbol.rank else
        f"const {cpp_type(argument.symbol)} &{argument.name}" for argument in abi)
    worker = _cpu_worker(Unit(0, region, (), None), signature, "pointwise_worker")
    arguments = []
    for argument in abi:
        arguments.append("n" if argument.dimension is not None else
                         {"a": "a", "b": "b", "output": "out", "n": "original_n"}[argument.symbol.name])
    return "\n".join([
        "#pragma once", read_common_header(), "#include <omp.h>",
        f"static_assert(CALIBRATION_PRECISION == {precision}, \"generated CPU control precision mismatch\");",
        "namespace fort_cpu_protocol_generated {", "using namespace generated_kernels::indexing;", *worker,
        "}", "static __attribute__((noinline)) void generated_cpu(const real* a, const real* b,",
        "                                                    real* out, std::size_t n, int threads) {",
        "    const int original_n = static_cast<int>(n);",
        "    #pragma omp parallel num_threads(threads)", "    {",
        *indent(["fort_cpu_protocol_generated::pointwise_worker(" + ", ".join(arguments) +
                 ", omp_get_thread_num(), omp_get_num_threads());"], 2),
        "    }", "}", "",
    ])


def generated_control_identity(precision):
    return sha256(generated_cpu_control_source(precision).encode()).hexdigest()


def worker_renderer_identities():
    root = Path(__file__).parents[1]
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in WORKER_RENDERER_DEPENDENCIES}

# Link only into the separate proof executable, against the exact same objects
# as the uninstrumented timed executable. GNU's public runtime ABI is used;
# no architecture-specific instructions or symbol spelling heuristics occur.
TEAM_PROOF_SOURCE = r'''
#include <atomic>
#include <omp.h>
#include <iostream>
#include <vector>
namespace {
struct capture { void (*body)(void*); void* data; };
std::atomic<int> entries{0}, wrong_team{0};
int expected = 0;
std::vector<int> visits;
void observed(void* data) {
    const auto* args = static_cast<capture*>(data);
    const int id = omp_get_thread_num();
    if (omp_get_num_threads() != expected || id < 0 || id >= expected) ++wrong_team;
    else ++visits[id];
    args->body(args->data);
}
}
extern "C" void __real_GOMP_parallel(void (*)(void*), void*, unsigned, unsigned);
extern "C" void __wrap_GOMP_parallel(void (*body)(void*), void* data, unsigned n, unsigned flags) {
    ++entries;
    capture args{body, data};
    __real_GOMP_parallel(observed, &args, n, flags);
}
extern "C" void fort_cpu_protocol_proof_begin(int threads) {
    expected = threads; entries = 0; wrong_team = 0; visits.assign(threads, 0);
}
extern "C" void fort_cpu_protocol_proof_end() {
    std::cout << ",\"parallel_entries\":" << entries.load()
              << ",\"wrong_team_visits\":" << wrong_team.load() << ",\"thread_visits\":[";
    for (int i = 0; i != expected; ++i) { if (i) std::cout << ','; std::cout << visits[i]; }
    std::cout << ']';
}
'''


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def cpu_protocol_identity():
    """Version the producer, reader, timed source and separate proof source."""
    source = Path(__file__)
    try:
        return sha256(source.read_bytes() + source.with_suffix(".cpp").read_bytes() +
                      TEAM_PROOF_SOURCE.encode() + _hash(worker_renderer_identities()).encode() +
                      generated_cpu_control_source(32).encode() + generated_cpu_control_source(64).encode()).hexdigest()
    except OSError as error:
        raise NumericalCalibrationError("CPU protocol source unavailable") from error


def _base(profile):
    section = profile.get("numerical")
    if not isinstance(section, dict) or section.get("schema_version") != 2:
        raise NumericalCalibrationError("CPU protocol requires numerical v2 observations")
    validate_numerical_profile(section, profile)
    memory = section.get("protocol", {}).get("memory", {})
    if memory.get("fit_sizes") != list(MEMORY_FIT_SIZES):
        raise NumericalCalibrationError("CPU protocol requires bounded working-set memory observations")
    return section


def _execution_identity(profile, identity):
    original = profile["numerical"]["identity"]
    if not isinstance(identity, dict):
        raise NumericalCalibrationError("CPU protocol execution identity missing")
    for name in ("cpu_threads", "precision_bits"):
        if type(identity.get(name)) is not int or identity[name] != profile[name]:
            raise NumericalCalibrationError("CPU protocol " + name + " mismatch")
    actual, limit = identity.get("actual_team_threads"), identity.get("thread_limit")
    if (type(actual) is not int or actual != profile["cpu_threads"] or type(limit) is not int or
            not profile["cpu_threads"] <= limit <= (1 << 31) - 1):
        raise NumericalCalibrationError("CPU protocol actual OpenMP team or thread limit mismatch")
    for name in ("omp_wait_policy", "gomp_spincount"):
        value = identity.get(name)
        if (name not in identity or value is not None and
                (not isinstance(value, str) or len(value) > OPENMP_ENVIRONMENT_LIMIT or
                 any(ord(c) < 32 or ord(c) > 126 for c in value))):
            raise NumericalCalibrationError("CPU protocol bounded OpenMP environment identity missing")
    if (identity.get("cpu_affinity") != original["cpu_affinity"] or
            any(type(cpu) is not int for cpu in identity.get("cpu_affinity", [])) or
            identity.get("omp_dynamic") is not False or identity.get("omp_proc_bind") != "false"):
        raise NumericalCalibrationError("CPU protocol placement mismatch")
    actual = identity.get("fortran")
    if not isinstance(actual, dict):
        raise NumericalCalibrationError("CPU protocol native Fortran identity missing")
    for name in ("compiler_version", "semantic_options"):
        if actual.get(name) != original["fortran"][name]:
            raise NumericalCalibrationError("CPU protocol native Fortran " + name + " mismatch")
    if (not isinstance(actual.get("compiler_options"), str) or
            normalize_fortran_options(actual["compiler_options"]) != actual["semantic_options"]):
        raise NumericalCalibrationError("CPU protocol native Fortran option normalization mismatch")
    host = identity.get("host")
    if (not isinstance(host, dict) or host.get("compiler_version") != profile["toolchain"]["host_cxx_version"] or
            host.get("semantic_options") != list(HOST_FLAGS)):
        raise NumericalCalibrationError("CPU protocol generated host backend mismatch")
    if identity.get("proof_backend") != "GNU GOMP_parallel wrap v1":
        raise NumericalCalibrationError("CPU protocol team proof backend unsupported")
    objects = identity.get("timed_objects")
    if (not isinstance(objects, dict) or set(objects) != {"native", "driver"} or
            any(not isinstance(value, str) or len(value) != 64 or
                any(c not in "0123456789abcdef" for c in value) for value in objects.values())):
        raise NumericalCalibrationError("CPU protocol timed object identities missing")


def _indexed(records, proofs, threads):
    maximum = len(MEASURED_BACKENDS) * len(STARTUP_SIZES) + len(CPU_BACKENDS) * len(MEMORY_SIZES)
    if not isinstance(records, list) or len(records) > maximum:
        raise NumericalCalibrationError("CPU protocol observations exceed bounded protocol")
    indexed, proved, memory = {}, {}, {}
    for row in records:
        if isinstance(row, dict) and row.get("kind") == "cpu_memory_cost_v1":
            backend, items = row.get("backend"), row.get("items")
            if backend not in CPU_BACKENDS or type(items) is not int or items not in MEMORY_SIZES:
                raise NumericalCalibrationError("unknown fresh CPU memory observation")
            if row.get("role") != ("fit" if items in MEMORY_FIT_SIZES else "holdout"):
                raise NumericalCalibrationError("fresh CPU memory fit/holdout role mismatch")
            _compute_samples(row)
            key = "memory_v2", backend, items
            if key in memory:
                raise NumericalCalibrationError("duplicate fresh CPU memory observation")
            memory[key] = row
    startup_rows = [row for row in records if not isinstance(row, dict) or row.get("kind") != "cpu_memory_cost_v1"]
    for rows, target, kind in ((startup_rows, indexed, "cpu_startup_cost_v1"),
                               (proofs, proved, "cpu_team_proof_v1")):
        if not isinstance(rows, list) or len(rows) > 6:
            raise NumericalCalibrationError("CPU protocol team proofs exceed bounded protocol")
        for row in rows:
            if (not isinstance(row, dict) or row.get("kind") != kind or
                    row.get("backend") not in MEASURED_BACKENDS or
                    type(row.get("items")) is not int or row["items"] not in STARTUP_SIZES):
                raise NumericalCalibrationError("unknown CPU protocol observation")
            key = row["backend"], row["items"]
            if key in target:
                raise NumericalCalibrationError("duplicate CPU protocol observation")
            if kind == "cpu_startup_cost_v1":
                _compute_samples(row)
            else:
                # A proof failure is evidence, not malformed data. It must keep
                # the backend unavailable even when all measured costs fit.
                if (type(row.get("parallel_entries")) is not int or
                        type(row.get("wrong_team_visits")) is not int or
                        not isinstance(row.get("thread_visits"), list) or
                        len(row["thread_visits"]) != threads or
                        any(type(n) is not int or n < 0 for n in row["thread_visits"])):
                    raise NumericalCalibrationError("malformed CPU team participation proof")
            target[key] = row
    return indexed, proved, memory


def _startup(backend, indexed, proofs):
    if backend == "native_serial":
        return {"status": "accepted", "fixed_seconds": 0.0, "reason_codes": [], "controls": [],
                "coverage": "per-region serial counterfactual; no unrelated ABI call overhead"}
    result = {"status": "rejected", "reason_codes": [], "controls": [], "coverage": FIXED_COVERAGE}
    rows = [indexed.get((backend, n)) for n in STARTUP_SIZES]
    proof_rows = [proofs.get((backend, n)) for n in STARTUP_SIZES]
    if any(row is None for row in rows + proof_rows):
        result["reason_codes"] = ["missing_startup_or_team_proof"]
        return result
    for row in proof_rows:
        if (row.get("agreement_passed") is not True or row["parallel_entries"] != 1 or
                row["wrong_team_visits"] != 0 or any(n != 1 for n in row["thread_visits"])):
            result["reason_codes"].append("team_protocol_or_sentinel_proof_failed")
    if any(row.get("agreement_passed") is not True for row in rows):
        result["reason_codes"].append("startup_sentinel_agreement_failed")
    fixed = statistics.median(_compute_samples(rows[0]))
    result["fixed_seconds"] = fixed
    for row in rows[1:]:
        measured = statistics.median(_compute_samples(row))
        error = abs(fixed - measured) / measured
        result["controls"].append({"items": row["items"], "measured_seconds": measured,
                                   "predicted_seconds": fixed, "relative_error": error})
        if error > MAX_HOLDOUT_RELATIVE_ERROR:
            result["reason_codes"].append("startup_dominance_control_error")
    result["reason_codes"] = sorted(set(result["reason_codes"]))
    result["status"] = "rejected" if result["reason_codes"] else "accepted"
    return result


def _raw_index(section):
    return {(row["family"], row["backend"], row["items"]): row for row in section["measurements"]
            if row.get("kind") == "compute_cost_v2"}


def _corrected_memory(indexed, backend, startup, precision, *, fresh=False):
    result = {"status": "rejected", "reason_codes": [], "holdouts": [],
              "coverage": "three distinct pointwise arrays; startup removed from byte rates"}
    if startup["status"] != "accepted":
        result["reason_codes"] = ["startup_rejected"]
        return result
    rows = [indexed.get(("memory_v2", backend, n)) for n in MEMORY_SIZES]
    if any(row is None or row.get("agreement_passed") is not True for row in rows):
        result["reason_codes"] = ["missing_or_incorrect_memory_observation"]
        return result
    fixed, width = startup["fixed_seconds"], 3 * (precision // 8)
    if fresh and any(type(row.get(name)) is not int or row[name] != width * row["items"]
                     for row in rows for name in ("traffic_bytes", "working_set_bytes")):
        raise NumericalCalibrationError("fresh CPU memory physical footprint mismatch")
    knots = []
    for row in rows:
        if row["items"] not in MEMORY_FIT_SIZES:
            continue
        duration = statistics.median(_compute_samples(row)) - fixed
        if not math.isfinite(duration) or duration <= 0:
            result["reason_codes"] = ["measured_startup_exceeds_memory_measurement"]
            return result
        traffic = width * row["items"]
        knots.append({"working_set_bytes": traffic, "seconds_per_traffic_byte": duration / traffic})
    model = {"kind": "piecewise_bandwidth_v1", "knots": knots,
             "working_set_range": [knots[0]["working_set_bytes"], knots[-1]["working_set_bytes"]],
             "fixed_cost_coverage": "independently measured startup removed"}
    result["model"] = model
    for row in rows:
        if row["items"] in MEMORY_FIT_SIZES:
            continue
        traffic = width * row["items"]
        actual = statistics.median(_compute_samples(row))
        predicted = fixed + memory_compute_seconds(model, traffic, traffic)
        error = abs(predicted - actual) / actual
        result["holdouts"].append({"items": row["items"], "measured_seconds": actual,
                                   "predicted_seconds": predicted, "relative_error": error})
        if error > MAX_HOLDOUT_RELATIVE_ERROR:
            result["reason_codes"].append("memory_size_holdout_error")
    result["reason_codes"] = sorted(set(result["reason_codes"]))
    result["status"] = "rejected" if result["reason_codes"] else "accepted"
    return result


def _coefficient_validation(section, indexed, backend, startup):
    result = {}
    for family in COMPUTE_COEFFICIENT_FAMILIES:
        if family == "memory_v2":
            continue
        old = section["families"][family][backend]
        evidence = {"status": "rejected", "reason_codes": [], "holdouts": []}
        result[family] = evidence
        if startup["status"] != "accepted" or old["status"] != "accepted":
            evidence["reason_codes"] = ["startup_or_original_coefficient_rejected"]
            continue
        slope = old["model"]["seconds_per_item"]
        evidence["seconds_per_item"] = slope
        for n in COMPUTE_HOLDOUT_SIZES:
            row = indexed[(family, backend, n)]
            actual = statistics.median(_compute_samples(row))
            predicted = startup["fixed_seconds"] + n * slope
            error = abs(predicted - actual) / actual
            evidence["holdouts"].append({"items": n, "measured_seconds": actual,
                                         "predicted_seconds": predicted, "relative_error": error})
            if error > MAX_HOLDOUT_RELATIVE_ERROR:
                evidence["reason_codes"].append("corrected_coefficient_holdout_error")
        evidence["reason_codes"] = sorted(set(evidence["reason_codes"]))
        evidence["status"] = "rejected" if evidence["reason_codes"] else "accepted"
    return result


def _class_validation(indexed, backend, startup, memory, coefficients, precision):
    result = {"memory_only_v1": {"status": memory["status"], "reason_codes": memory["reason_codes"]}}
    for name, families in COMPUTE_CLASSES.items():
        evidence = {"status": "rejected", "reason_codes": [], "holdouts": []}
        result[name] = evidence
        primitives = sorted({p for family in families for p in COMPUTE_RECIPES[family]["intrinsics"]})
        required = ("arithmetic_v2", *("primitive_" + p + "_v2" for p in primitives))
        if (startup["status"] != "accepted" or memory["status"] != "accepted" or
                any(coefficients[family]["status"] != "accepted" for family in required)):
            evidence["reason_codes"] = ["required_corrected_family_rejected"]
            continue
        arithmetic = coefficients["arithmetic_v2"]["seconds_per_item"] / COMPUTE_RECIPES["arithmetic_v2"]["arithmetic"]
        extras = {p: max(0.0, (coefficients["primitive_" + p + "_v2"]["seconds_per_item"] -
                              COMPUTE_RECIPES["primitive_" + p + "_v2"]["arithmetic"] * arithmetic) /
                             COMPUTE_RECIPES["primitive_" + p + "_v2"]["intrinsics"][p]) for p in primitives}
        for family in families:
            recipe = COMPUTE_RECIPES[family]
            slope = recipe["arithmetic"] * arithmetic + math.fsum(
                count * extras[p] for p, count in recipe["intrinsics"].items())
            for n in COMPUTE_SIZES:
                row = indexed.get((family, backend, n))
                if row is None or row.get("agreement_passed") is not True:
                    evidence["reason_codes"].append("missing_or_incorrect_expression_holdout")
                    continue
                actual = statistics.median(_compute_samples(row))
                traffic = n * recipe.get("memory_arrays", 2) * (precision // 8)
                predicted = startup["fixed_seconds"] + max(n * slope, memory_compute_seconds(memory["model"], traffic, traffic))
                error = abs(predicted - actual) / actual
                evidence["holdouts"].append({"family": family, "items": n, "measured_seconds": actual,
                                             "predicted_seconds": predicted, "relative_error": error})
                if error > MAX_HOLDOUT_RELATIVE_ERROR:
                    evidence["reason_codes"].append("corrected_expression_holdout_error")
        evidence["reason_codes"] = sorted(set(evidence["reason_codes"]))
        evidence["status"] = "rejected" if evidence["reason_codes"] else "accepted"
    return result


def profile_from_cpu_protocol_measurements(profile, records, execution_identity, proofs, *, calibration=None):
    """Derive optional costs; keep every rejected observation and old subtree."""
    if calibration is not None and not isinstance(calibration, dict):
        raise NumericalCalibrationError("CPU protocol calibration receipt must be an object")
    section = _base(profile)
    _execution_identity(profile, execution_identity)
    indexed, proved, fresh_memory = _indexed(records, proofs, profile["cpu_threads"])
    raw = _raw_index(section)
    backends = {}
    for backend in CPU_BACKENDS:
        startup = _startup(backend, indexed, proved)
        memory = _corrected_memory(fresh_memory, backend, startup, profile["precision_bits"], fresh=True)
        legacy_memory = _corrected_memory(raw, backend, startup, profile["precision_bits"])
        coefficients = _coefficient_validation(section, raw, backend, startup)
        backends[backend] = {"startup": startup, "memory": memory,
            "workload_validation": {"memory_only_v1": {"status": memory["status"], "reason_codes": memory["reason_codes"]}},
            "legacy_memory_diagnostic": legacy_memory, "legacy_compute_diagnostic": {
                "reason": "historical numerical observations lack original team and wait/spin identity",
                "coefficients": coefficients,
                "workload_validation": _class_validation(raw, backend, startup, legacy_memory, coefficients, profile["precision_bits"])}}
    result = deepcopy(profile)
    result["cpu_execution_protocol"] = {
        "schema_version": 1, "protocol_id": CPU_PROTOCOL_ID,
        "protocol_source_sha256": cpu_protocol_identity(),
        "generated_control_sha256": generated_control_identity(profile["precision_bits"]),
        "worker_renderer_identities": worker_renderer_identities(),
        "native_source_sha256": sha256(native_fixture_source(profile["precision_bits"]).encode()).hexdigest(),
        "numerical_generator_id": section["generator_id"], "numerical_identity": deepcopy(section["identity"]),
        "numerical_observations_sha256": _hash(section["measurements"]),
        "execution_identity": deepcopy(execution_identity), "measurements": deepcopy(records), "team_proofs": deepcopy(proofs),
        "backends": backends, "maximum_relative_error": MAX_HOLDOUT_RELATIVE_ERROR,
        "coverage": {"fixed": FIXED_COVERAGE, "memory_access_class": MEMORY_ACCESS_CLASS,
                     "memory": "fresh same-identity pointwise three-array working-set lattice; no extrapolation",
                     "native_serial": "startup zero by per-region convention", "existing_team": "unavailable",
                     "indirect_inspector": "unavailable", "stencil_reuse": "unvalidated"},
        "calibration": {**(calibration or {}), "application_profiled": False, "coefficients_refitted": False}}
    return result


def validate_cpu_protocol(profile):
    """Reconstruct optional evidence; saved statuses cannot grant eligibility."""
    saved = profile.get("cpu_execution_protocol")
    if not isinstance(saved, dict) or type(saved.get("schema_version")) is not int or saved["schema_version"] != 1:
        raise NumericalCalibrationError("CPU protocol evidence missing or unsupported")
    rebuilt = profile_from_cpu_protocol_measurements(profile, saved.get("measurements"),
        saved.get("execution_identity"), saved.get("team_proofs"), calibration=saved.get("calibration"))
    if saved != rebuilt["cpu_execution_protocol"]:
        raise NumericalCalibrationError("CPU protocol evidence differs from raw observations or source identity")
    return saved


def cpu_protocol_costs(profile, backend, *, access_class):
    """Return startup+memory independently from unvalidated compute classes."""
    if access_class != MEMORY_ACCESS_CLASS:
        raise NumericalCalibrationError("CPU memory access class has no independent validation")
    if backend not in CPU_BACKENDS:
        raise NumericalCalibrationError("CPU protocol backend unavailable")
    section = validate_cpu_protocol(profile)
    evidence = section["backends"][backend]
    if evidence["startup"]["status"] != "accepted" or evidence["memory"]["status"] != "accepted":
        raise NumericalCalibrationError("CPU startup or memory protocol rejected for " + backend)
    return {"backend_identity": backend, "access_class": MEMORY_ACCESS_CLASS,
            "fixed_seconds": evidence["startup"]["fixed_seconds"],
            "memory_cost_model": deepcopy(evidence["memory"]["model"]), "coverage": deepcopy(section["coverage"]),
            "protocol_source_sha256": section["protocol_source_sha256"]}


def parse_cpu_protocol_measurements(text):
    records, proofs, identities = [], [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, TypeError) as error:
            raise NumericalCalibrationError("CPU protocol emitted non-JSON output") from error
        if not isinstance(row, dict):
            raise NumericalCalibrationError("CPU protocol emitted non-object output")
        kind = row.get("kind")
        if kind == "cpu_protocol_identity_v1":
            identities.append(row)
        elif kind in {"cpu_startup_cost_v1", "cpu_memory_cost_v1"}:
            records.append(row)
        elif kind == "cpu_team_proof_v1":
            proofs.append(row)
        else:
            raise NumericalCalibrationError("unknown CPU protocol output")
    if len(identities) != 1:
        raise NumericalCalibrationError("exactly one CPU protocol identity required")
    identity = {key: value for key, value in identities[0].items() if key != "kind"}
    try:
        identity["fortran"]["semantic_options"] = normalize_fortran_options(identity["fortran"]["compiler_options"])
    except (KeyError, TypeError, ValueError) as error:
        raise NumericalCalibrationError("CPU protocol Fortran identity missing") from error
    return records, proofs, identity


def calibrate_cpu_protocol(profile, args, *, run=_run):
    """Compile once, prove the linked objects untimed, then measure once."""
    section = _base(profile)
    flags = list(args.fortran_flag or [])
    if (not flags or "-fopenmp" not in flags or
            any("fast-math" in flag or flag == "-Ofast" for flag in flags)):
        raise NumericalCalibrationError("supply original native Fortran flags including -fopenmp without fast-math")
    affinity = section["identity"]["cpu_affinity"]
    if not set(affinity) <= os.sched_getaffinity(0):
        raise NumericalCalibrationError("CPU protocol calibrated placement is unavailable")
    fortran, host = _tool(args.fortran, ("gfortran",)), _tool(args.host_cxx, ("g++",))
    target = Path(args.build_dir).resolve()
    target.mkdir(parents=True, exist_ok=False)
    source, native = Path(__file__).with_suffix(".cpp"), target / "native.f90"
    native.write_text(native_fixture_source(profile["precision_bits"]))
    (target / GENERATED_CONTROL_HEADER).write_text(generated_cpu_control_source(profile["precision_bits"]))
    proof_source = target / "team-proof.cpp"
    proof_source.write_text(TEAM_PROOF_SOURCE)
    obj, driver = target / "native.o", target / "driver.o"
    commands = [[fortran, *flags, "-c", str(native), "-o", str(obj)],
        [host, *HOST_FLAGS, "-I", str(target), "-DCALIBRATION_PRECISION=" + str(profile["precision_bits"]),
                 "-c", str(source), "-o", str(driver)]]
    for index, command in enumerate(commands):
        run(command, target, target / f"build-{index}.log", timeout=180)
    binary, proof_binary = target / "cpu-protocol", target / "cpu-protocol-proof"
    commands.extend([[host, "-fopenmp", str(obj), str(driver), "-lgfortran", "-o", str(binary)],
                     [host, *HOST_FLAGS, str(obj), str(driver), str(proof_source), "-lgfortran",
                      "-Wl,--wrap=GOMP_parallel", "-o", str(proof_binary)]])
    for index, command in enumerate(commands[2:], 2):
        run(command, target, target / f"build-{index}.log", timeout=180)
    host_version = run([host, "--version"], target, target / "host-version.txt", timeout=30).strip()
    environment = measurement_environment(os.environ)
    environment["OMP_NUM_THREADS"] = str(profile["cpu_threads"])
    prefix = ["taskset", "-c", ",".join(map(str, affinity))]
    invocation = [*prefix, str(binary), str(profile["cpu_threads"])]
    _, _, identity = parse_cpu_protocol_measurements(run([*invocation, "--identity"], target,
        target / "identity.jsonl", timeout=30, env=environment))
    identity.update(host={"compiler_version": host_version, "semantic_options": list(HOST_FLAGS)},
        timed_objects={"native": sha256(obj.read_bytes()).hexdigest(), "driver": sha256(driver.read_bytes()).hexdigest()})
    _execution_identity(profile, identity)
    proof_text = run([*prefix, str(proof_binary), str(profile["cpu_threads"]), "--proof"], target,
                     target / "team-proof.jsonl", timeout=30, env=environment)
    _, proofs, proof_identity = parse_cpu_protocol_measurements(proof_text)
    if any(proof_identity.get(key) != identity.get(key) for key in proof_identity):
        raise NumericalCalibrationError("timed and proof execution identities differ")
    text = run(invocation, target, target / "measurements.jsonl", timeout=180, env=environment)
    records, _, timed_identity = parse_cpu_protocol_measurements(text)
    if any(timed_identity.get(key) != identity.get(key) for key in timed_identity):
        raise NumericalCalibrationError("CPU protocol identity changed before timing")
    result = profile_from_cpu_protocol_measurements(profile, records, identity, proofs, calibration={
        "build_commands": commands, "run_command": invocation, "artifacts": str(target),
        "sampling": "seven >=200ms batches for startup controls and seven global rounds over all fresh CPU memory cells; no retries",
        "proof": "separate GNU wrapper executable linked against exact timed native/driver objects"})
    raw = target / "cpu-protocol-raw-samples.jsonl"
    if raw.is_file():
        result["cpu_execution_protocol"]["calibration"]["raw_samples"] = {"path": str(raw), "sha256": sha256(raw.read_bytes()).hexdigest()}
    (target / "evidence.json").write_text(json.dumps(result["cpu_execution_protocol"], indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--fortran")
    parser.add_argument("--host-cxx")
    parser.add_argument("--fortran-flag", action="append", default=[])
    args = parser.parse_args(argv)
    if args.profile.resolve() == args.output.resolve() or args.output.exists():
        parser.error("output must be new; preserve original profile")
    try:
        profile = json.loads(args.profile.read_text())
        result = calibrate_cpu_protocol(profile, args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, CalibrationError) as error:
        parser.exit(1, "CPU protocol calibration: " + str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
