"""Optional, exact-family numerical calibration with independent holdouts.

These workloads measure costs which an FMA-throughput profile cannot supply.
They deliberately do not classify arbitrary expressions as a calibrated mix:
an emitter must provide the matching recipe, backend and generator identities.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import statistics

from compiler.numerical_contract import (
    cuda_compile_options,
    numerical_build_contract,
    require_explicit_cuda_environment,
)

SCHEMA_VERSION = 1
BACKEND_ID = "standalone-cuda-openmp-cxx17-v1"
INTRINSIC_BACKEND_ID = "cxx17-scalar-real-math-v1"
PRIMITIVES = ("sqrt", "acos", "cos")
PRIMITIVE_STEPS = 16
MIX_FAMILIES = {"angle_mix_v1": {"sqrt": 8, "acos": 1, "cos": 3},
                "angle_mix_skew_v1": {"sqrt": 2, "acos": 3, "cos": 1}}
FAMILIES = ("transcendental_chain_v1", "private_matrix_v1",
            *("primitive_" + name + "_v1" for name in PRIMITIVES), *MIX_FAMILIES)
FIT_SIZES = (16384, 65536, 262144)
HOLDOUT_SIZES = (32768, 131072)
MAX_HOLDOUT_RELATIVE_ERROR = 0.25
HARDWARE_FIELDS = ("cpu_name", "gpu_uuid", "gpu_name", "compute_capability")
TOOLCHAIN_FIELDS = ("nvcc_version", "host_cxx_version", "cuda_runtime_version", "driver_version")


class NumericalCalibrationError(ValueError):
    """Missing, incompatible or unvalidated numerical measurements."""


def require_numerical_profile_contract(profile) -> None:
    """Keep historical observations readable without pricing new arithmetic."""
    from .profile import ProfileError, require_profile_numerical_contract
    try:
        require_profile_numerical_contract(profile)
    except ProfileError as error:
        raise NumericalCalibrationError(str(error)) from error


def apply_numerical_costs(analysis, profile):
    """Resolve static counts only for the independently validated scalar backend."""
    units = []
    for unit in analysis.units:
        if (unit.intrinsic_work_per_iteration and unit.arithmetic_work_per_iteration is not None
                and not unit.work_is_upper_bound and profile is not None):
            try:
                costs = numerical_intrinsic_costs(profile, unit.intrinsic_work_per_iteration,
                    backend_id=INTRINSIC_BACKEND_ID, generator_id=generator_identity())
                unit = replace(unit, work_per_iteration=unit.arithmetic_work_per_iteration,
                    cpu_numerical_seconds_per_iteration=costs["cpu_seconds_per_iteration"],
                    gpu_numerical_seconds_per_iteration=costs["gpu_seconds_per_iteration"],
                    work_estimate_reason=None)
            except NumericalCalibrationError as error:
                unit = replace(unit, work_estimate_reason=str(error))
        units.append(unit)
    return replace(analysis, units=tuple(units))


def generator_identity() -> str:
    """Identify both the fixture implementation and its fitting protocol."""
    root = Path(__file__).parent
    return sha256(b"\n".join((root / name).read_bytes() for name in
                             ("numerical_calibration.py", "numerical_calibration.cu"))
                  + numerical_build_contract()["identity"].encode()).hexdigest()


def recipe_identity(family: str, *, generator_id: str | None = None) -> str:
    if family not in FAMILIES:
        raise NumericalCalibrationError("unknown numerical workload family")
    return sha256((BACKEND_ID + ":" + (generator_id or generator_identity()) + ":" + family).encode()).hexdigest()


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise NumericalCalibrationError(name + " must be finite and positive")
    return float(value)


def _median(record):
    durations = record.get("seconds")
    if not isinstance(durations, list) or len(durations) != 5:
        raise NumericalCalibrationError("numerical observations require five duration samples")
    return statistics.median(_positive(value, "numerical duration") for value in durations)


def parse_measurements(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            raise NumericalCalibrationError("numerical benchmark emitted non-JSON output") from error
        if not isinstance(record, dict) or record.get("kind") not in {"numerical_identity", "numerical_cost"}:
            raise NumericalCalibrationError("unknown numerical measurement kind")
        if record["kind"] == "numerical_cost":
            _median(record)
        records.append(record)
    if not records:
        raise NumericalCalibrationError("numerical benchmark emitted no observations")
    return records


def _fit(records):
    """Fit only training points; neither holdout can affect coefficients."""
    points = [(record["items"], _median(record)) for record in records]
    mean_x = statistics.mean(size for size, _ in points)
    mean_y = statistics.mean(value for _, value in points)
    slope = math.fsum((size - mean_x) * (value - mean_y) for size, value in points) / math.fsum(
        (size - mean_x) ** 2 for size, _ in points)
    intercept = mean_y - slope * mean_x
    if intercept < 0:
        intercept = 0.0
        slope = math.fsum(size * value for size, value in points) / math.fsum(size * size for size, _ in points)
    _positive(slope, "numerical seconds per item")
    return {"fixed_seconds": intercept, "seconds_per_item": slope}


def profile_from_measurements(profile: dict, records: list[dict], *, calibration: dict) -> dict:
    """Attach validated optional costs without changing any existing rates."""
    identities = [record for record in records if record.get("kind") == "numerical_identity"]
    if len(identities) != 1:
        raise NumericalCalibrationError("exactly one numerical identity is required")
    identity = identities[0]
    for name in ("gpu_uuid", "gpu_name", "compute_capability"):
        if identity.get(name) != profile["hardware"][name]:
            raise NumericalCalibrationError("numerical hardware " + name + " mismatch")
    for name in ("cuda_runtime_version", "driver_version"):
        if identity.get(name) != profile["toolchain"][name]:
            raise NumericalCalibrationError("numerical toolchain " + name + " mismatch")
    for name in ("precision_bits", "cpu_threads"):
        if type(identity.get(name)) is not int or identity[name] != profile[name]:
            raise NumericalCalibrationError("numerical " + name + " mismatch")
    if identity.get("backend_id") != BACKEND_ID:
        raise NumericalCalibrationError("numerical backend mismatch")
    indexed = {}
    for record in records:
        if record.get("kind") == "numerical_identity":
            continue
        family, device, items = (record.get(name) for name in ("family", "device", "items"))
        if (record.get("kind") != "numerical_cost" or family not in FAMILIES or device not in {"cpu", "gpu"}
                or type(items) is not int or items not in FIT_SIZES + HOLDOUT_SIZES):
            raise NumericalCalibrationError("unknown numerical workload observation")
        expected_role = "fit" if items in FIT_SIZES else "holdout"
        if record.get("role") != expected_role:
            raise NumericalCalibrationError("numerical fit/holdout role mismatch")
        key = (family, device, items)
        if key in indexed:
            raise NumericalCalibrationError("duplicate numerical workload observation")
        _median(record)
        if record.get("agreement_passed") is not True:
            raise NumericalCalibrationError("numerical workload CPU/GPU agreement failed")
        indexed[key] = record
    expected = {(family, device, items) for family in FAMILIES for device in ("cpu", "gpu")
                for items in FIT_SIZES + HOLDOUT_SIZES}
    if set(indexed) != expected:
        raise NumericalCalibrationError("missing numerical workload observations")
    generator_id = generator_identity()
    families = {}
    for family in FAMILIES:
        fitted = {}
        for device in ("cpu", "gpu"):
            model = _fit([indexed[family, device, size] for size in FIT_SIZES])
            validations = []
            for size in HOLDOUT_SIZES:
                actual = _median(indexed[family, device, size])
                predicted = model["fixed_seconds"] + size * model["seconds_per_item"]
                relative_error = abs(predicted - actual) / actual
                if relative_error > MAX_HOLDOUT_RELATIVE_ERROR:
                    raise NumericalCalibrationError(f"{family} {device} independent holdout exceeds 25% error")
                validations.append({"items": size, "predicted_seconds": predicted,
                                    "measured_seconds": actual, "relative_error": relative_error})
            fitted[device] = {**model, "holdouts": validations}
        families[family] = {"recipe_id": recipe_identity(family, generator_id=generator_id),
                            "fit_sizes": list(FIT_SIZES), "holdout_sizes": list(HOLDOUT_SIZES),
                            "item_range": [min(FIT_SIZES), max(FIT_SIZES)], **fitted}
    result = deepcopy(profile)
    result["numerical"] = {"schema_version": SCHEMA_VERSION, "backend_id": BACKEND_ID,
        "generator_id": generator_id, "precision_bits": profile["precision_bits"], "cpu_threads": profile["cpu_threads"],
        "hardware": {name: profile["hardware"][name] for name in HARDWARE_FIELDS},
        "toolchain": {name: profile["toolchain"][name] for name in TOOLCHAIN_FIELDS}, "families": families,
        "holdout_max_relative_error": MAX_HOLDOUT_RELATIVE_ERROR,
        "applicability": "exact recipe, backend and generator identity only; arbitrary intrinsic mixes remain unestimated",
        "measurements": deepcopy(records), "calibration": {**calibration, "application_profiled": False}}
    result["numerical"]["intrinsic_model"] = _intrinsic_model(families, generator_id)
    validate_numerical_profile(result["numerical"], result)
    return result


def _intrinsic_model(families: dict, generator_id: str) -> dict:
    """Validate additive primitive costs against two unfit expression mixes.

    Each measured primitive includes its surrounding bounded arithmetic; this
    deliberately charges extra arithmetic rather than subtracting an unrelated
    FMA benchmark. Mixed holdouts never adjust any primitive coefficient.
    """
    rates = {device: {name: families["primitive_" + name + "_v1"][device]["seconds_per_item"] / PRIMITIVE_STEPS
                      for name in PRIMITIVES} for device in ("cpu", "gpu")}
    validations, reasons = [], []
    for family, counts in MIX_FAMILIES.items():
        for device in ("cpu", "gpu"):
            variable = PRIMITIVE_STEPS * math.fsum(counts[name] * rates[device][name] for name in PRIMITIVES)
            # All operations share one launch/team. The largest primitive
            # fixed cost is a conservative estimate of that single startup.
            fixed = max(families["primitive_" + name + "_v1"][device]["fixed_seconds"] for name in PRIMITIVES)
            for observed in families[family][device]["holdouts"]:
                predicted = fixed + observed["items"] * variable
                actual = observed["measured_seconds"]
                error = abs(predicted - actual) / actual
                validations.append({"family": family, "device": device, "items": observed["items"],
                                    "predicted_seconds": predicted, "measured_seconds": actual,
                                    "relative_error": error})
                if error > MAX_HOLDOUT_RELATIVE_ERROR:
                    reasons.append(f"{family} {device} mixed holdout at {observed['items']} exceeds 25% error")
    return {"backend_id": INTRINSIC_BACKEND_ID, "generator_id": generator_id,
            "available": not reasons, "reasons": reasons, "seconds_per_operation": rates,
            "validated_intrinsics": list(PRIMITIVES), "mixed_holdouts": validations,
            "arithmetic_policy": "primitive coefficients include bounded fixture arithmetic; retain ordinary work separately"}


def _validate_numerical_profile_v1(section: dict, profile: dict) -> None:
    """Validate saved optional evidence without requiring a current generator."""
    if not isinstance(section, dict) or type(section.get("schema_version")) is not int or section["schema_version"] != SCHEMA_VERSION:
        raise NumericalCalibrationError("numerical costs require schema_version 1")
    if section.get("backend_id") != BACKEND_ID:
        raise NumericalCalibrationError("numerical backend mismatch")
    generator = section.get("generator_id")
    if not isinstance(generator, str) or len(generator) != 64 or any(c not in "0123456789abcdef" for c in generator):
        raise NumericalCalibrationError("numerical generator_id must be a lowercase SHA-256")
    for name in ("precision_bits", "cpu_threads", "hardware", "toolchain"):
        expected = ({field: profile[name][field] for field in (HARDWARE_FIELDS if name == "hardware" else TOOLCHAIN_FIELDS)}
                    if name in {"hardware", "toolchain"} else profile[name])
        if section.get(name) != expected:
            raise NumericalCalibrationError("numerical " + name + " mismatch")
    if section.get("holdout_max_relative_error") != MAX_HOLDOUT_RELATIVE_ERROR:
        raise NumericalCalibrationError("numerical holdout threshold mismatch")
    families = section.get("families")
    if not isinstance(families, dict) or set(families) != set(FAMILIES):
        raise NumericalCalibrationError("missing numerical workload families")
    for family, costs in families.items():
        if not isinstance(costs, dict) or costs.get("recipe_id") != recipe_identity(family, generator_id=generator):
            raise NumericalCalibrationError("numerical recipe identity mismatch")
        if (costs.get("fit_sizes") != list(FIT_SIZES) or costs.get("holdout_sizes") != list(HOLDOUT_SIZES)
                or costs.get("item_range") != [min(FIT_SIZES), max(FIT_SIZES)]):
            raise NumericalCalibrationError("numerical workload range mismatch")
        for device in ("cpu", "gpu"):
            model = costs.get(device)
            if not isinstance(model, dict):
                raise NumericalCalibrationError("missing numerical device costs")
            _positive(model.get("seconds_per_item"), "numerical seconds per item")
            fixed = model.get("fixed_seconds")
            if (isinstance(fixed, bool) or not isinstance(fixed, (int, float)) or not math.isfinite(fixed) or fixed < 0):
                raise NumericalCalibrationError("numerical fixed seconds must be finite and nonnegative")
            holdouts = model.get("holdouts")
            if not isinstance(holdouts, list) or [row.get("items") for row in holdouts if isinstance(row, dict)] != list(HOLDOUT_SIZES):
                raise NumericalCalibrationError("missing numerical independent holdouts")
            for row in holdouts:
                actual = _positive(row.get("measured_seconds"), "holdout duration")
                predicted = _positive(row.get("predicted_seconds"), "holdout prediction")
                expected = fixed + row["items"] * model["seconds_per_item"]
                error = abs(predicted - actual) / actual
                reported_error = row.get("relative_error")
                if (isinstance(reported_error, bool) or not isinstance(reported_error, (int, float))
                        or not math.isfinite(reported_error) or reported_error < 0):
                    raise NumericalCalibrationError("numerical holdout error must be finite and nonnegative")
                if not math.isclose(predicted, expected, rel_tol=1e-12) or not math.isclose(reported_error, error, rel_tol=1e-12, abs_tol=1e-15) or error > MAX_HOLDOUT_RELATIVE_ERROR:
                    raise NumericalCalibrationError("numerical independent holdout is inconsistent or failed")
    if section.get("intrinsic_model") != _intrinsic_model(families, generator):
        raise NumericalCalibrationError("numerical intrinsic model differs from its independent mixed holdouts")


def numerical_costs(profile: dict, family: str, *, backend_id: str, generator_id: str, recipe_id: str) -> dict:
    """Expose a cost only to the exact implementation whose work was measured."""
    require_numerical_profile_contract(profile)
    section = profile.get("numerical")
    if section is None:
        raise NumericalCalibrationError("no numerical calibration")
    if section.get("schema_version") != 1:
        raise NumericalCalibrationError("legacy numerical reader requires schema_version 1")
    validate_numerical_profile(section, profile)
    if (backend_id != section["backend_id"] or generator_id != section["generator_id"]
            or family not in section["families"] or recipe_id != section["families"][family]["recipe_id"]):
        raise NumericalCalibrationError("numerical implementation identity mismatch; estimate unavailable")
    return deepcopy(section["families"][family])


def numerical_intrinsic_costs(profile: dict, counts, *, backend_id: str, generator_id: str) -> dict:
    """Return additional seconds per item for an independently validated class.

    Unknown intrinsics and retained-loop/conditional work must remain unknown;
    this API accepts only statically proved counts of the named scalar math
    primitives. A C++/CUDA scalar-math backend must explicitly identify itself.
    """
    require_numerical_profile_contract(profile)
    section = profile.get("numerical")
    if section is None:
        raise NumericalCalibrationError("no numerical calibration; intrinsic estimate unavailable")
    if section.get("schema_version") != 1:
        raise NumericalCalibrationError("legacy intrinsic reader requires schema_version 1")
    validate_numerical_profile(section, profile)
    model = section["intrinsic_model"]
    if backend_id != model["backend_id"] or generator_id != model["generator_id"]:
        raise NumericalCalibrationError("numerical intrinsic implementation identity mismatch")
    if model["available"] is not True:
        raise NumericalCalibrationError("numerical mixed holdout validation failed; intrinsic estimate unavailable")
    pairs = list(counts.items()) if isinstance(counts, dict) else list(counts)
    if (not pairs or len({name for name, _ in pairs}) != len(pairs)
            or any(name not in PRIMITIVES or type(count) is not int or count <= 0 for name, count in pairs)):
        raise NumericalCalibrationError("unsupported or unknown intrinsic counts; estimate unavailable")
    return {device + "_seconds_per_iteration": math.fsum(count * model["seconds_per_operation"][device][name]
                                                        for name, count in pairs) for device in ("cpu", "gpu")}


def calibrate_numerical(profile, args, directory: Path, nvcc: str, host: str, *, run) -> dict:
    require_explicit_cuda_environment()
    require_numerical_profile_contract(profile)
    target = directory / "numerical"
    target.mkdir(exist_ok=True)
    source = Path(__file__).with_suffix(".cu")
    binary = target / "numerical-calibration"
    command = [nvcc, "-O3", *cuda_compile_options(), "-std=c++17", "-arch=" + args.arch, "-ccbin", host,
               "-Xcompiler=-fopenmp", "-DCALIBRATION_PRECISION=" + str(args.precision), str(source), "-o", str(binary)]
    run(command, target, target / "build.log", timeout=180)
    environment = {name: value for name, value in os.environ.items()
                   if name not in {"FORT_RUNTIME_TRACE", "FORT_PHASE_TIMING", "CUDA_LAUNCH_BLOCKING"}
                   and not name.startswith("FORT_SCOPE_TEST_")}
    invocation = [str(binary), str(args.threads), str(args.device)]
    text = run(invocation, target, target / "measurements.jsonl", timeout=180, env=environment)
    return profile_from_measurements(profile, parse_measurements(text), calibration={
        "build_command": command, "run_command": invocation, "artifacts": str(target),
        "benchmark_source_sha256": sha256(source.read_bytes()).hexdigest(),
        "cpu_method": "wall time, fixed-budget OpenMP parallel for, five warmed measurements",
        "gpu_method": "CUDA events, resident input/output, five warmed measurements; transfer costs separate",
        "fit_method": "nonnegative linear fit on three fixed sizes; two independent sizes validate within 25%",
        "semantic_flags": "-O3 without fast-math; identical bounded expressions on CPU and GPU"})


# V2 is additive. Legacy observations/readers above retain their original
# meaning and cannot establish a native Fortran counterfactual.
COMPUTE_BACKEND_ID = "fortran-cxx17-cuda-roofline-v2"
COMPUTE_PROTOCOL_ID = "interleaved-seven-200ms-v2"
COMPUTE_BACKENDS = ("native_serial", "native_fork_join", "generated_cpu", "gpu")
COMPUTE_FIT_SIZES = (65536, 262144, 1048576)
COMPUTE_HOLDOUT_SIZES = (131072, 524288)
COMPUTE_SIZES = tuple(sorted(COMPUTE_FIT_SIZES + COMPUTE_HOLDOUT_SIZES))
COMPUTE_SAMPLES = 7
COMPUTE_MIN_BATCH_SECONDS = 0.2
COMPUTE_PRIMITIVES = (*PRIMITIVES, "divide")
MEMORY_FIT_SIZES = tuple(65536 * 3**k // 2**k for k in range(8))
MEMORY_HOLDOUT_SIZES = tuple(sorted({131072, 524288, *(round(math.sqrt(a * b))
    for a, b in zip(MEMORY_FIT_SIZES, MEMORY_FIT_SIZES[1:]))}))
MEMORY_SIZES = tuple(sorted(MEMORY_FIT_SIZES + MEMORY_HOLDOUT_SIZES))
MEMORY_MODEL_KIND = "piecewise_bandwidth_v1"
# Counts describe the fixture's ordinary scalar operations, independently of
# the application. Mixed families are validation-only, never coefficient fits.
COMPUTE_RECIPES = {
    "arithmetic_v2": {"arithmetic": 518, "intrinsics": {}},
    "memory_v2": {"arithmetic": 2, "intrinsics": {}, "memory_arrays": 3},
    "primitive_sqrt_v2": {"arithmetic": 80, "intrinsics": {"sqrt": 16}},
    "primitive_acos_v2": {"arithmetic": 48, "intrinsics": {"acos": 16}},
    "primitive_cos_v2": {"arithmetic": 32, "intrinsics": {"cos": 16}},
    "scalar_mix_v2": {"arithmetic": 736 + 112, "intrinsics": {"sqrt": 96, "acos": 32, "cos": 80, "divide": 16}, "memory_arrays": 3},
    "scalar_mix_skew_v2": {"arithmetic": 384 + 112, "intrinsics": {"sqrt": 32, "acos": 64, "cos": 16, "divide": 16}, "memory_arrays": 3},
    "private_mix_v2": {"arithmetic": 736 + 112 + 199, "intrinsics": {"sqrt": 96, "acos": 32, "cos": 80, "divide": 16}, "memory_arrays": 3},
    "private_mix_skew_v2": {"arithmetic": 384 + 112 + 37, "intrinsics": {"sqrt": 32, "acos": 64, "cos": 16, "divide": 16}, "memory_arrays": 3},
    "primitive_divide_v2": {"arithmetic": 326, "intrinsics": {"divide": 64}, "memory_arrays": 3},
    "ordinary_mix_v2": {"arithmetic": 576, "intrinsics": {"divide": 64}, "memory_arrays": 3},
}
COMPUTE_FAMILIES = tuple(COMPUTE_RECIPES)
COMPUTE_COEFFICIENT_FAMILIES = (*COMPUTE_FAMILIES[:5], "primitive_divide_v2")
COMPUTE_CLASSES = {
    "ordinary_expression_v2": ("ordinary_mix_v2",),
    "scalar_expression_v2": ("scalar_mix_v2", "scalar_mix_skew_v2"),
    "fixed_private_array_v2": ("private_mix_v2", "private_mix_skew_v2"),
}
COMPUTE_PRIVATE_LIMITS = {"private_array_elements": 64, "max_private_array_elements": 16,
                          "max_private_array_rank": 2, "private_array_groups": 4}


def compute_generator_identity() -> str:
    root = Path(__file__).parent
    return sha256(b"\n".join((root / name).read_bytes() for name in
        ("numerical_calibration.py", "numerical_calibration.cu", "numerical_calibration.f90"))
        + numerical_build_contract()["identity"].encode()).hexdigest()


def native_fixture_source(precision: int) -> str:
    """Specialize the maintained fixture before its original native DOs.

    Dispatch once per invocation, matching the generated C++/CUDA templates.
    The actual Fortran compiler retains its original semantic flags; no forced
    inlining or fast-math option is needed to remove a per-cell family switch.
    """
    if precision not in {32, 64}:
        raise NumericalCalibrationError("unsupported native fixture precision")
    source = Path(__file__).with_suffix(".f90").read_text()
    primitive_start = source.index("pure function primitive(")
    work_start = source.index("pure function numerical_work(")
    worker_start = source.index("subroutine fort_numerical_native_v2(")
    identity_start = source.index("subroutine fort_numerical_fortran_identity_v2(")
    primitive = source[primitive_start:work_start]
    work = source[work_start:worker_start]
    pdecl, pbody = primitive.split("select case(operation)\n", 1)
    pbody = pbody.split("end select\n", 1)[0]
    pmatches = list(re.finditer(r"(?m)^case\((\d)\)\n|^case default\n", pbody))
    # Fixed template assertions prevent silently benchmarking different work.
    if [match.group(1) for match in pmatches] != ["0", "1", None]:
        raise NumericalCalibrationError("native primitive template changed")
    primitives = []
    for operation, match in enumerate(pmatches):
        body = pbody[match.end():pmatches[operation + 1].start() if operation + 1 < len(pmatches) else None]
        decl = pdecl.replace("primitive(operation,x)", f"primitive_{operation}(x)")
        decl = decl.replace("integer,intent(in)::operation\n", "")
        primitives.append(decl + body + "end function\n\n")
    wdecl, wbody = work.split("select case(family)\n", 1)
    wbody = wbody.split("end select\n", 1)[0]
    matches = list(re.finditer(r"(?m)^case\(([^\n]+)\)\n|^case default\n", wbody))
    branches = {}
    for index, match in enumerate(matches):
        branches[match.group(1) or "default"] = wbody[match.end():matches[index + 1].start() if index + 1 < len(matches) else None]
    if set(branches) != {"0", "1", "2:4", "9", "10", "default"}:
        raise NumericalCalibrationError("native numerical template changed")
    functions, workers, dispatch = [], [], []
    for family in range(len(COMPUTE_FAMILIES)):
        branch = "2:4" if 2 <= family <= 4 else str(family) if str(family) in branches else "default"
        decl = wdecl.replace("numerical_work(family,input,other)", f"numerical_work_{family}(input,other)")
        decl = decl.replace("integer,intent(in)::family", f"integer,parameter::family={family}")
        body = branches[branch].replace("primitive(family-2,x)", f"primitive_{family-2}(x)")
        for operation in range(3):
            body = body.replace(f"primitive({operation},x)", f"primitive_{operation}(x)")
        functions.append(decl + body + "end function\n\n")
        workers.append(f"""subroutine numerical_worker_{family}(n,threads,fork_join,a,b,output)
integer(c_int),intent(in)::threads,fork_join
integer(c_size_t),intent(in)::n
real(rk),intent(in)::a(*),b(*)
real(rk),intent(out)::output(*)
integer(c_size_t)::i
if(fork_join==0) then
  do i=1,n
    output(i)=numerical_work_{family}(a(i),b(i))
  enddo
else
  !$omp parallel do num_threads(threads) schedule(static) default(none) &
  !$omp shared(n,a,b,output) private(i)
  do i=1,n
    output(i)=numerical_work_{family}(a(i),b(i))
  enddo
  !$omp end parallel do
endif
end subroutine

""")
        dispatch.append(f"case({family})\n  call numerical_worker_{family}(n,threads,fork_join,a,b,output)\n")
    entry = """subroutine fort_numerical_native_v2(family,n,threads,fork_join,a,b,output) bind(c)
integer(c_int),value::family,threads,fork_join
integer(c_size_t),value::n
real(rk),intent(in)::a(*),b(*)
real(rk),intent(out)::output(*)
select case(family)
""" + "".join(dispatch) + "end select\nend subroutine\n\n"
    result = source[:primitive_start] + "".join(primitives + functions + workers) + entry + source[identity_start:]
    return result.replace("integer, parameter :: rk = c_double", "integer, parameter :: rk = " + ("c_double" if precision == 64 else "c_float"))


def _compute_recipe_identity(family, generator):
    return sha256((COMPUTE_BACKEND_ID + ":" + generator + ":" + family).encode()).hexdigest()


def _compute_samples(record):
    samples = record.get("samples")
    if not isinstance(samples, list) or len(samples) != COMPUTE_SAMPLES:
        raise NumericalCalibrationError("compute observations require seven batch samples")
    result = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("batch") != index:
            raise NumericalCalibrationError("compute batch order mismatch")
        repetitions = sample.get("repetitions")
        if type(repetitions) is not int or repetitions <= 0:
            raise NumericalCalibrationError("compute repetitions must be positive integers")
        duration = _positive(sample.get("elapsed_seconds"), "compute batch duration")
        wall = _positive(sample.get("wall_seconds"), "compute wall duration")
        if wall < COMPUTE_MIN_BATCH_SECONDS:
            raise NumericalCalibrationError("compute batches must last at least 200 ms")
        result.append(duration / repetitions)
    return result


def parse_compute_measurements(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError as error:
            raise NumericalCalibrationError("compute benchmark emitted non-JSON output") from error
        if not isinstance(row, dict) or row.get("kind") not in {"compute_identity_v2", "compute_cost_v2"}:
            raise NumericalCalibrationError("unknown compute observation kind")
        if row["kind"] == "compute_cost_v2":
            _compute_samples(row)
        records.append(row)
    if not records:
        raise NumericalCalibrationError("compute benchmark emitted no observations")
    return records


def _compute_fit(rows):
    # Share the proven nonnegative linear fit without changing the v1 protocol.
    return _fit([{"items": row["items"], "seconds": [statistics.median(_compute_samples(row))] * 5}
                 for row in rows])


def _compute_family(family, backend, indexed, generator):
    record = {"status": "rejected", "recipe_id": _compute_recipe_identity(family, generator),
              "item_range": [min(COMPUTE_SIZES), max(COMPUTE_SIZES)],
              "reason_codes": [], "holdouts": [], "observation_ids": []}
    rows = [indexed.get((family, backend, size)) for size in COMPUTE_SIZES]
    record["observation_ids"] = [row["observation_id"] for row in rows if row is not None]
    if any(row is None for row in rows):
        record["reason_codes"].append("missing_observations")
        return record
    if any(row.get("agreement_passed") is not True for row in rows):
        record["reason_codes"].append("numerical_agreement_failed")
        return record
    try:
        model = _compute_fit([row for row in rows if row["role"] == "fit"])
    except NumericalCalibrationError:
        record["reason_codes"].append("nonpositive_fitted_rate")
        return record
    record["model"] = model
    for row in rows:
        if row["role"] != "holdout":
            continue
        actual = statistics.median(_compute_samples(row))
        predicted = model["fixed_seconds"] + row["items"] * model["seconds_per_item"]
        error = abs(predicted - actual) / actual
        record["holdouts"].append({"items": row["items"], "measured_seconds": actual,
                                   "predicted_seconds": predicted, "relative_error": error})
        if error > MAX_HOLDOUT_RELATIVE_ERROR:
            record["reason_codes"].append("size_holdout_error")
    record["reason_codes"] = sorted(set(record["reason_codes"]))
    record["status"] = "rejected" if record["reason_codes"] else "accepted"
    return record


def memory_compute_seconds(model, traffic_bytes, working_set_bytes):
    """The exact bounded memory formula shared by validation and readers.

    Working set is the physical union of accessed resource sections. Traffic
    includes repeated accesses; the two quantities must not be conflated.
    """
    try:
        traffic = float(traffic_bytes)
    except (OverflowError, ValueError, TypeError) as error:
        raise NumericalCalibrationError("memory traffic is unrepresentable") from error
    if (isinstance(traffic_bytes, bool) or not isinstance(traffic_bytes, (int, float)) or
            not math.isfinite(traffic) or traffic < 0 or type(working_set_bytes) is not int or
            not 0 <= working_set_bytes <= (1 << 64) - 1):
        raise NumericalCalibrationError("memory traffic/working set must be finite and nonnegative")
    if not isinstance(model, dict):
        raise NumericalCalibrationError("unsupported memory cost model")
    if model.get("kind") == "constant_bandwidth_v1":
        try:
            rate = _positive(model.get("seconds_per_traffic_byte"), "memory coefficient")
        except OverflowError as error:
            raise NumericalCalibrationError("memory coefficient is unrepresentable") from error
    elif model.get("kind") != MEMORY_MODEL_KIND:
        raise NumericalCalibrationError("unsupported memory cost model")
    else:
        knots = model.get("knots")
        if (not isinstance(knots, (list, tuple)) or not 2 <= len(knots) <= 8 or
                any(not isinstance(row, dict) or type(row.get("working_set_bytes")) is not int or
                    not 0 < row["working_set_bytes"] <= (1 << 64) - 1 for row in knots)):
            raise NumericalCalibrationError("memory model requires two to eight bounded working-set knots")
        coordinates = [row["working_set_bytes"] for row in knots]
        bounds = model.get("working_set_range")
        if (coordinates != sorted(set(coordinates)) or not isinstance(bounds, (list, tuple)) or
                len(bounds) != 2 or any(type(value) is not int for value in bounds) or
                list(bounds) != [coordinates[0], coordinates[-1]]):
            raise NumericalCalibrationError("memory working-set range mismatch")
        try:
            rates = [_positive(row.get("seconds_per_traffic_byte"), "memory coefficient") for row in knots]
        except OverflowError as error:
            raise NumericalCalibrationError("memory coefficient is unrepresentable") from error
        if not coordinates[0] <= working_set_bytes <= coordinates[-1]:
            raise NumericalCalibrationError("memory working set outside calibrated range")
        rate = None
        for index, (lower, upper) in enumerate(zip(coordinates, coordinates[1:])):
            if working_set_bytes <= upper:
                if working_set_bytes == lower:
                    rate = rates[index]
                elif working_set_bytes == upper:
                    rate = rates[index + 1]
                else:
                    fraction = float(working_set_bytes - lower) / float(upper - lower)
                    rate = rates[index] + fraction * (rates[index + 1] - rates[index])
                break
        if rate is None or not math.isfinite(rate) or rate <= 0:
            raise NumericalCalibrationError("memory interpolation failed")
    result = traffic * rate
    if not math.isfinite(result) or (traffic > 0 and result <= 0):
        raise NumericalCalibrationError("memory cost overflow or underflow")
    return result


def _compute_memory_family(backend, indexed, generator, precision, arithmetic):
    record = {"status": "rejected", "recipe_id": _compute_recipe_identity("memory_v2", generator),
              "item_range": [min(MEMORY_SIZES), max(MEMORY_SIZES)],
              "reason_codes": [], "holdouts": [], "observation_ids": []}
    rows = [indexed.get(("memory_v2", backend, size)) for size in MEMORY_SIZES]
    record["observation_ids"] = [row["observation_id"] for row in rows if row is not None]
    if any(row is None for row in rows):
        record["reason_codes"] = ["missing_observations"]
        return record
    if any(row.get("agreement_passed") is not True for row in rows):
        record["reason_codes"] = ["numerical_agreement_failed"]
        return record
    if arithmetic["status"] != "accepted":
        record["reason_codes"] = ["required_arithmetic_family_rejected"]
        return record
    fixed = arithmetic["model"]["fixed_seconds"] if backend in {"native_fork_join", "generated_cpu"} else 0.0
    width = 3 * (precision // 8)
    knots = []
    for row in rows:
        if row["role"] == "fit":
            duration = statistics.median(_compute_samples(row)) - fixed
            if duration <= 0:
                record["reason_codes"] = ["fixed_execution_exceeds_memory_measurement"]
                return record
            knots.append({"working_set_bytes": width * row["items"],
                          "seconds_per_traffic_byte": duration / (width * row["items"])})
    model = {"kind": MEMORY_MODEL_KIND, "working_set_range": [knots[0]["working_set_bytes"], knots[-1]["working_set_bytes"]],
             "knots": knots, "fixed_cost_coverage": "shared_cpu_execution_fixed_removed" if fixed else "none"}
    record["model"] = model
    for row in rows:
        if row["role"] != "holdout":
            continue
        traffic = row["items"] * width
        actual = statistics.median(_compute_samples(row))
        predicted = fixed + memory_compute_seconds(model, traffic, traffic)
        error = abs(predicted - actual) / actual
        record["holdouts"].append({"items": row["items"], "working_set_bytes": traffic,
            "measured_seconds": actual, "predicted_seconds": predicted, "relative_error": error})
        if error > MAX_HOLDOUT_RELATIVE_ERROR:
            record["reason_codes"].append("size_holdout_error")
    record["reason_codes"] = sorted(set(record["reason_codes"]))
    record["status"] = "rejected" if record["reason_codes"] else "accepted"
    return record


def _compute_components(families, backend, precision, primitives=COMPUTE_PRIMITIVES):
    required = ("arithmetic_v2", "memory_v2", *("primitive_" + p + "_v2" for p in primitives))
    if any(families[name][backend]["status"] != "accepted" for name in required):
        return None
    arithmetic = families["arithmetic_v2"][backend]["model"]["seconds_per_item"] / 518
    memory_model = families["memory_v2"][backend]["model"]
    if memory_model.get("kind") == MEMORY_MODEL_KIND:
        memory, cost_model = None, deepcopy(memory_model)
    else:
        memory = memory_model["seconds_per_item"] / (3 * (precision // 8))
        cost_model = {"kind": "constant_bandwidth_v1", "seconds_per_traffic_byte": memory}
    intrinsic = {}
    for primitive in primitives:
        name = "primitive_" + primitive + "_v2"
        # Ordinary scalar work is explicitly removed from the measured recipe;
        # independent expression and private-array holdouts test this model.
        intrinsic[primitive] = max(0.0, (families[name][backend]["model"]["seconds_per_item"] -
            COMPUTE_RECIPES[name]["arithmetic"] * arithmetic) / COMPUTE_RECIPES[name]["intrinsics"][primitive])
    return {"arithmetic_seconds_per_operation": arithmetic, "memory_seconds_per_byte": memory,
            "memory_cost_model": cost_model,
            "intrinsic_seconds_per_operation": intrinsic}


def _compute_fixed(families, backend):
    # Fork/join belongs to CPU numerical execution. GPU enqueue/coordination
    # is separately calibrated; its resident event intercept is diagnostic.
    sources = ("arithmetic_v2",) if families["memory_v2"][backend].get("model", {}).get("kind") == MEMORY_MODEL_KIND else ("arithmetic_v2", "memory_v2")
    return (max(families[name][backend]["model"]["fixed_seconds"] for name in sources)
            if backend in {"native_fork_join", "generated_cpu"} else 0.0)


def _compute_class(families, indexed, backend, workload_class, precision):
    record = {"status": "rejected", "reason_codes": [], "holdouts": []}
    primitives = tuple(name for name in COMPUTE_PRIMITIVES if any(
        name in COMPUTE_RECIPES[family]["intrinsics"] for family in COMPUTE_CLASSES[workload_class]))
    components = _compute_components(families, backend, precision, primitives)
    if components is None:
        record["reason_codes"] = ["required_family_rejected"]
        return record
    for family in COMPUTE_CLASSES[workload_class]:
        recipe = COMPUTE_RECIPES[family]
        # A single worker startup is charged once. GPU enqueue/team management
        # is external; its fitted resident-kernel intercept is not transferable.
        fixed = _compute_fixed(families, backend)
        compute_slope = recipe["arithmetic"] * components["arithmetic_seconds_per_operation"] +\
                    math.fsum(count * components["intrinsic_seconds_per_operation"][name]
                              for name, count in recipe["intrinsics"].items())
        # Mixed recipes never enter a coefficient fit. Every predefined size is
        # an independent expression holdout, including training-size coordinates.
        for size in COMPUTE_SIZES:
            row = indexed.get((family, backend, size))
            if row is None or row.get("agreement_passed") is not True:
                record["reason_codes"].append("missing_or_incorrect_expression_holdout")
                continue
            actual = statistics.median(_compute_samples(row))
            traffic = size * recipe.get("memory_arrays", 2) * (precision // 8)
            if components["memory_cost_model"]["kind"] == "constant_bandwidth_v1":
                predicted = fixed + size * max(compute_slope, recipe.get("memory_arrays", 2) *
                                               (precision // 8) * components["memory_seconds_per_byte"])
            else:
                predicted = fixed + max(size * compute_slope,
                                       memory_compute_seconds(components["memory_cost_model"], traffic, traffic))
            error = abs(predicted - actual) / actual
            record["holdouts"].append({"family": family, "items": size, "measured_seconds": actual,
                                       "predicted_seconds": predicted, "relative_error": error})
            if error > MAX_HOLDOUT_RELATIVE_ERROR:
                record["reason_codes"].append("expression_holdout_error")
    record["reason_codes"] = sorted(set(record["reason_codes"]))
    record["status"] = "rejected" if record["reason_codes"] else "accepted"
    return record


def profile_from_compute_measurements(profile, records, *, calibration, memory_model_kind=MEMORY_MODEL_KIND):
    identities = [row for row in records if row.get("kind") == "compute_identity_v2"]
    if len(identities) != 1:
        raise NumericalCalibrationError("exactly one compute identity is required")
    identity = identities[0]
    for name in ("gpu_uuid", "gpu_name", "compute_capability"):
        if identity.get(name) != profile["hardware"][name]:
            raise NumericalCalibrationError("compute hardware " + name + " mismatch")
    for name in ("cuda_runtime_version", "driver_version"):
        if identity.get(name) != profile["toolchain"][name]:
            raise NumericalCalibrationError("compute toolchain " + name + " mismatch")
    for name in ("cpu_threads", "precision_bits"):
        if type(identity.get(name)) is not int or identity[name] != profile[name]:
            raise NumericalCalibrationError("compute " + name + " mismatch")
    if identity.get("backend_id") != COMPUTE_BACKEND_ID:
        raise NumericalCalibrationError("compute backend mismatch")
    affinity = identity.get("cpu_affinity")
    if (not isinstance(affinity, list) or len(affinity) != profile["cpu_threads"] or
            any(type(cpu) is not int or cpu < 0 for cpu in affinity) or affinity != sorted(set(affinity))):
        raise NumericalCalibrationError("compute requires a fixed CPU affinity matching thread budget")
    fortran = identity.get("fortran")
    from .collective_calibration import normalize_fortran_options
    if not isinstance(fortran, dict) or any(not isinstance(fortran.get(name), str) or not fortran[name]
            for name in ("compiler_version", "compiler_options", "semantic_options")):
        raise NumericalCalibrationError("compute requires actual native Fortran compiler/options")
    if normalize_fortran_options(fortran["compiler_options"]) != fortran["semantic_options"]:
        raise NumericalCalibrationError("compute Fortran semantic options mismatch")
    indexed = {}
    for ordinal, row in enumerate(records):
        if row.get("kind") == "compute_identity_v2":
            continue
        family, backend, items = (row.get(name) for name in ("family", "backend", "items"))
        memory_piecewise = family == "memory_v2" and memory_model_kind == MEMORY_MODEL_KIND
        sizes, fit_sizes = (MEMORY_SIZES, MEMORY_FIT_SIZES) if memory_piecewise else (COMPUTE_SIZES, COMPUTE_FIT_SIZES)
        if row.get("kind") != "compute_cost_v2" or family not in COMPUTE_FAMILIES or backend not in COMPUTE_BACKENDS or type(items) is not int or items not in sizes:
            raise NumericalCalibrationError("unknown compute observation")
        if row.get("role") != ("fit" if items in fit_sizes else "holdout"):
            raise NumericalCalibrationError("compute fit/holdout role mismatch")
        if memory_piecewise and any(type(row.get(name)) is not int or row[name] != items * 3 * (profile["precision_bits"] // 8)
                                    for name in ("working_set_bytes", "traffic_bytes")):
            raise NumericalCalibrationError("memory observation physical footprint mismatch")
        _compute_samples(row)
        key = family, backend, items
        if key in indexed:
            raise NumericalCalibrationError("duplicate compute observation")
        indexed[key] = {**deepcopy(row), "observation_id": ordinal}
    generator = compute_generator_identity()
    families = {family: {backend: _compute_family(family, backend, indexed, generator)
                         for backend in COMPUTE_BACKENDS} for family in COMPUTE_COEFFICIENT_FAMILIES}
    if memory_model_kind == MEMORY_MODEL_KIND:
        families["memory_v2"] = {backend: _compute_memory_family(backend, indexed, generator, profile["precision_bits"],
            families["arithmetic_v2"][backend]) for backend in COMPUTE_BACKENDS}
    classes = {name: {backend: _compute_class(families, indexed, backend, name, profile["precision_bits"])
                      for backend in COMPUTE_BACKENDS} for name in COMPUTE_CLASSES}
    result = deepcopy(profile)
    result["numerical"] = {"schema_version": 2, "backend_id": COMPUTE_BACKEND_ID, "generator_id": generator,
        "precision_bits": profile["precision_bits"], "cpu_threads": profile["cpu_threads"],
        "hardware": {name: profile["hardware"][name] for name in HARDWARE_FIELDS},
        "toolchain": {name: profile["toolchain"][name] for name in TOOLCHAIN_FIELDS},
        "identity": deepcopy(identity), "protocol": {"id": COMPUTE_PROTOCOL_ID, "samples": COMPUTE_SAMPLES,
            "minimum_batch_seconds": COMPUTE_MIN_BATCH_SECONDS, "fit_sizes": list(COMPUTE_FIT_SIZES),
            "holdout_sizes": list(COMPUTE_HOLDOUT_SIZES), "maximum_relative_error": MAX_HOLDOUT_RELATIVE_ERROR,
            "backend_order": list(COMPUTE_BACKENDS), "cpu_affinity": affinity, "omp_proc_bind": "false"},
        "families": families, "workload_validation": classes, "private_limits": deepcopy(COMPUTE_PRIVATE_LIMITS),
        "measurements": deepcopy(records), "calibration": {**calibration, "application_profiled": False}}
    if memory_model_kind == MEMORY_MODEL_KIND:
        result["numerical"]["protocol"]["memory"] = {"kind": MEMORY_MODEL_KIND,
            "fit_sizes": list(MEMORY_FIT_SIZES), "holdout_sizes": list(MEMORY_HOLDOUT_SIZES),
            "working_set": "three distinct full arrays; 3*items*precision_bytes",
            "traffic": "one read per input and one write per output; 3*items*precision_bytes",
            "maximum_knots": 8, "extrapolation": False}
    return result


def validate_numerical_profile(section, profile):
    if isinstance(section, dict) and type(section.get("schema_version")) is int and section["schema_version"] == 2:
        # Reconstruct accepted and rejected evidence; saved status cannot bless
        # a failed holdout, nor may a rejected family poison another backend.
        rebuilt = profile_from_compute_measurements(profile, section.get("measurements", []),
            calibration=section.get("calibration", {}), memory_model_kind=section.get("protocol", {}).get("memory", {}).get("kind", "linear_v2"))["numerical"]
        # A saved generator may be older; validation verifies its recipes and
        # evidence, while the runtime reader separately requires current code.
        saved_generator = section.get("generator_id")
        if not isinstance(saved_generator, str) or len(saved_generator) != 64 or any(c not in "0123456789abcdef" for c in saved_generator):
            raise NumericalCalibrationError("compute generator must be a SHA-256 identity")
        rebuilt["generator_id"] = saved_generator
        for family in COMPUTE_COEFFICIENT_FAMILIES:
            for backend in COMPUTE_BACKENDS:
                rebuilt["families"][family][backend]["recipe_id"] = _compute_recipe_identity(family, saved_generator)
        if rebuilt != section:
            raise NumericalCalibrationError("saved compute evidence or identity is inconsistent")
        return
    _validate_numerical_profile_v1(section, profile)


def numerical_compute_model(profile, counts, *, workload_class="scalar_expression_v2", workload_features=None,
                            native_participation=None, backend_id=COMPUTE_BACKEND_ID, generator_id=None):
    """Return coefficients for total v3 compute, never a C++ native substitute.

    Bounds apply to runtime item counts. Callers must verify native compiler
    semantics/participation, selected device and process affinity before use.
    Fixed execution costs exclude runtime coordination, transfers and launches.
    """
    require_numerical_profile_contract(profile)
    section = profile.get("numerical")
    if not isinstance(section, dict) or section.get("schema_version") != 2:
        raise NumericalCalibrationError("native Fortran compute requires numerical calibration v2")
    validate_numerical_profile(section, profile)
    if section["backend_id"] != backend_id or section["generator_id"] != (generator_id or compute_generator_identity()):
        raise NumericalCalibrationError("compute implementation identity mismatch")
    if native_participation not in {"serial", "fork_join"}:
        raise NumericalCalibrationError("native participation unavailable; existing teams require separate calibration")
    if workload_class not in {"scalar_expression_v2", "fixed_private_array_v2"}:
        raise NumericalCalibrationError("unsupported compute workload class")
    if workload_features is not None:
        features = workload_features.to_dict() if hasattr(workload_features, "to_dict") else workload_features
        if (not isinstance(features, dict) or type(features.get("schema_version")) is not int or
                features["schema_version"] != 2 or features.get("classification_complete") is not True):
            raise NumericalCalibrationError("private-array classification incomplete")
    else:
        features = {}
    if workload_class == "fixed_private_array_v2":
        if (not features or any(type(features.get(name)) is not int or not 0 < features[name] <= limit
                               for name, limit in COMPUTE_PRIVATE_LIMITS.items())):
            raise NumericalCalibrationError("private-array workload outside calibrated applicability")
    elif features.get("private_array_elements", 0) != 0:
        raise NumericalCalibrationError("scalar workload contains private arrays")
    pairs = list(counts.items()) if isinstance(counts, dict) else list(counts)
    if len({name for name, _ in pairs}) != len(pairs) or any(name not in COMPUTE_PRIMITIVES or type(count) is not int or count <= 0 for name, count in pairs):
        raise NumericalCalibrationError("unsupported intrinsic counts")
    selected = {"native_fortran": "native_" + native_participation, "generated_cpu": "generated_cpu", "gpu": "gpu"}
    result = {"item_range": [min(COMPUTE_SIZES), max(COMPUTE_SIZES)], "native_participation": native_participation,
              "workload_class": workload_class, "cpu_affinity": section["identity"]["cpu_affinity"],
              "fortran": deepcopy(section["identity"]["fortran"]), "backend_id": backend_id,
              "generator_id": section["generator_id"]}
    for role, backend in selected.items():
        required = ("arithmetic_v2", "memory_v2", *("primitive_" + name + "_v2" for name, _ in pairs))
        for family in required:
            record = section["families"][family][backend]
            if record["status"] != "accepted":
                raise NumericalCalibrationError(f"{family}/{backend} rejected: {','.join(record['reason_codes'])}")
        validation_class = ("ordinary_expression_v2" if workload_class == "scalar_expression_v2" and
                            not any(name != "divide" for name, _ in pairs) else workload_class)
        validation = section["workload_validation"][validation_class][backend]
        if validation["status"] != "accepted":
            raise NumericalCalibrationError(f"{validation_class}/{backend} rejected: {','.join(validation['reason_codes'])}")
        components = _compute_components(section["families"], backend, profile["precision_bits"],
                                         tuple(name for name, _ in pairs))
        # Native serial has no team creation. Native fork/join and generated
        # CPU fixtures measure their actual parallel-for execution startup.
        fixed = _compute_fixed(section["families"], backend)
        result[role] = {**components,
            "intrinsic_seconds_per_item": math.fsum(count * components["intrinsic_seconds_per_operation"][name] for name, count in pairs),
            "fixed_seconds": fixed, "fixed_cost_coverage": "fork_join_execution_only" if fixed else "none",
            "backend_identity": backend}
    return result


def calibrate_numerical_v2(profile, args, directory: Path, nvcc: str, host: str, *, run, tool) -> dict:
    """Measure all predetermined components once, retaining rejected evidence."""
    require_explicit_cuda_environment()
    require_numerical_profile_contract(profile)
    from .collective_calibration import FORTRAN_FLAGS, normalize_fortran_options
    target = directory / "numerical-v2"
    target.mkdir(exist_ok=True)
    fortran = tool(getattr(args, "fortran", None), ("gfortran-15", "gfortran-14", "gfortran"))
    flags = list(getattr(args, "fortran_flag", None) or FORTRAN_FLAGS)
    if "-fopenmp" not in flags:
        raise NumericalCalibrationError("numerical v2 Fortran flags must include -fopenmp")
    allowed = sorted(os.sched_getaffinity(0))
    supplied = getattr(args, "cpu_affinity", None)
    try:
        affinity = sorted(set(int(cpu) for cpu in supplied.split(","))) if supplied else allowed[:args.threads]
    except ValueError as error:
        raise NumericalCalibrationError("CPU affinity must be comma-separated CPU indices") from error
    if len(affinity) != args.threads or not set(affinity) <= set(allowed):
        raise NumericalCalibrationError("CPU affinity must contain exactly the allowed thread budget")
    source = Path(__file__).with_suffix(".cu")
    obj, binary = target / "native.o", target / "numerical-calibration-v2"
    prepared_native = target / "native.f90"
    prepared_native.write_text(native_fixture_source(args.precision))
    compile_native = [fortran, *flags, "-c", str(prepared_native), "-o", str(obj)]
    run(compile_native, target, target / "native-build.log", timeout=180)
    # Link the Fortran runtime explicitly rather than relying on a host ABI
    # path. GNU Fortran is the currently supported native toolchain contract.
    command = [nvcc, "-O3", *cuda_compile_options(), "-std=c++17", "-arch=" + args.arch, "-ccbin", host,
               "-Xcompiler=-fopenmp", "-DCALIBRATION_V2=1", "-DCALIBRATION_PRECISION=" + str(args.precision),
               str(source), str(obj), "-lgfortran", "-o", str(binary)]
    run(command, target, target / "build.log", timeout=180)
    environment = {name: value for name, value in os.environ.items()
        if name not in {"FORT_RUNTIME_TRACE", "FORT_PHASE_TIMING", "CUDA_LAUNCH_BLOCKING", "OMP_PLACES", "GOMP_CPU_AFFINITY"}
        and not name.startswith("FORT_SCOPE_TEST_")}
    environment.update(OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE", OMP_PROC_BIND="false", OMP_SCHEDULE="static")
    invocation = ["taskset", "-c", ",".join(map(str, affinity)), str(binary), str(args.threads), str(args.device)]
    text = run(invocation, target, target / "measurements.jsonl", timeout=1800, env=environment)
    records = parse_compute_measurements(text)
    # The fixture emits compiler_version/options from the actual object; Python
    # performs the shared semantic normalization before storing the identity.
    for row in records:
        if row["kind"] == "compute_identity_v2":
            row["fortran"]["semantic_options"] = normalize_fortran_options(row["fortran"]["compiler_options"])
    result = profile_from_compute_measurements(profile, records, calibration={
        "build_commands": [compile_native, command], "run_command": invocation,
        "artifacts": str(target), "cpu_method": "actual native Fortran serial/fork-join and generated C++ fork-join; fixed affinity",
        "gpu_method": "resident CUDA events; transfer and enqueue management priced separately",
        "sampling": "seven predetermined interleaved batches of at least 200 ms per size/backend; no retry",
        "fit_method": "nonnegative linear fit; independent size and expression holdouts; 25% ceiling"})
    (target / "evidence.json").write_text(json.dumps(result["numerical"], indent=2, allow_nan=False) + "\n")
    return result
