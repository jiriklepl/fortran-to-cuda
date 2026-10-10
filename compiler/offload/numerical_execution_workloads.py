"""Predeclared source-backed CPU numerical execution controls.

The registry is data, not fitted evidence. Native Fortran owns its original
serial/static worksharing loops; generated CPU bodies use the production
cyclic worker. Nothing here examines application sources or observations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.frontend import lower_source
from compiler.ir import Assignment, Loop
from compiler.ir.plan import ParallelRegion
from compiler.numerical_contract import numerical_build_contract
from compiler.offload.analysis import _compute_operation_counts
from compiler.offload.compute_dependencies import analyze_compute_dependencies
from compiler.offload.cpu_dependency_workloads import (
    CpuDependencyInputs,
    CpuDependencyRecipe,
    _cpp_sources,
    _lattice,
    _round,
)
from compiler.offload.cpu_protocol_calibration import worker_renderer_identities

PROTOCOL_ID = "original-fortran-production-cyclic-v1"
BACKEND_ID = "native-fortran-production-cyclic-cxx17-v1"
CPU_BACKENDS = ("native_serial", "native_fork_join", "generated_cpu")
FIT_SIZES = (65536, 262144, 1048576)
HOLDOUT_SIZES = (131072, 524288)
SIZES = tuple(sorted(FIT_SIZES + HOLDOUT_SIZES))
MEMORY_FIT_SIZES = tuple(65536 * 3**k // 2**k for k in range(8))
MEMORY_HOLDOUT_SIZES = (80265, 120397, 131072, 180596, 270894, 406341, 524288, 609511, 914267)
MEMORY_SIZES = tuple(sorted(MEMORY_FIT_SIZES + MEMORY_HOLDOUT_SIZES))
STARTUP_SIZES = (0, 1, 8)
ROUNDS = 7
MIN_BATCH_SECONDS = 0.2
ACCESS_CLASS = "pointwise_three_array_v1"
NAMES = (
    "arithmetic",
    "memory",
    "sqrt",
    "acos",
    "cos",
    "scalar_mix",
    "scalar_mix_skew",
    "private_mix",
    "private_mix_skew",
    "divide",
    "ordinary_mix",
    "helper_scalar_mix",
    "helper_private_mix",
    "dependent_arithmetic",
    "domain_sqrt",
    "domain_acos",
    "domain_cos",
)
DOMAINS = {
    "sqrt": "finite_nonnegative_normal_or_zero",
    "acos": "finite_unit_interval",
    "cos": "finite_pi_interval",
    "divide": "finite_positive_denominator_lattice_v1",
}


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _precision(value):
    if type(value) is not int or value not in (32, 64):
        raise ValueError("numerical execution precision must be 32 or 64")
    return value // 8


def _body(name, precision):
    kind = _precision(precision)

    def real(value):
        return str(value) + f"_{kind}"

    def primitive(operation, x):
        argument = {
            "sqrt": f"sqrt({real('1.0')}+{real('0.25')}*{x}*{x})",
            "acos": f"acos({real('0.125')}*{x})",
            "cos": f"cos({x})",
        }[operation]
        return f"{real('0.25')}+{real('0.125')}*{argument}"

    base = name.removeprefix("helper_")
    lines = ["input_value=a(i)", "other_value=b(i)", "x=input_value"]
    if base == "memory":
        return "\n".join(lines[:2] + [f"output(i)=input_value+{real('0.25')}*other_value"])
    if base in {"arithmetic", "divide"}:
        lines += ["x1=x", f"x2=x*{real('0.5')}", f"x3=x*{real('0.25')}", f"x4=x*{real('0.125')}"]
        for _ in range(64 if base == "arithmetic" else 16):
            if base == "arithmetic":
                for column in range(1, 5):
                    lines.append(f"x{column}=x{column}*{real('1.00000' + str(column))}+{real('0.00000' + str(column))}")
            else:
                for column, next_column, add, scale in (
                    (1, 2, "0.125", "0.25"),
                    (2, 3, "0.25", "0.125"),
                    (3, 4, "0.5", "0.0625"),
                    (4, 1, "0.75", "0.03125"),
                ):
                    lines.append(
                        f"x{column}=(x{column}+{real(add)})/(other_value+x{next_column}*{real(scale)}+{real('1.0')})"
                    )
        lines.append("x=x1+x2+x3+x4")
    elif base in {"sqrt", "acos", "cos"}:
        lines += ["x=" + primitive(base, "x") for _ in range(16)]
    elif base == "dependent_arithmetic":
        lines += [f"x=x*{real('1.000001')}+{real('0.000001')}" for _ in range(64)]
    elif base.startswith("domain_"):
        lines.append("x=" + base[7:] + "(x)")
    elif base == "ordinary_mix":
        for _ in range(64):
            lines += [
                f"x=x*{real('1.000001')}+{real('0.000001')}",
                f"x=max({real('-0.9')},min({real('-0.1')},-abs(x)))",
                f"x=(x-{real('0.1')})/({real('-1.0')}-other_value)",
            ]
    elif base in {"scalar_mix", "scalar_mix_skew", "private_mix", "private_mix_skew"}:
        if base.startswith("private_"):
            width = 4 if base == "private_mix" else 2
            for col in range(1, width + 1):
                for row in range(1, width + 1):
                    lines += [
                        f"pa({row},{col})=x+{real(str((row + col) / 100))}",
                        f"pb({row},{col})={real('0.02')}*x+{real('1.0' if row == col else '0.0')}",
                    ]
            for col in range(1, width + 1):
                for row in range(1, width + 1):
                    lines.append(f"total=pa({row},1)*pb(1,{col})")
                    lines += [f"total=total+pa({row},{k})*pb({k},{col})" for k in range(2, width + 1)]
                    lines += [f"pc({row},{col})=total", f"pd({row},{col})=pc({row},{col})+pa({row},{col})"]
            lines += ["total=pd(1,1)"] + [f"total=total+pd({k},{k})" for k in range(2, width + 1)]
            lines.append(f"x={real('0.125')}+{real('0.01')}*total")
        roots, angles, cosines = (6, 2, 5) if base in {"scalar_mix", "private_mix"} else (2, 4, 1)
        for _ in range(16):
            for operation, count in (("sqrt", roots), ("acos", angles), ("cos", cosines)):
                lines += ["x=" + primitive(operation, "x") for _ in range(count)]
            lines += [
                f"x=max({real('-0.9')},min({real('-0.1')},-abs(x)))",
                f"x=(x-{real('0.1')})/({real('-1.0')}-other_value)",
            ]
    else:
        raise ValueError("unknown predeclared numerical execution recipe")
    # An explicit second input load and nonzero contribution make the common
    # two-read/one-write access contract source-backed, without assuming CSE.
    lines.append(f"output(i)=x+other_value*{real('0.001')}")
    return "\n".join(lines)


def _declarations(name, precision):
    private = name.removeprefix("helper_").startswith("private_")
    width = 2 if name == "private_mix_skew" else 4
    return f"real({_precision(precision)})::input_value,other_value,x,x1,x2,x3,x4,total" + (
        "," + ",".join(f"p{letter}({width},{width})" for letter in "abcd") if private else ""
    )


def _analysis_source(name, precision):
    return f"""module execution_{name}
implicit none
contains
subroutine evaluate(n,a,d,output)
integer,intent(in)::n
real({_precision(precision)}),intent(in)::a(:),d(:)
real({_precision(precision)}),intent(out)::output(:)
integer::i
{_declarations(name, precision)}
do i=1,n
{_body(name, precision).replace("b(i)", "d(i)")}
enddo
end subroutine
end module
"""


def _native_sources(name, precision):
    kind = "c_double" if precision == 64 else "c_float"
    helper = name.startswith("helper_")
    body = _body(name, precision)
    declarations = _declarations(name, precision)
    function_name = "execution_" + name + "_value"
    function_body = (
        body.replace("input_value=a(i)", "input_value=input")
        .replace("other_value=b(i)", "other_value=other")
        .replace("output(i)", "value")
    )
    function = f"""pure function {function_name}(input,other) result(value)
real({kind}),intent(in)::input,other
real({kind})::value
{declarations}
{function_body}
end function
"""
    prefix = f"module execution_native_{name}\nuse iso_c_binding\n"
    support = ()
    if helper:
        support = (
            (
                name + "_helper.f90",
                f"module execution_helper_{name}\nuse iso_c_binding\nimplicit none\ncontains\n{function}end module\n",
            ),
        )
        prefix += f"use execution_helper_{name},only:{function_name}\n"
    workers = []
    for role in ("serial", "fork_join"):
        directives = (
            (
                "!$omp parallel do num_threads(threads) schedule(static) default(none) &\n"
                "!$omp shared(n,threads,a,b,output) private(i)\n"
            )
            if role == "fork_join"
            else ""
        )
        end = "!$omp end parallel do\n" if role == "fork_join" else ""
        workers.append(f"""subroutine execution_{name}_{role}(n,threads,a,b,output) bind(c)
integer(c_int),value::n,threads
real({kind}),intent(in)::a(n),b(n)
real({kind}),intent(out)::output(n)
integer::i
{directives}do i=1,n
output(i)={function_name}(a(i),b(i))
enddo
{end}end subroutine
""")
    caller = prefix + "implicit none\ncontains\n" + ("" if helper else function) + "".join(workers) + "end module\n"
    return (*support, (name + ".f90", caller))


# Counts are declaration data. Tests verify them against the exact prepared IR;
# loading a profile must not parse or render numerical source.
_COUNTS = {
    "arithmetic": (520, {}),
    "memory": (2, {}),
    "sqrt": (82, {"sqrt": 16}),
    "acos": (50, {"acos": 16}),
    "cos": (34, {"cos": 16}),
    "scalar_mix": (850, {"sqrt": 96, "acos": 32, "cos": 80, "divide": 16}),
    "scalar_mix_skew": (498, {"sqrt": 32, "acos": 64, "cos": 16, "divide": 16}),
    "private_mix": (1031, {"sqrt": 96, "acos": 32, "cos": 80, "divide": 16}),
    "private_mix_skew": (529, {"sqrt": 32, "acos": 64, "cos": 16, "divide": 16}),
    "divide": (328, {"divide": 64}),
    "ordinary_mix": (578, {"divide": 64}),
    "dependent_arithmetic": (130, {}),
    "domain_sqrt": (2, {"sqrt": 1}),
    "domain_acos": (2, {"acos": 1}),
    "domain_cos": (2, {"cos": 1}),
}


def execution_registry(precision):
    """Return fresh metadata dictionaries only; no frontend or worker execution."""
    _precision(precision)
    rows = []
    for name in NAMES:
        base = name.removeprefix("helper_")
        arithmetic, intrinsics = _COUNTS[base]
        coefficient = base in {"arithmetic", "sqrt", "acos", "cos", "divide"} and not name.startswith("helper_")
        role = "memory" if base == "memory" else "coefficient" if coefficient else "holdout"
        private = base.startswith("private_")
        width = 2 if base == "private_mix_skew" else 4
        private_features = {
            "private_array_groups": 4 if private else 0,
            "private_array_elements": 4 * width * width if private else 0,
            "max_private_array_elements": width * width if private else 0,
            "max_private_array_rank": 2 if private else 0,
        }
        domains = {key: DOMAINS[key] for key in intrinsics}
        row = {
            "name": name,
            "family": base.removeprefix("domain_"),
            "role": role,
            "holdout_kind": "separate_helper"
            if name.startswith("helper_")
            else "domain"
            if name.startswith("domain_")
            else "dependency"
            if name == "dependent_arithmetic"
            else "expression"
            if role == "holdout"
            else None,
            "arithmetic": arithmetic,
            "intrinsics": dict(intrinsics),
            "workload_class": "fixed_private_array_v2"
            if private
            else "ordinary_expression_v2"
            if base in {"ordinary_mix", "dependent_arithmetic"}
            else "scalar_expression_v2",
            "private_features": private_features,
            "domain_ids": domains,
            "precision_bits": precision,
            "access_class": ACCESS_CLASS,
            "sizes": list(MEMORY_SIZES if role == "memory" else SIZES),
            "fit_sizes": list(MEMORY_FIT_SIZES if role == "memory" else FIT_SIZES if role == "coefficient" else ()),
            "source_sha256": _hash(
                {"analysis": _analysis_source(name, precision), "native": _native_sources(name, precision)}
            ),
            "input_lattices": {
                split: {
                    "a": list(_lattice(name[7:] if name.startswith("domain_") else None, precision, split)),
                    "b": list(
                        (2.0, 2.5, 3.0, 3.5, 4.0) if split == "training" else (2.125, 2.625, 3.125, 3.625, 3.875)
                    ),
                }
                for split in ("training", "holdout")
            },
            "helper_form": "separate" if name.startswith("helper_") else "same_translation_unit",
            "source_normalization": "predeclared bounded straight-line work item; no application source rewriting",
        }
        row["recipe_id"] = _hash(row)
        rows.append(row)
    return tuple(rows)


def registry_identity(precision):
    return _hash(execution_registry(precision))


def execution_generator_identity():
    here = Path(__file__).parent
    names = (
        "numerical_execution_workloads.py",
        "numerical_execution_driver.cpp",
        "numerical_execution_calibration.py",
        "cpu_protocol_calibration.py",
        "analysis.py",
        "compute_dependencies.py",
    )
    sources = {name: sha256((here / name).read_bytes()).hexdigest() for name in names}
    sources["numeric.hpp"] = sha256((here.parent / "runtime/numeric.hpp").read_bytes()).hexdigest()
    sources["common_functions.cuh"] = sha256(
        (here.parent / "emission/common/templates/common_functions.cuh").read_bytes()
    ).hexdigest()
    sources["cpu_dependency_workloads.py"] = sha256((here / "cpu_dependency_workloads.py").read_bytes()).hexdigest()
    # Parsing, source inlining and preparation are numerical backend authority.
    # Reading these hashes never runs frontend analysis when a profile loads.
    for folder in ("frontend", "ir", "driver", "passes"):
        for path in sorted((here.parent / folder).rglob("*.py")):
            sources[str(path.relative_to(here.parent))] = sha256(path.read_bytes()).hexdigest()
    return _hash({"sources": sources, "renderer": worker_renderer_identities(), "contract": numerical_build_contract()})


@dataclass(frozen=True)
class ExecutionRecipe:
    name: str
    metadata: dict
    analysis_source: str
    native_sources: tuple[tuple[str, str], ...]
    cpp_sources: tuple[tuple[str, str], ...]
    region: ParallelRegion

    def lattices(self, split):
        if split not in {"training", "holdout"}:
            raise ValueError("unknown input lattice split")
        primitive = self.name[7:] if self.name.startswith("domain_") else None
        a = tuple(_lattice(primitive, self.metadata["precision_bits"], split))
        b = (2.0, 2.5, 3.0, 3.5, 4.0) if split == "training" else (2.125, 2.625, 3.125, 3.625, 3.875)
        return tuple(_round(value, self.metadata["precision_bits"]) for value in a), tuple(
            _round(value, self.metadata["precision_bits"]) for value in b
        )

    def reference_period(self, split):
        a, b = self.lattices(split)
        n = math.lcm(len(a), len(b))
        inputs = CpuDependencyInputs(
            n,
            tuple(a[i % len(a)] for i in range(n)),
            tuple(b[i % len(b)] for i in range(n)),
            self.metadata["precision_bits"],
            split,
        )
        # The maintained typed interpreter evaluates assignments from this exact
        # lowered region; this is correctness evidence, never fitted work.
        return CpuDependencyRecipe.reference(
            SimpleNamespace(region=self.region, precision_bits=inputs.precision_bits), inputs
        )


@lru_cache(maxsize=2, typed=True)
def execution_recipes(precision):
    result = []
    for meta in execution_registry(precision):
        name = meta["name"]
        source = _analysis_source(name, precision)
        function = lower_source(source, f"execution_{name}::evaluate", source_name=name + "-authority.f90")
        (loop,) = function.body.statements
        if not isinstance(loop, Loop) or any(not isinstance(s, Assignment) for s in loop.body.statements):
            raise ValueError("execution recipe is not one exact straight-line numerical item")
        function, plan = prepare_function(function, options=CompilerOptions(opt_level=0))
        (region,) = plan.steps
        if not isinstance(region, ParallelRegion):
            raise ValueError("execution recipe is not independent")
        count, divisions, reason = _compute_operation_counts(region.body)
        if reason or count != meta["arithmetic"] or divisions != meta["intrinsics"].get("divide", 0):
            raise ValueError(f"execution recipe count declaration mismatch: {name} {count} {divisions}: {reason}")
        if name.startswith("helper_"):
            # Only fixture module placement is normalized. Actual native
            # compilation below keeps the helper in its separate source unit.
            native_helper = _native_sources(name, precision)[0][1]
            helper_body = native_helper.split("contains\n", 1)[1].rsplit("end module", 1)[0]
            normalized = (
                f"module execution_{name}\nimplicit none\ncontains\n"
                f"subroutine evaluate(n,a,d,output)\ninteger,intent(in)::n\n"
                f"real({_precision(precision)}),intent(in)::a(:),d(:)\n"
                f"real({_precision(precision)}),intent(out)::output(:)\ninteger::i\n"
                f"do i=1,n\noutput(i)=execution_{name}_value(a(i),d(i))\nenddo\nend subroutine\n"
                + helper_body.replace("real(c_double)", "real(8)").replace("real(c_float)", "real(4)")
                + "end module\n"
            )
            transitive = lower_source(
                normalized, f"execution_{name}::evaluate", source_name=name + "-normalized-helper-authority.f90"
            )
            transitive, tplan = prepare_function(transitive, options=CompilerOptions(opt_level=0))
            (tregion,) = tplan.steps
            direct_graph = analyze_compute_dependencies(region)
            helper_graph = analyze_compute_dependencies(tregion)
            if (
                not direct_graph.available
                or not helper_graph.available
                or direct_graph.operations != helper_graph.operations
                or direct_graph.outputs != helper_graph.outputs
                or direct_graph.floating_dtypes != helper_graph.floating_dtypes
                or direct_graph.unpriced_numerical_operations != helper_graph.unpriced_numerical_operations
            ):
                raise ValueError("separate native helper changes the transitive numerical computation")
        cpp = _cpp_sources("execution_" + name, function, region)
        result.append(ExecutionRecipe(name, meta, source, _native_sources(name, precision), cpp, region))
    return tuple(result)


def registry_source(recipes, generator_id):
    """Render the public standalone driver registry, not a parallel worker."""
    recipes = tuple(recipes)
    if len(recipes) != 17 or tuple(r.name for r in recipes) != NAMES:
        raise ValueError("the complete predeclared execution registry is required")
    precision = recipes[0].metadata["precision_bits"]
    if any(r.metadata["precision_bits"] != precision for r in recipes):
        raise ValueError("mixed registry precision")
    lines = [
        "#pragma once",
        "#include <cstddef>",
        "using execution_real=" + ("double" if precision == 64 else "float") + ";",
        "using execution_worker=void(*)(int,int,const execution_real*,const execution_real*,execution_real*);",
        "struct execution_recipe { const char* name; const char* identity; const char* role; const char* family;",
        "execution_worker native_serial,native_fork_join,generated_cpu;",
        "const int* sizes; std::size_t size_count; const int* fit_sizes; std::size_t fit_count;",
        "const execution_real* training_a; std::size_t training_a_count;",
        "const execution_real* training_b; std::size_t training_b_count;",
        "const execution_real* training_reference; std::size_t training_reference_count;",
        "const execution_real* holdout_a; std::size_t holdout_a_count;",
        "const execution_real* holdout_b; std::size_t holdout_b_count;",
        "const execution_real* holdout_reference; std::size_t holdout_reference_count; };",
        "static constexpr const char* execution_protocol_id=" + json.dumps(PROTOCOL_ID) + ";",
        "static constexpr const char* execution_backend_id=" + json.dumps(BACKEND_ID) + ";",
        "static constexpr const char* execution_generator_id=" + json.dumps(generator_id) + ";",
        "static constexpr const char* execution_registry_id=" + json.dumps(registry_identity(precision)) + ";",
    ]
    entries = []
    for r in recipes:
        prefix = "execution_" + r.name
        for role in ("serial", "fork_join", "generated"):
            lines.append(
                f'extern "C" void {prefix}_{role}(int,int,const execution_real*,const execution_real*,execution_real*);'
            )
        sizes = r.metadata["sizes"]
        fits = r.metadata["fit_sizes"]
        lines += [
            f"static constexpr int {prefix}_sizes[]={{" + ",".join(map(str, sizes)) + "};",
            f"static constexpr int {prefix}_fits[]={{" + ",".join(map(str, fits or [0])) + "};",
        ]
        for split in ("training", "holdout"):
            a, b = r.lattices(split)
            reference = r.reference_period(split)
            for label, values in (("a", a), ("b", b), ("reference", reference)):
                lines.append(
                    f"static constexpr execution_real {prefix}_{split}_{label}[]={{"
                    + ",".join("execution_real(" + format(v, ".17g") + ")" for v in values)
                    + "};"
                )
        fields = [
            json.dumps(r.name),
            json.dumps(r.metadata["recipe_id"]),
            json.dumps(r.metadata["role"]),
            json.dumps(r.metadata["family"]),
            prefix + "_serial",
            prefix + "_fork_join",
            prefix + "_generated",
            prefix + "_sizes",
            str(len(sizes)),
            prefix + "_fits",
            str(len(fits)),
        ]
        for split in ("training", "holdout"):
            a, b = r.lattices(split)
            for label, count in (("a", len(a)), ("b", len(b)), ("reference", math.lcm(len(a), len(b)))):
                fields += [prefix + "_" + split + "_" + label, str(count)]
        entries.append("{" + ",".join(fields) + "}")
    lines.append("static const execution_recipe execution_recipes[]={" + ",\n".join(entries) + "};")
    return "\n".join(lines) + "\n"


def native_identity_source():
    return """subroutine fort_numerical_execution_fortran_identity_v1(version,options,capacity) bind(c)
use iso_c_binding
use iso_fortran_env,only:compiler_version,compiler_options
implicit none
integer(c_int),value::capacity
character(c_char),intent(out)::version(*),options(*)
character(:),allocatable::v,o
integer::i
v=compiler_version()
o=compiler_options()
do i=1,min(len(v),capacity-1)
version(i)=v(i:i)
enddo
version(min(len(v),capacity-1)+1)=c_null_char
do i=1,min(len(o),capacity-1)
options(i)=o(i:i)
enddo
options(min(len(o),capacity-1)+1)=c_null_char
end subroutine
"""
