"""Predeclared generic CPU dependency fixtures, without measurements.

The 12 bases, eight structural holdouts and six domain holdouts are fixed
before sampling. Original Fortran and generated C++ use the same numerical
body. Native helper forms remain separate translation units; callers must
compile without LTO. Generated closures use the production cyclic worker.
Nothing here reads application sources or timings.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import abi_arguments, dimension_name
from compiler.emission.common.c_family import cpp_type
from compiler.emission.cuda.offload import _cpu_worker
from compiler.frontend import lower_source
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    IntrinsicCall,
    Literal,
    Loop,
    Reference,
    Unary,
)
from compiler.ir.plan import ParallelRegion
from compiler.offload.analysis import Unit
from compiler.offload.compute_dependencies import ComputeDependencies, analyze_compute_dependencies

PROTOCOL_VERSION = 1
BASIS_STEPS = 32
BASIS_FAMILIES = ("ordinary", "divide_dynamic", "divide_constant", "sqrt", "acos", "cos")
STRUCTURAL_GRAPHS = ("ordinary_guarded", "dependent_mixed", "diamond", "private_arrays")
DOMAIN_FAMILIES = ("acos", "sqrt", "cos")
PROTOCOL_REQUIREMENTS = (
    "seven interleaved batches of at least200ms per size and role",
    "basis widths1/4 identify span/work respectively; roles cannot be switched after sampling",
    "structural and domain holdouts never fit coefficients",
    "separate helper translation units; LTO prohibited",
    "analysis normalizes only helper module placement; native helpers remain separate",
    "ordinary holdout includes ABS, unary negate and ordered MIN/MAX",
    "private holdout fully populates and consumes two4x4 groups,32elements,rank2",
    "precision profiles and original native participation remain independent",
    "generated CPU uses the production cyclic worker and one original fork/join dispatch",
    "native helper holdouts remain separate; generated numerical closures are source-inlined",
)
_DENOMINATORS = ((2.0, 2.5, 3.0, 3.5, 4.0), (2.125, 2.625, 3.125, 3.625, 3.875))


def _round(value, precision_bits):
    return struct.unpack("f", struct.pack("f", value))[0] if precision_bits == 32 else float(value)


@dataclass(frozen=True)
class CpuDependencyInputs:
    """Two contiguous input arrays A(n),D(n); explicit ordinary numeric ABI."""

    n: int
    a: tuple[float, ...]
    d: tuple[float, ...]
    precision_bits: int
    split: str


@dataclass(frozen=True)
class CpuDependencyRecipe:
    name: str
    role: str
    coefficient_family: str | None
    width: int
    steps: int
    precision_bits: int
    helper_form: str
    source_body: str
    graph: ComputeDependencies
    domains: tuple[tuple[str, str], ...]
    workload_class: str
    memory_arrays: int
    fortran_sources: tuple[tuple[str, str], ...]
    cpp_sources: tuple[tuple[str, str], ...]
    native_serial_entry: str
    native_fork_join_entry: str
    generated_entry: str
    analysis_source: str
    region: ParallelRegion
    identity: str
    generated_dispatch_identity: str
    protocol_version: int = PROTOCOL_VERSION

    def to_dict(self):
        return {
            "protocol_version": self.protocol_version,
            "name": self.name,
            "role": self.role,
            "coefficient_family": self.coefficient_family,
            "width": self.width,
            "steps": self.steps,
            "precision_bits": self.precision_bits,
            "helper_form": self.helper_form,
            "identity": self.identity,
            "graph": self.graph.to_dict(),
            "domains": dict(self.domains),
            "workload_class": self.workload_class,
            "memory_arrays": self.memory_arrays,
            "native_serial_entry": self.native_serial_entry,
            "native_fork_join_entry": self.native_fork_join_entry,
            "generated_entry": self.generated_entry,
            "generated_dispatch_identity": self.generated_dispatch_identity,
            "generated_dispatch": "production _cpu_worker cyclic tid/team; one fixed-budget parallel team",
            "generated_helper_form": "source_inlined_proven_closure",
            "abi": "void(int n,int threads,const real*a,const real*d,real*output)",
            "compile_requirements": list(PROTOCOL_REQUIREMENTS),
            "fortran_sources": [{"path": name, "sha256": _sha(text)} for name, text in self.fortran_sources],
            "cpp_sources": [{"path": name, "sha256": _sha(text)} for name, text in self.cpp_sources],
        }

    def inputs(self, n: int, *, split: str = "holdout") -> CpuDependencyInputs:
        """Balanced deterministic lattices; domain rows reach their full range.

        Small n may contain only a lattice prefix. Fit/holdout sizes include
        all points. The complementary lattice is fixed before measurements.
        Normal-domain rows establish no subnormal/exceptional applicability.
        """
        if type(n) is not int or not 0 <= n <= 2**31 - 1:
            raise ValueError("fixture item count must fit the nonnegative INTEGER ABI")
        if split not in {"training", "holdout"}:
            raise ValueError("unknown fixture input split")
        lattice = _lattice(
            self.coefficient_family if self.role == "domain_holdout" else None, self.precision_bits, split
        )
        other = _DENOMINATORS[0 if split == "training" else 1]
        values = tuple(_round(lattice[index % len(lattice)], self.precision_bits) for index in range(n))
        denominators = tuple(_round(other[index % len(other)], self.precision_bits) for index in range(n))
        return CpuDependencyInputs(n, values, denominators, self.precision_bits, split)

    def reference(self, inputs: CpuDependencyInputs) -> tuple[float, ...]:
        """Typed scalar reference for correctness, not a native cost model."""
        if inputs.precision_bits != self.precision_bits or len(inputs.a) != inputs.n or len(inputs.d) != inputs.n:
            raise ValueError("fixture inputs do not match recipe precision or layout")
        result = []
        for item in range(inputs.n):
            values = {self.region.loops[0].iterator: item + 1}

            def expression(node, values=values):
                if isinstance(node, Literal):
                    return _round(float(node.value.replace("d", "e").replace("D", "e")), self.precision_bits)
                if isinstance(node, Reference):
                    return values[node.symbol]
                if isinstance(node, ArrayAccess):
                    coordinates = tuple(int(expression(index)) - 1 for index in node.indices)
                    if node.symbol.name == "a":
                        return inputs.a[coordinates[0]]
                    if node.symbol.name == "d":
                        return inputs.d[coordinates[0]]
                    raise ValueError("unexpected reference input array")
                if isinstance(node, Unary):
                    value = expression(node.operand)
                    return _round(-value if node.operator == "-" else value, self.precision_bits)
                if isinstance(node, Binary):
                    left, right = expression(node.left), expression(node.right)
                    if node.operator == "+":
                        value = left + right
                    elif node.operator == "-":
                        value = left - right
                    elif node.operator == "*":
                        value = left * right
                    elif node.operator == "/":
                        value = left / right
                    else:
                        raise ValueError("unexpected reference operator")
                    return _round(value, self.precision_bits)
                if isinstance(node, IntrinsicCall):
                    args = tuple(expression(arg) for arg in node.arguments)
                    if node.name in {"abs", "min", "max"}:
                        value = {"abs": abs, "min": min, "max": max}[node.name](*args)
                    else:
                        value = getattr(math, node.name)(*args)
                    return _round(value, self.precision_bits)
                raise ValueError("unexpected reference expression")

            output = None
            for statement in self.region.body.statements:
                value = expression(statement.value)
                if isinstance(statement.target, Reference):
                    values[statement.target.symbol] = _round(value, self.precision_bits)
                else:
                    output = value
            result.append(output)
        return tuple(result)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _lattice(family, precision_bits, split):
    epsilon = 2.0 ** (-24 if precision_bits == 32 else -53)
    if family == "acos":
        return (
            (-1.0, -1.0 + epsilon, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0 - epsilon, 1.0)
            if split == "training"
            else (-1.0, -1.0 + 4 * epsilon, -0.875, -0.5, -0.125, 0.0, 0.125, 0.5, 0.875, 1.0 - 4 * epsilon, 1.0)
        )
    if family == "sqrt":
        return (
            (0.0, 2.0**-60, 2.0**-30, 0.25, 1.0, 2.0, 2.0**30, 2.0**60)
            if split == "training"
            else (0.0, 2.0**-60, 2.0**-45, 2.0**-15, 0.5, 3.0, 9.0, 2.0**15, 2.0**45, 2.0**60)
        )
    if family == "cos":
        return (
            (-math.pi, -math.pi / 2, -1.0, 0.0, 1.0, math.pi / 2, math.pi)
            if split == "training"
            else (
                -math.pi,
                -math.pi / 2 - epsilon,
                -math.pi / 2 + epsilon,
                -2.0,
                -0.5,
                0.0,
                0.5,
                2.0,
                math.pi / 2 - epsilon,
                math.pi / 2 + epsilon,
                math.pi,
            )
        )
    return (
        (-0.875, -0.5, -0.125, 0.0, 0.125, 0.5, 0.875)
        if split == "training"
        else (-1.0, -0.625, -0.25, -0.0625, 0.0, 0.0625, 0.25, 0.625, 1.0)
    )


def _body(kind, width, precision_bits):
    suffix = "_8" if precision_bits == 64 else "_4"

    def real(value):
        return str(float(value)) + suffix

    lines = ["x0=a(i)"] + [f"x{column}=a(i)+{real(column / 8)}" for column in range(1, width)]
    if kind in BASIS_FAMILIES:
        for _ in range(BASIS_STEPS):
            for column in range(width):
                x = f"x{column}"
                expression = {
                    "ordinary": f"{x}*{real(0.75)}+{real(0.125)}",
                    "divide_dynamic": f"({x}+{real(1)})/d(i)",
                    "divide_constant": f"({x}+{real(1)})/{real(3)}",
                    "sqrt": f"sqrt(abs({x})+{real(0.125)})",
                    "acos": f"acos(max({real(-1)},min({real(1)},{x}*{real(0.25)})))*{real(0.25)}",
                    "cos": f"cos({x}*{real(0.5)})",
                }[kind]
                lines.append(f"{x}={expression}")
    elif kind == "ordinary_guarded":
        for _ in range(9):
            lines += [
                f"x0=max({real(-2)},min({real(2)},-abs(x0)*{real(0.75)}+{real(0.125)}))",
                f"x0=(x0+{real(0.25)})/d(i)",
                f"x0=x0/{real(3)}",
            ]
    elif kind == "dependent_mixed":
        for _ in range(9):
            lines += [
                f"x0=sqrt(abs(x0)+{real(0.125)})",
                f"x0=acos(max({real(-1)},min({real(1)},x0*{real(0.25)})))",
                f"x0=cos(x0)/(d(i)+{real(2)})",
                f"x0=x0/{real(3)}",
            ]
    elif kind == "diamond":
        for _ in range(9):
            lines += [
                f"x0=sqrt(abs(x0)+{real(0.5)})",
                f"x1=cos(x1*{real(0.5)})",
                f"t0=acos(max({real(-1)},min({real(1)},x0*{real(0.25)})))",
                f"t1=x1/(d(i)+{real(2)})",
                f"x0=t0*{real(0.25)}+t1",
                f"x1=t0*{real(0.125)}-t1",
            ]
    elif kind == "private_arrays":
        for column in range(1, 5):
            for row in range(1, 5):
                lines += [
                    f"small({row},{column})=x0*{real((row + column) / 64)}+x1*{real(row / 64)}",
                    f"other({row},{column})=x1*{real((row + column) / 64)}-x0*{real(column / 64)}",
                ]
        lines += [
            "t0=dot_product(small(:,1),other(:,1))",
            *(f"t0=t0+dot_product(small(:,{column}),other(:,{column}))" for column in range(2, 5)),
            f"x0=sqrt(abs(t0)+{real(0.5)})",
            f"x1=acos(max({real(-1)},min({real(1)},t0*{real(0.125)})))",
            f"x0=cos(x1)*x0/(d(i)+{real(2)})",
        ]
        # Both branches contribute through the fully consumed private arrays;
        # only the final scalar expression is published.
        width = 1
    elif kind.startswith("domain_"):
        lines.append(f"x0={kind[7:]}(x0)")
    else:
        raise ValueError("unknown predefined fixture")
    lines.append("output(i)=" + "+".join(f"x{column}" for column in range(width)) + f"+d(i)*{real(0.001)}")
    numerical = "\n".join(lines).replace("a(i)", "input_value").replace("d(i)", "denominator_value")
    return "input_value=a(i)\ndenominator_value=d(i)\n" + numerical


def _declarations(kind, precision_bits):
    result = f"real({precision_bits // 8})::x0,x1,x2,x3,t0,t1,input_value,denominator_value"
    if kind == "private_arrays":
        result += ",small(4,4),other(4,4)"
    return result


def _helper_source(name, body, declarations, precision_bits):
    helper_module, helper = "h_" + name, "f_" + name
    replacements = {"a(i)": "input", "d(i)": "denom", "output(i)": "value"}
    for original, replacement in replacements.items():
        body = body.replace(original, replacement)
    source = f"""module {helper_module}
implicit none
contains
pure function {helper}(input,denom) result(value)
real({precision_bits // 8}),intent(in)::input,denom
real({precision_bits // 8})::value
{declarations}
{body}
end function
end module
"""
    return helper_module, helper, source


def _lower(name, body, declarations, precision_bits, *, helper_source="", helper_module="", helper=""):
    # The numerical frontend currently follows same-module helpers. Normalize
    # only module placement in this analysis equivalent; compilation below
    # retains the real separate module and its original scalar argument ABI.
    helper_procedure = helper_source.split("contains\n", 1)[1].rsplit("end module", 1)[0] if helper_source else ""
    numerical = (
        (f"input_value=a(i)\ndenominator_value=d(i)\noutput(i)={helper}(input_value,denominator_value)")
        if helper_source
        else body
    )
    source = f"""module m_{name}
implicit none
contains
subroutine evaluate(n,a,d,output)
integer,intent(in)::n
real({precision_bits // 8}),intent(in)::a(:),d(:)
real({precision_bits // 8}),intent(out)::output(:)
integer::i
{declarations}
do i=1,n
{numerical}
enddo
end subroutine
{helper_procedure}
end module
"""
    function = lower_source(source, f"m_{name}::evaluate", source_name=name + "-authority.f90")
    (loop,) = function.body.statements
    if not isinstance(loop, Loop) or any(not isinstance(s, Assignment) for s in loop.body.statements):
        raise ValueError("fixture did not lower to one exact straight-line work item")
    # Use the actual scheduling/address preparation consumed by production
    # _cpu_worker, including logical coordinate and address expressions.
    function, plan = prepare_function(function, options=CompilerOptions(opt_level=0))
    if len(plan.steps) != 1 or not isinstance(plan.steps[0], ParallelRegion):
        raise ValueError("fixture did not prepare to one independent numerical region")
    region = plan.steps[0]
    graph = analyze_compute_dependencies(region)
    if not graph.available:
        raise ValueError("fixture dependency graph unavailable: " + str(graph.reason))
    return source, function, region, graph


def _native_sources(name, body, declarations, precision_bits, helper_form, helper_module, helper, helper_source):
    kind = "c_double" if precision_bits == 64 else "c_float"
    imports = f"use {helper_module},only:{helper}\n" if helper_form == "separate" else ""
    numerical = (
        (f"input_value=a(i)\ndenominator_value=d(i)\noutput(i)={helper}(input_value,denominator_value)")
        if imports
        else body
    )
    workers = []
    for role in ("serial", "fork_join"):
        opening = ""
        closing = ""
        if role == "fork_join":
            private = "i,x0,x1,x2,x3,t0,t1,input_value,denominator_value" + (
                ",small,other" if "small(" in declarations else ""
            )
            opening = (
                "!$omp parallel do num_threads(threads) schedule(static) default(none) &\n"
                f"!$omp shared(n,threads,a,d,output) &\n!$omp private({private})\n"
            )
            closing = "!$omp end parallel do\n"
        workers.append(f"""subroutine {name}_{role}(n,threads,a,d,output) bind(c)
integer(c_int),value::n,threads
real({kind}),intent(in)::a(n),d(n)
real({kind}),intent(out)::output(n)
integer::i
{declarations}
{opening}do i=1,n
{numerical}
enddo
{closing}end subroutine
""")
    caller = (
        f"module native_{name}\nuse iso_c_binding\n{imports}implicit none\ncontains\n"
        + "".join(workers)
        + "end module\n"
    )
    support = ((name + "_helper.f90", helper_source),) if helper_form == "separate" else ()
    return (*support, (name + ".f90", caller))


def _cpp_header():
    # Use the same indexing template and numeric support as production without
    # embedding unrelated allocation/storage units in every standalone fixture.
    source = files("compiler.emission.common").joinpath("templates/common_functions.cuh").read_text()
    template = source.split("// FORT_RUNTIME_UNITS", 1)[0] + "\n#endif\n"
    return template + '#include "numeric.hpp"\n#include <omp.h>\nusing namespace generated_kernels::indexing;\n'


def _cpp_sources(name, function, region):
    params = {symbol.name: symbol for symbol in function.parameters}
    n, a, d, output = (params[key] for key in ("n", "a", "d", "output"))
    real = cpp_type(a)
    signature = (
        f"int {n.cpp_name},int threads,const {real}*{a.cpp_name},const {real}*{d.cpp_name},{real}*{output.cpp_name}"
    )
    dims = "\n".join(
        f"const std::size_t {dimension_name(symbol, dimension)}="
        + (f"static_cast<std::size_t>({n.cpp_name})" if dimension == 1 else "4") + ";"
        for symbol in (a, d, output)
        for dimension in range(1, symbol.rank + 1)
    )
    arguments = abi_arguments(function.parameters)
    worker_signature = ", ".join(
        cpp_declaration(argument) if argument.symbol.rank else
        f"const {cpp_type(argument.symbol)} &{argument.name}" for argument in arguments
    )
    worker_name = name + "_cpu_worker"
    body = "\n".join(_cpu_worker(Unit(0, region, (), None), worker_signature, worker_name))
    call = ", ".join(argument.name for argument in arguments)
    worker = (
        _cpp_header()
        + body + "\n"
        + f'extern "C" void {name}_generated({signature}) {{\n'
        + dims
        + "\n#pragma omp parallel num_threads(threads)\n{\n"
        + f"{worker_name}({call}, omp_get_thread_num(), omp_get_num_threads());\n"
        + "}\n}\n"
    )
    return ((name + ".cpp", worker),)


def _recipe(kind, role, width, precision_bits, helper_form="inline"):
    name = f"dep_{kind}_w{width}_{helper_form}_p{precision_bits}"
    body = _body(kind, width, precision_bits)
    declarations = _declarations(kind, precision_bits)
    helper_module, helper, helper_source = _helper_source(name, body, declarations, precision_bits)
    # C++ uses the production worker for the source-inlined numerical closure.
    # The original native Fortran helper remains a separately compiled module
    # and is independently lowered to verify transitive numerical equivalence.
    inline_source, function, region, graph = _lower(name, body, declarations, precision_bits)
    if helper_form == "separate":
        analysis_source, _, _, transitive = _lower(
            name,
            body,
            declarations,
            precision_bits,
            helper_source=helper_source,
            helper_module=helper_module,
            helper=helper,
        )
        if (transitive.operations != graph.operations or transitive.outputs != graph.outputs or
                transitive.floating_dtypes != graph.floating_dtypes or
                transitive.unpriced_numerical_operations != graph.unpriced_numerical_operations):
            raise ValueError("separate helper changes the transitive numerical graph")
    else:
        analysis_source = inline_source
    family = kind[7:] if kind.startswith("domain_") else kind if role == "basis" else None
    domains = tuple(
        (primitive, contract)
        for primitive, contract in (
            ("acos", "finite_unit_interval"),
            ("sqrt", "finite_nonnegative_normal_or_zero"),
            ("cos", "finite_pi_interval"),
            ("divide_constant", "literal_three"),
        )
        if any(operation.family == primitive for operation in graph.operations)
    )
    native = _native_sources(
        name, body, declarations, precision_bits, helper_form, helper_module, helper, helper_source
    )
    cpp = _cpp_sources(name, function, region)
    dispatch_identity = _sha("\n".join(text for _, text in cpp))
    workload = (
        "fixed_private_array_v2"
        if kind == "private_arrays"
        else "ordinary_expression_v2"
        if kind == "ordinary_guarded"
        else "scalar_expression_v2"
    )
    identity = _sha(
        json.dumps(
            {
                "protocol": PROTOCOL_VERSION,
                "name": name,
                "body": body,
                "graph": graph.identity,
                "native": native,
                "cpp": cpp,
                "generated_dispatch_identity": dispatch_identity,
                "analysis_source": analysis_source,
                "role": role,
                "family": family,
                "width": width,
                "steps": BASIS_STEPS if role == "basis" else 1,
                "precision_bits": precision_bits,
                "helper_form": helper_form,
                "domains": domains,
                "workload_class": workload,
                "memory_arrays": 3,
                "input_lattices": [
                    _lattice(family if role == "domain_holdout" else None, precision_bits, split)
                    for split in ("training", "holdout")
                ],
                "denominator_lattices": _DENOMINATORS,
                "requirements": PROTOCOL_REQUIREMENTS,
            },
            separators=(",", ":"),
        )
    )
    return CpuDependencyRecipe(
        name,
        role,
        family,
        width,
        BASIS_STEPS if role == "basis" else 1,
        precision_bits,
        helper_form,
        body,
        graph,
        domains,
        workload,
        3,
        native,
        cpp,
        name + "_serial",
        name + "_fork_join",
        name + "_generated",
        analysis_source,
        region,
        identity,
        dispatch_identity,
    )


@lru_cache(maxsize=2, typed=True)
def recipes_for_precision(precision_bits: int) -> tuple[CpuDependencyRecipe, ...]:
    """Return the immutable, stable-order26recipe campaign before sampling."""
    if type(precision_bits) is not int or precision_bits not in {32, 64}:
        raise ValueError("dependency fixtures require precision32 or64")
    bases = tuple(_recipe(family, "basis", width, precision_bits) for family in BASIS_FAMILIES for width in (1, 4))
    structural = tuple(
        _recipe(kind, "structural_holdout", 2 if kind in {"diamond", "private_arrays"} else 1, precision_bits, form)
        for kind in STRUCTURAL_GRAPHS
        for form in ("inline", "separate")
    )
    domains = tuple(
        _recipe("domain_" + family, "domain_holdout", 1, precision_bits, form)
        for family in DOMAIN_FAMILIES
        for form in ("inline", "separate")
    )
    return bases + structural + domains


dependency_recipes = recipes_for_precision


def registry_source(recipes: tuple[CpuDependencyRecipe, ...], generator_id: str) -> str:
    """Emit the driver registry and periodic typed correctness references.

    Only initialization/checking uses the registry. Numerical workers contain
    no per-item recipe dispatch. Reference periods are the exact least common
    multiple of the two predetermined input lattices.
    """
    if (
        not isinstance(generator_id, str)
        or len(generator_id) != 64
        or any(char not in "0123456789abcdef" for char in generator_id)
    ):
        raise ValueError("fixture generator identity must be a SHA256")
    if (
        not isinstance(recipes, tuple)
        or not recipes
        or len(recipes) > 26
        or any(not isinstance(recipe, CpuDependencyRecipe) for recipe in recipes)
        or len({recipe.name for recipe in recipes}) != len(recipes)
        or len({recipe.precision_bits for recipe in recipes}) != 1
    ):
        raise ValueError("registry requires unique bounded recipes of one precision")
    precision = recipes[0].precision_bits
    real = "float" if precision == 32 else "double"
    lines = [
        "#pragma once",
        "#include <cstddef>",
        f"using dependency_real={real};",
        "using dependency_worker=void(*)(int,int,const dependency_real*,const dependency_real*,dependency_real*);",
        "struct dependency_recipe {",
        "const char*name;const char*identity;const char*role;const char*coefficient_family;int width;",
        "dependency_worker native_serial,native_fork_join,generated_cpu;",
    ]
    lines += [
        f"const dependency_real*{split}_{kind};std::size_t {split}_{kind}_count;"
        for split in ("training", "holdout")
        for kind in ("a", "d", "reference")
    ]
    lines += [
        "};",
        f"inline constexpr const char*dependency_generator_identity={json.dumps(generator_id)};",
        "inline constexpr int dependency_fit_sizes[]={65536,262144,1048576};",
        "inline constexpr int dependency_holdout_sizes[]={131072,524288};",
        'extern "C" void fort_cpu_dependency_fortran_identity_v1(char*,char*,int);',
    ]
    entries = []
    for index, recipe in enumerate(recipes):
        for entry in (recipe.native_serial_entry, recipe.native_fork_join_entry, recipe.generated_entry):
            lines.append(
                f'extern "C" void {entry}(int,int,const dependency_real*,const dependency_real*,dependency_real*);'
            )
        fields = [
            json.dumps(recipe.name),
            json.dumps(recipe.identity),
            json.dumps(recipe.role),
            json.dumps(recipe.coefficient_family or ""),
            str(recipe.width),
            recipe.native_serial_entry,
            recipe.native_fork_join_entry,
            recipe.generated_entry,
        ]
        for split_index, split in enumerate(("training", "holdout")):
            family = recipe.coefficient_family if recipe.role == "domain_holdout" else None
            a = tuple(_round(value, precision) for value in _lattice(family, precision, split))
            d = tuple(_round(value, precision) for value in _DENOMINATORS[split_index])
            period = math.lcm(len(a), len(d))
            reference = recipe.reference(recipe.inputs(period, split=split))
            for kind, values in (("a", a), ("d", d), ("reference", reference)):
                name = f"dependency_{index}_{split}_{kind}"
                literals = ",".join(float(value).hex() + ("f" if precision == 32 else "") for value in values)
                lines.append(f"inline constexpr dependency_real {name}[]={{" + literals + "};")
                fields.extend((name, str(len(values))))
        entries.append("{" + ",".join(fields) + "}")
    lines.append("inline constexpr dependency_recipe dependency_recipes[]={\n" + ",\n".join(entries) + "\n};")
    return "\n".join(lines) + "\n"


def write_dependency_registry(path, recipes: tuple[CpuDependencyRecipe, ...], generator_id: str):
    """Write only the explicit requested registry artifact."""
    from pathlib import Path

    Path(path).write_text(registry_source(recipes, generator_id))


def native_identity_source() -> str:
    """Compile with every original fixture's exact native semantic flags."""
    return """subroutine fort_cpu_dependency_fortran_identity_v1(version,options,capacity) bind(c)
use iso_c_binding,only:c_char,c_int,c_null_char
use iso_fortran_env,only:compiler_version,compiler_options
implicit none
character(kind=c_char),intent(out)::version(*),options(*)
integer(c_int),value::capacity
character(len=:),allocatable::native_version,native_options
integer::i,count
if(capacity<=0)return
native_version=compiler_version()
native_options=compiler_options()
count=min(len(native_version),capacity-1)
do i=1,count
version(i)=native_version(i:i)
enddo
version(count+1)=c_null_char
count=min(len(native_options),capacity-1)
do i=1,count
options(i)=native_options(i:i)
enddo
options(count+1)=c_null_char
end subroutine
"""
