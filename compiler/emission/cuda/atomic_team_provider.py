"""One preflighted original numerical alternative on one device transaction.

Normal source lowering, IEEE checks and exact physical/read proofs are required.
Every reached unit's geometry and access union is prepared before source_execute
can acquire device views.  Its non-reentrant callback only submits already
prepared kernels on the supplied stream; there is no per-unit CPU fallback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from compiler.emission.common.abi import abi_arguments
from compiler.emission.common.c_family import cpp_type
from compiler.emission.common.schedules import region_schedule
from compiler.emission.common.symbols import region_symbols
from compiler.emission.cuda.kernels import generate_kernel, kernel_name
from compiler.frontend.atomic_worksharing_team import AtomicWorksharingAlternative
from compiler.ir import CompilationError, ScalarType


@dataclass(frozen=True, slots=True)
class AtomicParameter:
    binding: object
    name: str
    kind: str
    axis: int | None


@dataclass(frozen=True, slots=True)
class AtomicTeamProvider:
    entry: str
    module: str
    cuda_source: str
    fortran_source: str
    fields: tuple
    parameters: tuple
    _record: str

    def public(self):
        return json.loads(self._record)


def emit_atomic_team_provider(alternative, prepared, effects, reads):
    from compiler.frontend.numerical_effects import NumericalEffectProof
    from compiler.frontend.numerical_preparation import PreparedOriginalRegion
    from compiler.frontend.reached_read_requirements import ReachedReadRequirementsProof

    if type(alternative) is not AtomicWorksharingAlternative:
        raise CompilationError("atomic provider requires its exact original alternative")
    if (type(prepared) is not PreparedOriginalRegion or type(effects) is not NumericalEffectProof
            or type(reads) is not ReachedReadRequirementsProof):
        raise CompilationError("atomic provider requires exact issued numerical, physical and read proofs")
    team, analysis = alternative.team, alternative.team._analysis
    selection, routine = team._nodes, team._routine
    alternative.validate(analysis, routine.qualified, selection)
    prepared.validate(analysis, routine.qualified, selection)
    if prepared.completion_proof is not alternative:
        raise CompilationError("atomic provider numerical preparation names another alternative")
    if prepared.requires_original_team_environment_preflight:
        raise CompilationError("atomic provider requires an independently qualified original-worker IEEE preflight")
    effects.validate(analysis, routine.qualified, selection, prepared)
    reads.validate(analysis, routine.qualified, selection, prepared)
    conditional_support, conditional_geometry, conditional_guard = "", [], []
    if prepared.conditional_context is not None:
        from compiler.emission.cuda.conditional_affine import conditional_prework
        conditional_support, conditional_geometry, conditional_guard = conditional_prework(alternative, prepared)
    units = prepared.offload.units
    if not units or len(units) != len(effects.steps):
        raise CompilationError("atomic provider requires exact complete numerical unit effects")
    function = prepared.function
    originals = {item.name: item for item in prepared.extraction.parameters}
    bindings = {item.root: item for item in prepared.extraction.bindings}
    arrays = tuple(symbol for symbol in function.parameters if symbol.rank)
    scalars = tuple(symbol for symbol in function.parameters if not symbol.rank)
    fields = tuple(bindings[originals[symbol.name].resource] for symbol in arrays)
    field_index = {binding.root: index for index, binding in enumerate(fields)}
    array_index = dict(zip(arrays, range(len(arrays)), strict=True))
    lower, scalar_by_root, parameters = {}, {}, []
    for symbol in scalars:
        original = originals[symbol.name]
        binding = bindings[original.resource]
        kind = {ScalarType.INTEGER: "c_int", ScalarType.REAL32: "c_float", ScalarType.REAL: "c_double"}.get(symbol.dtype)
        if kind is None:
            raise CompilationError("atomic provider scalar has an unsupported numerical ABI")
        parameters.append(AtomicParameter(binding, symbol.cpp_name, kind, original.lower_bound_dimension))
        if original.lower_bound_dimension is not None:
            lower[binding.root, original.lower_bound_dimension] = symbol.cpp_name
        else:
            scalar_by_root[binding.root] = symbol.cpp_name

    def bound(value):
        if value.kind == "literal":
            return f"number({value.value}LL)"
        if value.kind == "scalar" and value.resource in scalar_by_root:
            return f"number(*input_{scalar_by_root[value.resource]})"
        if value.kind == "lbound" and (value.resource, value.dimension) in lower:
            return f"number(*input_{lower[value.resource, value.dimension]})"
        if value.kind == "size" and value.resource in field_index:
            return f"extent(layouts[{field_index[value.resource]}].extents[{value.dimension-1}])"
        if value.kind == "unary" and value.operator in {"+", "-"}:
            child = bound(value.children[0])
            return child if value.operator == "+" else f"minus(number(0),{child})"
        if value.kind == "binary" and value.operator in {"+", "-", "*"}:
            operation = {"+": "plus", "-": "minus", "*": "times"}[value.operator]
            return f"{operation}({bound(value.children[0])},{bound(value.children[1])})"
        raise CompilationError("atomic provider has an unsupported original affine bound")

    entry = "fort_atomic_" + sha256((team.identity + prepared.identity).encode()).hexdigest()[:24]
    module = "fort_atomic_module_" + entry[-24:]
    kernel_sources, geometry, body, state_fields = [], [], [], []
    read_units = {item.index: item.sections for item in reads.units}
    for index, unit in enumerate(units):
        region = unit.region
        if region_schedule(region).tile_sizes:
            raise CompilationError("atomic provider first path requires ordinary untiled kernel schedules")
        kernel_sources += generate_kernel(region)
        state_fields += [f"int lower{index}[{len(region.loops)}]{{}}, stride{index}[{len(region.loops)}]{{}};",
                         f"std::size_t extents{index}[{len(region.loops)}]{{}}, total{index}=1;"]
        iterations = effects.steps[index].sections.resources[0].reads or effects.steps[index].sections.resources[0].writes
        # All physical rectangles retain this exact unit's original domain.
        # An empty unit is skipped before its later-axis metadata/scalar reads.
        if not iterations:
            raise CompilationError("atomic provider unit lost its original iteration guard")
        iterations = iterations[0].iterations
        if len(iterations) != len(region.loops):
            raise CompilationError("atomic provider original iteration rank differs")
        for axis, iteration in enumerate(iterations):
            geometry += [f"if (state.total{index}) {{", f"const auto lo={bound(iteration.lower)},hi={bound(iteration.upper)};",
                         "if (!lo.ok || !hi.ok || lo.value<INT32_MIN || lo.value>INT32_MAX || hi.value<INT32_MIN || hi.value>INT32_MAX) return fail(FORT_SCOPE_ARGUMENT);",
                         f"state.lower{index}[{axis}]=static_cast<int>(lo.value); state.stride{index}[{axis}]={iteration.step};",
                         f"state.extents{index}[{axis}]=({iteration.step}>0 ? hi.value>=lo.value : lo.value>=hi.value) ? "
                         f"static_cast<std::size_t>(({iteration.step}>0 ? hi.value-lo.value : lo.value-hi.value)+1):0;",
                         f"if (state.extents{index}[{axis}] && state.total{index}>SIZE_MAX/state.extents{index}[{axis}]) return fail(FORT_SCOPE_RESOURCE);",
                         f"state.total{index}*=state.extents{index}[{axis}];", "}"]
        geometry += [f"if (state.total{index}) {{", "++expected;"]
        for symbol in region_symbols(region):
            if not symbol.rank:
                geometry.append(f"state.{symbol.cpp_name}=*input_{symbol.cpp_name};")
        sections = effects.steps[index].sections
        exact_reads = {resource.resource: resource for resource in read_units[index].resources}
        for resource in sections.resources:
            field = field_index[resource.resource]
            geometry.append(f"state.used[{field}]=true;")
            source_reads = exact_reads.get(resource.resource)
            # Read proof already checked that exact original reads cover all
            # numerical preservation/upload envelopes.  Retain the exact union.
            for label, boxes in (("reads", () if source_reads is None else source_reads.reads),
                                 ("writes", resource.writes)):
                for rectangle in boxes:
                    geometry += ["{ OwnedBox box;"]
                    for axis, coordinate in enumerate(rectangle.axes, 1):
                        original_lower = "*input_" + lower[resource.resource, axis]
                        geometry += ["{", f"const auto first={bound(coordinate.lower)},last={bound(coordinate.upper)};",
                                     f"const auto lo=minus(first,number({original_lower})),hi=plus(minus(last,number({original_lower})),number(1));",
                                     f"if (!lo.ok || !hi.ok || lo.value<0 || hi.value<lo.value || static_cast<std::uint64_t>(hi.value)>layouts[{field}].extents[{axis-1}]) return fail(FORT_SCOPE_BOUNDARY);",
                                     "box.lower.push_back(static_cast<std::size_t>(lo.value)); box.upper.push_back(static_cast<std::size_t>(hi.value));", "}"]
                    geometry += [f"accesses[{field}].{label}.push_back(std::move(box));", "}"]
        geometry += ["}"]
        values = []
        for argument in abi_arguments(region_symbols(region)):
            symbol = argument.symbol
            if symbol.rank:
                field = array_index[symbol]
                values.append(f"views[{field}].extents[{argument.dimension-1}]" if argument.dimension else
                              f"static_cast<{cpp_type(symbol)}*>(views[{field}].device)")
            else:
                values.append("state." + symbol.cpp_name)
        values += [value for axis in range(len(region.loops)) for value in (
            f"state.lower{index}[{axis}]", f"state.stride{index}[{axis}]", f"state.extents{index}[{axis}]")]
        values.append(f"state.total{index}")
        body += [f"if (state.total{index}) {{", "*started=1;",
                 f"const auto blocks=static_cast<unsigned>(std::min<std::size_t>(65535,(state.total{index}-1)/256+1));",
                 f"{kernel_name(region)}<<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(" + ",".join(values) + ");",
                 "if (cudaGetLastError()!=cudaSuccess) return FORT_SCOPE_EXECUTION; ++*accepted;", "}"]
    arguments = ["fort_scope_t context", *[f"fort_buffer_t field{index}" for index in range(len(fields))],
                 *[f"const {cpp_type(symbol)}* input_{symbol.cpp_name}" for symbol in scalars],
                 "fort_scope_external_report_v1* report"]
    layout, view_checks = [], []
    writes = set(root for unit in team.public()["units"] for root in unit["writes"])
    for index, (symbol, binding) in enumerate(zip(arrays, fields, strict=True)):
        dtype = {ScalarType.INTEGER: "FORT_SCOPE_INTEGER32", ScalarType.REAL: "FORT_SCOPE_REAL64",
                 ScalarType.REAL32: "FORT_SCOPE_REAL32"}[symbol.dtype]
        for prior in range(index):
            if prepared.conditional_context is not None or binding.root in writes or fields[prior].root in writes:
                layout.append(f"if (field{index}==field{prior}) return fail(FORT_SCOPE_ARGUMENT);")
        layout += [f"if (const int status=fort_scope_layout_get(context,field{index},&layouts[{index}])) return fail(status);",
                   f"if (layouts[{index}].rank!={binding.rank} || layouts[{index}].type!={dtype} || layouts[{index}].element_bytes!={binding.kind}) return fail(FORT_SCOPE_ARGUMENT);",
                   f"bindings[{index}].buffer=field{index}; bindings[{index}].type={dtype}; state.handles[{index}]=field{index};",
                   f"bindings[{index}].rank={binding.rank}; bindings[{index}].element_bytes={binding.kind};"]
        if prepared.conditional_context is not None:
            layout.append(f"state.generations[{index}]=layouts[{index}].generation;")
            view_checks.append(f"if (views[{index}].generation!=state.generations[{index}]) return FORT_SCOPE_STALE;")
        for axis in range(binding.rank):
            layout.append(f"state.dimensions[{index}][{axis}]=layouts[{index}].extents[{axis}];")
        view_checks += [f"if (views[{index}].buffer!=state.handles[{index}] || views[{index}].rank!={binding.rank} || "
                        f"views[{index}].type!={dtype} || views[{index}].element_bytes!={binding.kind} || "
                        f"!views[{index}].extents || !views[{index}].strides || (state.used[{index}] && !views[{index}].device)) return FORT_SCOPE_ARGUMENT;",
                        f"{{ std::size_t stride={binding.kind};"]
        for axis in range(binding.rank):
            view_checks += [f"if (views[{index}].extents[{axis}]!=state.dimensions[{index}][{axis}] || views[{index}].strides[{axis}]!=stride) return FORT_SCOPE_ARGUMENT;",
                            f"if (views[{index}].extents[{axis}] && stride>SIZE_MAX/views[{index}].extents[{axis}]) return FORT_SCOPE_ARGUMENT;",
                            f"stride*=views[{index}].extents[{axis}];"]
        view_checks.append("}")
    source = ["#include <cuda_runtime.h>", "#include <cstdint>", "#include <climits>", "#include <limits>",
              "#include <algorithm>", "#include <vector>", "#include <new>", '#include "scoped_runtime.h"',
              '#include "common_functions.cuh"', '#include "numeric.hpp"', "using generated_kernels::indexing::F_IDX;",
              "namespace {", _SUPPORT, *([conditional_support] if conditional_support else []), "struct State {", *state_fields,
              f"bool used[{len(fields)}]{{}}; fort_buffer_t handles[{len(fields)}]{{}};",
              *([f"std::uint64_t generations[{len(fields)}]{{}};"] if prepared.conditional_context is not None else []),
              f"std::size_t dimensions[{len(fields)}][{max(binding.rank for binding in fields)}]{{}};",
              *[f"{cpp_type(symbol)} {symbol.cpp_name}{{}};" for symbol in scalars], "};", *kernel_sources,
              "static int submit(void* opaque,const fort_scope_external_view_v1* views,std::size_t count,void* stream,int* started,std::uint64_t* accepted) {",
              "*started=0; *accepted=0; auto& state=*static_cast<State*>(opaque);",
              f"if (count!={len(fields)}) return FORT_SCOPE_ARGUMENT;", *view_checks, *body, "return FORT_SCOPE_OK;", "}", "}",
              'extern "C" int ' + entry + "(" + ",".join(arguments) + ") {",
              "if (!report) return FORT_SCOPE_ARGUMENT; *report={1,FORT_SCOPE_ARGUMENT,0,0};",
              "auto fail=[&](int status) { report->status=status; return status; };",
              "try {", "if (!fort_scope_serial_caller() || !fort_scope_numerical_environment_supported()) return fail(FORT_SCOPE_BOUNDARY);",
              *[f"if (!input_{symbol.cpp_name}) return fail(FORT_SCOPE_ARGUMENT);" for symbol in scalars],
              "State state{}; std::uint64_t expected=0;",
              f"fort_scope_layout layouts[{len(fields)}]{{}}; OwnedAccess accesses[{len(fields)}];",
              f"fort_scope_external_binding_v1 bindings[{len(fields)}]{{}};", *layout, *conditional_geometry, *geometry,
              "if (!expected) return fail(FORT_SCOPE_OK);",
              *[f"accesses[{index}].finish(bindings[{index}].access);" for index in range(len(fields))],
              *conditional_guard,
              *([f"if (const int status=fort_scope_admit_source_reads_v1(context,field{index},bindings[{index}].access.reads,bindings[{index}].access.read_count)) return fail(status);"
                 for index in range(len(fields))] if prepared.conditional_context is not None else []),
              f"fort_scope_external_call_v1 call{{1,0,bindings,{len(fields)},0,0}};",
              "return fort_scope_source_execute_v1(context,&call,expected,submit,&state,report);",
              "} catch (const std::bad_alloc&) { return fail(FORT_SCOPE_RESOURCE); }",
              "catch (...) { return fail(FORT_SCOPE_EXECUTION); }", "}", ""]
    names = ["context", *[f"field{index}" for index in range(len(fields))], *[item.name for item in parameters], "report"]
    iso = {"c_int": "integer(c_int)", "c_float": "real(c_float)", "c_double": "real(c_double)"}
    fortran = [f"module {module}", "use iso_c_binding", "use fort_scoped_memory, only: fort_scope_external_report_v1", "implicit none", "interface",
               f"integer(c_int) function {entry}( &", " & " + ", &\n & ".join(names) + f') bind(c,name="{entry}")',
               "import", "integer(c_int64_t),value::context", *[f"integer(c_int64_t),value::field{index}" for index in range(len(fields))],
               *[f"{iso[item.kind]},intent(in)::{item.name}" for item in parameters],
               "type(fort_scope_external_report_v1),intent(out)::report", "end function", "end interface", "end module", ""]
    cuda, fortran = "\n".join(source), "\n".join(fortran)
    record = {"version": 3 if prepared.conditional_context is not None else 2, "kind": "atomic_original_team_alternative", "team_identity": team.identity,
              "alternative_identity": alternative.identity, "preparation_identity": prepared.identity,
              "physical_effect_identity": effects.identity, "source_read_identity": reads.identity,
              "entry": entry, "module": module, "cuda_sha256": sha256(cuda.encode()).hexdigest(),
              "fortran_sha256": sha256(fortran.encode()).hexdigest(), "units": len(units),
              "fields": [binding.public() for binding in fields],
              "parameters": [{"binding": item.binding.public(), "name": item.name,
                              "kind": item.kind, "axis": item.axis} for item in parameters],
              "completion": "one transaction at the original whole parent join; supplied-stream source order",
              "preparation": "all reached unit geometry and exact physical/source-read unions before submission",
              "fallback": "original whole team only before first attempted launch; no per-unit CPU fallback",
              "definitions_authorized": False, "placement_authorized": False, "pricing": "unpriced; explicit sections only",
              **({"conditional_numerical_context_identity": prepared.conditional_context.identity,
                  "conditional_guard_before_source_read_admission": True,
                  "zero_trip_unit": "decline whole original team before later bound evaluation",
                  "original_integer_prework": "reused checked original INT4 ranges and terminal values",
                  "guard_arithmetic": "checked INTEGER64; overflow declines before source-read admission",
                  "generation_snapshot_checked": True}
                 if prepared.conditional_context is not None else {})}
    return AtomicTeamProvider(entry, module, cuda, fortran, fields, tuple(parameters), json.dumps(record, sort_keys=True))


_SUPPORT = r"""
struct Number { bool ok; std::int64_t value; };
static Number number(std::int64_t value) { return {true,value}; }
static Number checked(__int128 value) { return {value>=INT64_MIN && value<=INT64_MAX,static_cast<std::int64_t>(value)}; }
static Number extent(std::size_t value) { return {value<=INT64_MAX,static_cast<std::int64_t>(value)}; }
static Number plus(Number a,Number b) { auto r=checked(static_cast<__int128>(a.value)+b.value);r.ok=r.ok&&a.ok&&b.ok;return r; }
static Number minus(Number a,Number b) { auto r=checked(static_cast<__int128>(a.value)-b.value);r.ok=r.ok&&a.ok&&b.ok;return r; }
static Number times(Number a,Number b) { auto r=checked(static_cast<__int128>(a.value)*b.value);r.ok=r.ok&&a.ok&&b.ok;return r; }
struct OwnedBox { std::vector<std::size_t> lower,upper; };
struct OwnedAccess {
 std::vector<OwnedBox> reads,writes;
 std::vector<fort_scope_section> read_views,write_views;
 void finish(fort_scope_access& access) {
  for (auto& box:reads) read_views.push_back({box.lower.data(),box.upper.data()});
  for (auto& box:writes) write_views.push_back({box.lower.data(),box.upper.data()});
  access.read_count=read_views.size();access.reads=read_views.data();
  access.write_count=write_views.size();access.writes=write_views.data();
  access.overwrite_count=write_views.size();access.overwrites=write_views.data();
 }
};
"""
