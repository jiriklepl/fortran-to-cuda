"""Compose bounded direct numerical leaves into one full-layout batch callback."""

from dataclasses import replace

from fparser.two import Fortran2003 as F

from compiler.emission.common.c_family import cpp_type
from compiler.emission.cuda.batch import (
    STORAGE,
    attempt_helper,
    model_preparation,
    window_signature,
)
from compiler.emission.cuda.offload import _host_profile_compatibility, _precision
from compiler.emission.fortran.formatting import _fortran_list
from compiler.ir import Block, CompilationError, FunctionIR, SourceLocation, Symbol
from compiler.offload.analysis import Interval, _merge_footprints, scope_slab_candidates, scope_slab_plan
from compiler.offload.profile import ProfileError, scoped_costs, scoped_transfer_costs
from compiler.scopes.numerical import resource_binding
from compiler.transforms.fusion import _substitute


def compose(builder, calls):
    """Map only whole-storage direct leaves; no name-based application rules."""
    canonical, actuals, units, entries, definitions, unit_ids = {}, {}, [], [], {}, {}
    for call in calls:
        sources = builder.numerical(call.procedure)
        if sources is None:
            raise CompilationError("cross-entry batching requires direct numerical leaves")
        function, plan = builder.numerical_ir[call.procedure]
        analysis, _ = scope_slab_candidates(function, plan)
        public, _directory = builder.entry_artifacts(call.procedure)
        if not analysis.available or not public["batch_execution"]["window_entry"]:
            raise CompilationError(analysis.reason or "numerical entry has no proved window worker")
        if entries and _precision(function) != _precision(entries[0][1]):
            raise CompilationError("cross-entry batch precision differs")
        routine = builder.analysis.routines[call.procedure]
        package = builder.packages.get(call.procedure)
        parameters = {item.name: item for item in package.parameters} if package else {}
        mapping = {}
        for symbol in function.parameters:
            parameter = parameters.get(symbol.name.lower())
            root = parameter.resource if parameter else "argument::" + symbol.name.lower()
            if root.startswith("argument::"):
                binding = call.bindings.get(root)
                if binding is None:
                    raise CompilationError("cross-entry batching requires source-backed scalar actual mappings")
                root = binding.root
            if parameter and parameter.lower_bound_dimension is not None:
                key = (root, "lower_bound", parameter.lower_bound_dimension)
                actual = ("lower_bound", root, parameter)
            else:
                key = (root, "value")
                actual = ("resource", root, None)
                if parameter and not symbol.rank:
                    binding = resource_binding(builder.analysis, routine, parameter.resource)
                    if "parameter" in binding.attributes and binding.root.startswith(routine.qualified + "::"):
                        value = routine.scope.kinds.integer(F.Name(binding.name), SourceLocation(str(routine.scope.path)))
                        key = ("constant", value, symbol.dtype.value)
                        actual = ("constant", str(value), None)
            if key not in canonical:
                canonical[key] = Symbol(len(canonical)+1, "resource_" + str(len(canonical)), symbol.dtype,
                                        symbol.rank, "inout" if symbol.rank else "in", True)
                actuals[canonical[key]] = actual
            replacement = canonical[key]
            if (replacement.rank, replacement.dtype) != (symbol.rank, symbol.dtype):
                raise CompilationError("cross-entry canonical resource types differ")
            mapping[symbol] = replacement
        start = len(units)
        events = []
        # A normalized worker does not own its source entry's OUT events.
        for argument in routine.arguments:
            binding = routine.scope.bindings[argument]
            if binding.rank and binding.intent == "out":
                root = call.bindings["argument::" + argument].root
                target = next((symbol for key, symbol in canonical.items() if key == (root, "value")), None)
                if target is None:
                    raise CompilationError("batch mapping omits an original definition event")
                events.append(target)
        definitions[start] = tuple(events)
        for original in analysis.units:
            mapped = _substitute(original, mapping)
            index = len(units)
            mapped = replace(mapped, index=index, region=replace(mapped.region, id=index),
                             footprints=_merge_footprints(mapped.footprints))
            units.append(mapped)
            unit_ids[index] = public["planning"]["units"][original.index]["id"]
        entries.append((call, function, public, mapping, start, len(units)))
    parameters = (*[symbol for symbol in canonical.values() if symbol.rank],
                  *[symbol for symbol in canonical.values() if not symbol.rank])
    slab, reason = scope_slab_plan(tuple(units), parameters)
    if slab is None:
        raise CompilationError(reason)
    function = FunctionIR("batch_chain", "generated_scope", parameters, parameters, Block(()), "source-backed direct chain")
    return function, tuple(units), entries, definitions, unit_ids, slab, actuals


def emit_artifacts(builder, calls, identity, *, terminal=False):
    """Emit one host coordinator; every leaf's existing kernels remain shared."""
    from compiler.scopes.source import _name, _span
    function, units, entries, definitions, unit_ids, slab, actuals = compose(builder, calls)
    try:
        costs = scoped_costs(builder.config.profile, builder.runtime["runtime_id"])
        transfers = scoped_transfer_costs(builder.config.profile, builder.runtime["runtime_id"])
    except (ProfileError, TypeError, KeyError) as error:
        raise CompilationError("cross-entry batch transfer costs unavailable: " + str(error)) from error
    name = _name("fort_scope_batch_", identity)
    c_name = "cpp_" + name
    prepare, callback = c_name + "_prepare", c_name + "_worker"
    arrays = [symbol for symbol in function.parameters if symbol.rank]
    scalars = [symbol for symbol in function.parameters if not symbol.rank]
    interval = Interval(0, len(units), _merge_footprints(fp for unit in units for fp in unit.footprints))
    text = ['#include <cuda_runtime.h>', '#include <memory>', '#include <vector>',
            '#define FORT_OFFLOAD_ENABLED 1', '#include "common_functions.cuh"', '#include "scoped_runtime.h"',
            f"namespace generated_batches::{name} {{", "using namespace generated_kernels;", *STORAGE]
    for _call, original, public, _mapping, _start, _stop in entries:
        text += [f'extern "C" int {public["batch_execution"]["window_entry"]}({",".join(window_signature(original,""))});']
    written = {}
    if terminal:
        for position, unit in enumerate(units):
            for symbol in definitions.get(position, ()):
                written.pop(symbol, None)
            for footprint in unit.footprints:
                if footprint.writes:
                    written.setdefault(footprint.symbol, []).append(replace(footprint, reads=(), full_read=False))
    exports = _merge_footprints(fp for footprints in written.values() for fp in footprints)
    text += model_preparation(function, units, prepare, unit_ids, definition_events=definitions, exports=exports)
    state = callback + "_state"
    text += [f"struct {state} {{", "    std::size_t first,stop; unsigned int axis;",
             *[f"    fort_buffer_t {symbol.cpp_name}_handle;" for symbol in arrays],
             *[f"    const {cpp_type(symbol)} *fort_scalar_{symbol.cpp_name};" for symbol in scalars], "};",
             f"static int {callback}(const fort_scope_batch_window *window,void *user,uint64_t *launches) {{",
             f"    auto &state=*static_cast<{state} *>(user);"]
    for _call, original, public, mapping, _start, _stop in entries:
        arguments = [*["state." + mapping[symbol].cpp_name + "_handle" for symbol in original.parameters if symbol.rank],
                     *["state.fort_scalar_" + mapping[symbol].cpp_name for symbol in original.parameters if not symbol.rank]]
        count = len(public["planning"]["units"])
        text += [f"    if (const int status={public['batch_execution']['window_entry']}(window,0,{count}ULL,state.axis,launches,{','.join(arguments)})) return status;"]
    text += ["    return FORT_SCOPE_OK;", "}"]
    text += [*_host_profile_compatibility(builder.config.host_threads, _precision(entries[0][1])), "",
             *attempt_helper(function, ((interval, slab, None),), c_name, prepare, callback,
                             builder.config.profile, costs, transfers, builder.config.host_threads,
                             _precision(entries[0][1])), "}"]
    fortran = [f"module {name}", "use iso_c_binding", "implicit none", "interface"]
    arguments = ["context", "mode", "first", "stop", *[symbol.cpp_name for symbol in function.parameters]]
    fortran += _fortran_list("function run(", arguments, f") bind(C,name='{c_name}') result(status)", 0)
    fortran += ["import :: c_int,c_int64_t,c_size_t,c_double,c_float,c_bool", "integer(c_int)::status",
                "integer(c_int64_t),value::context", "integer(c_int),value::mode", "integer(c_size_t),value::first",
                "integer(c_size_t),intent(out)::stop"]
    types = {"integer": "integer(c_int)", "real": "real(c_double)", "real32": "real(c_float)", "logical": "logical(c_bool)"}
    for symbol in function.parameters:
        fortran += [f"integer(c_int64_t),value::{symbol.cpp_name}" if symbol.rank else
                    f"{types[symbol.dtype.value]},intent(in)::{symbol.cpp_name}"]
    fortran += ["end function", "end interface", "end module", ""]
    directory = "batches/" + name
    builder.outputs[directory + "/scope_batch.cu"] = "\n".join(text) + "\n"
    builder.outputs[directory + "/scope_batch.f90"] = "\n".join(fortran)
    from compiler.emission.common.resources import read_common_header
    builder.outputs[directory + "/common_functions.cuh"] = read_common_header()
    builder.outputs[directory + "/scoped_runtime.h"] = builder.runtime_outputs["scoped_runtime.h"]
    public = {"abi_version": 1, "eligible": True, "reason": None, "entry": c_name, "fortran_module": name,
              "fortran_procedure": "run", "first_line": _span(calls[0].node)[0], "last_line": _span(calls[-1].node)[1],
              "calls": [call.procedure for call in calls], "units": len(units), "slab": slab.to_dict(),
              "artifacts": [directory + "/scope_batch.cu", directory + "/scope_batch.f90"],
              "resource_mappings": [{"parameter": symbol.name, "resource": actuals[symbol][1],
                                     "kind": actuals[symbol][0]} for symbol in function.parameters],
              "execution": "complete ordered direct-leaf chain per GPU window",
              "publication": "exact final written sections before final owner close" if terminal else
                             "original native boundaries or owner close",
              "terminal_exports": [{"resource": actuals[footprint.symbol][1], "rectangles": len(footprint.writes)}
                                   for footprint in exports],
              "kernels": "reuse numerical entry window workers"}
    return name, function, actuals, public


def execute_calls(builder, calls, mode, handles, parameters, actuals, imports, ordinary, *, terminal=False):
    """Try maximal bounded direct-leaf runs, retaining untouched ordinary paths."""
    from compiler.scopes.source import _name, _span
    result, position = [], 0
    while position < len(calls):
        stop = position
        while stop < len(calls) and builder.numerical(calls[stop].procedure):
            stop += 1
        group = calls[position:stop]
        if len(group) < 2:
            result += ordinary(calls[position:position+1])
            position += 1
            continue
        identity = f"{builder.entry.qualified}:{_span(group[0].node)}:{_span(group[-1].node)}"
        if identity not in builder.batch_chains:
            try:
                builder.batch_chains[identity] = emit_artifacts(builder, group, identity, terminal=terminal and stop==len(calls))
            except CompilationError as error:
                builder.batch_chains[identity] = (None, None, None,
                    {"eligible": False, "reason": str(error), "first_line": _span(group[0].node)[0],
                     "last_line": _span(group[-1].node)[1], "calls": [call.procedure for call in group]})
        name, function, arguments, _public = builder.batch_chains[identity]
        if name is None:
            result += ordinary(group)
        else:
            alias = _name("fort_batch_run_", identity)
            imports.append(f"use {name}, only: {alias} => run")
            values = []
            for symbol in function.parameters:
                kind, root, detail = arguments[symbol]
                if symbol.rank:
                    values.append(handles[root])
                elif kind == "constant":
                    values.append(root + "_c_int")
                else:
                    visible = parameters.get(root, builder.visible(builder.entry, root))
                    values.append(builder.lower_bound_actual(detail, visible) if kind == "lower_bound" else
                                  f"logical({visible},kind=c_bool)" if symbol.dtype.value == "logical" else visible)
            result += ["block", "integer(c_size_t)::fort_batch_stop"]
            result += _fortran_list("fort_status = " + alias + "(", ["fort_context", mode, "0_c_size_t", "fort_batch_stop", *values], ")", 0)
            result += ["if (fort_status /= FORT_SCOPE_OK) error stop 'source GPU batch failed after preflight'",
                       "if (fort_batch_stop == 0) then", *ordinary(group), "endif", "end block"]
        position = stop
    return result
