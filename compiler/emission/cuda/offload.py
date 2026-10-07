"""Opt-in ordinary-call policies; all numerical workers use checked compiler IR."""

from dataclasses import dataclass
from hashlib import sha256

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.c.generator import cpp_plan_lines
from compiler.emission.common.abi import abi_arguments, dimension_name
from compiler.emission.common.c_family import cpp_type, indent
from compiler.emission.common.loops import mapped_coordinates, mapped_snapshots, sequential_block
from compiler.emission.common.schedules import region_schedule
from compiler.emission.common.symbols import host_symbols, region_body
from compiler.emission.cuda.kernels import generate_launch
from compiler.ir import ScalarType, referenced_symbols
from compiler.offload.analysis import analyze_offload
from compiler.offload.codegen import profile_expression
from compiler.offload.codegen import query_expression as _value


@dataclass(frozen=True)
class OffloadEmission:
    helpers: str
    body: list[str]
    decision_body: list[str]
    query_name: str
    query_scalars: frozenset
    report: dict


def _precision(function):
    kinds = {s.dtype for s in function.parameters if s.dtype in {ScalarType.REAL, ScalarType.REAL32}}
    return 32 if kinds == {ScalarType.REAL32} else 64 if kinds <= {ScalarType.REAL} else 0


def _cpu_worker(unit, signature, name):
    region = unit.region
    lines = [f"static void {name}({signature}, int tid, int team) {{", *indent(mapped_snapshots(region))]
    factors = " * ".join(f"fort_internal_extent{a}" for a in range(len(region.loops))) or "0"
    # Dispatch metadata checked the product; standalone fallback below uses the
    # ordinary compiler renderer for unsupported/invalid metadata.
    lines += [f"    const std::size_t total = {factors};",
              "    for (std::size_t flat = tid; flat < total; flat += team) {",
              "        std::size_t remainder = flat;"]
    for axis in region_schedule(region).axis_order:
        lines += [f"        const std::size_t fort_internal_ordinal{axis} = remainder % fort_internal_extent{axis};",
                  f"        remainder /= fort_internal_extent{axis};"]
    lines += indent(mapped_coordinates(region), 2)
    lines += indent([f"{cpp_type(s)} {s.cpp_name};" for s in region.private_symbols], 2)
    lines += sequential_block(region_body(region), 2, [0], addressing=region.addressing)
    return [*lines, "    }", "}"]


def _metadata(function, analysis, signature, name):
    arrays = [s for s in function.parameters if s.rank]
    array_indices = {s: i for i, s in enumerate(arrays)}
    lines = [f"static offload::Data {name}({signature}) {{", "    offload::Data d;"]
    for symbol in arrays:
        dims = ", ".join(dimension_name(symbol, a + 1) for a in range(symbol.rank))
        lines += ["    {", "        offload::Array a;",
                  f"        a.host = const_cast<void*>(static_cast<const void*>({symbol.cpp_name}));",
                  f"        a.element_bytes = sizeof({cpp_type(symbol)}); a.dimensions = {{{dims}}};",
                  "        a.bytes = a.element_bytes;",
                  "        for (auto extent : a.dimensions) if (!offload::mul(a.bytes, extent, a.bytes)) d.valid = false;",
                  "        if (a.bytes && !a.host) d.valid = false;",
                  "        d.arrays.push_back(a);", "    }"]
    for unit in analysis.units:
        lines += ["    {", "        offload::Unit u; u.arrays.resize(d.arrays.size());",
                  "        bool active = d.valid; std::size_t points = 1;"]
        for loop in unit.region.loops:
            stride = str(loop.step) + ".0L" if isinstance(loop.step, int) else _value(loop.step)
            lines += ["        if (active) {",
                      f"            const auto lo = offload::index({_value(loop.lower)}, d.valid);",
                      f"            const auto hi = offload::index({_value(loop.upper)}, d.valid);",
                      f"            const auto step = offload::index({stride}, d.valid);",
                      "            if (lo < INT_MIN || lo > INT_MAX || hi < INT_MIN || hi > INT_MAX ||",
                      "                !step || step < INT_MIN || step > INT_MAX) d.valid = false;",
                      "            std::size_t extent = 0;",
                      "            if (d.valid && step > 0 && hi >= lo) extent = (hi - lo) / step + 1;",
                      "            if (d.valid && step < 0 && lo >= hi) extent = (lo - hi) / -step + 1;",
                      "            active = d.valid && extent != 0;",
                      "            if (active && !offload::mul(points, extent, points)) d.valid = false;",
                      "        }"]
        lines += ["        u.iterations = active && d.valid ? points : 0;",
                  f"        u.flops = static_cast<double>(u.iterations) * {unit.work_per_iteration or 0};",
                  "        d.units.push_back(u);", "    }"]
    for unit in analysis.units:
        for footprint in unit.footprints:
            idx = array_indices[footprint.symbol]
            for direction, boxes, full in (("upload", footprint.uploads, footprint.full_upload),
                                           ("download", footprint.downloads, footprint.full_write)):
                target = f"d.units[{unit.index}].arrays[{idx}].{direction}"
                lines += [f"    if (d.valid && d.units[{unit.index}].iterations) {{"]
                if full:
                    lines += [f"        offload::Box b; b.lower.assign(d.arrays[{idx}].dimensions.size(), 0);",
                              f"        for (auto n : d.arrays[{idx}].dimensions) {{ if (!n) d.valid = false; b.upper.push_back(n ? n-1 : 0); }}",
                              f"        if (d.valid) offload::append_box({target}, b);"]
                else:
                    for box in boxes:
                        lines += ["        {", "            offload::Box b;"]
                        for axis, (lo, hi) in enumerate(zip(box.lower, box.upper, strict=True)):
                            lines += ["            {",
                                      f"                auto lo = offload::index({_value(lo)}, d.valid);",
                                      f"                auto hi = offload::index({_value(hi)}, d.valid);",
                                      f"                if (lo < 1 || hi < lo || static_cast<unsigned long long>(hi) > d.arrays[{idx}].dimensions[{axis}]) d.valid = false;",
                                      "                b.lower.push_back(lo > 0 ? lo-1 : 0); b.upper.push_back(hi > 0 ? hi-1 : 0);", "            }"]
                        lines += [f"            if (d.valid) offload::append_box({target}, b);", "        }"]
                lines += ["    }"]
        # Distinct touched bytes are a conservative memory-traffic floor. The
        # same estimate is used for both devices; uncertain work selects CPU.
        lines += ["    for (std::size_t a=0; a<d.arrays.size(); ++a) {",
                  f"        for (const auto &b : d.units[{unit.index}].arrays[a].upload) d.units[{unit.index}].memory_bytes += offload::box_bytes(d.arrays[a], b, d.valid);",
                  f"        for (const auto &b : d.units[{unit.index}].arrays[a].download) d.units[{unit.index}].memory_bytes += offload::box_bytes(d.arrays[a], b, d.valid);", "    }"]
    return [*lines, "    return d;", "}"]


def generate_offload(function, plan, config):
    analysis = analyze_offload(function, plan)
    digest = sha256(f"{function.module}::{function.name}".encode()).hexdigest()[:12]
    name = f"fort_offload_{digest}"
    query_name = f"{function.name[:40]}_offload_{digest[:8]}"
    abi = abi_arguments(function.parameters)
    signature = ", ".join(cpp_declaration(a) for a in abi)
    query_signature = ", ".join(cpp_declaration(a) if a.symbol.rank else
                                f"const {cpp_type(a.symbol)} &{a.name}" for a in abi)
    arguments = ", ".join(a.name for a in abi)
    scalars = set()
    for unit in analysis.units:
        for loop in unit.region.loops:
            values = (loop.lower, loop.upper) + (() if isinstance(loop.step, int) else (loop.step,))
            for expr in values:
                scalars.update(s for s in referenced_symbols(expr) if not s.rank)
        for footprint in unit.footprints:
            for box in (*footprint.uploads, *footprint.downloads):
                for expr in (*box.lower, *box.upper):
                    scalars.update(s for s in referenced_symbols(expr) if not s.rank)
    report = {"policy": config.policy, "analysis": analysis.to_dict(), "native_fallback_query": query_name,
              "collective_entry": config.collective, "host_threads": config.host_threads,
              "profile_available": config.profile is not None, "profile_reason": config.profile_reason}
    known = analysis.available and all(u.work_per_iteration is not None and not u.work_is_upper_bound for u in analysis.units)
    report["estimate_available"] = bool(known and config.profile is not None)
    report["work_per_iteration"] = [u.work_per_iteration for u in analysis.units]
    report["launches_max"] = len(analysis.units)
    report["transfer_volume"] = "runtime union of physical sections; unknown accesses use full arrays"
    report["supported_strategies"] = ["native"] + (["sections", "auto"] if analysis.available else []) + (["chunked", "hybrid"] if analysis.chunk is not None else [])
    fallback = [f"{cpp_type(s)} {s.cpp_name};" for s in host_symbols(function, plan)] + cpp_plan_lines(plan)
    serial_fallback = [line for line in fallback if not line.lstrip().startswith("#pragma omp")]
    if config.collective:
        fallback = ["#pragma omp single", "{", *indent(serial_fallback), "}"]
    else:
        fallback = [line.replace("#pragma omp parallel for", f"#pragma omp parallel for num_threads({config.host_threads})") for line in fallback]
        fallback = ["if (offload::in_parallel()) {", *indent(serial_fallback), "} else {", *indent(fallback), "}"]
    if not analysis.available:
        return OffloadEmission("", fallback,
            [f'offload::decision_trace("{function.name}", "native", 0, 0);', "return 0;"],
            query_name, frozenset(), report)
    if config.policy in {"hybrid", "chunked"}:
        from compiler.emission.cuda.hybrid import generate_hybrid
        hybrid = generate_hybrid(function, plan, analysis, config.profile, config.policy,
                                 host_threads=config.host_threads, collective=config.collective)
        report["hybrid_available"] = hybrid.available
        report["hybrid_reason"] = hybrid.reason
        if hybrid.available:
            return OffloadEmission(hybrid.helpers, hybrid.body, hybrid.decision_body,
                                   query_name, frozenset(scalars), report)
        return OffloadEmission("", fallback,
            [f'offload::decision_trace("{function.name}", "native", 0, {len(analysis.units)});', "return 0;"],
            query_name, frozenset(), report)
    lines = [f"static const offload::Profile {name}_profile = {profile_expression(config.profile)};"]
    lines += _metadata(function, analysis, query_signature, name + "_data")
    for unit in analysis.units:
        lines += _cpu_worker(unit, signature, name + f"_cpu_{unit.index}")
    arrays = [s for s in function.parameters if s.rank]
    lines += [f"static void {name}_gpu({signature}, const offload::Data& d, std::size_t begin, std::size_t end) {{",
              "    offload::DeviceScope device_scope(d.device);",
              f"    const auto footprints = offload::interval(d, begin, end, {name}_profile);",
              "    std::vector<std::unique_ptr<offload::Allocation>> buffers(d.arrays.size());"]
    for i, symbol in enumerate(arrays):
        lines += [f"    if (!footprints[{i}].upload.empty() || !footprints[{i}].download.empty())",
                  f"        buffers[{i}] = std::make_unique<offload::Allocation>(d.arrays[{i}].bytes);",
                  f"    auto *{symbol.cpp_name}_device = buffers[{i}] ? static_cast<{cpp_type(symbol)}*>(buffers[{i}]->get()) : nullptr;"]
    lines += ["    for (std::size_t a=0; a<buffers.size(); ++a)",
              "        for (const auto &b : footprints[a].upload) offload::copy_box(d.arrays[a], b, buffers[a]->get(), true);"]
    for unit in analysis.units:
        lines += [f"    if (begin <= {unit.index} && {unit.index} < end && d.units[{unit.index}].iterations) {{",
                  *indent(generate_launch(unit.region), 2), "    }"]
    lines += ["    CUCH(cudaDeviceSynchronize());",
              "    for (std::size_t a=0; a<buffers.size(); ++a)",
              "        for (const auto &b : footprints[a].download) offload::copy_box(d.arrays[a], b, buffers[a]->get(), false);",
              "    for (auto &buffer : buffers) if (buffer) buffer->release_completed();", "}"]
    context = f"offload::context_valid({config.host_threads}, {'true' if config.collective else 'false'})"
    compatible = f"offload::compatible({name}_profile, {config.host_threads}, {_precision(function)})"
    ready = f"{name}_profile.valid && {context}"
    if config.policy == "sections":
        ready = context
        compatible = "true"
    elif not known:
        ready = "false"
        report["estimate_reason"] = "work estimate is unknown or conditional"
    automatic = "true" if config.policy == "auto" else "false"
    decision = [f"if (!({ready})) {{",
                f'    offload::decision_trace("{function.name}", "native", 0, {len(analysis.units)}); return 0;', "}",
                f"const auto d = {name}_data({arguments});",
                f"auto selected = offload::select(d, {name}_profile, {automatic});",
                f"if (selected.has_gpu() && !({compatible})) selected = offload::native_plan(d);",
                f'if (!selected.has_gpu()) offload::decision_trace("{function.name}", "native", 0, {len(analysis.units)});',
                f'if (!selected.has_gpu()) offload::plan_trace("{function.name}", d, {name}_profile, selected);',
                "return selected.has_gpu() ? 1 : 0;"]
    lines += [f"static void {name}_team({signature}, offload::Data *prepared = nullptr) {{", "    offload::Data *shared = prepared;",
              "    #pragma omp single copyprivate(shared)", "    {",
              "        offload::DecisionRange decision_range;",
              "        if (!prepared) {",
              f"            shared = new offload::Data({name}_data({arguments}));",
              f"            shared->plan = offload::select(*shared, {name}_profile, {automatic});",
              f"            if (shared->plan.has_gpu() && !({compatible})) shared->plan = offload::native_plan(*shared);",
              "            if (shared->plan.has_gpu()) CUCH(cudaGetDevice(&shared->device));",
              "        }",
              f"        if (offload::team_size() != {config.host_threads}) shared->plan = offload::native_plan(*shared);",
              "    }",
              "    std::size_t gpu=0, cpu=0;",
              "    for (const auto &c : shared->plan.choices) (c.gpu ? gpu : cpu) += c.end-c.begin;",
              f'    offload::decision_trace("{function.name}", gpu ? (cpu ? "mixed" : "gpu") : "native", gpu, cpu);',
              f'    if (offload::thread_id() == 0) offload::plan_trace("{function.name}", *shared, {name}_profile, shared->plan);',
              "    for (const auto &choice : shared->plan.choices) {",
              "        if (choice.gpu) {", "            #pragma omp single", "            {",
              f"                {name}_gpu({arguments}, *shared, choice.begin, choice.end);", "            }",
              "        } else {", "            switch (choice.begin) {"]
    for unit in analysis.units:
        lines += [f"            case {unit.index}: {name}_cpu_{unit.index}({arguments}, offload::thread_id(), offload::team_size()); break;"]
    lines += ["            }", "            #pragma omp barrier", "        }", "    }",
              "    #pragma omp single", "    { if (!prepared) delete shared; }", "}"]
    body = ["#pragma omp barrier"] if config.collective else []
    body += [f"if (!({ready})) {{", *indent(fallback), "    return;", "}",
             f"auto check = {name}_data({arguments});", "if (!check.valid) {", *indent(fallback), "    return;", "}"]
    if config.collective:
        body += [f"{name}_team({arguments});"]
    else:
        body += ["{", "    offload::DecisionRange decision_range;",
                 f"    check.plan = offload::select(check, {name}_profile, {automatic});",
                 f"    if (check.plan.has_gpu() && !({compatible})) check.plan = offload::native_plan(check);",
                 "    if (check.plan.has_gpu()) CUCH(cudaGetDevice(&check.device));", "}",
                 f"#pragma omp parallel num_threads({config.host_threads})", "{",
                 f"    {name}_team({arguments}, &check);", "}"]
    return OffloadEmission("\n".join(lines), body, decision, query_name, frozenset(scalars), report)
