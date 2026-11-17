"""Opt-in ordinary-call policies; all numerical workers use checked compiler IR."""


from compiler.emission.common.abi import dimension_name
from compiler.emission.common.c_family import cpp_type, indent
from compiler.emission.common.loops import mapped_coordinates, mapped_snapshots, sequential_block
from compiler.emission.common.schedules import region_schedule
from compiler.emission.common.symbols import region_body
from compiler.ir import ScalarType
from compiler.offload.codegen import query_expression as _value


def _precision(function):
    kinds = {s.dtype for s in function.parameters if s.dtype in {ScalarType.REAL, ScalarType.REAL32}}
    return 32 if kinds == {ScalarType.REAL32} else 64 if kinds <= {ScalarType.REAL} else 0


def _host_profile_compatibility(host_threads, precision):
    # A native-only preview must not initialize CUDA, but its CPU estimates
    # still require the calibrated host and compiled toolchain identities.
    return ["static bool scoped_host_compatible(const offload::Profile &profile) {",
              f"    if (!profile.valid || profile.threads != {host_threads} || profile.precision != {precision}) return false;",
              "    const auto &cpu_name = offload::cpu_identity();",
              '    const auto host = std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);',
              '    const auto cuda = std::to_string(__CUDACC_VER_MAJOR__) + "." + std::to_string(__CUDACC_VER_MINOR__) + "." + std::to_string(__CUDACC_VER_BUILD__);',
              "    return !cpu_name.empty() && cpu_name == profile.cpu_name && host == profile.host_compiler && cuda == profile.cuda_compiler;",
              "}"]


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


def _metadata(function, analysis, signature, name, value=_value, *, host_pointer=None,
              cost_footprints=True):
    arrays = [s for s in function.parameters if s.rank]
    array_indices = {s: i for i, s in enumerate(arrays)}
    lines = [f"static offload::Data {name}({signature}) {{", "    offload::Data d;"]
    for symbol in arrays:
        dims = ", ".join(dimension_name(symbol, a + 1) for a in range(symbol.rank))
        pointer = symbol.cpp_name if host_pointer is None else host_pointer(symbol)
        lines += ["    {", "        offload::Array a;",
                  f"        a.host = const_cast<void*>(static_cast<const void*>({pointer}));",
                  f"        a.element_bytes = sizeof({cpp_type(symbol)}); a.dimensions = {{{dims}}};",
                  "        a.bytes = a.element_bytes;",
                  "        for (auto extent : a.dimensions) if (!offload::mul(a.bytes, extent, a.bytes)) d.valid = false;",
                  "        if (a.bytes && !a.host) d.valid = false;",
                  "        d.arrays.push_back(a);", "    }"]
    for unit in analysis.units:
        lines += ["    {", "        offload::Unit u;" + (" u.arrays.resize(d.arrays.size());" if cost_footprints else ""),
                  "        bool active = d.valid; std::size_t points = 1;"]
        for loop in unit.region.loops:
            stride = str(loop.step) + ".0L" if isinstance(loop.step, int) else value(loop.step)
            lines += ["        if (active) {",
                      f"            const auto lo = offload::index({value(loop.lower)}, d.valid);",
                      f"            const auto hi = offload::index({value(loop.upper)}, d.valid);",
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
                  *([f"        u.cpu_numerical_seconds = static_cast<double>(u.iterations) * {unit.cpu_numerical_seconds_per_iteration!r};",
                     f"        u.gpu_numerical_seconds = static_cast<double>(u.iterations) * {unit.gpu_numerical_seconds_per_iteration!r};"]
                    if unit.cpu_numerical_seconds_per_iteration or unit.gpu_numerical_seconds_per_iteration else []),
                  "        d.units.push_back(u);", "    }"]
    if not cost_footprints:
        # Definition-only scoped queries emit their exact access descriptors
        # separately. Keep layout/domain validation and numerical costs, but do
        # not construct a second set of boxes whose byte estimates are unused.
        return [*lines, "    return d;", "}"]
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
                                      f"                auto lo = offload::index({value(lo)}, d.valid);",
                                      f"                auto hi = offload::index({value(hi)}, d.valid);",
                                      f"                if (lo < 1 || hi < lo || static_cast<unsigned long long>(hi) > d.arrays[{idx}].dimensions[{axis}]) d.valid = false;",
                                      "                b.lower.push_back(lo > 0 ? lo-1 : 0); b.upper.push_back(hi > 0 ? hi-1 : 0);", "            }"]
                        lines += [f"            if (d.valid) offload::append_box({target}, b);", "        }"]
                lines += ["    }"]
        # Exact unions are separate for reads and writes, matching the triad
        # calibration's two reads plus one write. Repeated stencil loads share
        # a read footprint; read/modify/write storage still owes both directions.
        # This is a traffic floor, not a cache or write-allocation model.
        # Cost fragmentation does not invalidate legal layouts/effects or
        # forced execution. Nonfinite work already declines AUTO estimates
        # independently of definition-only validation in the scoped planner.
        lines += ["    {", "        bool memory_known = true;",
                  "        for (std::size_t a=0; a<d.arrays.size(); ++a) {",
                  f"            d.units[{unit.index}].memory_bytes += offload::footprint_bytes(d.arrays[a], d.units[{unit.index}].arrays[a].upload, memory_known);",
                  f"            d.units[{unit.index}].memory_bytes += offload::footprint_bytes(d.arrays[a], d.units[{unit.index}].arrays[a].download, memory_known);", "        }",
                  f"        if (!memory_known) d.units[{unit.index}].memory_bytes = std::numeric_limits<double>::quiet_NaN();", "    }"]
    return [*lines, "    return d;", "}"]

