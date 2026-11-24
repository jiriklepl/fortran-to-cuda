"""The host worker of a region; it uses the same checked compiler IR as the kernel."""


from compiler.emission.common.c_family import cpp_type, indent
from compiler.emission.common.loops import mapped_coordinates, sequential_block
from compiler.emission.common.schedules import region_schedule
from compiler.emission.common.symbols import region_body


def _cpu_worker(unit, signature, name):
    region = unit.region
    bound_params = ", ".join(
        f"int fort_internal_lower{a}, int fort_internal_stride{a}, std::size_t fort_internal_extent{a}"
        for a in range(len(region.loops))
    )
    extra = (f", {bound_params}, std::size_t fort_internal_total" if bound_params else ", std::size_t fort_internal_total")
    lines = [
        f"static void {name}({signature}{extra}, int tid, int team) {{",
        "    const std::size_t chunk = (fort_internal_total + static_cast<std::size_t>(team) - 1) / static_cast<std::size_t>(team);",
        "    const std::size_t start = static_cast<std::size_t>(tid) * chunk;",
        "    const std::size_t stop = start < fort_internal_total ? (fort_internal_total - start < chunk ? fort_internal_total : start + chunk) : start;",
        "    for (std::size_t flat = start; flat < stop; ++flat) {",
        "        std::size_t remainder = flat;",
    ]
    axis_order = region_schedule(region).axis_order
    for position, axis in enumerate(axis_order):
        if position + 1 == len(axis_order):
            lines += [f"        const std::size_t fort_internal_ordinal{axis} = remainder;"]
        else:
            lines += [f"        const std::size_t fort_internal_ordinal{axis} = remainder % fort_internal_extent{axis};",
                      f"        remainder /= fort_internal_extent{axis};"]
    lines += indent(mapped_coordinates(region), 2)
    lines += indent([f"{cpp_type(s)} {s.cpp_name};" for s in region.private_symbols], 2)
    lines += sequential_block(region_body(region), 2, [0], addressing=region.addressing)
    return [*lines, "    }", "}"]
