"""The host worker of a region; it uses the same checked compiler IR as the kernel."""


from compiler.emission.common.c_family import cpp_type, indent
from compiler.emission.common.loops import mapped_coordinates, mapped_snapshots, sequential_block
from compiler.emission.common.schedules import region_schedule
from compiler.emission.common.symbols import region_body


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
