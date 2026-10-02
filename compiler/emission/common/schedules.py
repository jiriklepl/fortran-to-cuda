"""Render size arithmetic and coordinate decoding from explicit schedules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from compiler.ir import RegionSchedule

if TYPE_CHECKING:
    from compiler.ir import ParallelRegion


def region_schedule(region: ParallelRegion) -> RegionSchedule:
    """Keep source order for callers using the original unscheduled plan API."""
    if region.schedule is not None:
        return region.schedule
    return RegionSchedule(tuple(reversed(range(len(region.loops)))))


def checked_product(name: str, factors: tuple[str, ...]) -> list[str]:
    """Check products before any launch or flattened iteration count narrows."""
    # Test zero factors first: an empty domain must not overflow due to other axes.
    active = " && ".join(f"({factor}) != 0" for factor in factors) or "true"
    lines = [f"std::size_t {name} = 0;", f"if ({active}) {{", f"    {name} = 1;"]
    for factor in factors:
        lines.extend(
            [
                f"    if ({name} > static_cast<std::size_t>(-1) / ({factor})) {{",
                '        std::cerr << "Iteration size product overflows size_t" << std::endl;',
                "        std::abort();",
                "    }",
                f"    {name} *= ({factor});",
            ]
        )
    lines.append("}")
    return lines


def tile_counts(region: ParallelRegion) -> list[str]:
    schedule = region_schedule(region)
    return [
        f"const std::size_t fort_internal_tiles{axis} = fort_internal_extent{axis} == 0 ? 0 : "
        f"(fort_internal_extent{axis} - 1) / {size}ULL + 1;"
        for axis, size in enumerate(schedule.tile_sizes)
    ]
