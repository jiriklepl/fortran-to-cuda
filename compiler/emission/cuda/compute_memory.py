"""Emit bounded memory pricing from physical working sets, without extrapolation.

The caller supplies checked traffic bytes separately from the size of the
physical union of accessed resources. Repeated accesses increase traffic, not
the working set. This helper provides no source access or alias proof.
"""

import math
import re

COMPUTE_MEMORY_INCLUDES = ("#include <cmath>", "#include <cstddef>",
                           "#include <cstdint>", "#include <limits>")
_MAX_COORDINATE = (1 << 64) - 1


def _piecewise_model(model):
    if not isinstance(model, dict) or model.get("kind") != "piecewise_bandwidth_v1":
        return None
    knots = model.get("knots")
    if not isinstance(knots, (list, tuple)) or not 2 <= len(knots) <= 8:
        return None
    coordinates, rates = [], []
    for row in knots:
        if not isinstance(row, dict):
            return None
        coordinate, rate = row.get("working_set_bytes"), row.get("seconds_per_traffic_byte")
        if type(coordinate) is not int or not 0 < coordinate <= _MAX_COORDINATE:
            return None
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            return None
        try:
            rate = float(rate)
        except (OverflowError, ValueError):
            return None
        if not math.isfinite(rate) or rate <= 0:
            return None
        coordinates.append(coordinate)
        rates.append(rate)
    bounds = model.get("working_set_range")
    if (coordinates != sorted(set(coordinates)) or not isinstance(bounds, (list, tuple))
            or len(bounds) != 2 or any(type(value) is not int for value in bounds)
            or list(bounds) != [coordinates[0], coordinates[-1]]):
        return None
    return coordinates, rates


def generate_compute_memory(name, memory_cost_model):
    """Return a standalone C++ bool helper; include COMPUTE_MEMORY_INCLUDES first.

    The generated signature is ``bool name(double traffic_bytes,
    std::size_t working_set_bytes, double &seconds) noexcept``. False means an
    unavailable estimate and always leaves seconds zero. Malformed models emit
    a helper that returns false; unsupported constant models remain available
    through the caller's legacy path. The supplied function name must be a C++
    identifier. Definitions may be placed inside the generated entry namespace.
    """
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError("memory helper requires a C++ identifier")
    lines = [f"static bool {name}(double traffic_bytes, std::size_t working_set_bytes, double &seconds) noexcept {{",
             "    seconds = 0.0;"]
    model = _piecewise_model(memory_cost_model)
    if model is None:
        return tuple((*lines, "    (void)traffic_bytes; (void)working_set_bytes;", "    return false;", "}"))
    coordinates, rates = model
    lines += ["    if (!std::isfinite(traffic_bytes) || traffic_bytes < 0.0) return false;",
              "    static constexpr std::uint64_t coordinates[] = {" +
              ", ".join(str(value) + "ULL" for value in coordinates) + "};",
              "    static constexpr double rates[] = {" + ", ".join(value.hex() for value in rates) + "};",
              f"    constexpr std::size_t count = {len(coordinates)};",
              "    if (coordinates[count-1] > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) return false;",
              "    const std::uint64_t working_set = static_cast<std::uint64_t>(working_set_bytes);",
              "    if (working_set < coordinates[0] || working_set > coordinates[count-1]) return false;",
              "    double rate = 0.0;",
              "    for (std::size_t i=0; i+1<count; ++i) {",
              "        if (working_set > coordinates[i+1]) continue;",
              "        if (working_set == coordinates[i]) rate = rates[i];",
              "        else if (working_set == coordinates[i+1]) rate = rates[i+1];",
              "        else {",
              "            const double fraction = static_cast<double>(working_set-coordinates[i]) /",
              "                                    static_cast<double>(coordinates[i+1]-coordinates[i]);",
              "            rate = rates[i] + fraction * (rates[i+1]-rates[i]);",
              "        }",
              "        break;",
              "    }",
              "    if (!std::isfinite(rate) || rate <= 0.0) return false;",
              "    if (rate > 1.0 && traffic_bytes > std::numeric_limits<double>::max()/rate) return false;",
              "    const double result = traffic_bytes * rate;",
              "    if (!std::isfinite(result) || (traffic_bytes > 0.0 && result <= 0.0)) return false;",
              "    seconds = result;",
              "    return true;", "}"]
    return tuple(lines)
