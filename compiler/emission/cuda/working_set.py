"""Checked unique array footprints for optional numerical cost applicability.

Traffic counts are intentionally separate. This geometry proves neither GPU
legality nor coherence, and a failed estimate must not invalidate either one.
"""

import re


def working_set_lines(handles, *, data="unit", unit_index=0, prefix="fort_working_set", exact=True):
    """Declare ``prefix_bytes`` and ``prefix_available`` in emitted C++.

    ``handles`` follow the metadata arrays' order and identify validated
    canonical runtime roots. Only distinct touched roots are admitted: logical
    rectangles from aliased formal views cannot be merged as physical boxes.
    Validated rectangular views are injective, so their cardinality is exact
    for distinct roots regardless of original pitches or retained axes.

    The caller must supply ``exact=False`` for unknown/full or conservative
    source footprints. Rectangles use offload::Data's zero-based inclusive
    logical coordinates. Read/write overlaps are united, never enclosed in a
    bounding volume. Input, output and union work retain the existing budgets.
    This code reads metadata only, leaves Data/coherence unchanged and reports
    unavailable on geometry, allocation or arithmetic failure.
    """
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", prefix):
        raise ValueError("working-set prefix must be a C++ identifier")
    if type(unit_index) is not int or unit_index < 0:
        raise ValueError("working-set unit index must be a nonnegative integer")
    if type(exact) is not bool:
        raise ValueError("working-set exactness must be an explicit boolean")
    handles = tuple(handles)
    if not all(isinstance(handle, str) and handle for handle in handles):
        raise ValueError("working-set roots require compiler-owned C++ expressions")
    lines = [f"std::size_t {prefix}_bytes = 0;"]
    if not exact:
        return tuple([*lines, f"const bool {prefix}_available = false;"])
    lines += [f"const bool {prefix}_available = [&](const offload::Data &metadata,",
              "    std::initializer_list<std::uint64_t> roots) {",
              f"    if (!metadata.valid || metadata.units.size() <= {unit_index}ULL ||",
              f"        metadata.arrays.size() != {len(handles)}ULL) return false;",
              f"    const auto &footprints = metadata.units[{unit_index}].arrays;",
              "    if (footprints.size() != metadata.arrays.size()) return false;",
              "    std::size_t total = 0;",
              "    try {",
              "        for (std::size_t a = 0; a < metadata.arrays.size(); ++a) {",
              "            const auto &fp = footprints[a];",
              "            if (fp.upload.empty() && fp.download.empty()) continue;",
              "            if (!roots.begin()[a]) return false;",
              "            for (std::size_t b = 0; b < a; ++b)",
              "                if (roots.begin()[a] == roots.begin()[b] &&",
              "                    (!footprints[b].upload.empty() || !footprints[b].download.empty())) return false;",
              "            if (fp.upload.size() > offload::section_union_limit ||",
              "                fp.download.size() > offload::section_union_limit ||",
              "                fp.upload.size() > offload::section_union_limit - fp.download.size()) return false;",
              "            const auto &array = metadata.arrays[a];",
              "            if (!array.element_bytes || array.dimensions.empty()) return false;",
              "            bool valid = true;",
              "            for (const auto &box : fp.upload) (void)offload::box_bytes(array, box, valid);",
              "            for (const auto &box : fp.download) (void)offload::box_bytes(array, box, valid);",
              "            if (!valid) return false;",
              "            std::vector<offload::Box> incoming(fp.upload);",
              "            incoming.insert(incoming.end(), fp.download.begin(), fp.download.end());",
              "            std::vector<offload::Box> united;",
              "            if (!offload::disjoint_union(incoming, united)) return false;",
              "            for (const auto &box : united) {",
              "                const auto bytes = offload::box_bytes(array, box, valid);",
              "                if (!valid || !offload::add(total, bytes, total)) return false;",
              "            }",
              "        }",
              "    } catch (...) { return false; }",
              f"    {prefix}_bytes = total;",
              "    return true;",
              f"}}(({data}), {{{', '.join(handles)}}});"]
    return tuple(lines)
