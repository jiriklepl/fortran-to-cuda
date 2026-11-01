"""Conservative source-index envelopes on the existing external access ABI.

Boxes are derived only from ordinary lowering's checked original coordinates.
Writes receive no overwrite or initialization credit: the coherence runtime
preserves their prior values, including holes in a conservative envelope.
"""


def access_preflight(fields, checks, envelopes, *, layouts_argument=False):
    lines = [
        "struct OwnedTileBox { std::vector<std::size_t> lower,upper; };",
        "struct OwnedTileAccess {",
        " std::vector<OwnedTileBox> reads,writes;",
        " std::vector<fort_scope_section> read_views,write_views;",
        " void finish(fort_scope_access& access) {",
        "  for (auto& box:reads) read_views.push_back({box.lower.data(),box.upper.data()});",
        "  for (auto& box:writes) write_views.push_back({box.lower.data(),box.upper.data()});",
        "  access={}; access.read_count=read_views.size(); access.reads=read_views.data();",
        "  access.write_count=write_views.size(); access.writes=write_views.data();",
        " }",
        "};",
        "static int prepare_tile_access(fort_scope_t context,const tile_state& state,",
        " fort_scope_external_binding_v1* bindings,OwnedTileAccess* accesses" +
        (",fort_scope_layout* layouts) {" if layouts_argument else ") {"),
        "try {",
        *([] if layouts_argument else [f"fort_scope_layout layouts[{len(fields)}]{{}};"]),
    ]
    for index, (binding, _) in enumerate(fields):
        dtype = {("real", 4): "FORT_SCOPE_REAL32", ("real", 8): "FORT_SCOPE_REAL64",
                 ("integer", 4): "FORT_SCOPE_INTEGER32"}[binding.dtype, binding.kind]
        lines += [
            f"if (const int status=fort_scope_layout_get(context,bindings[{index}].buffer,&layouts[{index}])) return status;",
            f"if (layouts[{index}].rank!={binding.rank} || layouts[{index}].type!={dtype} || "
            f"layouts[{index}].element_bytes!={binding.kind} || !layouts[{index}].extents) return FORT_SCOPE_ARGUMENT;",
        ]
    lines += checks
    for index, access, coordinates in envelopes:
        lower = ",".join(f"static_cast<std::size_t>({bounds}.lo-static_cast<long long>({lb}))"
                         for bounds, lb in coordinates)
        upper = ",".join(f"static_cast<std::size_t>({bounds}.hi-static_cast<long long>({lb})+1)"
                         for bounds, lb in coordinates)
        lines += [f"accesses[{index}].{access}.push_back({{{{{lower}}},{{{upper}}}}});"]
    lines += [
        f"for (std::size_t field=0;field<{len(fields)};++field) accesses[field].finish(bindings[field].access);",
        "return FORT_SCOPE_OK;",
        "} catch (const std::bad_alloc&) { return FORT_SCOPE_RESOURCE; }",
        "catch (...) { return FORT_SCOPE_EXECUTION; }",
        "}",
    ]
    return lines
