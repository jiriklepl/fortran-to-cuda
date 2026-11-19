"""Transport the descriptor-only empty-CALL proof and its exact emitted guard."""

from hashlib import sha256

from .checked_execution_plan import _require, _sha256, _text
from .conditional_return_record import _false, _keys, _sealed


def unwrap_empty_call(record, preservation, returned, lines):
    _sealed(record, {"identity", "version", "kind", "procedure", "call_preservation", "native_graph_identity",
                     "descriptors", "routes", "original_call_executes_once", "empty_path", "nonempty_path",
                     "payload_access_authorized", "entry_definition_authority", "allocation_state_assumed"})
    _require(returned.get("native_call_preservation_policy_version") == 3
             and type(record["version"]) is int and record["version"] == 1
             and record["kind"] == "empty_native_call_descriptor_guard"
             and record["procedure"] == preservation["procedure"]
             and record["call_preservation"] == preservation["identity"]
             and record["native_graph_identity"] == preservation["native_graph_identity"],
             "empty native CALL lost its original lifetime/source linkage")
    _false(record, ("payload_access_authorized", "entry_definition_authority", "allocation_state_assumed"))
    _require(record["original_call_executes_once"] is True
             and record["empty_path"] == "original descriptor guards and zero-trip loops; no payload or externally visible write"
             and record["nonempty_path"] == "existing full-domain retirement before unchanged original CALL",
             "empty native CALL changed its execution or descriptor-only policy")
    descriptors = record["descriptors"]
    _require(isinstance(descriptors, list) and bool(descriptors), "empty native CALL needs original descriptors")
    for row in descriptors:
        _keys(row, {"resource", "name", "import"})
        _require(_text(row["resource"]) and isinstance(row["name"], str) and row["name"].isascii()
                 and row["name"].isidentifier(), "empty native CALL descriptor name differs")
        imported = row["import"]
        if imported is not None:
            _keys(imported, {"module", "remote"})
            _require(all(isinstance(v, str) and v.isascii() and v.isidentifier() for v in imported.values())
                     and row["resource"] == imported["module"] + "::" + imported["remote"]
                     and row["name"] == "fort_empty_meta_" + sha256(row["resource"].encode()).hexdigest()[:12],
                     "empty native CALL changed its original public descriptor import")
    resources = [row["resource"] for row in descriptors]
    _require(resources == sorted(set(resources)), "empty native CALL repeats descriptor storage")
    routes = record["routes"]
    _require(isinstance(routes, list) and bool(routes), "empty native CALL needs its original child inventory")
    for row in routes:
        _keys(row, {"procedure", "source_graph", "source_call", "association", "original_guard", "original_loop", "descriptor"})
        _require(all(_text(row[key]) for key in ("procedure", "source_call", "original_guard", "original_loop"))
                 and _sha256(row["source_graph"]) and isinstance(row["association"], list) and bool(row["association"])
                 and row["descriptor"] in resources, "empty native CALL child source differs")
        for item in row["association"]:
            procedure = "source_target" in item
            _keys(item, {"formal", "actual", "association", "source_target" if procedure else "resource"})
            _require(all(_text(value) for value in item.values())
                     and item["association"] == ("original procedure argument; never invoked before close" if procedure
                                                 else "original descriptor or scalar reference; no synthetic dummy"),
                     "empty native CALL changed its original child association")
    _require({row["descriptor"] for row in routes} == set(resources), "empty native CALL has detached descriptors")
    flag = "fort_native_nonempty_" + record["identity"][:12]
    imports = [f"use {d['import']['module']}, only: {d['name']} => {d['import']['remote']}"
               for d in descriptors if d["import"] is not None]
    prefix = ["block", *imports, "intrinsic :: allocated, size", "logical :: " + flag, flag + " = .false."]
    for row in descriptors:
        name = row["name"]
        prefix += [f"if (allocated({name})) then", f"if (size({name}) /= 0) {flag} = .true.", "endif"]
    prefix += [f"if ({flag}) then"]
    _require(lines[:len(prefix)] == prefix and lines[-2:] == ["endif", "end block"],
             "empty native CALL changed its exact guarded descriptor evaluation")
    return lines[len(prefix):-2]
