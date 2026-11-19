"""Closed integrity schema for source-issued tile and line tasks; never execution authority."""
from __future__ import annotations

import re
from hashlib import sha256

from .checked_execution_plan import _digest, _require, _sha256, _text

_BINDING = {"name", "resource", "type", "kind", "rank", "intent", "attributes", "lower_bounds", "shape"}
_SOURCE = {"identity", "version", "kind", "procedure", "source_graph", "source_inputs", "selected_node_ids",
           "team", "resources", "scalars", "scalar_captures", "scratch", "transaction", "definitions_authorized", "placement_authorized",
           "motion_authorized", "pricing", "position", "source_proof_identity", "execution_binding"}
_TEAM = {"identity", "version", "procedure", "source_graph", "source_inputs", "selected_node_ids", "span",
         "original_team", "unit_proof_identities", "domain", "fields", "scalars", "scratch", "source_order", "fallback",
         "definitions_authorized", "placement_authorized", "requires_existing_capture_alias_bounds_definition_and_FP_guards",
         "pricing"}
_LINE_KIND = "original_line_snapshot_team"
_LINE_TEAM = {"identity", "version", "kind", "procedure", "source_graph", "source_inputs", "selected_node_ids",
              "unit_proof_identities", "transverse_domain", "complete_combined_team", "completion", "source_order",
              "fields", "scalars", "scratch", "definitions_authorized", "placement_authorized",
              "pricing_authorized", "cooperative_mapping_authorized"}
_LINE_PREWORK = {
    "version": 1,
    "source": "exact original line proof and regenerated producer/consumer IR",
    "domain": "original independent transverse loops; no tile width or MIN certificate",
    "scratch": "reached original rank-one descriptor; exact source-line extent; bounded existing lease",
    "integer": "checked original operations, coordinates and terminal values before payload acquisition",
    "initialization_authorized": False, "pricing_authorized": False, "cooperative_mapping_authorized": False,
}
_METADATA = {"version", "kind", "team_identity", "unit_proof_identities", "mapping", "entry", "module", "parameters",
             "cuda_sha256", "fortran_sha256", "scalar_readability_identity", "transaction", "scratch", "definitions_authorized", "placement_authorized", "pricing"}
_IDENT = re.compile(r"[a-z][a-z0-9_]{0,62}\Z")


def _keys(value, fields):
    _require(isinstance(value, dict) and set(value) == fields, "source device task requires its exact schema")


def _binding(value, *, rank=None, max_rank=3):
    _keys(value, _BINDING)
    _require(isinstance(value["name"], str) and _IDENT.fullmatch(value["name"]) and _text(value["resource"])
             and isinstance(value["type"], str) and value["type"] in {"integer", "real"}
             and type(value["kind"]) is int
             and (value["type"], value["kind"]) in {("integer", 4), ("real", 4), ("real", 8)}
             and type(value["rank"]) is int and 0 <= value["rank"] <= max_rank
             and (rank is None or value["rank"] == rank) and value["intent"] in (None, "in", "out", "inout")
             and isinstance(value["attributes"], list) and all(_text(item) for item in value["attributes"])
             and len(set(value["attributes"])) == len(value["attributes"])
             and isinstance(value["lower_bounds"], list) and all(_text(item) for item in value["lower_bounds"])
             and isinstance(value["shape"], list), "source device task original binding differs")
    for axis in value["shape"]:
        _keys(axis, {"kind", "source", "bounds"})
        _require(_text(axis["kind"]) and _text(axis["source"]) and isinstance(axis["bounds"], list)
                 and all(item is None or isinstance(item, str) for item in axis["bounds"]),
                 "source device task original shape differs")


def _inventory(values, *, scalar=False):
    _require(isinstance(values, list), "source device task requires an explicit original binding inventory")
    for item in values:
        _binding(item, rank=0 if scalar else None)
        if not scalar:
            _require(item["rank"] > 0, "source device task array inventory contains a scalar")
    _require(len({item["resource"] for item in values}) == len(values), "source device task repeats original storage")


def _identity_module_capture(binding, canonical, resource, calls):
    """Globals keep storage identity and have no formal argument association."""
    parts = binding["resource"].split("::")
    return (binding["resource"] == canonical and len(parts) == 2 and parts[0] != "argument"
            and all(_IDENT.fullmatch(part) for part in parts) and parts[1] == binding["name"]
            and binding["intent"] is None and _digest(binding) == _digest(resource["descriptor"])
            and not set(binding["attributes"]).intersection(
                {"pointer", "target", "optional", "value", "volatile", "asynchronous"})
            and bool(calls) and all(isinstance(call.get("resource_mappings"), dict)
                                    and canonical not in call["resource_mappings"] for call in calls))


def validate_source_task(record, member, procedure, inputs, resources, *, owner, manifest, capture_facts=None,
                         graph_execution=None, task_ordinal=None):
    if isinstance(record, dict) and record.get("kind") == "queued_original_tile_private_team":
        from .queued_team_source_task_record import validate_queued_team_source_task

        validate_queued_team_source_task(record, member, procedure, inputs, resources, owner=owner,
                                         manifest=manifest, capture_facts=capture_facts)
        return
    if isinstance(record, dict) and record.get("kind") == "queued_single_original_tile_private_team":
        from .queued_source_task_record import validate_queued_source_task

        validate_queued_source_task(record, member, procedure, inputs, resources, owner=owner,
                                    manifest=manifest, capture_facts=capture_facts)
        return
    if isinstance(record, dict) and record.get("kind") == "original_atomic_conditional_joined_team":
        from .atomic_source_task_record import validate_atomic_source_task

        validate_atomic_source_task(record, member, procedure, inputs, resources, owner=owner, manifest=manifest,
                                    graph_execution=graph_execution, task_ordinal=task_ordinal)
        return
    pricing = None
    native_live = False
    placement = None
    if isinstance(record, dict) and type(record.get("version")) is int and record["version"] in {2, 3, 4} and record.get("kind") != "original_tile_private_team":
        scheduled = record["version"] == 4
        native_live = record["version"] in {3, 4}
        _keys(record, {"identity", "version", "kind", "source_proof", "pricing", "execution_binding"}
              | ({"placement"} if scheduled else set()))
        placement = record["placement"] if scheduled else None
        _require(not scheduled or isinstance(placement, dict), "source device task requires its placement record")
        _require(record["kind"] == ("scheduled_original_tile_private_team_live" if scheduled else
                                    "quoted_original_tile_private_team_live" if native_live
                                    else "quoted_original_tile_private_team") and _sha256(record["identity"])
                 and record["identity"] == _digest({k: v for k, v in record.items() if k != "identity"}),
                 "source device task quoted execution identity differs")
        source, pricing = record["source_proof"], record["pricing"]
        _keys(source, _SOURCE - {"source_proof_identity", "execution_binding"})
        _require(_sha256(source["identity"])
                 and source["identity"] == _digest({k: v for k, v in source.items() if k != "identity"})
                 and isinstance(pricing, dict) and _sha256(pricing.get("identity"))
                 and type(pricing.get("version")) is int
                 and pricing["version"] in ({2, 3} if native_live else {1}),
                 "source device task quoted source or pricing identity differs")
        # This private validation view combines separately checked envelopes;
        # it neither issues a replacement proof nor changes its source digest.
        record = {**source, "source_proof_identity": source["identity"],
                  "execution_binding": record["execution_binding"]}
    else:
        _keys(record, _SOURCE)
        _require(_sha256(record["identity"])
                 and record["identity"] == _digest({k: v for k, v in record.items() if k != "identity"})
                 and _sha256(record["source_proof_identity"]) and record["source_proof_identity"] == _digest({
                     k: v for k, v in record.items() if k not in {"identity", "source_proof_identity", "execution_binding"}}),
                 "source device task proof or execution identity differs")
    team, fields, binding = _validate_source_authority(record, member, procedure, inputs, resources,
                                                      owner=owner, capture_facts=capture_facts, manifest=manifest,
                                                      allow_line=pricing is None and placement is None and not native_live,
                                                      allow_bytes=pricing is None and placement is None and not native_live,
                                                      allow_storage=pricing is None and placement is None and not native_live)
    if placement is not None:
        from .source_placement_record import validate_source_placement

        validate_source_placement(placement, record, pricing, binding, graph_execution, task_ordinal)
    line_team = record["kind"] == _LINE_KIND
    _require(not line_team or (pricing is None and placement is None and not native_live),
             "line source task cannot borrow tile pricing or placement")
    provider = binding["provider"]
    _keys(provider, {"entry", "module", "cuda_source_sha256", "fortran_source_sha256", "metadata"})
    metadata = provider["metadata"]
    _require(isinstance(metadata, dict) and type(metadata.get("version")) is int
             and metadata["version"] in ({6} if placement is not None else
                                          {4} if native_live else {3} if pricing is not None else {1, 2, 7, 8, 9, 10}),
             "source device task provider version differs")
    version = metadata["version"]
    _require(not line_team or version == 7, "line source task requires its exact access-envelope provider version")
    _keys(metadata, _METADATA | ({"line_snapshot_prework"} if line_team else set()) | ({"batching"} if version in {2, 3, 4, 6, 7, 8, 9, 10} else set())
          | ({"access_envelopes"} if version in {7, 8, 9, 10} else set())
          | ({"scalar_transport"} if version in {9, 10} else set())
          | ({"field_transport"} if version == 10 else set())
          | ({"native_prepare_entry", "native_transaction_api"} if native_live else set())
          | ({"placement", "selected_backend"} if placement is not None else set()))
    if line_team:
        prework = metadata["line_snapshot_prework"]
        _keys(prework, set(_LINE_PREWORK))
        _require(prework == _LINE_PREWORK and type(prework["version"]) is int
                 and all(prework[key] is False for key in
                         ("initialization_authorized", "pricing_authorized", "cooperative_mapping_authorized"))
                 and metadata["mapping"] == "thread",
                 "line source task prework or mapping differs")
    scratch = "one bounded logical tile; same-stream source-order units"
    if version in {2, 3, 4, 6, 7, 8, 9, 10}:
        batching = metadata["batching"]
        _keys(batching, {"version", "target_bytes", "minimum_tiles", "planned_launches"})
        _require(type(batching["version"]) is int and batching["version"] == (2 if version == 8 else 1)
                 and type(batching["target_bytes"]) is int
                 and (batching["target_bytes"] in {4194304, 16777216, 67108864}
                      if version == 8 else batching["target_bytes"] == 4194304)
                 and type(batching["minimum_tiles"]) is int and batching["minimum_tiles"] == 1
                 and batching["planned_launches"] == "sum ceil(unit_tiles/batch_tiles)"
                 and isinstance(provider["entry"], str)
                 and (placement is not None or provider["entry"].endswith(
                     "_batch_b" + str(batching["target_bytes"] // 1048576) + "m_boxes" if version == 8 else
                     "_batch_storage" if version == 10 else "_batch_boxes_bytes" if version == 9 else "_batch_boxes" if version == 7 else
                     "_batch" if version == 2 else "_batch_q_" + pricing["identity"][:10])),
                 "source device task bounded batching protocol differs")
        scratch = "bounded original tile batches; same-stream source-order units"
    if version in {7, 8, 9, 10}:
        access = metadata["access_envelopes"]
        _keys(access, {"version", "source", "reads", "writes", "overwrites_authorized",
                       "initialization_authorized", "preflight", "pricing"})
        _require(type(access["version"]) is int and access["version"] == 1
                 and access["source"] == "exact ordinary lowered original indices"
                 and access["reads"] == "conservative per-access Cartesian envelopes"
                 and access["writes"] == "conservative per-access Cartesian envelopes; preserve prior values"
                 and access["overwrites_authorized"] is access["initialization_authorized"] is False
                 and access["preflight"] == "metadata only before CUDA, scratch and payload acquisition"
                 and access["pricing"] == "unpriced; explicit sections only",
                 "source device task access envelopes cannot grant definitions or overwrite credit")
        suffix = team["identity"][:20] + ("_cta" if metadata["mapping"] == "cooperative" else "_thread") + "_batch"
        if version == 8:
            suffix += "_b" + str(metadata["batching"]["target_bytes"] // 1048576) + "m"
        suffix += "_boxes"
        if version in {9, 10}:
            suffix = (team["identity"][:10] + ("_cta" if metadata["mapping"] == "cooperative" else "_thread")
                      + "_s" + record["source_proof_identity"][:8]
                      + ("_batch_storage" if version == 10 else "_batch_boxes_bytes"))
        _require(provider["entry"] == "fort_tile_task_" + suffix
                 and provider["module"] == "fort_tile_module_" + suffix,
                 "source device task access envelope provider identity differs")
    byte_roots = [c["resource"] for c in record["scalar_captures"]
                  if c["authority"]["kind"] == "original real scalar storage; byte transport without value read"]
    _require(version == 10 or bool(byte_roots) == (version == 9), "byte scalar source requires its exact direct transport ABI")
    _require((version == 10) == (record["version"] == 3), "source storage requires its exact direct transport ABI")
    if version in {9, 10}:
        transport = metadata["scalar_transport"]
        _keys(transport, {"version", "resources", "host", "device", "abi", "definitions_authorized"})
        _require(type(transport["version"]) is int and transport["version"] == 1
                 and transport["resources"] == byte_roots
                 and transport["host"] == "copy object representation after positive domain; no typed real value read"
                 and transport["device"] == "decode at each original scalar reference; retain original control"
                 and transport["abi"] == "trivial aligned unsigned-char aggregate; original scalar size and alignment"
                 and transport["definitions_authorized"] is False,
                 "byte scalar transport cannot authorize value reads or definitions")
    if version == 10:
        field_transport = metadata["field_transport"]
        _require(field_transport == {
            "version": 2, "source_proof_identity": record["source_proof_identity"],
            "regions": "original physical may-access bytes; preserve partial writes without initialization",
            "conditional_reads": "original source control retained",
            "definitions_authorized": False, "pricing_authorized": False,
        } and type(field_transport.get("version")) is int
            and field_transport["definitions_authorized"] is field_transport["pricing_authorized"] is False,
            "source field transport cannot grant definitions, motion or prices")
    if placement is not None:
        suffix = team["identity"][:12] + ("_cta" if metadata["mapping"] == "cooperative" else "_thread")
        suffix += "_g_" + placement["identity"][:16]
        _require(provider["entry"] == "fort_tile_task_" + suffix
                 and provider["module"] == "fort_tile_module_" + suffix
                 and metadata["placement"] == placement
                 and metadata["selected_backend"] == {"argument": "selected_backend", "cpu": 0, "gpu": 1}
                 and type(metadata["selected_backend"].get("cpu")) is int
                 and type(metadata["selected_backend"].get("gpu")) is int,
                 "source device task authoritative backend ABI differs")
    if native_live:
        _require(isinstance(provider["entry"], str) and _IDENT.fullmatch(provider["entry"])
                 and metadata["native_prepare_entry"] == "fort_tile_native_" + sha256(provider["entry"].encode()).hexdigest()[:20]
                 and metadata["native_transaction_api"] == {
                     name: "fort_scope_source_host_" + name + "_v1" for name in ("prepare", "start", "complete", "cancel")},
                 "source device task native transaction identity differs")
    _require(all(isinstance(provider[key], str) and _IDENT.fullmatch(provider[key]) for key in ("entry", "module"))
             and metadata["kind"] == record["kind"] and metadata["team_identity"] == team["identity"]
             and metadata["scalar_readability_identity"] == record["source_proof_identity"]
             and metadata["unit_proof_identities"] == team["unit_proof_identities"]
             and isinstance(metadata["mapping"], str) and metadata["mapping"] in {"thread", "cooperative"}
             and metadata["entry"] == provider["entry"] and metadata["module"] == provider["module"]
             and _sha256(provider["cuda_source_sha256"]) and _sha256(provider["fortran_source_sha256"])
             and metadata["cuda_sha256"] == provider["cuda_source_sha256"]
             and metadata["fortran_sha256"] == provider["fortran_source_sha256"]
             and metadata["transaction"] == ("fort_scope_source_execute_v2" if version == 10 else "fort_scope_source_execute_v1")
             and metadata["scratch"] == scratch
             and metadata["definitions_authorized"] is metadata["placement_authorized"] is False
             and metadata["pricing"] == ("unpriced; explicit sections only" if pricing is None else pricing),
             "source device task compiled provider identity or protocol differs")
    artifacts = manifest.get("artifacts_sha256", {})
    _require(isinstance(artifacts, dict)
             and artifacts.get(provider["entry"] + ".cu") == provider["cuda_source_sha256"]
             and artifacts.get(provider["module"] + ".f90") == provider["fortran_source_sha256"],
             "source device task compiled artifacts differ")
    _validate_parameters(metadata, team, fields)
    if pricing is not None:
        from .source_task_pricing_record import validate_source_pricing

        validate_source_pricing(pricing, record, manifest, native_live=native_live)
        _require(pricing["work"]["mapping"] == metadata["mapping"], "source pricing compiled mapping differs")


def _validate_source_authority(record, member, procedure, inputs, resources, *, owner, capture_facts=None,
                               allow_line=False, manifest=None, allow_bytes=False, allow_storage=False):
    """Common original numerical/scalar/storage evidence, separate from transport."""
    _require(type(record["version"]) is int and record["version"] in {1, 2, 3}
             and (record["version"] == 1 or record["kind"] == "original_tile_private_team"
                  and (record["version"] == 2 and allow_bytes or record["version"] == 3 and allow_storage))
             and (record["kind"] == "original_tile_private_team" or allow_line and record["kind"] == _LINE_KIND),
             "source device task source proof version differs")
    storage_transport = record["version"] == 3
    if storage_transport:
        runtime = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
        capability = runtime.get("source_storage_tasks")
        _require(capability == {
            "abi_version": 2, "execute": "fort_scope_source_execute_v2", "bindings_abi_version": 1,
            "access": "original conditional may-access envelopes; byte transport without value evaluation",
            "definitions": "only independently proved overwrites; representation freshness is separate",
            "readonly": "retain original replica and add a current destination replica",
            "failure": "same source submission counts and external transaction completion contract",
            "cost_estimate_available": False,
        } and type(capability.get("abi_version")) is int and type(capability.get("bindings_abi_version")) is int
            and capability["cost_estimate_available"] is False
            and runtime.get("specification_entry") == {
                "abi_version": 3, "observe": "fort_scope_observe_v1", "fallback": "fort_scope_observe_all_v1",
                "handle_check": "fort_scope_buffer_valid_v1",
                "regions": "tracked physical metadata regions without definition authority; whole domain if mappings or live handles are unavailable",
                "replicas": "read-only publication preserves current device replicas and generations",
                "failure": "stop before original entry; never replay"}
            and type(runtime["specification_entry"]["abi_version"]) is int,
            "source storage transport requires physical observation and its exact runtime ABI")
    _require(record["procedure"] == member["procedure"] == procedure["procedure"]
             and record["source_graph"] == member["source_graph"] and record["source_inputs"] == inputs,
             "source device task original source linkage differs")
    nodes = record["selected_node_ids"]
    _require(isinstance(nodes, list) and bool(nodes)
             and all(_text(node) and node.startswith(record["procedure"] + "/") for node in nodes)
             and len(set(nodes)) == len(nodes),
             "source device task requires its original complete team selection")
    _require(all(record[key] is False for key in ("definitions_authorized", "placement_authorized", "motion_authorized"))
             and record["pricing"] == "unpriced; explicit sections only"
             and record["position"] == "exact original complete OpenMP team"
             and record["transaction"] == {"runtime": "fort_scope_source_execute_v2" if storage_transport else "fort_scope_source_execute_v1", "stream": "supplied",
                 "scratch": "prework lease outside non-reentrant transaction",
                 "submission": "started before first attempt; exact accepted source-kernel count",
                 "failure": "prework original whole span once; attempted failure poisons; no replay"},
             "source device task cannot grant new execution, definitions, motion or pricing")
    team = record["team"]
    line_team = record["kind"] == _LINE_KIND
    if line_team:
        _validate_line_team(team, record)
    else:
        _keys(team, _TEAM)
        _require(type(team["version"]) is int and team["version"] == 1 and _sha256(team["identity"])
                 and team["identity"] == _digest({k: v for k, v in team.items() if k != "identity"})
                 and all(team[k] == record[k] for k in ("procedure", "source_graph", "source_inputs", "selected_node_ids"))
                 and team["definitions_authorized"] is team["placement_authorized"] is False
                 and team["requires_existing_capture_alias_bounds_definition_and_FP_guards"] is True
                 and team["pricing"] == "unpriced; forced sections only"
                 and team["source_order"] == "all original tile units and phase evaluations retained"
                 and team["fallback"] == "unchanged entire original OpenMP team once; prework only",
                 "source device task complete-team authority differs")
        _require(isinstance(team["span"], list) and len(team["span"]) == 2
                 and all(type(n) is int for n in team["span"]) and 1 <= team["span"][0] <= team["span"][1]
                 and _text(team["original_team"]) and isinstance(team["unit_proof_identities"], list)
                 and bool(team["unit_proof_identities"]) and all(_sha256(n) for n in team["unit_proof_identities"])
                 and len(set(team["unit_proof_identities"])) == len(team["unit_proof_identities"]),
                 "source device task original team or unit inventory differs")
        _require(isinstance(team["domain"], list) and 1 <= len(team["domain"]) <= 3,
                 "source device task requires its original rectangular domain")
        for axis in team["domain"]:
            _keys(axis, {"lower", "upper", "width"})
            _require(all(_text(value) for value in axis.values()), "source device task domain expressions differ")
    for key in ("scalars", "scratch"):
        _inventory(record[key], scalar=key == "scalars")
        _require(record[key] == team[key], "source device task scalar or scratch inventory differs")
    captures = record["scalar_captures"]
    _require(isinstance(captures, list) and len(captures) == len(record["scalars"]),
             "source device task scalar readability inventory differs")
    for capture, scalar in zip(captures, record["scalars"], strict=True):
        _keys(capture, {"resource", "descriptor", "authority"})
        _require(capture["resource"] == scalar["resource"] and capture["descriptor"] == scalar,
                 "source device task scalar readability binding differs")
        authority = capture["authority"]
        _require(isinstance(authority, dict), "source device task scalar readability requires typed authority")
        kind = authority.get("kind")
        if kind == "explicit checked whole-defined scalar entry fact":
            _keys(authority, {"kind", "finite_or_constant_authorized", "fact"})
            _scalar_entry_fact(authority["fact"], capture["resource"], capture_facts)
        elif kind == "original real scalar storage; byte transport without value read":
            _keys(authority, {"kind", "finite_or_constant_authorized", "fact"})
            _require(record["version"] in {2, 3} and not line_team and scalar["type"] == "real"
                     and scalar["kind"] in {4, 8} and scalar["rank"] == 0
                     and not set(scalar["attributes"]) & {
                         "pointer", "allocatable", "optional", "target", "value", "volatile", "asynchronous"},
                     "byte scalar capture requires stable plain real scalar storage")
            _scalar_storage_fact(authority["fact"], capture["resource"], capture_facts)
        elif kind == "original reached scalar prefix definition":
            _keys(authority, {"kind", "proof", "finite_or_constant_authorized"})
            _require(line_team, "reached scalar prefix cannot authorize tile, queued or priced work")
            _scalar_prefix(authority["proof"], scalar, record, capture_facts, procedure, manifest)
        else:
            _keys(authority, {"kind", "finite_or_constant_authorized"})
            _require((kind == "original positive-domain integer read" and scalar["type"] == "integer")
                     or (kind == "original PARAMETER declaration" and "parameter" in scalar["attributes"]),
                     "source device task scalar lacks independent readability authority")
        _require(authority["finite_or_constant_authorized"] is False,
                 "source device task readability cannot grant finite or constant values")
    _require(storage_transport or (record["version"] == 2) == any(c["authority"]["kind"] ==
             "original real scalar storage; byte transport without value read" for c in captures),
             "byte scalar source version or capture inventory differs")
    fields, effects = team["fields"], record["resources"]
    if line_team:
        _require(isinstance(fields, list), "line source task requires its original field inventory")
        for field in fields:
            _keys(field, {"descriptor", "written"})
        # A private validation view reuses the common binding/ABI checks. The
        # original record and its source/team digests remain unchanged.
        fields = [{"binding": field["descriptor"], "written": field["written"]} for field in fields]
    _require(isinstance(fields, list) and bool(fields) and isinstance(effects, list) and len(fields) == len(effects),
             "source device task requires its complete original fields")
    for field, effect in zip(fields, effects, strict=True):
        _keys(field, {"binding", "written"})
        _binding(field["binding"])
        _require(field["binding"]["rank"] > 0 and type(field["written"]) is bool,
                 "source device task field access differs")
        _keys(effect, {"resource", "descriptor", "access", "regions", "initialization_authorized"})
        _require(effect["descriptor"] == field["binding"] and effect["resource"] == field["binding"]["resource"]
                 and effect["access"] == ("readwrite" if field["written"] else "read")
                 and effect["regions"] == ("original physical may-access bytes; preserve partial writes without initialization"
                                           if storage_transport else "complete physical root; read before partial writes")
                 and effect["initialization_authorized"] is False,
                 "source device task must retain whole reads and conservative partial writes")
    local_roots = [field["binding"]["resource"] for field in fields]
    _require(len(set(local_roots)) == len(local_roots), "source device task repeats a field")
    binding = record["execution_binding"]
    projected = isinstance(binding, dict) and "call_projection" in binding
    _keys(binding, {"owner", "canonical_resources", "registration_slots", "provider"}
          | ({"call_projection"} if projected else set()))
    _require(not projected or line_team and allow_line,
             "source CALL projection cannot authorize tile, queued or priced work")
    canonical, slots = binding["canonical_resources"], binding["registration_slots"]
    _require(binding["owner"] == owner and isinstance(canonical, dict) and isinstance(slots, dict)
             and set(canonical) == set(slots) == set(local_roots)
             and all(_text(value) for value in canonical.values())
             and len(set(canonical.values())) == len(canonical), "source device task owning storage projection differs")
    projected_roots = None
    if projected:
        from .source_task_call_record import validate_call_projection

        projected_roots = validate_call_projection(binding["call_projection"], record, fields, procedure,
                                                    resources, owner, manifest)
    elif line_team:
        from .source_task_call_record import require_single_association

        require_single_association(procedure, owner, manifest)
    for field in fields:
        local = field["binding"]["resource"]
        root = canonical[local]
        _require(root in resources and resources[root].get("device_replica_authorized") is True
                 and type(slots[local]) is int and slots[local] == resources[root]["registration_identity"]
                 and all(field["binding"][key] == resources[root]["descriptor"].get(key) for key in ("type", "kind", "rank")),
                 "source device task physical registration differs")
        calls = procedure["calls"]
        _require((projected_roots is not None and root in projected_roots[local])
                 or (member["procedure"] == owner and local == root)
                 or local == "argument::"+field["binding"]["name"]
                 and bool(calls) and all(isinstance(call.get("resource_mappings"), dict)
                                       and call["resource_mappings"].get(local) == root for call in calls)
                 or _identity_module_capture(field["binding"], root, resources[root], calls),
                 "source device task child storage projection differs")
    return team, fields, binding



def _validate_line_team(team, record):
    """Closed line schema; a line team cannot inherit tile-width authority."""
    _keys(team, _LINE_TEAM)
    _require(type(team["version"]) is int and team["version"] == 1 and team["kind"] == _LINE_KIND
             and _sha256(team["identity"])
             and team["identity"] == _digest({k: v for k, v in team.items() if k != "identity"})
             and all(team[k] == record[k] for k in ("procedure", "source_graph", "source_inputs", "selected_node_ids"))
             and team["complete_combined_team"] is True
             and all(team[k] is False for k in ("definitions_authorized", "placement_authorized",
                                                 "pricing_authorized", "cooperative_mapping_authorized"))
             and team["completion"] == "exact underlying original line proof and joined-completion token"
             and team["source_order"] == "complete original line snapshot before original serial consumer",
             "line source task complete-team authority differs")
    units = team["unit_proof_identities"]
    _require(isinstance(units, list) and len(units) == 1 and _sha256(units[0]),
             "line source task requires exactly one original unit")
    domain = team["transverse_domain"]
    _require(isinstance(domain, list) and 1 <= len(domain) <= 2,
             "line source task requires independent original transverse loops")
    for axis in domain:
        _keys(axis, {"iterator", "resource", "lower", "upper", "step"})
        _require(isinstance(axis["iterator"], str) and _IDENT.fullmatch(axis["iterator"])
                 and axis["resource"] == team["procedure"] + "::" + axis["iterator"]
                 and _text(axis["lower"]) and _text(axis["upper"])
                 and type(axis["step"]) is int and axis["step"] == 1,
                 "line source task transverse coordinate or stride differs")
    _require(len({axis["resource"] for axis in domain}) == len(domain),
             "line source task repeats a transverse coordinate")
    scratch = team["scratch"]
    _require(isinstance(scratch, list) and len(scratch) == 1, "line source task requires one private vector")
    _binding(scratch[0], rank=1)
    _require(scratch[0]["type"] == "real" and scratch[0]["intent"] is None
             and scratch[0]["resource"] == team["procedure"] + "::" + scratch[0]["name"],
             "line source task private vector differs")

def _validate_parameters(metadata, team, fields):
    """Exact typed generated argument order shared by distinct backend readers."""
    expected = []
    iso = {("integer", 4): "c_int", ("real", 4): "c_float", ("real", 8): "c_double"}
    for scalar in team["scalars"]:
        expected.append(("scalar", scalar, None, iso[scalar["type"], scalar["kind"]]))
    for field in fields:
        expected.extend(("external_lower", field["binding"], axis, "c_int")
                        for axis in range(1, field["binding"]["rank"] + 1))
    for scratch in team["scratch"]:
        expected.extend((role, scratch, axis, "c_int") for axis in range(1, scratch["rank"] + 1)
                        for role in ("scratch_lower", "scratch_upper"))
    parameters = metadata["parameters"]
    _require(isinstance(parameters, list) and len(parameters) == len(expected), "source device task typed ABI inventory differs")
    for index, (parameter, (role, descriptor, axis, kind)) in enumerate(zip(parameters, expected, strict=True)):
        _keys(parameter, {"role", "binding", "axis", "name", "kind"})
        _require(parameter == {"role": role, "binding": descriptor, "axis": axis, "name": "p" + str(index), "kind": kind},
                 "source device task typed parameter order or descriptor recipe differs")


def _scalar_storage_fact(fact, resource, capture_facts):
    _require(isinstance(fact, dict) and {"storage", "allocation_changes", "escapes"} <= set(fact)
             <= {"storage", "allocation_changes", "escapes", "lifetime"}
             and fact["storage"] == "stable" and fact["allocation_changes"] is fact["escapes"] is False
             and isinstance(fact.get("lifetime", "invocation"), str)
             and fact.get("lifetime", "invocation") in {"invocation", "selected_window"},
             "byte scalar storage fact cannot grant initialization")
    if capture_facts is not None:
        captures = capture_facts.get("captures") if isinstance(capture_facts, dict) else None
        supplied = captures.get(resource) if isinstance(captures, dict) else None
        _require(isinstance(supplied, dict) and supplied.get("storage") == "stable"
                 and supplied.get("allocation_changes") is supplied.get("escapes") is False
                 and isinstance(supplied.get("lifetime", "invocation"), str)
                 and supplied.get("lifetime", "invocation") in {"invocation", fact.get("lifetime", "invocation")},
                 "byte scalar storage fact differs from the supplied request")


def _scalar_entry_fact(fact, resource, capture_facts):
    _require(isinstance(fact, dict)
             and {"storage", "initialized", "allocation_changes", "escapes"} <= set(fact)
             <= {"storage", "initialized", "allocation_changes", "escapes", "lifetime"}
             and fact["storage"] == "stable" and fact["initialized"] == "whole"
             and fact["allocation_changes"] is fact["escapes"] is False
             and isinstance(fact.get("lifetime", "invocation"), str)
             and fact.get("lifetime", "invocation") in {"invocation", "selected_window"},
             "source device task scalar entry fact differs")
    if capture_facts is not None:
        captures = capture_facts.get("captures") if isinstance(capture_facts, dict) else None
        supplied = captures.get(resource) if isinstance(captures, dict) else None
        _require(isinstance(supplied, dict)
                 and supplied.get("allocation_changes") is False and supplied.get("escapes") is False
                 and all(fact.get(key, "invocation" if key == "lifetime" else None)
                         == supplied.get(key, "invocation" if key == "lifetime" else None)
                         for key in ("storage", "initialized", "allocation_changes", "escapes", "lifetime")),
                 "source device task scalar entry fact differs from the supplied request")


def _scalar_prefix(proof, scalar, record, capture_facts, procedure, manifest):
    """Transport a reached source definition without reconstructing source proof."""
    flags = {"entry_definition_authorized", "array_payload_definition_authorized", "finite_or_constant_authorized",
             "motion_authorized", "execution_authorized", "lifetime_authorized"}
    _keys(proof, {"identity", "version", "kind", "procedure", "source_inputs", "source_graph", "selected_node_ids",
                  "selection", "resources", "assignments", "dependencies", "effects_identities", "semantics"} | flags)
    _require(type(proof["version"]) is int and proof["version"] == 2
             and proof["kind"] == "original_reached_scalar_prefix_definitions"
             and _sha256(proof["identity"])
             and proof["identity"] == _digest({k: v for k, v in proof.items() if k != "identity"})
             and all(proof[key] == record[key] for key in ("procedure", "source_inputs", "source_graph", "selected_node_ids"))
             and proof["semantics"] == "original unchanged prefix and predicates execute before scalar capture at this selection"
             and all(proof[key] is False for key in flags),
             "reached scalar prefix source, effects or authority differs")
    selection = proof["selection"]
    _keys(selection, {"schema_version", "procedure", "selected_node_ids", "first_line", "last_line",
                      "original_source_sha256", "activation", "prefix", "enclosing_controls", "fallback"})
    _require(type(selection["schema_version"]) is int and selection["schema_version"] == 1
             and selection["procedure"] == proof["procedure"] and selection["selected_node_ids"] == proof["selected_node_ids"]
             and all(type(selection[key]) is int for key in ("first_line", "last_line"))
             and 1 <= selection["first_line"] <= selection["last_line"]
             and _sha256(selection["original_source_sha256"])
             and selection["activation"] == "original complete team after unchanged native prefix and branch evaluation"
             and selection["fallback"] == "close before unchanged original whole team once; never replay executed work"
             and isinstance(selection["prefix"], list) and bool(selection["prefix"])
             and all(isinstance(node, str) for node in selection["prefix"])
             and isinstance(selection["enclosing_controls"], list)
             and all(_text(node) for node in selection["enclosing_controls"]),
             "reached scalar prefix original selection differs")
    executable_count = sum(bool(node.strip()) and not node.lstrip().startswith("!")
                           for node in selection["prefix"])
    _require(isinstance(proof["effects_identities"], list) and executable_count > 0
             and len(proof["effects_identities"]) == executable_count
             and all(_sha256(identity) for identity in proof["effects_identities"]),
             "reached scalar prefix requires ordered per-statement original effect identities")
    definitions = proof["resources"]
    _inventory(definitions, scalar=True)
    _require(bool(definitions) and scalar in definitions
             and all(binding["type"] == "integer" and binding["kind"] == 4
                     and binding["resource"].startswith(proof["procedure"] + "::")
                     and binding["intent"] is None
                     and not set(binding["attributes"]) & {"save", "pointer", "target", "optional", "volatile",
                                                           "asynchronous", "allocatable", "parameter"}
                     for binding in definitions),
             "reached scalar prefix requires exact original local INTEGER definitions")
    assignments = proof["assignments"]
    _require(isinstance(assignments, list) and bool(assignments), "reached scalar prefix lost original assignments")
    previous_line, previous_position, assigned = 0, -1, set()
    for assignment in assignments:
        _keys(assignment, {"resource", "first_line", "last_line", "source"})
        _require(_text(assignment["resource"]) and assignment["resource"].startswith(proof["procedure"] + "::")
                 and all(type(assignment[key]) is int for key in ("first_line", "last_line"))
                 and previous_line <= assignment["first_line"] <= assignment["last_line"] < selection["first_line"]
                 and assignment["first_line"] > 0 and _text(assignment["source"]),
                 "reached scalar prefix assignment source span differs")
        positions = [index for index, node in enumerate(selection["prefix"])
                     if index > previous_position and node == assignment["source"]]
        _require(bool(positions), "reached scalar prefix assignment order or original text differs")
        previous_position, previous_line = positions[0], assignment["last_line"]
        assigned.add(assignment["resource"])
    _require(all(binding["resource"] in assigned for binding in definitions),
             "reached scalar prefix omitted a demanded original definition")
    dependencies = proof["dependencies"]
    _require(isinstance(dependencies, list), "reached scalar prefix requires original dependency inventory")
    names = []
    for dependency in dependencies:
        _keys(dependency, {"resource", "descriptor", "authority"})
        binding, authority = dependency["descriptor"], dependency["authority"]
        if isinstance(binding, dict) and binding.get("type") == "logical":
            # This scalar is read by the unchanged host prefix, never passed
            # through the numerical provider ABI. Validate its original shape
            # without widening the provider's supported scalar types.
            _require(type(binding.get("kind")) is int and binding["kind"] in {1, 4},
                     "reached scalar prefix logical dependency kind differs")
            _binding({**binding, "type": "integer", "kind": 4}, rank=0)
        else:
            _binding(binding, rank=0)
        _require(dependency["resource"] == binding["resource"] and isinstance(authority, dict),
                 "reached scalar prefix dependency binding differs")
        names.append(dependency["resource"])
        if authority.get("kind") == "original PARAMETER declaration":
            _keys(authority, {"kind"})
            _require("parameter" in binding["attributes"], "reached scalar prefix parameter declaration differs")
        elif authority.get("kind") == "original source-constant scalar call inputs":
            from .scalar_call_record import validate_scalar_call_input

            validate_scalar_call_input(authority, binding, proof, procedure, manifest,
                                       call_projection=record["execution_binding"].get("call_projection"))
        else:
            _keys(authority, {"kind", "fact"})
            _require(authority["kind"] == "explicit checked whole-defined scalar entry fact"
                     and not binding["resource"].startswith(proof["procedure"] + "::")
                     and "parameter" not in binding["attributes"]
                     and (binding["type"] != "logical" or binding["kind"] == 1),
                     "reached scalar prefix cannot substitute local entry facts or unsupported capture kinds")
            _scalar_entry_fact(authority["fact"], binding["resource"], capture_facts)
            _require(authority["fact"].get("lifetime", "invocation") == "invocation",
                     "reached scalar prefix dependency requires invocation lifetime")
    _require(names == sorted(set(names)), "reached scalar prefix repeats or reorders dependencies")
