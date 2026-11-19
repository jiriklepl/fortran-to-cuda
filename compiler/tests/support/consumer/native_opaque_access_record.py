"""Validate typed native reservations without inventing source or payload authority.

The source compiler owns complete READ proofs. This reader checks their closed
transport schema, original operation/call links and final canonical role union.
"""
from .checked_execution_plan import _digest, _require, _sha256, _text
from .source_task_record import _binding

_PERMIT = {"identity", "version", "authority", "procedure", "source_graph", "selected_node_ids", "binding",
           "analysis_identity", "summary_identity", "structured_identity", "demand_identity", "allocation_changes",
           "calls_or_escapes", "device_capture", "managed_definitions", "lifetime", "range_validation"}
_FALSE = {"device_capture", "initialization_authorized", "managed_definitions", "alias_authorized"}


def _keys(value, fields):
    _require(isinstance(value, dict) and set(value) == fields, "native opaque access schema differs")


def _sealed(value):
    _require(_sha256(value.get("identity"))
             and value["identity"] == _digest({key: item for key, item in value.items() if key != "identity"}),
             "native opaque access content identity differs")


_WRITE_FALSE = {
    "managed_payload_access_authorized", "managed_definition_authorized", "entry_definition_authority",
    "declaration_initialization_credit", "capture_authorized", "device_replica_authorized", "alias_authority",
    "initial_quiescence_authority", "ownership_or_lifetime_grant", "motion_authority", "gpu_legality_established",
    "cost_authority", "execution_authorized", "whole_output_definedness_authorized",
}
_WRITE = {"identity", "schema_version", "proof_role", "procedure", "source_session", "source_context",
    "source_inputs", "source_graph", "selected_node_ids", "source", "first_line", "last_line", "original_selection",
    "original_call", "resource", "descriptor", "bounds", "element_count", "storage_bytes", "mode",
    "native_graph_identity", "reachability_identity", "storage_identity", "completion_identities",
    "joined_team_identities", "association_identities", "incoming_instances", "original_effect_inventory",
    "target_actions", "storage_disjointness_obligations", "original_definition_kills", "complete_write_effect_inventory",
    "completion_scope", "execution"} | _WRITE_FALSE
_ASSOC_FALSE = {"payload_access_authorized", "payload_definition_authority", "entry_definition_authority",
    "initialization_authorized", "capture_authorized", "alias_authority", "ownership_authority", "lifetime_authority",
    "completion_authority", "execution_authorized", "native_write_authority", "gpu_legality_established",
    "numerical_lowering_authorized", "motion_authorized"}
_ASSOC = {"identity", "schema_version", "proof_role", "source_session", "source_context", "policy", "caller", "callee",
    "graph_identity", "selected_node_ids", "source", "first_line", "last_line", "original_selection", "original_call",
    "source_call", "formal", "formal_descriptor", "actual_descriptor", "formal_bounds", "actual_bounds", "element_count",
    "actual_index_minus_formal_index", "association", "requires_original_execution"} | _ASSOC_FALSE
_ACTIONS = {"read", "write", "undefine", "association", "descriptor"}


def _texts(value, *, nonempty=False):
    _require(isinstance(value, list) and (bool(value) or not nonempty) and all(_text(item) for item in value),
             "native opaque source text inventory differs")


def _identities(value):
    _texts(value, nonempty=True)
    _require(len(value) == len(set(value)) and all(_sha256(item) for item in value),
             "native opaque source proof inventory differs")


def _native_descriptor(value):
    # Native ABI validation does not confer the GPU scalar/type restrictions of _binding.
    from .source_task_record import _BINDING, _IDENT

    _keys(value, _BINDING)
    _require(isinstance(value["name"], str) and _IDENT.fullmatch(value["name"]) and _text(value["resource"])
             and isinstance(value["type"], str) and value["type"] in {"real", "integer", "logical"} and type(value["kind"]) is int
             and value["kind"] in {4, 8} and type(value["rank"]) is int and 0 <= value["rank"] <= 3
             and value["intent"] in (None, "in", "out", "inout"), "native opaque original descriptor differs")
    _texts(value["attributes"])
    _texts(value["lower_bounds"])
    _require(len(value["attributes"]) == len(set(value["attributes"]))
             and len(value["lower_bounds"]) == value["rank"]
             and isinstance(value["shape"], list) and len(value["shape"]) == value["rank"],
             "native opaque original descriptor dimensions differ")
    for shape in value["shape"]:
        _keys(shape, {"kind", "source", "bounds"})
        _require(_text(shape["kind"]) and _text(shape["source"]) and isinstance(shape["bounds"], list)
                 and all(item is None or isinstance(item, str) for item in shape["bounds"]),
                 "native opaque original shape differs")


def _bounds(value, count):
    _require(isinstance(value, list) and len(value) == 2
             and all(type(item) is int and -(2**31) <= item < 2**31 for item in value)
             and type(count) is int and 0 < count < 2**31 and value[1] - value[0] + 1 == count,
             "native opaque fixed bounds or extent differs")


def _storage(value):
    _keys(value, {"resource", "declaring_scope", "base_resource", "component_path", "rank", "type", "attributes",
                  "numerical_capture_authority"})
    _require(all(_text(value[key]) for key in ("resource", "declaring_scope", "base_resource", "type"))
             and type(value["rank"]) is int and value["rank"] >= 0
             and value["numerical_capture_authority"] is False, "native opaque effect storage differs")
    _texts(value["attributes"])
    _texts(value["component_path"])


def _original_write_call(call, original):
    """Compare both existing CALL transports; no expression evaluation or new association."""
    from .source_task_call_record import _CALL, _MAPPING, _REQUIREMENTS

    common = {"procedure", "target", "source_kind", "formal_arguments", "actual_arguments", "original_arguments"}
    authority = {"effect_authority", "capture_authority", "alias_authority", "completion_authority", "execution_authority"}
    _keys(original, common | {"schema_version", "associations", "actual_evaluation"} | authority)
    _keys(call, _CALL | {"caller", "source", "first_line", "last_line"})
    _require(type(original["schema_version"]) is int and original["schema_version"] == 1
             and type(call["schema_version"]) is int and call["schema_version"] == 2
             and all(original[key] is False for key in authority)
             and original["actual_evaluation"] == "unchanged original call position and guards"
             and call["actual_evaluation"] == "original call position and guards; no analysis-time evaluation"
             and all(call[key] == original[key] for key in common), "native opaque original CALL transport differs")
    _texts(call["formal_arguments"], nonempty=True)
    formals, actuals = call["formal_arguments"], call["actual_arguments"]
    _require(len(formals) == len(set(formals)) and isinstance(actuals, list) and len(actuals) == len(formals)
             and all(item is None or _text(item) for item in actuals)
             and isinstance(original["associations"], list) and len(original["associations"]) == len(formals)
             and isinstance(call["resource_mappings"], list) and len(call["resource_mappings"]) == len(formals),
             "native opaque original CALL argument inventory differs")
    supplied = set()
    _require(isinstance(call["original_arguments"], list), "native opaque original argument order differs")
    for index, item in enumerate(call["original_arguments"]):
        _keys(item, {"position", "formal", "keyword", "actual"})
        _require(type(item["position"]) is int and item["position"] == index and item["formal"] in formals
                 and item["formal"] not in supplied and item["actual"] == actuals[formals.index(item["formal"])]
                 and _text(item["actual"]) and (item["keyword"] is None
                     or isinstance(item["keyword"], str) and item["keyword"].lower() == item["formal"]),
                 "native opaque original argument order differs")
        supplied.add(item["formal"])
    _require(supplied == {f for f, a in zip(formals, actuals, strict=True) if a is not None},
             "native opaque original supplied arguments differ")
    mappings = {}
    for formal, actual, receipt, mapping in zip(formals, actuals, original["associations"], call["resource_mappings"], strict=True):
        _keys(receipt, {"formal", "actual", "formal_descriptor", "actual_type", "entry_definition_kill"})
        _keys(mapping, _MAPPING)
        descriptor = mapping["formal_descriptor"]
        _native_descriptor(descriptor)
        _require(receipt["formal"] == mapping["formal"] == formal and receipt["actual"] == mapping["actual"] == actual
                 and descriptor == receipt["formal_descriptor"] and descriptor["name"] == formal
                 and descriptor["resource"] == mapping["formal_resource"] == "argument::" + formal
                 and type(receipt["entry_definition_kill"]) is bool
                 and receipt["entry_definition_kill"] is (descriptor["intent"] == "out")
                 and mapping["section"] is None, "native opaque original managed association differs")
        actual_type, actual_descriptor = receipt["actual_type"], mapping["actual_descriptor"]
        if actual_type is not None:
            _keys(actual_type, {"type", "kind", "rank", "storage_resource", "declaring_type"})
            _require(actual_type["declaring_type"] is None and type(actual_type["kind"]) is int
                     and type(actual_type["rank"]) is int, "native opaque original actual type differs")
        if actual_descriptor is not None:
            _native_descriptor(actual_descriptor)
            _require(actual_type is not None and all(actual_type[k] == actual_descriptor[k] for k in ("type", "kind", "rank"))
                     and mapping["resource"] == actual_descriptor["resource"] == actual_type["storage_resource"],
                     "native opaque original actual storage differs")
            mappings[mapping["formal_resource"]] = mapping["resource"]
        else:
            _require(mapping["resource"] is None and (actual_type is None or actual_type["storage_resource"] is None),
                     "native opaque original expression gained storage")
        _require(mapping["storage"] == ("omitted" if actual is None else "whole" if actual_descriptor else "expression")
                 and mapping["presence"] == ("omitted" if actual is None else "supplied"),
                 "native opaque original presence or storage differs")
        _keys(mapping["requirements"], _REQUIREMENTS)
        _require(mapping["requirements"] == {"presence_preserving_forwarding": False,
                 "descriptor_dependent_presence": False,
                 "original_actual_presence": bool(actual_descriptor and "optional" in actual_descriptor["attributes"]),
                 "original_allocation_descriptor": "allocatable" in descriptor["attributes"], "physical_section_mapping": False}
                 and all(type(value) is bool for value in mapping["requirements"].values()),
                 "native opaque original actual requirements differ")
    _require(call["resource_mapping"] == mappings, "native opaque complete original resource mapping differs")


def _write_guard(control, permit, procedure, unit, scope):
    guard = control["original_guard"]
    _require(guard is None or _text(guard), "native opaque original guard differs")
    selected = permit["selected_node_ids"]
    _require(isinstance(selected, list) and len(selected) == 1 and _text(selected[0])
             and control["selected_node_ids"] == ([selected[0] + "/body/0"] if guard is not None else selected)
             and permit["original_selection"] == ("IF (" + guard + ") " + permit["original_call"]
                                                    if guard is not None else permit["original_call"]),
             "native opaque whole IF to original action linkage differs")
    graphs = [item for item in scope.get("graph_execution", {}).get("procedures", [])
              if item.get("procedure") == procedure["procedure"]]
    _require(len(graphs) == 1, "native opaque CALL requires its original task graph")
    task_id = graphs[0]["admitted_unit_tasks"].get(str(unit["ordinal"]))
    pending, found = [(graphs[0]["task_graph"], ())], []
    while pending:
        graph, parents = pending.pop()
        _require(isinstance(graph, dict) and isinstance(graph.get("tasks"), list), "native opaque original task graph differs")
        for task in graph["tasks"]:
            if task.get("id") == task_id:
                found.append((task, parents))
            if "branch" in task:
                for index, child in enumerate(task["branch"]):
                    pending.append((child, parents + ((task, index),)))
            if "loop" in task:
                pending.append((task["loop"], parents + ((task, None),)))
    _require(len(found) == 1, "native opaque original CALL task missing or duplicated")
    task, parents = found[0]
    effects = {}
    for operation in unit["operations"]["native_operations"]:
        for root, actions in operation["resource_effects"].items():
            effects.setdefault(root, set()).update(actions)
    _require(task.get("implementations") == ["cpu"] and task.get("ordered") is True
             and all(task.get(key) == sorted(root for root, actions in effects.items() if action in actions)
                     for key, action in (("reads", "read"), ("may_writes", "write")))
             and all(permit["resource"] not in node.get(key, [])
                     for node in [task, *(parent for parent, _ in parents)]
                     for key in ("must_writes", "overwrites")),
             "native opaque CALL task effects or managed definition credit differs")
    if guard is not None:
        _require(any(parent.get("id") == selected[0] + ":alternative:0" and index == 0
                     and parent.get("predicate") == [guard] for parent, index in parents),
                 "native opaque CALL is detached from its exact original guard")


def _write_access(access, procedure, operation, plan, manifest, unit, scope):
    _keys(access, {"resource", "mode", "permit_kind", "permit", "association_receipts", "execution_control"})
    permit, root, control = access["permit"], access["resource"], access["execution_control"]
    _keys(permit, _WRITE)
    _sealed(permit)
    _keys(control, {"version", "source_graph", "selected_node_ids", "original_guard", "source_call"})
    _require(type(control["version"]) is int and control["version"] == 1
             and type(permit["schema_version"]) is int and permit["schema_version"] == 1
             and permit["proof_role"] == "original_native_opaque_array_write"
             and _text(root) and root == permit["resource"] and access["mode"] == permit["mode"]
             and isinstance(access["mode"], str) and access["mode"] in {"W", "RW"} and all(permit[k] is False for k in _WRITE_FALSE)
             and permit["complete_write_effect_inventory"] is True and permit["completion_scope"] == "normal return only"
             and permit["execution"] == "unchanged original control after separate alias and managed host transactions",
             "native opaque WRITE authority or mode differs")
    _require(permit["procedure"] == procedure["procedure"]
             and control["source_graph"] == procedure["structured_summary_identity"]
             and permit["source_graph"] == operation["structured_summary_identity"]
             and control["selected_node_ids"] == operation["selected_original_nodes"]
             and permit["source"] == procedure["source"]
             and _digest(permit["source_inputs"]) == plan["source_inputs_identity"]
             and all(type(permit[k]) is int and permit[k] == operation[k] for k in ("first_line", "last_line"))
             and all(_sha256(permit[k]) for k in ("source_session", "source_context", "source_graph", "native_graph_identity",
                                                 "reachability_identity", "storage_identity")),
             "native opaque WRITE source or complete graph differs")
    binding = permit["descriptor"]
    _native_descriptor(binding)
    _require(binding["resource"] == root and binding["rank"] == 1 and binding["type"] in {"real", "integer"}
             and binding["intent"] is None and not root.startswith("argument::")
             and not set(binding["attributes"]) & {"allocatable", "pointer", "target", "optional", "volatile", "asynchronous", "value", "parameter"}
             and binding["shape"][0]["kind"] == "Explicit_Shape_Spec", "native opaque WRITE storage differs")
    _bounds(permit["bounds"], permit["element_count"])
    _require(type(permit["storage_bytes"]) is int and 0 < permit["storage_bytes"] < 2**63
             and permit["storage_bytes"] == permit["element_count"] * binding["kind"], "native opaque WRITE byte extent differs")
    for key in ("association_identities", "completion_identities", "joined_team_identities"):
        _identities(permit[key])
    receipts = access["association_receipts"]
    _require(isinstance(receipts, list) and receipts and all(isinstance(r, dict) for r in receipts) and [r.get("identity") for r in receipts] == permit["association_identities"],
             "native opaque WRITE lost exact incoming association receipts")
    call = control["source_call"]
    _require(isinstance(call, dict) and call in manifest.get("resolved_calls", [])
             and all(call.get(k) == permit[p] for k, p in (("caller", "procedure"), ("source", "source"),
                   ("first_line", "first_line"), ("last_line", "last_line"))), "native opaque WRITE original CALL is missing")
    for receipt in receipts:
        _keys(receipt, _ASSOC)
        _sealed(receipt)
        _require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
                 and receipt["proof_role"] == "native_whole_fixed_array_call_association"
                 and all(receipt[k] is False for k in _ASSOC_FALSE) and receipt["requires_original_execution"] is True
                 and receipt["association"] == "original whole-array element order; declared lower bounds retained"
                 and all(receipt[k] == permit[k] for k in ("source_session", "source_context", "selected_node_ids", "source",
                          "first_line", "last_line", "original_selection", "original_call", "element_count"))
                 and receipt["caller"] == permit["procedure"] and receipt["graph_identity"] == permit["source_graph"]
                 and receipt["callee"] == call["procedure"] and receipt["actual_descriptor"] == binding
                 and receipt["actual_bounds"] == permit["bounds"], "native opaque WRITE incoming source association differs")
        _native_descriptor(receipt["formal_descriptor"])
        _bounds(receipt["formal_bounds"], receipt["element_count"])
        _require(type(receipt["actual_index_minus_formal_index"]) is int
                 and receipt["actual_index_minus_formal_index"] == receipt["actual_bounds"][0] - receipt["formal_bounds"][0]
                 and receipt["formal_descriptor"]["intent"] in {"out", "inout"}
                 and receipt["formal_descriptor"]["name"] == receipt["formal"]
                 and all(receipt["formal_descriptor"][k] == binding[k] for k in ("type", "kind", "rank")),
                 "native opaque WRITE incoming lower-bound mapping differs")
        policy = receipt["policy"]
        _keys(policy, {"version", "depth_limit", "procedure_limit", "operation_limit", "element_limit", "bounds_abi"})
        _require(type(policy["version"]) is int and policy["version"] == 1 and policy["element_limit"] is None
                 and all(type(policy[k]) is int and policy[k] > 0 for k in ("depth_limit", "procedure_limit", "operation_limit"))
                 and policy["bounds_abi"] == "default INTEGER bounds and extent", "native opaque association policy differs")
        _original_write_call(call, receipt["source_call"])
        mapping = next((m for m in call["resource_mappings"] if m["formal"] == receipt["formal"]), None)
        _require(mapping is not None and mapping["actual_descriptor"] == binding
                 and mapping["formal_descriptor"] == receipt["formal_descriptor"], "native opaque WRITE target mapping differs")
    incoming = permit["incoming_instances"]
    _require(isinstance(incoming, list) and incoming, "native opaque WRITE incoming instance inventory missing")
    seen = set()
    for item in incoming:
        _keys(item, {"association_identity", "caller", "callee", "formal", "instance", "original_call"})
        receipt = next((r for r in receipts if r["identity"] == item["association_identity"]), None)
        _require(receipt is not None and type(item["instance"]) is int and item["instance"] > 0
                 and all(item[k] == receipt[k] for k in ("caller", "callee", "formal", "original_call")),
                 "native opaque WRITE incoming instance differs")
        key = item["instance"], item["formal"]
        _require(key not in seen, "native opaque WRITE repeats an incoming instance")
        seen.add(key)
    _require({r["association_identity"] for r in incoming} == set(permit["association_identities"]),
             "native opaque WRITE leaves an incoming association unused")
    effects, actions = {}, set()
    _require(isinstance(permit["original_effect_inventory"], list) and permit["original_effect_inventory"],
             "native opaque WRITE complete original effects missing")
    for item in permit["original_effect_inventory"]:
        _keys(item, {"storage", "kind", "procedure", "source", "external"})
        _storage(item["storage"])
        _require(isinstance(item["kind"], str) and item["kind"] in _ACTIONS and type(item["external"]) is bool
                 and _text(item["procedure"]) and _text(item["source"]), "native opaque original effect differs")
        storage = item["storage"]
        if item["external"]:
            _require(not storage["component_path"], "native opaque CALL leaves an unmapped component")
            if storage["resource"] == root:
                actions.add(item["kind"])
            if storage["rank"] and item["kind"] in {"read", "write"}:
                effects.setdefault(storage["resource"], set()).add(item["kind"])
    _require(permit["target_actions"] == sorted(actions) and "write" in actions
             and access["mode"] == ("RW" if "read" in actions else "W")
             and operation["resource_effects"] == {k: sorted(v) for k, v in sorted(effects.items())}
             and root in operation["resources"] and root not in operation["managed_resources"]
             and operation["completion"].get("available") is True, "native opaque WRITE complete effect inventory differs")
    _require(isinstance(permit["storage_disjointness_obligations"], list), "native opaque alias obligations differ")
    opaque_roots = {item["resource"] for item in operation["native_opaque_accesses"]["accesses"]}
    for item in permit["storage_disjointness_obligations"]:
        _keys(item, {"native_storage", "demanded_resource", "demanded_declaring_scope", "actions", "discharged",
                     "distinct_source_names_prove_disjointness", "requirement"})
        _storage(item["native_storage"])
        _texts(item["actions"], nonempty=True)
        _require(item["demanded_resource"] in opaque_roots and _text(item["demanded_declaring_scope"])
                 and isinstance(item["actions"], list) and set(item["actions"]) <= _ACTIONS
                 and item["discharged"] is False and item["distinct_source_names_prove_disjointness"] is False
                 and item["requirement"] == "exact storage-association proof or safe reached runtime range separation",
                 "native opaque source names cannot discharge alias obligations")
    _require(isinstance(permit["original_definition_kills"], list), "native opaque original definition kills differ")
    for item in permit["original_definition_kills"]:
        _keys(item, {"kind", "original_source", "resource", "via"})
        _require(item["kind"] == "original INTENT(OUT) entry kill" and item["resource"] in opaque_roots
                 and _text(item["original_source"]) and _text(item["via"]), "native opaque original entry kill differs")
    metadata = operation.get("host_metadata", {}).get(root)
    _require(isinstance(metadata, dict) and metadata.get("device_capture") is False
             and metadata.get("descriptor") == binding, "native opaque WRITE host BYTE descriptor differs")
    _write_guard(control, permit, procedure, unit, scope)


def _accesses(value, procedure, operation, plan, manifest, unit, scope):
    _keys(value, {"identity", "version", "accesses"} | _FALSE)
    _sealed(value)
    _require(type(value["version"]) is int and value["version"] == 2
             and all(value[key] is False for key in _FALSE), "native opaque access cannot authorize managed storage")
    accesses = value["accesses"]
    _require(isinstance(accesses, list) and accesses, "native opaque access requires a nonempty typed inventory")
    roots = []
    for access in accesses:
        _require(isinstance(access, dict), "native opaque access requires a typed record")
        if access.get("permit_kind") == "original_fixed_array_write":
            _write_access(access, procedure, operation, plan, manifest, unit, scope)
            roots.append(access["resource"])
            continue
        _keys(access, {"resource", "mode", "permit_kind", "permit"})
        root, permit = access["resource"], access["permit"]
        _require(_text(root) and access["mode"] == "R" and access["permit_kind"] == "original_readonly",
                 "native opaque access has no supported typed source permit")
        _keys(permit, _PERMIT)
        _sealed(permit)
        binding = permit["binding"]
        _require(isinstance(binding, dict) and type(binding.get("rank")) is int and binding["rank"] > 0,
                 "native opaque READ requires its original array descriptor")
        _binding(binding, rank=binding["rank"])
        _require(binding["resource"] == root and binding["type"] in {"real", "integer"}
                 and binding["kind"] in {4, 8}
                 and not set(binding["attributes"]) & {"pointer", "optional", "volatile", "asynchronous", "value", "parameter"}
                 and (not root.startswith("argument::") or binding["intent"] == "in"
                      or permit["version"] == 2 and binding["intent"] == "inout"
                      and "allocatable" in binding["attributes"]),
                 "native opaque READ descriptor or borrowed IN role differs")
        _require(type(permit["version"]) is int and permit["version"] in {1, 2}
                 and (permit["version"] == 1 or root.startswith("argument::")
                      and "allocatable" in binding["attributes"] and "target" not in binding["attributes"])
                 and permit["authority"] == "complete original reached native operation"
                 and permit["procedure"] == procedure["procedure"]
                 and permit["source_graph"] == procedure["structured_summary_identity"]
                 and permit["selected_node_ids"] == operation["selected_original_nodes"]
                 and isinstance(permit["selected_node_ids"], list) and permit["selected_node_ids"]
                 and all(_text(node) for node in permit["selected_node_ids"])
                 and all(_sha256(permit[key]) for key in ("analysis_identity", "summary_identity"))
                 and permit["structured_identity"] == operation["structured_summary_identity"]
                 and permit["demand_identity"] == operation["demand_identity"],
                 "native opaque READ source or operation selection differs")
        _require(all(permit[key] is False for key in
                     ("allocation_changes", "calls_or_escapes", "device_capture", "managed_definitions"))
                 and permit["lifetime"] == "original stable native operation; allocation changes close ownership"
                 and permit["range_validation"] == "allocated guard then undefined BYTE reservation; managed aliases close before native work",
                 "native opaque READ lifetime or authority differs")
        effects = operation.get("resource_effects", {}).get(root)
        _require(isinstance(effects, list) and set(effects) <= {"read", "descriptor_read"}
                 and root in operation["resources"] and root not in operation["managed_resources"]
                 and operation["completion"].get("available") is True,
                 "native opaque READ differs from the complete native operation effects")
        metadata = operation.get("host_metadata", {}).get(root)
        _require(isinstance(metadata, dict) and metadata.get("device_capture") is False
                 and _digest(metadata.get("descriptor")) == _digest(binding),
                 "native opaque READ range preparation descriptor differs")
        roots.append(root)
    _require(roots == sorted(set(roots)), "native opaque READ inventory repeats or changes order")
    return accesses


def validate_native_opaque_accesses(epoch, plan, manifest, scope):
    canonical = {item["resource"]: item for item in epoch["resources"]}

    from .source_task_call_record import validate_call_inventory

    inventory = epoch.get("native_opaque_accesses", [])
    _require(isinstance(inventory, list), "native opaque epoch inventory differs")
    recorded = {}
    for row in inventory:
        _keys(row, {"procedure", "unit_ordinal", "operation_id", "access_identity", "resources", "call_projection"})
        key = row["procedure"], row["unit_ordinal"], row["operation_id"]
        _require(_text(row["procedure"]) and type(row["unit_ordinal"]) is int and row["unit_ordinal"] >= 0
                 and type(row["operation_id"]) is int and row["operation_id"] >= 0 and key not in recorded,
                 "native opaque operation identity repeats or differs")
        recorded[key] = row
    expected, opaque = [], set()
    for procedure in plan["procedures"]:
        for unit in procedure["units"]:
            for operation_id, operation in enumerate(unit["operations"].get("native_operations", ())):
                value = operation.get("native_opaque_accesses")
                if value is None:
                    continue
                _require(epoch["version"] == 4, "typed native opaque access requires storage epoch v4")
                _require(type(operation.get("operation_id")) is int and operation["operation_id"] == operation_id
                         and "host_only_native_reads" not in operation,
                         "typed native opaque operation order or legacy role differs")
                accesses = _accesses(value, procedure, operation, plan, manifest, unit, scope)
                selected = operation["selected_original_nodes"]
                _require(all(any(node == parent or node.startswith((parent + "/", parent + ":"))
                                 for parent in unit["selected_node_ids"]) for node in selected),
                         "native opaque READ lies outside its original selected unit")
                key = procedure["procedure"], unit["ordinal"], operation_id
                _require(key in recorded, "native opaque operation is missing from its epoch")
                projection = recorded[key]["call_projection"]
                _keys(projection, {"members", "calls", "field_slots", "field_roots"})
                target_roots, _scope, _plan = validate_call_inventory(projection,
                    {access["resource"]: access for access in accesses}, procedure, canonical, plan["entry"], manifest)
                local_roots = {access["resource"] for access in accesses}
                _require(all(not local_roots.intersection(unit["ordered_definitions"].get(role, ()))
                             for role in ("required_whole", "whole_on_return", "possible_definition_changes")),
                         "native opaque READ cannot establish managed definitions")
                rows = []
                for access in accesses:
                    roots = target_roots[access["resource"]]
                    binding = access["permit"]["descriptor" if access["permit_kind"] == "original_fixed_array_write" else "binding"]
                    for root in roots:
                        resource = canonical[root]
                        _require(resource.get("device_replica_authorized") is False
                                 and resource.get("initialization_authorized") is False
                                 and resource["storage"] == "opaque host reservation"
                                 and all(resource["descriptor"][key] == binding[key] for key in ("type", "kind", "rank")),
                                 "native opaque READ conflicts with a canonical numeric role or descriptor")
                    opaque.update(roots)
                    rows.append({"resource": access["resource"], "canonical_resources": sorted(roots),
                                 "mode": access["mode"], "permit_identity": access["permit"]["identity"]})
                expected.append({"procedure": procedure["procedure"], "unit_ordinal": unit["ordinal"],
                    "operation_id": operation_id, "access_identity": value["identity"], "resources": rows,
                    "call_projection": projection})
    if epoch["version"] == 4:
        _require(expected and epoch["native_opaque_accesses"] == expected,
                 "storage epoch native opaque inventory differs from its exact original operations and calls")
    else:
        _require(not expected, "legacy storage epoch cannot carry typed native opaque access")
    managed = {root for procedure in plan["procedures"] for unit in procedure["units"]
               for root in unit["resources"].values()}
    scalar_backing = {item["resource"] for item in scope.get("resource_bindings", {}).get("resources", ())
                      if item.get("role") == "opaque_scalar_element_alias_reservation"}
    _require(not opaque & (managed | scalar_backing), "native opaque storage overlaps a numeric or scalar backing role")
