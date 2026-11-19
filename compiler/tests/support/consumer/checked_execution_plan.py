"""Validate compiler-issued diagnostic plans without choosing execution.

The compiler retains the executable plan and its proof objects. This consumer
checks the versioned public record's integrity and source links only; operation,
definition, policy and cost facts do not authorize additional transformations.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_PLAN_FIELDS = {
    "schema_version", "identity", "execution_identity", "entry", "sources", "source_inputs_identity",
    "configuration_identity", "capture_facts_identity", "procedures", "resources",
    "numerical_invocations", "gpu_leaves", "policy", "cost_requirements",
}
_EPOCH_FIELDS = {
    "identity", "version", "entry", "source_inputs", "resources", "members", "activation", "lifetime",
    "generation", "readonly", "writes", "return", "definitions_authorized", "placement_authorized",
    "cross_cut_retention_authorized",
}
_EPOCH_POLICY = {
    "activation": "first reached managed access; no descriptor evaluation on zero-trip paths",
    "lifetime": "checked original invocation between mandatory retirement cuts",
    "generation": "runtime-checked allocation generation; never inferred from address reuse",
    "readonly": "add a valid replica; preserve other current replicas",
    "writes": "invalidate only overlapping regions after ordered completion",
    "return": "checked children borrow; unmanaged return publishes observable storage",
}
_TRANSFER_FIELDS = {
    "identity", "version", "procedure", "source_graph", "selected_node_ids", "callee", "source_inputs", "call",
    "body", "resources", "transition", "payload_reads", "payload_writes", "initialization_authorized",
    "runtime_guard", "on_unproved_mutation",
}
_ORIGINAL_CALL_FIELDS = {
    "schema_version", "procedure", "target", "source_kind", "formal_arguments", "actual_arguments",
    "original_arguments", "associations", "actual_evaluation", "effect_authority", "capture_authority",
    "alias_authority", "completion_authority", "execution_authority",
}
_PROCEDURE_FIELDS = {
    "procedure", "source", "source_sha256", "structured_summary_identity", "emission_identity",
    "units", "boundaries", "calls",
}
_UNIT_FIELDS = {
    "ordinal", "selected_node_ids", "first_line", "last_line", "resources", "ordered_definitions", "operations",
}


def _require(condition, reason):
    if not condition:
        raise ValueError("Compiler checked execution plan " + reason)


def _text(value):
    return isinstance(value, str) and bool(value)


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _digest(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Compiler checked execution plan requires finite JSON facts") from error
    return hashlib.sha256(encoded).hexdigest()


def _source_inputs_identity(manifest):
    analysis = manifest.get("analysis_sources")
    _require(analysis is None or isinstance(analysis, dict), "requires configured source metadata")
    if analysis is None:
        return _digest({"schema_version": 1, "sources": manifest["source_inputs"],
                        "dependencies": {}, "configuration": None, "entries": []})
    entries = analysis.get("entries")
    _require(isinstance(entries, list) and all(isinstance(item, dict) and _text(item.get("source"))
                                              for item in entries), "requires configured source entries")
    _require(analysis.get("source_inputs") == manifest["source_inputs"], "configured source inputs differ")
    return _digest({"schema_version": 1, "sources": manifest["source_inputs"],
                    "dependencies": analysis.get("dependencies"), "configuration": analysis.get("configuration"),
                    "entries": sorted(entries, key=lambda item: item["source"])})


def validate_checked_execution_plans(manifest, *, capture_facts=None):
    """Check historical and versioned storage-epoch records in scope schema 1.

    Missing records preserve compatibility with earlier compiler packages.
    Capture facts are checked against the request when available. Their compact
    plan identity differs from the outer manifest's legacy JSON hash recipe.
    Saved packages retain both identities unchanged and remain subject to the
    existing saved-manifest, artifact and source verification before application.
    """
    if "composition" in manifest or "component_manifests" in manifest:
        from .composed_plan_record import validate_composed_plans
        validate_composed_plans(manifest, capture_facts=capture_facts)
        return
    if "provider_only_lifetime" in manifest:
        from .provider_lifetime_record import validate_provider_only_lifetime
        validate_provider_only_lifetime(manifest)
    scopes = manifest.get("scopes", [])
    _require(isinstance(scopes, list), "requires a list of source scopes")
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        _require("external_device_tasks" not in scope or "checked_execution_plan" in scope,
                 "external device tasks require their checked plan")
        _require("source_device_tasks" not in scope or "checked_execution_plan" in scope,
                 "source device tasks require their checked plan")
        _require("storage_epoch" not in scope or "checked_execution_plan" in scope,
                 "storage epoch requires its checked plan")
        _require("owner_local_call_group" not in scope or "checked_execution_plan" in scope,
                 "owner-local call group requires its checked plan")
        _require("supported_path_epoch" not in scope or "checked_execution_plan" in scope,
                 "supported path requires its checked plan")
        if "checked_execution_plan" not in scope:
            continue
        _validate_plan(scope["checked_execution_plan"], manifest, scope, capture_facts)
        _require("storage_epoch" not in scope or "storage_epoch" in scope["checked_execution_plan"]
                 and _digest(scope["storage_epoch"]) == _digest(scope["checked_execution_plan"]["storage_epoch"]),
                 "original worksharing scope epoch differs from its checked plan")
        _require("owner_local_call_group" not in scope or scope["checked_execution_plan"]["schema_version"] == 4,
                 "owner-local call group requires a v4 checked plan")
        _require("supported_path_epoch" not in scope or scope["checked_execution_plan"]["schema_version"] == 5,
                 "supported path requires a v5 checked plan")
        if "external_device_tasks" in scope:
            plan = scope["checked_execution_plan"]
            _require(plan["schema_version"] in (2, 3, 4, 5), "external device tasks require a storage-epoch checked plan")
            records = [task for member in plan["storage_epoch"]["members"] for task in member.get("device_tasks", [])]
            _require(bool(records) and _digest(scope["external_device_tasks"]) == _digest(records),
                     "external device tasks differ from their checked epoch")
        if "source_device_tasks" in scope:
            plan = scope["checked_execution_plan"]
            _require(plan["schema_version"] in (2, 3, 4, 5), "source device tasks require a storage-epoch checked plan")
            records = [task for member in plan["storage_epoch"]["members"]
                       for task in member.get("source_device_tasks", [])]
            _require(bool(records) and _digest(scope["source_device_tasks"]) == _digest(records),
                     "source device tasks differ from their checked epoch")

    from .original_worksharing_record import validate_original_worksharing_constraints

    validate_original_worksharing_constraints(manifest)


def _validate_plan(plan, manifest, scope, capture_facts):
    _require(isinstance(plan, dict), "requires the complete v1 schema")
    version = plan.get("schema_version")
    _require(type(version) is int and version in (1, 2, 3, 4, 5),
             "has an unsupported schema version")
    fields = _PLAN_FIELDS if version == 1 else _PLAN_FIELDS | {"storage_epoch"}
    _require(set(plan) == fields, "requires the complete v" + str(version) + " schema")
    _require(all(_sha256(plan[key]) for key in ("identity", "execution_identity", "source_inputs_identity",
                                               "configuration_identity", "capture_facts_identity")),
             "requires SHA256 identities")
    _require(plan["identity"] == _digest({key: value for key, value in plan.items() if key != "identity"}),
             "identity differs from its contents")
    sources = plan["sources"]
    _require(isinstance(sources, dict) and bool(sources)
             and all(_text(path) and Path(path).is_absolute() and _sha256(digest)
                     for path, digest in sources.items())
             and sources == manifest.get("source_inputs"), "original source identities differ")
    _require(plan["source_inputs_identity"] == _source_inputs_identity(manifest),
             "configured source identity differs")
    if capture_facts is not None:
        _require(plan["capture_facts_identity"] == _digest(capture_facts), "capture facts identity differs")
    _require(_text(plan["entry"]) and plan["entry"] == manifest.get("entry"), "entry differs from its manifest")

    resources = plan["resources"]
    _require(isinstance(resources, list), "requires canonical resource records")
    roots, registrations = set(), set()
    for resource in resources:
        _require(isinstance(resource, dict) and set(resource) == {"resource", "registration_identity", "descriptor"}
                 and _text(resource["resource"]) and isinstance(resource["descriptor"], dict),
                 "requires canonical resource descriptors")
        number = resource["registration_identity"]
        _require(type(number) is int and number > 0 and number not in registrations
                 and resource["resource"] not in roots, "repeats or invalidates canonical registrations")
        roots.add(resource["resource"])
        registrations.add(number)

    procedures = plan["procedures"]
    _require(isinstance(procedures, list) and bool(procedures), "requires original procedure records")
    names = set()
    for procedure in procedures:
        _require(isinstance(procedure, dict) and set(procedure) == _PROCEDURE_FIELDS,
                 "requires complete procedure records")
        name, source = procedure["procedure"], procedure["source"]
        _require(_text(name) and name not in names, "repeats or invalidates an original procedure")
        names.add(name)
        _require(_text(source) and source in sources and procedure["source_sha256"] == sources[source]
                 and _sha256(procedure["structured_summary_identity"]), "procedure source linkage differs")
        _require(_sha256(procedure["emission_identity"]), "requires a procedure emission SHA256 identity")
        _require(all(isinstance(procedure[key], list) and all(isinstance(item, dict) for item in procedure[key])
                     for key in ("boundaries", "calls")), "requires native boundary and call records")
        units = procedure["units"]
        _require(isinstance(units, list), "requires ordered procedure units")
        for ordinal, unit in enumerate(units):
            _require(isinstance(unit, dict) and set(unit) == _UNIT_FIELDS, "requires complete unit records")
            _require(type(unit["ordinal"]) is int and unit["ordinal"] == ordinal, "unit ordinals are not ordered")
            _require(isinstance(unit["selected_node_ids"], list)
                     and all(_text(node) for node in unit["selected_node_ids"]), "requires original node identities")
            _require(type(unit["first_line"]) is int and type(unit["last_line"]) is int
                     and 1 <= unit["first_line"] <= unit["last_line"], "requires original ordered line spans")
            _require(isinstance(unit["resources"], dict)
                     and all(_text(local) and _text(root) and root in roots
                             for local, root in unit["resources"].items()), "unit canonical resource linkage differs")
            _require(isinstance(unit["ordered_definitions"], dict) and isinstance(unit["operations"], dict),
                     "requires definition and operation records")
            from .original_host_access_record import validate_original_host_accesses

            validate_original_host_accesses(procedure, unit, manifest)
    root = procedures[0]
    _require(root["procedure"] == plan["entry"] and root["source"] == scope.get("path"),
             "root procedure differs from its source scope")
    authority = scope.get("source_authority")
    _require(isinstance(authority, dict)
             and authority.get("structured_summary_identity") == root["structured_summary_identity"],
             "root summary differs from its source scope")
    _require(type(plan["numerical_invocations"]) is int and plan["numerical_invocations"] >= 0,
             "requires a nonnegative numerical invocation count")
    _require(isinstance(plan["gpu_leaves"], list) and all(_text(leaf) for leaf in plan["gpu_leaves"])
             and len(set(plan["gpu_leaves"])) == len(plan["gpu_leaves"]), "requires distinct numerical leaf identities")
    _require(isinstance(plan["policy"], dict), "requires compiler policy facts")
    _require(isinstance(plan["cost_requirements"], list)
             and all(isinstance(item, dict) and _text(item.get("kind")) and _text(item.get("reason"))
                     for item in plan["cost_requirements"]), "requires explanatory cost requirements")
    if version in (2, 3, 4, 5):
        _validate_epoch(plan["storage_epoch"], plan, manifest, capture_facts, scope)
        execution = {"execution_identity_version": 2,
                     **{key: plan[key] for key in ("entry", "sources", "source_inputs_identity", "configuration_identity",
                                                   "capture_facts_identity", "resources", "numerical_invocations", "gpu_leaves")},
                     "procedures": [{"procedure": item["procedure"], "emission_identity": item["emission_identity"]}
                                    for item in procedures], "storage_epoch_identity": plan["storage_epoch"]["identity"]}
        _require(plan["execution_identity"] == _digest(execution), "storage epoch execution identity differs")


def _validate_epoch(epoch, plan, manifest, capture_facts, scope):
    fields = _EPOCH_FIELDS | ({"return_contract"} if plan["schema_version"] == 3 else set())
    _require(isinstance(epoch, dict) and type(epoch.get("version")) is int
             and epoch["version"] in (1, 2, 3, 4), "storage epoch has an unsupported version")
    if epoch["version"] == 2 or epoch["version"] in (3, 4) and "private_allocation_obligations" in epoch:
        fields = fields | {"private_allocation_obligations"}
    if epoch["version"] == 3 or epoch["version"] == 4 and "original_worksharing_storage_obligations" in epoch:
        fields = fields | {"original_worksharing_storage_obligations"}
    if epoch["version"] == 4:
        fields = fields | {"native_opaque_accesses"}
    allowed = (fields | {"supported_path_epoch"} if plan["schema_version"] == 5
               else fields | {"selected_position_epoch"})
    _require(isinstance(epoch, dict) and (set(epoch) == allowed if plan["schema_version"] in (4, 5)
                                        else set(epoch) in (fields, allowed)),
             f"storage epoch requires the complete v{epoch['version']} schema")
    _require(_sha256(epoch["identity"])
             and epoch["identity"] == _digest({key: value for key, value in epoch.items() if key != "identity"}),
             "storage epoch identity differs from its contents")
    _require(epoch["entry"] == plan["entry"], "storage epoch entry differs")
    inputs = epoch["source_inputs"]
    _require(isinstance(inputs, dict)
             and set(inputs) == {"schema_version", "sources", "dependencies", "configuration", "entries"}
             and type(inputs["schema_version"]) is int and inputs["schema_version"] == 1
             and inputs["sources"] == plan["sources"] and _digest(inputs) == plan["source_inputs_identity"],
             "storage epoch source inputs differ")
    _require(all(epoch[key] == value for key, value in _EPOCH_POLICY.items()), "storage epoch lifetime policy differs")
    _require(all(epoch[key] is False for key in ("definitions_authorized", "placement_authorized",
                                                "cross_cut_retention_authorized")),
             "storage epoch cannot authorize definitions, placement or retention across cuts")
    resources = epoch["resources"]
    canonical = {item["resource"]: item for item in plan["resources"]}
    if "private_allocation_obligations" in epoch:
        obligations = epoch["private_allocation_obligations"]
        _require(isinstance(obligations, list) and obligations,
                 "storage epoch requires nonempty private allocation obligations")
        keys = []
        for item in obligations:
            _require(isinstance(item, dict) and set(item) == {
                "descriptor_identity", "descriptor_procedure", "unregistered_private_allocations"}
                and _sha256(item["descriptor_identity"])
                and isinstance(item["descriptor_procedure"], str)
                and re.fullmatch(r"[a-z][a-z0-9_]*(?:::[a-z][a-z0-9_]*)+", item["descriptor_procedure"]),
                "storage epoch private allocation descriptor identity differs")
            allocations = item["unregistered_private_allocations"]
            _require(isinstance(allocations, list) and allocations,
                     "storage epoch private allocation lacks its declaring identity")
            roots = []
            for allocation in allocations:
                _require(isinstance(allocation, dict) and set(allocation) == {"resource", "declaring_procedure"}
                         and isinstance(allocation["resource"], str)
                         and isinstance(allocation["declaring_procedure"], str)
                         and re.fullmatch(r"[a-z][a-z0-9_]*(?:::[a-z][a-z0-9_]*)+", allocation["declaring_procedure"])
                         and re.fullmatch(re.escape(allocation["declaring_procedure"]) + r"::[a-z][a-z0-9_]*",
                                          allocation["resource"])
                         and allocation["resource"] not in canonical,
                         "storage epoch private allocation overlaps a registration or lacks its declaring identity")
                roots.append(allocation["resource"])
            _require(roots == sorted(set(roots)),
                     "storage epoch private allocation overlaps a registration or lacks its declaring identity")
            keys.append((item["descriptor_procedure"], item["descriptor_identity"]))
        _require(keys == sorted(set(keys)), "storage epoch private allocation obligations repeat or change order")
    if "original_worksharing_storage_obligations" in epoch:
        from .original_worksharing_record import validate_original_worksharing

        validate_original_worksharing(epoch, plan)
    managed = {root for procedure in plan["procedures"] for unit in procedure["units"]
               for root in unit["resources"].values()}
    _require(isinstance(resources, list) and len(resources) == len(canonical), "storage epoch canonical resources differ")
    _require([item.get("resource") for item in resources if isinstance(item, dict)] == sorted(canonical),
             "storage epoch canonical resource order differs")
    for item in resources:
        original = canonical[item["resource"]]
        _require(_digest({key: item.get(key) for key in original}) == _digest(original),
                 "storage epoch canonical descriptor differs")
        if item.get("device_replica_authorized") is True:
            _require(set(item) == set(original) | {"storage", "lifetime", "allocation_changes", "escapes", "device_replica_authorized"}
                     and item["storage"] == "stable" and item["lifetime"] in ("invocation", "selected_window")
                     and item["allocation_changes"] is False and item["escapes"] is False,
                     "storage epoch managed capture facts differ")
        else:
            _require(set(item) == set(original) | {"storage", "lifetime", "device_replica_authorized", "initialization_authorized"}
                     and item["device_replica_authorized"] is False and item["initialization_authorized"] is False
                     and item["resource"] not in managed
                     and item["storage"] == "opaque host reservation"
                     and item["lifetime"] == "original checked native access",
                     "storage epoch opaque reservation cannot authorize capture or initialization")
    members = epoch["members"]
    _require(isinstance(members, list) and len(members) == len(plan["procedures"]), "storage epoch members differ")
    provider_slot = 0
    for member, procedure in zip(members, plan["procedures"], strict=True):
        required = {"procedure", "source_graph", "units", "storage_transfers", "retirement_cuts"}
        _require(isinstance(member, dict)
                 and required <= set(member) <= required | {"device_tasks", "source_device_tasks"},
                 "storage epoch requires complete member records")
        _require(member["procedure"] == procedure["procedure"]
                 and member["source_graph"] == procedure["structured_summary_identity"],
                 "storage epoch member source graph differs")
        expected = [{"ordinal": unit["ordinal"], "selected_node_ids": unit["selected_node_ids"],
                     "canonical_resources": unit["resources"]} for unit in procedure["units"]]
        _require(_digest(member["units"]) == _digest(expected), "storage epoch original selected units or canonical mappings differ")
        _require(_digest(member["retirement_cuts"]) == _digest(procedure["boundaries"]), "storage epoch mandatory retirement cuts differ")
        transfers = member["storage_transfers"]
        _require(isinstance(transfers, list), "storage epoch requires original storage transfer records")
        seen = set()
        for transfer in transfers:
            _validate_storage_transfer(transfer, member, inputs)
            _require(transfer["identity"] not in seen, "storage epoch repeats an original storage transfer")
            seen.add(transfer["identity"])
        if "device_tasks" in member:
            from .device_task_record import validate_device_task

            tasks = member["device_tasks"]
            _require(isinstance(tasks, list) and bool(tasks), "device tasks require a nonempty explicit list")
            seen_tasks = set()
            for task in tasks:
                if isinstance(task, dict) and type(task.get("version")) is int and task["version"] == 2:
                    provider_slot += 1
                validate_device_task(task, member, procedure, inputs, {item["resource"]: item for item in resources},
                                     owner=plan["entry"], provider_slot=provider_slot,
                                     provider_lifetime_identity=manifest.get("provider_only_lifetime", {}).get("identity"))
                _require(task["identity"] not in seen_tasks, "device task repeats an original call")
                seen_tasks.add(task["identity"])
        if "source_device_tasks" in member:
            from .source_task_record import validate_source_task

            tasks = member["source_device_tasks"]
            _require(isinstance(tasks, list) and bool(tasks), "source device tasks require a nonempty explicit list")
            seen_tasks = set()
            for ordinal, task in enumerate(tasks):
                validate_source_task(task, member, procedure, inputs,
                                     {item["resource"]: item for item in resources},
                                     owner=plan["entry"], manifest=manifest, capture_facts=capture_facts,
                                     graph_execution=scope.get("graph_execution"), task_ordinal=ordinal)
                _require(task["identity"] not in seen_tasks, "source device task repeats an original team")
                seen_tasks.add(task["identity"])
    from .native_opaque_access_record import validate_native_opaque_accesses

    validate_native_opaque_accesses(epoch, plan, manifest, scope)
    if plan["schema_version"] == 5:
        from .supported_path_epoch_record import validate_supported_path_epoch

        validate_supported_path_epoch(epoch["supported_path_epoch"], epoch, plan, manifest, scope,
                                      capture_facts=capture_facts)
    elif plan["schema_version"] == 4:
        from .owner_local_call_group_record import validate_owner_local_call_group

        validate_owner_local_call_group(epoch["selected_position_epoch"], epoch, plan, manifest, scope,
                                       capture_facts=capture_facts)
    elif "selected_position_epoch" in epoch:
        from .selected_epoch_record import validate_selected_epoch

        validate_selected_epoch(epoch["selected_position_epoch"], epoch, plan, scope)
    if plan["schema_version"] == 3:
        from .conditional_return_record import validate_conditional_return

        validate_conditional_return(epoch["return_contract"], epoch, plan, manifest, scope)


def _validate_storage_transfer(record, member, inputs):
    _require(isinstance(record, dict) and set(record) == _TRANSFER_FIELDS,
             "storage transfer requires the complete v1 schema")
    _require(type(record["version"]) is int and record["version"] == 1, "storage transfer has an unsupported version")
    _require(_sha256(record["identity"])
             and record["identity"] == _digest({key: value for key, value in record.items() if key != "identity"}),
             "storage transfer identity differs from its contents")
    _require(record["procedure"] == member["procedure"] and record["source_graph"] == member["source_graph"]
             and record["source_inputs"] == inputs and _text(record["callee"]), "storage transfer source linkage differs")
    nodes, resources = record["selected_node_ids"], record["resources"]
    _require(isinstance(nodes, list) and len(nodes) == 1 and _text(nodes[0])
             and nodes[0].startswith(member["procedure"] + "/"), "storage transfer original call selection differs")
    _require(isinstance(resources, list) and len(resources) == 2 and all(_text(root) for root in resources)
             and len(set(resources)) == 2, "storage transfer requires distinct original storage slots")
    _require(all(record[key] is False for key in ("payload_reads", "payload_writes", "initialization_authorized")),
             "storage transfer cannot authorize payload access or initialization")
    _require(record["transition"] == "exchange allocation associations; physical generations and replicas survive"
             and record["runtime_guard"] == "both current layouts match opposite live allocations after the original call"
             and record["on_unproved_mutation"] == "publish and retire before original source operation",
             "storage transfer descriptor transition policy differs")
    call = record["call"]
    _require(isinstance(call, dict) and set(call) == _ORIGINAL_CALL_FIELDS
             and type(call["schema_version"]) is int and call["schema_version"] == 1,
             "storage transfer requires the exact original call schema")
    _require(call["procedure"] == record["callee"] and _text(call["target"])
             and call["source_kind"] in ("module", "internal", "external")
             and call["actual_evaluation"] == "unchanged original call position and guards"
             and all(call[key] is False for key in ("effect_authority", "capture_authority", "alias_authority",
                                                  "completion_authority", "execution_authority")),
             "storage transfer original call cannot grant effect or execution authority")
    formals, actuals = call["formal_arguments"], call["actual_arguments"]
    _require(isinstance(formals, list) and len(formals) == 2
             and all(isinstance(name, str) and re.fullmatch(r"[a-z]\w*", name) for name in formals)
             and len(set(formals)) == 2
             and isinstance(actuals, list) and len(actuals) == 2 and all(_text(name) for name in actuals),
             "storage transfer original arguments differ")
    arguments, associations = call["original_arguments"], call["associations"]
    _require(isinstance(arguments, list) and len(arguments) == 2
             and isinstance(associations, list) and len(associations) == 2,
             "storage transfer original associations differ")
    seen = set()
    for index, argument in enumerate(arguments):
        _require(isinstance(argument, dict) and set(argument) == {"position", "formal", "keyword", "actual"}
                 and type(argument["position"]) is int and argument["position"] == index
                 and argument["formal"] in formals and argument["formal"] not in seen
                 and (argument["keyword"] is None or isinstance(argument["keyword"], str)
                      and argument["keyword"].lower() == argument["formal"])
                 and argument["actual"] == actuals[formals.index(argument["formal"])],
                 "storage transfer original argument order differs")
        seen.add(argument["formal"])
    for index, association in enumerate(associations):
        _require(isinstance(association, dict)
                 and set(association) == {"formal", "actual", "formal_descriptor", "actual_type", "entry_definition_kill"}
                 and association["formal"] == formals[index] and association["actual"] == actuals[index]
                 and association["entry_definition_kill"] is False, "storage transfer original association differs")
        descriptor, actual_type = association["formal_descriptor"], association["actual_type"]
        _require(isinstance(descriptor, dict)
                 and set(descriptor) == {"name", "resource", "type", "kind", "rank", "intent", "attributes", "lower_bounds", "shape"}
                 and descriptor["name"] == formals[index] and descriptor["resource"] == "argument::" + formals[index]
                 and descriptor["type"] in ("real", "integer") and type(descriptor["kind"]) is int
                 and descriptor["intent"] == "inout" and isinstance(descriptor["attributes"], list)
                 and all(_text(value) for value in descriptor["attributes"])
                 and descriptor["attributes"] == sorted(set(descriptor["attributes"]))
                 and "allocatable" in descriptor["attributes"]
                 and not set(descriptor["attributes"]) & {"pointer", "optional", "volatile", "asynchronous", "value", "save", "target", "protected", "parameter"}
                 and type(descriptor["rank"]) is int and descriptor["rank"] > 0,
                 "storage transfer requires original allocatable INOUT descriptors")
        rank = descriptor["rank"]
        _require(isinstance(descriptor["lower_bounds"], list) and len(descriptor["lower_bounds"]) == rank
                 and all(_text(bound) for bound in descriptor["lower_bounds"])
                 and isinstance(descriptor["shape"], list) and len(descriptor["shape"]) == rank,
                 "storage transfer descriptor shape differs")
        for axis in descriptor["shape"]:
            _require(isinstance(axis, dict) and set(axis) == {"kind", "source", "bounds"}
                     and _text(axis["kind"]) and _text(axis["source"]) and isinstance(axis["bounds"], list)
                     and all(bound is None or _text(bound) for bound in axis["bounds"]),
                     "storage transfer descriptor axis differs")
        _require(isinstance(actual_type, dict)
                 and set(actual_type) == {"type", "kind", "rank", "storage_resource", "declaring_type"}
                 and actual_type["storage_resource"] == resources[index] and actual_type["declaring_type"] is None
                 and all(actual_type[key] == descriptor[key] for key in ("type", "kind", "rank"))
                 and type(actual_type["kind"]) is int and type(actual_type["rank"]) is int
                 and (actual_type["type"], actual_type["kind"]) in {("real", 4), ("real", 8), ("integer", 4)},
                 "storage transfer actual storage or numeric descriptor differs")
    _require(all(associations[0]["formal_descriptor"][key] == associations[1]["formal_descriptor"][key]
                 for key in ("type", "kind", "rank")), "storage transfer descriptors have different element types or ranks")
    body = record["body"]
    _require(isinstance(body, list) and len(body) == 3 and all(isinstance(line, str) for line in body),
             "storage transfer requires the original three MOVE_ALLOC statements")
    pairs = [re.fullmatch(r"\s*call\s+move_alloc\s*\(\s*([a-z]\w*)\s*,\s*([a-z]\w*)\s*\)\s*", line, re.IGNORECASE)
             for line in body]
    _require(all(pairs), "storage transfer requires the original three MOVE_ALLOC statements")
    moves = [tuple(name.lower() for name in pair.groups()) for pair in pairs]
    temporary = moves[0][1]
    _require(temporary not in formals and moves == [(formals[0], temporary), (formals[1], formals[0]), (temporary, formals[1])],
             "storage transfer original MOVE_ALLOC order differs")
