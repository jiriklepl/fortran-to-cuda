"""Integrity checks for explanatory external-device source proofs, never placement."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .checked_execution_plan import (
    _ORIGINAL_CALL_FIELDS,
    _digest,
    _require,
    _sha256,
    _text,
)

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,62}\Z")
_COMPONENT = re.compile(r"[a-z][a-z0-9_]{0,62}(?:%[a-z][a-z0-9_]{0,62})*\Z")
_FIELDS = {"identity", "version", "procedure", "source_graph", "selected_node_ids", "source_inputs", "call",
           "contract", "provider", "canonical_resources", "source_association_authorized", "definitions_authorized",
           "placement_authorized", "numerical_equivalence", "lifetime", "position", "source_effects"}


def _keys(record, required, optional=()):
    _require(isinstance(record, dict) and set(required) <= set(record) <= set(required) | set(optional),
             "device task requires exact record fields")


def _reference(record, *, path_key="path"):
    _keys(record, (path_key, "sha256"))
    _require(_text(record[path_key]) and Path(record[path_key]).is_absolute() and _sha256(record["sha256"]),
             "device task artifact reference differs")
    try:
        with Path(record[path_key]).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as error:
        raise ValueError("Compiler checked execution plan device task artifact is unavailable") from error
    _require(actual == record["sha256"], "device task artifact content differs")
    return Path(record[path_key])


def _expression(value, ranks):
    if type(value) is int and -(1 << 63) <= value < (1 << 63):
        return
    _keys(value, ("descriptor", "formal", "axis", "offset"))
    _require(value["descriptor"] in ("size", "lbound") and _text(value["formal"]) and value["formal"] in ranks
             and type(value["axis"]) is int and 1 <= value["axis"] <= ranks[value["formal"]]
             and type(value["offset"]) is int and -(1 << 31) <= value["offset"] < (1 << 31),
             "device task descriptor expression differs")


def _provider(record, contract):
    path = _reference(contract["provider_manifest"])
    try:
        current = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Compiler checked execution plan device provider requires JSON") from error
    _require(_digest(current) == _digest(record), "device task provider manifest differs")
    version = contract["version"]
    _keys(record, ("version", "kind", "fortran", "artifacts", "numerical_contract", "execution",
                   *(("metadata_query",) if version == 2 else ())), ("pressure_backend", "retained_state_qualification"))
    _require(type(record["version"]) is int and record["version"] == version and record["kind"] == "external_device_provider",
             "device task provider version differs")
    _keys(record["fortran"], ("module", "config_type", "create", "execute", "destroy"))
    _require(all(isinstance(value, str) and _IDENTIFIER.fullmatch(value) for value in record["fortran"].values()),
             "device task provider interface differs")
    artifacts = record["artifacts"]
    _require(isinstance(artifacts, dict) and bool(artifacts), "device task provider requires implementation artifacts")
    for name, digest in artifacts.items():
        _reference({"path": name, "sha256": digest})
    numeric = record["numerical_contract"]
    _keys(numeric, ("identity", "qualification", "original_bodies", "task_contracts"))
    _require(_sha256(numeric["identity"]) and isinstance(numeric["original_bodies"], list)
             and all(_sha256(value) for value in numeric["original_bodies"])
             and contract["source"]["callee_body_sha256"] in numeric["original_bodies"],
             "device task numerical qualification lost its original body")
    scope = {"original_body": contract["source"]["callee_body_sha256"],
             "resources": contract["resources"], "configuration": contract["configuration"]}
    if version == 2:
        scope.update(opaque_arguments=contract["opaque_arguments"], completion=contract["completion"])
    if "entry_read_requirements" in contract:
        scope["entry_read_requirements"] = contract["entry_read_requirements"]
    task_scope = _digest(scope)
    _require(isinstance(numeric["task_contracts"], list) and all(_sha256(value) for value in numeric["task_contracts"])
             and task_scope in numeric["task_contracts"], "device task numerical qualification lost its exact boxes or configuration")
    _reference(numeric["qualification"])
    _require(record["execution"] == {"stream": "supplied", "completion": "same_stream", "workspace": "domain_claim",
                                      "fallback": "prework_only"}, "device task provider stream or fallback differs")
    if "retained_state_qualification" in record:
        from .provider_lifetime_record import validate_provider_retained_state
        _require(version == 2, "provider-only lifetime requires provider version2")
        validate_provider_retained_state(record)
    if "pressure_backend" in record:
        _reference(record["pressure_backend"], path_key="manifest")
    if version == 2:
        query = record["metadata_query"]
        _keys(query, ("module", "procedure", "opaque_formals", "configuration_argument", "read_only",
                      "configuration_components"))
        names = query["opaque_formals"]
        _require(all(isinstance(query[key], str) and _IDENTIFIER.fullmatch(query[key])
                     for key in ("module", "procedure")) and query["configuration_argument"] == "config"
                 and query["read_only"] is True and isinstance(names, list) and bool(names)
                 and all(isinstance(name, str) and _IDENTIFIER.fullmatch(name) for name in names)
                 and len(set(names)) == len(names)
                 and names == [item["formal"] for item in contract["opaque_arguments"]],
                 "device task metadata query opaque interface differs")
        components = query["configuration_components"]
        _require(isinstance(components, list) and bool(components), "device task metadata query components differ")
        seen = set()
        for component in components:
            _keys(component, ("component", "kind", "count"))
            name = component["component"]
            _require(isinstance(name, str) and _COMPONENT.fullmatch(name) and name not in seen
                     and component["kind"] in ("int32", "int64", "size", "real64")
                     and type(component["count"]) is int and 1 <= component["count"] <= 64,
                     "device task metadata query typed component differs")
            seen.add(name)


def _opaque_state(state, association, contract, record, inputs):
    _keys(state, ("identity", "version", "procedure", "source_graph", "source_inputs", "selected_node_ids", "formal",
                  "resource", "descriptor", "actual_type", "constructor", "qualification", "observers", "workspace_effects",
                  "capture_authorized", "definitions_authorized", "gpu_access_authorized", "metadata_reads_authorized", "storage"))
    _require(type(state["version"]) is int and state["version"] == 1 and _sha256(state["identity"])
             and state["identity"] == _digest({key: value for key, value in state.items() if key != "identity"})
             and state["procedure"] == record["procedure"] and state["source_graph"] == record["source_graph"]
             and _digest(state["source_inputs"]) == _digest(inputs)
             and state["selected_node_ids"] == record["selected_node_ids"]
             and state["formal"] == contract["formal"] and state["actual_type"] == association["actual_type"]
             and state["resource"] == association["actual_type"]["storage_resource"]
             and state["resource"] == record["procedure"] + "::" + association["actual"].lower(),
             "device task opaque state original identity differs")
    descriptor = state["descriptor"]
    _keys(descriptor, ("name", "resource", "type", "kind", "rank", "intent", "attributes", "lower_bounds", "shape"))
    attributes = descriptor["attributes"]
    _require(descriptor["name"] == association["actual"].lower() and descriptor["resource"] == state["resource"]
             and descriptor["type"] == association["actual_type"]["type"] and descriptor["kind"] is None
             and type(descriptor["rank"]) is int and descriptor["rank"] == 0 and descriptor["intent"] is None
             and descriptor["lower_bounds"] == [] and descriptor["shape"] == []
             and isinstance(attributes, list) and all(_text(name) for name in attributes)
             and attributes == sorted(set(attributes)) and "save" in attributes
             and not set(attributes) & {"pointer", "target", "allocatable", "optional", "volatile", "asynchronous", "value"},
             "device task opaque state requires exact local SAVE storage")
    constructor = state["constructor"]
    _keys(constructor, ("span", "statement_sha256", "selected_node_ids", "execution"))
    _require({key: constructor[key] for key in ("span", "statement_sha256")} == contract["constructor"]
             and isinstance(constructor["selected_node_ids"], list) and bool(constructor["selected_node_ids"])
             and all(_text(node) and node.startswith(record["procedure"] + "/") for node in constructor["selected_node_ids"])
             and constructor["execution"] == "unchanged original source and guards"
             and state["qualification"] == contract["qualification"], "device task opaque constructor or qualification differs")
    _reference(state["qualification"])
    _require(all(state[key] is False for key in ("capture_authorized", "definitions_authorized", "gpu_access_authorized",
                                                "metadata_reads_authorized"))
             and state["observers"] == "only original constructor LHS and exact contracted receiver argument"
             and state["workspace_effects"] == "explicit supplied private-scratch qualification; never a readonly object claim"
             and state["storage"] == "original SAVE declaration; no copy or C address export",
             "device task opaque state cannot widen observer or capture authority")


def validate_device_task(record, member, procedure, inputs, resources, *, owner=None, provider_slot=None,
                         provider_lifetime_identity=None):
    version = record.get("version") if isinstance(record, dict) else None
    _keys(record, _FIELDS | ({"source_proof_identity", "execution_binding"} if version == 2 else set()))
    _require(type(version) is int and version in (1, 2) and _sha256(record["identity"])
             and record["identity"] == _digest({key: value for key, value in record.items() if key != "identity"}),
             "device task version or identity differs")
    if version == 2:
        _require(_sha256(record["source_proof_identity"]) and record["source_proof_identity"] == _digest({
            key: value for key, value in record.items() if key not in ("identity", "source_proof_identity", "execution_binding")}),
            "device task original source proof identity differs")
    _require(record["procedure"] == member["procedure"] and record["source_graph"] == member["source_graph"]
             and _digest(record["source_inputs"]) == _digest(inputs), "device task original source linkage differs")
    nodes = record["selected_node_ids"]
    _require(isinstance(nodes, list) and len(nodes) == 1 and _text(nodes[0])
             and nodes[0].startswith(member["procedure"] + "/"), "device task original call selection differs")
    _require(record["source_association_authorized"] is True and record["definitions_authorized"] is False
             and record["placement_authorized"] is False, "device task cannot authorize definitions or placement")
    _require(record["numerical_equivalence"] == "explicit supplied adapter qualification"
             and record["lifetime"] == "existing checked capture and current caller-owned domain only"
             and record["position"] == "exact original reached CALL; no motion or eager descriptor reads",
             "device task cannot widen numerical or lifetime authority")
    call, contract = record["call"], record["contract"]
    opaque = {}
    if version == 2:
        _keys(contract, ("version", "source", "provider_manifest", "resources", "configuration", "opaque_arguments", "completion"),
              ("entry_read_requirements",))
        _require(type(contract["version"]) is int and contract["version"] == 2
                 and contract["completion"] == "wait_without_export"
                 and isinstance(contract["opaque_arguments"], list) and bool(contract["opaque_arguments"]),
                 "device task opaque contract or completion differs")
        for item in contract["opaque_arguments"]:
            _keys(item, ("formal", "constructor", "qualification"))
            _require(isinstance(item["formal"], str) and _IDENTIFIER.fullmatch(item["formal"])
                     and item["formal"] not in opaque, "device task opaque formal differs")
            _keys(item["constructor"], ("span", "statement_sha256"))
            span = item["constructor"]["span"]
            _require(isinstance(span, list) and len(span) == 2 and all(type(line) is int for line in span)
                     and 1 <= span[0] <= span[1] and _sha256(item["constructor"]["statement_sha256"]),
                     "device task opaque constructor differs")
            _reference(item["qualification"])
            opaque[item["formal"]] = item
    _keys(call, _ORIGINAL_CALL_FIELDS)
    _require(type(call["schema_version"]) is int and call["schema_version"] == 1
             and _text(call["procedure"]) and _text(call["target"]) and call["source_kind"] in ("module", "external", "internal")
             and call["actual_evaluation"] == "unchanged original call position and guards"
             and all(call[key] is False for key in ("effect_authority", "capture_authority", "alias_authority",
                                                  "completion_authority", "execution_authority")),
             "device task original call cannot grant execution authority")
    formals, actuals, associations = call["formal_arguments"], call["actual_arguments"], call["associations"]
    _require(isinstance(formals, list) and bool(formals)
             and all(isinstance(name, str) and _IDENTIFIER.fullmatch(name) for name in formals)
             and len(set(formals)) == len(formals) and isinstance(actuals, list) and len(actuals) == len(formals)
             and all(isinstance(name, str) and re.fullmatch(r"[A-Za-z]\w*", name) for name in actuals)
             and isinstance(associations, list) and len(associations) == len(formals),
             "device task requires exact ordinary whole-array arguments")
    ranks, canonical = {}, {}
    for formal, actual, association in zip(formals, actuals, associations, strict=True):
        _keys(association, ("formal", "actual", "formal_descriptor", "actual_type", "entry_definition_kill"))
        _require(association["formal"] == formal and association["actual"] == actual, "device task association differs")
        descriptor, actual_type = association["formal_descriptor"], association["actual_type"]
        _keys(descriptor, ("name", "resource", "type", "kind", "rank", "intent", "attributes", "lower_bounds", "shape"))
        _keys(actual_type, ("type", "kind", "rank", "storage_resource", "declaring_type"))
        if formal in opaque:
            declaring = actual_type["declaring_type"]
            _keys(declaring, ("module", "name"))
            _require(descriptor["name"] == formal and descriptor["resource"] == "argument::" + formal
                     and isinstance(descriptor["type"], str) and re.fullmatch(r"type\([a-z][a-z0-9_]*\)", descriptor["type"])
                     and isinstance(actual_type["type"], str) and re.fullmatch(r"type\([a-z][a-z0-9_]*\)", actual_type["type"])
                     and descriptor["type"] in (actual_type["type"], "type(" + str(declaring["name"]) + ")")
                     and descriptor["kind"] is None
                     and actual_type["kind"] is None and type(descriptor["rank"]) is int and descriptor["rank"] == 0
                     and type(actual_type["rank"]) is int and actual_type["rank"] == 0
                     and descriptor["intent"] == "inout" and association["entry_definition_kill"] is False
                     and descriptor["lower_bounds"] == [] and descriptor["shape"] == []
                     and isinstance(descriptor["attributes"], list) and all(_text(name) for name in descriptor["attributes"])
                     and descriptor["attributes"] == sorted(set(descriptor["attributes"]))
                     and not set(descriptor["attributes"]) & {"optional", "pointer", "allocatable", "value"}
                     and all(isinstance(name, str) and _IDENTIFIER.fullmatch(name) for name in declaring.values())
                     and _text(actual_type["storage_resource"]), "device task opaque argument descriptor differs")
            continue
        _require(descriptor["name"] == formal and descriptor["resource"] == "argument::" + formal
                 and descriptor["type"] in ("real", "integer") and type(descriptor["kind"]) is int
                 and (descriptor["type"], descriptor["kind"]) in {("real", 4), ("real", 8), ("integer", 4)}
                 and type(descriptor["rank"]) is int and descriptor["rank"] > 0
                 and descriptor["intent"] in (("in", "inout", "out") if version == 2 else ("in", "inout"))
                 and association["entry_definition_kill"] is (version == 2 and descriptor["intent"] == "out")
                 and isinstance(descriptor["attributes"], list) and all(_text(item) for item in descriptor["attributes"])
                 and descriptor["attributes"] == sorted(set(descriptor["attributes"]))
                 and not set(descriptor["attributes"]) & {"optional", "pointer", "allocatable", "value"}
                 and all(type(actual_type[key]) is int and actual_type[key] == descriptor[key] for key in ("kind", "rank"))
                 and actual_type["type"] == descriptor["type"] and actual_type["declaring_type"] is None
                 and _text(actual_type["storage_resource"]), "device task array descriptors differ")
        rank = descriptor["rank"]
        _require(isinstance(descriptor["lower_bounds"], list) and len(descriptor["lower_bounds"]) == rank
                 and all(_text(bound) for bound in descriptor["lower_bounds"])
                 and isinstance(descriptor["shape"], list) and len(descriptor["shape"]) == rank,
                 "device task descriptor shape differs")
        for axis in descriptor["shape"]:
            _keys(axis, ("kind", "source", "bounds"))
            _require(_text(axis["kind"]) and _text(axis["source"]) and isinstance(axis["bounds"], list)
                     and all(bound is None or _text(bound) for bound in axis["bounds"]),
                     "device task descriptor axis differs")
        ranks[formal], canonical[formal] = descriptor["rank"], actual_type["storage_resource"]
    _require(set(opaque) <= set(formals) and bool(canonical) and record["canonical_resources"] == canonical
             and len(set(canonical.values())) == len(canonical)
             and (version == 2 or all(root in resources and resources[root].get("device_replica_authorized") is True
                                     for root in canonical.values())), "device task canonical storage differs")
    arguments = call["original_arguments"]
    _require(isinstance(arguments, list) and len(arguments) == len(formals), "device task original argument order differs")
    seen = set()
    for index, item in enumerate(arguments):
        _keys(item, ("position", "formal", "keyword", "actual"))
        _require(type(item["position"]) is int and item["position"] == index and item["formal"] in formals
                 and item["formal"] not in seen and item["actual"] == actuals[formals.index(item["formal"])]
                 and (item["keyword"] is None or isinstance(item["keyword"], str)
                      and item["keyword"].lower() == item["formal"]), "device task original argument order differs")
        seen.add(item["formal"])
    _keys(contract, ("version", "source", "provider_manifest", "resources", "configuration",
                     *(("opaque_arguments", "completion") if version == 2 else ())),
          ("entry_read_requirements",) if version == 2 else ())
    _require(type(contract["version"]) is int and contract["version"] == version, "device task contract version differs")
    source = contract["source"]
    _keys(source, ("sha256", "span", "call_sha256", "callee", "callee_body_sha256"))
    _require(source["sha256"] == procedure["source_sha256"] and source["callee"] == call["procedure"]
             and _sha256(source["call_sha256"]) and _sha256(source["callee_body_sha256"])
             and isinstance(source["span"], list) and len(source["span"]) == 2
             and all(type(line) is int for line in source["span"]) and 1 <= source["span"][0] <= source["span"][1],
             "device task original call/body identity differs")
    effects = contract["resources"]
    array_formals = [formal for formal in formals if formal not in opaque]
    _require(isinstance(effects, list) and [row.get("formal") for row in effects if isinstance(row, dict)] == array_formals,
             "device task physical resources differ from original formal order")
    for resource in effects:
        _keys(resource, ("formal", "effects"))
        _require(isinstance(resource["effects"], list) and bool(resource["effects"]), "device task requires explicit effects")
        for effect in resource["effects"]:
            _keys(effect, ("kind", "lower", "upper"))
            _require(effect["kind"] in ("read", "write", "overwrite") and isinstance(effect["lower"], list)
                     and isinstance(effect["upper"], list) and len(effect["lower"]) == len(effect["upper"]) == ranks[resource["formal"]],
                     "device task requires exact physical half-open boxes")
            for value in [*effect["lower"], *effect["upper"]]:
                _expression(value, ranks)
    if version == 2:
        _provider(record["provider"], contract)
    original = record["source_effects"]
    if version == 1:
        _keys(original, ("summary_identity", "summary_version", "complete", "effects", "host_effects", "persistent_state"))
        _require(_sha256(original["summary_identity"]) and type(original["summary_version"]) is int
             and original["summary_version"] > 0 and original["complete"] is True
             and original["host_effects"] is False and original["persistent_state"] is False
             and isinstance(original["effects"], dict) and set(original["effects"]) == set(formals),
             "device task cannot erase original unknown, host or persistent effects")
        original_effects = original["effects"]
    else:
        _keys(original, ("version", "original_body_sha256", "resource_effects", "entry_definition_kills", "opaque_state",
                         "qualification", "wrapper_effects"),
              ("entry_read_requirements",))
        _require(type(original["version"]) is int and original["version"] == 2
                 and original["original_body_sha256"] == source["callee_body_sha256"]
                 and original["qualification"] == record["provider"]["numerical_contract"]["qualification"]
                 and original["wrapper_effects"] == "unchanged original source; only exact inner CALL replaced"
                 and original["entry_definition_kills"] == [item["formal"] for item in associations
                                                            if item["entry_definition_kill"]]
                 and isinstance(original["resource_effects"], dict) and set(original["resource_effects"]) == set(array_formals)
                 and isinstance(original["opaque_state"], list)
                 and [item.get("formal") for item in original["opaque_state"] if isinstance(item, dict)] == list(opaque),
                 "device task qualified original effects or definition kills differ")
        for state in original["opaque_state"]:
            association = associations[formals.index(state["formal"])]
            _opaque_state(state, association, opaque[state["formal"]], record, inputs)
        _require(("entry_read_requirements" in contract) == ("entry_read_requirements" in original),
                 "external entry-read prerequisite is absent or invented")
        if "entry_read_requirements" in original:
            from .external_entry_read_record import validate_external_entry_reads

            validate_external_entry_reads(original["entry_read_requirements"], record, ranks, associations)
        original_effects = original["resource_effects"]
    for resource in effects:
        actions = original_effects[resource["formal"]]
        _require(isinstance(actions, list) and all(action in ("read", "write", "overwrite") for action in actions)
                 and actions == sorted(set(actions)), "device task original payload effects differ")
        supplied = {effect["kind"] for effect in resource["effects"]}
        if version == 2:
            intent = associations[formals.index(resource["formal"])]["formal_descriptor"]["intent"]
            _require(actions == sorted(supplied) and (intent != "in" or supplied == {"read"})
                     and (intent != "out" or "read" not in supplied and bool(supplied & {"write", "overwrite"})),
                     "device task qualified effects contradict original INTENT")
        _require(("read" not in actions or "read" in supplied)
                 and bool(set(actions) & {"write", "overwrite"}) == bool(supplied & {"write", "overwrite"}),
                 "device task omits or invents original payload actions")
    _require(isinstance(contract["configuration"], list), "device task configuration differs")
    components = set()
    limits = {"int32": (-(1 << 31), (1 << 31) - 1), "int64": (-(1 << 63), (1 << 63) - 1), "size": (0, (1 << 63) - 1)}
    for item in contract["configuration"]:
        _keys(item, ("component", "kind", "values"))
        _require(isinstance(item["component"], str) and re.fullmatch(r"[a-z][a-z0-9_]{0,62}(?:%[a-z][a-z0-9_]{0,62})*", item["component"])
                 and item["component"] not in components and item["kind"] in ("int32", "int64", "size", "real64")
                 and isinstance(item["values"], list) and bool(item["values"]), "device task configuration component differs")
        components.add(item["component"])
        for value in item["values"]:
            if item["kind"] == "real64":
                _require(type(value) in (int, float) and -1.7976931348623157e308 <= value <= 1.7976931348623157e308
                         and math.isfinite(value), "device task REAL configuration differs")
            else:
                _expression(value, ranks)
                _require(type(value) is not int or limits[item["kind"]][0] <= value <= limits[item["kind"]][1],
                         "device task INTEGER configuration differs")
    if version == 1:
        _provider(record["provider"], contract)
    if version == 2:
        query = record["provider"]["metadata_query"]
        _require(not components & {item["component"] for item in query["configuration_components"]},
                 "device task query and literal configuration overlap")
        binding = record["execution_binding"]
        _keys(binding, {"owner", "canonical_resources", "registration_slots", "provider_slot", "configuration_words", "ownership"}
              | ({"provider_only_lifetime"} if "provider_only_lifetime" in binding else set()))
        if "provider_only_lifetime" in binding:
            _require(_sha256(provider_lifetime_identity) and binding["provider_only_lifetime"] == provider_lifetime_identity,
                     "device task provider-only lifetime binding differs")
        physical, slots = binding["canonical_resources"], binding["registration_slots"]
        _require(binding["owner"] == owner and binding["ownership"] ==
                 "owning invocation controls forwarded by reference through every checked ENTRY"
                 and isinstance(physical, dict) and set(physical) == set(canonical)
                 and isinstance(slots, dict) and set(slots) == set(canonical)
                 and all(_text(root) and root in resources and resources[root].get("device_replica_authorized") is True
                         and type(slots[formal]) is int and slots[formal] == resources[root]["registration_identity"]
                         for formal, root in physical.items())
                 and len(set(physical.values())) == len(physical)
                 and type(binding["provider_slot"]) is int and binding["provider_slot"] == provider_slot
                 and type(binding["configuration_words"]) is int and binding["configuration_words"] ==
                 sum(len(item["values"]) for item in contract["configuration"]) +
                 sum(item["count"] for item in query["configuration_components"]),
                 "device task owning execution binding differs")
        for formal, local in canonical.items():
            _require(all(isinstance(call.get("resource_mappings"), dict) for call in procedure["calls"]),
                     "device task child canonical projection differs")
            mappings = [call["resource_mappings"] for call in procedure["calls"]]
            expected = physical[formal]
            _require((member["procedure"] == owner and local == expected)
                     or bool(mappings) and all(mapping.get(local) == expected for mapping in mappings),
                     "device task child canonical projection differs")
