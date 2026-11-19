"""Integrity of a source-issued selected lifetime; no recreated source authority."""
from __future__ import annotations

from .checked_execution_plan import _digest, _require, _sha256, _text
from .source_placement_record import _tasks

_FIELDS = {"identity", "version", "procedure", "selected_node_ids", "cut_before_node_id", "descriptor_identity",
           "resources", "calls", "allocation_presence", "repeated_activation", "entry_definition_authority",
           "execution_authorized", "cross_frontier_retention_authorized"}



def _original_sequence(selected, cut, procedure, *, source_graph=None):
    """Check exact original sibling order and the immediately following cut."""
    _require(isinstance(selected, list) and bool(selected) and all(_text(node) for node in selected)
             and len(set(selected)) == len(selected) and _text(cut) and cut not in selected,
             "selected position epoch requires its complete original selection and distinct cut")
    # These are the original structured-source sequence IDs, not graph task
    # ordinals (IF diagnostics can expand into multiple alternative tasks).
    positions = [node.rpartition("/") for node in [*selected, cut]]
    parent = positions[0][0]
    _require(parent.startswith(procedure + "/")
             and all(prefix == parent and separator == "/" and number.isascii() and number.isdecimal()
                     for prefix, separator, number in positions),
             "selected position epoch original sequence or immediate tail cut differs")
    if source_graph is None:
        _require([int(number) for _prefix, _separator, number in positions]
                 == list(range(int(positions[0][2]), int(positions[0][2]) + len(positions))),
                 "selected position epoch original sequence or immediate tail cut differs")
    else:
        # Parser indices include comments. Contiguity belongs to the retained
        # original sequence, whose complete structural digest is rechecked.
        fields = {"schema_version", "procedure", "structured_identity", "analysis_identity", "root", "entry",
                  "available", "reasons", "node_limit", "operation_limit", "operation_count",
                  "raw_structure_rejection", "nodes"}
        _require(isinstance(source_graph, dict) and set(source_graph) == fields,
                 "selected position epoch requires its complete original source graph")
        payload = {key: value for key, value in source_graph.items()
                   if key not in {"schema_version", "structured_identity", "node_limit"}}
        payload["version"] = source_graph["schema_version"]
        _require(type(payload["version"]) is int and payload["version"] == 7
                 and source_graph["procedure"] == procedure
                 and source_graph["structured_identity"] == _digest(payload)
                 and isinstance(source_graph["nodes"], list)
                 and all(isinstance(node, dict) and _text(node.get("id")) for node in source_graph["nodes"]),
                 "selected position epoch original source graph identity differs")
        nodes = {node["id"]: node for node in source_graph["nodes"]}
        _require(len(nodes) == len(source_graph["nodes"]) and parent in nodes
                 and nodes[parent].get("kind") == "sequence",
                 "selected position epoch original parent sequence differs")
        children = nodes[parent].get("children")
        _require(isinstance(children, list) and all(_text(node) and node in nodes for node in children)
                 and len(children) == len(set(children)) and selected[0] in children,
                 "selected position epoch original sibling inventory differs")
        first = children.index(selected[0])
        _require(children[first:first + len(selected) + 1] == [*selected, cut],
                 "selected position epoch original sequence or immediate tail cut differs")
    return parent


def validate_selected_epoch(value, epoch, plan, scope):
    version = value.get("version") if isinstance(value, dict) else None
    fields = _FIELDS | ({"control_local_regions"} if version == 2 else set())
    _require(isinstance(value, dict) and set(value) == fields
             and type(version) is int and version in (1, 2)
             and _sha256(value["identity"])
             and value["identity"] == _digest({k: v for k, v in value.items() if k != "identity"})
             and _sha256(value["descriptor_identity"]) and value["procedure"] == plan["entry"] == epoch["entry"],
             "selected position epoch requires its exact original owner and versioned identity")
    _require(value["allocation_presence"] == "lazy original guards at the first reached managed access"
             and value["repeated_activation"] == "same checked source position; NONE after retirement"
             and all(value[key] is False for key in ("entry_definition_authority", "execution_authorized",
                                                    "cross_frontier_retention_authorized")),
             "selected position epoch cannot grant definitions, execution or cross-frontier retention")
    selected, cut, roots = value["selected_node_ids"], value["cut_before_node_id"], value["resources"]
    _require(isinstance(selected, list) and bool(selected) and all(_text(node) for node in selected)
             and len(set(selected)) == len(selected) and _text(cut) and cut not in selected,
             "selected position epoch requires its complete original selection and distinct cut")
    _original_sequence(selected, cut, value["procedure"])
    canonical = {item["resource"]: item for item in epoch["resources"]}
    _require(isinstance(roots, list) and bool(roots) and all(_text(root) for root in roots)
             and len(set(roots)) == len(roots)
             and all(root in canonical and canonical[root].get("device_replica_authorized") is True
                     and canonical[root].get("lifetime") == "selected_window" for root in roots),
             "selected position epoch requires independent selected-window storage facts")
    root = plan["procedures"][0]
    _require(any(boundary.get("reason") == "checked selected-position allocation epoch retirement"
                 and boundary.get("conditional_close") is None and boundary.get("reopen") is True
                 and boundary.get("execution") == "publish and close when reached; original continuation once"
                 for boundary in root["boundaries"]), "selected position epoch lost its mandatory retirement")
    execution = scope.get("graph_execution")
    _require(isinstance(execution, dict) and execution.get("source_order_executor") is True
             and isinstance(execution.get("procedures"), list), "selected position epoch requires its original graph")
    graphs = [item for item in execution["procedures"]
              if isinstance(item, dict) and item.get("procedure") == value["procedure"]]
    _require(len(graphs) == 1, "selected position epoch requires its exact original graph procedure")
    graph_ids = [task.get("id") for task in _tasks(graphs[0].get("task_graph"))]
    _require(all(any(isinstance(node, str) and (node == identity or node.startswith(identity + ":alternative:"))
                     for node in graph_ids) for identity in [*selected, cut]),
             "selected position epoch selection or cut is absent from its original graph")
    calls = value["calls"]
    _require(isinstance(calls, list) and bool(calls), "selected position epoch requires its original call inventory")
    seen = set()
    procedures = {item["procedure"]: item for item in plan["procedures"]}
    for call in calls:
        _require(isinstance(call, dict) and set(call) == {"node_id", "procedure", "canonical_resources"}
                 and _text(call["node_id"]) and call["node_id"] not in seen and _text(call["procedure"])
                 and call["node_id"] in graph_ids
                 and any(call["node_id"] == identity or call["node_id"].startswith(identity + "/")
                         for identity in selected), "selected position epoch original call selection differs")
        seen.add(call["node_id"])
        mappings = call["canonical_resources"]
        _require(isinstance(mappings, dict) and all(_text(local) and _text(mapped) and mapped in roots
                                                  for local, mapped in mappings.items())
                 and len(set(mappings.values())) == len(mappings),
                 "selected position epoch canonical call associations differ")
        child = procedures.get(call["procedure"])
        if child is not None:
            sites = [site for site in child["calls"] if site.get("caller") == value["procedure"]
                     and site.get("resource_mappings") == mappings]
            _require(bool(sites), "selected position epoch checked child association differs")
    if version == 2:
        from .control_local_epoch_record import validate_control_local_regions

        validate_control_local_regions(value["control_local_regions"], value, plan, graphs[0])
