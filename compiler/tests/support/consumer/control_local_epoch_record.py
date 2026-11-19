"""Transport consistency for compiler-issued regions and native retirements.

Descriptor and lifecycle identities refer to private compiler proofs. This
reader does not recreate those proofs or grant source, payload or execution
authority. It checks their public owner, storage, graph and boundary linkage.
"""
from __future__ import annotations

from .checked_execution_plan import _digest, _require, _sha256, _text
from .source_placement_record import _tasks

_FIELDS = {
    "proof_identity", "version", "kind", "procedure", "structured_identity", "selected_node_ids",
    "parent_sequence", "roots", "module_roots", "resources", "source_call_closure", "regions",
    "retirement_subtrees", "entry_definition_authority", "payload_access_authorized", "execution_authorized",
    "cross_cut_retention_authorized",
}
_REGION_FIELDS = {"node_id", "descriptor_identity"}
_CUT_FIELDS = {"node_id", "first_line", "last_line", "original_source_sha256", "storage_projection_identity",
               "reopening"}
_STATE_FIELDS = {"resource", "resource_kind", "original_binding", "stable", "original_descriptor_required",
                 "descriptor_changes_within_selection", "escapes_within_selection",
                 "execution_requires_runtime_guard", "entry_definition_authority"}
_ALLOCATION_FIELDS = {"requires_prior_noescape_and_current_allocation_proof",
                      "requires_original_allocation_presence_and_alias_guards"}
_MODULE_FIELDS = {"original_visible_spelling", "declaration_source", "declaration_source_sha256"}
_BORROWED_FIELDS = {"borrowed_descriptor", "descriptor_scope", "caller_storage_lifetime_authority",
                    "physical_allocation_origin_authority", "payload_access_authorized", "alias_authority"}
_FIXED_FIELDS = {"fixed_bounds", "storage_lifetime", "requires_prior_noescape_proof",
                 "requires_original_alias_guards", "payload_access_authorized", "alias_authority"}
_STATE_KINDS = {
    "original_allocatable_formal": _STATE_FIELDS | _ALLOCATION_FIELDS,
    "original_local_save_allocatable": _STATE_FIELDS | _ALLOCATION_FIELDS
        | {"local_storage_identity", "requires_original_runtime_bounds"},
    "original_ordinary_array_formal": _STATE_FIELDS | _ALLOCATION_FIELDS | _BORROWED_FIELDS,
    "original_module_allocatable": _STATE_FIELDS | _ALLOCATION_FIELDS | _MODULE_FIELDS
        | {"requires_original_runtime_bounds"},
    "original_module_fixed_array": _STATE_FIELDS | _MODULE_FIELDS | _FIXED_FIELDS,
}


def _inside(node, parent):
    return node == parent or node.startswith(parent + "/")


def _state(state, canonical, modules, sources):
    kind = state.get("resource_kind") if isinstance(state, dict) else None
    _require(_text(kind) and kind in _STATE_KINDS and set(state) == _STATE_KINDS[kind],
             "selected position epoch control-local storage declaration differs")
    resource = state["resource"]
    _require(resource in canonical and state["original_binding"] == canonical[resource]["descriptor"]
             and (resource in modules) == kind.startswith("original_module_")
             and all(state[key] is True for key in ("stable", "original_descriptor_required",
                                                    "execution_requires_runtime_guard"))
             and all(state[key] is False for key in ("descriptor_changes_within_selection",
                                                     "escapes_within_selection", "entry_definition_authority")),
             "selected position epoch control-local canonical declaration or authority differs")
    for key in _ALLOCATION_FIELDS | {"requires_original_runtime_bounds", "borrowed_descriptor",
                                     "requires_prior_noescape_proof", "requires_original_alias_guards"}:
        _require(key not in state or state[key] is True,
                 "selected position epoch control-local descriptor guards differ")
    for key in {"caller_storage_lifetime_authority", "physical_allocation_origin_authority",
                "payload_access_authorized", "alias_authority"}:
        _require(key not in state or state[key] is False,
                 "selected position epoch control-local descriptor cannot grant authority")
    if kind == "original_local_save_allocatable":
        _require(_sha256(state["local_storage_identity"]),
                 "selected position epoch control-local local storage reference differs")
    if kind == "original_ordinary_array_formal":
        _require(state["descriptor_scope"] == "exact original procedure activation and checked selected closure",
                 "selected position epoch control-local borrowed descriptor scope differs")
    if resource in modules:
        _require(_text(state["original_visible_spelling"]) and _text(state["declaration_source"])
                 and state["declaration_source"] in sources
                 and state["declaration_source_sha256"] == sources[state["declaration_source"]],
                 "selected position epoch control-local module declaration source differs")
    if kind == "original_module_fixed_array":
        bounds = state["fixed_bounds"]
        _require(state["storage_lifetime"] == "original fixed module object within checked selected closure"
                 and isinstance(bounds, list) and len(bounds) == state["original_binding"].get("rank")
                 and all(isinstance(axis, list) and len(axis) == 2
                         and all(type(bound) is int for bound in axis) for axis in bounds),
                 "selected position epoch control-local fixed module bounds differ")


def validate_control_local_regions(value, selected, plan, graph):
    root = plan["procedures"][0]
    _require(isinstance(value, dict) and set(value) == _FIELDS
             and type(value["version"]) is int and value["version"] == 1
             and value["kind"] == "original_control_local_descriptor_regions"
             and _sha256(value["proof_identity"])
             and value["proof_identity"] == selected["descriptor_identity"]
             and value["proof_identity"] == _digest({key: item for key, item in value.items()
                                                     if key != "proof_identity"})
             and value["procedure"] == selected["procedure"]
             and value["structured_identity"] == root["structured_summary_identity"]
             and value["selected_node_ids"] == selected["selected_node_ids"]
             and value["parent_sequence"] == selected["selected_node_ids"][0].rpartition("/")[0],
             "selected position epoch control-local original owner, selection or proof differs")
    _require(all(value[key] is False for key in ("entry_definition_authority", "payload_access_authorized",
                                               "execution_authorized", "cross_cut_retention_authorized")),
             "selected position epoch control-local regions cannot grant authority")
    roots, modules = value["roots"], value["module_roots"]
    _require(all(isinstance(items, list) and all(_text(item) for item in items) for items in (roots, modules))
             and roots + modules == selected["resources"]
             and len(set(roots + modules)) == len(roots) + len(modules),
             "selected position epoch control-local canonical roots differ")
    states = value["resources"]
    _require(isinstance(states, list) and all(isinstance(state, dict) for state in states)
             and [state.get("resource") for state in states] == roots + modules,
             "selected position epoch control-local complete storage inventory differs")
    canonical = {item["resource"]: item for item in plan["resources"]}
    for state in states:
        _state(state, canonical, modules, plan["sources"])
    closure = value["source_call_closure"]
    _require(isinstance(closure, dict) and all(_text(name) and _sha256(identity)
                                             for name, identity in closure.items()),
             "selected position epoch control-local source closure differs")
    # An independently consumed child can have a different summary policy;
    # source closure IDs are compiler references, not new reader authority.
    _require(all(call["procedure"] in closure for call in selected["calls"]),
             "selected position epoch control-local retained CALL closure is incomplete")
    if selected["procedure"] in closure:
        _require(closure[selected["procedure"]] == root["structured_summary_identity"],
                 "selected position epoch control-local owner closure differs")
    regions, cuts = value["regions"], value["retirement_subtrees"]
    _require(isinstance(regions, list) and bool(regions) and isinstance(cuts, list) and bool(cuts),
             "selected position epoch control-local regions and cuts must both be present")
    roles = []
    for rows, fields, identity_key in ((regions, _REGION_FIELDS, "descriptor_identity"),
                                      (cuts, _CUT_FIELDS, "storage_projection_identity")):
        for item in rows:
            _require(isinstance(item, dict) and set(item) == fields and _text(item["node_id"])
                     and _sha256(item[identity_key])
                     and any(_inside(item["node_id"], node) for node in selected["selected_node_ids"]),
                     "selected position epoch control-local role escapes its original selection")
            roles.append(item["node_id"])
    _require(len(set(roles)) == len(roles)
             and not any(_inside(left, right) for left in roles for right in roles if left != right),
             "selected position epoch control-local retained and retired subtrees overlap")
    retained = {item["node_id"] for item in regions}
    _require(all(call["node_id"] in retained for call in selected["calls"]),
             "selected position epoch control-local CALL is not an independently retained leaf")
    graph_ids = [item.get("id") for item in _tasks(graph.get("task_graph"))]
    boundaries = [item for item in root["boundaries"]
                  if item.get("reason") == "checked control-local allocation retirement"]
    _require(len(boundaries) == len(cuts),
             "selected position epoch control-local retirement inventory differs")
    used_spans = []
    for cut in cuts:
        first, last = cut["first_line"], cut["last_line"]
        _require(type(first) is int and type(last) is int and 1 <= first <= last
                 and _sha256(cut["original_source_sha256"])
                 and cut["reopening"] == "complete original no-escape lifecycle; fresh generation and NONE"
                 and any(isinstance(node, str) and (node == cut["node_id"]
                         or node.startswith(cut["node_id"] + ":alternative:")) for node in graph_ids),
                 "selected position epoch control-local retirement source or reopening differs")
        matches = [item for item in boundaries if item.get("first_line") == first
                   and item.get("last_line") == last]
        _require(len(matches) == 1 and matches[0].get("conditional_close") is None
                 and matches[0].get("reopen") is True
                 and matches[0].get("execution") == "publish and close when reached; original continuation once"
                 and not any(first <= end and start <= last for start, end in used_spans),
                 "selected position epoch control-local mandatory retirement boundary differs")
        used_spans.append((first, last))
    # Region/lifecycle digests are opaque compiler references. Public boundary
    # linkage cannot reconstruct original-source or no-escape lifetime proofs.
    # Native graph grouping may replace original operation IDs with synthetic
    # tasks; retained CALL IDs are checked against the graph by the v1 reader.
