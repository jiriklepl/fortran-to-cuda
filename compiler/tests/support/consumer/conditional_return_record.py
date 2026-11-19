"""Transport exact conditional-return contracts without issuing source authority."""
from __future__ import annotations

from hashlib import sha256

from .checked_execution_plan import _digest, _require, _sha256, _text

_RETURN = {"identity", "version", "kind", "entry", "source_inputs", "selection_identity", "members", "controls",
           "retained_return", "unproved_boundary", "unproved_boundary_admission", "known_retirement",
           "escape_completion_assumed", "entry_definition_authority", "payload_access_authorized", "motion_authorized",
           "placement_authorized", "cross_cut_replica_retention_authorized"}
_SELECTION = {"identity", "version", "kind", "entry", "source_inputs", "original_entry_body_sha256", "device_roots",
              "selection_semantics", "call_authority", "native_execution", "unselected_native_once",
              "payload_access_authorized", "descriptor_stability_authorized", "entry_definition_authority",
              "execution_authorized", "placement_authorized", "alias_authority"}

_ROOT_SELECTION = {"identity", "version", "kind", "procedure", "source_graph", "parent_sequence",
                   "selected_node_ids", "cut_before", "source_span", "original_source_sha256", "activation",
                   "cleanup", "outside_selection", "payload_access_authorized", "descriptor_stability_authorized",
                   "entry_definition_authority", "execution_authorized", "cross_exit_retention_authorized", "alias_authority"}
_END_RETIREMENT = {"version", "kind", "source", "insertion_span", "context", "replacement_sha256"}
_LOOP_ENTRY = {"identity", "version", "kind", "source_graph", "selected_node_ids", "header_node_id",
               "header_summary_identity", "source_span", "header_span", "header_source_sha256",
               "body_source_sha256", "flag", "context", "header_evaluation", "retirement",
               "empty_loop_preserves_domain", "final_iterator", "payload_access_authorized", "motion_authorized"}


def _keys(value, keys):
    _require(isinstance(value, dict) and set(value) == keys, "conditional return requires its exact schema")


def _sealed(value, keys):
    _keys(value, keys)
    _require(_sha256(value["identity"]) and value["identity"] == _digest({k: v for k, v in value.items() if k != "identity"}),
             "conditional return content identity differs")


def _false(value, keys):
    _require(all(value[key] is False for key in keys), "conditional return cannot grant payload, definition, placement or escape authority")


def _slot(procedure, resource):
    # Exact public v5 module-control ABI: local names are independently mapped
    # to caller slots and canonical registration identities.
    return int(sha256((procedure + "|" + resource).encode()).hexdigest()[:8], 16) + 1


def _context_member(name, scope, coordinators):
    row = coordinators.get(name, {})
    relation = row.get("lexical_namespace")
    return (relation["namespace_owner"] or scope["checked_execution_plan"]["entry"]
            if relation is not None else name)


def validate_conditional_return(record, epoch, plan, manifest, scope):
    version = record.get("version") if isinstance(record, dict) else None
    _sealed(record, _RETURN | ({"root_selection"} if isinstance(record,dict) and "root_selection" in record else set())
            | ({"native_call_preservation_policy_version"} if version == 3 or version in (4, 5)
               and "native_call_preservation_policy_version" in record else set())
            | ({"lexical_child_policy_version"} if version == 4 or version == 5 and "lexical_child_policy_version" in record else set())
            | ({"specification_entry_policy_version"} if version == 5 else set()))
    _require(type(version) is int and version in (1, 2, 3, 4, 5) and record["kind"] == "checked_conditional_return"
             and record["entry"] == plan["entry"] and record["source_inputs"] == epoch["source_inputs"]
             and _sha256(record["selection_identity"]), "conditional return source or version linkage differs")
    _false(record, ("escape_completion_assumed", "entry_definition_authority", "payload_access_authorized", "motion_authorized",
                    "placement_authorized", "cross_cut_replica_retention_authorized"))
    _require(record["retained_return"] == "only checked effects and coherent identity transfers on the executed path"
             and record["unproved_boundary"] == "publish and retire before original code; disable the unchecked suffix and callers"
             and record["unproved_boundary_admission"] == "entire conditional owner remains native; no cross-invocation escape completion is assumed"
             and record["known_retirement"] == "publish and retire; new registrations have no inferred definitions",
             "conditional return lifetime/escape policy differs")
    control = record["controls"]
    _keys(control, {"retention_allowed", "module_control_version", "propagation"})
    _require(control["retention_allowed"] == "fort_retention_allowed_" + sha256(plan["entry"].encode()).hexdigest()[:12]
             and type(control["module_control_version"]) is int and control["module_control_version"] == 5
             and control["propagation"] == "INOUT by original checked CALL; false is never rearmed in this invocation",
             "conditional return original ENTRY controls differ")
    if version == 5:
        _require(type(record["specification_entry_policy_version"]) is int and record["specification_entry_policy_version"] == 2,
                 "conditional return specification entry policy differs")
        from .specification_entry_record import validate_runtime
        validate_runtime(manifest.get("runtime"), record)
    if version == 4 or "lexical_child_policy_version" in record:
        _require(type(record["lexical_child_policy_version"]) is int and record["lexical_child_policy_version"] == 1,
                 "conditional return lexical child policy differs")
    if version == 3 or "native_call_preservation_policy_version" in record:
        _require(type(record["native_call_preservation_policy_version"]) is int
                 and record["native_call_preservation_policy_version"] in (2, 3),
                 "conditional return native CALL preservation policy differs")
    selected_root = None
    if version == 2 or version in (3, 4, 5) and "root_selection" in record:
        selected_root = _selected_root(record, epoch, plan, manifest, scope)
    else:
        selection = manifest.get("native_ancestor_selection")
        _sealed(selection, _SELECTION)
        _require(type(selection["version"]) is int and selection["version"] == 1
                 and selection["kind"] == "original_native_ancestor_selection" and selection["entry"] == plan["entry"]
                 and selection["identity"] == record["selection_identity"] and selection["source_inputs"] == record["source_inputs"]
                 and _sha256(selection["original_entry_body_sha256"]) and selection["unselected_native_once"] is True
                 and selection["selection_semantics"] == "only exact selected original module subtrees may undergo device discovery; all ancestors remain original native source"
                 and selection["call_authority"] == "independent exact original call association and checked companion admission"
                 and selection["native_execution"] == "existing precise native effects and coherence hooks or reached publication and full domain retirement",
                 "conditional return native ancestor selection differs")
        _false(selection, ("payload_access_authorized", "descriptor_stability_authorized", "entry_definition_authority",
                            "execution_authorized", "placement_authorized", "alias_authority"))
    procedures = {p["procedure"]: p for p in plan["procedures"]}
    members = record["members"]
    _require(isinstance(members, list) and [m.get("procedure") for m in members if isinstance(m, dict)] == list(procedures),
             "conditional return complete member inventory differs")
    if selected_root is None:
        roots = selection["device_roots"]
        _require(isinstance(roots, list) and bool(roots), "conditional return requires its original selected roots")
        seen = set()
        for root in roots:
            _keys(root, {"procedure", "source", "source_sha256", "original_body_sha256"})
            _require(_text(root["procedure"]) and root["procedure"] in procedures and root["procedure"] not in seen
                     and _sha256(root["original_body_sha256"]) and root["source"] == procedures[root["procedure"]]["source"]
                     and root["source_sha256"] == procedures[root["procedure"]]["source_sha256"],
                     "conditional return selected original root differs")
            seen.add(root["procedure"])
    coordinators = scope.get("module_coordinators", [])
    _require(isinstance(coordinators, list), "conditional return requires its original module ENTRY records")
    module = {row.get("procedure"): row for row in coordinators if isinstance(row, dict)}
    lexical_rows = scope.get("internal_coordinators", []) if "lexical_child_policy_version" in record else []
    _require(isinstance(lexical_rows, list), "conditional return requires its original lexical ENTRY records")
    lexical = {row.get("procedure"): row for row in lexical_rows if isinstance(row, dict)}
    _require(len(module) == len(coordinators) and len(lexical) == len(lexical_rows)
             and not set(module) & set(lexical) and set(module) | set(lexical) == set(procedures) - {plan["entry"]},
             "conditional return checked companion inventory differs")
    for name, row in module.items():
        _require(type(row.get("control_version")) is int and row["control_version"] == 5
                 and row.get("structured_summary_identity") == procedures[name]["structured_summary_identity"]
                 and row.get("interface") == "original_module_entry_v1"
                 and row.get("native_abi") == "original entry; compiler-control arguments are never referenced"
                 and isinstance(row.get("resource_mappings"), dict)
                 and row.get("calls") == [call for call in procedures[name]["calls"] if "handle_bindings" in call],
                 "conditional return companion ENTRY ABI or CALL inventory differs")
    from .source_task_call_record import validate_lexical_namespace

    all_coordinators = {**module, **lexical}
    for name, row in lexical.items():
        _require(row.get("structured_summary_identity") == procedures[name]["structured_summary_identity"]
                 and row.get("control") == "host association; original declarations and body"
                 and row.get("calls") == [call for call in procedures[name]["calls"]
                                           if call.get("caller") == name],
                 "conditional return lexical original ABI or CALL inventory differs")
        validate_lexical_namespace(row.get("lexical_namespace"), name, procedures,
                                   all_coordinators, scope, manifest)
    for name, row in lexical.items():
        namespace = row["lexical_namespace"]["namespace_owner"]
        current, seen = name, set()
        while current in lexical:
            _require(current not in seen and lexical[current]["lexical_namespace"]["namespace_owner"] == namespace,
                     "conditional return lexical namespace chain is cyclic or changes owner")
            seen.add(current)
            current = lexical[current]["lexical_namespace"]["parent"]
        _require(current == (namespace or plan["entry"]),
                 "conditional return lexical namespace chain lost its original owner")
    canonical = {r["resource"]: r["registration_identity"] for r in plan["resources"]}
    from .scalar_element_record import validate_scalar_element_reservations

    resource_rows, scalar_uses = validate_scalar_element_reservations(
        scope, plan, manifest, epoch, members, all_coordinators)
    epoch_members = {m["procedure"]: m for m in epoch["members"]}
    relations = []
    for member in members:
        relations.extend(_member(member, procedures[member["procedure"]], epoch_members[member["procedure"]], procedures, all_coordinators, canonical, scope,
                selected_root=selected_root if member["procedure"] == plan["entry"] else None,
                resource_rows=resource_rows, source_edits=manifest.get("source_edits"), return_record=record))
    uses = _closed_implicit_uses(relations, procedures, scalar_uses, scope, manifest)
    for relation in relations:
        _require(relation["implicit"] <= uses[relation["callee"]],
                 "conditional return child formal-to-canonical association differs")



def _selected_root(record, epoch, plan, manifest, scope):
    """Bind the v2 root region to existing source graph and emitted cleanup."""
    from .selected_epoch_record import _original_sequence
    from .source_placement_record import _tasks

    _require("native_ancestor_selection" not in manifest,
             "hierarchical return cannot borrow native-ancestor selection authority")
    selection = record["root_selection"]
    _sealed(selection, _ROOT_SELECTION)
    _require(type(selection["version"]) is int and selection["version"] == 1
             and selection["kind"] == "original_hierarchical_root_selection"
             and selection["procedure"] == plan["entry"]
             and selection["identity"] == record["selection_identity"]
             and _sha256(selection["source_graph"]) and _sha256(selection["original_source_sha256"])
             and selection["activation"] == "fresh lazy domain at each reached original selection; no prefix definitions"
             and selection["cleanup"] == "publish and wait; retire the domain immediately after the selected original span"
             and selection["outside_selection"] == "unchanged original source; no context or handles retained across the selection end",
             "hierarchical return original root selection differs")
    _false(selection, ("payload_access_authorized", "descriptor_stability_authorized", "entry_definition_authority",
                       "execution_authorized", "cross_exit_retention_authorized", "alias_authority"))
    selected, cut = selection["selected_node_ids"], selection["cut_before"]
    structure = manifest.get("native_effects", {}).get("structured_effects")
    _require(isinstance(structure, dict)
             and structure.get("structured_identity") == selection["source_graph"],
             "hierarchical return requires its retained original source graph")
    parent = _original_sequence(selected, cut, plan["entry"], source_graph=structure)
    _require(selection["parent_sequence"] == parent, "hierarchical return parent sequence differs")
    roots = [p for p in plan["procedures"] if p["procedure"] == plan["entry"]]
    members = [m for m in epoch["members"] if m["procedure"] == plan["entry"]]
    _require(len(roots) == len(members) == 1, "hierarchical return requires its exact original root member")
    root = roots[0]
    authority = scope.get("source_authority")
    _require(isinstance(authority, dict)
             and selection["source_graph"] == root["structured_summary_identity"] == members[0]["source_graph"]
             == authority.get("structured_summary_identity"), "hierarchical return root source graph differs")
    span = selection["source_span"]
    _require(isinstance(span, list) and len(span) == 2 and all(type(n) is int for n in span)
             and 1 <= span[0] <= span[1] and span == [scope.get("first_line"), scope.get("last_line")]
             and root["source"] == scope.get("path"), "hierarchical return original source span differs")
    execution = scope.get("graph_execution")
    _require(isinstance(execution, dict) and execution.get("source_order_executor") is True
             and isinstance(execution.get("procedures"), list), "hierarchical return requires its original checked graph")
    graphs = [g for g in execution["procedures"] if isinstance(g, dict) and g.get("procedure") == plan["entry"]]
    _require(len(graphs) == 1, "hierarchical return original root graph inventory differs")
    ids = [t.get("id") for t in _tasks(graphs[0].get("task_graph"))]
    _require(all(any(isinstance(node, str) and (node == selected_id or node.startswith(selected_id + ":alternative:"))
                     for node in ids) for selected_id in [*selected, cut]),
             "hierarchical return selection or immediate cut is absent from its original graph")
    end = scope.get("selected_end_retirement")
    _keys(end, _END_RETIREMENT)
    _require(type(end["version"]) is int and end["version"] == 1
             and end["kind"] == "emitted_selected_end_domain_retirement"
             and end["source"] == root["source"] and end["insertion_span"] == [span[1] + 1, span[1]]
             and isinstance(end["insertion_span"], list) and all(type(n) is int for n in end["insertion_span"])
             and _text(end["context"]) and end["context"] == scope.get("owner")
             and _sha256(end["replacement_sha256"]), "hierarchical return mandatory selected-end cleanup differs")
    edits = manifest.get("source_edits")
    _require(isinstance(edits, list), "hierarchical return requires its actual emitted source edits")
    matches = [e for e in edits if isinstance(e, dict) and e.get("source") == end["source"]
               and [e.get("first_line"), e.get("last_line")] == end["insertion_span"]]
    _require(len(matches) == 1 and matches[0].get("source_sha256") == root["source_sha256"]
             and type(matches[0].get("first_line")) is int and type(matches[0].get("last_line")) is int
             and isinstance(matches[0].get("replacement"), str)
             and sha256(matches[0]["replacement"].encode()).hexdigest() == end["replacement_sha256"],
             "hierarchical return selected-end source edit binding differs")
    lines = [line.strip() for line in matches[0]["replacement"].splitlines()]
    close = "fort_close_status = fort_scope_resident_close_v1(" + end["context"] + ")"
    reset = end["context"] + " = 0"
    _require(lines.count(close) == lines.count(reset) == 1 and lines.index(close) < lines.index(reset),
             "hierarchical return emitted cleanup lost its exact close/reset context")
    return selected

def _loop_entry(entry, nodes, procedure, coordinators, scope, source_edits):
    """Transport the existing header proof and exact emitted once-only cut.

    Summary identity remains compiler-issued source authority. These checks bind
    its original selection and byte-preserving header/body split to the actual
    source edit; they do not reconstruct Fortran effects or issue lifetime facts.
    """
    _sealed(entry, _LOOP_ENTRY)
    name = procedure["procedure"]
    context_member = _context_member(name, scope, coordinators)
    expected_context = (scope["owner"] if context_member == scope["checked_execution_plan"]["entry"]
                        else "fort_child_context_" + sha256(context_member.encode()).hexdigest()[:12])
    _require(name == scope["checked_execution_plan"]["entry"] or name in coordinators,
             "reached loop retirement requires its checked original context owner")
    _require(type(entry["version"]) is int and entry["version"] == 1
             and entry["kind"] == "original_counted_body_retirement"
             and entry["source_graph"] == procedure["structured_summary_identity"]
             and entry["selected_node_ids"] == nodes and len(nodes) == 1
             and entry["header_node_id"] == nodes[0] + "/header"
             and _sha256(entry["header_summary_identity"])
             and entry["context"] == expected_context
             and entry["flag"] == "fort_loop_retire_" + sha256((name + "|" + nodes[0]).encode()).hexdigest()[:12],
             "reached loop retirement source/header/context association differs")
    _require(entry["header_evaluation"] == "unchanged original counted DO exactly once"
             and entry["retirement"] == "once before the first original body statement"
             and entry["final_iterator"] == "unchanged original counted DO"
             and entry["empty_loop_preserves_domain"] is True,
             "reached loop retirement control semantics differ")
    _false(entry, ("payload_access_authorized", "motion_authorized"))
    span, header = entry["source_span"], entry["header_span"]
    _require(isinstance(span, list) and isinstance(header, list) and len(span) == len(header) == 2
             and all(type(n) is int for n in [*span, *header])
             and 1 <= span[0] == header[0] <= header[1] < span[1]
             and _sha256(entry["header_source_sha256"]) and _sha256(entry["body_source_sha256"]),
             "reached loop retirement original source split differs")
    boundaries = [b for b in procedure["boundaries"]
                  if [b.get("first_line"), b.get("last_line")] == span]
    _require(len(boundaries) == 1 and boundaries[0].get("loop_entry") == entry
             and boundaries[0].get("conditional_close") is None and boundaries[0].get("reopen") is True
             and boundaries[0].get("lifecycle") == "proved_lifecycle",
             "reached loop retirement is detached from its original checked boundary")
    _require(isinstance(source_edits, list), "reached loop retirement needs its emitted source edit")
    edits = [e for e in source_edits if isinstance(e, dict) and e.get("source") == procedure["source"]
             and [e.get("first_line"), e.get("last_line")] == span]
    _require(len(edits) == 1 and edits[0].get("source_sha256") == procedure["source_sha256"]
             and isinstance(edits[0].get("replacement"), str),
             "reached loop retirement original edit association differs")
    replacement, flag = edits[0]["replacement"], entry["flag"]
    prefix = "block\nlogical :: " + flag + "\n" + flag + " = .true.\n"
    marker = "! fort original loop body " + flag + "\n"
    _require(replacement.startswith(prefix) and replacement.endswith("end block\n")
             and replacement.count(marker) == 1, "reached loop retirement lost its private entry flag")
    lines = replacement[len(prefix):].splitlines(keepends=True)
    count = header[1] - header[0] + 1
    original_header = "".join(lines[:count])
    tail = "".join(lines[count:])
    guarded, body = tail.split(marker, 1)
    _require(sha256(original_header.encode()).hexdigest() == entry["header_source_sha256"]
             and sha256(body[:-len("end block\n")].encode()).hexdigest() == entry["body_source_sha256"],
             "reached loop retirement changed or duplicated original header/body bytes")
    _require(guarded.startswith("if (" + flag + ") then\n" + flag + " = .false.\n")
             and guarded.endswith("endif\n") and guarded.count(flag + " = .false.") == 1,
             "reached loop retirement lost its first-body once-only close")
    emitted = [line.strip() for line in guarded.splitlines()]
    close = "fort_close_status = fort_scope_resident_close_v1(" + expected_context + ")"
    reset = expected_context + " = 0"
    _require(emitted.count(close) == emitted.count(reset) == 1 and emitted.index(close) < emitted.index(reset),
             "reached loop retirement changed its exact domain close/reset")
    return entry["identity"]


def _member(member, procedure, epoch_member, procedures, coordinators, canonical, scope, *, selected_root=None,
            resource_rows=None, source_edits=None, return_record=None):
    _keys(member, {"procedure", "source_graph", "entry_exit", "routes"})
    name = procedure["procedure"]
    _require(member["source_graph"] == procedure["structured_summary_identity"], "conditional return member source graph differs")
    entry = member["entry_exit"]
    _keys(entry, {"procedure", "original_body_sha256", "specification", "exit", "opaque_resources", "definition_authority"}
          | ({"specification_reads"} if "specification_reads" in entry else set()))
    if "specification_reads" in entry:
        from .specification_entry_record import validate_projection
        validate_projection(entry["specification_reads"], entry, return_record, scope)
    _require(entry["procedure"] == name and _sha256(entry["original_body_sha256"])
             and entry["specification"] == "original numeric scalar/descriptor declarations; no hidden calls, aliases or finalization"
             and entry["exit"] == "no retained child-local allocation; original RETURN keeps caller controls"
             and isinstance(entry["opaque_resources"], list) and all(_text(r) for r in entry["opaque_resources"])
             and len(set(entry["opaque_resources"])) == len(entry["opaque_resources"])
             and entry["definition_authority"] is False, "conditional return entry/exit declaration contract differs")
    routes = member["routes"]
    _require(isinstance(routes, list), "conditional return requires all original execution routes")
    covered, unit_ordinals, native_ordinals, retirement = set(), set(), set(), 0
    checked_devices, checked_transfers, children, lexical_children, loop_entries = set(), set(), [], [], set()
    relations = []
    for route in routes:
        _require(isinstance(route, dict) and _text(route.get("kind")), "conditional return route differs")
        kind, nodes = route["kind"], route.get("selected_node_ids")
        _require(isinstance(nodes, list) and bool(nodes) and all(_text(n) and n.startswith(name + "/") for n in nodes)
                 and len(set(nodes)) == len(nodes) and not covered.intersection(nodes), "conditional return repeats or loses original nodes")
        if selected_root is not None:
            _require(all(any(node == root or node.startswith(root + "/") for root in selected_root)
                         for node in nodes), "hierarchical return root route escaped the original selected subtree")
        covered.update(nodes)
        base = {"kind", "selected_node_ids"}
        if kind in {"checked_unit", "checked_native"}:
            _keys(route, base | {"ordinal"})
            ordinal = route["ordinal"]
            _require(type(ordinal) is int and ordinal >= 0, "conditional return unit ordinal differs")
            target = unit_ordinals if kind == "checked_unit" else native_ordinals
            _require(ordinal not in target, "conditional return repeats a checked unit")
            target.add(ordinal)
            if kind == "checked_unit":
                _require(ordinal < len(procedure["units"]) and nodes == procedure["units"][ordinal]["selected_node_ids"], "conditional return numerical unit source differs")
        elif kind in {"checked_device", "checked_identity_transfer"}:
            _keys(route, base | {"proof_identity"})
            _require(_sha256(route["proof_identity"]), "conditional return task proof identity differs")
            (checked_devices if kind == "checked_device" else checked_transfers).add(route["proof_identity"])
        elif kind == "reached_retirement":
            _keys(route, base | {"lifetime", "rearm_allowed"} | ({"loop_entry"} if "loop_entry" in route else set())
                  | ({"call_preservation"} if "call_preservation" in route else set())
                  | ({"empty_call_guard"} if "empty_call_guard" in route else set()))
            _require("empty_call_guard" not in route or "call_preservation" in route,
                     "empty native CALL guard requires its complete original retirement")
            _require(route["lifetime"] == "proved_lifecycle" and route["rearm_allowed"] is True,
                     "conditional return cannot admit an unproved escape across invocations")
            if "loop_entry" in route:
                identity = _loop_entry(route["loop_entry"], nodes, procedure, coordinators, scope, source_edits)
                _require(identity not in loop_entries, "conditional return repeats a reached loop retirement")
                loop_entries.add(identity)
            if "call_preservation" in route:
                _require(return_record is not None and return_record["version"] in (3, 4, 5)
                         and return_record.get("native_call_preservation_policy_version") in (2, 3) and "loop_entry" not in route,
                         "native CALL preservation requires its distinct conditional return version")
                _call_preservation(route["call_preservation"],nodes,member,procedure,coordinators,scope,source_edits,return_record,
                                   empty_guard=route.get("empty_call_guard"))
            retirement += 1
        elif kind == "checked_child":
            _keys(route, base | {"procedure", "canonical_handles", "canonical_resources", "handle_forwarding"}
                  | ({"specification_entry"} if "specification_entry" in route else set()))
            relations.append(_child(route, name, procedures, coordinators, canonical, scope, resource_rows))
            from .specification_entry_record import validate_call
            validate_call(route, procedure, procedures, coordinators, scope, source_edits, return_record)
            children.append(route)
        elif kind == "checked_lexical_child":
            _keys(route, base | {"procedure", "source_call", "namespace_owner"})
            _require(return_record is not None and return_record["version"] in (4, 5)
                     and return_record.get("lexical_child_policy_version") == 1,
                     "conditional return lexical child requires its distinct policy version")
            _lexical_child(route, name, procedures, coordinators)
            relations.append({"caller": name, "callee": route["procedure"], "resources": None,
                              "explicit": set(), "implicit": set()})
            lexical_children.append(route)
        elif kind == "checked_condition":
            _keys(route, base | {"summary_identity"})
            _require(_sha256(route["summary_identity"]), "conditional return original predicate summary differs")
        elif kind in {"checked_counted_header", "checked_return"}:
            _keys(route, base)
            _require(len(nodes) == 1 and (kind != "checked_return" or name != scope["checked_execution_plan"]["entry"]),
                     "conditional return original control route differs")
        else:
            _require(False, "conditional return has an unknown route kind")
    _require(unit_ordinals == set(range(len(procedure["units"]))) and retirement == len(procedure["boundaries"]),
             "conditional return lost a numerical unit or reached retirement")
    _require(loop_entries == {b["loop_entry"].get("identity") for b in procedure["boundaries"] if "loop_entry" in b},
             "conditional return lost its reached loop retirement binding")
    _require(native_ordinals == set(range(len(native_ordinals))), "conditional return native unit ordinals differ")
    _require(len(children) == sum(call.get("caller") == name and "handle_bindings" in call
                                 for p in procedures.values() for call in p["calls"])
             and len(lexical_children) == sum(call.get("caller") == name and call.get("execution") ==
                 "original internal procedure; host-associated owner control" for call in procedure["calls"]),
             "conditional return lost a checked original child CALL")
    if children or lexical_children or native_ordinals:
        from .source_placement_record import _tasks

        graph = scope.get("graph_execution")
        _require(isinstance(graph, dict) and isinstance(graph.get("procedures"), list), "conditional return requires its checked source graph")
        rows = [row for row in graph["procedures"] if isinstance(row, dict) and row.get("procedure") == name]
        _require(len(rows) == 1, "conditional return source graph procedure differs")
        tasks = {task["id"]: task for task in _tasks(rows[0].get("task_graph"))}
        _require(all(name + ":native:" + str(ordinal) in tasks for ordinal in native_ordinals)
                 and all(len(child["selected_node_ids"]) == 1 and child["selected_node_ids"][0] in tasks
                         and tasks[child["selected_node_ids"][0]].get("opaque") is True
                         for child in [*children, *lexical_children]),
                 "conditional return native unit or original child graph coverage differs")
    device_ids = set()
    for task in [*epoch_member.get("device_tasks", []), *epoch_member.get("source_device_tasks", [])]:
        source = task.get("source_proof")
        identity = source.get("identity") if isinstance(source, dict) else task.get("source_proof_identity")
        _require(_sha256(identity), "conditional return device source identity differs")
        device_ids.add(identity)
    _require(checked_devices == device_ids and checked_transfers == {t["identity"] for t in epoch_member["storage_transfers"]},
             "conditional return device or identity-transfer coverage differs")
    return relations


def _lexical_child(route, caller, procedures, coordinators):
    name = route["procedure"]
    _require(_text(name) and name in procedures and name in coordinators,
             "conditional return lexical child differs")
    relation = coordinators[name].get("lexical_namespace")
    _require(isinstance(relation, dict) and relation["parent"] == caller
             and relation["namespace_owner"] == route["namespace_owner"]
             and any(row["source_call"] == route["source_call"]
                     and row["selected_node_ids"] == route["selected_node_ids"]
                     and row["caller_graph"] == procedures[caller]["structured_summary_identity"]
                     for row in relation["source_calls"]),
             "conditional return lexical child lost its exact original parent CALL")



def _call_preservation(value, nodes, member, procedure, coordinators, scope, source_edits, returned, *, empty_guard=None):
    keys = {"identity","version","kind","procedure","source_graph","selected_node_ids","source_span",
            "original_source","original_source_sha256","original_call","completion_identity","native_graph_identity",
            "selection_identity","capture_facts_identity","entry_exit_identity","writable_actual_families","entry_rely","retirement",
            "completion_scope","no_imports_or_publication","initial_quiescence_authorized","payload_access_authorized",
            "stable_storage_authorized","entry_definition_authority","alias_authority","motion_authorized",
            "cross_cut_replica_retention_authorized"}
    _sealed(value,keys)
    plan=scope["checked_execution_plan"]
    name=procedure["procedure"]
    _require(type(value["version"]) is int and value["version"]==2
             and value["kind"]=="same_family_native_call_retirement" and value["procedure"]==name
             and value["source_graph"]==procedure["structured_summary_identity"]
             and value["selected_node_ids"]==nodes and len(nodes)==1
             and value["selection_identity"]==returned["selection_identity"]
             and value["capture_facts_identity"]==plan["capture_facts_identity"]
             and value["entry_exit_identity"]==_digest(member["entry_exit"])
             and _sha256(value["completion_identity"]) and _sha256(value["native_graph_identity"]),
             "native CALL preservation source/admission linkage differs")
    families=value["writable_actual_families"]
    _require(isinstance(families,list) and all(_text(name) for name in families)
             and families==sorted(set(families))
             and set(families)<={row["resource"]for row in plan["resources"]},
             "native CALL writable actuals escaped the inherited managed allocation family")
    _require(value["entry_rely"]=="same conditional allocation-family access exclusion as first checked activation"
             and value["retirement"]=="full owning domain before untouched original CALL; next activation remains fresh NONE"
             and value["completion_scope"]=="normal return only" and value["no_imports_or_publication"] is True,
             "native CALL preservation contract differs")
    _false(value,("initial_quiescence_authorized","payload_access_authorized","stable_storage_authorized",
                  "entry_definition_authority","alias_authority","motion_authorized","cross_cut_replica_retention_authorized"))
    span=value["source_span"]
    _require(isinstance(span,list) and len(span)==2 and all(type(n) is int for n in span)
             and 1<=span[0]<=span[1] and _text(value["original_source"]) and _text(value["original_call"])
             and value["original_source_sha256"]==sha256(value["original_source"].encode()).hexdigest(),
             "native CALL preservation original source differs")
    boundaries=[b for b in procedure["boundaries"] if [b.get("first_line"),b.get("last_line")]==span]
    _require(len(boundaries)==1 and boundaries[0].get("call_preservation")==value
             and boundaries[0].get("conditional_close") is None and boundaries[0].get("reopen") is True
             and boundaries[0].get("lifecycle")=="proved_lifecycle" and "loop_entry" not in boundaries[0],
             "native CALL preservation is detached from its exact boundary")
    _require(isinstance(source_edits,list),"native CALL preservation requires its emitted edit")
    edits=[e for e in source_edits if isinstance(e,dict) and e.get("source")==procedure["source"]
           and [e.get("first_line"),e.get("last_line")]==span]
    _require(len(edits)==1 and edits[0].get("source_sha256")==procedure["source_sha256"]
             and isinstance(edits[0].get("replacement"),str),"native CALL preservation edit differs")
    text=edits[0]["replacement"]
    original=value["original_source"]
    _require(text.endswith(original) and text.count(original)==1,"native CALL must execute unchanged exactly once after retirement")
    context_member = _context_member(name, scope, coordinators)
    context=(scope["owner"] if context_member==plan["entry"] else "fort_child_context_"+sha256(context_member.encode()).hexdigest()[:12])
    _require(name==plan["entry"] or name in coordinators,"native CALL retirement lost its owning context")
    prefix=[line.strip() for line in text[:-len(original)].splitlines()]
    _require(boundaries[0].get("empty_call_guard") == empty_guard,
             "empty native CALL guard is detached from its original boundary")
    if empty_guard is not None:
        from .native_empty_record import unwrap_empty_call
        prefix = unwrap_empty_call(empty_guard, value, returned, prefix)
    close="fort_close_status = fort_scope_resident_close_v1("+context+")"
    reset=context+" = 0"
    _require(prefix.count(close)==prefix.count(reset)==1 and prefix.index(close)<prefix.index(reset),
             "native CALL retirement lost its full-domain close/reset before original work")
    # The close must execute on exactly the authentic context path. Counting
    # lines alone would accept an always-false wrapper or an inverted guard.
    import re
    expected=(["if (fort_borrowed_"+sha256(context_member.encode()).hexdigest()[:12]+") then"] if context_member!=plan["entry"] else [])
    expected.append("if ("+context+" /= 0) then")
    stack=[]
    for line in prefix:
        lower=line.lower()
        if re.fullmatch(r"if\s*\(.*\)\s*then",lower):
            stack.append(lower)
        elif lower in {"else","else then"} or lower.startswith(("else if", "elseif")):
            _require(bool(stack),"native CALL retirement has an unmatched alternative")
            stack[-1]="alternative:"+lower
        elif lower in {"endif","end if"}:
            _require(bool(stack),"native CALL retirement has an unmatched guard")
            stack.pop()
        elif line in {close,reset}:
            _require(stack==expected,"native CALL retirement changed its authentic context/borrowed guard")
        _require(not re.match(r"(?:return|go\s*to|goto|exit|cycle|stop|error\s+stop)\b",lower),
                 "native CALL retirement cannot bypass its mandatory close")
    _require(not stack,"native CALL retirement has an unterminated wrapper")
    position=prefix.index(close)
    _require(prefix[position+1:position+3]==[
        "if (fort_close_status /= FORT_SCOPE_OK) error stop 'lexical owner publication failed'",reset],
        "native CALL retirement must stop after a failed publication before resetting context")
    # Character actuals are outside this narrow preservation role. Normalize
    # only insignificant source whitespace/comments/continuations, not tokens.
    def original_call(text):
        return re.sub(r"(?m)![^\n]*|[\s&]","",text).upper()
    _require(original_call(value["original_source"])==original_call(value["original_call"])
             and original_call(value["original_call"]).startswith("CALL"),
             "native CALL preservation changed the exact original CALL spelling")


def _projected_call_site(route, caller, callee, scope):
    """Choose the exact original site from already authenticated source DAGs.

    Equal forwarding can occur at multiple reached CALLs. Select by the
    original node IDs first; canonical allocation maps never select a site.
    Separate source-task projections may retain the same consistent edge.
    """
    candidates = []
    projected_pair = False
    for member in scope["checked_execution_plan"]["storage_epoch"]["members"]:
        for task in member.get("source_device_tasks", []):
            projection = task.get("execution_binding", {}).get("call_projection")
            if projection is None:
                continue
            for edge in projection["calls"]:
                projected_pair |= (edge["caller"], edge["callee"]) == (caller, callee)
                if edge["selected_node_ids"] != route["selected_node_ids"]:
                    continue
                _require((edge["caller"], edge["callee"]) == (caller, callee)
                         and edge["handle_bindings"] == route["handle_forwarding"],
                         "conditional return projected original CALL linkage differs")
                candidates.append(edge)
    if not candidates:
        _require(not projected_pair, "conditional return requires its exact projected original CALL site")
        return None
    _require(all(value == candidates[0] for value in candidates),
             "conditional return has conflicting original CALL site projections")
    return candidates[0]


def _projected_implicit_uses(scope, procedure):
    """Transport typed uses over their already-validated CALL DAG.

    Each path starts at the exact checked operation and follows incoming slot
    forwarding. An unrelated sibling cannot justify an implicit capture.
    This runs after source and native access validation and grants no permits.
    """
    used = set()
    epoch = scope["checked_execution_plan"]["storage_epoch"]
    projections = [(row["procedure"], row["call_projection"]) for row in epoch.get("native_opaque_accesses", [])]
    projections.extend((member["procedure"], projection) for member in epoch["members"]
                       for task in member.get("source_device_tasks", [])
                       if (projection := task.get("execution_binding", {}).get("call_projection")) is not None)
    for leaf, projection in projections:
        nodes = {node["procedure"]: node for node in projection["members"]}
        if procedure not in nodes:
            continue
        incoming = {name: [] for name in nodes}
        for edge in projection["calls"]:
            incoming[edge["callee"]].append(edge)
        seeds = [(leaf, slot, set(projection["field_roots"][local]))
                 for local, slot in projection["field_slots"].items()]
        for start, slot, canonical in seeds:
            pending = [(start, slot)]
            seen = set()
            while pending:
                name, slot = pending.pop()
                if (name, slot) in seen:
                    continue
                seen.add((name, slot))
                node = nodes[name]
                root = next(root for root, value in node["resource_slots"].items() if value == slot)
                if name == procedure and root in canonical and not root.startswith("argument::"):
                    used.add(root)
                if "inherited_namespace" in node:
                    pending.append((node["inherited_namespace"]["parent"], slot))
                elif node["parameterized"]:
                    pending.extend((edge["caller"], edge["handle_bindings"][str(slot)]) for edge in incoming[name])
    return used


def _closed_implicit_uses(relations, procedures, scalar_uses, scope, manifest):
    """Close uses only after every existing original CALL route is checked.

    The first phase has checked selections, associations, descriptors and slot
    forwarding. This finite closure discharges its remaining use obligations;
    it issues no new CALL, capture, allocation or payload authority.
    """
    from heapq import heappop, heappush

    plan = scope["checked_execution_plan"]
    descriptors = {row["resource"]: row["descriptor"] for row in plan["resources"]}
    reservations = {row["resource"] for row in plan["storage_epoch"]["resources"]
                    if row.get("storage") == "opaque host reservation"
                    and row.get("device_replica_authorized") is False
                    and row.get("initialization_authorized") is False}
    formal_reservations_validated = False
    members = {member["procedure"]: member for member in plan["storage_epoch"]["members"]}
    uses, parents, children = {}, {name: [] for name in procedures}, {name: [] for name in procedures}
    for name, procedure in procedures.items():
        direct = {root for unit in procedure["units"] for root in unit["resources"].values()}
        for task in members[name].get("source_device_tasks", ()):
            binding = task["execution_binding"]
            if "call_projection" not in binding:
                # Source tasks have already passed their typed field, canonical
                # descriptor and original CALL association checks. Direct module
                # fields need the same use accounting as ordinary numeric units;
                # they need not also appear in a unit or a projected CALL DAG.
                # Formal actuals and varying associations retain their existing
                # explicit-route and projected-use checks below.
                direct.update(root for local, root in binding["canonical_resources"].items()
                              if local == root and not root.startswith("argument::"))
        direct.update(scalar_uses.get(name, ()))
        direct.update(access["resource"] for unit in procedure["units"]
                      for operation in unit["operations"].get("native_operations", ())
                      for access in (operation.get("native_opaque_accesses") or {}).get("accesses", ()))
        for unit in procedure["units"]:
            for operation in unit["operations"].get("native_operations", ()):
                # A reached BYTE reservation is a physical use of its module
                # storage, just as registration in unit.resources is. It is
                # not a numeric capture or an inferred use of an unused actual.
                # Joined and inline native work can reserve record metadata
                # too; their checked native operations already retain that use.
                # Bind it to the same canonical descriptor and opaque epoch role.
                for root, metadata in operation.get("host_metadata", {}).items():
                    if root.startswith("argument::"):
                        # A dummy keeps its own name, bounds and INTENT. Its
                        # canonical allocation belongs to an explicit CALL
                        # association, not this implicit-module-use closure.
                        # Reuse the typed reader's complete local descriptor,
                        # CALL projection and canonical reservation checks.
                        if not formal_reservations_validated:
                            from .native_opaque_access_record import validate_native_opaque_accesses

                            validate_native_opaque_accesses(plan["storage_epoch"], plan, manifest, scope)
                            formal_reservations_validated = True
                        _require(root in {access["resource"] for access in
                            (operation.get("native_opaque_accesses") or {}).get("accesses", ())},
                            "conditional return native reservation formal lacks its typed original access")
                        continue
                    _require(root in reservations and metadata.get("device_capture") is False
                             and metadata.get("descriptor") == descriptors.get(root),
                             "conditional return native reservation descriptor or storage role differs")
                    direct.add(root)
        direct.update(_projected_implicit_uses(scope, name))
        uses[name] = {root for root in direct if not root.startswith("argument::")}
    for relation in relations:
        caller, callee = relation["caller"], relation["callee"]
        _require(caller in procedures and callee in procedures and caller != callee,
                 "conditional return checked CALL relation has a foreign member or cycle")
        parents[callee].append(relation)
        children[caller].append(callee)
        uses[caller].update(root for root in relation["explicit"] if not root.startswith("argument::"))
    reached, pending = set(), [scope["checked_execution_plan"]["entry"]]
    while pending:
        name = pending.pop()
        if name not in reached:
            reached.add(name)
            pending.extend(children[name])
    _require(reached == set(procedures), "conditional return checked CALL relation has a disconnected member")
    remaining = {name: len(callees) for name, callees in children.items()}
    ready = []
    for name, count in remaining.items():
        if count == 0:
            heappush(ready, name)
    completed = set()
    while ready:
        name = heappop(ready)
        completed.add(name)
        for relation in parents[name]:
            caller, resources = relation["caller"], relation["resources"]
            forwarded = uses[name] if resources is None else {
                root for root, canonical in resources.items() if root == canonical and root in uses[name]}
            uses[caller].update(forwarded)
            remaining[caller] -= 1
            if remaining[caller] == 0:
                heappush(ready, caller)
    _require(completed == set(procedures), "conditional return checked CALL relation contains a cycle")
    return uses


def _child(route, caller, procedures, coordinators, canonical, scope, resource_rows):
    callee = route["procedure"]
    _require(_text(callee) and callee in coordinators and callee in procedures, "conditional return checked child differs")
    mappings, handles, forward = (route[key] for key in ("canonical_resources", "canonical_handles", "handle_forwarding"))
    _require(isinstance(mappings, dict) and all(_text(k) and _text(v) and v in canonical for k, v in mappings.items())
             and isinstance(handles, dict) and isinstance(forward, dict), "conditional return child canonical storage differs")
    expected = {str(_slot(callee, root)): canonical[value] for root, value in mappings.items()}
    _require(handles == expected and set(forward) == set(expected)
             and all(type(n) is int and n > 0 for n in [*handles.values(), *forward.values()]),
             "conditional return canonical registration or local-slot identity differs")
    projected = _projected_call_site(route, caller, callee, scope)
    calls = [call for call in procedures[callee]["calls"] if call.get("caller") == caller and call.get("handle_bindings") == forward]
    if projected is not None:
        calls = [call for call in calls if (call.get("first_line"), call.get("last_line")) ==
                 (projected["first_line"], projected["last_line"])
                 and call.get("resource_mappings") == projected["resource_mappings"]]
        _require(len(calls) == 1 and projected["caller_graph"] == procedures[caller]["structured_summary_identity"],
                 "conditional return projected CALL differs from its exact retained companion site")
    _require(bool(calls), "conditional return child handles differ from the checked original CALL")
    parent_namespace = _context_member(caller, scope, coordinators)
    parent_map = (coordinators[parent_namespace]["resource_mappings"] if parent_namespace in coordinators
                  else {root: root for root in canonical})
    for call in calls:
        actuals = call.get("resource_mappings")
        if not isinstance(actuals, dict):
            continue
        # An unused actual in an empty child has no retained resource slot.
        # Only associations already needed by the checked child seed a use.
        explicit = {mappings[root] for root in actuals if root in mappings}
        missing = set(mappings) - set(actuals)
        if missing:
            # Implicit module captures have no dummy/actual association. Their
            # global identity, descriptors and slots are checked immediately.
            # Only evidence of an actual use is deferred until all original
            # child routes pass the same checks; explicit formals stay exact.
            bindings = scope.get("resource_bindings")
            _require(isinstance(bindings, dict)
                     and isinstance(bindings.get("resources"), list)
                     and _digest(resource_rows) == _digest(scope["checked_execution_plan"]["resources"]),
                     "conditional return implicit module descriptor inventory differs")
            descriptors = {row["resource"]: row["descriptor"] for row in resource_rows}
            if not all(not root.startswith("argument::") and root == mappings[root]
                       and root in descriptors and descriptors[root].get("resource") == root
                       for root in missing):
                continue
            actuals = {**actuals, **{root: root for root in missing}}
        if all(actuals[root] == value for root, value in mappings.items()):
            local_actuals = projected["source_call"]["resource_mapping"] if projected is not None else None
            for root, value in mappings.items():
                slot = str(_slot(callee, root))
                local = actuals[root]
                if local_actuals is not None and root in local_actuals:
                    local = local_actuals[root]
                    _require(parent_map.get(local, local) == actuals[root],
                             "conditional return projected local/canonical association differs")
                elif parent_namespace in coordinators:
                    # Retained CALL mappings are canonical allocation names;
                    # helper handle numbers belong to the parent's local names.
                    locals_for_actual = [name for name, target in parent_map.items() if target == actuals[root]]
                    if not locals_for_actual and not local.startswith("argument::"):
                        locals_for_actual = [local]  # Implicit module capture.
                    _require(len(locals_for_actual) == 1,
                             "conditional return child requires an unambiguous parent-local association")
                    local = locals_for_actual[0]
                parent_slot = _slot(parent_namespace, local) if parent_namespace in coordinators else canonical[value]
                _require(forward[slot] == parent_slot, "conditional return child-to-parent local handles differ")
            return {"caller": caller, "callee": callee, "resources": dict(mappings),
                    "explicit": explicit, "implicit": missing}
    _require(False, "conditional return child formal-to-canonical association differs")
