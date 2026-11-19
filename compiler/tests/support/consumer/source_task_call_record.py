"""Validate a projection of checked CALLs; issue no source or runtime authority."""
from .checked_execution_plan import _digest, _require, _sha256, _text
from .source_task_record import _IDENT, _binding, _keys

_FLAGS = {"definition_authorized", "lifetime_authorized", "alias_authorized", "value_specialization_authorized"}
_CALL = {"schema_version", "procedure", "target", "source_kind", "formal_arguments", "actual_arguments",
         "original_arguments", "resource_mapping", "resource_mappings", "actual_evaluation"}
_MAPPING = {"formal", "formal_resource", "formal_descriptor", "actual", "actual_descriptor", "resource",
            "storage", "presence", "section", "requirements"}
_REQUIREMENTS = {"descriptor_dependent_presence", "original_actual_presence", "original_allocation_descriptor",
                 "physical_section_mapping", "presence_preserving_forwarding"}


def require_single_association(procedure, owner, manifest):
    """An old singleton binding cannot hide varying enclosing CALL mappings."""
    for scope in manifest.get("scopes", []):
        plan = scope.get("checked_execution_plan", {})
        if plan.get("entry") != owner or procedure not in plan.get("procedures", []):
            continue
        procedures = {p["procedure"]: p for p in plan["procedures"]}
        parameterized = {p["procedure"] for p in scope.get("module_coordinators", [])
                         if p.get("parameterized_handles") is True}
        inherited = {p["procedure"]: p.get("lexical_namespace")
                     for p in scope.get("internal_coordinators", [])}
        coordinators = {p["procedure"]: p for p in scope.get("module_coordinators", [])
                        + scope.get("internal_coordinators", [])}
        _require(not set(inherited) & {p["procedure"] for p in scope.get("module_coordinators", [])},
                 "source line task has conflicting lexical and module namespaces")
        pending, seen = [procedure["procedure"]], set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            relation = inherited.get(name)
            if relation is not None:
                parent, _namespace = validate_lexical_namespace(
                    relation, name, procedures, coordinators, scope, manifest)
                pending.append(parent)
                continue
            if name not in parameterized:
                continue  # This invocation already uses the owning registrations.
            calls = [call for call in procedures[name]["calls"]
                     if call.get("procedure", name) == name and call.get("caller") != name]
            if calls:
                _require(all(c.get("resource_mappings") == calls[0].get("resource_mappings") for c in calls),
                         "source line task requires its varying CALL projection")
                pending.extend(c["caller"] for c in calls if c["caller"] in procedures)


def _slots(value):
    _require(isinstance(value, dict) and all(_text(k) and type(v) is int and v > 0 for k, v in value.items())
             and len(set(value.values())) == len(value), "source CALL projection requires distinct exact slots")


def validate_lexical_namespace(value, name, procedures, coordinators, scope, manifest):
    """Bind an existing argumentless lexical ABI to its original parent CALLs."""
    _keys(value, {"version", "kind", "parent", "namespace_owner", "source_calls"})
    parent, namespace = value["parent"], value["namespace_owner"]
    _require(type(value["version"]) is int and value["version"] == 1
             and value["kind"] == "original_argumentless_lexical_namespace"
             and _text(parent) and parent in procedures and name.startswith(parent + "::")
             and "::" not in name[len(parent) + 2:]
             and (namespace is None or _text(namespace) and namespace in coordinators
                  and coordinators[namespace].get("parameterized_handles") is True),
             "source CALL projection inherited lexical namespace differs")
    rows = value["source_calls"]
    _require(isinstance(rows, list) and bool(rows), "source CALL projection lost its original lexical CALLs")
    expected = [call for call in procedures[parent]["calls"]
                if call.get("procedure") == name and call.get("execution") ==
                "original internal procedure; host-associated owner control"]
    calls, selections, sites = [], set(), []
    from .source_placement_record import _tasks

    graphs = [row for row in scope.get("graph_execution", {}).get("procedures", [])
              if row.get("procedure") == parent]
    _require(len(graphs) == 1, "source CALL projection lost its original lexical parent graph")
    tasks = {task["id"]: task for task in _tasks(graphs[0].get("task_graph"))}
    for row in rows:
        _keys(row, {"source_call", "caller_graph", "selected_node_ids"})
        call, selected = row["source_call"], row["selected_node_ids"]
        _keys(call, _CALL | {"caller", "source", "first_line", "last_line", "execution"})
        _require(type(call["schema_version"]) is int and call["schema_version"] == 2
                 and call["procedure"] == name and call["caller"] == parent and call["source_kind"] == "internal"
                 and _text(call["target"]) and call["source"] == procedures[parent]["source"]
                 and call["formal_arguments"] == call["actual_arguments"] == call["original_arguments"] == []
                 and call["resource_mapping"] == {} and call["resource_mappings"] == []
                 and call["actual_evaluation"] == "original call position and guards; no analysis-time evaluation"
                 and call["execution"] == "original internal procedure; host-associated owner control"
                 and type(call["first_line"]) is int and type(call["last_line"]) is int
                 and 1 <= call["first_line"] <= call["last_line"]
                 and call in manifest.get("resolved_calls", []) and call in expected,
                 "source CALL projection changed its original argumentless lexical CALL")
        _require(row["caller_graph"] == procedures[parent]["structured_summary_identity"]
                 and isinstance(selected, list) and len(selected) == 1 and _text(selected[0])
                 and selected[0].startswith(parent + "/") and selected[0] in tasks
                 and tasks[selected[0]].get("opaque") is True and selected[0] not in selections,
                 "source CALL projection changed its original lexical selection")
        selections.update(selected)
        calls.append(call)
        sites.append((call["first_line"], call["last_line"]))
    _require(calls == expected and sites == sorted(set(sites)),
             "source CALL projection omitted or repeated an original lexical CALL")
    return parent, namespace


def _original_call(call, callee, caller, *, scalar_elements_validated=False):
    _keys(call, _CALL)
    _require(type(call["schema_version"]) is int and call["schema_version"] == 2
             and call["procedure"] == callee["procedure"] and _text(call["target"]) and _text(call["source_kind"])
             and call["actual_evaluation"] == "original call position and guards; no analysis-time evaluation",
             "source CALL projection changed original evaluation")
    formals, actuals = call["formal_arguments"], call["actual_arguments"]
    _require(isinstance(formals, list) and bool(formals)
             and all(isinstance(n, str) and _IDENT.fullmatch(n) for n in formals)
             and len(formals) == len(set(formals)) and isinstance(actuals, list) and len(actuals) == len(formals)
             and all(a is None or _text(a) for a in actuals), "source CALL projection argument inventory differs")
    original = call["original_arguments"]
    _require(isinstance(original, list), "source CALL projection original arguments differ")
    seen = set()
    for index, row in enumerate(original):
        _keys(row, {"position", "formal", "keyword", "actual"})
        _require(type(row["position"]) is int and row["position"] == index
                 and isinstance(row["formal"], str) and row["formal"] in formals and row["formal"] not in seen
                 and _text(row["actual"]) and row["actual"] == actuals[formals.index(row["formal"])]
                 and (row["keyword"] is None or isinstance(row["keyword"], str)
                      and row["keyword"].lower() == row["formal"]), "source CALL projection argument order differs")
        seen.add(row["formal"])
    _require(seen == {f for f, a in zip(formals, actuals, strict=True) if a is not None},
             "source CALL projection lost original supplied arguments")
    rows = call["resource_mappings"]
    _require(isinstance(rows, list) and len(rows) == len(formals), "source CALL projection mappings differ")
    mapped, arrays = {}, {}
    for formal, actual, row in zip(formals, actuals, rows, strict=True):
        has_element = isinstance(row, dict) and row.get("storage") == "scalar_element"
        _keys(row, _MAPPING | ({"scalar_element"} if has_element else set()))
        _require(not has_element or scalar_elements_validated,
                 "source CALL projection scalar element lacks its checked coordinate contract")
        descriptor = row["formal_descriptor"]
        _require(row["formal"] == formal and row["formal_resource"] == "argument::" + formal
                 and row["actual"] == actual and isinstance(descriptor, dict)
                 and descriptor.get("name") == formal and descriptor.get("resource") == row["formal_resource"]
                 and type(descriptor.get("rank")) is int and descriptor["rank"] >= 0
                 and (row["resource"] is None or _text(row["resource"]))
                 and _text(row["presence"]) and _text(row["storage"]),
                 "source CALL projection original association differs")
        _keys(row["requirements"], _REQUIREMENTS)
        _require(all(type(v) is bool for v in row["requirements"].values()),
                 "source CALL projection original requirements differ")
        if row["resource"] is not None:
            mapped[row["formal_resource"]] = row["resource"]
        if not descriptor["rank"]:
            continue
        registered = row["formal_resource"] in callee["resource_slots"]
        # Host-only original associations can exceed the device task ABI rank.
        # They do not become registered fields or authorize device access.
        _binding(descriptor, max_rank=3 if registered else 15)
        _binding(row["actual_descriptor"], max_rank=3 if registered else 15)
        _require(row["presence"] == "supplied" and row["storage"] == "whole" and row["section"] is None
                 and row["resource"] == row["actual_descriptor"]["resource"]
                 and all(descriptor[k] == row["actual_descriptor"][k] for k in ("type", "kind", "rank")),
                 "source CALL projection requires original whole-array slot associations")
        # Original CALL associations also include unused array formals. Only
        # registered formals transport handles; an unused formal grants no slot.
        if row["formal_resource"] in callee["resource_slots"]:
            _require(row["resource"] in caller["resource_slots"],
                     "source CALL projection registered formal lacks its caller slot")
        arrays[row["formal_resource"]] = row
    _require(call["resource_mapping"] == mapped, "source CALL projection source mapping inventory differs")
    return arrays


def _graph_effects(graph, procedure, ordinal, effects):
    _require(isinstance(graph, dict) and isinstance(graph.get("procedures"), list),
             "source CALL projection requires its containing task graph")
    rows = [p for p in graph["procedures"] if isinstance(p, dict) and p.get("procedure") == procedure]
    _require(len(rows) == 1, "source CALL projection task graph procedure differs")
    target = procedure + ":source_device:" + str(ordinal)
    found = []
    pending = [(rows[0].get("task_graph"), ())]
    while pending:
        value, ancestors = pending.pop()
        if isinstance(value, list):
            pending.extend((item, ancestors) for item in value)
        elif isinstance(value, dict):
            if value.get("id") == target:
                found.append(value)
                for key in ("reads", "may_writes", "metadata_reads"):
                    _require(value.get(key) == effects[key], "source CALL projection task effects differ")
                    for ancestor in ancestors:
                        _require(isinstance(ancestor.get(key), list) and set(effects[key]) <= set(ancestor[key]),
                                 "source CALL projection enclosing graph effects omit a root")
                _require(value.get("implementations") == ["gpu"] and value.get("opaque") is False
                         and value.get("ordered") is True, "source CALL projection task placement differs")
                _require(all(value.get(key, []) == [] for key in ("must_writes", "overwrites")),
                         "source CALL projection cannot grant graph definition or overwrite credit")
                for ancestor in ancestors:
                    _require(all(isinstance(ancestor.get(key, []), list)
                                 and all(_text(root) for root in ancestor.get(key, []))
                                 and not set(ancestor.get(key, [])) & set(effects["reads"])
                                 for key in ("must_writes", "overwrites")),
                             "source CALL projection enclosing graph cannot define a projected root")
            next_ancestors = ancestors + (value,) if "id" in value else ancestors
            pending.extend((child, next_ancestors) for child in value.values() if isinstance(child, (dict, list)))
    _require(len(found) == 1, "source CALL projection task graph lost or repeated its source task")


def validate_call_inventory(value, local_fields, procedure, resources, owner, manifest, *, require_calls=False):
    """The existing checked companion DAG, shared by numeric and opaque roles."""
    contexts = [s for s in manifest.get("scopes", []) if isinstance(s, dict)
                and isinstance(s.get("checked_execution_plan"), dict)
                and s["checked_execution_plan"].get("entry") == owner
                and procedure in s["checked_execution_plan"].get("procedures", [])]
    _require(len(contexts) == 1, "source CALL projection requires one containing owning plan")
    scope = contexts[0]
    plan = scope["checked_execution_plan"]
    procedures = {p["procedure"]: p for p in plan["procedures"]}
    coordinators = {p["procedure"]: p for p in scope.get("module_coordinators", [])
                    + scope.get("internal_coordinators", [])}
    _require(isinstance(value["members"], list) and bool(value["members"]), "source CALL projection members differ")
    nodes = {}
    for node in value["members"]:
        _require(isinstance(node, dict) and type(node.get("parameterized")) is bool,
                 "source CALL projection member namespace differs")
        _keys(node, {"procedure", "source_graph", "source", "source_sha256", "parameterized", "resource_slots"}
              | (set() if node["parameterized"] else
                 {"inherited_namespace"} if "inherited_namespace" in node else {"owner_resources"}))
        name = node["procedure"]
        _require(_text(name) and name in procedures and name not in nodes
                 and all(node[k] == procedures[name][p] for k, p in
                         (("source", "source"), ("source_sha256", "source_sha256"),
                          ("source_graph", "structured_summary_identity"))),
                 "source CALL projection member source differs")
        _slots(node["resource_slots"])
        coordinator = coordinators.get(name, {})
        if node["parameterized"]:
            incoming_calls = [call for call in procedures[name]["calls"] if "handle_bindings" in call]
            _require(incoming_calls == coordinator.get("calls"),
                     "source CALL projection original array association or companion site inventory differs")
        _require(node["parameterized"] is (coordinator.get("parameterized_handles") is True),
                 "source CALL projection namespace differs from its existing companion ABI")
        if "inherited_namespace" in node:
            _require(name in {row["procedure"] for row in scope.get("internal_coordinators", [])}
                     and node["inherited_namespace"] == coordinator.get("lexical_namespace"),
                     "source CALL projection inherited namespace differs from its lexical ABI")
            validate_lexical_namespace(node["inherited_namespace"], name, procedures, coordinators, scope, manifest)
        elif not node["parameterized"]:
            canonical = node["owner_resources"]
            _require(isinstance(canonical, dict) and set(canonical) == set(node["resource_slots"])
                     and all(_text(root) and root in resources and node["resource_slots"][local]
                             == resources[root]["registration_identity"] for local, root in canonical.items()),
                     "source CALL projection terminal owner registrations differ")
            _require(all(root == local if name == owner else
                         coordinator.get("resource_mappings", {}).get(local, local) == root
                         for local, root in canonical.items()),
                     "source CALL projection terminal canonical associations differ")
        nodes[name] = node
    _require(list(nodes) == sorted(nodes) and procedure["procedure"] in nodes,
             "source CALL projection member order or leaf differs")
    for name, node in nodes.items():
        relation = node.get("inherited_namespace")
        if relation is None:
            continue
        parent, namespace = relation["parent"], relation["namespace_owner"]
        _require(parent in nodes and namespace in nodes and namespace != name
                 and nodes[namespace]["parameterized"] and node["resource_slots"] == nodes[parent]["resource_slots"],
                 "source CALL projection inherited slots or namespace owner differ")
        seen, current = set(), name
        while current != namespace:
            _require(current not in seen and current in nodes and "inherited_namespace" in nodes[current]
                     and nodes[current]["inherited_namespace"]["namespace_owner"] == namespace,
                     "source CALL projection inherited namespace chain is incomplete or cyclic")
            seen.add(current)
            current = nodes[current]["inherited_namespace"]["parent"]
    _slots(value["field_slots"])
    leaf = nodes[procedure["procedure"]]
    _require(set(value["field_slots"]) == set(local_fields)
             and all(leaf["resource_slots"].get(k) == v for k, v in value["field_slots"].items()),
             "source CALL projection field slots differ")
    edges = value["calls"]
    _require(isinstance(edges, list) and (bool(edges) if require_calls else
                 bool(edges) or not any(node["parameterized"] for node in nodes.values())),
             "source CALL projection requires its complete original CALL DAG")
    incoming, edge_keys, selections = {name: [] for name in nodes}, [], set()
    scalar_elements_validated = False
    for edge in edges:
        _keys(edge, {"caller", "callee", "first_line", "last_line", "selected_node_ids", "caller_graph",
                     "handle_bindings", "storage_identities", "resource_mappings", "source_call"})
        caller, callee = edge["caller"], edge["callee"]
        _require(_text(caller) and _text(callee) and caller in nodes and callee in nodes and caller != callee
                 and nodes[callee]["parameterized"] and edge["caller_graph"] == nodes[caller]["source_graph"]
                 and type(edge["first_line"]) is int and type(edge["last_line"]) is int
                 and 1 <= edge["first_line"] <= edge["last_line"], "source CALL projection original site differs")
        selected = edge["selected_node_ids"]
        _require(isinstance(selected, list) and bool(selected)
                 and all(_text(n) and n.startswith(caller + "/") for n in selected)
                 and len(selected) == len(set(selected)) and not selections.intersection(selected),
                 "source CALL projection original selection differs")
        selections.update(selected)
        key = caller, callee, edge["first_line"], edge["last_line"]
        edge_keys.append(key)
        calls = [c for c in procedures[callee]["calls"] if (c.get("caller"), c.get("first_line"), c.get("last_line"))
                 == (caller, edge["first_line"], edge["last_line"])]
        _require(len(calls) == 1 and all(calls[0].get(k) == edge[k] for k in ("handle_bindings", "resource_mappings"))
                 and any(_digest(edge["source_call"]) == _digest(call) for call in manifest.get("resolved_calls", [])),
                 "source CALL projection differs from its retained companion CALL")
        mapping = edge["handle_bindings"]
        _slots(mapping)
        _require(set(mapping) == {str(v) for v in nodes[callee]["resource_slots"].values()}
                 and set(mapping.values()) <= set(nodes[caller]["resource_slots"].values()),
                 "source CALL projection forwarding is incomplete")
        identities = edge["storage_identities"]
        _require(isinstance(identities, dict) and set(identities) == set(mapping),
                 "source CALL projection storage forwarding differs")
        for slot, actual in mapping.items():
            _keys(identities[slot], {"kind", "slot"})
            _require(type(identities[slot]["slot"]) is int and identities[slot]["slot"] == actual
                     and identities[slot]["kind"] == ("forwarded_slot" if nodes[caller]["parameterized"]
                         or "inherited_namespace" in nodes[caller] else "owner_registration"),
                     "source CALL projection changed storage identity forwarding")
        _keys(edge["source_call"], _CALL)
        associations = edge["source_call"]["resource_mappings"]
        _require(isinstance(associations, list), "source CALL projection mappings differ")
        has_elements = any(isinstance(row, dict) and row.get("storage") == "scalar_element"
                           for row in associations)
        if has_elements:
            coordinates = calls[0].get("original_scalar_element_coordinates")
            _require(isinstance(coordinates, dict) and coordinates.get("call") == edge["source_call"],
                     "source CALL projection scalar elements differ from their original coordinate contract")
            if not scalar_elements_validated:
                from .scalar_element_record import validate_scalar_element_reservations

                epoch = plan["storage_epoch"]
                validate_scalar_element_reservations(scope, plan, manifest, epoch,
                                                     epoch["return_contract"]["members"], coordinators)
                scalar_elements_validated = True
        arrays = _original_call(edge["source_call"], nodes[callee], nodes[caller],
                                scalar_elements_validated=has_elements and scalar_elements_validated)
        _require(isinstance(edge["resource_mappings"], dict) and set(edge["resource_mappings"]) == set(arrays),
                 "source CALL projection canonical actual inventory differs")
        for local, row in arrays.items():
            if local not in nodes[callee]["resource_slots"]:
                continue
            _require(mapping[str(nodes[callee]["resource_slots"][local])] == nodes[caller]["resource_slots"][row["resource"]],
                     "source CALL projection handle differs from original array association")
        for local, slot in nodes[callee]["resource_slots"].items():
            if local not in arrays:
                _require(not local.startswith("argument::") and local in nodes[caller]["resource_slots"]
                         and mapping[str(slot)] == nodes[caller]["resource_slots"][local],
                         "source CALL projection implicit module storage changed identity")
        incoming[callee].append(edge)
    _require(edge_keys == sorted(set(edge_keys)), "source CALL projection repeats or reorders original CALLs")
    for name, node in nodes.items():
        if node["parameterized"]:
            expected = {(c["caller"], name, c["first_line"], c["last_line"]) for c in procedures[name]["calls"]
                        if "handle_bindings" in c}
            _require(bool(expected) and expected == {k for k in edge_keys if k[1] == name},
                     "source CALL projection omits a retained incoming CALL")
    # Finite transport consistency over the already-issued DAG. It does not
    # reconstruct source association proofs or enumerate joint invocation paths.
    memo, active, reached = {}, set(), set()
    def roots(name, slot):
        key = name, slot
        if key in memo:
            return memo[key]
        _require(key not in active, "source CALL projection contains a cycle")
        active.add(key)
        reached.add(name)
        node = nodes[name]
        reverse = {number: root for root, number in node["resource_slots"].items()}
        _require(slot in reverse, "source CALL projection lost a resource slot")
        if "inherited_namespace" in node:
            result = roots(node["inherited_namespace"]["parent"], slot)
        elif node["parameterized"]:
            result = set().union(*(roots(e["caller"], e["handle_bindings"][str(slot)]) for e in incoming[name]))
        else:
            result = {node["owner_resources"][reverse[slot]]}
        active.remove(key)
        memo[key] = result
        return result
    expected_roots = {field: sorted(roots(procedure["procedure"], slot)) for field, slot in value["field_slots"].items()}
    _require(reached == set(nodes) and value["field_roots"] == expected_roots,
             "source CALL projection field closure differs or includes foreign members")
    for edge in edges:
        for local, canonical in edge["resource_mappings"].items():
            _require(_text(canonical), "source CALL projection canonical association differs")
            if local not in nodes[edge["callee"]]["resource_slots"]:
                # No handle or effect is inferred from this original association.
                # Where its caller already has a slot, retain the canonical check.
                row = next(row for row in edge["source_call"]["resource_mappings"]
                           if row["formal_resource"] == local)
                actual_slot = nodes[edge["caller"]]["resource_slots"].get(row["resource"])
                if actual_slot is not None:
                    _require(canonical in roots(edge["caller"], actual_slot),
                             "source CALL projection unregistered formal changed its caller association")
                continue
            slot = nodes[edge["callee"]]["resource_slots"][local]
            _require(_text(canonical) and canonical in roots(edge["caller"], edge["handle_bindings"][str(slot)]),
                     "source CALL projection canonical representative is outside its original slot closure")
    return expected_roots, scope, plan


def validate_call_projection(value, record, fields, procedure, resources, owner, manifest):
    # Atomic v4 retains the exact source team separately from this CALL
    # transport. Its numerical alternatives and providers keep their versions.
    atomic = record.get("kind") == "original_atomic_conditional_joined_team"
    if atomic:
        _require(type(record.get("version")) is int and record["version"] == 4
                 and isinstance(record.get("source_proof"), dict),
                 "source CALL projection requires projected atomic transport")
        source = record["source_proof"]
        source_identity = source.get("identity")
    else:
        source = record
        source_identity = record.get("source_proof_identity")
    _keys(value, {"identity", "version", "kind", "owner", "procedure", "source_proof_identity", "field_slots",
                  "members", "calls", "field_roots", "root_registrations", "projection_semantics",
                  "conservative_effects", "runtime_validation"} | _FLAGS)
    projected_version = 2 if any("inherited_namespace" in node for node in value.get("members", [])
                                 if isinstance(node, dict)) else 1
    _require(type(value["version"]) is int and value["version"] == projected_version
             and value["kind"] == "original_source_task_companion_projection"
             and _sha256(value["identity"]) and value["identity"] == _digest({k: v for k, v in value.items() if k != "identity"})
             and value["owner"] == owner and value["procedure"] == procedure["procedure"] == source["procedure"]
             and value["source_proof_identity"] == source_identity
             and all(value[k] is False for k in _FLAGS)
             and value["projection_semantics"] == "per-field effect closure; joint associations remain in the checked CALL DAG"
             and value["runtime_validation"] == "original reached registration and existing source access transaction validate each invocation",
             "source CALL projection identity or authority differs")
    local_fields = {f["binding"]["resource"]: f for f in fields}
    expected_roots, scope, plan = validate_call_inventory(
        value, local_fields, procedure, resources, owner, manifest, require_calls=True)
    actions = {}
    for local, field in local_fields.items():
        for root in expected_roots[local]:
            _require(root in resources and resources[root].get("device_replica_authorized") is True
                     and all(field["binding"][k] == resources[root]["descriptor"].get(k) for k in ("type", "kind", "rank")),
                     "source CALL projection target type or capture authority differs")
            actions[root] = actions.get(root, False) or field["written"]
    registrations = value["root_registrations"]
    _slots(registrations)
    _require(registrations == {root: resources[root]["registration_identity"] for root in sorted(actions)},
             "source CALL projection final root registrations differ")
    effects = {"reads": sorted(actions), "may_writes": sorted(root for root, written in actions.items() if written),
               "metadata_reads": sorted(actions), "must_writes": [], "overwrites": []}
    _require(value["conservative_effects"] == effects, "source CALL projection conservative effects differ")
    members = plan["storage_epoch"]["members"]
    tasks = next(m["source_device_tasks"] for m in members if m["procedure"] == source["procedure"])
    positions = [i for i, task in enumerate(tasks) if task == record]
    _require(len(positions) == 1, "source CALL projection task position differs")
    _graph_effects(scope.get("graph_execution"), source["procedure"], positions[0], effects)
    return expected_roots
