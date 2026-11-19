"""Check opaque scalar-element reservations against existing original CALL facts.

This validates transport consistency. The retained coordinate issuer owns the
original selector, declaration and readonly closure; this reader never evaluates
them or creates payload, definition, lifetime or source authority.
"""
from .checked_execution_plan import _digest, _require, _sha256, _text
from .source_task_record import _IDENT, _binding, _keys

_BASE = {"resource", "registration_identity", "descriptor"}
_ROLE = {"role", "managed_payload", "definition_authority"}
_PROOF_FALSE = {"bounds_evaluated", "execution_authorized", "entry_definition_authority",
                "allocation_freshness_authority", "gpu_legality_established"}
_PROOF_TRUE = {"requires_original_call_and_guards", "requires_registered_child_effect_and_completion_proofs",
               "requires_runtime_layout_alias_and_integer_checks", "mapping_does_not_apply_entry_kills"}
_PROOF = {"proof_identity", "schema_version", "kind", "caller", "callee", "node_id", "guard",
          "structured_identity", "analysis_identity", "source_identity", "call", "coordinate_associations",
          "scalar_mapping", "scalar_bounds", "original_scalar_components", "original_scalar_elements",
          "scalar_element_closure", "unmanaged_immutable_associations", "entry_definition_changes",
          "source_work", "requires_immutable_array_input_for_immutable_associations", "limits"} | _PROOF_FALSE | _PROOF_TRUE
_ELEMENT = {"resource", "actual", "backing_descriptor", "index", "bounds", "association",
            "declaration_initializer", "declaration_scan_work", "payload_evaluation",
            "managed_storage_authority", "definition_authority"}
_MAPPING = {"formal", "formal_resource", "actual", "resource", "storage", "presence",
            "formal_descriptor", "actual_descriptor", "section", "requirements"}
_CALL = {"schema_version", "procedure", "target", "source_kind", "formal_arguments", "actual_arguments",
         "original_arguments", "resource_mapping", "resource_mappings", "actual_evaluation"}
_REQUIREMENTS = {"presence_preserving_forwarding", "descriptor_dependent_presence", "original_actual_presence",
                 "original_allocation_descriptor", "physical_section_mapping"}
_UNSAFE = {"pointer", "target", "allocatable", "optional", "value", "volatile", "asynchronous", "parameter"}


def _selector(index):
    _require(isinstance(index, dict) and _text(index.get("expression")), "scalar element selector differs")
    kind = index.get("kind")
    if kind == "scalar":
        _keys(index, {"kind", "expression", "resource", "dependencies"})
        _require(_IDENT.fullmatch(index["expression"].lower()) and _text(index["resource"])
                 and index["dependencies"] == [{"resource": index["resource"], "kind": "scalar_read"}],
                 "scalar element original selector dependency differs")
    elif kind == "literal":
        _keys(index, {"kind", "expression", "value", "dependencies"})
        _require(type(index["value"]) is int and -(2**31) <= index["value"] < 2**31
                 and index["dependencies"] == [], "scalar element literal INTEGER selector differs")
    else:
        _keys(index, {"kind", "expression", "operator", "children", "dependencies"})
        _require(kind == "unary" and index["operator"] in {"+", "-"}
                 and isinstance(index["children"], list) and len(index["children"]) == 1
                 and index["children"][0].get("kind") == "literal" and index["dependencies"] == [],
                 "scalar element selector exceeds its original unary literal contract")
        _selector(index["children"][0])


def _element(element, mapping, reservations):
    _keys(element, _ELEMENT)
    root = element["resource"]
    _require(isinstance(root, str) and root in reservations, "scalar element lacks its exact opaque registration")
    backing = element["backing_descriptor"]
    _binding(backing, rank=1)
    _require(backing == reservations[root]["descriptor"] == mapping["actual_descriptor"]
             and backing["resource"] == root and backing["type"] == "real" and backing["kind"] in (4, 8)
             and backing["intent"] is None and not set(backing["attributes"]) & _UNSAFE
             and len(root.split("::")) == 2 and root.split("::")[0] != "argument"
             and root.split("::")[1] == backing["name"],
             "scalar element backing descriptor or module storage differs")
    bounds = element["bounds"]
    _require(isinstance(bounds, list) and len(bounds) == 2
             and all(type(value) is int and -(2**31) <= value < 2**31 for value in bounds)
             and bounds[0] <= bounds[1] and len(backing["shape"]) == 1
             and len(backing["lower_bounds"]) == 1
             and backing["shape"][0]["kind"] == "Explicit_Shape_Spec",
             "scalar element fixed bounds differ")
    # Literal descriptor bounds can be cross-checked without evaluating a
    # specification expression; PARAMETER expressions remain issuer-owned.
    for spelling, expected in zip(backing["shape"][0]["bounds"], bounds, strict=True):
        if isinstance(spelling, str) and spelling.lstrip('+-').isdigit():
            _require(int(spelling) == expected, "scalar element declared bounds differ")
    _selector(element["index"])
    def compact(value):
        return ''.join(value.split()).lower()

    _require(_text(element["actual"]) and element["actual"] == mapping["actual"]
             and compact(element["actual"]) == compact(backing["name"] + '(' + element["index"]["expression"] + ')')
             and element["association"] == "original CALL element association"
             and _text(element["declaration_initializer"]) and element["declaration_initializer"].startswith("= ")
             and not element["declaration_initializer"].startswith("=>")
             and type(element["declaration_scan_work"]) is int and 0 < element["declaration_scan_work"] <= 65536
             and element["payload_evaluation"] == "original scalar dummy uses; no call-entry value snapshot"
             and element["managed_storage_authority"] is False and element["definition_authority"] is False,
             "scalar element original association or declaration differs")
    formal = mapping["formal_descriptor"]
    _binding(formal, rank=0)
    _require(formal["type"] == backing["type"] and formal["kind"] == backing["kind"]
             and formal["intent"] == "in" and formal["lower_bounds"] == formal["shape"] == []
             and not set(formal["attributes"]) & _UNSAFE,
             "scalar element requires its readonly matching scalar formal")
    return root


def _call(call, callee, reservations):
    _keys(call, _CALL)
    _require(type(call["schema_version"]) is int and call["schema_version"] == 2
             and call["procedure"] == callee and _text(call["target"]) and _text(call["source_kind"])
             and call["actual_evaluation"] == "original call position and guards; no analysis-time evaluation",
             "scalar element original CALL differs")
    formals, actuals = call["formal_arguments"], call["actual_arguments"]
    _require(isinstance(formals, list) and bool(formals)
             and all(isinstance(name, str) and _IDENT.fullmatch(name) for name in formals)
             and len(set(formals)) == len(formals) and isinstance(actuals, list) and len(actuals) == len(formals)
             and all(value is None or _text(value) for value in actuals), "scalar element formal inventory differs")
    original = call["original_arguments"]
    _require(isinstance(original, list), "scalar element original argument ordering differs")
    seen = set()
    for position, row in enumerate(original):
        _keys(row, {"position", "formal", "keyword", "actual"})
        _require(type(row["position"]) is int and row["position"] == position
                 and row["formal"] in formals and row["formal"] not in seen
                 and row["actual"] == actuals[formals.index(row["formal"])]
                 and (row["keyword"] is None or _text(row["keyword"]) and row["keyword"].lower() == row["formal"]),
                 "scalar element original argument association differs")
        seen.add(row["formal"])
    _require(seen == {name for name, value in zip(formals, actuals, strict=True) if value is not None},
             "scalar element original argument inventory differs")
    mappings = call["resource_mappings"]
    _require(isinstance(mappings, list) and len(mappings) == len(formals), "scalar element mapping inventory differs")
    elements, used, resources = [], set(), {}
    for formal, actual, row in zip(formals, actuals, mappings, strict=True):
        _keys(row, _MAPPING | ({"scalar_element"} if row.get("storage") == "scalar_element" else set()))
        _require(row["formal"] == formal and row["formal_resource"] == 'argument::' + formal
                 and row["actual"] == actual and isinstance(row["formal_descriptor"], dict)
                 and row["formal_descriptor"].get("name") == formal
                 and row["formal_descriptor"].get("resource") == row["formal_resource"],
                 "scalar element formal descriptor or original actual differs")
        _keys(row["requirements"], _REQUIREMENTS)
        _require(all(type(value) is bool for value in row["requirements"].values()),
                 "scalar element association requirements differ")
        if row["resource"] is not None:
            _require(_text(row["resource"]), "scalar element source resource differs")
            resources[row["formal_resource"]] = row["resource"]
        if row["storage"] == "scalar_element":
            _require(row["resource"] is None and row["presence"] == "supplied" and row["section"] is None
                     and all(value is False for value in row["requirements"].values()),
                     "scalar element acquired whole-array or optional association authority")
            used.add(_element(row["scalar_element"], row, reservations))
            elements.append(row["scalar_element"])
    _require(call["resource_mapping"] == resources and elements
             and not used.intersection(resources.values()), "scalar element backing aliases an ordinary actual")
    return elements, used


def validate_scalar_element_reservations(scope, plan, manifest, epoch, members, coordinators):
    """Return validated canonical triples and exact descendant reservation uses."""
    bindings = scope.get("resource_bindings")
    rows = bindings.get("resources") if isinstance(bindings, dict) else None
    has_annotations = isinstance(rows, list) and any(isinstance(row, dict) and "role" in row for row in rows)
    procedures = {p["procedure"]: p for p in plan["procedures"]}
    has_proofs = any("original_scalar_element_coordinates" in call
                     for procedure in procedures.values() for call in procedure["calls"])
    if not has_annotations and not has_proofs:
        return rows, {}
    _keys(bindings, {"schema_version", "identity", "resources"})
    _require(type(bindings["schema_version"]) is int and bindings["schema_version"] == 1
             and bindings["identity"] == "invocation-local canonical registration; matches runtime buffer trace identity"
             and isinstance(rows, list) and len(rows) == len(plan["resources"]),
             "scalar element canonical resource inventory differs")
    canonical, reservations = [], {}
    for row, original in zip(rows, plan["resources"], strict=True):
        annotated = isinstance(row, dict) and "role" in row
        _keys(row, _BASE | (_ROLE if annotated else set()))
        base = {key: row[key] for key in _BASE}
        _require(_digest(base) == _digest(original), "scalar element canonical registration/descriptor differs")
        canonical.append(original)
        if annotated:
            _require(row["role"] == "opaque_scalar_element_alias_reservation"
                     and row["managed_payload"] is False and row["definition_authority"] is False,
                     "scalar element opaque reservation acquired payload authority")
            reservations[row["resource"]] = original
    _require(bool(reservations), "scalar element coordinates lack an opaque reservation")
    epoch_resources = {row["resource"]: row for row in epoch["resources"]}
    for root in reservations:
        row = epoch_resources[root]
        _require(row["device_replica_authorized"] is False and row["initialization_authorized"] is False
                 and row["storage"] == "opaque host reservation" and row["lifetime"] == "original checked native access"
                 and not any(root in unit["resources"].values() for p in procedures.values() for unit in p["units"]),
                 "scalar element backing conflicts with managed payload storage")
    by_member = {member["procedure"]: member for member in members}
    uses, reached, coordinate_uses = {}, set(), set()
    for callee, procedure in procedures.items():
        for site in procedure["calls"]:
            if "original_scalar_element_coordinates" not in site:
                continue
            _keys(site, {"caller", "first_line", "last_line", "resource_mappings", "handle_bindings",
                         "mapped_call_node_id", "coordinate_proof_identity", "original_scalar_element_coordinates"})
            proof = site["original_scalar_element_coordinates"]
            _keys(proof, _PROOF)
            caller = site.get("caller")
            _require(caller in procedures and proof["caller"] == caller and proof["callee"] == callee
                     and type(proof["schema_version"]) is int and proof["schema_version"] == 3
                     and proof["kind"] == "original_native_call_coordinates"
                     and _sha256(proof["proof_identity"])
                     and proof["proof_identity"] == _digest({k: v for k, v in proof.items() if k != "proof_identity"})
                     and site.get("coordinate_proof_identity") == proof["proof_identity"]
                     and proof["structured_identity"] == procedures[caller]["structured_summary_identity"]
                     and _sha256(proof["analysis_identity"]) and proof["source_identity"] == epoch["source_inputs"]
                     and isinstance(proof["guard"], list) and all(_text(value) for value in proof["guard"])
                     and all(proof[key] is False for key in _PROOF_FALSE)
                     and all(proof[key] is True for key in _PROOF_TRUE),
                     "scalar element exact original coordinate proof differs")
            siblings = [row for row in procedure["calls"] if row.get("caller") == caller]
            routes = [row for row in by_member[caller]["routes"]
                      if row.get("kind") == "checked_child" and row.get("procedure") == callee]
            _require(len(routes) == len(siblings)
                     and all(type(row.get("first_line")) is int and type(row.get("last_line")) is int
                             and 1 <= row["first_line"] <= row["last_line"] for row in siblings)
                     and all(left["last_line"] < right["first_line"]
                             for left, right in zip(siblings, siblings[1:], strict=False))
                     and (caller, proof["node_id"]) not in coordinate_uses,
                     "scalar element repeats or reorders an original CALL use")
            route = routes[siblings.index(site)]
            _require(site in coordinators[callee].get("calls", [])
                     and route.get("selected_node_ids") == [proof["node_id"]]
                     and route.get("handle_forwarding") == site.get("handle_bindings"),
                     "scalar element coordinate use differs from its exact checked CALL")
            coordinate_uses.add((caller, proof["node_id"]))
            _require(any(_digest(proof["call"]) == _digest(call) for call in manifest.get("resolved_calls", [])),
                     "scalar element source CALL differs from the original resolved inventory")
            elements, roots = _call(proof["call"], callee, reservations)
            _require(proof["original_scalar_elements"] == elements,
                     "scalar element coordinates differ from original selector/declaration metadata")
            closure = proof["scalar_element_closure"]
            _keys(closure, {"version", "procedures", "source_work", "operations", "backing_resources",
                            "no_other_backing_reference", "only_scalar_in_forwarding", "effect_authority",
                            "definition_authority", "completion_authority"})
            _require(type(closure["version"]) is int and closure["version"] == 2
                     and closure["backing_resources"] == sorted(roots)
                     and closure["no_other_backing_reference"] is True and closure["only_scalar_in_forwarding"] is True
                     and all(closure[key] is False for key in ("effect_authority", "definition_authority", "completion_authority"))
                     and isinstance(closure["procedures"], list) and bool(closure["procedures"]),
                     "scalar element readonly backing closure differs")
            for row in closure["procedures"]:
                _keys(row, {"procedure", "associated_formals", "source", "source_sha256"})
                _require(_text(row["procedure"]) and isinstance(row["associated_formals"], list)
                         and all(_text(value) and value.startswith('argument::') for value in row["associated_formals"])
                         and row["source"] in plan["sources"] and row["source_sha256"] == plan["sources"][row["source"]],
                         "scalar element closure source/formal inventory differs")
            root_closure = closure["procedures"][0]
            expected_formals = sorted(row["formal_resource"] for row in proof["call"]["resource_mappings"]
                                      if row["storage"] == "scalar_element")
            _require(root_closure["procedure"] == callee and root_closure["associated_formals"] == expected_formals,
                     "scalar element closure does not retain its original readonly formals")
            for value in (proof["source_work"], closure["source_work"], closure["operations"]):
                _require(type(value) is int and value > 0, "scalar element source-work accounting differs")
            _require(isinstance(proof["coordinate_associations"], list)
                     and isinstance(proof["scalar_mapping"], dict) and isinstance(proof["scalar_bounds"], dict)
                     and isinstance(proof["original_scalar_components"], list)
                     and isinstance(proof["entry_definition_changes"], list)
                     and isinstance(proof["unmanaged_immutable_associations"], list)
                     and proof["requires_immutable_array_input_for_immutable_associations"] is bool(proof["unmanaged_immutable_associations"])
                     and isinstance(proof["limits"], list) and all(_text(value) for value in proof["limits"]),
                     "scalar element coordinate diagnostic inventory differs")
            uses.setdefault(caller, set()).update(roots)
            reached.update(roots)
    _require(reached == set(reservations), "scalar element opaque reservation lacks an original scalar CALL use")
    # Globals preserve their exact module identity through checked descendants.
    # Existing return-route validation separately checks every edge and slot.
    for _ in procedures:
        changed = False
        for name, member in by_member.items():
            for route in member["routes"]:
                if route.get("kind") not in {"checked_child", "checked_lexical_child"}:
                    continue
                child_uses = uses.get(route["procedure"], set())
                if route["kind"] == "checked_lexical_child":
                    relation = coordinators.get(route["procedure"], {}).get("lexical_namespace")
                    _require(isinstance(relation, dict) and relation.get("parent") == name
                             and any(row.get("source_call") == route.get("source_call")
                                     and row.get("selected_node_ids") == route.get("selected_node_ids")
                                     for row in relation.get("source_calls", [])),
                             "scalar element lexical forwarding lost its exact original parent CALL")
                    forwarded = child_uses & set(reservations)
                else:
                    forwarded = {root for root, canonical_root in route["canonical_resources"].items()
                                 if root == canonical_root and root in child_uses}
                before = len(uses.get(name, set()))
                uses.setdefault(name, set()).update(forwarded)
                changed |= before != len(uses[name])
        if not changed:
            break
    return canonical, uses
