"""Closed transport of issued atomic teams; no reconstructed source authority.

The compiler owns source participation, numerical preparation and physical/read
proof issuance. These checks authenticate their public linkage and compiled ABI;
JSON checksums are not an independent Fortran or numerical proof.
"""
from __future__ import annotations

import re

from .checked_execution_plan import _digest, _require, _sha256, _text

_TASK = {"identity", "version", "kind", "source_proof", "alternatives", "execution_binding", "definitions_authorized", "placement_authorized", "motion_authorized", "pricing"}
_TEAM = {"identity", "version", "kind", "procedure", "source_graph", "source_inputs", "selected_node_ids", "parent_completion", "units", "predicates", "predicate_read_sites", "tree", "alternatives", "private_resources", "analysis_limits", "bindings", "physical_alias_requirement", "completion", "preparation", "fallback", "execution_authorized", "gpu_legality_established", "definitions_authorized", "capture_authorized", "alias_authorized", "motion_authorized"}
_ALTERNATIVE = {"proof_identity", "schema_version", "proof_role", "team_identity", "alternative", "selected_original_units", "parent_completion_identity", "available", "has_openmp_in_closure", "caller_contract", "join", "private_resources", "standalone_execution_authority", "individual_unit_completion_authorized", "native_effects_authority", "gpu_legality_established"}
_PREP = {"schema_version", "proof_role", "preparation_identity", "provider_identity", "procedure", "selected_node_ids", "source_identity", "extraction_source_identity", "preconditions", "scalar_changes", "completion", "physical_analysis_available", "physical_analysis_reason", "physical_backend", "all_runtime_guards_typed", "original_storage_effects", "requires_original_team_environment_preflight", "capture_authorized", "artifacts_emitted", "execution_authorized", "definition_transfer"}
_EFFECT = {"proof_identity", "schema_version", "proof_role", "procedure", "analysis_identity", "structured_identity", "source_identity", "selected_node_ids", "preparation_identity", "sections", "units", "steps", "resources", "source_demands", "entry_definition_changes", "scalar_changes", "ordered_effect_count", "preconditions", "physical_visits", "rectangle_limit", "transfer_scope", "joined_reads", "input_definitions_authorized", "execution_authorized", "native_hook_authorized", "gpu_legality_established", "descriptor_lifetime_authorized", "alias_authorized", "unconditional_return_definitions_authorized", "bounds_evaluated"}
_READ = {"proof_identity", "schema_version", "proof_role", "procedure", "source_identity", "preparation_identity", "numerical_effect_identity", "units", "sections", "preconditions", "physical_visits", "validity", "joined_reads", "prefix_definitions_authorized", "capture_authorized", "descriptor_lifetime_authorized", "alias_authorized", "execution_authorized", "bounds_evaluated"}
_PROVIDER = {"version", "kind", "team_identity", "alternative_identity", "preparation_identity", "physical_effect_identity", "source_read_identity", "entry", "module", "cuda_sha256", "fortran_sha256", "units", "fields", "parameters", "completion", "preparation", "fallback", "definitions_authorized", "placement_authorized", "pricing"}
_IDENT = re.compile(r"[a-z][a-z0-9_]{0,62}\Z")


def _keys(record, expected):
    _require(isinstance(record, dict) and set(record) == expected, "atomic source task requires its exact schema")


def _sealed(record, fields, *, identity="identity"):
    _keys(record, fields)
    _require(_sha256(record[identity]) and record[identity] == _digest({k: v for k, v in record.items() if k != identity}),
             "atomic source task content identity differs")


def _false(record, keys):
    _require(all(record[key] is False for key in keys), "atomic source task cannot grant new authority")


def _predicate_sites(team, manifest):
    """Compare the uniform scanner's sites with the retained source graph."""
    sites, predicates = team["predicate_read_sites"], team["predicates"]
    _require(isinstance(sites, list) and len(sites) == len(predicates),
             "atomic predicate original header inventory differs")
    seen = set()
    for site, predicate in zip(sites, predicates, strict=True):
        _keys(site, {"source_node_id", "original_source", "expression", "host_point_reads"})
        identity = site["source_node_id"]
        _require(_text(identity) and identity not in seen and _text(site["expression"])
                 and any(identity.startswith(node + "/") for node in team["selected_node_ids"])
                 and site["original_source"] in {"IF (" + site["expression"] + ") THEN",
                                                  "ELSE IF (" + site["expression"] + ") THEN"}
                 and all(site[key] == predicate.get(key) for key in ("source_node_id", "original_source", "host_point_reads")),
                 "atomic predicate does not match its exact parent uniform header")
        points = site["host_point_reads"]
        _require(isinstance(points, list) and len({_digest(point) for point in points}) == len(points),
                 "atomic predicate duplicates an original point read")
        # Compare only spellings already issued by the parent scanner. This is
        # transport linkage to its original expression, not Fortran analysis.
        expected = []
        for point in team["parent_completion"].get("uniform_array_reads", []):
            match = re.search(r"(?<![A-Za-z0-9_])" + re.escape(point["source_reference"])
                              + r"(?![A-Za-z0-9_])", site["expression"])
            projection = {key: point[key] for key in ("resource", "source_reference", "subscript",
                "source_subscript", "parameter_resource", "original_lower_bound", "original_upper_bound")}
            if match and not any(projection == item[1] for item in expected):
                expected.append((match.start(), projection))
        _require(points == [point for _position, point in sorted(expected, key=lambda item: item[0])],
                 "atomic predicate points differ from its original expression spellings")
        seen.add(identity)
    structured = manifest.get("native_effects", {}).get("structured_effects", {})
    if isinstance(structured, dict) and structured.get("procedure") == team["procedure"]:
        _require(structured.get("structured_identity") == team["source_graph"]
                 and isinstance(structured.get("nodes"), list), "atomic predicate original graph differs")
        records = {node.get("id"): node for node in structured["nodes"] if isinstance(node, dict)}
        units = [unit["source_node_id"] for unit in team["units"]]
        conditions = [node["id"] for node in records.values() if node.get("evaluation") == "condition"
                      and any(node["id"].startswith(selected + "/") for selected in team["selected_node_ids"])
                      and not any(node["id"].startswith(unit + "/") for unit in units)]
        _require(conditions == [site["source_node_id"] for site in sites],
                 "atomic predicate order differs from the original source graph")
        for site in sites:
            bodies = [records.get(branch.get("body"), {}) for node in records.values()
                      for branch in node.get("alternatives", []) if branch.get("condition") == site["source_node_id"]]
            _require(len(bodies) == 1 and isinstance(bodies[0].get("guard"), list)
                     and bodies[0]["guard"] and bodies[0]["guard"][-1] == site["expression"],
                     "atomic predicate expression differs from its original branch guard")


def _sections(record, descriptors):
    """Check typed transport, without re-deriving any original rectangles."""
    _keys(record, {"schema_version", "available", "reason", "coordinate_system", "rectangle_limit", "resources"})
    _require(type(record["schema_version"]) is int and record["schema_version"] == 2
             and record["available"] is True and record["reason"] is None
             and record["coordinate_system"] == "original logical bounds; checked zero-based full-storage half-open mapping"
             and type(record["rectangle_limit"]) is int and record["rectangle_limit"] == 32
             and isinstance(record["resources"], list), "atomic source physical section schema differs")
    seen = set()
    required = {"resource", "rank", "logical_lower_bounds", "empty_dimension_lower_bound", "dependencies", "reads", "writes", "overwrites"}
    optional = {"guarded_explicit_dummy_extents", "original_allocation_lower_bounds", "original_dummy_lower_bounds", "logical_lower_bound_expressions"}
    for resource in record["resources"]:
        _require(isinstance(resource, dict) and required <= set(resource) <= required | optional
                 and _text(resource["resource"]) and resource["resource"] in descriptors
                 and resource["resource"] not in seen and type(resource["rank"]) is int
                 and resource["rank"] == descriptors[resource["resource"]]["rank"]
                 and resource["rank"] > 0 and isinstance(resource["logical_lower_bounds"], list)
                 and len(resource["logical_lower_bounds"]) == resource["rank"]
                 and all(value is None or type(value) is int for value in resource["logical_lower_bounds"])
                 and type(resource["empty_dimension_lower_bound"]) is int and resource["empty_dimension_lower_bound"] == 1,
                 "atomic source physical resource descriptor differs")
        seen.add(resource["resource"])
        for name in ("original_allocation_lower_bounds", "original_dummy_lower_bounds"):
            _require(name not in resource or resource[name] is True, "atomic source physical origin flag differs")
        for label in ("reads", "writes", "overwrites"):
            _require(isinstance(resource[label], list) and len(resource[label]) <= 32, "atomic source physical rectangle inventory differs")
            for box in resource[label]:
                _require(isinstance(box, dict) and {"axes", "source_access"} <= set(box) <= {"axes", "source_access", "iteration_guards"}
                         and isinstance(box["axes"], list) and len(box["axes"]) == resource["rank"]
                         and _text(box["source_access"]), "atomic source physical rectangle differs")
                for axis in box["axes"]:
                    _keys(axis, {"kind", "lower", "upper"})
                    _require(isinstance(axis["kind"], str) and axis["kind"] in {"point", "unit_stride_range"}
                             and isinstance(axis["lower"], dict) and isinstance(axis["upper"], dict),
                             "atomic source physical axis differs")
                if "iteration_guards" in box:
                    _require(isinstance(box["iteration_guards"], list), "atomic source physical iteration guard differs")
                    for guard in box["iteration_guards"]:
                        _keys(guard, {"lower", "upper", "step"})
                        _require(isinstance(guard["lower"], dict) and isinstance(guard["upper"], dict)
                                 and type(guard["step"]) is int and guard["step"] in {-1, 1},
                                 "atomic source physical iteration direction differs")
    return seen


def _descriptor(value):
    from .source_task_record import _binding

    # Logical source predicates are absent from the numerical ABI. Reuse the
    # structural binding check without changing the authenticated descriptor.
    if isinstance(value, dict) and value.get("type") == "logical":
        _require(type(value.get("kind")) is int and value["kind"] in {1, 4} and value.get("rank") == 0,
                 "atomic source predicate type differs")
        _binding({**value, "type": "integer", "kind": 4})
    else:
        _binding(value)


def _tree_paths(tree, units, predicates):
    _require(isinstance(tree, list), "atomic source task requires its complete original branch tree")
    paths = [[]]
    for item in tree:
        _require(isinstance(item, dict), "atomic source branch node differs")
        if set(item) == {"unit"}:
            _require(type(item["unit"]) is int and 0 <= item["unit"] < len(units), "atomic source unit index differs")
            paths = [[*path, item["unit"]] for path in paths]
            continue
        _keys(item, {"branches"})
        branches = item["branches"]
        _require(isinstance(branches, list) and len(branches) >= 2, "atomic source IF requires its complete alternatives")
        choices = []
        for position, branch in enumerate(branches):
            _keys(branch, {"predicate", "body"})
            predicate = branch["predicate"]
            _require((predicate is None and position == len(branches) - 1)
                     or (type(predicate) is int and 0 <= predicate < len(predicates)),
                     "atomic source predicate index differs")
            choices.extend(_tree_paths(branch["body"], units, predicates))
        _require(branches[-1]["predicate"] is None, "atomic source IF lost its original else/empty alternative")
        paths = [[*prefix, *suffix] for prefix in paths for suffix in choices]
    return paths


def _numerical(variant, team, index, descriptors):
    conditional = team["version"] == 3
    _keys(variant, {"source", "preparation", "physical_effects", "source_reads", "provider"})
    source, prep, effects, reads = (variant[k] for k in ("source", "preparation", "physical_effects", "source_reads"))
    _sealed(source, _ALTERNATIVE, identity="proof_identity")
    _require(type(source["schema_version"]) is int and source["schema_version"] == 1
             and source["proof_role"] == "atomic_original_team_alternative"
             and type(source["alternative"]) is int and source["alternative"] == index
             and source["team_identity"] == team["identity"]
             and source["selected_original_units"] == team["alternatives"][index]
             and source["parent_completion_identity"] == team["parent_completion"]["proof_identity"]
             and source["private_resources"] == team["private_resources"]
             and source["available"] is source["has_openmp_in_closure"] is True
             and source["join"] == "original parent PARALLEL completion only"
             and source["caller_contract"] == "selected original conditional path inside one atomic whole-team replacement",
             "atomic source alternative or parent join linkage differs")
    _false(source, ("standalone_execution_authority", "individual_unit_completion_authorized", "native_effects_authority", "gpu_legality_established"))
    _keys(prep, _PREP | ({"conditional_numerical_context"} if conditional else set()))
    _require(type(prep["schema_version"]) is int and prep["schema_version"] == 2
             and prep["proof_role"] == "prepared_original_numerical_region" and prep["completion"] == source
             and all(_sha256(prep[key]) for key in ("preparation_identity", "provider_identity", "extraction_source_identity", "source_identity"))
             and prep["physical_analysis_available"] is prep["all_runtime_guards_typed"] is True
             and prep["physical_analysis_reason"] is None and prep["physical_backend"] == "scoped_pointer_abi"
             and prep["scalar_changes"] == []
             and prep["definition_transfer"] == "conditional on all original typed preconditions; separate effects proof required",
             "atomic source numerical preparation differs")
    _false(prep, ("requires_original_team_environment_preflight", "capture_authorized", "artifacts_emitted", "execution_authorized"))
    _sealed(effects, _EFFECT | ({"conditional_numerical_context_identity"} if conditional else set()), identity="proof_identity")
    _sealed(reads, _READ | ({"conditional_numerical_context_identity", "read_admission_requires_same_conditional_guard"} if conditional else set()), identity="proof_identity")
    for proof, role in ((effects, "prepared_numerical_physical_effects"), (reads, "exact_original_reached_read_preconditions")):
        _require(type(proof["schema_version"]) is int and proof["schema_version"] == 1 and proof["proof_role"] == role
                 and proof["preparation_identity"] == prep["preparation_identity"]
                 and proof["procedure"] == team["procedure"] and proof["preconditions"] == prep["preconditions"]
                 and type(proof["physical_visits"]) is int and proof["physical_visits"] > 0,
                 "atomic source physical/read proof linkage differs")
    _require(prep["procedure"] == team["procedure"] and prep["selected_node_ids"] == team["selected_node_ids"]
             and effects["selected_node_ids"] == team["selected_node_ids"]
             and effects["structured_identity"] == team["source_graph"]
             and reads["source_identity"] == effects["source_identity"]
             and reads["numerical_effect_identity"] == effects["proof_identity"]
             and effects["source_demands"] == [prep["preparation_identity"]]
             and effects["scalar_changes"] == []
             and type(effects["rectangle_limit"]) is int and effects["rectangle_limit"] == 32,
             "atomic source preparation, source or geometry identity differs")
    _false(effects, ("input_definitions_authorized", "execution_authorized", "native_hook_authorized", "gpu_legality_established", "descriptor_lifetime_authorized", "alias_authorized", "unconditional_return_definitions_authorized", "bounds_evaluated"))
    _false(reads, ("prefix_definitions_authorized", "capture_authorized", "descriptor_lifetime_authorized", "alias_authorized", "execution_authorized", "bounds_evaluated"))
    count = len(source["selected_original_units"])
    for proof in (effects, reads):
        _require(isinstance(proof["units"], list) and [unit.get("unit_index") for unit in proof["units"] if isinstance(unit, dict)] == list(range(count))
                 and all(isinstance(unit, dict) and set(unit) == {"unit_index", "sections"} for unit in proof["units"]),
                 "atomic source lost a reached numerical unit")
        _sections(proof["sections"], descriptors)
        for unit in proof["units"]:
            _sections(unit["sections"], descriptors)
    _require(isinstance(effects["resources"], list) and bool(effects["resources"]), "atomic source physical inventory differs")
    for resource in effects["resources"]:
        _require(isinstance(resource, dict) and _text(resource.get("resource"))
                 and resource["resource"] in descriptors and type(resource.get("rank")) is int
                 and resource["rank"] == descriptors[resource["resource"]]["rank"],
                 "atomic source numerical resource differs")
    if conditional:
        from .conditional_affine_record import validate_context
        context = validate_context(prep["conditional_numerical_context"], team, prep, descriptors)
        _require(effects["conditional_numerical_context_identity"] == reads["conditional_numerical_context_identity"] == context["identity"]
                 and reads["read_admission_requires_same_conditional_guard"] is True,
                 "conditional SECTION read/effect guard linkage differs")
    return prep, effects, reads


def validate_atomic_source_task(record, member, procedure, inputs, resources, *, owner, manifest, graph_execution, task_ordinal):
    _sealed(record, _TASK)
    _require(type(record["version"]) is int and record["version"] in {2, 3, 4}
             and record["kind"] == "original_atomic_conditional_joined_team"
             and record["pricing"] == "unpriced; explicit sections only", "atomic source task version or policy differs")
    _false(record, ("definitions_authorized", "placement_authorized", "motion_authorized"))
    team = record["source_proof"]
    projected = record["version"] == 4
    _require(isinstance(team, dict) and type(team.get("version")) is int and team["version"] in {2, 3},
             "atomic source task original team version differs")
    conditional = team["version"] == 3
    _sealed(team, _TEAM | ({"conditional_affine_context_required"} if conditional else set()))
    _require((projected or team["version"] == record["version"]) and team["kind"] == "atomic_original_conditional_joined_team"
             and team["procedure"] == member["procedure"] == procedure["procedure"]
             and team["source_graph"] == member["source_graph"] and team["source_inputs"] == inputs,
             "atomic source task original authority linkage differs")
    _false(team, ("execution_authorized", "gpu_legality_established", "definitions_authorized", "capture_authorized", "alias_authorized", "motion_authorized"))
    _require(team["completion"] == "whole parent PARALLEL join only; no independent NOWAIT unit completion"
             and team["preparation"] == "reached original predicates admit only missing exact read regions before host acquisition; existing device replicas retained; no writes or whole-entry definitions; entire reached alternative before first device submission"
             and team["fallback"] == "complete original team once before submission; fatal after any attempt; no individual native replay"
             and team["physical_alias_requirement"] == "every distinct writable/read root must be checked disjoint at reached preparation",
             "atomic source task changed its completion/preparation/replay contract")
    nodes = team["selected_node_ids"]
    _require(isinstance(nodes, list) and bool(nodes) and all(_text(n) and n.startswith(team["procedure"] + "/") for n in nodes)
             and len(set(nodes)) == len(nodes), "atomic source task lost its original complete selection")
    parent = team["parent_completion"]
    uniform_keys = {"uniform_array_reads", "uniform_array_read_contract", "guarded_fixed_bound_array_conditions_authorized",
                    "guarded_assumed_shape_array_conditions_authorized", "gpu_independence_established"}
    _keys(parent, {"available", "caller_contract", "executable_original_nodes", "has_opaque_calls_in_closure", "has_openmp_in_closure", "join", "private_resources", "procedure", "proof_identity", "reason", "requires_serial_caller", "retains_original_team_and_directives", "schema_version", "selected_original_nodes", "structured_identity"}
          | (uniform_keys if "uniform_array_reads" in parent else set())
          | ({"native_only", "native_sections_contract", "original_section_count", "section_independence_established", "internal_cuts_authorized", "gpu_legality_established"} if conditional else set()))
    uniform = parent.get("uniform_array_reads", [])
    if uniform:
        _require(isinstance(uniform, list) and parent["uniform_array_read_contract"] == "fixed-rank-one-shared-constant-point-v1"
                 and parent["guarded_assumed_shape_array_conditions_authorized"] is False
                 and parent["gpu_independence_established"] is False
                 and type(parent["guarded_fixed_bound_array_conditions_authorized"]) is bool,
                 "atomic predicate parent point contract differs")
    else:
        _require(not (set(parent) & uniform_keys), "atomic predicate parent point inventory is empty")
    _require(type(parent["schema_version"]) is int and parent["schema_version"] == 7 and _sha256(parent["proof_identity"])
             and parent["procedure"] == team["procedure"] and parent["structured_identity"] == team["source_graph"]
             and parent["selected_original_nodes"] == nodes and parent["private_resources"] == team["private_resources"]
             and parent["available"] is parent["has_openmp_in_closure"] is parent["requires_serial_caller"] is parent["retains_original_team_and_directives"] is True
             and parent["has_opaque_calls_in_closure"] is False and parent["caller_contract"] == "serial_source_scope"
             and parent["join"] == "explicit original parallel end", "atomic source task requires the complete parent join")
    bindings = team["bindings"]
    _require(isinstance(bindings, list) and bool(bindings), "atomic source task requires original typed bindings")
    for descriptor in bindings:
        _descriptor(descriptor)
    descriptors = {item["resource"]: item for item in bindings}
    _require(len(descriptors) == len(bindings), "atomic source task repeats an original binding")
    point_keys = {"resource", "source_reference", "subscript", "source_subscript", "parameter_resource",
                  "original_lower_bound", "original_upper_bound"}
    for point in uniform:
        _keys(point, point_keys | {"type", "kind", "rank", "guarded_condition", "compound_logical_condition",
            "shared_and_unwritten_in_complete_team", "requires_registered_storage_and_alias_validation",
            "requires_host_coherence_before_original_team", "condition_evaluation", "storage"})
        descriptor = descriptors.get(point["resource"])
        _require(descriptor is not None and (descriptor["type"], descriptor["kind"], descriptor["rank"]) == ("integer", 4, 1)
                 and (point["type"], point["kind"], point["rank"]) == ("integer", 4, 1)
                 and point["storage"] == "fixed_explicit_shape"
                 and all(type(point[key]) is int for key in ("subscript", "original_lower_bound", "original_upper_bound"))
                 and point["original_lower_bound"] <= point["subscript"] <= point["original_upper_bound"]
                 and _text(point["source_reference"]) and _text(point["source_subscript"])
                 and (point["parameter_resource"] is None or _text(point["parameter_resource"]))
                 and all(type(point[key]) is bool for key in ("guarded_condition", "compound_logical_condition"))
                 and all(point[key] is True for key in ("shared_and_unwritten_in_complete_team",
                     "requires_registered_storage_and_alias_validation", "requires_host_coherence_before_original_team"))
                 and point["condition_evaluation"] == "unchanged original team condition; no proof-time payload read",
                 "atomic predicate parent point binding differs")
    if uniform:
        _require(parent["guarded_fixed_bound_array_conditions_authorized"] is any(point["guarded_condition"] for point in uniform),
                 "atomic predicate parent guard inventory differs")
    units, predicates = team["units"], team["predicates"]
    _require(isinstance(units, list) and bool(units) and isinstance(predicates, list) and bool(predicates), "atomic source task requires original loops and conditional predicates")
    arrays, writes = set(), set()
    for unit in units:
        _keys(unit, {"source_node_id", "reads", "writes", "nowait"} | ({"unit_kind", "section_directive_node_id", "section_group_node_id"} if conditional else set()))
        _require(_text(unit["source_node_id"]) and unit["source_node_id"].startswith(team["procedure"] + "/")
                 and type(unit["nowait"]) is bool and isinstance(unit["reads"], list) and isinstance(unit["writes"], list)
                 and bool(unit["writes"]) and all(_text(root) and root in descriptors and descriptors[root]["rank"] > 0 for root in [*unit["reads"], *unit["writes"]]),
                 "atomic source unit storage identity differs")
        arrays.update([*unit["reads"], *unit["writes"]])
        writes.update(unit["writes"])
    if conditional:
        from .conditional_affine_record import section_inventory
        section_inventory(team)
    used_points = []
    for predicate in predicates:
        _keys(predicate, {"source_node_id", "original_source", "scalar_resources", "host_point_reads"})
        _require(_text(predicate["source_node_id"]) and _text(predicate["original_source"])
                 and isinstance(predicate["scalar_resources"], list)
                 and all(_text(root) and root in descriptors and descriptors[root]["rank"] == 0
                         and descriptors[root]["type"] in {"integer", "logical"} for root in predicate["scalar_resources"]),
                 "atomic source original predicate binding differs")
        points = predicate["host_point_reads"]
        _require(isinstance(points, list), "atomic predicate host point inventory differs")
        for point in points:
            _keys(point, point_keys)
            _require(any(point == {key: candidate[key] for key in point_keys} for candidate in uniform),
                     "atomic predicate point lacks its parent source binding")
            _require(point["resource"] not in writes, "atomic predicate point is written inside the original team")
            arrays.add(point["resource"])
            used_points.append(point)
    _require(all({key: point[key] for key in point_keys} in used_points for point in uniform),
             "atomic predicate omitted a parent point read")
    _predicate_sites(team, manifest)
    paths = _tree_paths(team["tree"], units, predicates)
    _require(paths == team["alternatives"] and {index for path in paths for index in path} == set(range(len(units))), "atomic source task lost an original path or tail")
    for path in paths:
        _require(len(set(path)) == len(path), "atomic source task repeats a reached worksharing unit")
        for position, index in enumerate(path) if not conditional else ():
            for other in path[:position]:
                _require(not (set(units[index]["writes"]) & {*units[other]["reads"], *units[other]["writes"]}
                              or set(units[other]["writes"]) & set(units[index]["reads"])), "atomic source cross-unit storage dependency differs")
    binding = record["execution_binding"]
    _keys(binding, {"owner", "canonical_resources", "registration_slots"}
          | ({"call_projection"} if projected else set()))
    _require(binding["owner"] == owner and isinstance(binding["canonical_resources"], dict) and isinstance(binding["registration_slots"], dict)
             and set(binding["canonical_resources"]) == set(binding["registration_slots"]) == arrays,
             "atomic source physical root inventory differs")
    projected_roots = None
    from .source_task_call_record import (
        require_single_association,
        validate_call_projection,
    )

    if projected:
        fields = [{"binding": descriptors[root], "written": root in writes} for root in sorted(arrays)]
        projected_roots = validate_call_projection(binding["call_projection"], record, fields, procedure,
                                                    resources, owner, manifest)
    else:
        require_single_association(procedure, owner, manifest)
    for root in arrays:
        physical = binding["canonical_resources"][root]
        _require(_text(physical) and physical in resources, "atomic source canonical physical root differs")
        if projected:
            _require(physical in projected_roots[root], "atomic source representative is outside its CALL projection")
        original = resources[physical]
        _require(type(binding["registration_slots"][root]) is int
                 and binding["registration_slots"][root] == original["registration_identity"]
                 and original["device_replica_authorized"] is True
                 and all(descriptors[root][key] == original["descriptor"][key] for key in ("type", "kind", "rank")),
                 "atomic source physical slot or typed descriptor differs")
    alternatives = record["alternatives"]
    indices = [i for i, path in enumerate(paths) if path]
    _require(isinstance(alternatives, list) and len(alternatives) == len(indices), "atomic source numerical alternative inventory differs")
    shared_context = None
    for variant, index in zip(alternatives, indices, strict=True):
        prep, effects, reads = _numerical(variant, team, index, descriptors)
        provider = variant["provider"]
        _keys(provider, {"entry", "module", "cuda_source_sha256", "fortran_source_sha256", "metadata"})
        metadata = provider["metadata"]
        from .conditional_affine_record import PROVIDER_KEYS, validate_provider
        _keys(metadata, _PROVIDER | (PROVIDER_KEYS if conditional else set()))
        if conditional:
            context = prep["conditional_numerical_context"]
            _require(shared_context is None or context == shared_context,
                     "conditional SECTION alternatives do not share the same complete context")
            shared_context = context
            validate_provider(metadata, context)
        _require(type(metadata["version"]) is int and metadata["version"] == team["version"] and metadata["kind"] == "atomic_original_team_alternative"
                 and metadata["team_identity"] == team["identity"] and metadata["alternative_identity"] == variant["source"]["proof_identity"]
                 and metadata["preparation_identity"] == prep["preparation_identity"]
                 and metadata["physical_effect_identity"] == effects["proof_identity"] and metadata["source_read_identity"] == reads["proof_identity"]
                 and type(metadata["units"]) is int and metadata["units"] == len(paths[index]), "atomic source provider proof linkage differs")
        _false(metadata, ("definitions_authorized", "placement_authorized"))
        _require(metadata["completion"] == "one transaction at the original whole parent join; supplied-stream source order"
                 and metadata["preparation"] == "all reached unit geometry and exact physical/source-read unions before submission"
                 and metadata["fallback"] == "original whole team only before first attempted launch; no per-unit CPU fallback"
                 and metadata["pricing"] == "unpriced; explicit sections only", "atomic source provider execution contract differs")
        for name, suffix, sha_key in (("entry", ".cu", "cuda"), ("module", ".f90", "fortran")):
            _require(isinstance(provider[name], str) and _IDENT.fullmatch(provider[name]) and metadata[name] == provider[name]
                     and _sha256(provider[sha_key + "_source_sha256"]) and metadata[sha_key + "_sha256"] == provider[sha_key + "_source_sha256"]
                     and manifest.get("artifacts_sha256", {}).get(provider[name] + suffix) == provider[sha_key + "_source_sha256"], "atomic source provider compiled artifact differs")
        fields = metadata["fields"]
        _require(isinstance(fields, list) and bool(fields), "atomic source provider requires typed field ABI")
        for field in fields:
            _descriptor(field)
            _require(field["resource"] in arrays and field == descriptors[field["resource"]], "atomic source provider field ABI differs")
        _require(len({field["resource"] for field in fields}) == len(fields)
                 and {field["resource"] for field in fields} == {r["resource"] for r in effects["resources"]}, "atomic source provider field coverage differs")
        parameters = metadata["parameters"]
        _require(isinstance(parameters, list), "atomic source provider scalar ABI differs")
        names = set()
        for parameter in parameters:
            _keys(parameter, {"binding", "name", "kind", "axis"})
            descriptor = parameter["binding"]
            _descriptor(descriptor)
            _require(descriptor["resource"] in descriptors and descriptor == descriptors[descriptor["resource"]]
                     and isinstance(parameter["name"], str) and _IDENT.fullmatch(parameter["name"]) and parameter["name"] not in names,
                     "atomic source provider parameter original binding differs")
            names.add(parameter["name"])
            axis = parameter["axis"]
            if axis is None:
                _require(descriptor["rank"] == 0 and parameter["kind"] == {("integer", 4): "c_int", ("real", 4): "c_float", ("real", 8): "c_double"}.get((descriptor["type"], descriptor["kind"])), "atomic source provider scalar type differs")
            else:
                _require(type(axis) is int and 1 <= axis <= descriptor["rank"] and parameter["kind"] == "c_int", "atomic source provider lower-bound recipe differs")
    _graph(team, binding, arrays, writes, graph_execution, task_ordinal)


def _graph(team, binding, arrays, writes, graph, ordinal):
    from .source_placement_record import _placements, _tasks

    _require(type(ordinal) is int and ordinal >= 0 and isinstance(graph, dict) and graph.get("source_order_executor") is True
             and isinstance(graph.get("procedures"), list), "atomic source task requires its complete source-order graph")
    procedures = [item for item in graph["procedures"] if isinstance(item, dict) and item.get("procedure") == team["procedure"]]
    _require(len(procedures) == 1, "atomic source task original graph procedure differs")
    task_id = team["procedure"] + ":source_device:" + str(ordinal)
    tasks = [item for item in _tasks(procedures[0].get("task_graph")) if item.get("id") == task_id]
    placements = [item for item in _placements(procedures[0].get("schedule")) if item.get("task") == task_id]
    _require(len(tasks) == len(placements) == 1, "atomic source task/schedule identity differs")
    task, placement = tasks[0], placements[0]
    projection = binding.get("call_projection")
    roots = (projection["conservative_effects"]["reads"] if projection is not None
             else sorted(binding["canonical_resources"][root] for root in arrays))
    write_roots = (projection["conservative_effects"]["may_writes"] if projection is not None
                   else sorted(binding["canonical_resources"][root] for root in writes))
    _require(task.get("implementations") == ["gpu"] and task.get("reads") == task.get("metadata_reads") == roots
             and task.get("may_writes") == write_roots
             and task.get("ordered") is True and task.get("opaque") is False
             and task.get("scalar_writes") == task.get("lifetime_resources") == []
             and task.get("scalar_reads") == sorted(item["resource"] for item in team["bindings"]
                 if item["rank"] == 0 and item["resource"] not in team["private_resources"])
             and placement.get("target") == "gpu" and "deferred_placement" not in task and "deferred_placement" not in placement,
             "atomic source task must retain its forced complete GPU alternative")
