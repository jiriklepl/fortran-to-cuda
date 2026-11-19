"""Closed linkage for original SECTION guards; no reconstructed ISL authority.

The compiler owns source ordering and the exact projected mathematical sets.
This reader checks their complete typed transport and source/provider linkage.
The internal model digest is opaque: JSON cannot independently issue a proof.
"""
from __future__ import annotations

from .checked_execution_plan import _digest, _require, _sha256, _text

_CONTEXT = {"identity", "version", "kind", "team_identity", "provider_identity", "procedure", "source_graph", "selected_node_ids", "source_identity", "source_inputs", "extraction_identity", "payload_identity", "parameters", "paths", "model_units", "source_work_budget", "isl_operation_budget", "guard", "incoming_reads", "original_integer_prework_required", "physical_bounds_and_alias_guards_required", "independent_section_completion_authorized", "definition_transfer_authorized", "execution_authorized", "source_read_admission_authorized"}
_DOMAIN = {"model_identity", "version", "kind", "parameters", "units", "guard_conjunction", "domain", "source_authority", "execution_authorized", "incoming_read_admission_authorized", "alias_authorized", "requires_original_integer_prework", "requires_exact_source_binding_and_guard_before_admission"}
PROVIDER_KEYS = {"conditional_numerical_context_identity", "conditional_guard_before_source_read_admission", "zero_trip_unit", "guard_arithmetic", "original_integer_prework", "generation_snapshot_checked"}
PARENT_KEYS = {"native_only", "native_sections_contract", "original_section_count", "section_independence_established", "internal_cuts_authorized", "gpu_legality_established"}
UNIT_KEYS = {"unit_kind", "section_directive_node_id", "section_group_node_id"}


def _keys(value, keys):
    _require(isinstance(value, dict) and set(value) == keys, "conditional SECTION exact schema differs")


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def section_inventory(team):
    parent, units = team["parent_completion"], team["units"]
    _require(team["conditional_affine_context_required"] is True
             and parent["native_only"] is True
             and parent["native_sections_contract"] == "original-explicit-counted-sections-v1"
             and type(parent["original_section_count"]) is int
             and parent["original_section_count"] == len(units)
             and all(parent[key] is False for key in ("section_independence_established", "internal_cuts_authorized", "gpu_legality_established")),
             "conditional SECTION cannot issue independent completion")
    loops, directives = set(), set()
    for unit in units:
        _require(unit["unit_kind"] == "section", "conditional SECTION unit kind differs")
        names = [unit[key] for key in ("source_node_id", "section_directive_node_id", "section_group_node_id")]
        _require(all(_text(name) and any(name.startswith(root + "/") for root in team["selected_node_ids"]) for name in names)
                 and len(set(names)) == 3 and names[0] not in loops and names[1] not in directives,
                 "conditional SECTION original inventory differs")
        loops.add(names[0])
        directives.add(names[1])
    _require(not loops.intersection(directives)
             and not {u["section_group_node_id"] for u in units}.intersection(loops | directives),
             "conditional SECTION marker roles overlap")


def _domain(domain, parameters, path):
    _keys(domain, _DOMAIN)
    expected = [{"model_name": "p" + str(p["symbol_id"]), "symbol_id": p["symbol_id"],
                 "name": p["name"], "dtype": "integer"} for p in parameters if p["rank"] == 0]
    _require(type(domain["version"]) is int and domain["version"] == 2
             and domain["kind"] == "unaccepted_conditional_affine_domain"
             and _sha256(domain["model_identity"])
             and domain["parameters"] == expected
             and type(domain["units"]) is int and domain["units"] == len(path)
             and domain["domain"] == "intersection of exact projected legal sets over original INTEGER4 inputs"
             and all(domain[key] is False for key in ("source_authority", "execution_authorized", "incoming_read_admission_authorized", "alias_authorized"))
             and domain["requires_original_integer_prework"] is domain["requires_exact_source_binding_and_guard_before_admission"] is True,
             "conditional SECTION model linkage or authority differs")
    for parameter in domain["parameters"]:
        _keys(parameter, {"model_name", "symbol_id", "name", "dtype"})
        _require(type(parameter["symbol_id"]) is int, "conditional SECTION model symbol is not an integer")
    clauses = domain["guard_conjunction"]
    _require(isinstance(clauses, list), "conditional SECTION guard conjunction differs")
    names, identities = {p["model_name"] for p in expected}, set()
    for clause in clauses:
        _keys(clause, {"identity", "legal_set", "forbidden_set", "constraints"})
        _require(_sha256(clause["identity"]) and clause["identity"] not in identities
                 and clause["identity"] == _digest({key: value for key, value in clause.items() if key != "identity"})
                 and _text(clause["legal_set"]) and _text(clause["forbidden_set"])
                 and isinstance(clause["constraints"], list), "conditional SECTION guard clause identity differs")
        identities.add(clause["identity"])
        for arm in clause["constraints"]:
            _require(isinstance(arm, list), "conditional SECTION guard alternative differs")
            for condition in arm:
                _keys(condition, {"constant", "coefficients", "equality"})
                _require(_integer(condition["constant"], -(2**63), 2**63-1)
                         and type(condition["equality"]) is bool and isinstance(condition["coefficients"], list),
                         "conditional SECTION checked guard coefficient differs")
                used = []
                for row in condition["coefficients"]:
                    _require(isinstance(row, list) and len(row) == 2 and _text(row[0]) and row[0] in names
                             and _integer(row[1], -(2**63), 2**63-1) and row[1] != 0,
                             "conditional SECTION guard references a foreign parameter")
                    used.append(row[0])
                _require(len(set(used)) == len(used), "conditional SECTION guard duplicates a parameter")


def validate_context(context, team, prep, descriptors):
    _keys(context, _CONTEXT)
    _require(_sha256(context["identity"])
             and context["identity"] == _digest({key: value for key, value in context.items() if key != "identity"})
             and type(context["version"]) is int and context["version"] == 1
             and context["kind"] == "original_conditional_affine_numerical_context"
             and context["team_identity"] == team["identity"]
             and context["provider_identity"] == prep["provider_identity"]
             and context["procedure"] == team["procedure"]
             and context["source_graph"] == team["source_graph"]
             and context["selected_node_ids"] == team["selected_node_ids"]
             and context["source_inputs"] == team["source_inputs"]
             and context["source_identity"] == prep["source_identity"]
             and all(_sha256(context[key]) for key in ("extraction_identity", "payload_identity"))
             and type(context["model_units"]) is int and context["model_units"] == len(team["units"])
             and type(context["source_work_budget"]) is int and context["source_work_budget"] == 65536
             and type(context["isl_operation_budget"]) is int and context["isl_operation_budget"] == 1000000,
             "conditional SECTION context source or policy differs")
    _require(context["guard"] == "same checked INTEGER64 legal domain before source-read admission and device submission"
             and context["incoming_reads"] == "perfect mapped nests; one unconditional assignment; no retained loops, intrinsics or conditional operands"
             and context["original_integer_prework_required"] is context["physical_bounds_and_alias_guards_required"] is True
             and all(context[key] is False for key in ("independent_section_completion_authorized", "definition_transfer_authorized", "execution_authorized", "source_read_admission_authorized")),
             "conditional SECTION context grants unsupported authority")
    parameters = context["parameters"]
    _require(isinstance(parameters, list) and bool(parameters), "conditional SECTION typed parameters missing")
    ids, names = set(), set()
    for parameter in parameters:
        _keys(parameter, {"symbol_id", "name", "dtype", "rank", "resource", "lower_bound_dimension", "runtime_lower_bound", "binding"})
        _require(_text(parameter["resource"]), "conditional SECTION resource is not a source identity")
        descriptor = descriptors.get(parameter["resource"])
        _require(descriptor is not None and parameter["binding"] == descriptor
                 and _integer(parameter["symbol_id"], 0, 2**31-1) and parameter["symbol_id"] not in ids
                 and _text(parameter["name"]) and parameter["name"] not in names
                 and type(parameter["rank"]) is int and parameter["rank"] >= 0,
                 "conditional SECTION original typed parameter differs")
        ids.add(parameter["symbol_id"])
        names.add(parameter["name"])
        axis = parameter["lower_bound_dimension"]
        if axis is not None:
            _require(_integer(axis, 1, descriptor["rank"]) and parameter["runtime_lower_bound"] is True
                     and parameter["rank"] == 0 and parameter["dtype"] == "integer",
                     "conditional SECTION lower-bound association differs")
        else:
            _require(parameter["runtime_lower_bound"] is False and parameter["rank"] == descriptor["rank"],
                     "conditional SECTION parameter rank differs")
            if parameter["rank"] == 0:
                _require((descriptor["type"], descriptor["kind"], parameter["dtype"]) == ("integer", 4, "integer"),
                         "conditional SECTION requires original INTEGER4 scalars")
            else:
                expected = {("integer", 4): "integer", ("real", 4): "real32", ("real", 8): "real"}
                _require(parameter["dtype"] == expected.get((descriptor["type"], descriptor["kind"])),
                         "conditional SECTION array element type differs")
    paths = context["paths"]
    _require(isinstance(paths, list) and len(paths) == len(team["alternatives"]),
             "conditional SECTION omitted an original path")
    for item, original in zip(paths, team["alternatives"], strict=True):
        _keys(item, {"original_units", "domain"})
        _require(isinstance(item["original_units"], list)
                 and all(type(index) is int for index in item["original_units"])
                 and item["original_units"] == original, "conditional SECTION path selection differs")
        if original:
            _domain(item["domain"], parameters, original)
        else:
            _require(item["domain"] is None, "conditional SECTION empty path cannot gain authority")
    return context


def validate_provider(metadata, context):
    _require(metadata["conditional_numerical_context_identity"] == context["identity"]
             and metadata["conditional_guard_before_source_read_admission"] is True
             and metadata["generation_snapshot_checked"] is True
             and metadata["zero_trip_unit"] == "decline whole original team before later bound evaluation"
             and metadata["guard_arithmetic"] == "checked INTEGER64; overflow declines before source-read admission"
             and metadata["original_integer_prework"] == "reused checked original INT4 ranges and terminal values",
             "conditional SECTION provider guard ordering differs")
    expected = [{"binding": p["binding"], "name": "fort_v" + str(p["symbol_id"]) + "_" + p["name"],
                 "kind": "c_int", "axis": p["lower_bound_dimension"]}
                for p in context["parameters"] if p["rank"] == 0]
    _require(metadata["parameters"] == expected
             and metadata["fields"] == [p["binding"] for p in context["parameters"] if p["rank"] > 0],
             "conditional SECTION provider changed its exact parameter snapshots")
