"""Transport original scalar CALL associations; never evaluate or specialize actuals."""
from .checked_execution_plan import _require, _sha256, _text
from .source_task_record import _IDENT, _binding, _keys

_FLAGS = {"value_specialization_authorized", "evaluation_motion_authorized",
          "array_payload_definition_authorized", "lifetime_authorized", "entry_fact_authorized"}
_REQUIREMENTS = {"descriptor_dependent_presence", "original_actual_presence", "original_allocation_descriptor",
                 "physical_section_mapping", "presence_preserving_forwarding"}


def _source_call(call, binding, procedure):
    _keys(call, {"schema_version", "procedure", "target", "source_kind", "formal_arguments", "actual_arguments",
                 "original_arguments", "resource_mapping", "resource_mappings", "actual_evaluation"})
    _require(type(call["schema_version"]) is int and call["schema_version"] == 2
             and call["procedure"] == procedure and _text(call["target"]) and _text(call["source_kind"])
             and call["actual_evaluation"] == "original call position and guards; no analysis-time evaluation",
             "scalar call input original call identity differs")
    formals, actuals = call["formal_arguments"], call["actual_arguments"]
    _require(isinstance(formals, list) and bool(formals)
             and all(isinstance(name, str) and _IDENT.fullmatch(name) for name in formals)
             and len(set(formals)) == len(formals) and binding["name"] in formals
             and isinstance(actuals, list) and len(actuals) == len(formals)
             and all(actual is None or _text(actual) for actual in actuals),
             "scalar call input formal/actual inventory differs")
    original = call["original_arguments"]
    _require(isinstance(original, list), "scalar call input original argument order differs")
    seen = set()
    for index, row in enumerate(original):
        _keys(row, {"position", "formal", "keyword", "actual"})
        _require(type(row["position"]) is int and row["position"] == index
                 and isinstance(row["formal"], str) and row["formal"] in formals and row["formal"] not in seen
                 and _text(row["actual"]) and row["actual"] == actuals[formals.index(row["formal"])]
                 and (row["keyword"] is None or isinstance(row["keyword"], str)
                      and row["keyword"].lower() == row["formal"]),
                 "scalar call input original argument association differs")
        seen.add(row["formal"])
    _require(seen == {formal for formal, actual in zip(formals, actuals, strict=True) if actual is not None},
             "scalar call input lost supplied original arguments")
    mappings = call["resource_mappings"]
    _require(isinstance(mappings, list) and len(mappings) == len(formals),
             "scalar call input source mappings differ")
    resources = {}
    for formal, actual, row in zip(formals, actuals, mappings, strict=True):
        _keys(row, {"formal", "formal_resource", "formal_descriptor", "actual", "actual_descriptor", "resource",
                    "storage", "presence", "section", "requirements"})
        _require(row["formal"] == formal and row["formal_resource"] == "argument::" + formal
                 and row["actual"] == actual and isinstance(row["formal_descriptor"], dict)
                 and row["formal_descriptor"].get("name") == formal
                 and row["formal_descriptor"].get("resource") == row["formal_resource"]
                 and (row["resource"] is None or _text(row["resource"]))
                 and _text(row["storage"]) and _text(row["presence"]),
                 "scalar call input source association differs")
        _keys(row["requirements"], _REQUIREMENTS)
        _require(all(type(value) is bool for value in row["requirements"].values()),
                 "scalar call input association requirements differ")
        if row["resource"] is not None:
            resources[row["formal_resource"]] = row["resource"]
        if formal != binding["name"]:
            continue
        _require(row["formal_descriptor"] == binding and row["presence"] == "supplied"
                 and row["section"] is None and row["storage"] in {"whole", "expression"}
                 and all(value is False for value in row["requirements"].values()),
                 "scalar call input requires the exact ordinary supplied scalar")
        descriptor = row["actual_descriptor"]
        if descriptor is None:
            _require(row["resource"] is None and row["storage"] == "expression",
                     "scalar call input expression acquired a storage identity")
        else:
            _binding(descriptor, rank=0)
            _require("parameter" in descriptor["attributes"] and descriptor["intent"] is None
                     and descriptor["type"] == binding["type"] and descriptor["kind"] == binding["kind"]
                     and descriptor["lower_bounds"] == descriptor["shape"] == []
                     and row["resource"] == descriptor["resource"] and row["storage"] == "whole",
                     "scalar call input named actual requires its original PARAMETER declaration")
    _require(call["resource_mapping"] == resources, "scalar call input resource association inventory differs")


def validate_scalar_call_input(authority, binding, prefix, procedure, manifest, *, call_projection=None):
    _keys(authority, {"kind", "version", "caller", "procedure", "source_inputs", "caller_graph", "formal",
                      "calls", "semantics"} | _FLAGS)
    _require(type(authority["version"]) is int and authority["version"] == 1
             and authority["kind"] == "original source-constant scalar call inputs"
             and authority["procedure"] == prefix["procedure"] == procedure["procedure"]
             and authority["source_inputs"] == prefix["source_inputs"] and _sha256(authority["caller_graph"])
             and _text(authority["caller"]) and authority["caller"] != authority["procedure"]
             and authority["formal"] == binding and binding["resource"] == "argument::" + binding["name"]
             and binding["intent"] == "in" and binding["rank"] == 0
             and binding["lower_bounds"] == binding["shape"] == []
             and not set(binding["attributes"]) & {"pointer", "optional", "volatile", "asynchronous", "allocatable", "value", "parameter"}
             and all(authority[key] is False for key in _FLAGS)
             and authority["semantics"] == "original actual evaluation and association at each unchanged guarded CALL",
             "scalar call input source, formal or authority differs")
    _require(isinstance(manifest, dict) and isinstance(manifest.get("scopes"), list)
             and isinstance(manifest.get("resolved_calls"), list),
             "scalar call input requires the containing checked call inventory")
    contexts = [scope["checked_execution_plan"] for scope in manifest["scopes"]
                if isinstance(scope, dict) and isinstance(scope.get("checked_execution_plan"), dict)
                and procedure in scope["checked_execution_plan"].get("procedures", [])]
    _require(bool(contexts) and all(any(p.get("procedure") == authority["caller"]
             and p.get("structured_summary_identity") == authority["caller_graph"] for p in context["procedures"])
             for context in contexts), "scalar call input caller graph differs from its checked scope")
    calls = authority["calls"]
    _require(isinstance(calls, list) and bool(calls), "scalar call input requires original call sites")
    previous, selections, by_span = 0, set(), {}
    for row in calls:
        _keys(row, {"selected_node_ids", "first_line", "last_line", "source_call"})
        nodes = row["selected_node_ids"]
        _require(isinstance(nodes, list) and bool(nodes)
                 and all(_text(node) and node.startswith(authority["caller"] + "/") for node in nodes)
                 and len(set(nodes)) == len(nodes) and not selections.intersection(nodes)
                 and type(row["first_line"]) is int and type(row["last_line"]) is int
                 and previous < row["first_line"] <= row["last_line"],
                 "scalar call input original site order or span differs")
        previous = row["last_line"]
        selections.update(nodes)
        _source_call(row["source_call"], binding, prefix["procedure"])
        by_span[(row["first_line"], row["last_line"])] = row["source_call"]
    emitted = procedure.get("calls")
    _require(isinstance(emitted, list) and bool(emitted), "scalar call input requires actual companion calls")
    for site in emitted:
        _require(isinstance(site, dict) and site.get("caller") == authority["caller"]
                 and type(site.get("first_line")) is int and type(site.get("last_line")) is int,
                 "scalar call input emitted caller differs")
        call = by_span.get((site["first_line"], site["last_line"]))
        _require(call is not None and call in manifest["resolved_calls"],
                 "scalar call input does not cover an emitted resolved call")
        direct = isinstance(site.get("resource_mappings"), dict) and all(
            call["resource_mapping"].get(root) == actual for root, actual in site["resource_mappings"].items())
        # The containing source-task validator subsequently validates the whole
        # projection and exact slot associations. This local check only binds
        # the scalar's original CALL to that same transport record.
        projected = isinstance(call_projection, dict) and isinstance(call_projection.get("calls"), list) and any(
            isinstance(edge, dict) and edge.get("callee") == procedure["procedure"]
            and all(edge.get(key) == site.get(key) for key in
                    ("caller", "first_line", "last_line", "handle_bindings", "resource_mappings"))
            and edge.get("source_call") == call for edge in call_projection["calls"])
        _require(direct or projected,
                 "scalar call input emitted association differs")
    # Extra original guarded/outside-selection calls need not be retained in
    # resolved_calls. Their constant-expression proof stays compiler-owned;
    # this reader neither evaluates actual expressions nor specializes values.
