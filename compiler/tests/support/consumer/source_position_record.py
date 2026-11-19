"""Transport a reached original root position without issuing source authority."""
from __future__ import annotations

import re

from .checked_execution_plan import _digest, _require, _sha256
from .entry_preservation_record import _direct_children
from .sequence_exit_record import _span

_FLAGS = {"execution_authorized", "entry_definition_authority", "descriptor_lifetime_authority",
          "payload_access_authorized", "exit_authority", "cross_exit_retention_authorized", "alias_authority"}
_FIELDS = _FLAGS | {"identity", "version", "kind", "procedure", "source_inputs", "source_graph",
                   "parent_sequence", "selected_node_ids", "activation_node_id", "source_span",
                   "original_units", "activation", "prefix_scope"}
_NUMBER = r"(?:0|[1-9][0-9]*)"


def validate_source_position(value, group, epoch, plan, *, original_tasks):
    """Check exact public links; parser-object and lifetime checks stay independent."""
    _require(isinstance(value, dict) and set(value) == _FIELDS
             and type(value["version"]) is int and value["version"] == 1
             and value["kind"] == "original_root_sequence_position"
             and _sha256(value["identity"])
             and value["identity"] == _digest({key: item for key, item in value.items() if key != "identity"}),
             "source position requires its exact versioned original record")
    entry = plan["entry"]
    _require(value["procedure"] == group["procedure"] == entry
             and value["source_inputs"] == epoch["source_inputs"]
             and value["source_graph"] == group["source_graph"]
             and value["parent_sequence"] == entry+"/body"
             and value["selected_node_ids"] == group["selected_node_ids"],
             "source position original source, root or selected interval differs")
    _require(all(value[flag] is False for flag in _FLAGS)
             and value["activation"] == "before the first selected original root unit after its complete original prefix"
             and value["prefix_scope"] == "original root sequence only; no enclosing loop, association or branch",
             "source position cannot grant execution, definitions, lifetime, payloads, exits or aliases")
    selected = value["selected_node_ids"]
    grammar = re.escape(entry)+"/body/"+_NUMBER
    _require(isinstance(selected, list) and bool(selected) and all(isinstance(node, str) and re.fullmatch(grammar, node)
             for node in selected) and len(set(selected)) == len(selected)
             and value["activation_node_id"] == selected[0],
             "source position requires distinct exact original root units")
    _require(isinstance(original_tasks, list) and all(isinstance(task, dict) and isinstance(task.get("id"), str)
                                                    for task in original_tasks),
             "source position requires its existing complete original task inventory")
    children = [entry+"/body/"+str(number) for number in _direct_children({task["id"] for task in original_tasks}, entry+"/body")]
    _require(selected[0] in children and selected[-1] in children
             and selected == children[children.index(selected[0]):children.index(selected[-1])+1],
             "source position omits or reorders original root units")
    units = value["original_units"]
    _require(isinstance(units, list) and len(units) == len(selected) and _span(value["source_span"]),
             "source position requires every original physical source unit")
    previous = 0
    for identity, unit in zip(selected, units, strict=True):
        _require(isinstance(unit, dict) and set(unit) == {"node_id", "source_span", "original_source_sha256"}
                 and unit["node_id"] == identity and _span(unit["source_span"])
                 and unit["source_span"][0] > previous and _sha256(unit["original_source_sha256"]),
                 "source position original unit span or syntax differs")
        previous = unit["source_span"][1]
    _require(value["source_span"] == [units[0]["source_span"][0], units[-1]["source_span"][1]]
             and value["source_span"] == group["sequence_exit"]["source_span"],
             "source position differs from its exact original interval extent")
    return value
