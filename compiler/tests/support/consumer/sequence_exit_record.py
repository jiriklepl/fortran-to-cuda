"""Transport an original branch exit without recreating source authority."""
from __future__ import annotations

from .checked_execution_plan import _digest, _require, _sha256, _text

_FIELDS = {
    "identity", "version", "kind", "procedure", "source_inputs", "source_graph",
    "parent_sequence", "branch_node_id", "branch_ordinal", "selected_node_ids",
    "last_call_node_id", "source_span", "exit_anchor", "cleanup", "activation",
    "execution_authorized", "entry_definition_authority", "cross_exit_retention_authorized", "alias_authority",
}


def _span(value):
    return (isinstance(value, list) and len(value) == 2
            and all(type(number) is int for number in value) and 1 <= value[0] <= value[1])


def validate_sequence_exit(value, epoch, plan):
    """Check exact identity, graph/source linkage and the versioned exit grammar.

    The compiler's registered original parser objects establish the real branch
    and insertion site. A portable reader checks their public links; it cannot
    use this record as permission to reconstruct or move source operations.
    """
    _require(isinstance(value, dict) and set(value) == _FIELDS
             and type(value["version"]) is int and value["version"] == 1
             and value["kind"] == "original_branch_sequence_exit"
             and _sha256(value["identity"])
             and value["identity"] == _digest({key: item for key, item in value.items() if key != "identity"}),
             "local sequence exit requires its exact versioned content identity")
    _require(value["procedure"] == plan["entry"] == epoch["entry"]
             and value["source_inputs"] == epoch["source_inputs"]
             and value["source_graph"] == plan["procedures"][0]["structured_summary_identity"],
             "local sequence exit original source linkage differs")
    _require(all(value[key] is False for key in ("execution_authorized", "entry_definition_authority",
                                                "cross_exit_retention_authorized", "alias_authority"))
             and value["cleanup"] == "publish observable storage and wait; destroy domain and handles immediately after last CALL"
             and value["activation"] == "inside unchanged original branch after predicate evaluation",
             "local sequence exit cannot grant definitions, execution, aliases or retention")
    ordinal, branch, parent = value["branch_ordinal"], value["branch_node_id"], value["parent_sequence"]
    _require(type(ordinal) is int and ordinal >= 0 and _text(branch)
             and branch.startswith(plan["entry"] + "/") and parent == branch + "/body" + str(ordinal),
             "local sequence exit original branch or parent sequence differs")
    selected = value["selected_node_ids"]
    _require(isinstance(selected, list) and len(selected) >= 2
             and all(_text(node) for node in selected) and len(set(selected)) == len(selected)
             and value["last_call_node_id"] == selected[-1],
             "local sequence exit requires its distinct original CALL suffix")
    positions = [node.rpartition("/") for node in selected]
    _require(all(prefix == parent and separator == "/" and number.isascii() and number.isdecimal()
                 and str(int(number)) == number for prefix, separator, number in positions)
             and [int(number) for _prefix, _separator, number in positions]
             == list(range(int(positions[0][2]), int(positions[0][2]) + len(positions))),
             "local sequence exit original CALL order or contiguity differs")
    anchor = value["exit_anchor"]
    _require(_span(value["source_span"]) and isinstance(anchor, dict)
             and set(anchor) == {"parser_kind", "source_span", "original_source_sha256"}
             and _text(anchor["parser_kind"])
             and anchor["parser_kind"] in {"Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}
             and _span(anchor["source_span"]) and _sha256(anchor["original_source_sha256"])
             and value["source_span"][1] < anchor["source_span"][0],
             "local sequence exit lost its independent original cleanup anchor")
    return value
