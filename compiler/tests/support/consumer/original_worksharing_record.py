"""Transport original native worksharing constraints without granting execution."""
from __future__ import annotations

import hashlib
import re

from .checked_execution_plan import _digest, _require, _sha256, _text

_QUALIFIED = re.compile(r"[a-z][a-z0-9_]*(?:::[a-z][a-z0-9_]*)+\Z")
_FALSE = {
    "internal_cuts_authorized", "descriptor_preservation_authorized", "storage_lifetime_authority",
    "caller_participation_authority", "work_completion_authority", "execution_authorized",
    "payload_access_authorized", "entry_definition_authority", "alias_authority", "gpu_legality_established",
}
_TRUE = {"requires_complete_original_callee_and_body_rescan", "original_native_source_unchanged",
         "requires_whole_original_native_callee"}
_WRAPPER_FALSE = {"internal_cuts_authorized", "execution_authorized", "payload_access_authorized",
                  "entry_definition_authority"}
_WRAPPER = {"descriptor_identity", "descriptor_procedure", "descriptor_structured_identity",
            "descriptor_source_call_closure", "worksharing_requirements", "native_source_mode"} | _WRAPPER_FALSE
_REQUIREMENT = {"version", "kind", "procedure", "source_inputs", "structured_identity",
                "whole_original_procedure_node_id", "regions", "callee_source", "proof_identity"} | _FALSE | _TRUE


def _check(condition, reason):
    _require(condition, "original worksharing " + reason)


def _keys(value, fields):
    _check(isinstance(value, dict) and set(value) == fields, "requires its exact schema")


def _procedure(value):
    return isinstance(value, str) and _QUALIFIED.fullmatch(value) is not None


def _span(value):
    _check(isinstance(value, list) and len(value) == 2
           and all(type(n) is int for n in value) and 1 <= value[0] <= value[1],
           "requires a positive original source span")
    return value


def _directive(record):
    _keys(record, {"source_span", "original_source", "syntax_sha256"})
    _span(record["source_span"])
    text = record["original_source"]
    _check(_text(text) and record["syntax_sha256"] == hashlib.sha256(text.encode()).hexdigest(),
           "directive source digest differs")
    # Only the finite original directive forms accepted by the compiler issuer.
    match = re.fullmatch(r"\s*!\$omp\s+([^\r\n]+?)\s*", text, re.IGNORECASE)
    _check(match is not None, "requires an original OpenMP directive")
    return re.sub(r"\s+", " ", match[1].lower())


def _requirement(record, closure, plan):
    _keys(record, _REQUIREMENT)
    _check(type(record["version"]) is int and record["version"] == 1
           and record["kind"] == "original_orphan_worksharing_storage_structure",
           "has an unsupported structure version")
    _check(_sha256(record["proof_identity"])
           and record["proof_identity"] == _digest({k: v for k, v in record.items() if k != "proof_identity"}),
           "structure identity differs")
    procedure = record["procedure"]
    _check(_procedure(procedure) and record["structured_identity"] == closure.get(procedure)
           and record["source_inputs"] == plan["storage_epoch"]["source_inputs"]
           and record["whole_original_procedure_node_id"] == procedure + "/body",
           "source graph or input identity differs")
    _check(all(record[key] is False for key in _FALSE) and all(record[key] is True for key in _TRUE),
           "cannot grant execution, completion or descriptor authority")
    source = record["callee_source"]
    _keys(source, {"path", "source_sha256", "source_span", "original_callee_sha256"})
    _check(_text(source["path"]) and _sha256(source["source_sha256"])
           and source["source_sha256"] == plan["sources"].get(source["path"])
           and _sha256(source["original_callee_sha256"]), "callee source input differs")
    first, last = _span(source["source_span"])
    regions = record["regions"]
    _check(isinstance(regions, list) and bool(regions), "requires nonempty original regions")
    previous = first - 1
    for region in regions:
        _keys(region, {"directive", "associated_loop", "end_directive", "collapse", "retained_nowait"})
        opening, ending = _directive(region["directive"]), _directive(region["end_directive"])
        match = re.fullmatch(r"do(?: collapse\s*\(\s*([1-4])\s*\))?", opening)
        _check(match is not None and type(region["collapse"]) is int
               and region["collapse"] == int(match[1] or 1), "original collapse clause differs")
        _check(ending in {"end do", "end do nowait"} and type(region["retained_nowait"]) is bool
               and region["retained_nowait"] == (ending == "end do nowait"), "original NOWAIT differs")
        loop = region["associated_loop"]
        _keys(loop, {"source_span", "syntax_sha256"})
        _check(_sha256(loop["syntax_sha256"]), "loop syntax digest differs")
        start, stop = _span(loop["source_span"])
        a, b = region["directive"]["source_span"]
        c, d = region["end_directive"]["source_span"]
        _check(previous < a <= b < start <= stop < c <= d <= last,
               "regions overlap or escape the whole original callee")
        previous = d


def validate_original_worksharing(epoch, plan):
    obligations = epoch["original_worksharing_storage_obligations"]
    _check(isinstance(obligations, list) and bool(obligations), "requires nonempty obligations")
    keys = []
    identities = set()
    for item in obligations:
        _keys(item, _WRAPPER)
        _check(_sha256(item["descriptor_identity"]) and _procedure(item["descriptor_procedure"])
               and _sha256(item["descriptor_structured_identity"]), "descriptor identity differs")
        _check(item["native_source_mode"] == "unchanged original callee; no generated body or companion"
               and all(item[key] is False for key in _WRAPPER_FALSE), "must preserve the whole original native callee")
        closure = item["descriptor_source_call_closure"]
        _check(isinstance(closure, dict) and bool(closure)
               and all(_procedure(name) and _sha256(identity) for name, identity in closure.items()),
               "requires its original descriptor call closure")
        requirements = item["worksharing_requirements"]
        _check(isinstance(requirements, list) and bool(requirements), "requires nonempty structure requirements")
        requirement_keys = []
        for record in requirements:
            _requirement(record, closure, plan)
            requirement_keys.append((record["procedure"], record["proof_identity"]))
        _check(requirement_keys == sorted(set(requirement_keys))
               and len({name for name, _ in requirement_keys}) == len(requirement_keys)
               and len({identity for _, identity in requirement_keys}) == len(requirement_keys),
               "structure requirements repeat or change order")
        keys.append((item["descriptor_procedure"], item["descriptor_identity"]))
        identities.add(item["descriptor_identity"])
    _check(keys == sorted(set(keys)) and len(identities) == len(keys), "obligations repeat or change order")


def _overlaps(first, last, protected):
    # An insertion at either original endpoint also changes the protected callee.
    return first <= protected[1] and max(first, last) >= protected[0]


def validate_original_worksharing_constraints(manifest):
    """Check constraints across every validated scope in the same source package."""
    validate_original_worksharing_package_constraints((manifest,))


def validate_original_worksharing_package_constraints(manifests):
    """Check independently validated packages before applying any replacement."""
    manifests = tuple(manifests)
    plans = [scope["checked_execution_plan"] for manifest in manifests for scope in manifest.get("scopes", [])
             if isinstance(scope, dict) and "checked_execution_plan" in scope]
    records = [record for plan in plans
               for item in plan.get("storage_epoch", {}).get("original_worksharing_storage_obligations", [])
               for record in item["worksharing_requirements"]]
    if not records:
        return
    edits = []
    for manifest in manifests:
        original_edits = manifest.get("source_edits", [])
        _check(isinstance(original_edits, list), "requires original source edit intervals")
        edits.extend((edit, manifest["source_inputs"]) for edit in original_edits)
    for record in records:
        source = record["callee_source"]
        for plan in plans:
            for procedure in plan["procedures"]:
                _check(procedure["procedure"] != record["procedure"],
                       "whole native callee cannot be a managed member or companion")
                if procedure["source"] == source["path"]:
                    for unit in procedure["units"]:
                        _check(not _overlaps(unit["first_line"], unit["last_line"], source["source_span"]),
                               "managed unit overlaps the whole native callee")
            for member in plan.get("storage_epoch", {}).get("members", []):
                for task in member.get("device_tasks", []):
                    _check(task["call"]["procedure"] != record["procedure"],
                           "whole native callee cannot be an external device task")
        for edit, inputs in edits:
            _keys(edit, {"source", "source_sha256", "first_line", "last_line", "replacement"})
            _check(_text(edit["source"]) and _sha256(edit["source_sha256"])
                   and edit["source_sha256"] == inputs.get(edit["source"])
                   and type(edit["first_line"]) is int and type(edit["last_line"]) is int
                   and 1 <= edit["first_line"] <= edit["last_line"] + 1
                   and isinstance(edit["replacement"], str), "source edit interval differs")
            _check(edit["source"] != source["path"]
                   or not _overlaps(edit["first_line"], edit["last_line"], source["source_span"]),
                   "source edit overlaps the whole native callee")
