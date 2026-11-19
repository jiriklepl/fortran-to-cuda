"""Check the public source/graph dispatch linkage, without issuing authority."""
from __future__ import annotations

from .checked_execution_plan import _digest, _require, _sha256

_FIELDS = {"identity", "version", "kind", "policy", "task_id", "owner", "procedure", "source_proof_identity",
           "pricing_identity", "canonical_resources", "registration_slots", "implementations", "unavailable_target",
           "native_endpoint", "gpu_endpoint", "execution_authorized"}


def _tasks(graph):
    """Follow only the existing explicit structured diagnostic graph grammar."""
    pending, result, seen = [graph], [], set()
    while pending:
        current = pending.pop()
        _require(isinstance(current, dict) and isinstance(current.get("tasks"), list),
                 "source placement requires its original task graph")
        _require(id(current) not in seen, "source placement repeats a structured graph")
        seen.add(id(current))
        for task in current["tasks"]:
            _require(isinstance(task, dict), "source placement requires original task records")
            result.append(task)
            if "branch" in task:
                _require(isinstance(task["branch"], list) and len(task["branch"]) == 2,
                         "source placement original branch differs")
                pending.extend(task["branch"])
            if "loop" in task:
                pending.append(task["loop"])
    return result


def _placements(schedule):
    pending, result, seen = [schedule], [], set()
    while pending:
        current = pending.pop()
        _require(isinstance(current, dict) and isinstance(current.get("placements"), list),
                 "source placement requires its original schedule")
        _require(id(current) not in seen, "source placement repeats a structured schedule")
        seen.add(id(current))
        for placement in current["placements"]:
            _require(isinstance(placement, dict) and isinstance(placement.get("children"), list),
                     "source placement requires complete scheduled task records")
            result.append(placement)
            pending.extend(placement["children"])
    return result


def validate_source_placement(value, source, pricing, binding, graph_execution, ordinal):
    _require(isinstance(value, dict) and set(value) == _FIELDS
             and type(value["version"]) is int and value["version"] == 1
             and _sha256(value["identity"])
             and value["identity"] == _digest({k: v for k, v in value.items() if k != "identity"}),
             "source placement requires its exact versioned identity")
    _require(type(ordinal) is int and ordinal >= 0
             and value["task_id"] == source["procedure"] + ":source_device:" + str(ordinal)
             and value["kind"] == "checked_source_runtime_placement" and value["policy"] == "live_source_quote_v2"
             and value["procedure"] == source["procedure"] and value["owner"] == binding["owner"]
             and value["source_proof_identity"] == source["source_proof_identity"]
             and value["pricing_identity"] == pricing["identity"]
             and value["canonical_resources"] == binding["canonical_resources"]
             and value["registration_slots"] == binding["registration_slots"]
             and all(type(slot) is int for slot in value["registration_slots"].values())
             and value["implementations"] == ["cpu", "gpu"] and value["unavailable_target"] == "cpu"
             and value["native_endpoint"] == "typed_host_live" and value["gpu_endpoint"] == "source_device_live"
             and value["execution_authorized"] is False,
             "source placement source, pricing, physical storage or endpoint differs")
    _require(isinstance(graph_execution, dict) and graph_execution.get("source_order_executor") is True
             and isinstance(graph_execution.get("procedures"), list), "source placement requires its source-order graph")
    procedures = [item for item in graph_execution["procedures"]
                  if isinstance(item, dict) and item.get("procedure") == source["procedure"]]
    _require(len(procedures) == 1, "source placement requires one exact original graph procedure")
    procedure = procedures[0]
    tasks = [task for task in _tasks(procedure.get("task_graph")) if task.get("id") == value["task_id"]]
    scheduled = [task for task in _placements(procedure.get("schedule")) if task.get("task") == value["task_id"]]
    _require(len(tasks) == len(scheduled) == 1, "source placement task or schedule identity differs")
    task, selected = tasks[0], scheduled[0]
    deferred = {"version": 1, "policy": value["policy"], "identity": value["identity"]}
    roots = sorted(binding["canonical_resources"].values())
    writes = sorted(binding["canonical_resources"][effect["resource"]]
                    for effect in source["resources"] if effect["access"] == "readwrite")
    _require(set(task) == {"id", "implementations", "reads", "may_writes", "scalar_reads", "scalar_writes",
                           "lifetime_resources", "metadata_reads", "opaque", "ordered", "deferred_placement"}
             and set(selected) == {"task", "target", "estimated_cost", "transfers", "evictions", "children",
                                  "deferred_placement"}
             and task.get("deferred_placement") == deferred and selected.get("deferred_placement") == deferred
             and type(task["deferred_placement"].get("version")) is int
             and type(selected["deferred_placement"].get("version")) is int
             and task.get("implementations") == ["cpu", "gpu"] and task.get("ordered") is True
             and task.get("opaque") is False and task.get("reads") == task.get("metadata_reads") == roots
             and task.get("may_writes") == writes and "branch" not in task and "loop" not in task
             and task.get("scalar_reads") == sorted(item["resource"] for item in source["scalars"])
             and task.get("scalar_writes") == task.get("lifetime_resources") == []
             and selected.get("target", "missing") is None and selected.get("estimated_cost", "missing") is None
             and selected.get("transfers") == selected.get("evictions") == selected.get("children") == [],
             "source placement must remain deferred with both exact physical implementations")
