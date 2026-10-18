"""Measured complete-process placement for one checked original source interval.

The checked plan remains the execution authority. A finalized entry projection
supplies safe root geometry; internal native work is observed without hoisting
its controls. Complete-process paired observations use the existing exact-window
issuer. Empirical applicability does not assert a future invocation count or
price cold startup using warm calls.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256

from compiler.ir import CompilationError
from compiler.offload.exact_window import (
    HOLDOUTS,
    SAMPLES,
    _issue_placement,
    _paired_timing_decision,
    _request_snapshot,
    digest,
    implementation_identity,
)

PROTOCOL = "checked-interval-process-v4"
PREFIX = "FORT_SELECTION_OBSERVATION "


def _acceptance(value):
    """An explicit source-bound performance criterion never changes legality."""
    if (type(value) is not dict or set(value) != {"criterion"}
            or type(value["criterion"]) is not str
            or value["criterion"] not in {"strict_advantage", "paired_non_regression"}):
        raise CompilationError("selected timing acceptance requires strict_advantage or paired_non_regression")
    return dict(value)


def _valid_key(key, count):
    return (type(key) is list and len(key) == count
            and all(type(value) is int and -(2**63) <= value < 2**63 for value in key))


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _valid_workload(workload):
    return (type(workload) is dict and set(workload) == {"identity", "argv", "environment", "files", "completion"}
            and type(workload["argv"]) is list and bool(workload["argv"])
            and all(type(arg) is str and "\0" not in arg for arg in workload["argv"])
            and type(workload["environment"]) is dict
            and all(type(name) is str and bool(name) and "\0" not in name
                    and (value is None or type(value) is str and "\0" not in value)
                    for name, value in workload["environment"].items())
            and type(workload["files"]) is dict and 0 < len(workload["files"]) <= 64
            and all(type(name) is str and bool(name) and not name.startswith("/") and ".." not in name.split("/")
                    and "\0" not in name and _sha(value) for name, value in workload["files"].items())
            and type(workload["completion"]) is dict and bool(workload["completion"])
            and workload["identity"] == digest({key: value for key, value in workload.items() if key != "identity"}))


def process_evidence_decision(evidence, contract):
    """Keep build verification local to one decision and recheck before return."""
    try:
        from compiler.offload.application_build import application_build_validation

        with application_build_validation():
            return _process_evidence_decision(evidence, contract)
    except (CompilationError, KeyError, TypeError, ValueError, OverflowError, OSError) as error:
        return {"available": False, "selection": "native", "reason": str(error)}


def _process_evidence_decision(evidence, contract):
    """Reconstruct complete-process acceptance and every usable reached key.

    Every campaign is one explicit observed workload class. A key observed in an
    unprofitable campaign is not admitted merely because another campaign wins.
    The native observations have an independently observed CPU/build identity;
    resident observations must report actual completed launches, not a request.
    """
    try:
        from compiler.offload.application_build import validate_application_build_observation
        from compiler.offload.selection_validation import validate_campaign_numerics

        if (type(evidence) is not dict or evidence.get("protocol") != PROTOCOL
                or type(evidence.get("campaigns")) is not list or not evidence["campaigns"]):
            raise ValueError("complete-process selection calibration missing")
        if (type(contract) is not dict or type(contract.get("schema_version")) is not int
                or contract["schema_version"] != 2 or contract.get("protocol") != PROTOCOL):
            raise ValueError("complete-process selection contract version differs")
        acceptance = _acceptance(contract.get("acceptance"))
        if len(contract["selection_bundle"]["groups"]) != 1:
            raise ValueError("complete-process production pricing currently requires one admitted selection per bundle")
        accepted, rejected, identities, campaigns, workloads = {}, set(), {}, [], {}
        count = len(contract["key"])
        for campaign in evidence["campaigns"]:
            if type(campaign) is not dict or campaign.get("contract") != contract:
                raise ValueError("complete-process source/selection/build identity differs")
            workload = campaign.get("workload")
            if not _valid_workload(workload) or workload not in contract["workloads"]:
                raise ValueError("complete-process workload identity is incomplete")
            rows = campaign.get("observations")
            if campaign.get("selection_bundle") != contract["selection_bundle"]:
                raise ValueError("complete-process enabled selection bundle differs")
            if type(rows) is not list or len(rows) != 2 * (SAMPLES + HOLDOUTS):
                raise ValueError("complete-process calibration requires seven pairs and two holdouts")
            key_sequence, all_keys, per_key_outcomes, builds = None, set(), {}, {}
            projected = deepcopy(campaign)
            first_key = None
            for order, row in enumerate(rows):
                if type(row) is not dict:
                    raise ValueError("complete-process observation must be an object")
                mode = ("native", "resident")[(order // 2 + order % 2) % 2]
                process = row.get("process")
                if (type(process) is not dict or process.get("returncode") != 0
                        or type(process.get("returncode")) is not int
                        or process.get("workload_identity") != workload["identity"]
                        or process.get("argv") != workload["argv"]
                        or process.get("environment") != workload["environment"]
                        or process.get("files") != workload["files"]
                        or process.get("completion") != workload["completion"]
                        or process.get("selection_bundle_identity") != contract["selection_bundle"]["identity"]
                        or process.get("wall_time_includes") != ["startup", "first_invocation", "all_invocations", "output", "teardown"]
                        or process.get("seconds") != row.get("seconds")):
                    raise ValueError("complete-process observation does not establish unchanged workload completion")
                validate_application_build_observation(row.get("build"), contract=contract["build_contract"], mode=mode,
                                                      selection_contract_identity=digest(contract))
                if mode in builds and row["build"] != builds[mode]:
                    raise ValueError("complete-process numerical validation requires the same measured build per mode")
                builds[mode] = row["build"]
                invocations = row.get("invocations")
                if type(invocations) is not list or not invocations:
                    raise ValueError("complete-process observation lacks actual selected invocations")
                sequence, summed, launched = [], 0.0, 0
                for ordinal, invocation in enumerate(invocations):
                    if (type(invocation) is not dict or invocation.get("ordinal") != ordinal
                            or type(invocation.get("ordinal")) is not int
                            or invocation.get("selection") != contract["selection_identity"]
                            or not _valid_key(invocation.get("key"), count)
                            or type(invocation.get("launches")) is not int or invocation["launches"] < 0
                            or invocation.get("complete") is not True
                            or type(invocation.get("fallbacks")) is not int or invocation["fallbacks"] != 0
                            or type(invocation.get("seconds")) not in {int, float}
                            or not math.isfinite(invocation["seconds"]) or invocation["seconds"] <= 0):
                        raise ValueError("selected invocation key/outcome/timing is invalid")
                    key = tuple(invocation["key"])
                    sequence.append(key)
                    all_keys.add(key)
                    summed += invocation["seconds"]
                    launched += invocation["launches"]
                    if mode == "native":
                        if invocation.get("outcome") != "native" or invocation["launches"] != 0:
                            raise ValueError("independent native process executed a resident alternative")
                    else:
                        resident = invocation.get("outcome") == "resident"
                        if not resident:
                            raise ValueError("selected resident outcome is incomplete or mixed")
                        per_key_outcomes.setdefault(key, []).append(resident)
                if mode == "resident" and not launched:
                    raise ValueError("resident process has no completed GPU work")
                if summed > row["seconds"]:
                    raise ValueError("selected invocation times exceed complete-process elapsed time")
                if key_sequence is None:
                    key_sequence = sequence
                    first_key = list(sequence[0])
                elif sequence != key_sequence:
                    raise ValueError("paired complete-process selection key sequence changed")
                # Reuse the strict original raw-pair order, hardware, independent
                # CPU and complete elapsed-time checks. Keys
                # are authenticated above as a sequence, not silently collapsed.
                projected["observations"][order]["key"] = first_key
                projected["observations"][order]["observed_key"] = first_key
            projected["key"] = first_key
            decision = _paired_timing_decision(projected, contract, protocol=PROTOCOL,
                                              criterion=acceptance["criterion"])
            if not decision["available"]:
                raise ValueError(decision["reason"])
            identity = decision["identity"]
            numerical = validate_campaign_numerics(
                campaign, contract=contract, workload=workload, builds=builds,
                key_sequence=key_sequence, identity=identity)
            identity_key = digest(identity)
            identities[identity_key] = identity
            workloads[workload["identity"]] = workload
            usable = {key for key, outcomes in per_key_outcomes.items() if outcomes and all(outcomes)}
            if usable != all_keys:
                raise ValueError("resident process did not complete the entire measured selection alternative")
            if decision["selection"] != "resident":
                rejected.update(all_keys)
            else:
                for key in usable:
                    accepted.setdefault(key, set()).add((identity_key, workload["identity"]))
                rejected.update(all_keys - usable)
            campaigns.append({"workload_identity": workload["identity"], "invocations": len(key_sequence),
                              "native_seconds": decision["native_seconds"], "resident_seconds": decision["resident_seconds"],
                              "selection": decision["selection"], "numerical_validation": numerical})
        if any(campaign["selection"] != "resident" for campaign in campaigns):
            rejected.update(accepted)
        entries = [{"key": list(key), "identity": identities[identity], "workload": workloads[workload]}
                   for key in sorted(accepted.keys() - rejected) for identity, workload in sorted(accepted[key])]
        if len(identities) != 1:
            raise ValueError("complete-process campaigns require one compatible execution environment")
        requirement = ("paired non-regression" if acceptance["criterion"] == "paired_non_regression" else
                       "selection advantage")
        return {"available": True, "selection": "resident" if entries else "native",
                "reason": (f"measured complete-process {requirement}" if entries else
                           f"complete-process {requirement} not established"),
                "acceptance": acceptance,
                "accepted_keys": entries, "campaigns": campaigns, "evidence_sha256": digest(evidence),
                "applicability": "empirical observed complete-process workload classes; no future invocation-count guarantee"}
    except (CompilationError, KeyError, TypeError, ValueError, OverflowError, OSError) as error:
        return {"available": False, "selection": "native", "reason": str(error)}


def _workload_key(owner, interval):
    """Use only the finalized interval's safe root-entry projection.

    The key distinguishes reached geometry. Internal native branches, child
    tiling and private work remain in the empirical source/workload class;
    they are never hoisted or replayed to predict an instruction count.
    """
    entry = interval.cost_entry()
    if type(entry) is not dict or set(entry) != {"arrays", "integers", "identity"} or not _sha(entry["identity"]):
        raise CompilationError("interval pricing requires a finalized safe entry projection")
    key, expressions, guards, seen = [], [], [], set()
    for rank, bindings in ((True, entry["arrays"]), (False, entry["integers"])):
        for binding in bindings:
            if binding.root in seen:
                continue
            if bool(binding.rank) != rank or binding.attributes & {"optional", "pointer", "volatile", "asynchronous"}:
                raise CompilationError("interval cost entry has an unsafe original descriptor or control")
            name = owner.builder.visible(owner.routine, binding.root)
            if binding.rank:
                if "allocatable" in binding.attributes:
                    guards.append(f"allocated({name})")
                guards.append(f"is_contiguous({name})")
                for axis in range(1, binding.rank + 1):
                    for inquiry in ("size", "lbound"):
                        key.append({"resource": binding.root, "property": inquiry, "axis": axis})
                        expressions.append(f"{inquiry}({name},{axis},kind=c_int64_t)")
            else:
                if binding.dtype != "integer" or binding.kind not in {4, 8}:
                    raise CompilationError("interval cost entry requires an original INTEGER control")
                key.append({"resource": binding.root, "property": "integer_value"})
                expressions.append(f"int({name},c_int64_t)")
            seen.add(binding.root)
    if not key or len(key) > 256:
        raise CompilationError("interval workload entry geometry unavailable")
    return key, expressions, guards, entry["identity"]


def prepare(owner):
    """Price one exact admitted interval; measurements never widen legality."""
    from compiler.emission.common.resources import read_common_header
    from compiler.frontend.immutable_graph import freeze_graph
    from compiler.numerical_contract import numerical_build_contract
    from compiler.offload.application_build import require_application_build_contract
    from compiler.offload.selection_validation import normalize_policy

    builder = owner.builder
    request = deepcopy(builder.config.exact_window)
    snapshot = _request_snapshot(request)
    acceptance = _acceptance(request.get("acceptance", {"criterion": "strict_advantage"}))
    plan = owner.execution_plan.validate()
    interval = getattr(plan, "interval", None)
    if interval is None:
        raise CompilationError("interval pricing requires an issued checked source interval")
    interval.validate(plan)
    if owner is not plan._owner or owner.parent is not None:
        raise CompilationError("interval pricing requires its exact owning root")
    units = tuple(unit for member in plan.members for unit in plan.units(member))
    if not units or not plan.leaves:
        raise CompilationError("interval pricing requires checked numerical work")
    variant = request.get("calibration")
    if variant is not None and variant not in {"native", "resident"}:
        raise CompilationError("selected calibration requires an explicit native or resident artifact")
    if builder.config.policy != "auto":
        raise CompilationError("selected calibration and production require identical automatic numerical options")
    key, expressions, guards, entry_identity = _workload_key(owner, interval)
    workloads = request.get("workloads")
    if (type(workloads) is not list or not workloads or len(workloads) > 32
            or any(not _valid_workload(workload) for workload in workloads)
            or len({workload["identity"] for workload in workloads}) != len(workloads)):
        raise CompilationError("selected calibration requires explicit runtime workload input identities")
    public = plan.public()
    selection = deepcopy(interval.selection_context)
    bundle = getattr(owner, "selection_bundle", None)
    if bundle is None:
        bundle = {"schema_version": 1, "entry": owner.routine.qualified,
                  "groups": [{"selection_identity": digest(selection), **{name: selection[name] for name in
                              ("first_line", "last_line", "original_source_sha256")}}]}
        bundle["identity"] = digest(bundle)
    workers, artifacts = {}, {}
    for leaf in sorted(plan.leaves):
        generated = builder.numerical(leaf)
        if generated is None:
            raise CompilationError("selected pricing requires the checked numerical artifacts")
        workers[leaf] = {name: sha256(text.encode()).hexdigest() for name, text in generated.artifacts.items()
                         if name in {"shared_entry.cu", "shared_interface.f90"}}
        _public, directory = builder.entry_artifacts(leaf)
        artifacts.update({directory + "/" + name: value for name, value in workers[leaf].items()})
    implementation = implementation_identity()
    build = require_application_build_contract(
        request.get("build_contract"), sources=dict(builder.analysis.sources),
        source_inputs_identity=public["source_inputs_identity"],
        configuration_identity=public["configuration_identity"],
        compiler_identity=digest(implementation), options_identity=digest(asdict(builder.options)),
        worker_artifacts=artifacts, runtime_identity=builder.runtime["runtime_id"])
    validation_policy = normalize_policy(request.get("validation_policy"), build_contract=build)
    contract = {
        "schema_version": 2, "protocol": PROTOCOL, "entry": owner.routine.qualified,
        "acceptance": acceptance,
        "sources": dict(sorted(builder.analysis.sources.items())), "selection": selection,
        "selection_bundle": deepcopy(bundle),
        "workloads": workloads,
        "selection_identity": digest(selection), "checked_execution_identity": plan.execution_identity,
        "entry_projection_identity": entry_identity, "completion_manifest": interval.completion_manifest(),
        "workers": workers, "runtime": builder.runtime["runtime_id"], "implementation": implementation,
        "common_functions_sha256": sha256(read_common_header().encode()).hexdigest(),
        "source_inputs": builder.analysis.inputs.identity(), "facts": builder.facts,
        "compiler_options": asdict(builder.options), "numerical_contract": numerical_build_contract(),
        "build_contract": build, "validation_policy": validation_policy, "threads": builder.config.host_threads,
        "original_participation": [unit.scope.completion.public() for unit in units
                                   if getattr(unit.scope, "completion", None) is not None],
        "transfers": builder.config.scope_transfers, "device_budget_bytes": builder.device_budget, "key": key,
        "outcome_contract": {"successful_completion": True, "fallbacks": 0, "requires_process_gpu_launch": True},
        "coverage": "complete application process; checked interval includes original controls, native interruptions, teams, transfers and cleanup",
        "applicability": "empirical observed source/workload class and safe entry geometry; internal native work is measured, not predicted; no future invocation-count guarantee",
    }
    # This versioned contract is a JSON wire record. Normalize tuple-valued
    # options now, before both capture and consumption, so saved evidence does
    # not acquire a different meaning merely by crossing the public interface.
    contract = json.loads(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False))
    evidence = request.get("evidence")
    if isinstance(evidence, dict) and "selections" in evidence:
        evidence = evidence["selections"].get(contract["selection_identity"])
    decision = ({"available": False, "selection": "calibration", "reason": "independent complete-process calibration artifact"}
                if variant else process_evidence_decision(evidence, contract))
    record = freeze_graph({"contract": contract, "contract_sha256": digest(contract), **decision,
                           "expressions": expressions, "guards": guards,
                           "capture": variant is not None, "capture_variant": variant})
    return _issue_placement(owner, record, snapshot)


def _names(owner):
    record = owner.exact_window.record
    return "fort_selection_" + record["contract_sha256"][:12], record


def _interface(name, arguments="", declarations=()):
    return ["interface", f"subroutine {name}({arguments}) bind(C)",
            "import c_int64_t,c_char", *declarations, "end subroutine", "end interface"]


def _imports():
    return ["use iso_c_binding, only: c_int,c_int64_t,c_char,c_null_char",
            "use, intrinsic :: iso_fortran_env, only: fort_version => compiler_version, fort_options => compiler_options"]


def _publish(owner):
    from compiler.emission.common.resources import read_common_header

    name, record = _names(owner)
    owner.builder.outputs["exact/" + name + ".cu"] = _observer_source(name, record)
    owner.builder.outputs["exact/common_functions.cuh"] = read_common_header()
    for header in ("scoped_runtime.h", "scoped_team_observer.hpp"):
        owner.builder.outputs["exact/" + header] = owner.builder.runtime_outputs[header]


def _entry_control(owner, *, native):
    name, record = _names(owner)
    _publish(owner)
    count = len(record["expressions"])
    suffix = "_native" if native else "_key"
    declarations = [*_imports(), *_interface(name + "_begin"), *_interface(name + "_end"),
                    *_interface(name + suffix, "key,version,options", (
                        f"integer(c_int64_t), intent(in) :: key({count})",
                        "character(c_char), intent(in) :: version(*),options(*)")),
                    "logical :: fort_interval_key_ready", f"integer(c_int64_t) :: fort_interval_key({count})"]
    entry = [f"call {name}_begin()", "fort_interval_key_ready = .true."]
    # Never depend on Fortran conjunction short-circuiting or re-evaluate a
    # child initializer. The finalized projection owns every queried binding.
    entry += [f"if (fort_interval_key_ready) fort_interval_key_ready = {guard}" for guard in record["guards"]]
    entry += ["if (fort_interval_key_ready) then"]
    entry += [f"fort_interval_key({index}) = {expression}" for index, expression in enumerate(record["expressions"], 1)]
    entry += [f"call {name}{suffix}(fort_interval_key,fort_version()//c_null_char,fort_options()//c_null_char)", "endif"]
    return declarations, entry


def control(owner):
    """Store a safe root key without CUDA, before original interval execution."""
    _name, record = _names(owner)
    if not record["capture"] and record["selection"] != "resident":
        return [], [owner.enabled + " = .false."]
    return _entry_control(owner, native=False)


def choice(owner, unit, handles, values, *, context="fort_team_context", enabled="fort_team_enabled",
           rejected="fort_native_ready"):
    """Choose after existing reached legality checks, using the stored root key."""
    del handles, values
    root = owner.owner
    plan = root.execution_plan
    if not any(unit is item for member in plan.members for item in plan.units(member)):
        raise CompilationError("interval workload choice requires an exact checked operation")
    name, _record = _names(root)
    lines = ["block", *_imports(), "interface",
             f"function {name}_choose() bind(C) result(selected)",
             "import c_int", "integer(c_int) :: selected", "end function", "end interface"]
    from copy import copy

    close_owner = copy(owner)
    close_owner.context, close_owner.enabled, close_owner.control_guard = context, enabled, None
    lines += [f"if ({name}_choose() == 0) then", *close_owner.close(),
              f"{rejected} = .false.", "endif", "end block"]
    return lines


def fallback(owner):
    """Record an owning GPU/native or suffix transition, never a planned CALL."""
    root = owner.owner
    name, record = _names(root)
    if not record["capture"] and record["selection"] != "resident":
        return []
    return ["block", "use iso_c_binding, only: c_int64_t,c_char", *_interface(name + "_fallback"),
            f"call {name}_fallback()", "end block"]


def before_close(owner, context):
    """Retain actual launches before destruction; close time remains measured."""
    name, record = _names(owner.owner)
    if not record["capture"] and record["selection"] != "resident":
        return []
    return ["block", "use iso_c_binding, only: c_int64_t,c_char",
            *_interface(name + "_close", "context", ("integer(c_int64_t), value :: context",)),
            f"call {name}_close({context})", "end block"]


def finish(owner):
    """Only the successful final cleanup reaches this completion event."""
    name, record = _names(owner.owner)
    if not record["capture"] and record["selection"] != "resident":
        return []
    return ["block", "use iso_c_binding, only: c_int64_t,c_char", *_interface(name + "_end"),
            f"call {name}_end()", "end block"]


def native_control(owner):
    """The same root key, followed by the exact original interval once."""
    declarations, entry = _entry_control(owner, native=True)
    return declarations, entry, finish(owner)


def _observer_source(name, record):
    from compiler.offload.exact_window import ENVIRONMENT, selector_source
    from compiler.offload.selection_workload import workload_source

    capture = record["capture"]
    native = record.get("capture_variant") == "native"
    observe = name + "_remember"
    if native:
        source = """#include <omp.h>
#include <sched.h>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <cstdio>
#include <iomanip>
#define FORT_OFFLOAD_ENABLED 1
#include "common_functions.cuh"
#include "scoped_runtime.h"
static std::string quoted(const std::string &s) {
 std::ostringstream o; o << '"';
 for(unsigned char c:s){if(c=='"'||c=='\\\\')o<<'\\\\'<<c;
 else if(c<32)o<<"\\\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c);else o<<c;}
 o<<'"';return o.str();}
"""
    else:
        # Choice occurs inside the original master after all participants vote.
        # The shared exact selector's serial check is replaced by the identical
        # original-team identity, while environment/key/build checks remain.
        source = selector_source(name + "_select", record, observer=observe)
        source = source.replace("omp_get_level()!=0",
                                "(omp_get_level()!=0&&(omp_get_level()!=1||omp_get_num_threads()!=@THREADS@))")
        source = source.replace("@THREADS@", str(record["contract"]["threads"]))
        source += workload_source([{"workload": workload} for workload in record["contract"]["workloads"]]
                                   if capture else record.get("accepted_keys", ()))
        source += ('\nextern "C" int ' + name + '(const int64_t *key,const char *version,const char *options) {\n'
                   + "if(!fort_selection_workload_matches(key))return 0;\n"
                   + "return " + name + "_select(key,version,options);\n}\n")
    if native:
        source += workload_source([{"workload": workload} for workload in record["contract"]["workloads"]])
    count = len(record["expressions"])
    environment = []
    for variable in sorted(ENVIRONMENT):
        environment.append("{const char *v=std::getenv(" + json.dumps(variable) + ");if(!first)environment<<\",\";"
                           "first=false;environment<<" + json.dumps(json.dumps(variable) + ":")
                           + "<<(v?quoted(v):\"null\");}")
    source += r'''
#include <chrono>
namespace {
struct selection_observation {
 bool active=false, keyed=false, has_key=false; uint64_t launches=0, fallbacks=0;
 int decision=-1;
 int64_t values[@COUNT@]{};
 std::string version, options;
 std::chrono::steady_clock::time_point start;
 std::string key="null", identity="null", cpu="null";
};
thread_local selection_observation selection_state;
static std::string key_json(const int64_t *key) {
 std::ostringstream s;s<<"[";for(int i=0;i<@COUNT@;++i){if(i)s<<",";s<<key[i];}s<<"]";return s.str();
}
static std::string cpu_json(const char *version,const char *options) {
 cpu_set_t affinity;CPU_ZERO(&affinity);if(sched_getaffinity(0,sizeof(affinity),&affinity))return "null";
 std::ostringstream a;a<<"[";bool first=true;
 for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&affinity)){if(!first)a<<",";first=false;a<<cpu;}a<<"]";
 omp_sched_t kind;int chunk;omp_get_schedule(&kind,&chunk);
 std::ostringstream environment;environment<<"{";first=true;
 @ENVIRONMENT@
 environment<<"}";
 std::ostringstream out;
 out<<"{\"affinity\":"<<a.str()<<",\"cpu_name\":"<<quoted(generated_kernels::offload::cpu_identity())
 <<",\"effective_schedule\":{\"chunk\":"<<chunk<<",\"kind\":"<<static_cast<uint32_t>(kind)<<"}"
 <<",\"environment\":"<<environment.str()<<",\"fortran_options\":"<<quoted(options)
 <<",\"fortran_version\":"<<quoted(version)<<",\"team\":["<<omp_get_max_threads()<<","<<omp_get_dynamic()
 <<","<<int(omp_get_proc_bind())<<","<<omp_get_thread_limit()<<","<<omp_get_max_active_levels()<<"]}";
 return out.str();
}
}
extern "C" void @NAME@_begin() {
 selection_state=selection_observation{};
 selection_state.active=@CAPTURE@||std::getenv("FORT_RUNTIME_TRACE");
 if(selection_state.active)selection_state.start=std::chrono::steady_clock::now();
}
static void @OBSERVER@(const int64_t *key,const std::string &identity) {
 if(!selection_state.active)return;
 selection_state.keyed=true;selection_state.key=key_json(key);selection_state.identity=identity;
}
extern "C" void @NAME@_key(const int64_t *key,const char *version,const char *options) {
 if(!key||!version||!options)return;
 for(int i=0;i<@COUNT@;++i)selection_state.values[i]=key[i];
 selection_state.version=version;selection_state.options=options;selection_state.has_key=true;
}
extern "C" int @NAME@_choose() {
 if(selection_state.decision>=0)return selection_state.decision;
 if(!selection_state.has_key)return selection_state.decision=0;
 return selection_state.decision=@SELECT@;
}
extern "C" void @NAME@_fallback() {
 ++selection_state.fallbacks;
 selection_state.decision=0;
}
extern "C" void @NAME@_native(const int64_t *key,const char *version,const char *options) {
 if(!selection_state.active||!key||!version||!options)return;
 if(!fort_selection_workload_matches(key))return;
 selection_state.keyed=true;selection_state.key=key_json(key);selection_state.cpu=cpu_json(version,options);
}
extern "C" void @NAME@_close(int64_t context) {
 if(!selection_state.active||!context)return;
 fort_scope_stats stats{};
 if(fort_scope_stats_get(context,&stats)!=FORT_SCOPE_OK){selection_state.keyed=false;++selection_state.fallbacks;return;}
 selection_state.launches+=stats.launches;
}
extern "C" void @NAME@_end() {
 if(!selection_state.active)return;
 const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-selection_state.start).count();
 const char *outcome=!selection_state.keyed?"unkeyed":
   (@NATIVE@?"native":(selection_state.fallbacks?"fallback":"resident"));
 std::fprintf(stderr,"FORT_SELECTION_OBSERVATION {\"selection\":\"@SELECTION@\",\"key\":%s,\"identity\":%s,\"cpu_identity\":%s,\"launches\":%llu,\"fallbacks\":%llu,\"complete\":true,\"outcome\":\"%s\",\"seconds\":%.17g}\n",
 selection_state.key.c_str(),selection_state.identity.c_str(),selection_state.cpu.c_str(),
 static_cast<unsigned long long>(selection_state.launches),static_cast<unsigned long long>(selection_state.fallbacks),
 outcome,seconds);
 selection_state.active=false;
}
'''
    for token, value in {"COUNT": str(count), "NAME": name, "OBSERVER": observe,
                         "ENVIRONMENT": "\n".join(environment), "CAPTURE": "true" if capture else "false",
                         "SELECTION": record["contract"]["selection_identity"], "NATIVE": "true" if native else "false",
                         "SELECT": ("0" if native else name + "(selection_state.values,selection_state.version.c_str(),selection_state.options.c_str())")}.items():
        source = source.replace("@" + token + "@", value)
    return source


def verify_capture_builds(request):
    """Public collector preflight; JSON records never supply execution proofs."""
    from compiler.offload.application_build import (
        application_build_validation,
        validate_application_build_observation,
        validate_diagnostic_relink,
    )
    from compiler.offload.selection_validation import _json, normalize_policy

    contract = _json(request["contract"])
    if contract.get("protocol") != PROTOCOL or contract.get("schema_version") != 2:
        raise ValueError("capture collection requires the current checked-interval contract")
    _acceptance(contract.get("acceptance"))
    policy = normalize_policy(contract["validation_policy"], build_contract=contract["build_contract"])
    if any(type(request.get(name)) is not dict or set(request[name]) != {"native", "resident"}
           for name in ("builds", "diagnostics")):
        raise ValueError("both application and diagnostic capture variants are required")
    variants = {}
    with application_build_validation():
        for mode in ("native", "resident"):
            build = request["builds"][mode]
            validate_application_build_observation(build, contract=contract["build_contract"], mode=mode,
                                                  selection_contract_identity=digest(contract))
            attestation = _json({"path": build["attestation"], "sha256": build["attestation_sha256"]})
            diagnostic = validate_diagnostic_relink(request["diagnostics"][mode], application_build=build,
                contract=contract["build_contract"], driver=policy["driver"], mode=mode)
            variants[mode] = {"application": {"binary": attestation["binary"], "binary_sha256": build["binary_sha256"]},
                              "diagnostic": {key: diagnostic[key] for key in ("binary", "binary_sha256")}}
    return {"status": "verified", "contract_sha256": digest(contract), "variants": variants,
            "placement_authority": False}


def evaluate_capture_evidence(request):
    """Canonicalize observed CPU options, then independently verify all evidence.

    The exact contract and resident execution identity are left untouched. This
    uses the same Fortran-option normalization as raw-log numerical validation;
    it does not change the authenticated compiler argv or build recipes.
    """
    from compiler.offload.collective_calibration import normalize_fortran_options
    from compiler.offload.selection_validation import _json

    contract, evidence = _json(request["contract"]), _json(request["evidence"])
    for campaign in evidence.get("campaigns", ()):
        if type(campaign) is not dict:
            continue
        for row in campaign.get("observations", ()):
            if type(row) is not dict:
                continue
            cpu = row.get("observed_cpu_identity")
            if type(cpu) is dict and type(cpu.get("fortran_options")) is str:
                cpu["fortran_options"] = normalize_fortran_options(cpu["fortran_options"])
    decision = process_evidence_decision(evidence, contract)
    return {"status": "verified" if decision["available"] else "rejected", "decision": decision,
            "evidence": evidence, "placement_authority": False}


def main(argv=None):
    """Verify collector inputs through the explicitly selected compiler CLI."""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("builds", "evaluate"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.request.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("collector request exceeds the supported bound")
        raw = args.request.read_bytes()
        request = json.loads(raw)
        if type(request) is not dict:
            raise ValueError("collector request must be a JSON object")
        result = (verify_capture_builds(request) if args.phase == "builds" else evaluate_capture_evidence(request))
        result["identity"] = digest(result)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"status": result["status"], "identity": result["identity"]}))
        if result["status"] != "verified":
            print(result["decision"]["reason"], file=sys.stderr)
            return 1
        return 0
    except (CompilationError, OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
