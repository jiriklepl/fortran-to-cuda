"""Measured complete-process placement for one checked original numerical group.

The checked plan remains the execution authority.  This module derives a reached
workload key from its prepared numerical IR and attaches complete-process paired
observations through the existing exact-window issuer.  Empirical applicability
does not assert a future invocation count or price cold startup using warm calls.
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
    _request_snapshot,
    digest,
    evidence_decision,
    implementation_identity,
)

PROTOCOL = "checked-selection-process-v1"
PREFIX = "FORT_SELECTION_OBSERVATION "


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
    """Reconstruct complete-process acceptance and every usable reached key.

    Every campaign is one explicit observed workload class. A key observed in an
    unprofitable campaign is not admitted merely because another campaign wins.
    The native observations have an independently observed CPU/build identity;
    resident observations must report actual completed launches, not a request.
    """
    try:
        from compiler.offload.application_build import validate_application_build_observation

        if (type(evidence) is not dict or evidence.get("protocol") != PROTOCOL
                or type(evidence.get("campaigns")) is not list or not evidence["campaigns"]):
            raise ValueError("complete-process selection calibration missing")
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
            key_sequence, all_keys, per_key_outcomes = None, set(), {}
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
                validate_application_build_observation(row.get("build"), contract=contract["build_contract"], mode=mode)
                invocations = row.get("invocations")
                if type(invocations) is not list or not invocations:
                    raise ValueError("complete-process observation lacks actual selected invocations")
                sequence, summed = [], 0.0
                for ordinal, invocation in enumerate(invocations):
                    if (type(invocation) is not dict or invocation.get("ordinal") != ordinal
                            or type(invocation.get("ordinal")) is not int
                            or invocation.get("selection") != contract["selection_identity"]
                            or not _valid_key(invocation.get("key"), count)
                            or type(invocation.get("launches")) is not int or invocation["launches"] < 0
                            or type(invocation.get("seconds")) not in {int, float}
                            or not math.isfinite(invocation["seconds"]) or invocation["seconds"] <= 0):
                        raise ValueError("selected invocation key/outcome/timing is invalid")
                    key = tuple(invocation["key"])
                    sequence.append(key)
                    all_keys.add(key)
                    summed += invocation["seconds"]
                    if mode == "native":
                        if invocation.get("outcome") != "native" or invocation["launches"] != 0:
                            raise ValueError("independent native process executed a resident alternative")
                    else:
                        resident = (invocation.get("outcome") == "resident"
                                    and invocation["launches"] == contract["expected_launches"])
                        fallback = invocation.get("outcome") == "native" and invocation["launches"] == 0
                        if not resident and not fallback:
                            raise ValueError("selected resident outcome is incomplete or mixed")
                        per_key_outcomes.setdefault(key, []).append(resident)
                if summed > row["seconds"]:
                    raise ValueError("selected invocation times exceed complete-process elapsed time")
                if key_sequence is None:
                    key_sequence = sequence
                    first_key = list(sequence[0])
                elif sequence != key_sequence:
                    raise ValueError("paired complete-process selection key sequence changed")
                # Reuse the strict original raw-pair order, hardware, independent
                # CPU, float64 agreement and complete elapsed-time checks. Keys
                # are authenticated above as a sequence, not silently collapsed.
                projected["observations"][order]["key"] = first_key
                projected["observations"][order]["observed_key"] = first_key
            projected["key"] = first_key
            decision = evidence_decision(projected, contract, protocol=PROTOCOL)
            if not decision["available"]:
                raise ValueError(decision["reason"])
            identity = decision["identity"]
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
                              "selection": decision["selection"]})
        if any(campaign["selection"] != "resident" for campaign in campaigns):
            rejected.update(accepted)
        entries = [{"key": list(key), "identity": identities[identity], "workload": workloads[workload]}
                   for key in sorted(accepted.keys() - rejected) for identity, workload in sorted(accepted[key])]
        if len(identities) != 1:
            raise ValueError("complete-process campaigns require one compatible execution environment")
        return {"available": True, "selection": "resident" if entries else "native",
                "reason": ("measured complete-process selection advantage" if entries else
                           "complete-process selection saving not established"),
                "accepted_keys": entries, "campaigns": campaigns, "evidence_sha256": digest(evidence),
                "applicability": "empirical observed complete-process workload classes; no future invocation-count guarantee"}
    except (CompilationError, KeyError, TypeError, ValueError, OverflowError, OSError) as error:
        return {"available": False, "selection": "native", "reason": str(error)}


def _workload_key(owner, unit):
    """Derive controls from the same lowered, prepared numerical computation."""
    from compiler.ir.nodes import (
        ArrayAccess,
        Assignment,
        Binary,
        If,
        IntrinsicCall,
        Literal,
        Loop,
        Reference,
        Size,
        Unary,
    )

    prepared = unit.scope.prepared
    extraction = prepared.extraction
    if extraction.operation_kind != "numerical_loop" or not prepared.all_runtime_guards_typed:
        raise CompilationError("selected pricing requires a complete typed original numerical group")
    if extraction.scalar_element_captures or extraction.immutable_array_reads:
        raise CompilationError("selected pricing lacks a workload contract for indirect scalar captures")
    if prepared.scalar_changes:
        raise CompilationError("selected pricing requires unchanged external scalar controls")
    parameters = {symbol.name.lower(): symbol for symbol in prepared.function.parameters}
    integer_parameters = {symbol.id for symbol in parameters.values() if not symbol.rank and symbol.dtype.value == "integer"}
    needed, visits = set(), 0

    def control(expression, iterators):
        nonlocal visits
        visits += 1
        if visits > 65536:
            raise CompilationError("selected workload analysis budget exhausted")
        if type(expression) is int or isinstance(expression, Literal) and expression.dtype.value == "integer":
            return
        if isinstance(expression, Reference):
            if expression.symbol.id in iterators:
                return
            if expression.symbol.id in integer_parameters:
                needed.add(expression.symbol.name.lower())
                return
        if isinstance(expression, Size):
            return
        if isinstance(expression, Unary) and expression.operator in {"+", "-", ".not."}:
            control(expression.operand, iterators)
            return
        if isinstance(expression, Binary) and expression.operator.lower() in {
                "+", "-", "*", "/", "<", ">", "<=", ">=", "==", "/=", ".and.", ".or."}:
            control(expression.left, iterators)
            control(expression.right, iterators)
            return
        if isinstance(expression, IntrinsicCall) and expression.name.lower() in {"min", "max", "int"}:
            for argument in expression.arguments:
                control(argument, iterators)
            return
        raise CompilationError("selected workload control depends on payload, modified private state or unsupported work")

    def expression_indices(expression, iterators):
        from compiler.ir.nodes import walk_expr

        for node in walk_expr(expression):
            if isinstance(node, ArrayAccess):
                for index in node.indices:
                    control(index, iterators)

    def block(body, iterators=frozenset()):
        for statement in body.statements:
            if isinstance(statement, Loop):
                for value in (statement.lower, statement.upper, statement.step):
                    control(value, iterators)
                block(statement.body, iterators | {statement.iterator.id})
            elif isinstance(statement, If):
                control(statement.condition, iterators)
                block(statement.then_body, iterators)
                block(statement.else_body, iterators)
            elif isinstance(statement, Assignment):
                expression_indices(statement.target, iterators)
                expression_indices(statement.value, iterators)
            else:
                raise CompilationError("selected workload has unsupported prepared numerical work")

    # PreparedOriginalRegion.function is the fully lowered helper closure, not
    # just the outer source body: calls have been inlined into this typed IR.
    # Its control walk therefore rejects variable helper work as well. Tiling
    # normalization can remove tile steps from numerical IR, so retain every
    # external INTEGER parameter of the authenticated extraction too.
    block(prepared.function.body)
    needed.update(symbol.name.lower() for symbol in parameters.values()
                  if not symbol.rank and symbol.dtype.value == "integer")
    binding_by_root = {binding.root: binding for binding in extraction.bindings}
    key, expressions, guards, seen = [], [], [], set()
    for parameter in extraction.parameters:
        binding = binding_by_root.get(parameter.resource)
        if binding is None:
            raise CompilationError("selected workload parameter lacks exact original binding")
        canonical = owner.canonical(binding)
        root = canonical.root
        if root in seen or "parameter" in binding.attributes:
            continue
        name = owner.builder.visible(owner.routine, binding.root)
        if binding.attributes & {"optional", "pointer", "volatile", "asynchronous"}:
            raise CompilationError("selected workload requires stable present original descriptors and controls")
        if binding.rank:
            if binding.dtype != "real" or binding.kind not in {4, 8}:
                raise CompilationError("selected workload descriptor requires a real numerical array")
            if "allocatable" in binding.attributes:
                guards.append(f"allocated({name})")
            guards.append(f"is_contiguous({name})")
            for axis in range(1, binding.rank + 1):
                for inquiry in ("size", "lbound"):
                    key.append({"resource": root, "property": inquiry, "axis": axis})
                    expressions.append(f"{inquiry}({name},{axis},kind=c_int64_t)")
        elif parameter.name.lower() in needed:
            if binding.dtype != "integer" or binding.kind != 4:
                raise CompilationError("selected workload requires original default INTEGER controls")
            if (binding.declaring_scope is owner.routine.scope and binding.name not in owner.routine.arguments
                    and not any(binding is item for item in owner.closed_initializations)):
                raise CompilationError("selected local workload control lacks an original reached definition")
            if binding.name in owner.routine.arguments and binding.intent != "in":
                raise CompilationError("selected dummy workload control requires original INTENT(IN)")
            key.append({"resource": root, "property": "integer_value"})
            expressions.append(f"int({name},c_int64_t)")
        else:
            continue
        seen.add(root)
    from compiler.frontend.native_private import original_private_array

    private_arrays = unit.scope.completion.public().get("original_private_arrays", ())
    declarations = {binding.root: binding for binding in owner.routine.scope.bindings.values()}
    for root in sorted({item["resource"] for item in private_arrays}):
        if root in seen:
            continue
        binding = declarations.get(root)
        if binding is None:
            raise CompilationError("selected original PRIVATE descriptor unavailable")
        original_private_array(owner.routine, binding)
        for axis in range(1, binding.rank + 1):
            for inquiry in ("size", "lbound"):
                key.append({"resource": root, "property": inquiry, "axis": axis})
                expressions.append(f"{inquiry}({binding.name},{axis},kind=c_int64_t)")
    if not key or len(key) > 256:
        raise CompilationError("selected workload descriptor/control inventory unavailable")
    return key, expressions, guards


def prepare(owner):
    """Price an already checked closed original group; never widen admission."""
    from compiler.emission.common.resources import read_common_header
    from compiler.frontend.immutable_graph import freeze_graph
    from compiler.numerical_contract import numerical_build_contract
    from compiler.offload.application_build import require_application_build_contract
    from compiler.scopes.combined_team import CombinedNumericalTeam

    builder = owner.builder
    request = deepcopy(builder.config.exact_window)
    snapshot = _request_snapshot(request)
    plan = owner.execution_plan.validate()
    units = plan.units(owner)
    if (not getattr(owner, "closed_selection", None) or len(plan.members) != 1
            or len(units) != 1 or not isinstance(units[0].scope, CombinedNumericalTeam)
            or owner.parent is not None or owner.boundaries or units[0].scope.native):
        raise CompilationError("selected pricing requires one checked closed original numerical team")
    variant = request.get("calibration")
    if variant is not None and variant not in {"native", "resident"}:
        raise CompilationError("selected calibration requires an explicit native or resident artifact")
    if builder.config.policy != "auto":
        raise CompilationError("selected calibration and production require identical automatic numerical options")
    key, expressions, guards = _workload_key(owner, units[0])
    workloads = request.get("workloads")
    if (type(workloads) is not list or not workloads or len(workloads) > 32
            or any(not _valid_workload(workload) for workload in workloads)
            or len({workload["identity"] for workload in workloads}) != len(workloads)):
        raise CompilationError("selected calibration requires explicit runtime workload input identities")
    public = plan.public()
    workers, artifacts = {}, {}
    for leaf in units[0].leaves:
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
    contract = {
        "schema_version": 1, "protocol": PROTOCOL, "entry": owner.routine.qualified,
        "sources": dict(sorted(builder.analysis.sources.items())), "selection": deepcopy(owner.selection_context),
        "selection_bundle": deepcopy(owner.selection_bundle),
        "workloads": workloads,
        "selection_identity": digest(owner.selection_context), "checked_execution_identity": plan.execution_identity,
        "workers": workers, "runtime": builder.runtime["runtime_id"], "implementation": implementation,
        "common_functions_sha256": sha256(read_common_header().encode()).hexdigest(),
        "source_inputs": builder.analysis.inputs.identity(), "facts": builder.facts,
        "compiler_options": asdict(builder.options), "numerical_contract": numerical_build_contract(),
        "build_contract": build, "threads": builder.config.host_threads,
        "original_participation": units[0].scope.completion.public(),
        "transfers": builder.config.scope_transfers, "device_budget_bytes": builder.device_budget, "key": key,
        "expected_launches": len(units[0].scope.prepared.offload.units),
        "coverage": "complete application process; reached group timing includes controls, original team, transfers and cleanup",
        "applicability": "empirical observed workload class; exact reached keys; no future invocation-count guarantee",
    }
    if contract["expected_launches"] < 1:
        raise CompilationError("selected pricing requires a nonempty checked numerical launch inventory")
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


def control(owner):
    """Start the complete-group observer before preparation; never read a key."""
    name, record = _names(owner)
    if not record["capture"] and record["selection"] != "resident":
        return [], [owner.enabled + " = .false."]
    _publish(owner)
    return [*_imports(), *_interface(name + "_begin")], [f"call {name}_begin()"]


def choice(owner, unit, handles, values):
    """Inspect descriptors only at the checked reached master preparation point."""
    del handles, values
    name, record = _names(owner)
    if unit.scope is not owner.execution_plan.units(owner)[0].scope:
        raise CompilationError("selected workload choice requires the exact checked numerical group")
    count = len(record["expressions"])
    # The original team has already validated its environment, descriptors,
    # typed activation preconditions and complete numerical query. No key is
    # evaluated for a rejected/inactive group. Block locals are master-private.
    lines = ["block", *_imports(), "interface",
             f"function {name}(key,version,options) bind(C) result(selected)",
             "import c_int,c_int64_t,c_char", f"integer(c_int64_t), intent(in) :: key({count})",
             "character(c_char), intent(in) :: version(*),options(*)", "integer(c_int) :: selected",
             "end function", "end interface", f"integer(c_int64_t) :: fort_selection_key({count})"]
    lines += [f"fort_selection_key({index}) = {expression}" for index, expression in enumerate(record["expressions"], 1)]
    from copy import copy

    close_owner = copy(owner)
    close_owner.context, close_owner.enabled, close_owner.control_guard = "fort_team_context", "fort_team_enabled", None
    lines += [f"if ({name}(fort_selection_key,fort_version()//c_null_char,fort_options()//c_null_char) == 0) then",
              *close_owner.close(), "fort_native_ready = .false.", "endif", "end block"]
    return lines


def before_close(owner, context):
    """Retain actual launches before destruction; final elapsed time includes close."""
    name, record = _names(owner)
    if not record["capture"] and record["selection"] != "resident":
        return []
    return ["block", "use iso_c_binding, only: c_int64_t,c_char",
            *_interface(name + "_close", "context", ("integer(c_int64_t), value :: context",)),
            f"call {name}_close({context})", "end block"]


def finish(owner):
    name, record = _names(owner)
    if not record["capture"] and record["selection"] != "resident":
        return []
    return ["block", "use iso_c_binding, only: c_int64_t,c_char", *_interface(name + "_end"),
            f"call {name}_end()", "end block"]


def native_control(owner):
    """Independent native artifact: original group, CPU identity, no CUDA query."""
    from compiler.scopes.numerical_preflight import preparation_preflight

    name, record = _names(owner)
    _publish(owner)
    unit, = owner.execution_plan.units(owner)
    call, = unit.scope.calls
    specification, preflight = preparation_preflight(owner.builder, call, {})
    count = len(record["expressions"])
    declarations = [*_imports(), "use fort_scoped_memory, only: fort_scope_numerical_environment_supported",
                    *_interface(name + "_begin"), *_interface(name + "_end"),
                    *_interface(name + "_native", "key,version,options", (
                        f"integer(c_int64_t), intent(in) :: key({count})",
                        "character(c_char), intent(in) :: version(*),options(*)")),
                    "logical :: fort_numerical_guard", f"integer(c_int64_t) :: fort_selection_key({count})",
                    *specification]
    # Presence/contiguity checks are ordered statements, never Fortran logical
    # conjunctions whose operands could be evaluated speculatively.
    entry = [f"call {name}_begin()", "fort_numerical_guard = .true."]
    entry += [f"if (fort_numerical_guard) fort_numerical_guard = {guard}" for guard in record["guards"]]
    entry += ["if (fort_numerical_guard) then", *preflight, "endif", "if (fort_numerical_guard) then"]
    entry += [f"fort_selection_key({index}) = {expression}" for index, expression in enumerate(record["expressions"], 1)]
    entry += [f"call {name}_native(fort_selection_key,fort_version()//c_null_char,fort_options()//c_null_char)", "endif"]
    return declarations, entry, [f"call {name}_end()"]


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
        source = source.replace("omp_get_level()!=0", "omp_get_level()!=1||omp_get_num_threads()!=@THREADS@")
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
 bool active=false, keyed=false; uint64_t launches=0;
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
extern "C" void @NAME@_native(const int64_t *key,const char *version,const char *options) {
 if(!selection_state.active||!key||!version||!options)return;
 if(!fort_selection_workload_matches(key))return;
 selection_state.keyed=true;selection_state.key=key_json(key);selection_state.cpu=cpu_json(version,options);
}
extern "C" void @NAME@_close(int64_t context) {
 if(!selection_state.active||!context)return;
 fort_scope_stats stats{};
 if(fort_scope_stats_get(context,&stats)!=FORT_SCOPE_OK){selection_state.keyed=false;return;}
 selection_state.launches+=stats.launches;
}
extern "C" void @NAME@_end() {
 if(!selection_state.active)return;
 const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-selection_state.start).count();
 std::fprintf(stderr,"FORT_SELECTION_OBSERVATION {\"selection\":\"@SELECTION@\",\"key\":%s,\"identity\":%s,\"cpu_identity\":%s,\"launches\":%llu,\"outcome\":\"%s\",\"seconds\":%.17g}\n",
 selection_state.key.c_str(),selection_state.identity.c_str(),selection_state.cpu.c_str(),
 static_cast<unsigned long long>(selection_state.launches),selection_state.keyed?(selection_state.launches?"resident":"native"):"unkeyed",seconds);
 selection_state.active=false;
}
'''
    for token, value in {"COUNT": str(count), "NAME": name, "OBSERVER": observe,
                         "ENVIRONMENT": "\n".join(environment), "CAPTURE": "true" if capture else "false",
                         "SELECTION": record["contract"]["selection_identity"]}.items():
        source = source.replace("@" + token + "@", value)
    return source
