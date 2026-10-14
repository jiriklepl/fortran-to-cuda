"""Measured, source-bound alternatives for a complete original lexical window.

This deliberately has no per-item coefficients or size interpolation. Its first
admission class is a complete branchless owner with stable array formals,
integer input controls, and original scalar outputs. The existing compiler still
proves every numerical operation, effect and definition. Costs authorize only
placement. Original joined teams require independent completion proofs and
whole-invocation observations. Missing/mismatching evidence leaves the original
owner native.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from compiler.ir import CompilationError
from compiler.offload.collective_calibration import normalize_fortran_options

VERSION = 2
PROTOCOL = "complete-original-window-cold-v2"
SAMPLES = 7
HOLDOUTS = 2
ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_WAIT_POLICY",
    "GOMP_SPINCOUNT",
    "OMP_SCHEDULE",
    "OMP_THREAD_LIMIT",
    "OMP_MAX_ACTIVE_LEVELS",
    "OMP_NESTED",
    "OMP_PLACES",
    "GOMP_CPU_AFFINITY",
    "CUDA_VISIBLE_DEVICES",
)


def build_contract(architecture="sm_86"):
    import re

    from compiler.numerical_contract import cuda_compile_options

    if not isinstance(architecture, str) or re.fullmatch(r"sm_[0-9]{2,3}", architecture) is None:
        raise CompilationError("exact-window requires an explicit CUDA architecture")
    return {
        "schema_version": 1,
        "cuda_options": ["-O3", "-std=c++17", "-Xcompiler=-fopenmp", *cuda_compile_options(), "-arch=" + architecture],
        "fortran_options": ["-O3", "-std=f2018", "-fopenmp", "-ffp-contract=off"],
        "consumer": "independent exact_window_calibration producer only",
        "attestation": "selector requires FORT_EXACT_WINDOW_BUILD_<contract_sha256>=1 after exact option validation",
    }


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def implementation_identity():
    root = Path(__file__).parents[1]
    # Cost reuse is intentionally conservative. A complete compiler inventory
    # avoids an expanding, silently incomplete hand-maintained dependency list.
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
        if "tests" not in path.relative_to(root).parts
    }


def evidence_decision(evidence, contract):
    """Reconstruct acceptance from raw paired observations; no trusted flag."""
    try:
        if not isinstance(evidence, dict) or evidence.get("protocol") != PROTOCOL:
            raise ValueError("complete-window calibration missing")
        if evidence.get("contract") != contract:
            raise ValueError("complete-window source/plan/implementation identity differs")
        key, identity, rows = evidence["key"], evidence["identity"], evidence["observations"]
        if (
            not isinstance(key, list)
            or len(key) != len(contract["key"])
            or any(type(value) is not int or not -(2**63) <= value < 2**63 for value in key)
        ):
            raise ValueError("complete-window descriptor/control key invalid")
        if not isinstance(identity, dict) or set(identity) != {
            "gpu_uuid",
            "gpu_name",
            "cpu_name",
            "cuda_runtime",
            "cuda_driver",
            "host_compiler",
            "cuda_compiler",
            "fortran_version",
            "fortran_options",
            "affinity",
            "threads",
            "dynamic",
            "proc_bind",
            "thread_limit",
            "max_active_levels",
            "environment",
        }:
            raise ValueError("complete-window execution identity incomplete")
        for field in (
            "gpu_uuid",
            "gpu_name",
            "cpu_name",
            "host_compiler",
            "cuda_compiler",
            "fortran_version",
            "fortran_options",
        ):
            if not isinstance(identity[field], str) or not identity[field] or len(identity[field]) > 16384:
                raise ValueError("complete-window hardware/toolchain identity invalid")
        if (
            not isinstance(identity["affinity"], list)
            or not isinstance(identity["environment"], dict)
            or any(value is not None and not isinstance(value, str) for value in identity["environment"].values())
        ):
            raise ValueError("complete-window placement identity invalid")
        if (
            identity["threads"] != contract["threads"]
            or identity["dynamic"] != 0
            or identity["proc_bind"] != 0
            or type(identity["thread_limit"]) is not int
            or identity["thread_limit"] < contract["threads"]
            or type(identity["max_active_levels"]) is not int
            or identity["max_active_levels"] < 1
            or not identity["affinity"]
            or any(type(cpu) is not int or not 0 <= cpu < 1024 for cpu in identity["affinity"])
            or identity["affinity"] != sorted(set(identity["affinity"]))
            or set(identity["environment"]) != set(ENVIRONMENT)
        ):
            raise ValueError("complete-window execution placement differs")
        if type(rows) is not list or len(rows) != 2 * (SAMPLES + HOLDOUTS):
            raise ValueError("complete-window requires seven pairs and two independent holdouts")
        expected_cpu = {
            "cpu_name": identity["cpu_name"],
            "fortran_version": identity["fortran_version"],
            "fortran_options": normalize_fortran_options(identity["fortran_options"]),
            "team": [
                identity["threads"],
                identity["dynamic"],
                identity["proc_bind"],
                identity["thread_limit"],
                identity["max_active_levels"],
            ],
            "affinity": identity["affinity"],
            "environment": identity["environment"],
        }
        values = {"native": [], "resident": []}
        for order, row in enumerate(rows):
            if type(row) is not dict:
                raise ValueError("complete-window observation must be an object")
            round_index, turn = divmod(order, 2)
            mode = ("native", "resident")[(round_index + turn) % 2]
            if (
                row.get("round") != round_index
                or row.get("mode") != mode
                or row.get("role") != ("fit" if round_index < SAMPLES else "holdout")
                or row.get("key") != key
                or row.get("identity") != identity
                or row.get("agreement") is not True
                or row.get("observed_cpu_identity") != expected_cpu
                or row.get("observed_key") != key
            ):
                raise ValueError("complete-window observation order/identity/agreement differs")
            field_hash = row.get("fields_sha256")
            if (
                not isinstance(field_hash, str)
                or len(field_hash) != 64
                or any(char not in "0123456789abcdef" for char in field_hash)
                or order % 2
                and field_hash != rows[order - 1].get("fields_sha256")
            ):
                raise ValueError("complete-window full-field comparison differs")
            elapsed = row.get("seconds")
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("complete-window time invalid")
            values[mode].append(float(elapsed))
        # The complete resident interval includes all planning, guards, native
        # interruptions, transfers, publication and close; never add them again.
        native = statistics.median(values["native"][:SAMPLES])
        resident = statistics.median(values["resident"][:SAMPLES])
        profitable = max(values["resident"]) < min(values["native"]) * 0.95
        return {
            "available": True,
            "selection": "resident" if profitable else "native",
            "reason": "measured complete-window advantage" if profitable else "complete-window saving not established",
            "native_seconds": native,
            "resident_seconds": resident,
            "key": key,
            "identity": identity,
            "evidence_sha256": digest(evidence),
        }
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return {"available": False, "selection": "native", "reason": str(error)}


def original_team_contract(owner):
    """Authenticate complete original teams before pricing their execution.

    Only top-level joined groups are admitted here. Their completion proofs
    preserve private state, worksharing and final joins; calibration measures
    the actual native groups rather than borrowing serial or per-loop rates.
    This does not provide numerical or native-section execution authority.
    """
    from fparser.two.utils import walk

    from compiler.frontend.source_effects import _children
    from compiler.scopes.segments import directive, grouped_nodes

    builder, routine = owner.builder, owner.routine
    covered, teams = set(), []
    for group in grouped_nodes(_children(routine.execution)):
        if not isinstance(group, tuple):
            continue
        directives = [node for node in walk(group) if directive(node) is not None]
        if not directives:
            continue
        proof = builder.analysis.joined_completion(routine.qualified, group)
        _, selection = builder.analysis._selected_source(routine.qualified, group)
        proof.validate(builder.analysis, routine.qualified, selection)
        teams.append(proof.public())
        covered.update(map(id, directives))
    if any(directive(node) is not None and id(node) not in covered for node in walk(routine.execution)):
        raise CompilationError("exact-window directive lacks a complete original joined-team proof")
    return {
        "kind": "original_joined_teams" if teams else "serial",
        "teams": teams,
        "coverage": "complete invocation; original team startup, worksharing and final joins included",
    }


def prepare(owner):
    """Bind one admitted complete owner, without extending source legality."""
    from fparser.two.utils import walk

    from compiler.frontend.source_effects import _kind

    builder, routine = owner.builder, owner.routine
    request = builder.config.exact_window
    if request is None:
        return None
    if owner.parent is not None or len(owner.members) != 1 or owner.boundaries:
        raise CompilationError("exact window requires one complete owner without child owners or closures")
    if getattr(builder, "_reached_window_binding", None) is not None:
        raise CompilationError("exact window currently requires the complete original invocation")
    if not owner.units or sum(len(unit.scope.calls) for unit in owner.units) < 2:
        raise CompilationError("exact window requires at least two admitted numerical operations")
    participation = original_team_contract(owner)
    # Payload-dependent branches, calls and changing lifetimes need independent
    # coverage. They are not inferred from descriptor or source identity.
    forbidden = {
        "If_Construct",
        "If_Stmt",
        "Select_Case_Construct",
        "Case_Construct",
        "Call_Stmt",
        "Allocate_Stmt",
        "Deallocate_Stmt",
        "Pointer_Assignment_Stmt",
        "Return_Stmt",
        "Exit_Stmt",
        "Cycle_Stmt",
        "Where_Construct",
        "Where_Stmt",
    }
    nodes = tuple(walk(routine.execution))
    controls = {
        name
        for name in routine.arguments
        if routine.scope.bindings[name].intent == "in"
        and not routine.scope.bindings[name].rank
        and routine.scope.bindings[name].dtype == "integer"
    }
    iterators = {
        str(node.items[1][0]).lower() for node in nodes if _kind(node) == "Loop_Control" and node.items[1] is not None
    }

    def fixed_control(expression, enclosing=()):
        # No data-dependent bounds/indexing, even through a private temporary.
        # All accepted varying names are guarded integer IN controls or source
        # loop indices. Constant parameter identities belong to source hashes.
        for item in walk(expression):
            kind = _kind(item)
            if kind in {"Part_Ref", "Function_Reference", "Intrinsic_Function_Reference", "Data_Ref"}:
                raise CompilationError("exact-window payload-dependent or indirect indexing/control unavailable")
            if kind == "Name":
                name = str(item).lower()
                binding = builder.analysis._binding(routine.scope, name)
                if name not in controls | set(enclosing) and not (
                    binding is not None and binding.dtype == "integer" and "parameter" in binding.attributes
                ):
                    raise CompilationError("exact-window control requires guarded integer inputs or loop indices")
            if kind in {"Mult_Operand", "Add_Operand", "Level_2_Expr"} and len(item.items) == 3:
                left, operator, right = item.items
                if str(operator) not in {"+", "-", "*"} or (
                    str(operator) == "*"
                    and _kind(left) != "Int_Literal_Constant"
                    and _kind(right) != "Int_Literal_Constant"
                ):
                    raise CompilationError("exact-window controls must be affine")

    for node in nodes:
        if _kind(node) in forbidden:
            raise CompilationError("exact-window control/effect coverage unavailable: " + _kind(node))
        if _kind(node) == "Loop_Control":
            if node.items[0] is not None or node.items[1] is None:
                raise CompilationError("exact-window requires fixed counted loops")
            fixed_control(node.items[1][1])
        if _kind(node) == "Part_Ref":
            binding = builder.analysis._binding(routine.scope, str(node.items[0]).lower())
            if binding is None or not binding.rank:
                raise CompilationError("exact-window expression helper coverage unavailable")
            enclosing, ancestor = set(), getattr(node, "parent", None)
            while ancestor is not None:
                if _kind(ancestor) == "Block_Nonlabel_Do_Construct":
                    header = ancestor.content[0]
                    control = header.items[1]
                    if control is not None and control.items[1] is not None:
                        enclosing.add(str(control.items[1][0]).lower())
                ancestor = getattr(ancestor, "parent", None)
            fixed_control(node.items[1], enclosing)
        if _kind(node) == "Intrinsic_Function_Reference" and str(node.items[0]).lower() not in {
            "sum",
            "real",
            "int",
            "size",
            "lbound",
            "ubound",
        }:
            raise CompilationError("exact-window primitive input-domain coverage unavailable")
        if (
            _kind(node) == "Intrinsic_Function_Reference"
            and str(node.items[0]).lower() == "sum"
            and len(node.items[1].items) != 1
        ):
            raise CompilationError("exact-window reduction mask/dimension coverage unavailable")
        if (
            _kind(node) == "Assignment_Stmt"
            and _kind(node.items[0]) == "Name"
            and str(node.items[0]).lower() in iterators
        ):
            raise CompilationError("exact-window loop indices may not be assigned outside their DO controls")
        if (
            _kind(node) in {"Mult_Operand", "Add_Operand"}
            and len(node.items) == 3
            and str(node.items[1]) in {"/", "**"}
        ):
            raise CompilationError("exact-window primitive input-domain coverage unavailable")
    if any(
        call.region is None
        or call.region.numerical_helpers
        or call.region.tile_domains
        or call.region.requires_numerical_environment
        for unit in owner.units
        for call in unit.scope.calls
    ):
        raise CompilationError("exact-window helper/tile/numerical-environment coverage unavailable")
    key, expressions, guards = [], [], []
    arrays = set()
    for name in routine.arguments:
        binding = routine.scope.bindings[name]
        if binding.attributes & {"optional", "pointer", "allocatable", "volatile", "asynchronous", "value"}:
            raise CompilationError("exact-window requires present stable ordinary formals")
        if binding.rank:
            if binding.dtype != "real" or binding.kind not in {4, 8}:
                raise CompilationError("exact-window requires real array formals")
            arrays.add(binding.root)
            guards.append(f"is_contiguous({name})")
            for axis in range(1, binding.rank + 1):
                for inquiry in ("size", "lbound"):
                    key.append({"resource": binding.root, "property": inquiry, "axis": axis})
                    expressions.append(f"{inquiry}({name},{axis},kind=c_int64_t)")
        elif binding.intent == "in":
            if binding.dtype != "integer" or binding.kind != 4:
                raise CompilationError("exact-window requires integer input controls")
            key.append({"resource": binding.root, "property": "integer_value"})
            expressions.append(f"int({name},c_int64_t)")
        elif binding.intent != "out":
            raise CompilationError("exact-window scalar state requires an independent control contract")
    # PRIVATE storage can change the native work/allocation cost even when its
    # payload is unused. Its original automatic descriptor already exists at
    # procedure entry; never re-evaluate specification expressions, whose
    # inputs may have changed since that descriptor was established. The same
    # array can belong to several original teams but needs only one key entry.
    private_arrays = {
        item["resource"]
        for team in participation["teams"]
        for item in team.get("original_private_arrays", ())
    }
    original_bindings = {binding.root: binding for binding in routine.scope.bindings.values()}
    for root in sorted(private_arrays):
        from compiler.frontend.native_private import original_private_array

        binding = original_bindings.get(root)
        if binding is None:
            raise CompilationError("exact-window original PRIVATE descriptor unavailable")
        original_private_array(routine, binding)
        for axis in range(1, binding.rank + 1):
            for inquiry in ("size", "lbound"):
                key.append({"resource": root, "property": inquiry, "axis": axis})
                expressions.append(f"{inquiry}({binding.name},{axis},kind=c_int64_t)")
    if not arrays or len(key) > 256:
        raise CompilationError("exact-window descriptor inventory unavailable")
    actual = {root for unit in owner.units for root in unit.arrays}
    if actual - arrays:
        raise CompilationError("exact-window module/local storage needs lifetime and key coverage")
    # Require source-bound workers already prepared by the normal compiler.
    workers = {}
    for unit in owner.units:
        for leaf in unit.leaves:
            generated = builder.numerical(leaf)
            if generated is None:
                raise CompilationError("exact-window numerical artifact unavailable")
            workers[leaf] = {
                name: sha256(text.encode()).hexdigest()
                for name, text in generated.artifacts.items()
                if name in {"shared_entry.cu", "shared_interface.f90"}
            }
    from compiler.emission.common.resources import read_common_header
    from compiler.numerical_contract import numerical_build_contract

    requested_build = request.get("build_contract")
    if requested_build is None and isinstance(request.get("evidence"), dict):
        evidence_contract = request["evidence"].get("contract")
        requested_build = evidence_contract.get("build_contract") if isinstance(evidence_contract, dict) else None
    architecture = "sm_86"
    if requested_build is not None:
        if (
            not isinstance(requested_build, dict)
            or not isinstance(requested_build.get("cuda_options"), list)
            or not requested_build["cuda_options"]
            or not isinstance(requested_build["cuda_options"][-1], str)
        ):
            raise CompilationError("exact-window semantic build contract invalid")
        architecture = requested_build["cuda_options"][-1].removeprefix("-arch=")
    expected_build = build_contract(architecture)
    if requested_build is not None and requested_build != expected_build:
        raise CompilationError("exact-window semantic build contract differs")
    contract = {
        "schema_version": VERSION,
        "protocol": PROTOCOL,
        "entry": routine.qualified,
        "sources": dict(sorted(builder.analysis.sources.items())),
        "source_graph": owner.graph.identity,
        "workers": workers,
        "runtime": builder.runtime["runtime_id"],
        "implementation": implementation_identity(),
        "common_functions_sha256": sha256(read_common_header().encode()).hexdigest(),
        "source_inputs": builder.analysis.inputs.identity(),
        "facts": builder.facts,
        "compiler_options": asdict(builder.options),
        "numerical_contract": numerical_build_contract(),
        "build_contract": expected_build,
        "threads": builder.config.host_threads,
        "original_participation": participation,
        "transfers": builder.config.scope_transfers,
        "device_budget_bytes": builder.device_budget,
        "key": key,
        "coverage": "one complete cold invocation including guards, transfers, native effects and publication",
        "admission": "branchless original owner with proved joined teams; stable real formals; integer controls",
    }
    capture = request.get("calibration") is True
    if capture and builder.config.policy != "auto":
        raise CompilationError("exact-window calibration uses the same automatic numerical artifacts")
    decision = (
        {"available": False, "selection": "calibration", "reason": "explicit offline calibration only"}
        if capture
        else evidence_decision(request.get("evidence"), contract)
    )
    owner.exact_window = {
        "contract": contract,
        "contract_sha256": digest(contract),
        **decision,
        "expressions": expressions,
        "guards": guards,
        "capture": capture,
    }
    return owner.exact_window


def control(owner):
    """Emit the only policy decision before original work and ownership."""
    from compiler.emission.common.resources import read_common_header

    record = owner.exact_window
    enabled = owner.enabled
    if not record["capture"] and record["selection"] != "resident":
        return [], [enabled + " = .false."]
    identity = record["contract_sha256"]
    name = "fort_exact_window_" + identity[:12]
    path = "exact/" + name + ".cu"
    owner.builder.outputs[path] = selector_source(name, record)
    owner.builder.outputs["exact/common_functions.cuh"] = read_common_header()
    imports = [
        "use iso_c_binding, only: c_int, c_int64_t, c_char, c_null_char",
        "use, intrinsic :: iso_fortran_env, only: fort_version => compiler_version, fort_options => compiler_options",
    ]
    interface = [
        "interface",
        f"function {name}(key,version,options) bind(C) result(selected)",
        "import c_int,c_int64_t,c_char",
        f"integer(c_int64_t), intent(in) :: key({len(record['expressions'])})",
        "character(c_char), intent(in) :: version(*),options(*)",
        "integer(c_int) :: selected",
        "end function",
        "end interface",
        f"integer(c_int64_t) :: fort_exact_key({len(record['expressions'])})",
    ]
    lines = [f"if ({enabled}) {enabled} = {guard}" for guard in record["guards"]]
    lines += [f"if ({enabled}) then"]
    lines += [f"fort_exact_key({index}) = {expression}" for index, expression in enumerate(record["expressions"], 1)]
    lines += [
        f"{enabled} = {name}(fort_exact_key,fort_version()//c_null_char,fort_options()//c_null_char) /= 0",
        "endif",
    ]
    return [*imports, *interface], lines


def selector_source(name, record):
    """The calibration selector records actual identities inside timed calls.

    Calibration override exists only in a separately compiled diagnostic
    artifact. Production source has no force environment variable.
    """
    expected = record.get("identity", {})
    key = record.get("key", [])
    count = len(record["expressions"])
    capture = record["capture"]
    expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    # Identity values are assembled in exactly Python's sorted-key order.
    source = r"""
#include <cuda_runtime.h>
#include <omp.h>
#include <sched.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <cstdio>
#include <iomanip>
#define FORT_OFFLOAD_ENABLED 1
#include "common_functions.cuh"
#include "scoped_runtime.h"
#include "scoped_team_observer.hpp"
static std::string quoted(const std::string &s) {
    std::ostringstream o; o << '"';
    for (unsigned char c:s) { if(c=='"'||c=='\\') o << '\\' << c;
      else if(c<32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
      else o << c; }
    o << '"'; return o.str();
}
extern "C" int @NAME@(const int64_t *key,const char *version,const char *options) {
    #if !defined(@ATTESTATION@) || @ATTESTATION@ != 1
    return 0;
    #endif
    if (!key||!version||!options||omp_get_level()!=0||omp_get_dynamic()
            ||omp_get_max_threads()!=@THREADS@||omp_get_proc_bind()!=omp_proc_bind_false
            ||omp_get_thread_limit()<@THREADS@||omp_get_max_active_levels()<1
            ||!fort_scope_numerical_environment_supported()) return 0;
    @KEYCHECK@
    @FORTRANCHECK@
    cudaDeviceProp p{}; int device=-1,runtime=0,driver=0;
    if(cudaGetDevice(&device)!=cudaSuccess||cudaGetDeviceProperties(&p,device)!=cudaSuccess
            ||cudaRuntimeGetVersion(&runtime)!=cudaSuccess||cudaDriverGetVersion(&driver)!=cudaSuccess) return 0;
    cpu_set_t affinity; CPU_ZERO(&affinity);
    if(sched_getaffinity(0,sizeof(affinity),&affinity))return 0;
    std::ostringstream a; a<<"["; bool first=true;
    for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&affinity)){
        if(!first)a<<","; first=false; a<<cpu;
    } a<<"]";
    std::ostringstream uuid; uuid<<std::hex<<std::setfill('0');
    for(unsigned char c:p.uuid.bytes)uuid<<std::setw(2)<<unsigned(c);
    std::ostringstream environment; environment<<"{"; first=true;
    @ENVIRONMENT@
    environment<<"}";
    const auto &cpu_name=generated_kernels::offload::cpu_identity();
    if(cpu_name.empty())return 0;
    std::ostringstream identity;
    identity<<"{\"affinity\":"<<a.str()<<",\"cpu_name\":"<<quoted(cpu_name)<<",\"cuda_compiler\":"
        <<quoted(std::to_string(__CUDACC_VER_MAJOR__)+"."+std::to_string(__CUDACC_VER_MINOR__)+"."+std::to_string(__CUDACC_VER_BUILD__))
        <<",\"cuda_driver\":"<<driver<<",\"cuda_runtime\":"<<runtime<<",\"dynamic\":"<<omp_get_dynamic()
        <<",\"environment\":"<<environment.str()<<",\"fortran_options\":"<<quoted(@FORTRANOPTIONS@)
        <<",\"fortran_version\":"<<quoted(version)<<",\"gpu_name\":"<<quoted(p.name)
        <<",\"gpu_uuid\":"<<quoted(uuid.str())<<",\"host_compiler\":"<<quoted(__VERSION__)
        <<",\"max_active_levels\":"<<omp_get_max_active_levels()
        <<",\"proc_bind\":"<<int(omp_get_proc_bind())<<",\"thread_limit\":"<<omp_get_thread_limit()
        <<",\"threads\":"<<omp_get_max_threads()<<"}";
    @DECISION@
}
"""
    env = []
    for variable in sorted(ENVIRONMENT):
        env.append(
            "{const char *value=std::getenv(" + json.dumps(variable) + ");"
            'if(!first)environment<<",";first=false;environment<<'
            + json.dumps(json.dumps(variable) + ":")
            + '<<(value?quoted(value):"null");}'
        )
    if capture:
        choice = (
            f'std::ostringstream k;k<<"[";for(int i=0;i<{count};++i){{if(i)k<<",";k<<key[i];}}k<<"]";\n'
            'std::fprintf(stderr,"FORT_EXACT_OBSERVATION {\\"identity\\":%s,\\"key\\":%s}\\n",identity.str().c_str(),k.str().c_str());\n'
            'const char *mode=std::getenv("FORT_EXACT_CALIBRATION_MODE");\n'
            'return mode && !std::strcmp(mode,"resident");'
        )
        keycheck = ""
    else:
        keycheck = "if(" + "||".join(f"key[{i}]!={v}LL" for i, v in enumerate(key)) + ")return 0;"
        choice = (
            "const bool compatible=identity.str()==" + json.dumps(expected_json) + ";\n"
            'if(std::getenv("FORT_RUNTIME_TRACE"))std::fprintf(stderr,'
            '"FORT_EXACT_DECISION selection=%s reason=%s\\n",compatible?"resident":"native",'
            'compatible?"measured_complete_window":"execution_identity_mismatch");\nreturn compatible;'
        )
    for token, value in {
        "NAME": name,
        "ATTESTATION": "FORT_EXACT_WINDOW_BUILD_" + record["contract_sha256"],
        "THREADS": str(record["contract"]["threads"]),
        "KEYCHECK": keycheck,
        "FORTRANCHECK": (
            ""
            if capture
            else "if(!fort_scope_team_fortran_compatible_v1(version,options,"
            + json.dumps(expected["fortran_version"])
            + ","
            + json.dumps(normalize_fortran_options(expected["fortran_options"]))
            + "))return 0;"
        ),
        "FORTRANOPTIONS": "options" if capture else json.dumps(expected["fortran_options"]),
        "ENVIRONMENT": "\n".join(env),
        "DECISION": choice,
    }.items():
        source = source.replace("@" + token + "@", value)
    return source
