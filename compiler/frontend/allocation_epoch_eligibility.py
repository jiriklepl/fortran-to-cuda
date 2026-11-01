"""Source-owned sticky exclusion gate, without resource or execution authority.

This proves a deliberately narrow, once-armed local SAVE flag. It cannot make
an unproved edge safe. A consumer must close before every barred edge and must
check this same flag before every later activation, including a new context.
The flag never certifies aliases, allocation presence or readable payloads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import Base

from compiler.frontend.call_bindings import resolve_original_call
from compiler.frontend.source_effects import _children, _kind, _part
from compiler.frontend.source_functions import _scope_roles, _validate_scope, _validate_scope_roles
from compiler.frontend.supported_paths import _association_receipt, _stamp, _syntax
from compiler.ir import CompilationError

_ISSUER = object()
_UNSAFE = {"pointer", "target", "optional", "volatile", "asynchronous", "value", "parameter", "protected"}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _flat(node):
    return str(node).replace(" ", "").lower()


def _initial(scope, name, value):
    declarations = [node for node, _path in _syntax((_part(scope.node, "Specification_Part"),))
                    if _kind(node) == "Entity_Decl" and str(node.items[0]).lower() == name]
    return (len(declarations) == 1 and declarations[0].items[3] is not None
            and _flat(declarations[0].items[3]) == "=" + value)


def _integer_metadata(analysis, scope, node):
    permitted = {"Name", "Int_Literal_Constant", "Allocate_Shape_Spec", "Allocate_Shape_Spec_List",
                 "Level_2_Expr", "Level_2_Unary_Expr", "Add_Operand", "Mult_Operand", "Parenthesis"}
    for item, _path in _syntax((node,)):
        if isinstance(item, (tuple, list)):
            continue
        if _kind(item) not in permitted:
            raise CompilationError("allocation gate bounds cannot call or expose fresh storage")
        if _kind(item) == "Name":
            binding = analysis._binding(analysis.source_scope_for(item, scope), item)
            if (binding is None or binding.dtype != "integer" or binding.rank
                    or binding.attributes & ((_UNSAFE - {"parameter"}) | {"allocatable"})):
                raise CompilationError("allocation gate bounds require original ordinary integer metadata")


def _prearm_clock(analysis, routine, node, resources):
    """Reuse exact original clock/descriptor authority before the final arm."""
    from compiler.frontend.descriptor_windows import prove_descriptor_window
    from compiler.frontend.scalar_clock import original_scalar_clock

    call = node
    if _kind(node) == "If_Stmt":
        call = node.items[1]
        allowed = {"Name", "Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant",
                   "Level_4_Expr", "Relational_Expr", "And_Operand", "Or_Operand", "Equiv_Operand",
                   "Parenthesis", "Level_1_Expr"}
        for item, _path in _syntax((node.items[0],)):
            if isinstance(item, (tuple, list)):
                continue
            if _kind(item) not in allowed:
                raise CompilationError("allocation gate pre-arm clock predicate requires original ordinary scalar metadata")
            if _kind(item) == "Name":
                binding = analysis._binding(analysis.source_scope_for(item, routine.scope), item)
                if (binding is None or binding.rank or binding.dtype not in {"integer", "real", "logical"}
                        or binding.attributes & ((_UNSAFE-{"parameter"}) | {"allocatable"})):
                    raise CompilationError("allocation gate pre-arm clock predicate cannot read associated storage")
    if _kind(call) != "Call_Stmt":
        raise CompilationError("allocation gate pre-arm work requires an exact original scalar clock")
    clock = original_scalar_clock(analysis, routine.qualified, call)
    graph, identities = analysis._selected_source(routine.qualified, (node,))
    if len(identities) != 1:
        raise CompilationError("allocation gate pre-arm metadata lost its exact original graph node")
    descriptor = prove_descriptor_window(analysis, routine.qualified, identities,
                                        tuple(binding.root for binding in resources))
    clock.validate(analysis, routine.qualified, call)
    descriptor.validate(analysis, routine.qualified, identities)
    return node, clock, descriptor


def _derive(analysis, routine, flag, resources, barriers):
    if (getattr(analysis, "_descriptor_source", None) is not None
            or analysis._require_original(routine.qualified) is not routine):
        raise CompilationError("allocation gate requires its exact original routine")
    roles, scope = _scope_roles(analysis), routine.scope
    _validate_scope(analysis, scope)
    inputs, input_identity = analysis.inputs, analysis.inputs.identity()
    if (scope.bindings.get(flag.name) is not flag or flag.declaring_scope is not scope
            or flag.dtype != "logical" or flag.kind != 4 or flag.rank or flag.name in routine.arguments
            or "save" not in flag.attributes or flag.attributes & _UNSAFE
            or not _initial(scope, flag.name, ".false.")):
        raise CompilationError("allocation gate requires a private local SAVE LOGICAL initialized false")
    syntax = tuple(_syntax((routine.scope.node,)))
    if any(_kind(node) in {"Internal_Subprogram_Part", "Namelist_Stmt", "Common_Stmt", "Equivalence_Stmt",
                         "Cray_Pointer_Stmt", "Codimension_Stmt", "Codimension_Attr_Spec"}
           or _kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$") for node, _path in syntax):
        raise CompilationError("allocation gate requires independent local storage without observers or concurrency")
    if (not isinstance(resources, tuple) or not resources or len({id(item) for item in resources}) != len(resources)
            or any(scope.bindings.get(item.name) is not item or item.declaring_scope is not scope
                   or item.name in routine.arguments or item.dtype not in {"real", "integer"}
                   or not item.rank or not {"save", "allocatable"} <= item.attributes
                   or item.attributes & _UNSAFE for item in resources)):
        raise CompilationError("allocation gate requires exact independent local SAVE allocation identities")
    execution = tuple(_children(routine.execution))
    significant = tuple(node for node in execution if _kind(node) != "Comment")
    guard = significant[0] if significant else None
    if _kind(guard) != "If_Construct":
        raise CompilationError("allocation gate must arm in the first original first-call guard")
    header, *middle, end = guard.content
    condition = header.items[0] if _kind(header) == "If_Then_Stmt" else None
    if (_kind(condition) not in {"Level_4_Expr", "Relational_Expr"} or len(condition.items) != 3
            or _kind(condition.items[0]) != "Name" or str(condition.items[1]) not in {"==", ".EQ."}
            or _flat(condition.items[2]) != "0" or _kind(end) != "End_If_Stmt"
            or any(_kind(node) in {"Else_Stmt", "Else_If_Stmt"} for node in middle)):
        raise CompilationError("allocation gate requires one original counter-equals-zero first-call guard")
    counter = analysis._binding(scope, condition.items[0])
    if (counter is None or counter.declaring_scope is not scope or scope.bindings.get(counter.name) is not counter
            or counter.dtype != "integer" or counter.rank or "save" not in counter.attributes
            or counter.attributes & _UNSAFE or not _initial(scope, counter.name, "0")):
        raise CompilationError("allocation gate requires an independent local SAVE integer counter initialized zero")
    body = tuple(node for node in middle if _kind(node) != "Comment")
    prefix = body[:len(resources)]
    if len(prefix) != len(resources):
        raise CompilationError("allocation gate requires every fresh original allocation before arming")
    for node, resource in zip(prefix, resources, strict=True):
        if _kind(node) != "Allocate_Stmt" or node.items[0] is not None or node.items[2] is not None:
            raise CompilationError("allocation gate requires unconditional original allocations without status or aliases")
        allocations = node.items[1].items
        if (len(allocations) != 1 or _kind(allocations[0]) != "Allocation"
                or _kind(allocations[0].items[0]) != "Name"
                or analysis._binding(scope, allocations[0].items[0]) is not resource):
            raise CompilationError("allocation gate lost its exact fresh allocation sequence")
        _integer_metadata(analysis, scope, allocations[0].items[1])
    arm_candidates = [node for node, _path in _syntax((routine.execution,))
                      if _kind(node) == "Assignment_Stmt" and _kind(node.items[0]) == "Name"
                      and analysis._binding(scope, node.items[0]) is flag and _flat(node.items[2]) == ".true."]
    if len(arm_candidates) != 1 or not any(node is arm_candidates[0] for node in body):
        raise CompilationError("allocation gate requires exactly one original arming assignment")
    arm, = arm_candidates
    position = next(index for index, node in enumerate(body) if node is arm)
    if position < len(resources):
        raise CompilationError("allocation gate cannot arm before every fresh allocation")
    if position != len(body)-1:
        raise CompilationError("allocation gate cannot leave unchecked work after its first-allocation arming")
    prearm = []
    for node in body[len(resources):position]:
        literal_assignment = (_kind(node) == "Assignment_Stmt" and _kind(node.items[0]) == "Name"
                and any(analysis._binding(scope, node.items[0]) is resource for resource in resources)
                and _kind(node.items[2]) in {"Int_Literal_Constant", "Real_Literal_Constant"})
        if not literal_assignment:
            prearm.append(_prearm_clock(analysis, routine, node, resources))
    # Metadata arming cannot erase a previously exposed allocation. The first
    # guard is executed at most once in a defined Fortran invocation sequence.
    increment = significant[1] if len(significant) > 1 else None
    if (_kind(increment) != "Assignment_Stmt" or _flat(increment) != counter.name + "=" + counter.name + "+1"):
        raise CompilationError("allocation gate requires immediate original monotonic counter advancement")

    sequences = {}
    uses, flag_assignments, counter_writes = [], [], []
    for node, path in _syntax((routine.execution,)):
        content = tuple(_children(node))
        if content and (hasattr(node, "content") or _kind(node) == "Execution_Part"):
            children = tuple(child for child in content if _kind(child) != "Comment")
            for index, child in enumerate(children):
                sequences[id(child)] = (children, index)
        if _kind(node) == "Assignment_Stmt":
            lhs = node.items[0]
            binding = analysis._binding(analysis.source_scope_for(lhs, scope), lhs) if _kind(lhs) == "Name" else None
            if binding is flag:
                flag_assignments.append(node)
            if binding is counter:
                counter_writes.append(node)
        if _kind(node) == "Name" and str(node).lower() in {flag.name, counter.name}:
            fresh = analysis._binding(analysis.source_scope_for(node, scope), node)
            if fresh not in (flag, counter):
                raise CompilationError("allocation gate original scalar binding changed")
            uses.append((node, path))
    if len(counter_writes) != 1 or counter_writes[0] is not increment:
        raise CompilationError("allocation gate counter cannot be reset or assigned indirectly")
    if not isinstance(barriers, tuple) or not barriers or len({id(node) for node in barriers}) != len(barriers):
        raise CompilationError("allocation gate requires exact distinct original excluded CALLs")
    clearings, resolutions, barrier_records = [], [], []
    for call in barriers:
        sequence = sequences.get(id(call))
        if _kind(call) != "Call_Stmt" or sequence is None or not sequence[1]:
            raise CompilationError("allocation gate barrier must precede its exact original CALL")
        clearing = sequence[0][sequence[1]-1]
        if (_kind(clearing) != "Assignment_Stmt" or _kind(clearing.items[0]) != "Name"
                or analysis._binding(scope, clearing.items[0]) is not flag or _flat(clearing.items[2]) != ".false."):
            raise CompilationError("allocation gate must clear immediately before every excluded CALL")
        resolved = resolve_original_call(analysis, analysis.source_scope_for(call, scope), call)
        receipt = _association_receipt(analysis, routine, call, resolved)
        resolutions.append((call, resolved, receipt))
        clearings.append(clearing)
        barrier_records.append({"call": str(call), "procedure": resolved.procedure,
                                "call_sha256": sha256(str(call).encode()).hexdigest(),
                                "associations": resolved.public()["associations"]})
    if (len(flag_assignments) != 1+len(clearings)
            or {id(node) for node in flag_assignments} != {id(arm), *(id(node) for node in clearings)}):
        raise CompilationError("allocation gate cannot rearm or clear at an unbound source occurrence")
    # A scalar may not be passed to a callee or exported as an address, even
    # when its only textual writes match the assignments above.
    for node, path in uses:
        ancestors = []
        cursor = routine.execution
        for index in path[1:]:
            ancestors.append(cursor)
            cursor = _children(cursor)[index]
        statement = next((parent for parent in reversed(ancestors) if _kind(parent).endswith("_Stmt")), None)
        if analysis._binding(scope, node) is flag:
            if any(statement is assignment for assignment in flag_assignments):
                if statement.items[0] is not node:
                    raise CompilationError("allocation gate cannot expose its saved state through an expression")
            elif not (_kind(statement) in {"If_Then_Stmt", "If_Stmt"} and statement.items[0] is node):
                raise CompilationError("allocation gate may only restrict an exact original IF condition")
        elif (_kind(statement) not in {"Assignment_Stmt", "If_Then_Stmt", "If_Stmt", "Else_If_Stmt"}
                or any(_kind(parent) in {"Part_Ref", "Function_Reference", "Intrinsic_Function_Reference",
                                        "Structure_Constructor"} for parent in ancestors)):
            raise CompilationError("allocation gate counter cannot escape its original frame")
    record = {"version": 1, "kind": "original_once_armed_allocation_gate", "procedure": routine.qualified,
              "source_inputs": input_identity, "original_body_sha256": sha256(str(routine.scope.node).encode()).hexdigest(),
              "flag": flag.public(), "resources": [item.public() for item in resources],
              "counter": counter.public(), "arm": _stamp(arm, ()), "barriers": barrier_records,
              "rule": "arm only at first original allocation; clearing survives return and every new context; no rearm after an excluded CALL",
              "required_consumer": "publish and close before excluded CALL; check this flag before every domain activation and child dispatch",
              "scope": "restriction on all managed canonical allocations, not evidence about aliases or hidden callee exposures",
              "lifetime_authorized": False, "allocation_presence_authority": False,
              "payload_definition_authority": False, "payload_access_authorized": False,
              "execution_authorized": False, "cross_frontier_retention_authorized": False}
    if prearm:
        record.update(version=2, prearm_metadata=[{"original_node": _stamp(node, ()),
            "clock_identity": clock.identity, "descriptor_identity": descriptor.identity}
            for node, clock, descriptor in prearm])
    encoded = _encoded(record)
    consumed = tuple((node, path) for node, path in syntax if isinstance(node, Base))
    fresh = dict((path, node) for node, path in _syntax((routine.scope.node,)))
    if any(fresh.get(path) is not node for node, path in consumed):
        raise CompilationError("allocation gate consumed original occurrence changed")
    for call, resolved, receipt in resolutions:
        if _association_receipt(analysis, routine, call, resolved) != receipt:
            raise CompilationError("allocation gate excluded CALL association changed")
    for node, clock, descriptor in prearm:
        call = node.items[1] if _kind(node) == "If_Stmt" else node
        clock.validate(analysis, routine.qualified, call)
        descriptor.validate(analysis, routine.qualified, analysis._selected_source(routine.qualified, (node,))[1])
    _validate_scope_roles(analysis, roles, {id(scope): scope})
    _validate_scope(analysis, scope)
    if analysis.inputs is not inputs or analysis.inputs.identity() != input_identity:
        raise CompilationError("allocation gate source inputs changed before publication")
    analysis.verify_publication()
    return encoded, consumed, tuple(item[2] for item in resolutions), tuple(prearm)


@dataclass(frozen=True, slots=True)
class OriginalAllocationGate:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _flag: object = field(repr=False, compare=False)
    _resources: tuple = field(repr=False, compare=False)
    _barriers: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _uses: tuple = field(repr=False, compare=False)
    _resolutions: tuple = field(repr=False, compare=False)
    _prearm: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self):
        analysis = self._analysis
        if (self._issuer is not _ISSUER or type(self) is not OriginalAllocationGate
                or type(self._record) is not str or sha256(self._record.encode()).hexdigest() != self.identity
                or getattr(analysis, "_allocation_gates", {}).get(self.identity) is not self):
            raise CompilationError("allocation gate requires its exact registered original proof")
        record, uses, resolutions, prearm = _derive(analysis, self._routine, self._flag, self._resources, self._barriers)
        if (record != self._record or resolutions != self._resolutions or len(uses) != len(self._uses)
                or any(a is not b or ap != bp for (a, ap), (b, bp) in zip(uses, self._uses, strict=True))
                or len(prearm) != len(self._prearm)
                or any(a is not b or p is not q or r is not s for (a, p, r), (b, q, s)
                       in zip(prearm, self._prearm, strict=True))
                or getattr(analysis, "_allocation_gates", {}).get(self.identity) is not self):
            raise CompilationError("allocation gate source or binding authority changed")
        analysis.verify_publication()
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def prove_allocation_gate(analysis, routine, flag, resources, barriers):
    resources, barriers = tuple(resources), tuple(barriers)
    registry = getattr(analysis, "_allocation_gates", None)
    if registry is None:
        registry = analysis._allocation_gates = {}
    if type(registry) is not dict:
        raise CompilationError("allocation gate requires its private original-analysis issuer registry")
    for proof in tuple(registry.values()):
        if type(proof) is not OriginalAllocationGate or proof._analysis is not analysis:
            raise CompilationError("allocation gate cannot consume public metadata as authority")
        if (proof._routine is routine and proof._flag is flag and proof._resources == resources
                and len(proof._barriers) == len(barriers)
                and all(a is b for a, b in zip(proof._barriers, barriers, strict=True))):
            return proof.validate()
    record, uses, resolutions, prearm = _derive(analysis, routine, flag, resources, barriers)
    if getattr(analysis, "_allocation_gates", None) is not registry:
        raise CompilationError("allocation gate registry changed before publication")
    proof = OriginalAllocationGate(sha256(record.encode()).hexdigest(), analysis, routine, flag, resources, barriers,
                                  record, uses, resolutions, prearm, _ISSUER)
    registry[proof.identity] = proof
    return proof.validate()
