"""Original-source proofs for bounded, read-only parallel reductions.

These records establish source form and effects, not runtime placement. Real
extrema and reassociated sums deliberately retain distinct native contracts.
No routine name supplied by an application grants numerical legality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json

from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError, SourceLocation

REDUCTION_PROOF_VERSION = 1
_REDUCTIONS = frozenset({"minval", "maxval", "all", "any", "sum"})
_IEEE_NAN = "$intrinsic::ieee_arithmetic::ieee_is_nan"


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ReductionProof:
    procedure: str
    node_id: str
    structured_identity: str
    identity: str
    available: bool
    reason: str | None
    _record: object = field(repr=False, compare=False)
    original_node: object = field(repr=False, compare=False)

    def public(self):
        return {"schema_version": REDUCTION_PROOF_VERSION, "procedure": self.procedure,
                "node_id": self.node_id, "structured_identity": self.structured_identity,
                "proof_identity": self.identity, "available": self.available,
                "source_analysis_available": self.available,
                "execution_supported": False,
                "execution_reason": "source proof only; no generated reduction worker or calibrated placement",
                "reason": self.reason, **_thaw(self._record)}

    def validate(self, analysis):
        """Recheck authority before attaching a generated implementation."""
        graph = analysis.structure(self.procedure)
        if (analysis._reduction_proofs.get(self.identity) is not self
                or graph.identity != self.structured_identity or graph.node_id(self.original_node) != self.node_id
                or str(self.original_node) != self._record["original_source"]):
            raise CompilationError("reduction proof no longer has original source authority")
        return self


def _function(node):
    while _kind(node) == "Parenthesis":
        node = node.items[1]
    if _kind(node) not in {"Intrinsic_Function_Reference", "Function_Reference", "Part_Ref"}:
        raise CompilationError("reduction requires one original intrinsic assignment")
    return node, str(node.items[0]).lower(), tuple(_children(node.items[1]))


def candidate_nodes(graph, identities=None):
    """Select only reduction-shaped original assignments in one local graph."""
    owned = None
    if identities is not None:
        from fparser.two.utils import walk
        owned = {id(item) for identity in identities for original in graph.source_nodes(identity)
                 for item in walk(original)}
    result = []
    for identity, node in graph.nodes.items():
        if node.kind != "operation" or node.details.get("evaluation") != "statement":
            continue
        originals = graph.source_nodes(identity)
        if len(originals) != 1 or _kind(originals[0]) != "Assignment_Stmt":
            continue
        statement = originals[0]
        if owned is not None and id(statement) not in owned:
            continue
        try:
            _original, name, _arguments = _function(statement.items[2])
        except CompilationError:
            continue
        if name in _REDUCTIONS:
            result.append(identity)
    return tuple(result)


def _arguments(original, formals):
    result, order, position, keyword_seen = {}, [], 0, False
    for item in original:
        if _kind(item) == "Actual_Arg_Spec":
            keyword_seen = True
            formal, value = str(item.items[0]).lower(), item.items[1]
            keyword = str(item.items[0])
        else:
            if keyword_seen or position >= len(formals):
                raise CompilationError("reduction arguments must retain valid original keyword order")
            formal, value, keyword = formals[position], item, None
            position += 1
        if formal not in formals or formal in result:
            raise CompilationError("reduction has an unknown or duplicate argument: " + formal)
        result[formal] = value
        order.append({"formal": formal, "keyword": keyword, "source": str(value)})
    if formals[0] not in result:
        raise CompilationError("reduction is missing its original array argument")
    return result, order


def _unshadowed(analysis, scope, name):
    if (analysis._binding(scope, name) is not None or analysis._candidates(scope, name)
            or analysis._unknown_exports(scope)):
        raise CompilationError("reduction intrinsic identity is unresolved or shadowed: " + name)


def _reference(analysis, scope, node):
    while _kind(node) == "Parenthesis":
        node = node.items[1]
    if _kind(node) == "Data_Ref":
        from compiler.frontend.component_bindings import component_access
        access = component_access(analysis, scope, node)
        if access is None:
            raise CompilationError("reduction component storage is unresolved")
        binding, indices = access.binding, access.indices
    elif _kind(node) == "Part_Ref":
        binding, indices = analysis._binding(scope, node.items[0]), tuple(_children(node.items[1]))
    elif _kind(node) == "Name":
        binding, indices = analysis._binding(scope, node), ()
    else:
        raise CompilationError("reduction value requires a whole array or proven rectangular section")
    if binding is None or not binding.rank:
        raise CompilationError("reduction value requires a source-backed array")
    if binding.attributes & {"pointer", "optional", "volatile", "asynchronous"}:
        raise CompilationError("reduction array association or presence requires additional proof")
    if indices:
        if len(indices) != binding.rank:
            raise CompilationError("reduction section rank differs from original storage")
        rank = sum(_kind(index) == "Subscript_Triplet" for index in indices)
    else:
        rank = binding.rank
    if rank < 1:
        raise CompilationError("reduction array argument is scalar")
    reason = analysis.resource_identity_boundary(binding)
    if reason:
        raise CompilationError(reason)
    return node, binding, rank


def _logical_scalar(analysis, scope, node):
    while _kind(node) == "Parenthesis":
        node = node.items[1]
    if _kind(node) == "Logical_Literal_Constant":
        if node.items[1] is not None and scope.kinds.integer(node.items[1], SourceLocation(str(scope.path))) != 4:
            raise CompilationError("reduction scalar MASK requires default LOGICAL")
        return {"kind": "literal", "source": str(node), "value": str(node.items[0]).lower() == ".true."}
    if _kind(node) not in {"Name", "Data_Ref"}:
        return None
    binding = analysis._binding(scope, node)
    if binding is None or binding.rank:
        return None
    if binding.dtype != "logical" or binding.kind != 4 or binding.attributes & {
            "pointer", "optional", "allocatable", "volatile", "asynchronous"}:
        raise CompilationError("reduction scalar MASK requires a safely reached default LOGICAL")
    return {"kind": "scalar", "source": str(node), "resource": binding.root}


def analyze_reduction(analysis, procedure, original):
    """Recognize one authenticated source assignment without executing inputs.

    An available record still requires descriptor/presence/alias checks, exact
    coherence hooks, resource acquisition and the stated numerical contract.
    Speculative GPU partials may write scratch only; scalar publication is one
    commit at the original assignment position. This is not an entry replay.
    """
    graph = analysis.structure(procedure)
    if isinstance(original, str):
        selected = graph.source_nodes(original)
        if len(selected) != 1:
            raise CompilationError("reduction ID does not identify one original assignment")
        statement, identity = selected[0], original
    else:
        statement, identity = original, graph.node_id(original)
    node = graph.nodes[identity]
    if (not graph.available or node.kind != "operation" or node.details.get("evaluation") != "statement"
            or _kind(statement) != "Assignment_Stmt"):
        raise CompilationError("reduction requires an original reached assignment source node")
    routine = analysis.routines[procedure]
    scope = analysis.source_scope_for(statement, routine.scope)
    target, _equals, value = statement.items
    record = {"source_form": "original_assignment", "original_source": str(statement),
              "source_span": list(node.span), "guard": list(node.guard),
              "original_value": str(value), "target": str(target),
              "target_resource": None, "operator": None, "value": None,
              "mask": None, "dim": None, "empty_identity": None,
              "numerical_contract": None, "canonical_reads": [],
              "scalar_reads": [], "descriptor_reads": [],
              "scalar_publication": None, "requirements": [],
              "argument_evaluation": "original assignment position under original guards",
              "speculative_effects": {"application_writes": [], "scratch_only": True}}
    available, reason = False, None
    try:
        if _kind(target) not in {"Name", "Data_Ref"}:
            raise CompilationError("reduction result requires an original scalar variable")
        target_binding = analysis._binding(scope, target)
        if target_binding is None or target_binding.rank or target_binding.intent == "in" or target_binding.attributes & {
                "pointer", "optional", "allocatable", "volatile", "asynchronous", "parameter"}:
            raise CompilationError("reduction scalar publication association or type is unproved")
        record["target_resource"] = target_binding.root
        function, name, original_arguments = _function(value)
        record["operator"] = name
        if name not in _REDUCTIONS:
            if name in {"min", "max"}:
                record["source_form"] = "scalar_extremum_assignment"
            raise CompilationError("source operation is not an admitted reduction intrinsic")
        record["source_form"] = "intrinsic_array_reduction"
        _unshadowed(analysis, scope, name)
        formals = ("mask", "dim") if name in {"all", "any"} else ("array", "dim", "mask")
        if (name not in {"all", "any"} and len(original_arguments) == 2
                and all(_kind(argument) != "Actual_Arg_Spec" for argument in original_arguments)):
            signature = analysis._signature(scope, original_arguments[1])
            if signature is not None and signature[0] == "logical":
                # The standard positional ARRAY,MASK overload omits DIM;
                # distinguish it by source-backed type, never by spelling.
                formals = ("array", "mask")
        arguments, order = _arguments(original_arguments, formals)
        record["original_argument_order"] = order
        array = arguments["mask" if name in {"all", "any"} else "array"]
        predicate = None
        if name in {"all", "any"} and _kind(array) in {
                "Function_Reference", "Intrinsic_Function_Reference", "Part_Ref"}:
            candidate, predicate_name, predicate_arguments = _function(array)
            if (analysis._binding(scope, predicate_name) is None
                    and not analysis._unknown_exports(scope)
                    and analysis._candidates(scope, predicate_name) == [_IEEE_NAN]):
                mapped, _order = _arguments(predicate_arguments, ("x",))
                array, predicate = mapped["x"], {"kind": "ieee_is_nan", "identity": _IEEE_NAN,
                                               "source": str(candidate)}
        array, binding, rank = _reference(analysis, scope, array)
        record["value"] = {"source": str(array), "resource": binding.root, "type": binding.dtype,
                           "kind": binding.kind, "logical_rank": rank, "predicate": predicate}
        if "dim" in arguments:
            dimension = scope.kinds.integer(arguments["dim"], SourceLocation(str(scope.path)))
            record["dim"] = {"source": str(arguments["dim"]), "value": dimension}
            if dimension != 1 or rank != 1:
                raise CompilationError("initial reduction DIM support requires a scalar result from a rank-one array")
        if name in {"all", "any"}:
            if target_binding.dtype != "logical" or target_binding.kind != 4:
                raise CompilationError("logical reduction publication requires default LOGICAL")
            if predicate is None and (binding.dtype, binding.kind) != ("logical", 4):
                raise CompilationError("ALL/ANY input requires default LOGICAL or typed IEEE_IS_NAN")
            if predicate is not None and (binding.dtype != "real" or binding.kind not in {4, 8}):
                raise CompilationError("IEEE_IS_NAN input requires supported REAL precision")
            record["empty_identity"] = {"type": "logical", "value": name == "all"}
            record["numerical_contract"] = {"kind": "associative_logical", "reassociation": "exact",
                "native_predicate_contract_required": predicate is not None,
                "nan_predicate": "requires verified native flag/signaling behavior" if predicate else None}
            if predicate:
                record["requirements"].append("verified native IEEE_IS_NAN classification and observable exception policy")
        else:
            if (target_binding.dtype, target_binding.kind) != (binding.dtype, binding.kind):
                raise CompilationError("reduction publication requires the original numerical result type and kind")
            if binding.dtype == "integer" and binding.kind in {4, 8}:
                huge = (1 << (8 * binding.kind - 1)) - 1
                record["empty_identity"] = {"type": "integer", "kind": binding.kind,
                                             "value": huge if name == "minval" else -huge}
                record["numerical_contract"] = {"kind": "associative_integer_extremum", "reassociation": "exact",
                    "requires_nonempty_tag": True,
                    "initialization": "first selected element; apply the empty identity only when none were selected"}
            elif binding.dtype == "real" and binding.kind in {4, 8}:
                record["empty_identity"] = {"type": "real", "kind": binding.kind,
                                             "source": "HUGE(source kind)" if name == "minval" else "-HUGE(source kind)"}
                record["numerical_contract"] = {"kind": "original_native_real_extremum",
                    "nan_operand_order": "unproved", "signed_zero": "unproved", "observable_exceptions": "unproved"}
            else:
                raise CompilationError("reduction input numerical type or kind is unsupported")
            if "mask" in arguments:
                mask = _logical_scalar(analysis, scope, arguments["mask"])
                if mask is None:
                    mask_node, mask_binding, mask_rank = _reference(analysis, scope, arguments["mask"])
                    if (mask_binding.dtype, mask_binding.kind) != ("logical", 4) or mask_rank != rank:
                        raise CompilationError("reduction MASK requires conforming default LOGICAL storage")
                    mask = {"kind": "array", "source": str(mask_node), "resource": mask_binding.root,
                            "logical_rank": mask_rank}
                    record["requirements"].append("runtime MASK extent conformity before speculative work")
                record["mask"] = mask
        # Refine the original complete expression, including masks, through the
        # same typed physical mapping used by native coherence hooks.
        sections = analysis.native_sections_for_nodes(procedure, (identity,), capture_locals=True)
        if not sections.available:
            raise CompilationError("reduction exact input footprint unavailable: " + (sections.reason or "unknown"))
        for resource in sections.resources:
            if resource.writes:
                raise CompilationError("reduction speculative inputs contain application array writes")
            if resource.reads:
                record["canonical_reads"].append({"resource": resource.resource, "rank": resource.rank,
                    "sections": resource.public(), "position": "before speculative read-only partials"})
            for dependency in resource.dependencies:
                key = "scalar_reads" if dependency.kind == "scalar_read" else "descriptor_reads"
                if dependency.resource not in record[key]:
                    record[key].append(dependency.resource)
        if record["mask"] is not None and record["mask"]["kind"] == "scalar":
            record["scalar_reads"].append(record["mask"]["resource"])
        record["scalar_reads"] = sorted(set(record["scalar_reads"]))
        record["descriptor_reads"] = sorted(set(record["descriptor_reads"]))
        record["scalar_publication"] = {"resource": target_binding.root, "type": target_binding.dtype,
            "kind": target_binding.kind, "position": "one commit at the original assignment",
            "original_owner": True, "repeats_procedure_entry_definition_event": False}
        record["requirements"] += ["original allocation/presence and alias proof",
            "checked exact input sections and full-layout logical coordinates",
            "complete outstanding partials before scalar commit or native consumer",
            "resource failure before work may retain native; execution errors never replay earlier work"]
        if name == "sum":
            record["numerical_contract"] = {"kind": "original_native_sum", "reassociation": "not authorized"}
            record["empty_identity"] = {"type": binding.dtype, "kind": binding.kind, "value": 0}
            raise CompilationError("SUM remains native without an authorized original numerical reassociation contract")
        if name in {"minval", "maxval"} and binding.dtype == "real":
            raise CompilationError("real MINVAL/MAXVAL require a separate verified NaN, signed-zero and exception contract")
        available = True
    except CompilationError as error:
        reason = str(error)
    payload = {"version": REDUCTION_PROOF_VERSION, "procedure": procedure, "node": identity,
               "graph": graph.identity, "available": available, "reason": reason, "record": record}
    digest = sha256(_canonical(payload).encode()).hexdigest()
    if digest not in analysis._reduction_proofs:
        if len(analysis._reduction_proofs) >= analysis.operation_limit:
            analysis._reduction_proofs.pop(next(iter(analysis._reduction_proofs)))
        analysis._reduction_proofs[digest] = ReductionProof(procedure, identity, graph.identity, digest,
                                                         available, reason, _freeze(record), statement)
    return analysis._reduction_proofs[digest]
