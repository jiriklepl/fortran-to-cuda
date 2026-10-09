"""Original OpenMP clauses can supply a bounded reduction-order contract.

This module proves source semantics only. It neither relaxes ordinary native
completion tokens nor supplies a generated executor or placement estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re

from fparser.two.utils import walk

from compiler.frontend.native_completion import _children, _directive, _kind
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError, ScalarType, SourceLocation
from compiler.ir.intrinsics import INTRINSICS, intrinsic_type

OPENMP_REDUCTION_PROOF_VERSION = 1
_UNSAFE = frozenset({"pointer", "allocatable", "optional", "target", "volatile", "asynchronous"})
_TYPES = {("integer", 4, 0): ScalarType.INTEGER, ("real", 4, 0): ScalarType.REAL32,
          ("real", 8, 0): ScalarType.REAL}
_SIGNATURES = {value: key for key, value in _TYPES.items()}


@dataclass(frozen=True)
class OpenMPReductionProof:
    procedure: str
    structured_identity: str
    identity: str
    selected_node_ids: tuple[str, ...]
    available: bool
    reason: str | None
    _record: object = field(repr=False, compare=False)
    _originals: tuple[object, ...] = field(repr=False, compare=False)

    def public(self):
        return {"schema_version": OPENMP_REDUCTION_PROOF_VERSION, "procedure": self.procedure,
                "structured_identity": self.structured_identity, "proof_identity": self.identity,
                "selected_original_nodes": list(self.selected_node_ids),
                "source_analysis_available": self.available, "execution_supported": False,
                "execution_reason": "source contract only; no generated OpenMP reduction executor or calibration",
                "reason": self.reason, **_thaw(self._record)}

    def validate(self, analysis):
        graph, identities = analysis._selected_source(self.procedure, self._originals)
        if (getattr(analysis, "_omp_reduction_proofs", {}).get(self.identity) is not self
                or graph.identity != self.structured_identity or identities != self.selected_node_ids):
            raise CompilationError("OpenMP reduction proof lacks original complete-group authority")
        return self


def analyze_openmp_reduction(analysis, procedure, selected):
    """Prove one original joined team with one local scalar '+' reduction.

    Whole effects remain conservative source effects. Before execution, a
    separate worker must establish exact physical reads, resource admission,
    numerical semantics and the original scalar publication protocol.
    """
    graph, identities = analysis._selected_source(procedure, selected)
    originals = tuple(node for identity in identities for node in graph.source_nodes(identity))
    routine = analysis.routines[procedure]
    location = SourceLocation(str(routine.scope.path))
    record = {"source_form": "original_openmp_clause_reduction", "operator": "+",
              "original_source": [str(node) for node in originals],
              "guard": list(graph.nodes[identities[0]].guard) if identities else [],
              "accumulator": None, "initial_value": None, "private_initializer": None,
              "contribution": None, "loop_domains": [], "private_resources": [],
              "reduction_private_copies": None, "per_item_statements": [],
              "canonical_reads": [], "scalar_reads": [], "descriptor_reads": [],
              "ordered_effects": [], "scalar_publication": None,
              "native_completion": None, "numerical_contract": None,
              "requirements": [], "preserves_original_owner": True}
    available, reason = False, None
    try:
        peeled, expanded = {}, []
        for node in originals:
            if _kind(node) == "Block_Nonlabel_Do_Construct":
                content = list(node.content)
                while content and _kind(content[0]) == "Comment":
                    expanded.append(content.pop(0))
                peeled[id(node)] = tuple(content)
            expanded.append(node)
        nodes = [node for node in expanded if _kind(node) != "Comment" or _directive(node) is not None]
        if not nodes or not (_directive(nodes[0]) or "").startswith("parallel"):
            raise CompilationError("OpenMP reduction requires one complete original PARALLEL group")
        combined = bool(re.match(r"parallel\s+do(?:\s|$)", _directive(nodes[0])))
        ending = "end parallel do" if combined else "end parallel"
        if _directive(nodes[-1]) != ending:
            raise CompilationError("OpenMP reduction requires its complete original joined region")
        private, shared, reductions = {}, {}, []
        collapse, default = 1, "shared"

        def binding(name):
            result = analysis._binding(routine.scope, name)
            if result is None or result.attributes & _UNSAFE or analysis.resource_identity_boundary(result):
                raise CompilationError("OpenMP reduction storage association or identity is unproved: " + str(name))
            module = analysis.modules.get(result.root.split("::", 1)[0])
            module_owned = module is not None and any(
                result.root == item.root or result.root.startswith(item.root + "%")
                for item in module.bindings.values())
            if module_owned:
                specification = next((item for item in _children(module.node)
                                      if _kind(item) == "Specification_Part"), None)
                if any(_directive(item) is not None for item in walk(specification)):
                    raise CompilationError("OpenMP reduction module thread ownership is unproved: " + result.root)
            return result

        def clauses(text, region):
            nonlocal collapse, default
            allowed = ({"private", "shared", "default", "reduction"} if region == "parallel" else
                       {"private", "collapse", "schedule", "reduction"} if region == "do" else
                       {"private", "shared", "default", "collapse", "schedule", "reduction"})
            while text.strip():
                match = re.match(r"\s*(private|shared|default|collapse|schedule|reduction)\s*\(([^()]*)\)\s*", text)
                if match is None:
                    raise CompilationError("unsupported original OpenMP reduction clause: " + text.strip())
                clause, value = match.group(1), match.group(2).strip()
                if clause not in allowed:
                    raise CompilationError("OpenMP reduction clause does not belong to its original construct: " + clause)
                if clause in {"private", "shared"}:
                    for name in value.split(","):
                        name = name.strip()
                        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                            raise CompilationError("OpenMP reduction clauses require resolved variable names")
                        item = binding(name)
                        if clause == "private":
                            if (not item.root.startswith(procedure + "::") or item.rank
                                    or item.attributes & {"save", "parameter"}
                                    or item.dtype not in {"real", "integer", "logical"} or item.kind not in {4, 8}):
                                raise CompilationError("OpenMP reduction PRIVATE requires original ordinary local scalars")
                            private[item.root] = item
                        else:
                            shared[item.root] = item
                elif clause == "reduction":
                    reduction = re.fullmatch(r"\+\s*:\s*([a-z][a-z0-9_]*)", value)
                    if reduction is None:
                        raise CompilationError("initial OpenMP source proof requires one builtin scalar '+' reduction")
                    reductions.append((binding(reduction.group(1)), region))
                elif clause == "default":
                    if value not in {"shared", "none"}:
                        raise CompilationError("unsupported original OpenMP DEFAULT")
                    default = value
                elif clause == "collapse":
                    if value not in {"1", "2", "3", "4"}:
                        raise CompilationError("OpenMP reduction COLLAPSE exceeds the bounded loop nest")
                    collapse = int(value)
                elif value not in {"static", "runtime"}:
                    raise CompilationError("unsupported original OpenMP reduction SCHEDULE")
                text = text[match.end():].lstrip(", ")

        clauses(_directive(nodes[0])[len("parallel do"):] if combined else _directive(nodes[0])[len("parallel"):],
                "parallel do" if combined else "parallel")
        inside = nodes[1:-1]
        if not combined:
            if len(inside) != 3 or not re.match(r"do(?:\s|$)", _directive(inside[0]) or ""):
                raise CompilationError("initial OpenMP reduction requires one complete worksharing DO")
            clauses(_directive(inside[0])[2:], "do")
            if _directive(inside[-1]) not in {"end do", "end do nowait"}:
                raise CompilationError("OpenMP reduction worksharing DO requires its original END DO")
            inside = inside[1:-1]
        if len(inside) != 1 or _kind(inside[0]) != "Block_Nonlabel_Do_Construct" or len(reductions) != 1:
            raise CompilationError("initial OpenMP reduction requires one loop and one scalar accumulator")
        accumulator, reduction_region = reductions[0]
        if (not accumulator.root.startswith(procedure + "::") or accumulator.rank
                or accumulator.attributes & {"save", "parameter"}
                or accumulator.signature() not in _TYPES):
            raise CompilationError("OpenMP sum requires an original unaliased local INTEGER(4) or REAL(4/8) accumulator")
        if (accumulator.root in private or (accumulator.root in shared and reduction_region != "do")
                or private.keys() & shared.keys()):
            raise CompilationError("conflicting OpenMP reduction data-sharing clauses")
        if reduction_region == "do" and default == "none" and accumulator.root not in shared:
            raise CompilationError("worksharing reduction requires original shared accumulator authority")
        record["accumulator"] = accumulator.public()
        record["reduction_clause_region"] = reduction_region
        record["initial_value"] = {"resource": accumulator.root, "source": accumulator.name,
            "position": "original pre-region value", "requires_original_definition": True}
        record["private_initializer"] = {"value": 0, "type": accumulator.dtype, "kind": accumulator.kind}
        active, reads, descriptors = {}, {}, {}
        defined = set()

        def read(item):
            if item.root == accumulator.root:
                raise CompilationError("OpenMP accumulator is observed outside its isolated update")
            if item.attributes & _UNSAFE:
                raise CompilationError("OpenMP contribution association or alias proof is unavailable: " + item.root)
            if item.root in private and item.root not in active and item.root not in defined:
                raise CompilationError("OpenMP private contribution state is read before its per-item definition")
            if item.root not in private and item.root not in active and "parameter" not in item.attributes:
                if default == "none" and item.root not in shared:
                    raise CompilationError("OpenMP DEFAULT(NONE) input lacks original shared data authority")
                reads[item.root] = item

        def numeric(node, depth=0):
            if depth > analysis.depth_limit:
                raise CompilationError("OpenMP reduction contribution exceeds bounded expression depth")
            kind, items = _kind(node), _children(node)
            if kind == "Parenthesis":
                return numeric(items[1], depth + 1)
            if kind in {"Int_Literal_Constant", "Real_Literal_Constant"}:
                signature = analysis._signature(routine.scope, node)
                if signature not in _TYPES:
                    raise CompilationError("OpenMP contribution numerical kind is unsupported")
                return signature
            if kind == "Name":
                item = binding(node)
                if item.rank or item.signature() not in _TYPES:
                    raise CompilationError("OpenMP contribution requires scalar numeric values")
                read(item)
                return item.signature()
            if kind == "Part_Ref":
                if analysis._binding(routine.scope, items[0]) is None:
                    raise CompilationError("OpenMP contribution contains an unsupported call or unresolved indexed storage: " + str(node))
                item = binding(items[0])
                indices = tuple(_children(items[1]))
                if not item.rank or len(indices) != item.rank or (item.dtype, item.kind, 0) not in _TYPES:
                    raise CompilationError("OpenMP contribution requires a scalar numeric array element")
                read(item)
                for index in indices:
                    if numeric(index, depth + 1) != ("integer", 4, 0):
                        raise CompilationError("OpenMP contribution index requires original INTEGER(4) scalar math")
                return item.dtype, item.kind, 0
            if len(items) == 2 and str(items[0]) in {"+", "-"}:
                return numeric(items[1], depth + 1)
            if len(items) == 3 and str(items[1]) in {"+", "-", "*", "/"}:
                left, right = numeric(items[0], depth + 1), numeric(items[2], depth + 1)
                if left != right:
                    raise CompilationError("OpenMP contribution mixed-kind arithmetic requires a separate numerical proof")
                return left
            if kind == "Intrinsic_Function_Reference":
                name = str(items[0]).lower()
                if (name not in INTRINSICS or analysis._binding(routine.scope, name)
                        or analysis._candidates(routine.scope, name) or analysis._unknown_exports(routine.scope)):
                    raise CompilationError("OpenMP contribution intrinsic identity or transitive computation is unproved")
                arguments = tuple(_children(items[1]))
                if any(_kind(argument) == "Actual_Arg_Spec" for argument in arguments):
                    raise CompilationError("initial OpenMP contribution intrinsics require original positional arguments")
                output_kind = None
                if name in {"real", "int"} and len(arguments) == 2:
                    output_kind = routine.scope.kinds.integer(arguments[1], location)
                types = tuple(_TYPES[numeric(argument, depth + 1)] for argument in arguments)
                return _SIGNATURES[intrinsic_type(name, types, location, kind=output_kind)]
            raise CompilationError("OpenMP contribution contains an unsupported call, array expression or scalar effect")

        def bound(node, depth=0):
            if depth > analysis.depth_limit:
                raise CompilationError("OpenMP reduction bounds exceed bounded expression depth")
            kind, items = _kind(node), _children(node)
            if kind == "Intrinsic_Function_Reference" and str(items[0]).lower() in {"size", "lbound", "ubound"}:
                name, arguments = str(items[0]).lower(), tuple(_children(items[1]))
                if (analysis._binding(routine.scope, name) or analysis._candidates(routine.scope, name)
                        or analysis._unknown_exports(routine.scope) or len(arguments) != 2 or _kind(arguments[0]) != "Name"):
                    raise CompilationError("OpenMP loop descriptor bound identity is unproved")
                item = binding(arguments[0])
                axis = routine.scope.kinds.integer(arguments[1], location)
                if not 1 <= axis <= item.rank:
                    raise CompilationError("OpenMP loop descriptor bound dimension is invalid")
                if default == "none" and item.root not in shared:
                    raise CompilationError("OpenMP DEFAULT(NONE) descriptor input lacks original shared data authority")
                descriptors[item.root] = item
                return
            if kind == "Parenthesis":
                return bound(items[1], depth + 1)
            if len(items) == 2 and str(items[0]) in {"+", "-"}:
                return bound(items[1], depth + 1)
            if len(items) == 3 and str(items[1]) in {"+", "-", "*"}:
                bound(items[0], depth + 1)
                bound(items[2], depth + 1)
                if str(items[1]) == "*" and not any(_kind(item) == "Int_Literal_Constant" for item in (items[0], items[2])):
                    raise CompilationError("OpenMP loop bound requires affine scalar arithmetic")
                return
            if kind == "Name":
                item = binding(node)
                if item.root in active or item.root in private:
                    raise CompilationError("OpenMP reduction requires independent rectangular loop bounds")
            elif kind != "Int_Literal_Constant":
                raise CompilationError("OpenMP reduction loop bounds require affine INTEGER source expressions")
            if numeric(node, depth + 1) != ("integer", 4, 0):
                raise CompilationError("OpenMP reduction loop bound requires INTEGER(4)")

        loop, domains = inside[0], []
        while _kind(loop) == "Block_Nonlabel_Do_Construct":
            if len(domains) >= analysis.depth_limit:
                raise CompilationError("OpenMP reduction exceeds the bounded rectangular loop depth")
            body = list(peeled.get(id(loop), loop.content))
            if any(_directive(item) is not None for child in body for item in walk(child)):
                raise CompilationError("nested OpenMP directives require a separate reduction completion proof")
            body = [item for item in body if _kind(item) != "Comment"]
            header = body[0]
            control = header.items[1]
            if (not control or control.items[0] is not None or control.items[1] is None
                    or any(item is not None for item in control.items[2:]) or _kind(body[-1]) != "End_Do_Stmt"):
                raise CompilationError("OpenMP reduction requires a complete ordinary counted loop")
            iterator_node, bounds = control.items[1]
            iterator = binding(iterator_node)
            if (not iterator.root.startswith(procedure + "::") or iterator.rank or iterator.signature() != ("integer", 4, 0)
                    or iterator.attributes & {"save", "parameter"}
                    or iterator.root in active or iterator.root == accumulator.root):
                raise CompilationError("OpenMP reduction iterator requires original independent local INTEGER(4) storage")
            bound(bounds[0])
            bound(bounds[1])
            step = 1 if len(bounds) == 2 else routine.scope.kinds.integer(bounds[2], location)
            if step not in {-1, 1}:
                raise CompilationError("initial OpenMP reduction loops require a constant unit stride")
            domains.append({"iterator": iterator.root, "lower": str(bounds[0]), "upper": str(bounds[1]), "step": step})
            active[iterator.root] = iterator
            body = body[1:-1]
            if len(body) == 1 and _kind(body[0]) == "Block_Nonlabel_Do_Construct":
                loop = body[0]
            else:
                break
        if collapse > len(domains):
            raise CompilationError("OpenMP COLLAPSE exceeds the complete rectangular source nest")
        updates = []
        for statement in body:
            if _kind(statement) != "Assignment_Stmt" or _kind(statement.items[0]) != "Name":
                raise CompilationError("OpenMP reduction body requires scalar assignments without calls, exits or array writes")
            target, _equals, value = statement.items
            item = binding(target)
            if item.root == accumulator.root:
                while _kind(value) == "Parenthesis":
                    value = value.items[1]
                parts = _children(value)
                if len(parts) != 3 or str(parts[1]) != "+":
                    raise CompilationError("OpenMP accumulator requires one isolated original '+' update")
                positions = [index for index in (0, 2) if _kind(parts[index]) == "Name"
                             and binding(parts[index]).root == accumulator.root]
                if len(positions) != 1:
                    raise CompilationError("OpenMP accumulator update must read its private value exactly once")
                contribution = parts[2 if positions[0] == 0 else 0]
                if numeric(contribution) != accumulator.signature():
                    raise CompilationError("OpenMP contribution must preserve the original accumulator type and kind")
                updates.append({"source": str(statement), "value": str(contribution),
                                "accumulator_operand": "left" if positions[0] == 0 else "right"})
            elif item.root in private and item.root not in active:
                if numeric(value) != item.signature():
                    raise CompilationError("OpenMP private scalar assignment requires matching numerical kinds")
                defined.add(item.root)
            else:
                raise CompilationError("OpenMP reduction body has an unrelated shared or iterator write")
        if len(updates) != 1:
            raise CompilationError("initial OpenMP reduction requires exactly one accumulator update per logical item")
        summary = analysis.segment_summary(procedure, identities, capture_locals=True)
        if not summary["complete"] or any(item["kind"] in {"call", "native_contract", "boundary", "control"}
                                          for item in summary["operations"]):
            raise CompilationError("OpenMP reduction original complete effects are unproved")
        private_roots = set(private) | set(active)
        # The base source summary deliberately describes conservative ordinary
        # Fortran effects. The authenticated clause proves these writes belong
        # to private copies; only the final combine writes the original scalar.
        original_effects = [item for item in summary["ordered_effects"]
                            if item.get("resource") not in private_roots | {accumulator.root}]
        accumulator_read = {"kind": "read", "resource": accumulator.root, "rank": 0,
            "position": "retain original scalar value for the reduction combine",
            "guard_frames": [{"procedure": procedure, "condition": value} for value in record["guard"]],
            "source_form": "original_openmp_clause_reduction", "source_access": accumulator.name}
        accumulator_commit = {"kind": "write", "resource": accumulator.root, "rank": 0,
            "position": "original proven region completion", "guard_frames": accumulator_read["guard_frames"],
            "source_form": "original_openmp_clause_reduction", "source_access": accumulator.name,
            "repeats_procedure_entry_definition_event": False}
        record.update({"contribution": updates[0], "loop_domains": domains,
            "private_resources": sorted(private_roots),
            "reduction_private_copies": {"original_resource": accumulator.root, "initializer": 0,
                "original_application_writes_before_commit": False},
            "per_item_statements": [str(statement) for statement in body],
            "canonical_reads": sorted(root for root, item in reads.items() if item.rank),
            "scalar_reads": sorted({root for root, item in reads.items() if not item.rank} | {accumulator.root}),
            "descriptor_reads": sorted(descriptors),
            "ordered_effects": [accumulator_read, *original_effects, accumulator_commit],
            "effect_summary_identity": summary["summary_identity"],
            "demand_identity": summary["demand_identity"],
            "native_completion": {"available": True, "kind": "original joined OpenMP reduction region",
                "retains_original_team_and_directives": True, "requires_serial_caller": True,
                "ordinary_completion_token": False},
            "scalar_publication": {"resource": accumulator.root, "position": "original proven region completion",
                "combines_original_value_once": True, "original_owner": True},
            "numerical_contract": {"kind": "original_openmp_builtin_sum",
                "combination_order": "unspecified by the original OpenMP reduction clause",
                "per_contribution_math": "preserve original computation and supported IEEE behavior",
                "serial_sum_permission": False, "field_tolerances": "unchanged"},
            "requirements": ["original accumulator is defined before the reached region",
                "original serial caller and complete native team/join semantics",
                "exact physical read mapping and checked original descriptor/bound guards before execution",
                "preserve original per-contribution numerical and IEEE contracts",
                "unchanged complete-field tolerances; no exception or rounding mode assumed",
                "one scalar commit at completion; resource failure after work never replays the owner",
                "generated executor and generic offline reduction calibration before GPU placement"]})
        if accumulator.dtype == "integer":
            raise CompilationError("integer OpenMP SUM requires a proof that every partial sum is representable")
        available = True
    except CompilationError as error:
        reason = str(error)
    payload = {"version": OPENMP_REDUCTION_PROOF_VERSION, "procedure": procedure, "graph": graph.identity,
               "nodes": identities, "available": available, "reason": reason, "record": record}
    identity = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    registry = getattr(analysis, "_omp_reduction_proofs", None)
    if registry is None:
        registry = analysis._omp_reduction_proofs = {}
    if identity not in registry:
        if len(registry) >= analysis.procedure_limit:
            registry.pop(next(iter(registry)))
        registry[identity] = OpenMPReductionProof(procedure, graph.identity, identity, identities,
                                               available, reason, _freeze(record), originals)
    return registry[identity]
