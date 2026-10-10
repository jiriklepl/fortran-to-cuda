"""Compiler-owned completion tokens for whole original native OpenMP groups."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError, SourceLocation
from compiler.ir.integers import INTEGER_MAX, INTEGER_MIN, integer_literal

NATIVE_COMPLETION_VERSION = 4


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _directive(node):
    text = str(node).lstrip().lower()
    return text[5:].strip() if _kind(node) == "Comment" and text.startswith("!$omp") else None


@dataclass(frozen=True)
class NativeCompletionProof:
    procedure: str
    structured_identity: str
    identity: str
    selected_node_ids: tuple[str, ...]
    executable_node_ids: tuple[str, ...]
    private_roots: tuple[str, ...]
    _record: object = field(repr=False, compare=False)

    def public(self):
        return {"schema_version": NATIVE_COMPLETION_VERSION, "proof_identity": self.identity,
                "procedure": self.procedure, "structured_identity": self.structured_identity,
                "selected_original_nodes": list(self.selected_node_ids),
                "executable_original_nodes": list(self.executable_node_ids),
                "private_resources": list(self.private_roots), **_thaw(self._record)}

    def validate(self, analysis, procedure, identities):
        graph = analysis.structure(procedure)
        if (procedure != self.procedure or graph.identity != self.structured_identity
                or analysis._joined_completions.get(self.identity) is not self
                or tuple(identities) not in {self.selected_node_ids, self.executable_node_ids}):
            raise CompilationError("native section completion lacks whole original joined-group authority")
        return self


def _joined_completion_facts(analysis, procedure, selected, *, call_completion=None, worksharing=None):
    """Verify one original group; serialized facts cannot grant a token.

    The retained native source supplies its original team, private storage,
    directives and synchronization. This proof never moves a worksharing loop
    outside its enclosing complete region or creates a generated OpenMP team.
    """
    graph, identities = analysis._selected_source(procedure, selected)
    deferred = graph.native_group_for_selection(selected)
    if deferred is None and graph.deferred_native_descendants(identities):
        raise CompilationError("deferred native completion requires the exact whole original source selection")
    if deferred is not None and not deferred.available:
        raise CompilationError("deferred native group completion unavailable: " + str(deferred.reason))
    routine = analysis.routines[procedure]
    originals = tuple(node for identity in identities for node in graph.source_nodes(identity))
    executable = tuple(identity for identity in identities
                       if any(_kind(node) != "Comment" for node in graph.source_nodes(identity)))
    private, written, peeled, uniform_reads = set(), set(), {}, {}

    def content(node):
        return peeled.get(id(node), _children(node))

    def normalize(items):
        result = []
        for item in items:
            if (_kind(item) in {"Block_Nonlabel_Do_Construct", "If_Construct", "Associate_Construct"}
                    and id(item) not in peeled):
                body = list(_children(item))
                prefix = []
                while body and _kind(body[0]) == "Comment":
                    prefix.append(body.pop(0))
                # A parent normalization may already have exposed the exact
                # attached original directives. Re-entering its body must
                # not insert that prefix a second time (or invent a nested
                # PARALLEL/DO before the same associated construct).
                peeled[id(item)] = tuple(body)
                result.extend(prefix)
            result.append(item)
        return result

    # Documentation comments may precede an attached original directive.
    # Ignore only those comments for structure; original selection identity
    # still retains them and all OpenMP directives remain authoritative.
    nodes = [node for node in normalize(originals)
             if _kind(node) != "Comment" or _directive(node) is not None]
    first = _directive(nodes[0]) if nodes else None
    combined = bool(re.match(r"parallel\s+do(?:\s|$)", first or ""))
    implicit_join = combined and len(nodes) == 2 and _kind(nodes[1]) == "Block_Nonlabel_Do_Construct"
    if not nodes or (not implicit_join and _directive(nodes[-1]) not in {"end parallel", "end parallel do"}):
        raise CompilationError("native parallel operation requires a joined END PARALLEL")
    for item in originals:
        for node in walk(item):
            if _kind(node) == "Assignment_Stmt":
                target = node.items[0]
                if _kind(target) == "Part_Ref":
                    target = target.items[0]
                binding = analysis._binding(analysis.source_scope_for(node, routine.scope), target)
                if binding is not None:
                    written.add(binding.root)
            elif _kind(node) == "Loop_Control" and node.items[1] is not None:
                binding = analysis._binding(analysis.source_scope_for(node, routine.scope), node.items[1][0])
                if binding is not None:
                    private.add(binding.root)
            elif _kind(node) == "Call_Stmt":
                if call_completion is None:
                    raise CompilationError("joined native call completion requires a separate complete source-call proof")
                written.update(call_completion(node))

    def local_private(binding):
        if (binding.attributes & {"save", "pointer", "allocatable", "optional", "volatile", "asynchronous", "parameter"}
                or not binding.root.startswith(procedure + "::")
                or binding.dtype not in {"real", "integer", "logical"} or binding.kind not in {4, 8}):
            raise CompilationError("native PRIVATE requires original fixed numeric local storage")
        if binding.rank:
            total = 1
            for axis in binding.shape_nodes:
                if _kind(axis) != "Explicit_Shape_Spec":
                    raise CompilationError("native PRIVATE array requires original fixed bounds")
                lo, hi = axis.items
                location = SourceLocation(str(routine.scope.path))
                lower = 1 if lo is None else routine.scope.kinds.integer(lo, location)
                upper = routine.scope.kinds.integer(hi, location)
                if not INTEGER_MIN <= lower <= INTEGER_MAX or not INTEGER_MIN <= upper <= INTEGER_MAX:
                    raise CompilationError("native PRIVATE array bounds exceed the INTEGER ABI")
                total *= max(0, upper - lower + 1)
                if total > 256:
                    raise CompilationError("native PRIVATE array exceeds bounded fixed storage")

    def clauses(text):
        remainder = text.strip()
        while remainder:
            match = re.match(r"(private|shared|default|collapse|schedule)\s*\(([^()]*)\)\s*", remainder)
            if match is None:
                raise CompilationError("unsupported joined native OpenMP clause: " + remainder)
            kind, values = match.group(1), match.group(2).strip()
            if kind == "default":
                if values not in {"shared", "none"}:
                    raise CompilationError("unsupported native OpenMP DEFAULT")
            elif kind == "collapse":
                if values not in {"1", "2", "3", "4"}:
                    raise CompilationError("unsupported native OpenMP COLLAPSE")
            elif kind == "schedule":
                if values not in {"static", "runtime"}:
                    raise CompilationError("unsupported native OpenMP SCHEDULE")
            else:
                for name in values.split(","):
                    if not re.fullmatch(r"[a-z][a-z0-9_]*", name.strip()):
                        raise CompilationError("native OpenMP clause needs resolved variable names")
                    binding = analysis._binding(routine.scope, name.strip())
                    if binding is None:
                        raise CompilationError("unresolved native OpenMP clause variable")
                    if kind == "private":
                        local_private(binding)
                        private.add(binding.root)
            remainder = remainder[match.end():].lstrip(", ")

    def constant_index(scope, node):
        """Authenticate one literal or original INTEGER PARAMETER, not a read."""
        location = SourceLocation(str(scope.path))
        kind, items = _kind(node), _children(node)
        if kind == "Parenthesis":
            return constant_index(scope, items[1])
        if len(items) == 2 and str(items[0]) in {"+", "-"}:
            value, parameter = constant_index(scope, items[1])
            value = -value if str(items[0]) == "-" else value
            return integer_literal(str(value), location), parameter
        if kind == "Int_Literal_Constant":
            width = 4 if node.items[1] is None else scope.kinds.integer(node.items[1], location)
            if width not in {4, 8} or not 0 <= int(node.items[0]) < 2 ** (width * 8 - 1):
                raise CompilationError("uniform array subscript requires a supported INTEGER literal")
            return integer_literal(str(node.items[0]), location), None
        if kind == "Name":
            binding = analysis._binding(scope, node)
            if (binding is None or binding.rank or binding.dtype != "integer" or binding.kind not in {4, 8}
                    or "parameter" not in binding.attributes or binding.root in private | written):
                raise CompilationError("uniform array subscript requires an original INTEGER PARAMETER")
            reason = analysis.resource_identity_boundary(binding)
            if reason:
                raise CompilationError(reason)
            owner = binding.declaring_scope
            if owner is None:
                raise CompilationError("uniform array subscript PARAMETER lacks original declaration authority")
            value = owner.kinds.integer(binding.name, SourceLocation(str(owner.path)))
            return integer_literal(str(value), location), binding.root
        raise CompilationError("uniform array subscript requires a literal or original INTEGER PARAMETER")

    def storage_ownership(binding, scope):
        # THREADPRIVATE belongs to the original declaring specification, not
        # to a synthetic dummy or to whichever spelling an importing scope uses.
        # Storage association is deliberately outside this narrow proof.
        seen = set()
        for start in (binding.declaring_scope, scope):
            owner = start
            while owner is not None and id(owner) not in seen:
                seen.add(id(owner))
                specification = next((node for node in _children(owner.node)
                                      if _kind(node) == "Specification_Part"), None)
                for node in walk(specification):
                    if _kind(node) in {"Common_Stmt", "Equivalence_Stmt"}:
                        raise CompilationError("uniform array condition storage association is uncertain")
                    directive = _directive(node)
                    if directive is None or not directive.startswith("threadprivate"):
                        continue
                    match = re.fullmatch(r"threadprivate\s*\(([^()]*)\)", directive)
                    if match is None:
                        raise CompilationError("uniform array condition THREADPRIVATE ownership is uncertain")
                    for name in match[1].split(","):
                        original = analysis._binding(owner, name.strip())
                        if original is None:
                            raise CompilationError("uniform array condition THREADPRIVATE ownership is unresolved")
                        if original.root == binding.root:
                            raise CompilationError("uniform array condition storage is THREADPRIVATE")
                owner = owner.parent

    def uniform_array(binding, reference, scope, *, guarded, compound):
        if (binding.rank != 1 or _kind(reference) != "Part_Ref"
                or _kind(reference.items[0]) != "Name"
                or binding.dtype not in {"real", "integer"} or binding.kind not in {4, 8}
                or binding.attributes & {"allocatable", "pointer", "optional", "target", "volatile",
                                         "asynchronous", "parameter", "value"}):
            raise CompilationError("uniform array condition requires fixed shared rank-one numeric storage")
        reason = analysis.resource_identity_boundary(binding)
        if reason:
            raise CompilationError(reason)
        storage_ownership(binding, scope)
        indices = tuple(_children(reference.items[1]))
        if len(indices) != 1:
            raise CompilationError("uniform array condition requires one constant scalar subscript")
        value, parameter = constant_index(scope, indices[0])
        axes = binding.shape_nodes
        if len(axes) != 1:
            raise CompilationError("uniform array condition lacks original rank-one bounds")
        owner = binding.declaring_scope
        if owner is None:
            raise CompilationError("uniform array condition lacks original declaration authority")
        axis = axes[0]
        fact = {"resource": binding.root, "type": binding.dtype, "kind": binding.kind, "rank": 1,
                "source_reference": str(reference), "subscript": value,
                "source_subscript": str(indices[0]), "parameter_resource": parameter,
                "guarded_condition": guarded,
                "compound_logical_condition": compound,
                "shared_and_unwritten_in_complete_team": True,
                "requires_registered_storage_and_alias_validation": True,
                "requires_host_coherence_before_original_team": True,
                "condition_evaluation": "unchanged original team condition; no proof-time payload read"}
        location = SourceLocation(str(owner.path))
        if _kind(axis) == "Explicit_Shape_Spec":
            lower, upper = axis.items
            lo = 1 if lower is None else owner.kinds.integer(lower, location)
            hi = owner.kinds.integer(upper, location)
            integer_literal(str(lo), location)
            integer_literal(str(hi), location)
            if not lo <= value <= hi:
                raise CompilationError("uniform array constant subscript is outside original fixed bounds")
            fact.update(storage="fixed_explicit_shape", original_lower_bound=lo, original_upper_bound=hi)
        elif (_kind(axis) == "Assumed_Shape_Spec" and binding.name in routine.arguments
              and owner is routine.scope):
            if guarded:
                # Native exact hooks prepare possible reads before the team.
                # An inactive assumed-shape point may be outside its actual
                # descriptor even when the original execution is valid.
                raise CompilationError("guarded uniform array condition requires original-position coherence")
            if compound:
                raise CompilationError("compound uniform array condition requires proved fixed bounds")
            fact.update(storage="stable_original_assumed_shape",
                        requires_original_descriptor_and_checked_coordinates=True)
        else:
            raise CompilationError("uniform array condition requires fixed or original assumed-shape storage")
        key = binding.root, str(reference), guarded, compound
        uniform_reads[key] = fact
        if len(uniform_reads) > analysis.operation_limit:
            raise CompilationError("uniform array condition resource budget exhausted")

    def uniform(header, *, guarded):
        condition = header.items[0]
        scope = analysis.source_scope_for(header, routine.scope)
        compound = any(_kind(node) in {"And_Operand", "Or_Operand", "Equiv_Operand"}
                       for node in walk(condition))
        for expression in walk(condition):
            kind = _kind(expression)
            if kind in {"Function_Reference", "Structure_Constructor"}:
                raise CompilationError("joined native OpenMP condition has unproved uniform call effects")
            if kind == "Part_Ref":
                binding = analysis._binding(scope, expression.items[0])
                if binding is None or not binding.rank:
                    raise CompilationError("joined native OpenMP condition has unproved uniform call effects")
            elif kind == "Intrinsic_Function_Reference":
                name = str(expression.items[0]).lower()
                if (name not in {"size", "lbound", "ubound"} or analysis._binding(scope, name)
                        or analysis._candidates(scope, name) or analysis._unknown_exports(scope)):
                    raise CompilationError("joined native OpenMP condition requires uniform scalar or descriptor reads")
        from compiler.frontend.component_bindings import references
        for binding, reference in references(analysis, scope, condition):
            if binding.root in private or binding.root in written:
                raise CompilationError("joined native OpenMP condition is private or changes inside the complete region")
            if binding.rank and _kind(reference) in {"Part_Ref", "Data_Ref"}:
                uniform_array(binding, reference, scope, guarded=guarded, compound=compound)
            if binding.attributes & {"pointer", "optional", "volatile", "asynchronous"}:
                raise CompilationError("joined native OpenMP condition association or observation is uncertain")

    def check_loop(loop):
        normalize((loop,))
        body = content(loop)
        if not body or _kind(body[-1]) != "End_Do_Stmt":
            raise CompilationError("native worksharing loop requires a complete counted DO")
        if any(_directive(item) is not None for original in body for item in walk(original)):
            raise CompilationError("nested native OpenMP directives inside a worksharing loop are unsupported")

    if first is None or not re.match(r"parallel(?:\s|$)", first):
        raise CompilationError("native operation is not one complete PARALLEL region")
    combined = bool(re.match(r"parallel\s+do(?:\s|$)", first))
    clauses(first[len("parallel do"):] if combined else first[len("parallel"):])

    def body(items, *, branch_depth=0):
        items, index = normalize(items), 0
        while index < len(items):
            node, directive = items[index], _directive(items[index])
            if _kind(node) == "If_Construct":
                branch = []
                for item in content(node):
                    label = _kind(item)
                    if label in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                        if branch:
                            body(branch, branch_depth=branch_depth + 1)
                        branch = []
                        if label in {"If_Then_Stmt", "Else_If_Stmt"}:
                            uniform(item, guarded=branch_depth > 0 or label == "Else_If_Stmt")
                    else:
                        branch.append(item)
                index += 1
                continue
            if directive is None:
                if _kind(node) != "Comment":
                    raise CompilationError("native PARALLEL body requires bounded worksharing DO loops")
                index += 1
                continue
            if not re.match(r"do(?:\s|$)", directive):
                raise CompilationError("unsupported joined native OpenMP directive: " + directive)
            clauses(directive[2:])
            index += 1
            while index < len(items) and _kind(items[index]) == "Comment" and _directive(items[index]) is None:
                index += 1
            if index >= len(items) or _kind(items[index]) != "Block_Nonlabel_Do_Construct":
                raise CompilationError("native OpenMP DO requires one complete associated loop")
            check_loop(items[index])
            loop, opening = items[index], node
            index += 1
            while index < len(items) and _kind(items[index]) == "Comment" and _directive(items[index]) is None:
                index += 1
            ending = None
            if index < len(items) and _directive(items[index]) in {"end do", "end do nowait"}:
                ending = items[index]
                index += 1
            elif index < len(items) and (_directive(items[index]) or "").startswith("end do"):
                raise CompilationError("native OpenMP DO has an unsupported ending directive")
            # An omitted END DO has the original implicit worksharing barrier.
            if worksharing is not None:
                worksharing.append((opening, loop, ending))

    if combined:
        items = [item for item in (nodes[1:] if implicit_join else nodes[1:-1]) if _kind(item) != "Comment"]
        if ((not implicit_join and _directive(nodes[-1]) != "end parallel do") or len(items) != 1
                or _kind(items[0]) != "Block_Nonlabel_Do_Construct"):
            raise CompilationError("combined PARALLEL DO requires one complete associated loop and join")
        check_loop(items[0])
    else:
        if _directive(nodes[-1]) != "end parallel":
            raise CompilationError("native PARALLEL operation requires a matching END PARALLEL")
        body(nodes[1:-1])
    record = {"available": True, "reason": "original complete native PARALLEL region joins before coherence commit",
              "caller_contract": "serial_source_scope", "requires_serial_caller": True,
              "has_openmp_in_closure": True, "has_opaque_calls_in_closure": False,
              "join": "implicit combined-loop completion" if implicit_join else "explicit original parallel end",
              "retains_original_team_and_directives": True}
    if uniform_reads:
        record.update(uniform_array_reads=list(uniform_reads.values()),
                      uniform_array_read_contract="fixed-rank-one-shared-constant-point-v1",
                      guarded_fixed_bound_array_conditions_authorized=any(
                          item["guarded_condition"] for item in uniform_reads.values()),
                      guarded_assumed_shape_array_conditions_authorized=False,
                      gpu_independence_established=False)
    if deferred is not None:
        record.update(deferred_native_group_identity=deferred.identity,
                      native_only=True, internal_cuts_authorized=False,
                      bounded_native_units=len(deferred.units))
    return graph, identities, executable, tuple(sorted(private)), record


def prove_joined_completion(analysis, procedure, selected):
    """Issue native-effect authority only for the ordinary conservative path."""
    graph, identities, executable, private, record = _joined_completion_facts(analysis, procedure, selected)
    payload = {"version": NATIVE_COMPLETION_VERSION, "procedure": procedure, "graph": graph.identity,
               "selected": identities, "executable": executable, "private": sorted(private), "record": record}
    identity = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return NativeCompletionProof(procedure, graph.identity, identity, identities, executable,
                                 private, _freeze(record))
