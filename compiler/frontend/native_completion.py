"""Compiler-owned completion tokens for whole original native OpenMP groups."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.call_bindings import _original_call_control
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError, SourceLocation
from compiler.ir.integers import integer_literal

NATIVE_COMPLETION_VERSION = 7
NATIVE_FIXED_ARRAY_COMPLETION_VERSION = 4
NATIVE_FORMAL_ARRAY_COMPLETION_VERSION = 5


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _directive(node):
    if _kind(node) != "Comment":
        return None
    text = str(node).lstrip().lower()
    return text[5:].strip() if text.startswith("!$omp") else None


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


def _joined_completion_facts(analysis, procedure, selected, *, call_completion=None, worksharing=None,
                             graph_pointers=None, uniform_sites=None, retirement_schedules=False, native_lifetime=None, section_units=None, reduction_team=None):
    """Verify one original group; serialized facts cannot grant a token.

    The retained native source supplies its original team, private storage,
    directives and synchronization. This proof never moves a worksharing loop
    outside its enclosing complete region or creates a generated OpenMP team.
    """
    graph, identities = analysis._selected_source(procedure, selected)
    deferred = graph.native_group_for_selection(selected)
    if deferred is None and graph.deferred_native_descendants(identities):
        raise CompilationError("deferred native completion requires the exact whole original source selection")
    if reduction_team is not None:
        from compiler.frontend.descriptor_teams import DescriptorTeamProof, _original_group
        if type(reduction_team) is not DescriptorTeamProof or worksharing is not None:
            raise CompilationError("fixed array native completion requires its issued whole-team storage proof")
        reduction_team.validate(analysis, procedure, identities)
        group = _original_group(analysis.routines[procedure], tuple(selected), graph_selection=True)
        if (len(group.original_nodes) != len(reduction_team._originals)
                or any(a is not b for a, b in zip(group.original_nodes, reduction_team._originals, strict=True))
                or not reduction_team.public().get("intrinsic_fixed_array_reductions")):
            raise CompilationError("fixed array native completion changed its complete original team")
    if deferred is not None and not deferred.available and reduction_team is None:
        raise CompilationError("deferred native group completion unavailable: " + str(deferred.reason))
    routine = analysis.routines[procedure]
    originals = tuple(node for identity in identities for node in graph.source_nodes(identity))
    executable = tuple(identity for identity in identities
                       if any(_kind(node) != "Comment" for node in graph.source_nodes(identity)))
    private, explicit_private, written, peeled, uniform_reads = set(), set(), set(), {}, {}
    section_count = 0
    environment, terminals = {}, {}
    private_arrays = {}

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
    if worksharing is not None and any(
            re.match(r"(?:sections|section|end\s+sections)(?:\s|$)", _directive(node) or "")
            for original in originals for node in walk(original)):
        # Never return only the ordinary worksharing loops from a larger team
        # containing sections, even if its END SECTIONS has a barrier. Native
        # sections remain an indivisible original operation through its join.
        raise CompilationError("native SECTIONS requires whole original joined-team authority; worksharing cuts are unsupported")
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
                field = native_lifetime.field(routine, target) if native_lifetime is not None else None
                binding = field.binding if field is not None else analysis._binding(
                    analysis.source_scope_for(node, routine.scope), target)
                if binding is not None:
                    written.add(binding.root)
            elif _kind(node) == "Loop_Control" and node.items[1] is not None:
                binding = analysis._binding(analysis.source_scope_for(node, routine.scope), node.items[1][0])
                if binding is not None:
                    private.add(binding.root)
            elif _kind(node) == "Call_Stmt":
                if call_completion is None:
                    try:
                        proof = analysis.native_environment(procedure, node)
                    except CompilationError as error:
                        raise CompilationError("joined native call completion requires a separate complete source-call proof: "
                                               + str(error)) from error
                    if worksharing is not None:
                        raise CompilationError("native IEEE environment operations require whole original joined-team authority; worksharing cuts are unsupported")
                    environment[id(node)] = proof
                    written.update(proof.written_roots)
                else:
                    completed = call_completion(node)
                    from compiler.frontend.resource_reachability import NativeTerminatingCall
                    if type(completed) is NativeTerminatingCall:
                        if completed.procedure != procedure or completed.original is not node:
                            raise CompilationError("native terminal completion changed its original call position")
                        terminals[id(node)] = completed.validate(analysis)
                    else:
                        written.update(completed)

    def local_private(binding):
        if graph_pointers is not None and any(pointer.binding is binding for pointer in graph_pointers):
            # Only the descriptor is private. Its pointed-to record remains a
            # shared effect in the separately authenticated native graph.
            return
        if (binding.attributes & {"save", "pointer", "allocatable", "optional", "volatile", "asynchronous", "parameter"}
                or not binding.root.startswith(procedure + "::")
                or binding.dtype not in {"real", "integer", "logical"} or binding.kind not in {4, 8}):
            raise CompilationError("native PRIVATE requires original fixed numeric local storage")
        if binding.rank:
            if any(binding.root in proof.state_roots for proof in environment.values()):
                analysis.native_environment_state(procedure, binding).validate(analysis, procedure, binding)
                return
            from compiler.frontend.native_private import original_private_array
            private_arrays[binding.root] = original_private_array(routine, binding)

    def clauses(text, original):
        remainder = text.strip()
        while remainder:
            pattern = r"(private|shared|default|collapse|schedule|reduction)\s*\(([^()]*)\)\s*" if reduction_team is not None else r"(private|shared|default|collapse|schedule)\s*\(([^()]*)\)\s*"
            match = re.match(pattern, remainder)
            if match is None:
                raise CompilationError("unsupported joined native OpenMP clause: " + remainder)
            kind, values = match.group(1), match.group(2).strip()
            if kind == "reduction":
                team_record = reduction_team.public()
                rows = (*team_record.get("intrinsic_scalar_reductions", ()),
                        *team_record["intrinsic_fixed_array_reductions"])
                if not any(row["original_directive"] == str(original) for row in rows):
                    raise CompilationError("fixed array reduction clause lacks exact original storage evidence")
            elif kind == "default":
                if values not in {"shared", "none"}:
                    raise CompilationError("unsupported native OpenMP DEFAULT")
            elif kind == "collapse":
                if values not in {"1", "2", "3", "4"}:
                    raise CompilationError("unsupported native OpenMP COLLAPSE")
            elif kind == "schedule":
                guided = graph_pointers is not None and re.fullmatch(r"guided(?:\s*,\s*[1-9][0-9]*)?", values)
                unchanged = retirement_schedules and re.fullmatch(r"(?:static|dynamic|guided)(?:\s*,\s*[1-9][0-9]*)?", values)
                if values not in {"static", "runtime"} and not guided and not unchanged:
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
                        explicit_private.add(binding.root)
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
        return fact

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
        points = []
        for binding, reference in references(analysis, scope, condition):
            if binding.root in private or binding.root in written:
                raise CompilationError("joined native OpenMP condition is private or changes inside the complete region")
            if binding.rank and _kind(reference) in {"Part_Ref", "Data_Ref"}:
                fact = uniform_array(binding, reference, scope, guarded=guarded, compound=compound)
                if fact not in points:
                    points.append(fact)
            if binding.attributes & {"pointer", "optional", "volatile", "asynchronous"}:
                raise CompilationError("joined native OpenMP condition association or observation is uncertain")
        if uniform_sites is not None:
            if len(uniform_sites) >= analysis.operation_limit:
                raise CompilationError("uniform condition site inventory budget exhausted")
            uniform_sites.append((header, tuple(points)))

    def check_loop(loop):
        normalize((loop,))
        body = content(loop)
        if not body or _kind(body[-1]) != "End_Do_Stmt":
            raise CompilationError("native worksharing loop requires a complete counted DO")
        if any(_directive(item) is not None for original in body for item in walk(original)):
            raise CompilationError("nested native OpenMP directives inside a worksharing loop are unsupported")

    def check_section_loop(loop):
        check_loop(loop)
        statements = {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Assignment_Stmt", "If_Stmt",
                      "If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt", "Continue_Stmt"}
        for nested in walk(loop):
            if _kind(nested).endswith("_Stmt") and _kind(nested) not in statements:
                raise CompilationError("native SECTION body has unsupported call, lifetime or exit effects: " + _kind(nested))
            if _kind(nested) != "Block_Nonlabel_Do_Construct":
                continue
            normalize((nested,))
            original = content(nested)
            if (not original or _kind(original[0]) != "Nonlabel_Do_Stmt"
                    or original[0].items[1] is None or original[0].items[1].items[1] is None):
                raise CompilationError("native SECTION body requires complete ordinary counted DO loops")
            control = original[0].items[1]
            binding = analysis._binding(analysis.source_scope_for(control, routine.scope), control.items[1][0])
            if binding is None or binding.root not in section_private:
                raise CompilationError("native SECTION counted DO iterator requires explicit original PRIVATE storage")

    if first is None or not re.match(r"parallel(?:\s|$)", first):
        raise CompilationError("native operation is not one complete PARALLEL region")
    combined = bool(re.match(r"parallel\s+do(?:\s|$)", first))
    clauses(first[len("parallel do"):] if combined else first[len("parallel"):], nodes[0])
    # A PRIVATE clause on a preceding worksharing DO ends with that DO; it
    # cannot establish the ownership of an ordinary loop in a later SECTION.
    section_private = frozenset(explicit_private)

    def body(items, *, branch_depth=0):
        nonlocal section_count
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
                if _kind(node) == "Call_Stmt" and id(node) in environment:
                    index += 1
                    continue
                if _kind(node) != "Comment":
                    raise CompilationError("native PARALLEL body requires bounded worksharing DO loops")
                index += 1
                continue
            if directive == "sections":
                from compiler.frontend.native_group_structure import original_section_loops
                loops, index = original_section_loops(items, index, limit=analysis.operation_limit,
                                                     inventory=section_units)
                for loop in loops:
                    check_section_loop(loop)
                section_count += len(loops)
                if section_count > analysis.operation_limit:
                    raise CompilationError("native SECTION source-unit budget exhausted")
                continue
            if not re.match(r"do(?:\s|$)", directive):
                raise CompilationError("unsupported joined native OpenMP directive: " + directive)
            clauses(directive[2:], node)
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
    if reduction_team is not None:
        record.update(native_only=True, internal_cuts_authorized=False, gpu_legality_established=False,
                      fixed_array_reduction_team_identity=reduction_team.identity,
                      reduction_arithmetic="unchanged original CPU OpenMP reduction; no reassociation")
    if private_arrays:
        record["original_private_arrays"] = [private_arrays[root] for root in sorted(private_arrays)]
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
    if section_count:
        record.update(native_sections_contract="original-explicit-counted-sections-v1",
                      original_section_count=section_count, native_only=True,
                      internal_cuts_authorized=False, section_independence_established=False,
                      gpu_legality_established=False)
    if environment:
        record.update(native_environment_contract="original-ieee-halting-mode-v1",
                      native_environment_operations=[proof.public() for proof in environment.values()],
                      native_only=True, internal_cuts_authorized=False,
                      requires_completed_device_work=True, gpu_legality_established=False)
    if terminals:
        record.update(completion_scope="normal return only", termination_paths_return=False,
                      original_terminal_calls=[item.public() for item in terminals.values()],
                      original_order_and_branch_required=True, motion_authorized=False,
                      internal_cuts_authorized=False, gpu_legality_established=False)
    return graph, identities, executable, tuple(sorted(private)), record


def prove_joined_completion(analysis, procedure, selected):
    """Issue native-effect authority only for the ordinary conservative path."""
    graph, identities, executable, private, record = _joined_completion_facts(analysis, procedure, selected)
    payload = {"version": NATIVE_COMPLETION_VERSION, "procedure": procedure, "graph": graph.identity,
               "selected": identities, "executable": executable, "private": sorted(private), "record": record}
    identity = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return NativeCompletionProof(procedure, graph.identity, identity, identities, executable,
                                 private, _freeze(record))


_GRAPH_COMPLETION_ISSUER = object()


@dataclass(frozen=True)
class NativeGraphCompletionProof:
    """Whole unchanged joined execution conditional on one complete effect graph.

    This token cannot be used by ordinary native-section or worksharing-cut
    consumers. In particular, a private pointer descriptor grants no private
    payload, independence, physical section or numerical authority.
    """
    procedure: str
    identity: str
    selected_node_ids: tuple
    _record: object = field(repr=False, compare=False)
    _graph: object = field(repr=False, compare=False)
    _structure: object = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)
    _call_storage: object = field(default=None, repr=False, compare=False)
    _call_joins: tuple = field(default=(), repr=False, compare=False)
    _reduction_teams: tuple = field(default=(), repr=False, compare=False)
    _context: object = field(default=None, repr=False, compare=False)
    _arena: object = field(default=None, repr=False, compare=False)

    def _require_handle(self, analysis):
        if analysis is not self._analysis or type(self) is not NativeGraphCompletionProof:
            raise CompilationError("native graph completion requires exact registered source and graph authority")
        if getattr(analysis, "_source_session", None) is None:
            if self._context is not None or self._arena is not None or self._reduction_teams:
                raise CompilationError("fixed array completion requires its sealed source authority")
            return
        context = analysis.source_authority_context()
        if self._context is not context:
            raise CompilationError("native graph completion changed its issuing source context")
        context.require(analysis)
        arena = context.derivation_arena(analysis)
        if not arena.accepts(self._arena):
            raise CompilationError("native graph completion belongs to a discarded generation")
        try:
            arena.require_handle(("native_graph_completion", id(analysis), self.identity), self)
        except CompilationError as error:
            raise CompilationError("native graph completion requires exact registered source and graph authority: " + str(error)) from error

    def validate(self, analysis, procedure, selected):
        attachments = _graph_completion_attachments(self)
        identity = self.identity
        session = getattr(analysis, "_source_session", None)
        self._require_handle(analysis)
        registry = analysis._native_graph_completions
        context = getattr(analysis, "_source_authority_context", None)
        arena = getattr(analysis, "_source_derivation_arena", None)
        self._validate_source(analysis, procedure, selected)
        _require_graph_completion_attachments(self, attachments)
        self._require_handle(analysis)
        if (getattr(analysis, "_source_session", None) is not session
                or getattr(analysis, "_source_authority_context", None) is not context
                or getattr(analysis, "_source_derivation_arena", None) is not arena
                or getattr(analysis, "_native_graph_completions", None) is not registry
                or getattr(analysis, "_native_graph_completions", {}).get(identity) is not self):
            raise CompilationError("native graph completion publication authority changed")
        _require_graph_completion_attachments(self, attachments)
        return self

    def _validate_source(self, analysis, procedure, selected):
        if (type(self) is not NativeGraphCompletionProof or self._issuer is not _GRAPH_COMPLETION_ISSUER or analysis is not self._analysis
                or procedure != self.procedure or tuple(selected) != self.selected_node_ids
                or getattr(analysis, "_native_graph_completions", {}).get(self.identity) is not self
                or analysis.structure(procedure) is not self._structure):
            raise CompilationError("native graph completion requires exact registered source and graph authority")
        self._graph.validate(analysis, self._graph.procedure, self._graph._selection)
        rows = self._record.get("fixed_array_reduction_teams", ())
        if rows or self._reduction_teams:
            from compiler.frontend.descriptor_teams import DescriptorTeamProof
            if self._call_storage is not None or not rows or len(rows) != len(self._reduction_teams):
                raise CompilationError("fixed array completion lost its exact team receipts")
            groups = {(name, tuple(map(id, originals))) for name, originals in _native_call_groups(self._graph)}
            for row, team in zip(rows, self._reduction_teams, strict=True):
                if (type(team) is not DescriptorTeamProof
                        or (team.procedure, tuple(map(id, team._originals))) not in groups):
                    raise CompilationError("fixed array completion changed its graph-owned original team")
                team.validate(analysis, team.procedure, team.selected_node_ids)
                if _freeze(team.public()) != row or not row.get("intrinsic_fixed_array_reductions"):
                    raise CompilationError("fixed array completion changed its original reduction storage")
        # Recompute every existing equation, even when both new fields were
        # stripped. Retirement's identity version4 is distinct from its public
        # schema3; ordinary graph records retain their original version2 bytes.
        formal_rows = _formal_array_call_rows(analysis, self._graph, self._reduction_teams)
        if formal_rows != _thaw(self._record.get("fixed_array_call_instances", ())):
            raise CompilationError("native array completion changed its exact incoming CALL coverage")
        version = _graph_completion_version(self._record, self._call_storage)
        expected = sha256(json.dumps({"version": version, "native_graph": self._graph.identity,
            "procedure": procedure, "selected": tuple(selected), "record": _thaw(self._record)},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if expected != self.identity:
            raise CompilationError("native graph completion record changed")
        if self._call_storage is not None:
            if self._record.get("storage_identity") != self._call_storage.identity:
                raise CompilationError("native call retirement changed its independent storage proof")
            _require_native_call_retirement(analysis, self._graph, self._call_storage)
            expected = tuple((item["procedure"], tuple(item["selected_node_ids"]),
                              item["ordinary_completion"]["proof_identity"])
                             for item in self._record["original_groups"] if "ordinary_completion" in item)
            retained = {(name, tuple(ids)): proof for template in self._graph.templates
                        for name, ids, proof in template.completion}
            if tuple((name, tuple(ids), proof.identity) for name, ids, proof in self._call_joins) != expected:
                raise CompilationError("native call retirement lost its original group completion receipts")
            for name, identities, proof in self._call_joins:
                if retained.get((name, tuple(identities))) is not proof:
                    raise CompilationError("native call retirement replaced a graph-owned completion receipt")
                proof.validate(analysis, name, identities)
        elif not self._graph.completion_eligible:
            raise CompilationError("native graph has unresolved effect or association barriers")
        return self

    def public(self):
        return {"schema_version": NATIVE_FORMAL_ARRAY_COMPLETION_VERSION if self._record.get("fixed_array_call_instances") else NATIVE_FIXED_ARRAY_COMPLETION_VERSION if self._reduction_teams else 3 if self._call_storage is not None else 2,
                "proof_identity": self.identity, "procedure": self.procedure,
                "native_graph_identity": self._graph.identity, **_thaw(self._record)}



def _graph_completion_attachments(proof):
    from compiler.frontend.descriptor_teams import _original_call_receipts

    owners = [proof]
    formal = [team for team in proof._reduction_teams if team._fixed_array_associations]
    if formal:
        owners.extend((proof._graph, *proof._graph.templates, *proof._graph.instances))
        for instance in proof._graph.instances:
            if instance.call is not None:
                owners.extend((instance.call, instance.call.edge,
                               *_original_call_receipts(instance.call.edge.resolved)))
        for team in formal:
            owners.extend((team, *team._fixed_array_associations))
            for item in team._fixed_array_associations:
                owners.extend(_original_call_receipts(item._resolved))
    return tuple((owner, item.name, object.__getattribute__(owner, item.name))
                 for owner in owners for item in fields(type(owner)))


def _require_graph_completion_attachments(proof, attachments):
    if any(object.__getattribute__(owner, name) is not value for owner, name, value in attachments):
        raise CompilationError("native graph completion attachments changed during publication")


def _publish_graph_completion(analysis, registry, proof):
    session = getattr(analysis, "_source_session", None)
    context = analysis.source_authority_context() if getattr(analysis, "_source_session", None) is not None else None
    arena = context.derivation_arena(analysis) if context is not None else None
    if context is None and proof._reduction_teams:
        raise CompilationError("fixed array graph completion requires a sealed original source session")
    object.__setattr__(proof, "_context", context)
    object.__setattr__(proof, "_arena", arena)
    attachments = _graph_completion_attachments(proof)
    identity = proof.identity
    registry[identity] = proof
    try:
        proof._validate_source(analysis, proof.procedure, proof.selected_node_ids)
        analysis.verify_publication()
        if context is not None:
            context.require(analysis)
            if context.derivation_arena(analysis) is not arena:
                raise CompilationError("native graph completion issuing generation changed")
        if (getattr(analysis, "_source_session", None) is not session
                or (context is not None and (
                    getattr(analysis, "_source_authority_context", None) is not context
                    or getattr(analysis, "_source_derivation_arena", None) is not arena))
                or getattr(analysis, "_native_graph_completions", None) is not registry
                or registry.get(identity) is not proof):
            raise CompilationError("native graph completion registry changed before publication")
        # No delegated source/publication check follows this attachment fence.
        # Frozen records and tuples make raw field identity sufficient here.
        _require_graph_completion_attachments(proof, attachments)
        if context is not None:
            arena.register_handle(("native_graph_completion", id(analysis), identity), proof)
        return proof
    except BaseException:
        if registry.get(identity) is proof:
            del registry[identity]
        raise


def prove_graph_joined_completion(analysis, procedure, selected, native_graph):
    """Validate joined structure using that graph's exact original call closure."""
    from compiler.frontend.resource_reachability import NativeEffectGraph

    if type(native_graph) is not NativeEffectGraph:
        raise CompilationError("native joined graph completion requires an issued native effect graph")
    native_graph.validate(analysis, native_graph.procedure, native_graph._selection)
    if not native_graph.completion_eligible:
        raise CompilationError("native graph has unresolved effect or association barriers")
    checked, reduction_teams = {}, []
    for obligation in native_graph.deferred_completion:
        key = obligation.procedure, tuple(map(id, obligation.originals))
        checked[key] = _graph_joined_facts(analysis, obligation.procedure, obligation.originals, native_graph, reduction_teams=reduction_teams)
    key = procedure, tuple(map(id, selected))
    ordinary = None
    if key not in checked:
        _structure, identities = analysis._selected_source(procedure, selected)
        retained = [proof for template in native_graph.templates for name, original_ids, proof in template.completion
                    if name == procedure and tuple(original_ids) == identities]
        if len(retained) != 1 or type(retained[0]) is not NativeCompletionProof:
            raise CompilationError("joined group is absent from the complete native graph's obligations")
        ordinary = retained[0].validate(analysis, procedure, identities)
        # A previously proved ordinary join is retained independently in the
        # same graph. Verify the full graph-owned structure as for a deferred
        # group; absence of an obligation never supplies completion evidence.
        checked[key] = _graph_joined_facts(analysis, procedure, selected, native_graph, reduction_teams=reduction_teams)
    structure, identities, record = checked[key]
    if ordinary is not None:
        record["retained_ordinary_completion_identity"] = ordinary.identity
    if reduction_teams:
        record["fixed_array_reduction_teams"] = [team.public() for team in reduction_teams]
    formal_rows = _formal_array_call_rows(analysis, native_graph, reduction_teams)
    if formal_rows:
        record["fixed_array_call_instances"] = formal_rows
    identity = sha256(json.dumps({"version": _graph_completion_version(record), "native_graph": native_graph.identity,
                                  "procedure": procedure, "selected": identities, "record": record},
                                 sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    registry = getattr(analysis, "_native_graph_completions", None)
    if registry is None:
        registry = analysis._native_graph_completions = {}
    if identity in registry:
        return registry[identity].validate(analysis, procedure, identities)
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("native graph completion proof registry budget exhausted")
    proof = NativeGraphCompletionProof(procedure, identity, identities, _freeze(record), native_graph,
                                        structure, analysis, _GRAPH_COMPLETION_ISSUER, _reduction_teams=tuple(reduction_teams))
    return _publish_graph_completion(analysis, registry, proof)


def _graph_completion_version(record, call_storage=None):
    return (4 if call_storage is not None else NATIVE_FORMAL_ARRAY_COMPLETION_VERSION
            if record.get("fixed_array_call_instances") else NATIVE_FIXED_ARRAY_COMPLETION_VERSION
            if record.get("fixed_array_reduction_teams") else 2)


def _incoming_array_calls(analysis, native_graph, procedure, binding):
    instances = [item for item in native_graph.instances if item.template.procedure == procedure]
    if not instances:
        raise CompilationError("native reduction formal has no original incoming CALL instance")
    for instance in instances:
        if (instance.call is None or instance.parent is None
                or not 0 <= instance.ordinal < len(native_graph.instances)
                or not 0 <= instance.parent < instance.ordinal
                or native_graph.instances[instance.ordinal] is not instance):
            raise CompilationError("native reduction formal requires every original incoming CALL")
        parent = native_graph.instances[instance.parent]
        edge = instance.call.edge
        if (not any(item is instance.call for item in parent.template.calls)
                or edge.caller != parent.template.procedure or edge.callee != procedure):
            raise CompilationError("native reduction CALL lost its exact original graph edge")
        edge.resolved.validate(analysis, edge.resolved._scope, edge.original)
        mapping = next((item for item in edge.resolved.associations if item.formal_binding is binding), None)
        if mapping is None:
            raise CompilationError("native reduction CALL lost its exact formal association")
        yield instance, edge, mapping


def _derive_formal_array_calls(analysis, native_graph, procedure, binding):
    from compiler.frontend.native_fixed_array_association import prove_native_fixed_array_association

    context = analysis.source_authority_context()
    proofs = []
    for _instance, edge, _mapping in _incoming_array_calls(analysis, native_graph, procedure, binding):
        parent = context.session.lexical_parent(edge.original)
        selection = (parent,) if _kind(parent) == "If_Stmt" else (edge.original,)
        proof = prove_native_fixed_array_association(analysis, edge.caller, selection, binding.name)
        if not any(previous is proof for previous in proofs):
            proofs.append(proof)
    return tuple(proofs)


def _formal_array_call_rows(analysis, native_graph, teams):
    from compiler.frontend.native_fixed_array_association import NativeFixedArrayAssociation

    by_formal = {}
    for team in teams:
        for proof in team._fixed_array_associations:
            if type(proof) is not NativeFixedArrayAssociation:
                raise CompilationError("native array completion requires exact original association receipts")
            proof.validate(analysis, proof._caller.qualified, proof._selection, proof._formal_name)
            key = team.procedure, id(proof.formal)
            proofs = by_formal.setdefault(key, [])
            if not any(previous is proof for previous in proofs):
                proofs.append(proof)
    rows = []
    for (procedure, _key), proofs in by_formal.items():
        binding = proofs[0].formal
        used = set()
        for instance, edge, mapping in _incoming_array_calls(analysis, native_graph, procedure, binding):
            matched = [proof for proof in proofs if proof._caller.qualified == edge.caller
                       and _original_call_control(proof._selection) is edge.original]
            if len(matched) != 1 or matched[0].actual is not mapping.binding:
                raise CompilationError("native array completion lacks exact association for an incoming CALL")
            proof = matched[0]
            used.add(id(proof))
            rows.append({"instance": instance.ordinal, "caller": edge.caller, "callee": procedure,
                         "formal": binding.name, "association_identity": proof.identity,
                         "original_call": str(edge.original)})
        if used != {id(proof) for proof in proofs}:
            raise CompilationError("native array completion has an unrelated original CALL association")
    return sorted(rows, key=lambda item: (item["instance"], item["callee"], item["formal"]))


def _graph_joined_facts(analysis, procedure, selected, native_graph, *, retirement_storage=None, reduction_teams=None):
    native_graph.validate(analysis, native_graph.procedure, native_graph._selection)
    template = next((item for item in native_graph.templates if item.procedure == procedure), None)
    if template is None:
        raise CompilationError("joined native procedure is absent from the effect graph")
    # Whole-call retirement has already published and destroyed the domain.
    # Its original plain team may keep native worksharing schedules through the
    # enclosing join; ordinary payload consumers retain their stronger limit.
    if retirement_storage is not None:
        _require_native_call_retirement(analysis, native_graph, retirement_storage)
    directives = [_directive(node) for original in selected for node in walk(original)
                  if _directive(node) is not None]
    reduction_team = None
    if reduction_teams is not None and any(re.search(r"\breduction\s*\(", text) for text in directives):
        from compiler.frontend.descriptor_teams import _FixedArrayAssociationRequired, prove_descriptor_team
        associations, requested = (), set()
        while True:
            try:
                team = prove_descriptor_team(analysis, procedure, tuple(selected),
                    fixed_array_associations=associations)
                break
            except _FixedArrayAssociationRequired as error:
                if type(error) is not _FixedArrayAssociationRequired:
                    raise
                routine = analysis._require_original(procedure)
                binding = error.binding
                if (id(binding) in requested or len(requested) >= len(routine.arguments)
                        or routine.scope.bindings.get(binding.name) is not binding
                        or binding.name not in routine.arguments):
                    raise CompilationError("native array completion has an invalid formal association request") from error
                requested.add(id(binding))
                associations += _derive_formal_array_calls(analysis, native_graph, procedure, binding)
        if team.public().get("intrinsic_fixed_array_reductions"):
            reduction_team = team.validate(analysis, procedure, team.selected_node_ids)
            if all(previous is not team for previous in reduction_teams):
                reduction_teams.append(team)
    opening = r"parallel(?:\s|$)" if retirement_storage is not None or reduction_team is not None else r"parallel\s+do(?:\s|$)"
    if not directives or not re.match(opening, directives[0]):
        raise CompilationError("graph-owned native completion requires one complete original PARALLEL DO")
    calls = {id(instance.call.edge.original): instance.call.edge
             for instance in native_graph.instances if instance.call is not None}

    terminals = {id(item.original): item for _ordinal, item in native_graph.terminal_calls}

    def complete_call(node):
        terminal = terminals.get(id(node))
        if terminal is not None:
            if terminal.procedure != procedure:
                raise CompilationError("native terminal belongs to another original procedure")
            return terminal.validate(analysis)
        edge = calls.get(id(node))
        if edge is None or edge.caller != procedure:
            raise CompilationError("native joined call is absent from the complete instantiated graph")
        edge.resolved.validate(analysis, edge.resolved._scope, node)
        # No call is hoisted across a worksharing bound or shared predicate in
        # this combined-loop-only contract. All payload effects remain in graph.
        return ()

    selected_nodes = {id(node) for original in selected for node in walk(original)}
    pointers = tuple(item for item in template.confined_pointers if id(item.loop) in selected_nodes)
    lifetime = None
    if retirement_storage is not None and retirement_storage._native_graph is not None:
        from compiler.frontend.native_lifetime_context import NativeLifetimeContext
        lifetime = NativeLifetimeContext(analysis, native_graph)
    structure, identities, _executable, private, record = _joined_completion_facts(
        analysis, procedure, selected, call_completion=complete_call, graph_pointers=pointers,
        retirement_schedules=retirement_storage is not None, native_lifetime=lifetime, reduction_team=reduction_team)
    if any(pointer.binding.root not in private for pointer in pointers):
        raise CompilationError("confined native pointer requires explicit original PRIVATE descriptor storage")
    record.update(native_only=True, internal_cuts_authorized=False, gpu_legality_established=False,
                  effect_graph_required=True, completion_scope="normal return only",
                  termination_paths_return=False, motion_authorized=False,
                  original_observable_terminal_instances=[{"instance": ordinal, **item.public()}
                      for ordinal, item in native_graph.terminal_calls],
                  private_pointer_descriptors=[item.binding.root for item in pointers],
                  private_pointer_payload_authority=False)
    return structure, identities, record


def _native_call_groups(native):
    native.validate(native._analysis, native.procedure, native._selection)
    from compiler.frontend.resource_reachability import NativeEffectNode, NativePresenceConditional

    groups = {}
    for template in native.templates:
        pending = list(template.nodes)
        while pending:
            node = pending.pop()
            if isinstance(node, NativeEffectNode):
                if node.kind == "joined_group":
                    groups[template.procedure, tuple(map(id, node.original))] = tuple(node.original)
                pending.extend(node.items)
            elif isinstance(node, NativePresenceConditional):
                pending.extend((*node.present, *node.absent))
    return tuple((name, originals) for (name, _ids), originals in groups.items())


def native_call_storage_structures(analysis, native):
    """Existing callees of one whole original CALL or single-line IF CALL."""
    from compiler.frontend.descriptor_teams import _FixedArrayAssociationRequired, prove_descriptor_team
    from compiler.frontend.resource_reachability import NativeEffectGraph

    if type(native) is not NativeEffectGraph:
        raise CompilationError("native call retirement requires its issued original effect graph")
    native.validate(analysis, native.procedure, native._selection)
    if (_original_call_control(native._selection) is None
            or not native.retirement_effects_eligible):
        raise CompilationError("native call retirement has unresolved original effects or selection")
    result = []
    lifetime = None
    if any(template.confined_pointers for template in native.templates):
        from compiler.frontend.native_lifetime_context import NativeLifetimeContext
        lifetime = NativeLifetimeContext(analysis, native)
    for name, originals in _native_call_groups(native):
        if name not in analysis.routines or name == native.procedure:
            raise CompilationError("native call team requires an original complete subroutine callee")
        try:
            team = prove_descriptor_team(analysis, name, originals,
                native_graph=native if lifetime is not None and lifetime.team_bindings(name, originals) else None)
        except _FixedArrayAssociationRequired as error:
            if type(error) is not _FixedArrayAssociationRequired:
                raise
            # A formal reduction needs the same graph-owned incoming CALL
            # receipts as ordinary joined completion. Descriptor storage alone
            # cannot manufacture that association or grant payload access.
            completion = prove_graph_joined_completion(analysis, name, originals, native)
            selected = analysis._selected_source(name, originals)[1]
            completion.validate(analysis, name, selected)
            matches = [proof for proof in completion._reduction_teams
                       if proof.procedure == name and len(proof._originals) == len(originals)
                       and all(left is right for left, right in zip(proof._originals, originals, strict=True))]
            if len(matches) != 1 or not matches[0]._fixed_array_associations:
                raise CompilationError("native call storage lost its exact formal reduction team") from error
            team = matches[0]
        result.append(team)
    for template in native.templates:
        result.extend(template.worksharing)
    return tuple({id(proof): proof for proof in result}.values())


def _require_native_call_retirement(analysis, native, storage):
    from compiler.frontend.resource_reachability import NativeAllocationObligation, NativeWorksharingObligation
    from compiler.frontend.storage_mutation_effects import StorageMutationEffects, _resource

    native.validate(analysis, native.procedure, native._selection)
    if (_original_call_control(native._selection) is None
            or not native.retirement_effects_eligible or type(storage) is not StorageMutationEffects):
        raise CompilationError("native call retirement requires exact independent graph and storage authority")
    storage.validate(analysis, native.procedure, selection=native._selection)
    record = storage._record
    if not record["complete"] or record["escapes"] or type(storage._joined_storage) is not tuple:
        raise CompilationError("native call retirement requires complete nonescaping original callee storage")
    if storage._native_graph is not None and storage._native_graph is not native:
        raise CompilationError("native retirement changed its independent lifetime graph")
    structures = storage._joined_storage
    from compiler.frontend.descriptor_teams import DescriptorTeamProof

    for proof in structures:
        if type(proof) is DescriptorTeamProof:
            proof.validate(analysis, proof.procedure, proof.selected_node_ids)
            if proof._native_graph is not None and proof._native_graph is not native:
                raise CompilationError("native retirement changed its confined-pointer team graph")
        else:
            proof.validate(analysis, proof._routine)
    # Storage keeps the exact team handles; retirement does not become an
    # executable fixed-array completion. Recheck every formal association
    # against this original CALL graph at issue and subsequent publication.
    _formal_array_call_rows(analysis, native,
        tuple(proof for proof in structures if type(proof) is DescriptorTeamProof))
    groups = _native_call_groups(native)
    expected_teams = {(name, analysis._selected_source(name, originals)[1]) for name, originals in groups}
    actual_teams = {(proof.procedure, tuple(proof.selected_node_ids)) for proof in structures
                    if type(proof) is DescriptorTeamProof}
    if actual_teams != expected_teams:
        raise CompilationError("native call retirement lost exact original group storage coverage")
    grouped_nodes = {}
    for name, originals in groups:
        grouped_nodes.setdefault(name, set()).update(id(node) for original in originals for node in walk(original))
    for ordinal, obligation in native.lifetime_obligations:
        if isinstance(obligation, NativeAllocationObligation):
            if (_kind(obligation.original) != "Assignment_Stmt"
                    or obligation.original.items[0] is not obligation.target
                    or not any(member["procedure"] == obligation.procedure and any(
                        event["kind"] == "whole allocatable assignment may replace storage"
                        and event["resource"] == _resource(obligation.binding)
                        and event["original_source"] == str(obligation.original)
                        for event in member["mutations"]) for member in record["closure"])):
                raise CompilationError("native allocation obligation lacks its exact original storage mutation")
        elif isinstance(obligation, NativeWorksharingObligation):
            if not any(proof is obligation.proof for proof in structures):
                raise CompilationError("native orphan obligation lost its exact whole-callee structure")
            obligation.proof.checkpoint()
            instance = native.instances[ordinal]
            if instance.template._selection is not None or instance.template._routine is not obligation.proof._routine:
                raise CompilationError("native orphan completion requires the whole original callee")
            while instance.call is not None:
                edge = instance.call.edge
                if id(edge.original) in grouped_nodes.get(edge.caller, ()):
                    raise CompilationError("orphan worksharing reached through a team requires independent participation")
                instance = native.instances[instance.parent]
        else:
            raise CompilationError("native call retirement contains an unknown lifetime obligation")


def prove_native_call_retirement(analysis, procedure, selected, native, storage):
    """Normal return of one unchanged CALL/control after complete domain retirement.

    Allocation mutations remain explicit. This proof lends neither stable
    storage nor payload access to ordinary native units. A selected IF keeps
    its original predicate, effects, action and source identity throughout.
    """
    _require_native_call_retirement(analysis, native, storage)
    structure, identities = analysis._selected_source(procedure, selected)
    if (procedure != native.procedure or len(selected) != len(native._selection)
            or any(a is not b for a, b in zip(selected, native._selection, strict=True))):
        raise CompilationError("native call retirement changed its original caller selection")
    joins, group_records = [], []
    for name, originals in _native_call_groups(native):
        _graph, selected_ids = analysis._selected_source(name, originals)
        ordinary = [proof for template in native.templates for owner, ids, proof in template.completion
                    if owner == name and ids == selected_ids]
        if len(ordinary) == 1:
            proof = ordinary[0].validate(analysis, name, selected_ids)
            joins.append((name, selected_ids, proof))
            group_records.append({"procedure": name, "selected_node_ids": list(selected_ids),
                                  "ordinary_completion": proof.public()})
        elif ordinary:
            raise CompilationError("native call retirement has ambiguous original group completion")
        else:
            reduction_teams = []
            _graph, checked_ids, facts = _graph_joined_facts(analysis, name, originals, native,
                retirement_storage=storage, reduction_teams=reduction_teams)
            if checked_ids != selected_ids:
                raise CompilationError("native call retirement changed a complete original group")
            if any(not any(team is retained for retained in storage._joined_storage) for team in reduction_teams):
                raise CompilationError("native retirement replaced its exact joined storage team")
            group_records.append({"procedure": name, "selected_node_ids": list(selected_ids),
                                  "graph_completion": facts})
    record = {"kind": "whole_original_native_call_retirement", "available": True,
              "caller_contract": "serial_source_scope", "requires_serial_caller": True,
              "requires_completed_device_work": True, "requires_domain_retirement": True,
              "completion_scope": "normal return only", "termination_paths_return": False,
              "motion_authorized": False, "internal_cuts_authorized": False,
              "gpu_legality_established": False, "payload_access_authorized": False,
              "stable_storage_authorized": False, "original_call_unchanged": True,
              "storage_identity": storage.identity, "original_groups": group_records,
              "discharged_lifetime_obligations": [{"instance": ordinal, **item.public()}
                  for ordinal, item in native.lifetime_obligations],
              "original_observable_terminal_instances": [{"instance": ordinal, **item.public()}
                  for ordinal, item in native.terminal_calls]}
    identity = sha256(json.dumps({"version": 4, "procedure": procedure, "selected": identities,
        "native_graph": native.identity, "record": record}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    registry = getattr(analysis, "_native_graph_completions", None)
    if registry is None:
        registry = analysis._native_graph_completions = {}
    if identity in registry:
        return registry[identity].validate(analysis, procedure, identities)
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("native graph completion proof registry budget exhausted")
    proof = NativeGraphCompletionProof(procedure, identity, identities, _freeze(record), native, structure,
        analysis, _GRAPH_COMPLETION_ISSUER, storage, tuple(joins))
    return _publish_graph_completion(analysis, registry, proof)
