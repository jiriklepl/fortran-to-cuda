"""Compiler-owned completion tokens for whole original native OpenMP groups."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re

from fparser.two.utils import walk

from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError, SourceLocation
from compiler.ir.integers import INTEGER_MIN, INTEGER_MAX

NATIVE_COMPLETION_VERSION = 1


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


def _joined_completion_facts(analysis, procedure, selected, *, call_completion=None):
    """Verify one original group; serialized facts cannot grant a token.

    The retained native source supplies its original team, private storage,
    directives and synchronization. This proof never moves a worksharing loop
    outside its enclosing complete region or creates a generated OpenMP team.
    """
    graph, identities = analysis._selected_source(procedure, selected)
    routine = analysis.routines[procedure]
    originals = tuple(node for identity in identities for node in graph.source_nodes(identity))
    executable = tuple(identity for identity in identities
                       if any(_kind(node) != "Comment" for node in graph.source_nodes(identity)))
    private, written, peeled = set(), set(), {}

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
    if not nodes or _directive(nodes[-1]) not in {"end parallel", "end parallel do"}:
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

    def uniform(header):
        condition = header.items[0]
        scope = analysis.source_scope_for(header, routine.scope)
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
                raise CompilationError("joined native OpenMP condition payload reads require a uniform-value proof")
            if binding.attributes & {"pointer", "optional", "volatile", "asynchronous"}:
                raise CompilationError("joined native OpenMP condition association or observation is uncertain")

    def check_loop(loop):
        normalize((loop,))
        body = content(loop)
        if not body or _kind(body[-1]) != "End_Do_Stmt":
            raise CompilationError("native worksharing loop requires a complete counted DO")
        if any(_directive(item) is not None for original in body for item in walk(original)):
            raise CompilationError("nested native OpenMP directives inside a worksharing loop are unsupported")

    first = _directive(nodes[0])
    if first is None or not re.match(r"parallel(?:\s|$)", first):
        raise CompilationError("native operation is not one complete PARALLEL region")
    combined = bool(re.match(r"parallel\s+do(?:\s|$)", first))
    clauses(first[len("parallel do"):] if combined else first[len("parallel"):])

    def body(items):
        items, index = normalize(items), 0
        while index < len(items):
            node, directive = items[index], _directive(items[index])
            if _kind(node) == "If_Construct":
                branch = []
                for item in content(node):
                    label = _kind(item)
                    if label in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                        if branch:
                            body(branch)
                        branch = []
                        if label in {"If_Then_Stmt", "Else_If_Stmt"}:
                            uniform(item)
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
            index += 1
            while index < len(items) and _kind(items[index]) == "Comment" and _directive(items[index]) is None:
                index += 1
            if index >= len(items) or _directive(items[index]) not in {"end do", "end do nowait"}:
                raise CompilationError("native OpenMP DO requires a matching END DO")
            index += 1

    if combined:
        items = [item for item in nodes[1:-1] if _kind(item) != "Comment"]
        if (_directive(nodes[-1]) != "end parallel do" or len(items) != 1
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
              "retains_original_team_and_directives": True}
    return graph, identities, executable, tuple(sorted(private)), record


def prove_joined_completion(analysis, procedure, selected):
    """Issue native-effect authority only for the ordinary conservative path."""
    graph, identities, executable, private, record = _joined_completion_facts(analysis, procedure, selected)
    payload = {"version": NATIVE_COMPLETION_VERSION, "procedure": procedure, "graph": graph.identity,
               "selected": identities, "executable": executable, "private": sorted(private), "record": record}
    identity = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return NativeCompletionProof(procedure, graph.identity, identity, identities, executable,
                                 private, _freeze(record))
