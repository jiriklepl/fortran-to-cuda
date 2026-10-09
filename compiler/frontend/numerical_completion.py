"""Synchronous source-helper completion, separate from native effect hooks."""

from dataclasses import dataclass, field
from hashlib import sha256
import json

from fparser.two.utils import walk

from compiler.frontend.native_completion import _children, _directive, _joined_completion_facts, _kind
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError
from compiler.ir.intrinsics import ARRAY_INQUIRIES, INTRINSICS, MODEL_INQUIRIES

NUMERICAL_COMPLETION_VERSION = 1
_INTRINSICS = set(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES | {
    "dot_product", "sum", "product", "all", "any", "count", "minval", "maxval", "allocated", "present"}
_STATEMENTS = {"Assignment_Stmt", "Call_Stmt", "Nonlabel_Do_Stmt", "Label_Do_Stmt", "End_Do_Stmt",
               "If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt", "If_Stmt", "Continue_Stmt"}


@dataclass(frozen=True)
class NumericalCompletionProof:
    procedure: str
    structured_identity: str
    identity: str
    selected_node_ids: tuple[str, ...]
    private_roots: tuple[str, ...]
    _record: object = field(repr=False, compare=False)

    def public(self):
        return {"schema_version": NUMERICAL_COMPLETION_VERSION,
                "proof_role": "numerical_source_completion", "proof_identity": self.identity,
                "procedure": self.procedure, "structured_identity": self.structured_identity,
                "selected_original_nodes": list(self.selected_node_ids),
                "private_resources": list(self.private_roots),
                "native_effects_authority": False, "gpu_legality_established": False, **_thaw(self._record)}

    def validate(self, analysis):
        graph = analysis.structure(self.procedure)
        if (graph.identity != self.structured_identity
                or analysis._numerical_completions.get(self.identity) is not self):
            raise CompilationError("numerical completion lacks registered original source authority")
        for item in _thaw(self._record).get("helper_closure", ()):
            helper = analysis.numerical_helpers.get(item["procedure"])
            original = analysis._numerical_roles.get(item["procedure"])
            if (helper is None or original is None or helper.scope.node is not original[0]
                    or analysis._routine_signature(helper) != original[1]
                    or str(helper.scope.node) != original[2]):
                raise CompilationError("numerical completion requires unchanged original source-backed helper authority")
        return self


def prove_numerical_completion(analysis, procedure, selected):
    """Admit source-proven pure synchronous helpers only for outlining.

    This token cannot authorize native memory hooks. Numerical lowering still
    proves private definitions, aliases, IEEE behavior and GPU computation.
    """
    graph, identities = analysis._selected_source(procedure, selected)
    routine = analysis.routines[procedure]
    originals = tuple(node for identity in identities for node in graph.source_nodes(identity))
    helpers, active, calls, top_writes = {}, [], {}, set()
    operations, call_sites = 0, 0

    def storage(owner, node):
        scope = analysis.source_scope_for(node, owner.scope)
        if _kind(node) == "Part_Ref":
            node = node.items[0]
        return analysis._binding(scope, node)

    def writable(owner, binding):
        if binding is None or binding.attributes & {"pointer", "allocatable", "save", "parameter", "volatile", "asynchronous"}:
            return False
        if binding.name in owner.arguments:
            return binding.intent in {"out", "inout"}
        return binding.root.startswith(owner.qualified + "::")

    def associate(helper, node):
        arguments, result, position, keyword_seen = tuple(_children(node.items[1])), {}, 0, False
        for argument in arguments:
            if _kind(argument) == "Actual_Arg_Spec":
                keyword_seen = True
                formal, value = str(argument.items[0]).lower(), argument.items[1]
            else:
                if keyword_seen or position >= len(helper.arguments):
                    raise CompilationError("numerical completion helper association is unproved")
                formal, value = helper.arguments[position], argument
                position += 1
            if formal not in helper.arguments or formal in result:
                raise CompilationError("numerical completion helper association is ambiguous")
            result[formal] = value
        if set(result) != set(helper.arguments):
            raise CompilationError("numerical completion requires all original helper arguments")
        return result

    def helper_call(owner, node):
        nonlocal call_sites
        call_sites += 1
        if call_sites > 128:
            raise CompilationError("numerical helper completion call budget exceeded")
        scope = analysis.source_scope_for(node, owner.scope)
        targets = analysis._candidates(scope, node.items[0])
        if len(targets) != 1 or targets[0] not in analysis.numerical_helpers:
            raise CompilationError("numerical completion has an unresolved synchronous source helper: " + str(node.items[0]))
        helper = analysis.numerical_helpers[targets[0]]
        actuals = associate(helper, node)
        outputs = set()
        for name, actual in actuals.items():
            formal = helper.scope.bindings.get(name)
            if (formal is None or formal.intent not in {"in", "out", "inout"}
                    or formal.attributes & {"pointer", "optional"}
                    or ("allocatable" in formal.attributes and formal.intent != "in")):
                raise CompilationError("numerical completion helper needs stable original explicit formal intents")
            if formal.intent in {"out", "inout"}:
                actual_binding = storage(owner, actual)
                if not writable(owner, actual_binding):
                    raise CompilationError("numerical completion helper writes nonlocal or unstable original storage")
                outputs.add(actual_binding.root)
        if owner is routine:
            calls[id(node)] = tuple(outputs)
            top_writes.update(outputs)
        visit_helper(helper)

    def scan(owner, body, *, specification=False):
        nonlocal operations
        if body is None:
            return
        from compiler.frontend.component_bindings import references
        for binding, _reference in references(analysis, owner.scope, body):
            if binding.attributes & {"pointer", "volatile", "asynchronous", "optional"}:
                raise CompilationError("numerical completion helper has uncertain storage or observation effects")
        selectors = {id(part) for reference in walk(body) if _kind(reference) == "Data_Ref"
                     for part in reference.items}
        for node in walk(body):
            kind = _kind(node)
            if kind == "Comment":
                if owner is not routine and _directive(node) is not None:
                    raise CompilationError("numerical completion helper contains OpenMP work or thread ownership")
                continue
            if not specification and kind.endswith("_Stmt"):
                operations += 1
                if operations > analysis.operation_limit:
                    raise CompilationError("numerical helper completion operation budget exceeded")
                if kind not in _STATEMENTS and not (kind == "Return_Stmt" and owner is not routine):
                    raise CompilationError("numerical completion helper has unsupported I/O, exit or lifetime effects: " + kind)
            if kind == "Assignment_Stmt" and owner is not routine:
                if not writable(owner, storage(owner, node.items[0])):
                    raise CompilationError("numerical completion helper writes nonlocal or unstable original storage")
            if kind == "Call_Stmt":
                helper_call(owner, node)
            elif kind == "Intrinsic_Function_Reference":
                name = str(node.items[0]).lower()
                scope = analysis.source_scope_for(node, owner.scope)
                if (name not in _INTRINSICS or analysis._binding(scope, name)
                        or analysis._candidates(scope, name) or analysis._unknown_exports(scope)):
                    raise CompilationError("numerical completion intrinsic identity or effects are unproved: " + name)
            elif kind in {"Function_Reference", "Part_Ref", "Structure_Constructor"} and id(node) not in selectors:
                if kind == "Part_Ref" and storage(owner, node) is not None:
                    continue
                helper_call(owner, node)

    def visit_helper(helper):
        if helper.qualified in active:
            raise CompilationError("recursive numerical helper completion is unsupported")
        if helper.qualified in helpers:
            return
        original = analysis._numerical_roles.get(helper.qualified)
        if (original is None or helper.scope.node is not original[0]
                or analysis._routine_signature(helper) != original[1] or str(helper.scope.node) != original[2]):
            raise CompilationError("numerical completion needs original source-backed helper authority")
        header = next((item for item in _children(helper.scope.node)
                       if _kind(item) in {"Subroutine_Stmt", "Function_Stmt"}), None)
        if header is None or not any(str(item).lower() == "pure" for item in _children(header.items[0])):
            raise CompilationError("numerical helper completion requires explicit PURE and transitive source proof")
        if len(active) + 1 >= analysis.depth_limit or len(helpers) >= min(128, analysis.procedure_limit):
            raise CompilationError("numerical helper completion closure budget exceeded")
        if any(binding.attributes & {"save", "volatile", "asynchronous"}
               and "parameter" not in binding.attributes for binding in helper.scope.bindings.values()):
            raise CompilationError("numerical completion helper has persistent or externally observed local state")
        helpers[helper.qualified] = helper
        active.append(helper.qualified)
        specification = next((item for item in _children(helper.scope.node)
                              if _kind(item) == "Specification_Part"), None)
        scan(helper, specification, specification=True)
        scan(helper, helper.execution)
        active.pop()

    for node in originals:
        scan(routine, node)

    def completed_call(node):
        if id(node) not in calls:
            raise CompilationError("numerical joined call lacks original helper-closure authority")
        return calls[id(node)]

    graph, identities, _executable, private, record = _joined_completion_facts(
        analysis, procedure, selected, call_completion=completed_call)
    if not top_writes.issubset(private):
        raise CompilationError("joined numerical helper outputs require original private scalar or array storage")
    record.update({"reason": "original joined numerical region and source-proven synchronous pure helper closure",
        "helper_closure": [{"procedure": name, "source": str(helper.scope.path),
                            "source_identity": analysis.sources[str(helper.scope.path)]}
                           for name, helper in sorted(helpers.items())],
        "requires_numerical_lowering": True, "native_effects_authority": False})
    payload = {"version": NUMERICAL_COMPLETION_VERSION, "graph": graph.identity, "nodes": identities,
               "private": private, "record": record}
    identity = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    registry = analysis._numerical_completions
    if identity not in registry:
        if len(registry) >= analysis.procedure_limit:
            registry.pop(next(iter(registry)))
        registry[identity] = NumericalCompletionProof(procedure, graph.identity, identity, identities, private, _freeze(record))
    return registry[identity]
