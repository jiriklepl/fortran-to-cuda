"""Complete effects of unchanged native calls before a residency owner exists.

Missing lifetime facts for a read-only module input do not constitute managed
capture facts.  A detached analysis may discharge that narrow analysis obstacle
only after the entire original closure proves the input is read-only and native
work returns synchronously.  The registered result grants no execution, device,
definition, alias, or allocation-freshness authority.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import Base

from compiler.frontend.call_bindings import resolve_source_call
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_functions import _original, _validate_scope
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.frontend.summary_cache import SummaryCache
from compiler.ir import CompilationError

NATIVE_PREFIX_VERSION = 1
_SCAN_LIMIT = 65536
_FORBIDDEN = {"pointer", "optional", "target", "volatile", "asynchronous", "value", "parameter"}
_CACHES = ("summaries", "_closures", "_structures", "_segments", "_descriptor_proofs",
           "_native_sections", "_joined_completions", "_numerical_completions",
           "_worksharing_completions", "_worksharing_native_completions", "_reduction_proofs",
           "_reduction_candidates", "_omp_reduction_proofs", "_native_environments",
           "_native_environment_states", "_native_predicates", "_native_functions",
           "_native_prefix_proofs", "_native_definition_proofs")


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _walk(node, visits):
    stack = [iter((node,))]
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        visits[0] += 1
        if visits[0] > _SCAN_LIMIT:
            raise CompilationError("native prefix original-source scan budget exhausted")
        if isinstance(item, (tuple, list)):
            stack.append(iter(item))
        elif isinstance(item, Base):
            yield item
            stack.append(iter(getattr(item, "content", getattr(item, "items", ())) or ()))


def _source_state(analysis, procedure):
    if getattr(analysis, "_descriptor_source", None) is not None:
        raise CompilationError("native prefix requires an original analysis, not a shortened projection")
    analysis._require_original(procedure)
    authority = analysis._summary_authority()
    if authority["role"] != "source":
        raise CompilationError("native prefix requires unchanged original source authority")
    scopes = analysis._source_function_scopes
    if len(scopes) > _SCAN_LIMIT:
        raise CompilationError("native prefix original scope budget exhausted")
    # Imports, kind state and generic maps are mutable compiler structures,
    # separate from source text. Authenticate them against the parsed snapshot.
    for scope, *_rest in scopes.values():
        try:
            _validate_scope(analysis, scope, track=False)
        except CompilationError as error:
            raise CompilationError("native prefix original binding resolution changed") from error
    if any(id(scope) not in scopes for scope in analysis.modules.values()) or any(
            id(routine.scope) not in scopes for routine in analysis.numerical_helpers.values()):
        raise CompilationError("native prefix original scope identity changed")
    return authority


def _fork(analysis, roots):
    result = copy.copy(analysis)
    # Detach every top-level mutable container, including source-only caches.
    # Original parser nodes remain shared, read-only and source-authenticated.
    for name, value in vars(analysis).items():
        if type(value) is dict:
            setattr(result, name, dict(value))
        elif type(value) is set:
            setattr(result, name, set(value))
        elif type(value) is list:
            setattr(result, name, list(value))
    for name in _CACHES:
        setattr(result, name, {})
    result.contracts = copy.deepcopy(analysis.contracts)
    result._source_function_used_scopes = set()
    result._summary_cache = SummaryCache(max_entries=0)
    result._closure_authority = None
    result._analysis_started = False
    # This is explicitly a native-only provisional set, never a caller capture
    # authorization. It cannot leak through any original cache or token table.
    result._stable_module_allocatables = analysis.stable_module_allocatables | frozenset(roots)
    return result


def _candidate(analysis, root):
    parts = root.split("::")
    owner = analysis.modules.get(parts[0]) if len(parts) == 2 else None
    binding = owner.bindings.get(parts[1]) if owner is not None else None
    if (binding is None or binding.root != root or binding.declaring_scope is not owner
            or not 1 <= binding.rank <= 4 or "allocatable" not in binding.attributes
            or binding.attributes & _FORBIDDEN
            or binding.dtype not in {"real", "integer"} or binding.kind not in {4, 8}):
        raise CompilationError("native prefix lifetime obligation is not an original numeric module allocation: " + root)
    reason = analysis.resource_identity_boundary(binding)
    if reason:
        raise CompilationError(reason)
    specification = next((node for node in getattr(owner.node, "content", ())
                          if type(node).__name__ == "Specification_Part"), None)
    for node in _walk(specification, [0]):
        if type(node).__name__ in {"Common_Stmt", "Equivalence_Stmt"}:
            raise CompilationError("native prefix module storage association is unproved")
        if type(node).__name__ == "Comment" and str(node).lstrip().lower().startswith("!$omp"):
            raise CompilationError("native prefix module OpenMP ownership is unproved")
    return {**binding.public(), "source": str(owner.path), "source_sha256": analysis.sources[str(owner.path)]}


def _directive(node):
    text = str(node).strip().lower()
    return text[5:].strip() if type(node).__name__ == "Comment" and text.startswith("!$omp") else None


def _clauses(analysis, routine, text):
    names, private, reductions, shared = set(), [], [], []
    while text:
        match = re.match(r"(private|shared|default|reduction)\s*\(([^()]*)\)\s*", text)
        if match is None:
            raise CompilationError("unsupported before-owner OpenMP clause: " + text)
        kind, content = match[1], match[2].strip()
        if kind == "default":
            if content not in {"shared", "none"}:
                raise CompilationError("unsupported before-owner OpenMP DEFAULT")
            key = "default"
            if key in names:
                raise CompilationError("duplicate before-owner OpenMP DEFAULT")
            names.add(key)
        else:
            operator = None
            if kind == "reduction":
                split = content.split(":")
                if len(split) != 2 or split[0].strip() not in {"max", "min", "+", ".or.", ".and."}:
                    raise CompilationError("before-owner reduction requires a builtin scalar operator")
                operator, content = split[0].strip(), split[1].strip()
            for name in content.split(","):
                name = name.strip()
                if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                    raise CompilationError("before-owner OpenMP clause requires original scalar names")
                binding = analysis._binding(routine.scope, name)
                if (binding is None or binding.rank or binding.kind not in {4, 8}
                        or binding.dtype not in {"real", "integer", "logical"}
                        or binding.attributes & (_FORBIDDEN | {"save", "allocatable"})):
                    raise CompilationError("before-owner OpenMP clause requires fixed original scalar storage")
                reason = analysis.resource_identity_boundary(binding)
                if reason:
                    raise CompilationError(reason)
                if binding.root in names:
                    raise CompilationError("conflicting before-owner OpenMP data-sharing clauses")
                names.add(binding.root)
                if kind == "private":
                    scope = binding.declaring_scope
                    if (scope is None or scope.parent is None or binding.intent is not None
                            or binding.root.startswith("argument::")):
                        raise CompilationError("before-owner PRIVATE requires plain original local scalars")
                    private.append(binding.public())
                elif kind == "shared":
                    shared.append(binding.public())
                else:
                    logical = operator in {".or.", ".and."}
                    if logical != (binding.dtype == "logical"):
                        raise CompilationError("before-owner reduction operator and scalar type disagree")
                    reductions.append({"operator": operator, "binding": binding.public(),
                                       "execution": "unchanged original native reduction"})
        text = text[match.end():].lstrip(", ")
    return {"private": private, "shared": shared, "reductions": reductions}


def _completion(analysis, closure):
    """Prove synchronous return only, separately from managed completion."""
    groups, calls, visits, checked_scopes = {}, {}, [0], set()
    for name, summary in closure.summaries.items():
        routine = _original(analysis, name)
        if (not summary.get("complete") or summary.get("effect_composition", {}).get("available") is not True
                or any(op["kind"] in {"native_contract", "native_environment"} for op in summary["operations"])):
            raise CompilationError("native prefix requires a complete nonopaque original source closure")
        calls[name] = tuple(op["procedure"] for op in summary["operations"] if op["kind"] == "call")
        groups[name] = []
        # Reject hidden declaration ownership, storage association and user
        # reduction declarations; do not reinterpret compiler/runtime clauses.
        scopes, seen = [routine.scope], set()
        resources = {effect.get("resource", "") for effect in summary["ordered_effects"]}
        scopes.extend(analysis.modules[module] for module in sorted({root.split("::", 1)[0] for root in resources})
                      if module in analysis.modules)
        while scopes:
            scope = scopes.pop()
            if scope is None or id(scope) in seen:
                continue
            seen.add(id(scope))
            scopes.append(scope.parent)
            if id(scope) in checked_scopes:
                continue
            checked_scopes.add(id(scope))
            specification = next((node for node in getattr(scope.node, "content", ())
                                  if type(node).__name__ == "Specification_Part"), None)
            for node in _walk(specification, visits):
                if type(node).__name__ in {"Common_Stmt", "Equivalence_Stmt"} or _directive(node) is not None:
                    raise CompilationError("native prefix declaration ownership is unproved")

        peeled = {}

        def sequence(items, *, peeled=peeled, routine=routine, name=name, depth=0):
            if depth > analysis.operation_limit:
                raise CompilationError("native prefix control depth budget exhausted")
            for item in grouped_original_sequence(items, peeled):
                if isinstance(item, NativeGroupCandidate):
                    nodes = tuple(node for node in item.items
                                  if type(node).__name__ != "Comment" or _directive(node) is not None)
                    opening = _directive(nodes[0]) if nodes else None
                    if (len(nodes) != 3 or opening is None
                            or re.match(r"parallel\s+do(?:\s|$)", opening) is None
                            or type(nodes[1]).__name__ != "Block_Nonlabel_Do_Construct"
                            or _directive(nodes[2]) != "end parallel do"):
                        raise CompilationError("native prefix requires a complete original PARALLEL DO and matching join")
                    clause = _clauses(analysis, routine, re.sub(r"^parallel\s+do\s*", "", opening))
                    nested_calls = []
                    for node in _walk(nodes[1], visits):
                        if _directive(node) is not None:
                            raise CompilationError("nested before-owner OpenMP constructs are unsupported")
                        if type(node).__name__ == "Call_Stmt":
                            resolved = resolve_source_call(analysis, analysis.source_scope_for(node, routine.scope), node)
                            nested_calls.append(resolved.procedure)
                    groups[name].append({"opening": str(nodes[0]), "loop": str(nodes[1]),
                                         "join": str(nodes[2]), **clause,
                                         "called_procedures": nested_calls})
                else:
                    if _directive(item) is not None:
                        raise CompilationError("unsupported before-owner OpenMP directive: " + str(item))
                    kind = type(item).__name__
                    if kind in {"If_Construct", "Block_Nonlabel_Do_Construct", "Associate_Construct"}:
                        children = peeled.get(id(item), getattr(item, "content", ()))
                        sequence(children[1:-1], peeled=peeled, routine=routine, name=name, depth=depth + 1)

        # Account for the full expression tree as well as the structural walk.
        tuple(_walk(routine.execution, visits))
        sequence(getattr(routine.execution, "content", ()))

    active, has_team = set(), {}

    def teams(name):
        if name in active or name not in groups:
            raise CompilationError("native prefix call closure is recursive or incomplete")
        if name not in has_team:
            active.add(name)
            has_team[name] = bool(groups[name]) | any(teams(child) for child in calls[name])
            active.remove(name)
        return has_team[name]

    for name in groups:
        teams(name)
    for procedure_groups in groups.values():
        for group in procedure_groups:
            if any(teams(child) for child in group["called_procedures"]):
                raise CompilationError("before-owner native team calls a nested team")
    return {"schema_version": 1, "available": True, "contract": "original_native_before_owner_synchronous_return",
            "requires_serial_caller": True, "requires_no_active_context": True,
            "original_native_completion_unchanged": True, "internal_cuts_authorized": False,
            "numerical_reassociation_authorized": False, "execution_authorized": False,
            "procedures": [{"procedure": name, "original_joined_groups": groups[name],
                            "called_procedures": list(calls[name])} for name in sorted(groups)]}


def _derive(analysis, procedure):
    authority = _source_state(analysis, procedure)
    roots, obligations = set(), {}
    initial_complete = None
    for _attempt in range(analysis.procedure_limit + 1):
        projected = _fork(analysis, roots)
        summary = projected.summarize(procedure)
        closure = projected._closures[procedure]
        if initial_complete is None:
            initial_complete = summary["complete"]
        added = set()
        if (len(closure.summaries) > analysis.procedure_limit
                or closure.operations > analysis.operation_limit):
            raise CompilationError("native prefix source closure budget exhausted")
        for name, child in closure.summaries.items():
            for requirement in child.get("capture_lifetime_requirements", ()):
                if type(requirement.get("authorized")) is not bool or type(requirement.get("resource")) is not str:
                    raise CompilationError("native prefix requires typed source lifetime obligations")
                root = requirement["resource"]
                if requirement["authorized"] or root.startswith("argument::"):
                    continue
                _candidate(analysis, root)
                added.add(root)
                record = {"procedure": name, **copy.deepcopy(requirement)}
                obligations[_hash(record)] = record
                if len(obligations) > analysis.operation_limit:
                    raise CompilationError("native prefix lifetime obligation budget exhausted")
        if added - roots:
            roots.update(added)
            if len(roots) > analysis.operation_limit:
                raise CompilationError("native prefix module input budget exhausted")
            continue
        if not summary["complete"] or summary.get("effect_composition", {}).get("available") is not True:
            raise CompilationError("native prefix original source closure incomplete: " + "; ".join(summary["reasons"]))
        completion = _completion(projected, closure)
        effects = summary["ordered_effects"]
        for root in roots:
            accesses = [effect for effect in effects if effect.get("resource") == root]
            if (not accesses or any(effect["kind"] not in {"read", "descriptor_read"} for effect in accesses)):
                raise CompilationError("native prefix provisional module input is not read-only: " + root)
        if _source_state(analysis, procedure) != authority:
            raise CompilationError("native prefix original source authority changed during proof")
        return {"schema_version": NATIVE_PREFIX_VERSION, "proof_role": "original_native_before_owner_effects",
                "procedure": procedure, "sources": analysis.inputs.identity(), "analysis_authority": authority,
                "original_summary_complete": initial_complete, "complete": True,
                "readonly_module_resources": sorted(roots),
                "candidate_lifetime_obligations": [obligations[key] for key in sorted(obligations)],
                "module_inputs": [_candidate(analysis, root) for root in sorted(roots)],
                "summary": summary, "synchronous_return": completion,
                "closure": [{"procedure": name, "summary_identity": item["summary_identity"],
                             "source_kind": item["source_kind"]} for name, item in sorted(closure.summaries.items())],
                "execution_authorized": False, "requires_original_native_execution": True,
                "requires_serial_caller": True, "requires_no_active_context": True,
                "original_guards_and_argument_evaluation_required": True,
                "capture_authority": False, "managed_definition_authority": False,
                "allocation_freshness_authority": False, "alias_authority": False,
                "gpu_legality_established": False, "numerical_lowering_authorized": False}
    raise CompilationError("native prefix provisional closure fixed-point budget exhausted")


@dataclass(frozen=True)
class NativePrefixProof:
    procedure: str
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _authority: object = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)
    _bindings: tuple = field(repr=False, compare=False)

    @property
    def readonly_module_resources(self):
        return tuple(_thaw(self._record)["readonly_module_resources"])

    @property
    def effects(self):
        return _thaw(self._record)["summary"]["ordered_effects"]

    @property
    def summary(self):
        """Native-only complete effects; never a general managed summary."""
        return _thaw(self._record)["summary"]

    def public(self):
        return {"proof_identity": self.identity, **_thaw(self._record)}

    def validate(self, analysis, procedure):
        if (analysis is not self._analysis or procedure != self.procedure
                or getattr(analysis, "_native_prefix_proofs", {}).get(self.identity) is not self):
            raise CompilationError("native prefix lacks registered original analysis authority")
        if _source_state(analysis, procedure) != _thaw(self._authority):
            raise CompilationError("native prefix original source authority changed")
        for scope, bindings in self._bindings:
            if (len(scope.bindings) != len(bindings)
                    or any(scope.bindings.get(name) is not binding for name, binding in bindings)):
                raise CompilationError("native prefix original binding identity changed")
        return self


def prove_native_prefix(analysis, procedure):
    """Register conditional source effects, without authorizing execution.

    The consuming owner-window proof must independently enforce unchanged
    original native execution by a serial caller before any context exists.
    Runtime nonalias, descriptors, definitions and selected-position freshness
    cannot be obtained by passing a flag or by reusing these provisional roots.
    """
    registry = getattr(analysis, "_native_prefix_proofs", None)
    if registry is not None and any(not isinstance(proof, NativePrefixProof) or proof._analysis is not analysis
                                    for proof in registry.values()):
        raise CompilationError("native prefix registry belongs to another original analysis")
    record = _derive(analysis, procedure)
    identity = _hash(record)
    if registry is None:
        registry = analysis._native_prefix_proofs = {}
    existing = registry.get(identity)
    if existing is not None:
        return existing.validate(analysis, procedure)
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("native prefix proof registry budget exhausted")
    bindings = tuple((scope, tuple(scope.bindings.items()))
                     for scope, *_rest in analysis._source_function_scopes.values())
    proof = NativePrefixProof(procedure, identity, analysis, _freeze(record["analysis_authority"]),
                              _freeze(record), bindings)
    registry[identity] = proof
    return proof
