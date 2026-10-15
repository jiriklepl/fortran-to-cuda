"""Effects of unchanged native source relative to a complete resource demand.

This consumer discharges storage separation; reachability alone cannot do so.
The initial contract uses whole-resource read/write obligations and never grants
definitions, captured storage, native timing estimates, or reordered execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fparser.two.utils import walk

from compiler.frontend.native_group_structure import NATIVE_GROUP_SCAN_LIMIT
from compiler.frontend.resource_reachability import (
    _authenticate_binding,
    _digest,
    prove_resource_reachability,
)
from compiler.ir import CompilationError

VERSION = 1
_ISSUER = object()
_UNSTABLE = frozenset({"pointer", "optional", "volatile", "asynchronous", "value", "external", "intrinsic"})


def _owned_storage(analysis, binding, declarations):
    """Exact distinct module and non-dummy local objects own their storage.

    A dummy, pointer, associated name or COMMON/EQUIVALENCE object cannot borrow
    this rule. The reachability proof independently rejects storage association,
    descriptor changes, unknown calls and escaping/implicit execution throughout
    the retained original selection, including consulted declaration scopes.
    """
    scope = binding.declaring_scope
    module = analysis.modules.get(getattr(scope, "module", None))
    if scope.bindings.get(binding.name) is not binding:
        return False
    if module is not scope:
        routine = analysis.numerical_helpers.get(getattr(scope, "qualified", None))
        if (routine is None or routine.scope is not scope or binding.name in routine.arguments
                or binding.intent is not None):
            return False
    if binding.attributes & _UNSTABLE:
        return False
    if not declarations.independent(scope):
        return False
    # Public C linkage can expose independently aliased storage outside the
    # source-owned graph. Unknown declaration attributes remain conservative.
    allowed = {"allocatable", "save", "target", "contiguous", "dimension", "public", "private", "parameter"}
    return not binding.attributes - allowed


class _Declarations:
    """Check every compared declaration scope, including future-only demands."""
    def __init__(self):
        self.checked, self.visits = {}, 0

    def independent(self, scope):
        if id(scope) in self.checked:
            return self.checked[id(scope)]
        specification = next((part for part in getattr(scope.node, "content", ())
                              if type(part).__name__ == "Specification_Part"), None)
        independent = True
        for node in walk(specification):
            self.visits += 1
            if self.visits > NATIVE_GROUP_SCAN_LIMIT:
                raise CompilationError("native storage separation declaration scan budget exhausted")
            kind = type(node).__name__
            if (kind in {"Common_Stmt", "Equivalence_Stmt", "Bind_Stmt", "Cray_Pointer_Stmt"}
                    or kind == "Comment" and str(node).lstrip().lower().startswith("!$omp")):
                independent = False
        self.checked[id(scope)] = independent
        return independent


def _separation(analysis, native, demanded, declarations):
    if (native.binding is demanded or native.attributes & _UNSTABLE
            or not _owned_storage(analysis, native.binding, declarations)
            or not _owned_storage(analysis, demanded, declarations)):
        raise CompilationError("native resource separation requires independent original owned storage: "
                               + native.resource + " versus " + demanded.root)
    return (native.resource, demanded.root)


@dataclass(frozen=True, slots=True)
class SourceBoundNativeDemandEffects:
    procedure: str
    identity: str
    demanded: tuple = field(repr=False, compare=False)
    effects: tuple = field(repr=False, compare=False)
    reachability: object = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _separated: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self, analysis, procedure, originals, demanded):
        if (self._issuer is not _ISSUER or type(self) is not SourceBoundNativeDemandEffects
                or analysis is not self._analysis or procedure != self.procedure
                or getattr(analysis, "_native_demand_effects", {}).get(self.identity) is not self):
            raise CompilationError("native demand effects require exact registered source authority")
        self.reachability.validate(analysis, procedure, originals, demanded)
        # Registered source tokens are immutable, but standalone source graphs
        # remain mutable. Revalidate storage identities and attributes too.
        for binding in demanded:
            _authenticate_binding(analysis, binding)
        declarations = _Declarations()
        for obligation in self.reachability.obligations:
            _separation(analysis, obligation.native, obligation.demanded, declarations)
        return self

    def public(self):
        return {
            "schema_version": VERSION,
            "identity": self.identity,
            "procedure": self.procedure,
            "reachability_identity": self.reachability.identity,
            "demanded_resources": [binding.root for binding in self.demanded],
            "effects": [{"resource": binding.root, "actions": sorted(actions), "coverage": "whole"}
                        for binding, actions in self.effects],
            "storage_separation": [{"native": native, "demanded": demanded,
                                    "basis": "distinct original owned module or non-dummy local storage"}
                                   for native, demanded in self._separated],
            "native_completion": {
                "available": True,
                "has_openmp_in_closure": bool(self.reachability._completion),
                "reason": "unchanged source calls with closed synchronous reachability and original joins",
                "internal_cuts_authorized": False,
            },
            "execution": "unchanged original selection after required whole-resource host coherence",
            "definitions": "no new definition or whole-overwrite authority",
            "demand_scope": "consumer must validate against the entire final checked interval",
            "capture_authorized": False,
            "placement_authorized": False,
            "cost_estimate_available": False,
        }


def prove_native_demand(analysis, procedure, originals, demanded):
    """Authorize coherent original native execution for exact demanded objects.

    The owner must independently establish whole entry definitions and stable
    descriptors, retain original source/guards, and verify demand completeness
    against its final checked execution plan before publishing any generated
    edits. Unknown effects or aliases reject before this token can be issued.
    """
    reachability = prove_resource_reachability(analysis, procedure, originals, demanded)
    if reachability.barriers:
        raise CompilationError("native demand reachability incomplete: " + "; ".join(
            dict.fromkeys(item.reason for item in reachability.barriers)))
    for binding in demanded:
        if binding.attributes & _UNSTABLE:
            raise CompilationError("native demand requires stable present original storage: " + binding.root)
        if binding.dtype not in {"real", "integer", "logical"} or binding.kind not in {4, 8}:
            raise CompilationError("native demand requires intrinsic numeric storage: " + binding.root)
    declarations = _Declarations()
    separated = tuple(_separation(analysis, item.native, item.demanded, declarations)
                      for item in reachability.obligations)
    effects = []
    for binding in demanded:
        accesses = tuple(item for item in reachability.accesses if item.storage.binding is binding)
        if any(item.storage.components for item in accesses):
            raise CompilationError("native managed component effects require separate physical mapping authority")
        actions = {item.kind for item in accesses}
        if actions - {"read", "write", "association", "descriptor"}:
            raise CompilationError("native demand cannot preserve definitions across an entry or lifetime change: "
                                   + binding.root)
        if not binding.rank and "write" in actions:
            raise CompilationError("native demanded scalar writes require reached planning authority: " + binding.root)
        if binding.rank and actions & {"read", "write"}:
            effects.append((binding, frozenset(actions & {"read", "write"})))
    identity = _digest({"version": VERSION, "reachability": reachability.identity,
                        "effects": [(id(binding), sorted(actions)) for binding, actions in effects],
                        "separated": separated})
    registry = getattr(analysis, "_native_demand_effects", None)
    if registry is None:
        registry = analysis._native_demand_effects = {}
    if identity in registry:
        return registry[identity].validate(analysis, procedure, originals, demanded)
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("native demand effect registry budget exhausted")
    result = SourceBoundNativeDemandEffects(procedure, identity, demanded, tuple(effects), reachability,
                                            analysis, originals, separated, _ISSUER)
    registry[identity] = result
    try:
        return result.validate(analysis, procedure, originals, demanded)
    except BaseException:
        if registry.get(identity) is result:
            del registry[identity]
        raise
