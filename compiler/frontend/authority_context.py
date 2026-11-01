"""Closed source/policy inputs separated from mutable generation work.

A context reads only the sealed session and immutable policy values. Replacing
any supported policy input discards the analysis's context; old handles then
fail rather than silently following a mutable analysis object. Parser/semantic
copies remain private projections and cannot acquire original membership.
"""
import json
from dataclasses import dataclass, field
from hashlib import sha256

from compiler.frontend.immutable_graph import GraphFreezer
from compiler.frontend.proof_arena import DerivationArena
from compiler.ir import CompilationError

# Every mutable source-policy input currently consumed by the source facade.
# Caches, counters and generated-builder state deliberately are not authority.
CONTEXT_FIELDS = frozenset({
    "depth_limit", "procedure_limit", "operation_limit", "contracts",
    "_stable_module_allocatables", "_allocation_authorizations", "_native_metadata",
    "_selected_descriptor_window", "_descriptor_source", "source_graph_operation_limit",
    "routines",
})
_ABSENT = object()


def immutable_policy(value):
    return GraphFreezer().freeze(value)


def normalize_context_field(name, value):
    """Own canonical first-consumer inputs at the supported mutation boundary."""
    from compiler.frontend.immutable_graph import FrozenDict, _thaw
    from compiler.frontend.summary_cache import _plain_json

    if name in {"depth_limit", "procedure_limit", "operation_limit"}:
        if type(value) is not int or value < 1:
            raise CompilationError("source context budgets require positive exact integers")
        return value
    if name == "_native_metadata":
        if type(value) is not bool:
            raise CompilationError("source context native metadata requires an exact boolean")
        return value
    if name == "_stable_module_allocatables":
        if type(value) not in {set, frozenset} or any(type(item) is not str for item in value):
            raise CompilationError("source context stable roots require exact source resource names")
        return frozenset(value)
    if name == "_descriptor_source":
        from compiler.frontend.source_effects import SourceEffects
        if value is not None and type(value) is not SourceEffects:
            raise CompilationError("source context descriptor source requires its typed analysis reference")
        return value
    if name == "_selected_descriptor_window":
        if value is not None:
            from compiler.frontend.descriptor_windows import DescriptorWindowProof
            from compiler.frontend.source_effects import SourceEffects
            from compiler.scopes.conditional_return import ConditionalUnitDescriptorProof

            if (type(value) is not tuple or len(value) != 4 or type(value[0]) is not SourceEffects
                    or type(value[1]) is not str or type(value[2]) is not tuple
                    or any(type(item) is not str for item in value[2])
                    or type(value[3]) not in {DescriptorWindowProof, ConditionalUnitDescriptorProof}):
                raise CompilationError("source context requires its immutable typed descriptor selection")
        return value
    if name in {"contracts", "_allocation_authorizations", "source_graph_operation_limit"}:
        if type(value) not in {dict, FrozenDict}:
            raise CompilationError("source context policy requires a compiler-owned JSON mapping")
        if name == "source_graph_operation_limit" and any(
                type(key) is not str or type(item) is not int or item < 1 for key, item in value.items()):
            raise CompilationError("source graph caps require named positive exact integers")
        owned = immutable_policy(value)
        try:
            _plain_json(_thaw(owned))
        except (TypeError, ValueError) as error:
            raise CompilationError("source context policy requires immutable JSON values") from error
        return owned
    return value


def require_context_fields(policy, session):
    """Validate canonical ownership without invoking an external serializer."""
    from compiler.frontend.immutable_graph import FrozenDict, FrozenList
    from compiler.frontend.source_session import ProjectedRoutines

    def closed(value):
        if value is None or type(value) in {bool, int, float, str}:
            return True
        if type(value) is FrozenDict:
            return all(type(key) is str and closed(item) for key, item in value.items())
        if type(value) in {FrozenList, tuple}:
            return all(closed(item) for item in value)
        return False

    for name, value in policy.items():
        if name in {"contracts", "_allocation_authorizations", "source_graph_operation_limit"}:
            if type(value) is not FrozenDict or not closed(value):
                raise CompilationError("source context contains an unowned mutable policy input")
        elif name == "routines":
            if value is session.roots["routines"]:
                continue
            if (type(value) is not ProjectedRoutines or value.base is not session.roots["routines"]
                    or any(type(key) is not str or not session.owns_projection(role)
                           for key, role in value.overrides.items())):
                raise CompilationError("source context contains an unowned routine projection")
        else:
            canonical = normalize_context_field(name, value)
            if canonical is not value and canonical != value:
                raise CompilationError("source context input is not canonical: " + name)
            if name == "_stable_module_allocatables" and type(value) is not frozenset:
                raise CompilationError("source context contains mutable stable roots")


@dataclass(frozen=True, slots=True, eq=False)
class SourceAuthorityContext:
    _intern_key: str = field(repr=False)
    session: object = field(repr=False)
    policy: object = field(repr=False)
    authority: str = field(repr=False)
    _fields_key: tuple = field(repr=False)
    _arena: object = field(default=None, repr=False)

    def require(self, analysis):
        from compiler.frontend.source_effects import SourceEffects
        if (type(analysis) is not SourceEffects or not self.session.owns_context(self)
                or getattr(analysis, "_source_session", None) is not self.session
                or getattr(analysis, "_source_authority_context", None) is not self):
            raise CompilationError("source proof belongs to a stale or foreign authority context")
        self.session.verify(analysis)
        return self

    def require_work_state(self, state):
        if not self.session.owns_context(self) or _policy_key(
                {name: state[name] for name in CONTEXT_FIELDS if name in state}, self.session) != self._fields_key:
            raise CompilationError("generation work state does not match its immutable authority context")

    def reader(self, analysis):
        """A private cache facade with closed source and policy references."""
        self.require(analysis)
        reader = object.__new__(type(analysis))
        values = object.__getattribute__(reader, "__dict__")
        values.update(self.session.roots)
        # Local-allocation reading consumes no mutable legacy owner or
        # selected-window proof. Joined prerequisites are imported separately.
        values.update((name, value) for name, value in self.policy.items()
                      if name not in {"_selected_descriptor_window", "_descriptor_source"})
        values.update(_source_session=self.session, _source_authority_context=self,
                      _source_authority_reader=True,
                      _source_function_used_scopes=set(), _original_call_associations={})
        return reader

    def derivation_arena(self, analysis):
        self.require(analysis)
        arena = getattr(analysis, "_source_derivation_arena", None)
        if type(arena) is not DerivationArena or arena._context is not self:
            arena = self._arena.fork()
            object.__setattr__(analysis, "_source_derivation_arena", arena)
        return arena

    @property
    def arena(self):
        return self._arena


def _context_key(value, *, original_routines=None):
    """Bind every context input without calling delegated public serializers."""
    from compiler.frontend.immutable_graph import FrozenDict, FrozenList
    from compiler.frontend.source_session import ProjectedRoutines
    if original_routines is not None:
        if value is original_routines:
            return "routine_roles", id(original_routines), ()
        if type(value) is ProjectedRoutines and value.base is original_routines:
            return "routine_roles", id(original_routines), tuple(
                (name, id(routine)) for name, routine in sorted(value.overrides.items())
                if original_routines.get(name) is not routine)
        raise CompilationError("source context requires its original routine base and immutable roles")
    if value is None or type(value) in {bool, int, str, float}:
        return type(value).__name__, value
    if type(value) in {dict, FrozenDict}:
        return "mapping", tuple((key, _context_key(item)) for key, item in sorted(dict.items(value)))
    if type(value) in {tuple, list, FrozenList}:
        return "sequence", tuple(_context_key(item) for item in value)
    if type(value) in {set, frozenset}:
        return "set", tuple(sorted(_context_key(item) for item in value))
    # Original projection roles and typed opaque prerequisite owners retain
    # exact identity. Their mutable analysis caches are never read as policy.
    return "reference", type(value).__module__, type(value).__qualname__, id(value)


def _policy_key(policy, session):
    return tuple((name, _context_key(value, original_routines=session.roots["routines"])
                  if name == "routines" else _context_key(value)) for name, value in sorted(policy.items()))


def context_for(analysis):
    from compiler.frontend.source_effects import SourceEffects
    if type(analysis) is not SourceEffects:
        raise CompilationError("source context requires the compiler-owned source analysis facade")
    session = getattr(analysis, "_source_session", None)
    if session is None:
        raise CompilationError("production source arenas require a sealed source session")
    previous = getattr(analysis, "_source_authority_context", None)
    if previous is not None:
        if type(previous) is not SourceAuthorityContext:
            raise CompilationError("source context requires its compiler-owned immutable context")
        return previous.require(analysis)
    session.verify(analysis)
    policy = {name: getattr(analysis, name) for name in CONTEXT_FIELDS if hasattr(analysis, name)}
    require_context_fields(policy, session)
    # Values are closed by SourceEffects' supported setters/seal, not copied
    # from arbitrary callers at a later successful token lookup.
    canonical = json.dumps(analysis._summary_authority(), sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    fields = _policy_key(policy, session)
    key = sha256(json.dumps((session.identity, canonical, fields),
                           separators=(",", ":")).encode()).hexdigest()
    context = session.authority_context(key, policy, canonical, fields)
    object.__setattr__(analysis, "_source_authority_context", context)
    return context


def fork_derivations(analysis):
    """A transaction/fork may read parent facts but cannot publish handles back."""
    context = getattr(analysis, "_source_authority_context", None)
    arena = getattr(analysis, "_source_derivation_arena", None)
    if context is not None and type(arena) is DerivationArena and arena._context is context:
        object.__setattr__(analysis, "_source_derivation_arena", arena.fork())
