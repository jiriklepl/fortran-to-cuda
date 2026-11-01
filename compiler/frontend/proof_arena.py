"""Issuer-owned immutable derivations under one closed authority context.

The append-only index is implementation state, not an analysis registry. A
published result never points at a mutable parser projection or generated IR.
Failed and incomplete derivations are not installed.
"""
from dataclasses import dataclass
from hashlib import sha256

from compiler.ir import CompilationError

_ISSUER = object()
_FAMILIES = {}


@dataclass(frozen=True, slots=True, eq=False)
class DerivationFamily:
    name: str
    version: int
    _producer: object
    _legacy_import: bool
    _issuer: object


def _register_family(name, version, producer, *, legacy_import=False):
    """Trusted compiler implementation registration, never a source API."""
    key = name, version
    if key in _FAMILIES:
        raise CompilationError("source derivation family is already registered")
    family = DerivationFamily(name, version, producer, legacy_import, _ISSUER)
    _FAMILIES[key] = family
    return family


def _require_family(family):
    if (type(family) is not DerivationFamily or family._issuer is not _ISSUER
            or _FAMILIES.get((family.name, family.version)) is not family):
        raise CompilationError("source derivation requires its compiler-owned family")



@dataclass(frozen=True, slots=True)
class SourcePayload:
    record: str
    source_nodes: tuple = ()


@dataclass(frozen=True, slots=True)
class DerivedRecord:
    family: object
    query: tuple
    record: str
    identity: str
    dependencies: tuple
    source_nodes: tuple
    _issuer: object


class DerivationArena:
    __slots__ = ("_context", "_parent", "_records", "_handles", "_active", "_derivations")

    def __init__(self, context, parent=None):
        if parent is not None and (type(parent) is not DerivationArena or parent._context is not context):
            raise CompilationError("derivation fork requires the exact immutable context")
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_records", parent._records if parent is not None else {})
        object.__setattr__(self, "_handles", {})
        object.__setattr__(self, "_active", parent._active if parent is not None else set())
        object.__setattr__(self, "_derivations", parent._derivations if parent is not None else {})

    def __setattr__(self, name, value):
        raise CompilationError("derivation arena ownership is immutable")

    def derive(self, family, query, inputs, owner, *, dependencies=()):
        _require_family(family)
        self._context.require(owner)
        dependencies = tuple(self.require(self._context, item) for item in dependencies)
        key = family, query
        previous = self._find(key)
        if previous is not None:
            if previous.dependencies != dependencies:
                raise CompilationError("source derivation dependencies changed for the same query")
            return previous
        if key in self._active:
            raise CompilationError("recursive source derivation is unavailable")
        self._active.add(key)
        try:
            reader = owner if family._legacy_import else self._context.reader(owner)
            payload = family._producer(reader, *inputs)
            self._context.require(owner)
            payload = SourcePayload(payload) if type(payload) is str else payload
            if type(payload) is not SourcePayload:
                raise CompilationError("source derivation requires its immutable payload")
            record = payload.record
            for identity in payload.source_nodes:
                self._context.session.original_node(identity)
            if type(record) is not str:
                raise CompilationError("source derivation must publish an immutable encoded record")
            result = DerivedRecord(family, query, record, sha256(record.encode()).hexdigest(),
                                   tuple(dependencies), tuple(payload.source_nodes), _ISSUER)
            self._records[key] = result
            self._derivations[family.name] = self._derivations.get(family.name, 0) + 1
            return result
        finally:
            self._active.remove(key)

    def require(self, context, result):
        if (context is not self._context or type(result) is not DerivedRecord
                or result._issuer is not _ISSUER
                or self._find((result.family, result.query)) is not result):
            raise CompilationError("source derivation belongs to a foreign authority arena")
        _require_family(result.family)
        return result

    def _find(self, key):
        # Source facts outlive speculative generation; only consumption
        # handles are branch-local. Every entry is a complete immutable result
        # under this exact context, with no generated/capture state.
        return self._records.get(key)

    def register_handle(self, key, handle):
        if self._find_handle(key) is not None:
            raise CompilationError("source proof handle is already issued in this arena")
        self._handles[key] = handle

    def _find_handle(self, key):
        current = self
        while current is not None:
            found = current._handles.get(key)
            if found is not None:
                return found
            current = current._parent
        return None

    def require_handle(self, key, handle):
        if self._find_handle(key) is not handle:
            raise CompilationError("source proof requires its exact arena-issued handle")

    def fork(self):
        return DerivationArena(self._context, self)

    def accepts(self, arena):
        current = self
        while current is not None:
            if current is arena:
                return True
            current = current._parent
        return False

    def derivation_count(self, family):
        return self._derivations.get(family, 0)
