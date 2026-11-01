"""Descriptor/storage facts owned by the existing immutable source arena.

The scanner is an issuer, not a repeated validation service. A completed scan
publishes one immutable fact. Consumers check its context and exact consumed
prerequisites; mutable generation state and runtime lifetimes are not facts.
"""
import json
from dataclasses import dataclass, replace

from compiler.frontend.proof_arena import DerivedRecord, _register_family
from compiler.ir import CompilationError

_ABSENT = object()


def _payload(_owner, encoded):
    return encoded


DESCRIPTOR = _register_family("original_descriptor_window", 1, _payload, legacy_import=True)
STORAGE = _register_family("original_storage_mutation", 1, _payload, legacy_import=True)
TEAM = _register_family("original_descriptor_team", 1, _payload, legacy_import=True)
_PREREQUISITE = _register_family("original_source_prerequisite", 1, _payload, legacy_import=True)


def _encoded(value):
    from compiler.frontend.structured_effects import _thaw
    return value if type(value) is str else json.dumps(_thaw(value), sort_keys=True,
                                                     separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class RegistryDependency:
    owner: object
    context: object
    arena: object
    name: str
    registry: object
    key: object
    token: object
    marker: object
    payload: object
    identity: object
    fact: object
    issued: tuple

    def require(self):
        self.context.require(self.owner)
        registry = getattr(self.owner, self.name, None)
        arena = self.context.derivation_arena(self.owner)
        rebound = arena is not self.arena and arena.accepts(self.arena)
        if (type(registry) is not dict
                or (registry is not self.registry and not rebound)
                or registry.get(self.key) is not self.token
                or getattr(self.owner, self.name + "_owner", _ABSENT) is not self.marker
                or getattr(self.token, "_record", _ABSENT) is not self.payload
                or getattr(self.token, "identity", _ABSENT) != self.identity):
            raise CompilationError("issued source fact consumed prerequisite changed")
        if self.fact is not None:
            if getattr(self.token, "_source_fact", None) is not self.fact:
                raise CompilationError("issued prerequisite fact handle changed")
            self.fact.require_ownership(self.token)
        if self.issued:
            context, origin, derived = self.issued
            context.require(self.owner)
            current = context.derivation_arena(self.owner)
            if not current.accepts(origin):
                raise CompilationError("source prerequisite belongs to a discarded generation")
            current.require(context, derived)
            current.require_handle((derived.family.name, id(self.owner), derived.query), self.token)


def dependencies(rows):
    """Retain used entries, including their imported transitive prerequisites.

    Adding unrelated issuer entries does not alter an already derived fact.
    Registries are compiler-owned; arbitrary Python callbacks are not source
    transformations. Replacing a consumed entry remains a hard error.
    """
    result = {}
    for owner, name, registry, key, token, *_unused in rows:
        if type(registry) is not dict or registry.get(key) is not token:
            raise CompilationError("issued source fact requires its exact prerequisite registry")
        if getattr(owner, "_source_session", None) is None:
            raise CompilationError("sealed source fact cannot import a mutable diagnostic prerequisite")
        context = owner.source_authority_context()
        derived = getattr(token, "_derived", None)
        issued = ((token._context, token._arena, derived) if type(derived) is DerivedRecord else ())
        dependency = RegistryDependency(owner, context, context.derivation_arena(owner), name, registry, key, token,
            getattr(owner, name + "_owner", _ABSENT), getattr(token, "_record", _ABSENT),
            getattr(token, "identity", _ABSENT), getattr(token, "_source_fact", None), issued)
        dependency.require()
        result[id(owner), name, key] = dependency
        fact = getattr(token, "_source_fact", None)
        if fact is not None:
            fact.require(token)
            for nested in fact.dependencies:
                result[id(nested.owner), nested.name, nested.key] = nested
    return tuple(result.values())


def token_dependencies(tokens):
    """Find ownership for explicitly consumed typed prerequisite tokens."""
    rows, seen = [], set()
    for token in tokens:
        if token is None or id(token) in seen:
            continue
        seen.add(id(token))
        owner = getattr(token, "_analysis", None)
        if owner is None:
            raise CompilationError("source prerequisite has no original analysis owner")
        from compiler.frontend.descriptor_associations import _ISSUER, FixedScalarAssociation

        if type(token) is FixedScalarAssociation:
            # This is a scan-local source value, freshly derived and checked
            # by the complete scanner, then embedded in its immutable record.
            # It has no mutable registration or generation/capture authority.
            context = owner.source_authority_context()
            if (token._issuer is not _ISSUER or type(token._record) is not str
                    or context.session.routine(owner, token._routine.qualified, function=True) is not token._routine
                    or context.session.original_node(id(token._original)) is not token._original):
                raise CompilationError("local source association requires its sealed original query")
            continue
        found = False
        for name, registry in vars(owner).items():
            if name.startswith("_") and type(registry) is dict:
                for key, value in registry.items():
                    if value is token:
                        rows.append((owner, name, registry, key, token))
                        found = True
        if not found:
            raise CompilationError("source prerequisite has no exact issuer registration")
    return dependencies(rows)


@dataclass(frozen=True, slots=True)
class SourceFact:
    owner: object
    context: object
    arena: object
    derived: object
    query: tuple
    payload: object
    identity: str
    dependencies: tuple

    def require_ownership(self, proof):
        self.context.require(self.owner)
        arena = self.context.derivation_arena(self.owner)
        if not arena.accepts(self.arena):
            raise CompilationError("source fact belongs to another generation branch")
        arena.require(self.context, self.derived)
        arena.require_handle((self.derived.family.name, id(self.owner), self.query), proof)
        if proof._record is not self.payload or proof.identity != self.identity:
            raise CompilationError("source fact immutable payload changed")
        return proof

    def require(self, proof):
        self.require_ownership(proof)
        for dependency in self.dependencies:
            dependency.require()
        return proof


def lookup(analysis, family, query):
    if getattr(analysis, "_source_session", None) is None:
        return None
    context = analysis.source_authority_context()
    arena = context.derivation_arena(analysis)
    proof = arena._find_handle((family.name, id(analysis), query))
    if proof is not None:
        fact = getattr(proof, "_source_fact", None)
        if fact is None or fact.derived.family is not family or fact.query != query:
            raise CompilationError("source fact handle has a different original query")
        fact.require(proof)
    return proof


def issue(analysis, family, query, proof, prerequisites=()):
    if getattr(analysis, "_source_session", None) is None:
        return proof
    registry_name = {DESCRIPTOR: "_descriptor_window_proofs", STORAGE: "_storage_mutation_effects",
                     TEAM: "_descriptor_team_proofs"}[family]
    registry = getattr(analysis, registry_name)
    if registry.get(proof.identity) is not proof:
        raise CompilationError("source fact lost its original issuer registration")
    try:
        context = analysis.source_authority_context()
        arena = context.derivation_arena(analysis)
        imported = []
        for dependency in prerequisites:
            dependency.require()
            key = (id(dependency.owner), dependency.context._intern_key, dependency.name,
                   dependency.key, str(dependency.identity))
            encoded = json.dumps((dependency.context._intern_key, dependency.name,
                                  dependency.key, str(dependency.identity)), separators=(",", ":"))
            imported.append(arena.derive(_PREREQUISITE, key, (encoded,), analysis))
        encoded = _encoded(proof._record)
        # Encoding is part of issuance. Only completed original facts may
        # enter the arena; live-file checks remain mandatory at publication.
        analysis.verify_publication()
        context.require(analysis)
        for dependency in prerequisites:
            dependency.require()
        derived_query = query, tuple(item.identity for item in imported)
        derived = arena.derive(family, derived_query, (encoded,), analysis, dependencies=tuple(imported))
        if derived.record != encoded:
            raise CompilationError("source fact differs from its immutable issued derivation")
        # Never retarget a prior handle, including an unsealed diagnostic
        # token issued before this source session was sealed.
        replacement = replace(proof, _source_fact=None)
        fact = SourceFact(analysis, context, arena, derived, query, replacement._record,
                          replacement.identity, tuple(prerequisites))
        object.__setattr__(replacement, "_source_fact", fact)
        arena.register_handle((family.name, id(analysis), query), replacement)
        registry[replacement.identity] = replacement
        return replacement
    except BaseException:
        if getattr(proof, "_source_fact", None) is None and registry.get(proof.identity) is proof:
            del registry[proof.identity]
        raise
