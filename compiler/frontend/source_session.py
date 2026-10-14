"""One authenticated, immutable source graph shared by compilation projections.

Only original declaration/syntax state is sealed. Effect caches and private
routine projections remain explicit mutable state on SourceEffects. A session
query authenticates graph identities; publication authenticates live files.
"""
import json
from collections.abc import MutableMapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

from compiler.frontend.immutable_graph import GraphFreezer
from compiler.frontend.sealed_syntax import sealed as syntax_sealed
from compiler.ir import CompilationError


class ProjectedRoutines(MutableMapping):
    """Small explicit override map over one immutable original registry."""
    __slots__ = ("base", "_overrides")

    def __init__(self, base, overrides=()):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "_overrides", dict(overrides))

    def __setattr__(self, name, value):
        raise CompilationError("private projection registry identity is immutable")

    @property
    def overrides(self):
        return MappingProxyType(self._overrides)

    def __getitem__(self, key):
        return self._overrides[key] if key in self._overrides else self.base[key]

    def __setitem__(self, key, value):
        self._overrides[key] = value

    def __delitem__(self, key):
        raise CompilationError("private projections cannot remove original source roles")

    def __iter__(self):
        yield from self.base
        yield from (key for key in self._overrides if key not in self.base)

    def __len__(self):
        return len(self.base) + len(self._overrides.keys() - self.base.keys())

    def __copy__(self):
        return ProjectedRoutines(self.base, self._overrides)


# Authoritative roots, distinct from mutable caches/proof token registries.
_ROOTS = ("inputs", "sources", "modules", "routines", "functions", "numerical_helpers",
          "kind_expressions", "_source_scopes", "_associate_scopes", "_source_roles",
          "_numerical_roles", "_source_module_roles", "_source_module_bindings",
          "_source_object_scopes", "_source_function_scopes", "_source_provenance",
          "_source_object_calls", "_source_syntax")


@dataclass(frozen=True, slots=True)
class SourceSession:
    identity: str
    roots: object
    objects: object

    @classmethod
    def create(cls, analysis):
        from compiler.frontend.component_bindings import AssociationScope, ComponentAccess, ComponentSchema
        from compiler.frontend.lowering import _KindScope
        from compiler.frontend.source_effects import Binding, Routine, Scope
        from compiler.frontend.source_functions import _validate_scope
        from compiler.frontend.source_inputs import SourceInputs
        from compiler.frontend.source_objects import _members, _TypeBuilder
        from compiler.frontend.source_syntax import SourceSyntaxSnapshot, _Group, _Layout, _Node, _Sequence

        analysis.inputs.verify_live()
        analysis.inputs.require_captured_parsing()
        syntax = analysis._require_original_global()
        for name in analysis.routines:
            analysis._require_original_routine(name, _syntax=syntax)
        for scope, *_ in analysis._source_function_scopes.values():
            _validate_scope(analysis, scope, track=False)
        for scope, *_ in analysis._source_object_scopes.values():
            record = _TypeBuilder(analysis).scope(scope)
            for name, schema in getattr(scope, "component_types", {}).items():
                original = record[5].get(name)
                if (original is None or original[0] is not schema or original[1] is not schema.node
                        or original[2] != str(schema.node) or original[3] != _members(schema)):
                    raise CompilationError("cannot seal changed original object schema")
        authority = analysis._summary_authority()
        if authority["role"] != "source":
            raise CompilationError("cannot seal a private or changed source graph")
        identity = sha256(json.dumps({"source_session_version": 1, "authority": authority},
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        analysis._source_syntax = analysis._source_syntax.seal()
        analysis.inputs.verify_live()
        graph_types = (Binding, Scope, Routine, _KindScope, ComponentSchema,
                       ComponentAccess, AssociationScope, SourceInputs, SourceSyntaxSnapshot)
        freezer = GraphFreezer(owned_types=graph_types,
                               borrowed_types=(_Node, _Sequence, _Group, _Layout),
                               borrow_if=syntax_sealed)
        roots = {name: freezer.freeze(getattr(analysis, name)) for name in _ROOTS}
        for name, value in roots.items():
            setattr(analysis, name, value)
        return cls(identity, MappingProxyType(roots), MappingProxyType(freezer.objects))

    def verify(self, analysis):
        for name, expected in self.roots.items():
            actual = getattr(analysis, name)
            if name == "routines" and isinstance(actual, ProjectedRoutines) and actual.base is expected:
                continue
            if actual is not expected:
                raise CompilationError("compilation-session original registry changed: " + name)

    def contains(self, value):
        return self.objects.get(id(value)) is value

    def routine(self, analysis, name, *, function=False):
        self.verify(analysis)
        registry = "numerical_helpers" if function else "routines"
        original = self.roots[registry].get(name)
        current = getattr(analysis, registry).get(name)
        if original is None or current is not original:
            raise CompilationError("structured effects require original source-backed procedure authority")
        return original

    def projections(self, analysis):
        self.verify(analysis)
        routines = analysis.routines
        if not isinstance(routines, ProjectedRoutines):
            return []
        return [{"procedure": name, "signature": analysis._routine_signature(routine),
                 "execution": str(routine.execution)}
                for name, routine in sorted(routines.overrides.items())
                if routine is not self.roots["routines"].get(name)]
