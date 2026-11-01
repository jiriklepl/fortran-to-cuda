"""One authenticated, immutable source graph shared by compilation projections.

Original declaration/syntax state and persistent projection values are sealed.
Effect caches, context interning and generated handles are separate work state.
A session query authenticates graph identities; publication authenticates files.
"""
import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

from compiler.frontend.immutable_graph import GraphFreezer
from compiler.frontend.sealed_syntax import sealed as syntax_sealed
from compiler.ir import CompilationError


class ProjectedRoutines(Mapping):
    """Persistent private routine selection over one immutable original registry."""
    __slots__ = ("base", "_overrides")

    def __init__(self, base, overrides=()):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "_overrides", MappingProxyType(dict(overrides)))

    def __setattr__(self, name, value):
        raise CompilationError("private projection registry identity is immutable")

    @property
    def overrides(self):
        return self._overrides

    def __getitem__(self, key):
        return self._overrides[key] if key in self._overrides else self.base[key]

    def __setitem__(self, key, value):
        raise CompilationError("routine projection is immutable; use project_routine")

    def with_override(self, key, value):
        return ProjectedRoutines(self.base, {**self._overrides, key: value})

    def __delitem__(self, key):
        raise CompilationError("private projections cannot remove original source roles")

    def __iter__(self):
        yield from self.base
        yield from (key for key in self._overrides if key not in self.base)

    def __len__(self):
        return len(self.base) + len(self._overrides.keys() - self.base.keys())

    def __copy__(self):
        return self


# Authoritative roots, distinct from mutable caches/proof token registries.
_ROOTS = ("inputs", "sources", "modules", "routines", "functions", "numerical_helpers",
          "kind_expressions", "_source_scopes", "_associate_scopes", "_block_scopes", "_block_scope_roles", "_source_roles",
          "_numerical_roles", "_source_module_roles", "_source_module_bindings",
          "_source_object_scopes", "_source_function_scopes", "_source_provenance",
          "_source_object_calls", "_source_syntax")


class _ContextIndex:
    """Issuer work state, distinct from the immutable semantic source maps."""
    __slots__ = ("_by_key", "_roles")

    def __init__(self):
        object.__setattr__(self, "_by_key", {})
        object.__setattr__(self, "_roles", {})

    def __setattr__(self, name, value):
        raise CompilationError("source context index ownership is immutable")

    def get(self, key):
        return self._by_key.get(key)

    def register(self, key, context):
        if key in self._by_key:
            raise CompilationError("source context is already interned")
        self._by_key[key] = context

    def register_role(self, routine):
        self._roles[id(routine)] = routine

    def owns_role(self, routine):
        return self._roles.get(id(routine)) is routine


@dataclass(frozen=True, slots=True)
class SourceSession:
    identity: str
    roots: object
    objects: object
    _execution_nodes: object
    _execution_membership: object
    _specification_comments: object
    _lexical_parents: object
    _contexts: object

    @classmethod
    def create(cls, analysis):
        from fparser.two.utils import walk

        from compiler.frontend.component_bindings import AssociationScope, ComponentAccess, ComponentSchema
        from compiler.frontend.lowering import _KindScope
        from compiler.frontend.source_effects import Binding, Routine, Scope, _kind, _part
        from compiler.frontend.source_functions import _validate_scope
        from compiler.frontend.source_inputs import SourceInputs
        from compiler.frontend.source_membership import ExecutionMembership
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
        identity = sha256(json.dumps({"source_session_version": 2, "authority": authority},
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
        # The syntax snapshot already bounded and authenticated these trees.
        # Keep exact per-routine membership beside that immutable authority;
        # consumers need not format or walk a whole procedure for every node.
        execution_membership = {name: ExecutionMembership.create(routine.execution)
                                for name, routine in roots["routines"].items()}
        execution_nodes = {name: membership.nodes for name, membership in execution_membership.items()}
        # Repeated participation queries need specification comments only.
        # Index immutable original syntax, without resolving directives or
        # changing when an unrelated malformed directive is diagnosed.
        scopes = (*roots["modules"].values(), *(routine.scope for routine in roots["routines"].values()))
        specification_comments = {
            id(scope.node): (scope.node, tuple(node for node in walk(_part(scope.node, "Specification_Part"))
                                               if _kind(node) == "Comment"))
            for scope in scopes
        }
        parents = {}
        pending, visited = [scope.node for scope in scopes], set()
        while pending:
            node = pending.pop()
            if id(node) in visited:
                continue
            visited.add(id(node))
            if isinstance(node, (tuple, list)):
                pending.extend(node)
            elif hasattr(node, "items") or hasattr(node, "content"):
                parents[id(node)] = (node, getattr(node, "parent", None))
                pending.extend(getattr(node, "content", getattr(node, "items", ())) or ())
        return cls(identity, MappingProxyType(roots), MappingProxyType(freezer.objects),
                   MappingProxyType(execution_nodes), MappingProxyType(execution_membership),
                   MappingProxyType(specification_comments), MappingProxyType(parents), _ContextIndex())

    def authority_context(self, identity, policy, authority, fields_key):
        from compiler.frontend.authority_context import SourceAuthorityContext
        from compiler.frontend.proof_arena import DerivationArena

        context = self._contexts.get(identity)
        if context is None:
            context = SourceAuthorityContext(identity, self, MappingProxyType(dict(policy)), authority, fields_key)
            object.__setattr__(context, "_arena", DerivationArena(context))
            self._contexts.register(identity, context)
        return context

    def owns_context(self, context):
        return self._contexts.get(context._intern_key) is context

    def _register_projection(self, routine):
        self._contexts.register_role(routine)

    def owns_projection(self, routine):
        return self.contains(routine) or self._contexts.owns_role(routine)

    def original_node(self, identity):
        found = self._lexical_parents.get(identity)
        if found is None:
            raise CompilationError("source capability refers to a foreign original node")
        return found[0]

    def lexical_parent(self, node):
        found = self._lexical_parents.get(id(node))
        if found is None or found[0] is not node:
            raise CompilationError("lexical parent requires an original source node")
        return found[1]

    def verify(self, analysis):
        for name, expected in self.roots.items():
            actual = getattr(analysis, name)
            if name == "routines" and isinstance(actual, ProjectedRoutines) and actual.base is expected:
                continue
            if actual is not expected:
                raise CompilationError("compilation-session original registry changed: " + name)

    def contains(self, value):
        return self.objects.get(id(value)) is value

    def execution_nodes(self, analysis, routine):
        if self.routine(analysis, routine.qualified) is not routine:
            raise CompilationError("original execution membership requires its exact source routine")
        return self._execution_nodes[routine.qualified]

    def specification_comments(self, analysis):
        """Exact original syntax only; foreign projection nodes need a fresh walk."""
        self.verify(analysis)
        return self._specification_comments

    def execution_overlaps(self, analysis, routine, selections):
        if self.routine(analysis, routine.qualified) is not routine:
            raise CompilationError("original execution membership requires its exact source routine")
        return self._execution_membership[routine.qualified].overlaps(selections)

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
