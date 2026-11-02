"""Read-only source indexes for an existing graph's native retirement scan.

This is not an issuer. The original native graph owns every association and
pointer occurrence; the independent storage scanner still visits all bodies.
"""
from types import MappingProxyType

from fparser.two.utils import walk

from compiler.frontend.call_bindings import _original_call_control
from compiler.frontend.component_bindings import _typename
from compiler.frontend.source_objects import _TypeBuilder
from compiler.ir import CompilationError


def _kind(node):
    return type(node).__name__


class NativeLifetimeContext:
    def __init__(self, analysis, graph):
        from compiler.frontend.resource_reachability import (
            NativeEffectGraph,
            NativeEffectNode,
            NativePresenceConditional,
            ResourceAccess,
        )
        if type(graph) is not NativeEffectGraph:
            raise CompilationError("native lifetime context requires its issued original graph")
        graph.validate(analysis, graph.procedure, graph._selection)
        if (_original_call_control(graph._selection) is None
                or not graph.retirement_effects_eligible):
            raise CompilationError("native lifetime context requires one complete original CALL graph")
        self.analysis, self.graph = analysis, graph
        self.types = _TypeBuilder(analysis)
        self.schemas = {}
        bindings, pointers, accesses, calls, groups = {}, {}, {}, {}, {}
        for template in graph.templates:
            for pointer in template.confined_pointers:
                schema = self.record_schema(pointer.target.binding)
                if self.record_schema(pointer.binding, pointer=True) is not schema:
                    raise CompilationError("native lifetime pointer changed its original declaring type")
                bindings[id(pointer.target.binding)] = pointer.target.binding
                bindings[id(pointer.binding)] = pointer.binding
                pointers[id(pointer.assignment)] = template.procedure, pointer
            for call in template.calls:
                calls[id(call.edge.original)] = call
            pending = list(template.nodes)
            while pending:
                node = pending.pop()
                if isinstance(node, NativeEffectNode):
                    if node.kind == "joined_group":
                        groups[template.procedure, tuple(map(id, node.original))] = tuple(node.original)
                    pending.extend(node.items)
                elif isinstance(node, NativePresenceConditional):
                    pending.extend((*node.present, *node.absent))
                elif isinstance(node, ResourceAccess):
                    accesses.setdefault(id(node.original), []).append(node)
        # Canonical record actuals remain linked to exact graph-issued CALL
        # associations. No source-name disjointness or payload range is inferred.
        for call in calls.values():
            for item in call.edge.resolved.associations:
                formal, actual = item.formal_binding, item.binding
                if not _typename(formal.dtype):
                    continue
                if (actual is None or _kind(item.actual) != "Name" or formal.rank != 1
                        or actual.rank != 1 or formal.intent not in {"in", "inout"}
                        or "target" not in formal.attributes or "target" not in actual.attributes
                        or formal.attributes & {"pointer", "allocatable", "optional", "value", "volatile", "asynchronous"}
                        or actual.attributes & {"pointer", "optional", "value", "volatile", "asynchronous"}):
                    continue
                if self.record_schema(formal) is not self.record_schema(actual):
                    raise CompilationError("native lifetime record association changed its declaring type")
                bindings[id(formal)], bindings[id(actual)] = formal, actual
        self.bindings = MappingProxyType(bindings)
        self.pointers = MappingProxyType(pointers)
        self.accesses = MappingProxyType({key: tuple(value) for key, value in accesses.items()})
        self.calls, self.groups = MappingProxyType(calls), MappingProxyType(groups)
        self.checkpoint()

    def record_schema(self, binding, *, pointer=False):
        scope = binding.declaring_scope
        if scope is None or scope.bindings.get(binding.name) is not binding:
            raise CompilationError("native lifetime record lost original declaring storage")
        if binding.attributes & {"optional", "volatile", "asynchronous", "value"}:
            raise CompilationError("native lifetime record has unsupported observation or association")
        if pointer and (binding.rank or binding.attributes != frozenset({"pointer"})):
            raise CompilationError("native lifetime pointer is not exact original unsaved local storage")
        schema = self.types.resolve(scope, _typename(binding.dtype))
        if schema is None:
            raise CompilationError("native lifetime record declaring type is unavailable")
        if id(schema) not in self.schemas:
            record = self.types.schema(schema)
            if (not record['fields'] or any(field['shape'] or field['type'] not in {'real', 'integer', 'logical'}
                    or field['kind'] not in {4, 8} for field in record['fields'])):
                raise CompilationError("native lifetime record requires fixed primitive scalar fields")
            self.schemas[id(schema)] = schema, record
        return schema

    def checkpoint(self):
        self.graph.validate(self.analysis, self.graph.procedure, self.graph._selection)
        return self

    def require_selection(self, procedure, originals):
        if (procedure != self.graph.procedure or len(originals) != len(self.graph._selection)
                or any(a is not b for a, b in zip(originals, self.graph._selection, strict=True))):
            raise CompilationError("native lifetime context changed its original CALL selection")
        return self.checkpoint()

    def team_bindings(self, procedure, originals):
        self.checkpoint()
        key = procedure, tuple(map(id, originals))
        if key not in self.groups:
            raise CompilationError("native lifetime team is not an exact graph-owned original group")
        nodes = {id(node) for original in originals for node in walk(original)}
        result = []
        for owner, pointer in self.pointers.values():
            if owner == procedure and id(pointer.assignment) in nodes and id(pointer.loop) in nodes:
                result.append((pointer.binding, pointer.target.binding))
        return tuple(result)

    def binding(self, binding):
        return binding is not None and self.bindings.get(id(binding)) is binding

    def field(self, routine, original):
        if _kind(original) != 'Data_Ref':
            return None
        paths = [item.storage for item in self.accesses.get(id(original), ())
                 if item.original is original and item.procedure == routine.qualified]
        if not paths:
            return None
        path = paths[0]
        if (not self.binding(path.binding) or len(path.components) != 1
                or any(item.binding is not path.binding or len(item.components) != 1
                       or item.components[0].schema is not path.components[0].schema
                       or item.components[0].declaration is not path.components[0].declaration
                       or item.components[0] != path.components[0] for item in paths)):
            return None
        component, = path.components
        schema, record = self.schemas[id(self.record_schema(path.binding))]
        if (component.schema is not schema or component.rank or component.attributes &
                {'pointer', 'allocatable', 'target', 'volatile', 'asynchronous'}
                or component.name not in {item['name'] for item in record['fields']}):
            raise CompilationError("native lifetime field lost its exact fixed scalar declaration")
        return path

    def association(self, resolved, item):
        call = self.calls.get(id(resolved._original))
        if call is None or call.edge.resolved is not resolved:
            return None
        if not self.binding(item.formal_binding) or not self.binding(item.binding):
            return None
        return item.binding

    def component_actual(self, routine, resolved, item):
        call = self.calls.get(id(resolved._original))
        if call is None or call.edge.resolved is not resolved:
            return None
        if (item.formal_binding.rank or item.formal_binding.dtype not in {'real', 'integer', 'logical'}
                or item.formal_binding.attributes & {'pointer', 'allocatable', 'target', 'value', 'volatile', 'asynchronous'}):
            return None
        return self.field(routine, item.actual)

    def pointer_assignment(self, routine, original):
        row = self.pointers.get(id(original))
        return row is not None and row[0] == routine.qualified and row[1].assignment is original

    def prerequisites(self):
        return (self.graph, *self.graph.templates,
                *(call.edge.resolved for template in self.graph.templates for call in template.calls),
                *(terminal.proof for template in self.graph.templates for terminal in template.terminations))

    def public(self):
        return {'kind': 'original_native_graph_lifetime_context', 'native_graph_identity': self.graph.identity,
                'procedure': self.graph.procedure, 'confined_pointer_count': len(self.pointers),
                'whole_original_call_and_all_callees_required': True, 'retirement_required': True,
                'payload_access_authorized': False, 'entry_definition_authorized': False,
                'alias_authorized': False, 'physical_sections_authorized': False}
