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
    def __init__(self, analysis, graph, *, joined_storage=None):
        from compiler.frontend.resource_reachability import (
            NativeEffectGraph,
            NativeEffectNode,
            NativePresenceConditional,
            ResourceAccess,
        )
        if type(graph) is not NativeEffectGraph:
            raise CompilationError("native lifetime context requires its issued original graph")
        graph.validate(analysis, graph.procedure, graph._selection)
        self._joined = None
        if joined_storage is not None:
            from compiler.frontend.descriptor_teams import DescriptorTeamProof, _team_attachments
            from compiler.frontend.native_completion import (
                NativeGraphCompletionProof,
                _graph_completion_attachments,
            )
            if (type(joined_storage) is not DescriptorTeamProof or not graph.completion_eligible
                    or joined_storage._analysis is not analysis
                    or joined_storage.procedure != graph.procedure
                    or len(joined_storage._originals) != len(graph._selection)
                    or any(a is not b for a,b in zip(joined_storage._originals,graph._selection,strict=True))):
                raise CompilationError("native group lifetime requires its exact original joined storage")
            joined_storage.validate(analysis,graph.procedure,joined_storage.selected_node_ids)
            completed = [proof for proof in getattr(analysis,"_native_graph_completions",{}).values()
                         if type(proof) is NativeGraphCompletionProof and proof._graph is graph
                         and proof._ordinary_join is not None and proof._call_storage is None
                         and proof.procedure == graph.procedure
                         and proof.selected_node_ids == joined_storage.selected_node_ids]
            if len(completed) != 1:
                raise CompilationError("native group lifetime requires its same-graph ordinary whole-group completion")
            completion = completed[0].validate(analysis,graph.procedure,joined_storage.selected_node_ids)
            self._joined = (joined_storage,completion,_team_attachments(joined_storage),
                            _graph_completion_attachments(completion),analysis._descriptor_team_proofs)
        elif (_original_call_control(graph._selection) is None
                or not graph.retirement_effects_eligible):
            raise CompilationError("native lifetime context requires one complete original CALL graph")
        self.analysis, self.graph = analysis, graph
        self.types = _TypeBuilder(analysis)
        self.schemas = {}
        self._readonly_target_schemas = {}
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

    @property
    def group_only(self):
        return self._joined is not None

    @property
    def group_completion(self):
        return None if self._joined is None else self._joined[1]

    def require_checkpoint(self):
        if self._joined is not None:
            from compiler.frontend.native_completion import _require_graph_completion_attachments
            team,completion,team_fields,completion_fields,registry = self._joined
            if (getattr(self.analysis,"_descriptor_team_proofs",None) is not registry
                    or type(registry) is not dict or dict.get(registry,team.identity) is not team
                    or any(object.__getattribute__(owner,name) is not value for owner,name,value in team_fields)
                    or completion._graph is not self.graph or completion._ordinary_join is None
                    or completion._call_storage is not None or team.procedure != self.graph.procedure
                    or team.selected_node_ids != completion.selected_node_ids
                    or len(team._originals) != len(self.graph._selection)
                    or any(a is not b for a,b in zip(team._originals,self.graph._selection,strict=True))):
                raise CompilationError("native group lifetime changed its original storage/completion receipts")
            _require_graph_completion_attachments(completion,completion_fields)
        return self

    def checkpoint(self):
        self.require_checkpoint()
        self.graph.validate(self.analysis, self.graph.procedure, self.graph._selection)
        if self._joined is not None:
            team,completion,*_rest = self._joined
            team.validate(self.analysis,team.procedure,team.selected_node_ids)
            completion.validate(self.analysis,team.procedure,team.selected_node_ids)
        return self.require_checkpoint()

    def readonly_target(self, binding):
        """Declaration/read role only, separate from binding()/pointer grants."""
        self.require_checkpoint()
        if (self._joined is None or binding is None or binding.rank != 1
                or not {"target","allocatable"} <= binding.attributes
                or binding.attributes - {"target","allocatable","public","dimension"}
                or _typename(binding.dtype) is None or binding.declaring_scope is None
                or _kind(binding.declaring_scope.node) != "Module"
                or binding.declaring_scope.bindings.get(binding.name) is not binding):
            return False
        paths = [item for accesses in self.accesses.values() for item in accesses
                 if item.storage.binding is binding and item.kind in {"read","descriptor"}
                 and item.procedure == self.graph.procedure]
        if not paths:
            return False
        cached = self._readonly_target_schemas.get(id(binding))
        if cached is None:
            cached = binding,self.record_schema(binding)  # Existing fixed primitive declaration check.
            self._readonly_target_schemas[id(binding)] = cached
        if cached[0] is not binding or any(item.storage.components
                and item.storage.components[0].schema is not cached[1] for item in paths):
            raise CompilationError("native storage target read changed its original declaring type")
        return True

    def require_target_read(self, routine, original, binding, *, descriptor=False):
        if not self.readonly_target(binding) or routine.qualified != self.graph.procedure:
            raise CompilationError("native storage target read requires its exact whole original joined group")
        accesses = self.accesses.get(id(original),())
        if not any(item.original is original and item.storage.binding is binding
                   and item.procedure == routine.qualified
                   and item.kind == ("descriptor" if descriptor else "read") for item in accesses):
            raise CompilationError("native storage target read is absent from its original effect graph")
        return self.require_checkpoint()

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

    def scalar_reduction_associations(self, procedure, originals, binding):
        """Exact original scalar actuals, only after whole-call retirement.

        No payload, initialization, alias or execution permission is issued.
        Pointer team bindings remain a separate role.
        """
        from compiler.frontend.native_completion import _incoming_array_calls
        from compiler.frontend.native_demand import _Declarations

        self.team_bindings(procedure, originals)  # Authenticate the whole group.
        routine = self.analysis._require_original(procedure)
        declarations = _Declarations()
        if (routine.scope.bindings.get(binding.name) is not binding
                or binding.name not in routine.arguments or binding.rank
                or binding.intent not in {"out", "inout"}
                or binding.dtype not in {"real", "integer", "logical"} or binding.kind not in {4, 8}
                or binding.attributes - {"intent"}):
            raise CompilationError("native scalar reduction requires an exact plain OUT or INOUT formal")
        rows = []
        for instance, edge, mapping in _incoming_array_calls(self.analysis, self.graph, procedure, binding):
            caller = self.analysis._require_original(edge.caller)
            actual = mapping.binding
            if (actual is None or _kind(mapping.actual) != "Name" or actual.rank
                    or actual.signature() != binding.signature() or actual.attributes
                    or actual.declaring_scope is not caller.scope or actual.name in caller.arguments
                    or caller.scope.bindings.get(actual.name) is not actual
                    or self.analysis._binding(edge.resolved._scope, str(mapping.actual).lower()) is not actual
                    or mapping.actual is not edge.resolved.actuals[routine.arguments.index(binding.name)]):
                raise CompilationError("native scalar reduction requires its exact caller-owned plain scalar local")
            specification = tuple(node for node in getattr(caller.scope.node, "content", ())
                                  if _kind(node) == "Specification_Part")
            if (not declarations.independent(caller.scope)
                    or any(_kind(node) in {"Namelist_Stmt", "Data_Stmt"} for node in walk(specification))):
                raise CompilationError("native scalar reduction actual has unproved storage association")
            if any(item is not mapping and item.binding is actual for item in edge.resolved.associations):
                raise CompilationError("native scalar reduction actual aliases another original argument")
            rows.append({"instance": instance.ordinal, "caller": edge.caller, "callee": procedure,
                         "formal": binding.public(), "actual": actual.public(),
                         "original_call": str(edge.original), "whole_original_call_retirement_required": True})
        return tuple(rows)

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
                *((self._joined[0],self._joined[1]) if self._joined is not None else ()),
                *(call.edge.resolved for template in self.graph.templates for call in template.calls),
                *(terminal.proof for template in self.graph.templates for terminal in template.terminations))

    def public(self):
        if self._joined is not None:
            return {'kind':'original_native_joined_group_storage_context','native_graph_identity':self.graph.identity,
                    'procedure':self.graph.procedure,'whole_original_group_retirement_required':True,
                    'completion_identity':self._joined[1].identity,'storage_team_identity':self._joined[0].identity,
                    'retirement_required':True,'payload_access_authorized':False,'entry_definition_authorized':False,
                    'alias_authorized':False,'ownership_authorized':False,'capture_authorized':False,
                    'physical_sections_authorized':False,'target_role':'original primitive-field reads and descriptor inquiries only'}
        return {'kind': 'original_native_graph_lifetime_context', 'native_graph_identity': self.graph.identity,
                'procedure': self.graph.procedure, 'confined_pointer_count': len(self.pointers),
                'whole_original_call_and_all_callees_required': True, 'retirement_required': True,
                'payload_access_authorized': False, 'entry_definition_authorized': False,
                'alias_authorized': False, 'physical_sections_authorized': False}
