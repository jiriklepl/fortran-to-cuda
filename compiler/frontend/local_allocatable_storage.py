"""Original local SAVE allocation identity, without payload or lifetime facts.

An independent numeric local allocation cannot acquire an alias merely because
it is selected for residency.  This token authenticates its original declaration
and direct uses.  A descriptor window must still inspect its entire selected
call closure, and reached runtime guards must establish allocation/bounds/alias
conditions.  Prefix writes, allocation statements and INTENT declarations grant
no current payload definitions or retention across an excluded frontier.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from hashlib import sha256

from fparser.two.utils import Base

from compiler.frontend.call_bindings import resolve_original_call
from compiler.frontend.descriptor_intrinsics import (
    intrinsic_scope_attachments,
    intrinsic_source_attachments,
    same_intrinsic_scope_attachments,
    same_intrinsic_source_attachments,
)
from compiler.frontend.descriptor_teams import DescriptorTeamProof
from compiler.frontend.proof_arena import SourcePayload, _register_family
from compiler.frontend.source_functions import (
    _original as _original_procedure,
)
from compiler.frontend.source_functions import (
    _scope_roles,
    _SessionScopeRoles,
    _validate_resolution,
    _validate_scope,
    _validate_scope_roles,
)
from compiler.ir import CompilationError
from compiler.ir.intrinsics import ARRAY_INQUIRIES, INTRINSICS, MODEL_INQUIRIES

_ISSUER = object()
_ABSENT = object()
_SCAN_LIMIT = 65536
_UNSAFE = {"pointer", "target", "optional", "volatile", "asynchronous", "value", "parameter", "protected"}
_FUNCTIONS = {"Part_Ref", "Function_Reference", "Intrinsic_Function_Reference", "Structure_Constructor"}
_INTRINSICS = set(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES | {
    "allocated", "sum", "product", "any", "all", "count", "minval", "maxval", "dot_product"}
_STATEMENTS = {"Assignment_Stmt", "Allocate_Stmt", "Deallocate_Stmt", "Call_Stmt", "If_Stmt",
               "If_Then_Stmt", "Else_If_Stmt", "Nonlabel_Do_Stmt", "Where_Stmt", "Where_Construct_Stmt",
               "Masked_Elsewhere_Stmt"}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _kind(node):
    return type(node).__name__


def _attached(value):
    return tuple((item.name, getattr(value, item.name)) for item in fields(value))


def _same(value, saved):
    return all(getattr(value, name) is item for name, item in saved)


def _raw_authority(analysis):
    """Strong, callback-free namespace receipts before any public serializer."""
    from compiler.frontend import source_session
    from compiler.frontend.source_session import ProjectedRoutines

    source = intrinsic_source_attachments(analysis)
    root_policy = source_session._ROOTS
    session = getattr(analysis, "_source_session", None)
    session_fields = _attached(session) if session is not None else ()
    projection = ((analysis.routines, analysis.routines.base, analysis.routines._overrides,
                   tuple(analysis.routines._overrides.items()))
                  if type(analysis.routines) is ProjectedRoutines else None)
    scopes, objects = {}, {}
    for table in (analysis.modules, analysis.routines, analysis.numerical_helpers,
                  analysis._associate_scopes, analysis._block_scopes):
        for value in table.values():
            objects[id(value)] = (value, _attached(value))
            scope = getattr(value, "scope", value)
            while hasattr(scope, "bindings") and id(scope) not in scopes:
                scopes[id(scope)] = scope
                scope = scope.parent
    states = tuple((intrinsic_scope_attachments(scope), getattr(scope, "qualified", _ABSENT),
                    tuple((binding, _attached(binding), getattr(binding, "component_object", _ABSENT),
                           getattr(binding, "associate_selector", _ABSENT)) for binding in scope.bindings.values()),
                    getattr(scope, "component_types", _ABSENT),
                    tuple(getattr(scope, "component_types", {}).items())) for scope in scopes.values())
    schemas = tuple((schema, _attached(schema), schema.members, tuple(schema.members.items()))
                    for _state, _qualified, _bindings, _mapping, items in states for _name, schema in items)
    parser, parser_containers, seen = [], [], set()
    pending = [state[0].node for state, *_rest in states]
    allowance = max(_SCAN_LIMIT, 8 * sum(len(text) for text in analysis.inputs._texts.values()))
    while pending:
        value = pending.pop()
        if not isinstance(value, (Base, tuple, list)) or id(value) in seen:
            continue
        seen.add(id(value))
        if len(seen) > allowance:
            raise CompilationError("local allocation original namespace syntax budget exhausted")
        if isinstance(value, (tuple, list)):
            children = tuple(value)
            parser_containers.append((value, children))
        else:
            attrs = tuple(vars(value).items())
            item = getattr(value, "item", None)
            parser.append((value, attrs, item, tuple(vars(item).items()) if item is not None else ()))
            children = getattr(value, "content", getattr(value, "items", ())) or ()
        pending.extend(children)
    inputs = analysis.inputs
    input_fields = tuple((name, getattr(inputs, name)) for name in
                         ("paths", "sources", "_texts", "entries", "dependencies", "configuration"))
    containers, seen, pending = [], set(), [value for _name, value in (*input_fields, *session_fields)]
    if projection is not None:
        pending.extend((projection[1], projection[2]))
    while pending:
        value = pending.pop()
        if id(value) in seen or not isinstance(value, (Mapping, list, tuple, set, frozenset)):
            continue
        seen.add(id(value))
        if isinstance(value, Mapping):
            items = tuple(value.items())
            containers.append((value, "mapping", items))
            pending.extend(item for _key, item in items)
        else:
            items = tuple(value)
            containers.append((value, "set" if isinstance(value, (set, frozenset)) else "sequence", items))
            pending.extend(items)
    return (source, (session, session_fields, projection, source_session, root_policy), tuple(objects.values()), states, schemas,
            (tuple(parser), tuple(parser_containers)),
            (inputs, input_fields, tuple(containers)))


def _same_authority(analysis, saved):
    source, anchors, objects, states, schemas, (parser, parser_containers), inputs_state = saved
    session, session_fields, projection, source_session, root_policy = anchors
    inputs, input_fields, containers = inputs_state
    return (same_intrinsic_source_attachments(analysis, source)
            and getattr(analysis, "_source_session", None) is session
            and source_session._ROOTS is root_policy
            and (session is None or _same(session, session_fields))
            and (projection is None or analysis.routines is projection[0]
                 and projection[0].base is projection[1] and projection[0]._overrides is projection[2]
                 and len(projection[2]) == len(projection[3])
                 and all(projection[2].get(name) is value for name, value in projection[3]))
            and all(len(vars(node)) == len(attrs)
                    and all(vars(node).get(name, _ABSENT) is value for name, value in attrs)
                    and getattr(node, "item", None) is item
                    and (item is None or len(vars(item)) == len(item_attrs)
                         and all(vars(item).get(name, _ABSENT) is value for name, value in item_attrs))
                    for node, attrs, item, item_attrs in parser)
            and all(len(value) == len(items) and all(left is right for left, right in zip(value, items, strict=True))
                    for value, items in parser_containers)
            and analysis.inputs is inputs and all(getattr(inputs, name) is value for name, value in input_fields)
            and all(len(value) == len(items) and (
                all(value.get(key, _ABSENT) is item for key, item in items) if kind == "mapping" else
                all(any(item is current for current in value) for item in items) if kind == "set" else
                all(left is right for left, right in zip(value, items, strict=True)))
                for value, kind, items in containers)
            and all(_same(value, attached) for value, attached in objects)
            and all(same_intrinsic_scope_attachments(state)
                    and getattr(state[0], "qualified", _ABSENT) is qualified
                    and all(_same(binding, attached)
                            and getattr(binding, "component_object", _ABSENT) is root
                            and getattr(binding, "associate_selector", _ABSENT) is selector
                            for binding, attached, root, selector in bindings)
                    and getattr(state[0], "component_types", _ABSENT) is mapping
                    and (mapping is _ABSENT or len(mapping) == len(items)
                         and all(mapping.get(name) is schema for name, schema in items))
                    for state, qualified, bindings, mapping, items in states)
            and all(_same(schema, attached) and schema.members is members and len(members) == len(items)
                    and all(members.get(name) is item for name, item in items)
                    for schema, attached, members, items in schemas))


def allocation_source_checkpoint(analysis):
    """Pure participating-source receipt for a synchronous proof composition."""
    if getattr(analysis, "_source_session", None) is not None:
        return analysis.source_authority_context()
    return _raw_authority(analysis)


def require_allocation_source_checkpoint(analysis, saved):
    """Reject later proof callbacks changing an earlier proof's source."""
    from compiler.frontend.authority_context import SourceAuthorityContext
    if type(saved) is SourceAuthorityContext:
        saved.require(analysis)
    elif not _same_authority(analysis, saved):
        raise CompilationError("local allocation participating source changed at publication")


def _joined_receipts(analysis, routine, joined_storage):
    if type(joined_storage) is not tuple:
        raise CompilationError("local allocation joined storage requires an exact tuple of issued teams")
    owner = getattr(analysis, "_source_context_owner", analysis)
    registry = getattr(owner, "_descriptor_team_proofs", None)
    result, seen = [], set()
    for proof in joined_storage:
        if (type(proof) is not DescriptorTeamProof or proof._analysis is not owner
                or proof.procedure != routine.qualified or id(proof) in seen
                or not isinstance(registry, dict) or registry.get(proof.identity) is not proof
                or type(proof._record) is not str
                or sha256(proof._record.encode()).hexdigest() != proof.identity):
            raise CompilationError("local allocation requires exact same-analysis original joined storage")
        seen.add(id(proof))
        result.append((proof, _attached(proof), registry, proof.identity))
    return tuple(result)


def _same_joined(analysis, saved):
    analysis = getattr(analysis, "_source_context_owner", analysis)
    return all(_same(proof, attached) and getattr(analysis, "_descriptor_team_proofs", None) is registry
               and registry.get(identity) is proof for proof, attached, registry, identity in saved)


def _contained_nonuse(analysis, routine, binding, original, scope_authority):
    """Unused original contained code cannot observe this exact host allocation."""
    parts = [node for node, _parents in original if _kind(node) == "Internal_Subprogram_Part"]
    if not parts:
        return ()
    nested = {id(item.scope.node): item for item in analysis.numerical_helpers.values()
              if item is not routine}
    records = {}
    for part in parts:
        for node, parents in _walk(part):
            owner = nested.get(id(node))
            if owner is not None:
                current, seen = owner.scope.parent, set()
                while current is not routine.scope:
                    if current is None or id(current) in seen:
                        raise CompilationError("local allocation contained source owner is uncertain")
                    seen.add(id(current))
                    current = current.parent
                if _original_procedure(analysis, owner.qualified) is not owner:
                    raise CompilationError("local allocation contained source owner changed")
                scope_authority(owner.scope)
                records[owner.qualified] = {"procedure": owner.qualified,
                    "original_procedure_sha256": sha256(str(owner.scope.node).encode()).hexdigest()}
            if _kind(node) != "Name" or str(node).lower() != binding.name:
                continue
            containing = next((nested[id(parent)] for parent in reversed(parents) if id(parent) in nested), None)
            if containing is None:
                raise CompilationError("local allocation contained declaration or export is uncertain")
            # Header/specification names are not executable index entries.
            # Their authentic lexical fallback is the original contained
            # procedure, rather than the allocation's host procedure.
            scope = analysis.source_scope_for(node, containing.scope)
            _validate_resolution(analysis, scope, (node,), _scope_validator=scope_authority)
            if analysis._binding(scope, node) is binding:
                raise CompilationError("local allocation cannot escape through contained host association")
    if not records:
        raise CompilationError("local allocation contained source scopes are unavailable")
    return tuple(records.values())


def _walk(node):
    """Bound source syntax work, retaining lexical parents for direct uses."""
    pending, count = [(node, ())], 0
    while pending:
        current, parents = pending.pop()
        count += 1
        if count > _SCAN_LIMIT:
            raise CompilationError("local allocation original-source scan budget exhausted")
        if isinstance(current, (tuple, list)):
            pending.extend((child, parents) for child in reversed(current))
        elif isinstance(current, Base):
            yield current, parents
            children = getattr(current, "content", getattr(current, "items", ())) or ()
            pending.extend((child, (*parents, current)) for child in reversed(children))


def _derive(analysis, routine, binding, joined_storage=(), *, _return_fence=False):
    context = (analysis._source_authority_context
               if getattr(analysis, "_source_authority_reader", False) else None)
    raw = None if context is not None else _raw_authority(analysis)
    joined = (_imported_joined_receipts(analysis, routine, joined_storage)
              if context is not None else _joined_receipts(analysis, routine, joined_storage))

    def checkpoint():
        if context is not None:
            context.require(analysis)
            context.session.verify(analysis)
        if (context is None and not _same_authority(analysis, raw)
                or not (_same_imported_joined(analysis, joined) if context is not None
                        else _same_joined(analysis, joined))):
            raise CompilationError("local allocation consumed original authority changed at publication")

    if (getattr(analysis, "_descriptor_source", None) is not None
            or analysis._require_original(routine.qualified) is not routine):
        raise CompilationError("local allocation requires its unchanged original declaring routine")
    roles, scopes = _scope_roles(analysis), {}

    def scope_authority(scope):
        _validate_scope(analysis, scope)
        scopes[id(scope)] = scope

    scope_authority(routine.scope)
    if (routine.scope.bindings.get(binding.name) is not binding or binding.declaring_scope is not routine.scope
            or binding.name in routine.arguments or not {"save", "allocatable"} <= binding.attributes
            or binding.attributes & _UNSAFE or binding.dtype not in {"real", "integer"}
            or binding.kind not in {4, 8} or not 1 <= binding.rank <= 4 or binding.intent is not None):
        raise CompilationError("local allocation requires an exact independent numeric local SAVE ALLOCATABLE declaration")
    original = tuple(_walk(routine.scope.node))
    contained = _contained_nonuse(analysis, routine, binding, original, scope_authority)
    specification = next((node for node, _parents in original if _kind(node) == "Specification_Part"), None)
    forbidden = {"Namelist_Stmt", "Common_Stmt", "Equivalence_Stmt", "Cray_Pointer_Stmt", "Codimension_Stmt",
                 "Codimension_Attr_Spec"}
    if any(_kind(node) in forbidden for node, _parents in _walk(specification)):
        raise CompilationError("local allocation requires independent storage without association, NAMELIST or coarrays")
    declarations = [node for node, _parents in _walk(specification)
                    if _kind(node) == "Entity_Decl" and str(node.items[0]).lower() == binding.name]
    if (len(declarations) != 1 or declarations[0].items[2] is not None or declarations[0].items[3] is not None
            or any(_kind(axis) not in {"Deferred_Shape_Spec", "Assumed_Shape_Spec"}
                   or any(bound is not None for bound in axis.items) for axis in binding.shape_nodes)):
        raise CompilationError("local allocation requires its original uninitialized deferred-shape declaration")
    # A contained procedure with proved nonuse cannot observe this allocation.
    # Every directive in the declaring procedure still needs its own genuine
    # complete joined-team receipt; PRIVATE is never a completion premise.
    own = tuple((node, parents) for node, parents in original
                if not any(_kind(parent) == "Internal_Subprogram_Part" for parent in parents))
    directives = tuple(node for node, _parents in own
                       if _kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp"))
    covered = {}
    for proof, _attached_proof, _registry, _identity in joined:
        proof.validate(getattr(analysis, "_source_context_owner", analysis),
                       routine.qualified, proof.selected_node_ids)
        checkpoint()
        for root in proof._originals:
            for node, _parents in _walk(root):
                if _kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp"):
                    if id(node) in covered or not any(node is original for original in directives):
                        raise CompilationError("local allocation joined storage has overlapping or foreign directives")
                    covered[id(node)] = node
    if len(covered) != len(directives):
        raise CompilationError("local allocation OpenMP storage/concurrency requires separate authority")

    syntax = tuple(_walk(routine.execution))
    uses, statements, calls = [], {}, {}
    for node, parents in syntax:
        if _kind(node) != "Name" or str(node).lower() != binding.name:
            continue
        scope = analysis.source_scope_for(node, routine.scope)
        _validate_resolution(analysis, scope, (node,), _scope_validator=scope_authority)
        if analysis._binding(scope, node) is not binding:
            continue
        uses.append(node)
        statement = next((parent for parent in reversed(parents) if _kind(parent).endswith("_Stmt")), None)
        if statement is None or _kind(statement) not in _STATEMENTS:
            raise CompilationError("local allocation has an unproved observer, association or address escape")
        if any(_kind(parent) in {"Associate_Construct", "Select_Type_Construct", "Select_Rank_Construct"}
               for parent in parents):
            raise CompilationError("local allocation alias constructs require separate source authority")
        statements[id(statement)] = statement
        for expression in parents:
            if _kind(expression) not in _FUNCTIONS:
                continue
            name = expression.items[0]
            expression_scope = analysis.source_scope_for(expression, routine.scope)
            _validate_resolution(analysis, expression_scope, (name,), _scope_validator=scope_authority)
            callee_binding = analysis._binding(expression_scope, name)
            if _kind(expression) == "Part_Ref" and callee_binding is binding:
                continue
            if (_kind(expression) != "Intrinsic_Function_Reference" or callee_binding is not None
                    or str(name).lower() not in _INTRINSICS or analysis._unknown_exports(expression_scope)
                    or analysis._candidates(expression_scope, name)):
                raise CompilationError("local allocation source function/address escape requires separate authority")
        if _kind(statement) == "Call_Stmt":
            calls[id(statement)] = statement
    if len(statements) > analysis.operation_limit:
        raise CompilationError("local allocation source-use budget exhausted")
    associations = []
    for call in calls.values():
        scope = analysis.source_scope_for(call, routine.scope)
        resolved = resolve_original_call(analysis, scope, call)
        matches = []
        for association in resolved.associations:
            if association.actual is None:
                continue
            names = [node for node, _parents in _walk(association.actual) if any(node is use for use in uses)]
            if not names:
                continue
            formal = association.formal_binding
            if (formal.dtype not in {"real", "integer"} or formal.kind != binding.kind
                    or formal.attributes & (_UNSAFE | {"allocatable"}) or formal.dtype != binding.dtype):
                raise CompilationError("local allocation CALL requires an exact nonescaping numeric receiving formal")
            matches.append(association)
        if not matches or sum(association.binding is binding for association in matches) > 1:
            raise CompilationError("local allocation CALL requires distinct original storage associations")
        associations.append({"procedure": resolved.procedure, "original_call": str(call),
                             "arguments": [association.public() for association in matches]})

    # Do not publish a token after a resolution callback/source change. These
    # checks authenticate original source and namespaces, never mutable facts.
    _validate_scope_roles(analysis, roles, scopes)
    for scope in scopes.values():
        scope_authority(scope)
    if analysis._require_original(routine.qualified) is not routine:
        raise CompilationError("local allocation original routine authority changed")
    record = {"version": 1, "procedure": routine.qualified, "resource": binding.root,
            "source_inputs": analysis.inputs.identity(),
            "original_procedure_sha256": sha256(str(routine.scope.node).encode()).hexdigest(),
            "original_binding": binding.public(), "direct_calls": associations,
            "declaration": "original independent local numeric SAVE ALLOCATABLE",
            "direct_uses": "no contained host association, direct address export or escape-capable receiving formal",
            "selected_call_closure_requires_independent_descriptor_proof": True,
            "requires_original_allocation_presence_and_alias_guards": True,
            "requires_original_runtime_bounds": True,
            "payload_definition_authority": False, "allocation_presence_authority": False,
            "selected_descriptor_stability_authority": False, "capture_authorized": False,
            "execution_authorized": False, "cross_frontier_retention_authorized": False}
    if contained or directives:
        record.update(version=2, contained_nonuse=list(contained),
                      joined_storage=[proof.public() for proof, *_receipt in joined],
                      direct_uses="exact contained nonuse and original joined coverage; no direct address export or escape-capable receiving formal",
                      parent_join_is_not_execution_authority=True)
    analysis.verify_publication()
    analysis.verify_source()
    analysis.inputs.verify_live()
    checkpoint()
    # Only the synchronous issuer consumes this fresh derivation fence. It
    # never enters a proof, analysis registry or persistent validated cache.
    return (record, checkpoint) if _return_fence else record


@dataclass(frozen=True, slots=True)
class OriginalLocalAllocationProof:
    identity: str
    binding: object
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _roles: object = field(repr=False, compare=False)
    _limits: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)
    _joined_storage: tuple = field(default=(), repr=False, compare=False)
    _joined_authority: tuple = field(default=(), repr=False, compare=False)
    _context: object = field(default=None, repr=False, compare=False)
    _arena: object = field(default=None, repr=False, compare=False)
    _derived: object = field(default=None, repr=False, compare=False)

    def checkpoint(self):
        """Stack-local exact attachments; no resolution or serialization."""
        if self._context is not None:
            self.validate(self._analysis, self._routine, self.binding)
            return self._context, self._arena, self._derived
        registry = getattr(self._analysis, "_local_allocation_proofs", None)
        if not isinstance(registry, dict) or registry.get(self.identity) is not self:
            raise CompilationError("local allocation requires its exact compiler-issued original proof")
        return (_attached(self), registry, self.identity, _raw_authority(self._analysis),
                self._joined_authority)

    def require_checkpoint(self, saved):
        if self._context is not None:
            if saved != (self._context, self._arena, self._derived):
                raise CompilationError("local allocation arena checkpoint changed")
            self.validate(self._analysis, self._routine, self.binding)
            return
        attached, registry, identity, raw, joined = saved
        if (not _same(self, attached) or getattr(self._analysis, "_local_allocation_proofs", None) is not registry
                or registry.get(identity) is not self or not _same_authority(self._analysis, raw)
                or not _same_joined(self._analysis, joined)):
            raise CompilationError("local allocation retained original checkpoint changed at publication")

    def validate(self, analysis, routine, binding):
        if self._context is not None:
            self._context.require(analysis)
            arena = self._context.derivation_arena(analysis)
            if (self._issuer is not _ISSUER or self._analysis is not analysis
                    or self._routine is not routine or self.binding is not binding
                    or not arena.accepts(self._arena)
                    or getattr(analysis, "_local_allocation_proofs", {}).get(self.identity) is not self):
                raise CompilationError("local allocation requires its exact compiler-issued arena proof")
            arena.require(self._context, self._derived)
            arena.require_handle(("original_local_allocation", id(analysis), self._derived.query), self)
            if self._record is not self._derived.record or self.identity != self._derived.identity:
                raise CompilationError("local allocation immutable arena record changed")
            return self
        if getattr(analysis, "_source_session", None) is not None:
            raise CompilationError("mutable diagnostic proof cannot enter a sealed authority context")
        attached = _attached(self)
        registry = getattr(analysis, "_local_allocation_proofs", None)
        if (self._issuer is not _ISSUER or analysis is not self._analysis or routine is not self._routine
                or binding is not self.binding
                or not isinstance(registry, dict) or registry.get(self.identity) is not self
                or type(self._record) is not str or sha256(self._record.encode()).hexdigest() != self.identity
                or not _same_joined(analysis, self._joined_authority)):
            raise CompilationError("local allocation requires its exact compiler-issued original proof")
        if self._limits != (analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit):
            raise CompilationError("local allocation original inspection policy changed")
        consulted = {id(scope) for scope in self._roles.values()} if isinstance(self._roles, dict) else set()
        _validate_scope_roles(analysis, self._roles, consulted)
        if _encoded(_derive(analysis, routine, binding, self._joined_storage)) != self._record:
            raise CompilationError("local allocation source, binding or direct-use authority changed")
        if (not _same(self, attached) or getattr(analysis, "_local_allocation_proofs", None) is not registry
                or registry.get(self.identity) is not self or not _same_joined(analysis, self._joined_authority)):
            raise CompilationError("local allocation retained issuer attachment changed at publication")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def _role_receipt(roles):
    if type(roles) not in {dict, _SessionScopeRoles}:
        raise CompilationError("local allocation retained source roles are unavailable")
    return tuple(dict.items(roles)) if type(roles) is dict else _attached(roles)


def _same_roles(roles, saved):
    if type(roles) not in {dict, _SessionScopeRoles}:
        return False
    return (len(roles) == len(saved)
            and all(dict.get(roles, name, _ABSENT) is value for name, value in saved)
            if type(roles) is dict else _same(roles, saved))


def _issuer_receipts(registry):
    if registry is None:
        return ()
    if type(registry) is not dict:
        raise CompilationError("local allocation requires its compiler-owned plain proof registry")
    result = []
    for key, proof in dict.items(registry):
        if type(proof) is not OriginalLocalAllocationProof:
            raise CompilationError("local allocation issuer registry contains foreign authority")
        roles = proof._roles
        roles_state = _role_receipt(roles)
        result.append((key, proof, _attached(proof), roles, roles_state))
    return tuple(result)


def _same_issuer_receipts(analysis, registry, saved):
    if getattr(analysis, "_local_allocation_proofs", None) is not registry:
        return False
    if registry is None:
        return not saved
    if type(registry) is not dict:
        return False
    return (len(registry) == len(saved)
            and all(dict.get(registry, key) is proof and _same(proof, attached)
                    and proof._roles is roles
                    and _same_roles(roles, state)
                    for key, proof, attached, roles, state in saved))


def original_local_allocation(analysis, routine, binding, *, joined_storage=()):
    """Derive once under a closed context; mutable diagnostics remain fresh."""
    if getattr(analysis, "_source_session", None) is not None:
        return _arena_local_allocation(analysis, routine, binding, joined_storage)
    policy = tuple((name, getattr(analysis, name)) for name in
                   ("depth_limit", "procedure_limit", "operation_limit"))
    descriptor_source = getattr(analysis, "_descriptor_source", _ABSENT)
    joined = _joined_receipts(analysis, routine, joined_storage)
    registry = getattr(analysis, "_local_allocation_proofs", None)
    existing = _issuer_receipts(registry)
    fresh, source_fence = _derive(analysis, routine, binding, joined_storage, _return_fence=True)
    record = _encoded(fresh)

    def publication_fence():
        if (not _same_joined(analysis, joined)
                or not _same_issuer_receipts(analysis, registry, existing)):
            raise CompilationError("local allocation issuer registry changed during source inspection")
        if (any(getattr(analysis, name) is not value for name, value in policy)
                or getattr(analysis, "_descriptor_source", _ABSENT) is not descriptor_source):
            raise CompilationError("local allocation original inspection policy changed before publication")
        analysis.verify_source()
        analysis.inputs.verify_live()
        source_fence()
        # The source fence is callback-free. Keep exact retained proof and
        # registry attachments on both sides, including mutable role entries.
        if (not _same_issuer_receipts(analysis, registry, existing)
                or any(getattr(analysis, name) is not value for name, value in policy)
                or getattr(analysis, "_descriptor_source", _ABSENT) is not descriptor_source):
            raise CompilationError("local allocation issuer registry or policy changed before publication")

    identity = sha256(record.encode()).hexdigest()
    if registry is not None and identity in registry:
        proof = registry[identity]
        if (proof._issuer is not _ISSUER or proof._analysis is not analysis
                or proof._routine is not routine or proof.binding is not binding
                or proof.identity != identity or proof._record != record
                or proof._limits != (analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit)
                or len(proof._joined_storage) != len(joined_storage)
                or any(left is not right for left, right in zip(proof._joined_storage, joined_storage, strict=True))
                or not _same_joined(analysis, proof._joined_authority)):
            raise CompilationError("local allocation cached proof belongs to another original query")
        consulted = {id(scope) for scope in proof._roles.values()} if isinstance(proof._roles, dict) else set()
        _validate_scope_roles(analysis, proof._roles, consulted)
        publication_fence()
        return proof
    roles = _scope_roles(analysis)
    roles_state = _role_receipt(roles)
    consulted = {id(scope) for scope in roles.values()} if isinstance(roles, dict) else set()
    _validate_scope_roles(analysis, roles, consulted)
    proof = OriginalLocalAllocationProof(identity, binding, analysis, routine,
        roles, tuple(value for _name, value in policy), record, _ISSUER, joined_storage, joined)
    attached = _attached(proof)
    publication_fence()
    if not _same(proof, attached) or not _same_roles(roles, roles_state):
        raise CompilationError("local allocation new proof or source roles changed before publication")
    # No delegated callbacks occur after the final source/registry fence.
    if registry is None:
        registry = analysis._local_allocation_proofs = {}
    registry[identity] = proof
    return proof


def _arena_local_allocation(analysis, routine, binding, joined_storage):
    context = analysis.source_authority_context()
    context.require(analysis)
    if (context.session.routine(analysis, routine.qualified) is not routine
            or routine.scope.bindings.get(binding.name) is not binding
            or not context.session.contains(binding)):
        raise CompilationError("local allocation requires exact original arena query objects")
    if getattr(analysis, "_descriptor_source", None) is not None:
        raise CompilationError("local allocation requires its unchanged original declaring routine")
    joined = _joined_receipts(analysis, routine, joined_storage)
    arena = context.derivation_arena(analysis)
    capabilities = []
    for proof, _attached_proof, _registry, _identity in joined:
        key = proof.procedure, tuple(proof.selected_node_ids), proof.identity
        imported = arena.derive(_LOCAL_JOINED_IMPORT_FAMILY, key, (routine, proof), analysis)
        capabilities.append(_ImportedJoinedStorage(context, imported))
    capabilities = tuple(capabilities)
    query = (routine.qualified, binding.root, tuple(proof.identity for proof in joined_storage))
    arena = context.derivation_arena(analysis)

    derived = arena.derive(_LOCAL_ALLOCATION_FAMILY, query, (routine, binding, capabilities), analysis,
                           dependencies=tuple(capability._derived for capability in capabilities))
    context.require(analysis)
    if getattr(analysis, "_local_allocation_context", None) is not context:
        analysis._local_allocation_proofs = {}
        analysis._local_allocation_context = context
    registry = analysis._local_allocation_proofs
    proof = registry.get(derived.identity)
    if proof is not None:
        return proof.validate(analysis, routine, binding)
    proof = OriginalLocalAllocationProof(derived.identity, binding, analysis, routine, _scope_roles(analysis),
        (analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit), derived.record, _ISSUER,
        capabilities, (), context, arena, derived)
    arena.register_handle(("original_local_allocation", id(analysis), query), proof)
    registry[proof.identity] = proof
    return proof


def _allocation_arena_derivation(reader, routine, binding, joined_storage):
    return _encoded(_derive(reader, routine, binding, joined_storage))


_LOCAL_ALLOCATION_FAMILY = _register_family("original_local_allocation", 1, _allocation_arena_derivation)


@dataclass(frozen=True, slots=True)
class _ImportedJoinedStorage:
    _context: object
    _derived: object

    @property
    def procedure(self):
        return self._derived.query[0]

    @property
    def selected_node_ids(self):
        return self._derived.query[1]

    @property
    def identity(self):
        return self._derived.query[2]

    @property
    def _originals(self):
        return tuple(self._context.session.original_node(node) for node in self._derived.source_nodes)

    def validate(self, analysis, procedure, selected):
        self._context.require(analysis)
        self._context.arena.require(self._context, self._derived)
        if (self._derived.family is not _LOCAL_JOINED_IMPORT_FAMILY or procedure != self.procedure
                or tuple(selected) != self.selected_node_ids):
            raise CompilationError("local allocation joined dependency belongs to another query")
        return self

    def public(self):
        return json.loads(self._derived.record)


def _imported_joined_receipts(analysis, routine, capabilities):
    if type(capabilities) is not tuple:
        raise CompilationError("local allocation requires its immutable joined prerequisites")
    result = []
    for capability in capabilities:
        if type(capability) is not _ImportedJoinedStorage:
            raise CompilationError("local allocation requires an imported joined storage capability")
        capability.validate(analysis, routine.qualified, capability.selected_node_ids)
        result.append((capability, None, None, capability.identity))
    return tuple(result)


def _same_imported_joined(analysis, saved):
    for capability, *_rest in saved:
        capability.validate(analysis, capability.procedure, capability.selected_node_ids)
    return True


def _import_local_joined_dependency(analysis, routine, proof):
    # The only transitional owner-reading family: fully validate this exact
    # existing joined-team prerequisite, then retain immutable data/node IDs.
    context = analysis.source_authority_context()
    receipts = _joined_receipts(analysis, routine, (proof,))
    proof.validate(analysis, routine.qualified, proof.selected_node_ids)
    record = _encoded(proof.public())
    nodes = tuple(id(node) for node in proof._originals)
    for identity in nodes:
        context.session.original_node(identity)
    context.require(analysis)
    if not _same_joined(analysis, receipts):
        raise CompilationError("local allocation joined prerequisite changed during import")
    return SourcePayload(record, nodes)


_LOCAL_JOINED_IMPORT_FAMILY = _register_family(
    "local_allocation_joined_storage", 1, _import_local_joined_dependency, legacy_import=True)
