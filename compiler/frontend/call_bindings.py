"""Source-backed call bindings without evaluating actuals or rebasing sections.

Resolved objects borrow the original syntax during analysis. Public records are
plain values suitable for summary caches; they never contain parser objects or
claim physical transfer coordinates without a runtime descriptor proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from compiler.ir import CompilationError, SourceLocation


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()




@dataclass(frozen=True)
class SourceDependency:
    resource: str
    kind: str
    binding: object = field(compare=False, repr=False)

    def public(self):
        return {"resource": self.resource, "kind": self.kind}












@dataclass(frozen=True)
class SourceCallArgument:
    position: int
    formal: str
    keyword: str | None
    actual: object = field(compare=False, repr=False)

    def public(self):
        return {"position": self.position, "formal": self.formal, "keyword": self.keyword,
                "actual": str(self.actual)}














def _arrange(callee, original, *, function=False):
    values, arguments, keyword_seen, position = {}, [], False, 0
    for original_position, item in enumerate(original):
        keyword = str(item.items[0]) if _kind(item) in ({"Actual_Arg_Spec", "Component_Spec"}
                                                       if function else {"Actual_Arg_Spec"}) else None
        actual = item.items[1] if keyword is not None else item
        if keyword is not None:
            keyword_seen = True
            formal = keyword.lower()
            if formal not in callee.arguments:
                raise CompilationError("unknown source-call keyword: " + keyword)
        else:
            if keyword_seen:
                raise CompilationError("positional source-call argument follows a keyword")
            if position >= len(callee.arguments):
                raise CompilationError("too many source-call arguments")
            formal = callee.arguments[position]
            position += 1
        if formal in values:
            raise CompilationError("duplicate source-call argument: " + formal)
        values[formal] = actual
        arguments.append(SourceCallArgument(original_position, formal, keyword, actual))
    for formal in callee.arguments:
        binding = callee.scope.bindings.get(formal)
        if binding is None:
            raise CompilationError("source-call formal declaration is unavailable: " + formal)
        if formal not in values and "optional" not in binding.attributes:
            raise CompilationError("required source-call argument is missing: " + formal)
    return tuple(values.get(formal) for formal in callee.arguments), tuple(arguments)














def _match_signature(declaration, signature):
    if (signature is None or None in (declaration.kind, signature[1])
            or declaration.signature() != signature):
        raise CompilationError("source-call type, kind or rank mismatch: " + declaration.name)


def _resolve_candidates(analysis, scope, call_node, associate, *, function=False):
    target, arguments = call_node.items
    targets = {"Name", "Type_Name"} if function else {"Name"}
    if function and _kind(call_node) == "Intrinsic_Function_Reference":
        targets.add("Intrinsic_Name")
    if _kind(target) not in targets:
        raise CompilationError("indirect calls are scope boundaries")
    original = tuple(_children(arguments))
    matches, errors = [], []
    candidates = analysis._candidates(scope, target)
    pool = ({name: routine for name, routine in analysis.numerical_helpers.items()
             if routine.source_kind in {"function", "internal_function"}} if function else analysis.routines)
    unavailable = [procedure for procedure in candidates if procedure not in pool]
    if unavailable:
        raise CompilationError("source call interface closure unavailable: " + ", ".join(unavailable))
    for procedure in candidates:
        callee = pool[procedure]
        try:
            if callee.source_kind == "external":
                interface = analysis.external_interface(callee)
                if not interface["available"]:
                    raise CompilationError(interface["reason"] + ": " + procedure)
                if any(_kind(item) == "Actual_Arg_Spec" for item in original):
                    raise CompilationError("external keyword arguments require a proven explicit interface: " + procedure)
            actuals, original_arguments = _arrange(callee, original, function=function)
            associations = associate(callee, actuals)
            matches.append((callee, actuals, associations, original_arguments))
        except OriginalCallBudgetExceeded:
            raise
        except CompilationError as error:
            errors.append(error)
    if len(matches) != 1:
        if not matches and errors and len({str(error) for error in errors}) == 1:
            raise errors[0]
        raise CompilationError("source call is unresolved or ambiguous: " + str(target))
    return matches[0]





# This association layer retains original source semantics only. In particular,
# its storage reference is not a rectangle, a capture, or a no-alias proof.
class OriginalCallBudgetExceeded(CompilationError):
    """Original association inspection exceeded a bounded source scan."""


_ORIGINAL_CALL_ISSUER = object()


@dataclass(frozen=True)
class OriginalActualType:
    dtype: str
    kind: int | None
    rank: int
    binding: object = field(default=None, compare=False, repr=False)
    schema: object = field(default=None, compare=False, repr=False)

    def public(self):
        return {"type": self.dtype, "kind": self.kind, "rank": self.rank,
                "storage_resource": self.binding.root if self.binding is not None else None,
                "declaring_type": ({"module": self.schema.scope.module, "name": self.schema.name}
                                   if self.schema is not None else None)}


@dataclass(frozen=True)
class OriginalArgumentAssociation:
    formal: str
    actual: object = field(compare=False, repr=False)
    formal_binding: object = field(compare=False, repr=False)
    actual_type: OriginalActualType | None

    @property
    def binding(self):
        return self.actual_type.binding if self.actual_type is not None else None

    def public(self):
        return {"formal": self.formal, "actual": str(self.actual) if self.actual is not None else None,
                "formal_descriptor": self.formal_binding.public(),
                "actual_type": self.actual_type.public() if self.actual_type is not None else None,
                "entry_definition_kill": self.formal_binding.intent == "out"}


@dataclass(frozen=True)
class ResolvedOriginalCall:
    procedure: str
    formals: tuple[str, ...]
    actuals: tuple = field(compare=False, repr=False)
    associations: tuple[OriginalArgumentAssociation, ...]
    original_arguments: tuple[SourceCallArgument, ...]
    target: str
    source_kind: str
    callee: object = field(compare=False, repr=False)
    _caller: object = field(compare=False, repr=False)
    _analysis: object = field(compare=False, repr=False)
    _scope: object = field(compare=False, repr=False)
    _original: object = field(compare=False, repr=False)
    _scopes: tuple = field(compare=False, repr=False)
    _roles: object = field(compare=False, repr=False)
    _schemas: tuple = field(compare=False, repr=False)
    _limits: tuple = field(compare=False, repr=False)
    _issuer: object = field(compare=False, repr=False)

    def public(self):
        return {"schema_version": 1, "procedure": self.procedure, "target": self.target,
                "source_kind": self.source_kind, "formal_arguments": list(self.formals),
                "actual_arguments": [str(actual) if actual is not None else None for actual in self.actuals],
                "original_arguments": [item.public() for item in self.original_arguments],
                "associations": [item.public() for item in self.associations],
                "actual_evaluation": "unchanged original call position and guards",
                "effect_authority": False, "capture_authority": False, "alias_authority": False,
                "completion_authority": False, "execution_authority": False}

    def validate(self, analysis, scope, original):
        from compiler.frontend.source_functions import _original, _validate_scope, _validate_scope_roles
        from compiler.frontend.source_objects import _members, _TypeBuilder

        if (self._issuer is not _ORIGINAL_CALL_ISSUER or analysis is not self._analysis
                or scope is not self._scope or original is not self._original
                or getattr(analysis, "_original_call_associations", {}).get((id(analysis), id(original))) is not self):
            raise CompilationError("original call requires exact issued source association")
        if self._limits != (analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit):
            raise CompilationError("original call association inspection budgets changed")
        analysis.verify_source()
        if (_original(analysis, self._caller.qualified) is not self._caller
                or _original(analysis, self.procedure) is not self.callee):
            raise CompilationError("original call procedure authority changed")
        for retained in self._scopes:
            _validate_scope(analysis, retained)
        _validate_scope_roles(analysis, self._roles, {id(scope) for scope in self._scopes})
        builder = _TypeBuilder(analysis)
        for schema in self._schemas:
            record = builder.scope(schema.scope)[5].get(schema.name)
            if (record is None or record[0] is not schema or record[1] is not schema.node
                    or record[2] != str(schema.node) or record[3] != _members(schema)):
                raise CompilationError("original call declaring type authority changed")
        return self


class _OriginalTypes:
    """Bounded declaration/type inspection; no layout, payload or index evaluation."""
    def __init__(self, analysis):
        from compiler.frontend.source_objects import _TypeBuilder

        self.analysis, self.scopes, self.schemas = analysis, {}, {}
        self.visits = 0
        reader = self

        class Types(_TypeBuilder):
            def scope(self, scope):
                reader.scope(scope)
                return super().scope(scope)

        self.types = Types(analysis)

    def scope(self, scope):
        from compiler.frontend.source_functions import _validate_scope

        if scope is not None and id(scope) not in self.scopes:
            _validate_scope(self.analysis, scope)
            self.scopes[id(scope)] = scope
        return scope

    def schema(self, scope, dtype):
        from compiler.frontend.component_bindings import _typename

        name = _typename(dtype)
        if name is None:
            if dtype not in {"real", "integer", "logical"}:
                raise CompilationError("original call requires a known nonpolymorphic declared type")
            return None
        schema = self.types.resolve(scope, name)
        if schema is None or schema.reason:
            raise CompilationError("original call declaring type is unavailable or unsupported")
        self.schemas[id(schema)] = schema
        return schema

    def declared(self, binding):
        self.scope(binding.declaring_scope)
        if binding.declaring_scope.bindings.get(binding.name) is not binding:
            raise CompilationError("original call requires an exact declared binding")
        return OriginalActualType(binding.dtype, binding.kind, binding.rank, binding,
                                  self.schema(binding.declaring_scope, binding.dtype))

    def indices(self, scope, part, rank, depth):
        if _kind(part) != "Part_Ref":
            return rank
        indices = tuple(_children(part.items[1]))
        if len(indices) != rank:
            raise CompilationError("original call array subscripts do not match declared rank")
        logical_rank = 0
        for index in indices:
            values = index.items if _kind(index) == "Subscript_Triplet" else (index,)
            logical_rank += _kind(index) == "Subscript_Triplet"
            for value in values:
                if value is not None:
                    typed = self.actual(scope, value, depth + 1)
                    if typed.dtype != "integer" or typed.rank:
                        raise CompilationError("original call requires scalar INTEGER section indices")
        return logical_rank

    def storage(self, scope, node, depth):
        parts = tuple(node.items) if _kind(node) == "Data_Ref" else (node,)
        base = parts[0].items[0] if _kind(parts[0]) == "Part_Ref" else parts[0]
        if _kind(base) != "Name":
            return None
        binding = self.analysis._binding(scope, base)
        if binding is None:
            return None
        typed = self.declared(binding)
        rank = self.indices(scope, parts[0], typed.rank, depth)
        dtype, width, schema = typed.dtype, typed.kind, typed.schema
        for part in parts[1:]:
            self.tick(depth)
            if schema is None:
                raise CompilationError("original call component requires a declared source type")
            name = str(part.items[0] if _kind(part) == "Part_Ref" else part).lower()
            declaration = schema.members.get(name)
            if declaration is None:
                raise CompilationError("original call component declaration unavailable: " + name)
            spec, attributes, entity = declaration
            flags = {str(item).split("(")[0].lower() for item in attributes}
            if (schema.private or "private" in flags) and scope.module != schema.scope.module:
                raise CompilationError("original call component is inaccessible: " + name)
            dimension = next((item.items[1] for item in attributes
                              if _kind(item) == "Dimension_Component_Attr_Spec"), None)
            shape = entity.items[1] if entity.items[1] is not None else dimension
            field_rank = len(_children(shape))
            if rank and field_rank:
                raise CompilationError("original call requires at most one array part in a component reference")
            if _kind(spec) == "Intrinsic_Type_Spec":
                base_type, selector = spec.items
                dtype = str(base_type).lower()
                width = 8 if dtype == "double precision" else 4
                dtype = "real" if dtype == "double precision" else dtype
                if selector is not None:
                    width = schema.scope.kinds.integer(selector.items[1], SourceLocation(str(scope.path)))
            else:
                dtype, width = str(spec).lower(), None
            owner = schema.scope
            schema = self.schema(owner, dtype)
            rank += self.indices(scope, part, field_rank, depth)
        return OriginalActualType(dtype, width, rank, binding, schema)

    def tick(self, depth):
        self.visits += 1
        if depth > self.analysis.depth_limit or self.visits > self.analysis.operation_limit:
            raise OriginalCallBudgetExceeded("original call type inspection budget exhausted")

    def actual(self, scope, node, depth=0):
        self.tick(depth)
        self.scope(scope)
        if _kind(node) in {"Name", "Part_Ref", "Data_Ref"}:
            storage = self.storage(scope, node, depth)
            if storage is not None:
                return storage
        items = _children(node)
        if _kind(node) == "Array_Constructor" and _kind(node.items[1]) == "Ac_Value_List":
            # Type association only: retain the original vector expression and
            # its evaluation point. Implied DOs, explicit type specifications,
            # nested arrays and mixed kinds require separate expression proof.
            values = tuple(_children(node.items[1]))
            if values:
                typed = tuple(self.actual(scope, value, depth + 1) for value in values)
                first = typed[0]
                if (first.dtype in {"integer", "real", "logical"} and first.kind in {1, 4, 8}
                        and not first.rank
                        and all((value.dtype, value.kind, value.rank) == (first.dtype, first.kind, 0)
                                for value in typed)):
                    return OriginalActualType(first.dtype, first.kind, 1)
        if _kind(node) == "Parenthesis":
            typed = self.actual(scope, items[1], depth + 1)
            return OriginalActualType(typed.dtype, typed.kind, typed.rank, schema=typed.schema)
        if len(items) == 2 and str(items[0]) in {"+", "-"}:
            typed = self.actual(scope, items[1], depth + 1)
            if typed.dtype in {"integer", "real"} and typed.kind in {4, 8} and not typed.rank:
                return OriginalActualType(typed.dtype, typed.kind, 0)
        if len(items) == 3 and str(items[1]) in {"+", "-", "*", "/", "**"}:
            from compiler.frontend.source_scalar_types import scalar_numeric_result

            left, right = (self.actual(scope, child, depth + 1) for child in (items[0], items[2]))
            signature = scalar_numeric_result(str(items[1]), (left.dtype, left.kind, left.rank),
                                              (right.dtype, right.kind, right.rank))
            if signature is not None:
                return OriginalActualType(*signature)
        if _kind(node) in {"Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant"}:
            signature = self.analysis._signature(scope, node)
            if signature is not None and signature[1] is not None:
                return OriginalActualType(*signature)
        raise CompilationError("original call actual type requires separate source expression authority: " + str(node))

    def associate(self, callee, actuals):
        result = []
        self.scope(callee.scope)
        for formal, actual in zip(callee.arguments, actuals, strict=True):
            declaration = callee.scope.bindings[formal]
            expected = self.declared(declaration)
            typed = self.actual(self.caller_scope, actual) if actual is not None else None
            if typed is not None:
                if expected.schema is not None:
                    if expected.schema is not typed.schema or expected.rank != typed.rank:
                        raise CompilationError("source-call canonical declaring type or rank mismatch: " + formal)
                else:
                    _match_signature(declaration, (typed.dtype, typed.kind, typed.rank))
                if typed.binding is None and (declaration.intent in {"out", "inout"}
                                              or declaration.attributes & {"pointer", "allocatable"}):
                    raise CompilationError("source-call writable or descriptor actual requires original storage: " + formal)
            result.append(OriginalArgumentAssociation(formal, actual, declaration, typed))
        return tuple(result)


def _resolve_original_invocation(analysis, scope, original, *, function=False):
    from compiler.frontend.source_functions import _original, _scope_roles, _validate_resolution, _walk

    analysis.verify_source()
    allowed = {"Part_Ref", "Function_Reference", "Structure_Constructor", "Intrinsic_Function_Reference"} if function else {"Call_Stmt"}
    if _kind(original) not in allowed:
        raise CompilationError("original association requires the corresponding CALL or function syntax")
    existing = getattr(analysis, "_original_call_associations", {}).get((id(analysis), id(original)))
    if existing is not None:
        return existing.validate(analysis, scope, original)
    # Find the exact enclosing original routine, including lexical ASSOCIATE
    # scopes. No projected statement or equal-text clone can issue authority.
    pending, caller = scope, None
    while pending is not None:
        caller = next((routine for routine in analysis.numerical_helpers.values()
                       if routine.scope is pending), None)
        if caller is not None:
            break
        pending = pending.parent
    if (caller is None or _original(analysis, caller.qualified) is not caller
            or not any(node is original for node in _walk(caller.execution))
            or analysis.source_scope_for(original, caller.scope) is not scope):
        raise CompilationError("original call requires its exact original statement and lexical scope")
    roles = _scope_roles(analysis)
    if isinstance(roles, dict):
        roles = MappingProxyType(roles)
    reader = _OriginalTypes(analysis)
    reader.caller_scope = scope
    _validate_resolution(analysis, scope, _walk(original), tick=lambda _kind: reader.tick(0),
                         _scope_validator=reader.scope)
    callee, actuals, associations, arguments = _resolve_candidates(
        analysis, scope, original, reader.associate, function=function)
    if _original(analysis, callee.qualified) is not callee:
        raise CompilationError("original call requires its exact original callee")
    proof = ResolvedOriginalCall(callee.qualified, callee.arguments, actuals, associations, arguments,
                                str(original.items[0]), callee.source_kind, callee, caller, analysis, scope,
                                original, tuple(reader.scopes.values()), roles, tuple(reader.schemas.values()),
                                (analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit),
                                _ORIGINAL_CALL_ISSUER)
    registry = getattr(analysis, "_original_call_associations", None)
    if registry is None:
        registry = analysis._original_call_associations = {}
    registry[id(analysis), id(original)] = proof
    try:
        return proof.validate(analysis, scope, original)
    except BaseException:
        del registry[id(analysis), id(original)]
        raise


def resolve_original_call(analysis, scope, call_node):
    """Match an unchanged original CALL without granting physical mapping."""
    return _resolve_original_invocation(analysis, scope, call_node)


