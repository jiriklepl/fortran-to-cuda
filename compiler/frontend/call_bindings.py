"""Source-backed call bindings without evaluating actuals or rebasing sections.

Resolved objects borrow the original syntax during analysis. Public records are
plain values suitable for summary caches; they never contain parser objects or
claim physical transfer coordinates without a runtime descriptor proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compiler.ir import CompilationError, SourceLocation


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _arrange(callee, original):
    """The actuals of a CALL in the order of the callee's dummies, ``None`` for an absent optional one."""
    values, keyword_seen, position = {}, False, 0
    for item in original:
        keyword = str(item.items[0]) if _kind(item) == "Actual_Arg_Spec" else None
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
    for formal in callee.arguments:
        binding = callee.scope.bindings.get(formal)
        if binding is None:
            raise CompilationError("source-call formal declaration is unavailable: " + formal)
        if formal not in values and "optional" not in binding.attributes:
            raise CompilationError("required source-call argument is missing: " + formal)
    return tuple(values.get(formal) for formal in callee.arguments)


_SUPPORTED = frozenset({("integer", 4, 0), ("integer", 8, 0),
                        ("real", 4, 0), ("real", 8, 0)})


def scalar_numeric_result(operator, left, right):
    """Return the supported target ABI signature or decline the operation.

    Fortran 2023 10.1.5.1 and 10.1.9.3 determine type and kind independently
    of evaluation. In particular REAL ** INTEGER keeps the REAL kind, while
    two INTEGER kinds use the greater exponent range, including for **.
    This service never folds, rewrites or authorizes evaluating an expression.
    """
    if operator not in {"+", "-", "*", "/", "**"} or left not in _SUPPORTED or right not in _SUPPORTED:
        return None
    reals = [value[1] for value in (left, right) if value[0] == "real"]
    return ("real", max(reals), 0) if reals else ("integer", max(left[1], right[1]), 0)


def _match_signature(declaration, signature):
    if (signature is None or None in (declaration.kind, signature[1])
            or declaration.signature() != signature):
        raise CompilationError("source-call type, kind or rank mismatch: " + declaration.name)


def _resolve_candidates(analysis, scope, call_node, associate):
    """The one procedure a CALL can mean whose dummies its actuals associate with."""
    target, arguments = call_node.items
    if _kind(target) != "Name":
        raise CompilationError("indirect calls are scope boundaries")
    original = tuple(_children(arguments))
    matches, errors = [], []
    candidates = analysis._candidates(scope, target)
    unavailable = [procedure for procedure in candidates if procedure not in analysis.routines]
    if unavailable:
        raise CompilationError("source call interface closure unavailable: " + ", ".join(unavailable))
    for procedure in candidates:
        callee = analysis.routines[procedure]
        try:
            if callee.source_kind == "external":
                reason = analysis.external_interface(callee)
                if reason is not None:
                    raise CompilationError(reason + ": " + procedure)
                if any(_kind(item) == "Actual_Arg_Spec" for item in original):
                    raise CompilationError("external keyword arguments require a proven explicit interface: " + procedure)
            associate(callee, _arrange(callee, original))
            matches.append(callee)
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


@dataclass(frozen=True)
class OriginalActualType:
    dtype: str
    kind: int | None
    rank: int
    binding: object = field(default=None, compare=False, repr=False)
    schema: object = field(default=None, compare=False, repr=False)


class _OriginalTypes:
    """Bounded declaration/type inspection; no layout, payload or index evaluation."""
    def __init__(self, analysis, caller_scope):
        self.analysis, self.caller_scope = analysis, caller_scope
        self.visits = self.type_lookups = 0

    def resolve(self, scope, name, active=()):
        """The derived type that a name means in a scope, through USE and host association."""
        if scope is None or (id(scope), name) in active:
            return None
        self.type_lookups += 1
        if self.type_lookups > self.analysis.procedure_limit or len(active) >= self.analysis.depth_limit:
            raise CompilationError("source object type resolution budget exhausted")
        local = getattr(scope, "component_types", {}).get(name)
        if local is not None:
            return local
        sources = self.analysis.use_sources(scope, name)
        if sources is None:
            return None
        found = {}
        for module, remote in sources:
            owner = self.analysis.modules.get(module)
            if owner is None:
                return None
            if not self.analysis._exported(owner, remote):
                if scope.imports.get(name) == (module, remote):
                    return None
                continue
            result = self.resolve(owner, remote, (*active, (id(scope), name)))
            if result is not None:
                found[id(result.node)] = result
        if found:
            return next(iter(found.values())) if len(found) == 1 else None
        return self.resolve(scope.parent, name, (*active, (id(scope), name))) if scope.parent else None

    def schema(self, scope, dtype):
        from compiler.frontend.component_bindings import _typename

        name = _typename(dtype)
        if name is None:
            if dtype not in {"real", "integer", "logical"}:
                raise CompilationError("original call requires a known nonpolymorphic declared type")
            return None
        schema = self.resolve(scope, name)
        if schema is None or schema.reason:
            raise CompilationError("original call declaring type is unavailable or unsupported")
        return schema

    def declared(self, binding):
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
            left, right = (self.actual(scope, child, depth + 1) for child in (items[0], items[2]))
            signature = scalar_numeric_result(str(items[1]), (left.dtype, left.kind, left.rank),
                                              (right.dtype, right.kind, right.rank))
            if signature is not None:
                return OriginalActualType(*signature)
        if _kind(node) in {"Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant"}:
            signature = self.analysis._literal_signature(scope, node)
            if signature[1] is not None:
                return OriginalActualType(*signature)
        raise CompilationError("original call actual type requires separate source expression authority: " + str(node))

    def associate(self, callee, actuals):
        """Check that each actual associates with its dummy: type, kind, rank and, where written, storage."""
        for formal, actual in zip(callee.arguments, actuals, strict=True):
            declaration = callee.scope.bindings[formal]
            expected = self.declared(declaration)
            if actual is None:
                continue
            typed = self.actual(self.caller_scope, actual)
            if expected.schema is not None:
                if expected.schema is not typed.schema or expected.rank != typed.rank:
                    raise CompilationError("source-call canonical declaring type or rank mismatch: " + formal)
            else:
                _match_signature(declaration, (typed.dtype, typed.kind, typed.rank))
            if typed.binding is None and (declaration.intent in {"out", "inout"}
                                          or declaration.attributes & {"pointer", "allocatable"}):
                raise CompilationError("source-call writable or descriptor actual requires original storage: " + formal)


def resolve_original_call(analysis, scope, call_node):
    """The procedure, a ``Routine`` of the analysis, that a CALL statement of a procedure calls.

    ``scope`` is the scope the statement stands in. The answer is kept per statement.
    """
    if _kind(call_node) != "Call_Stmt":
        raise CompilationError("original association requires the corresponding CALL syntax")
    if scope is None or analysis.source_scope_for(call_node) is not scope:
        raise CompilationError("original call requires its exact original statement and lexical scope")
    callee = analysis.resolved_calls.get(id(call_node))
    if callee is None:
        callee = analysis.resolved_calls[id(call_node)] = _resolve_candidates(
            analysis, scope, call_node, _OriginalTypes(analysis, scope).associate)
    return callee
