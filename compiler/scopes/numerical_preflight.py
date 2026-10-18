"""Guard conditional numerical effects before acquiring execution authority."""

from compiler.frontend.native_sections import NativeBound
from compiler.ir import CompilationError
from compiler.ir.integers import INTEGER_MAX, INTEGER_MIN


def preparation_preflight(builder, call, parameters, *, read_requirements=None):
    """Use original descriptors/scalars, in source activation order.

    The caller has already checked allocation, contiguity and descriptor ranges.
    No numerical-unit registration, definition event or query may precede this
    code. An outer owner can already have bound independently proved original
    descriptors, without crediting this unit's conditional definitions. Failure
    only clears the flag; the owning lexical block publishes and closes before
    running the unchanged complete original group.
    """
    from fparser.two import Fortran2003 as F

    from compiler.scopes.segments import renamed

    if read_requirements is None:
        region = builder._consume_call_numerical_region(call)
        if region is None:
            raise CompilationError("numerical preflight requires registered exact physical effects")
        effect = region.effects
    else:
        from compiler.frontend.reached_read_requirements import ReachedReadRequirementsProof
        if type(read_requirements) is not ReachedReadRequirementsProof:
            raise CompilationError("numerical preflight requires registered original read authority")
        prepared = builder.inline_for(call.procedure).prepared_regions.get(call.procedure)
        if prepared is not read_requirements.prepared:
            raise CompilationError("numerical preflight read preparation differs from the original call")
        read_requirements.validate(builder.analysis, builder.entry.qualified, prepared._selection, prepared)
        effect = read_requirements
    original = builder.inline.original_selection(call.node)
    specification, lines = [], ["fort_numerical_guard = .true."]
    ordinal = 0

    def name(resource):
        return parameters.get(resource) or builder.visible(builder.entry, resource)

    def resolve(bound):
        nonlocal ordinal
        if not isinstance(bound, NativeBound):
            raise CompilationError("numerical preflight requires typed original bounds")
        if bound.kind == "literal":
            if type(bound.value) is not int or not INTEGER_MIN <= bound.value <= INTEGER_MAX:
                raise CompilationError("numerical preflight literal exceeds the original INTEGER ABI")
            return f"({bound.value}_c_int64_t)"
        if bound.kind == "scalar":
            expression = f"int({name(bound.resource)},c_int64_t)"
        elif bound.kind in {"lbound", "ubound", "size"}:
            if bound.dimension is None or not 1 <= bound.dimension <= 4:
                raise CompilationError("numerical preflight inquiry requires a constant original dimension")
            expression = f"{bound.kind}({name(bound.resource)},{bound.dimension},kind=c_int64_t)"
        elif bound.kind == "parenthesis" and len(bound.children) == 1:
            return resolve(bound.children[0])
        elif bound.kind == "unary" and len(bound.children) == 1 and bound.operator in {"+", "-"}:
            expression = bound.operator + "(" + resolve(bound.children[0]) + ")"
        elif bound.kind == "binary" and len(bound.children) == 2 and bound.operator in {"+", "-", "*"}:
            left, right = (resolve(child) for child in bound.children)
            expression = f"({left} {bound.operator} {right})"
        else:
            raise CompilationError("numerical preflight requires checked original affine bounds")
        ordinal += 1
        temporary = "fort_precondition_" + str(ordinal)
        specification.append(f"integer(c_int64_t) :: {temporary}")
        # Each operand fits default INTEGER before a widened operation uses it;
        # even multiplication therefore fits INTEGER64. Failed intermediates
        # never feed another expression or an original descriptor inquiry.
        lines.extend(["if (fort_numerical_guard) then", f"{temporary} = {expression}",
                      f"fort_numerical_guard = {temporary} >= ({INTEGER_MIN}_c_int64_t) .and. &",
                      f"{temporary} <= {INTEGER_MAX}_c_int64_t", "endif"])
        return temporary

    for guard in effect.preconditions:
        # Resolve each next activation only inside all earlier original domains.
        for domain in guard.activation_domains:
            lower, upper = resolve(domain.lower), resolve(domain.upper)
            lines.extend(["if (fort_numerical_guard) then", f"if ({lower} <= {upper}) then"])
        predicate = guard.render(resolve)
        lines.append("if (fort_numerical_guard) fort_numerical_guard = " + predicate)
        # An inactive outer domain selects the unchanged group before any
        # inner control or by-value numerical argument can be read. Closing
        # conservatively also handles groups with another nonempty unit.
        lines.extend(["else", "fort_numerical_guard = .false.", "endif", "endif"]
                     * len(guard.activation_domains))

    # Additional ordinary semantic checks can restrict execution, but cannot
    # supply the typed conditional physical footprint used above.
    scope = builder.analysis.source_scope_for(original[0], builder.entry.scope)
    covered = frozenset(effect.prepared.covered_runtime_guards)
    guards = (*(guard for guard in call.region.runtime_guards if guard not in covered),
              *(("fort_scope_numerical_environment_supported() /= 0",)
                if call.region.requires_numerical_environment else ()))
    for guard in guards:
        expression = F.Assignment_Stmt("fort_numerical_guard = " + guard).items[2]
        lines.append("if (fort_numerical_guard) fort_numerical_guard = "
                     + renamed(builder, expression, parameters, lexical_scope=scope))
    return specification, lines
