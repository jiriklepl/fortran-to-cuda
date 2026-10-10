"""Apply compiler-emitted metadata proofs at a fresh original source entry."""

from dataclasses import dataclass, field
from hashlib import sha256


@dataclass(frozen=True)
class SourceNativePreflight:
    imports: tuple[str, ...] = ()
    expression: str | None = None
    public: dict = field(default_factory=dict)


def source_native_preflight(builder, calls, leaves, values, *, original_descriptors=False):
    """Map one direct numerical call without evaluating any source expression.

    The caller must apply this only after original descriptor/semantic guards
    and only while no owning context exists. Unsupported mappings add no code.
    Calls through coordinators, sections and mutable scalar expressions keep
    their existing reached planning path.
    """
    from fparser.two import Fortran2003 as F

    from compiler.scopes.source import view_call

    def unavailable(reason):
        return SourceNativePreflight(public={"abi_version": 1, "available": False, "reason": reason})

    if builder.config.policy != "auto":
        return unavailable("metadata preflight applies only to automatic placement")
    if len(calls) != 1 or len(leaves) != 1:
        return unavailable("metadata preflight requires one direct numerical call")
    call, = calls
    if call.procedure not in leaves or view_call(call):
        return unavailable("coordinator and section mappings require reached planning")
    generated = builder.numerical(call.procedure)
    if generated is None:
        return unavailable("numerical artifact is unavailable")
    contract = generated.scoped.get("native_preflight", {})
    if not contract.get("available"):
        return unavailable(contract.get("reason") or "metadata proof is unavailable")
    package = call.region or builder.packages.get(call.procedure)
    parameters = {parameter.name.lower(): parameter for parameter in package.parameters} if package else {}
    actual_bindings = {binding.root: binding for binding in call.bindings.values()}
    arguments = []
    for item in contract["parameters"]:
        name = item["parameter"].lower()
        parameter = parameters.get(name)
        formal = parameter.resource if parameter is not None else "argument::" + name
        binding = call.bindings.get(formal) or actual_bindings.get(formal)
        root = binding.root if binding is not None else formal
        if root not in values:
            return unavailable("metadata input lacks an original whole-variable mapping: " + name)
        visible = values[root]
        if item["kind"] == "array_extent":
            if ((parameter is not None and not parameter.rank)
                    or (binding is not None and not binding.rank)):
                return unavailable("metadata extent does not map to an original array")
            arguments.append(f"size({visible}, {item['dimension']}, kind=c_size_t)")
        elif item["kind"] == "integer_scalar":
            if (binding is not None and not binding.rank
                    and binding.attributes & {"volatile", "asynchronous", "optional", "pointer", "allocatable"}):
                return unavailable("metadata scalar has unsupported observable or guarded storage")
            if parameter is not None and parameter.lower_bound_dimension is not None:
                if not original_descriptors:
                    return unavailable("original lower bounds are unavailable before synthetic dummy rebasing")
                arguments.append(builder.lower_bound_actual(parameter, visible))
            elif binding is not None and binding.signature()[:2] == ("integer", 4):
                arguments.append(visible)
            else:
                return unavailable("metadata scalar does not map to an original INTEGER input")
        else:
            return unavailable("unsupported compiler metadata parameter")
    alias = "fort_native_preflight_" + sha256(call.procedure.encode()).hexdigest()[:12]
    if (builder.analysis._binding(builder.entry.scope, alias)
            or builder.analysis._candidates(builder.entry.scope, F.Name(alias))
            or alias in builder.entry.scope.imports or alias in builder.entry.scope.ambiguous_imports):
        return unavailable("metadata preflight alias conflicts with original source")
    public, _ = builder.entry_artifacts(call.procedure)
    return SourceNativePreflight(
        imports=(f"use {public['fortran_module']}, only: {alias} => {contract['fortran_procedure']}",),
        expression=alias + "(" + ", ".join(arguments) + ") == 1_c_int",
        public={**contract, "source_arguments": arguments,
                "position": "after original guards, before fresh context creation",
                "selection": "unchanged original reached unit; no prefix replay"})
