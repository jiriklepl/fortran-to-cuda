"""Authenticated original IEEE operations and strictly native local state.

These proofs retain the Fortran intrinsic, representation, lexical storage and
calling threads. They grant no device access, numerical legality or team cuts.
"""

from __future__ import annotations

_EXCEPTIONS = {
    "ieee_get_halting_mode", "ieee_set_halting_mode", "ieee_all", "ieee_usual",
    "ieee_invalid", "ieee_overflow", "ieee_divide_by_zero", "ieee_underflow", "ieee_inexact",
}
_EXPORTS = {module: {name: "$intrinsic::ieee_exceptions::" + name for name in _EXCEPTIONS}
            for module in ("ieee_exceptions", "ieee_arithmetic")}










def canonical_intrinsic_export(identity):
    """Recognize only the finite reviewed standard export set."""
    if not isinstance(identity, str):
        return None
    pieces = identity.split("::")
    return _EXPORTS.get(pieces[1], {}).get(pieces[2]) if len(pieces) == 3 and pieces[0] == "$intrinsic" else None


def intrinsic_export(analysis, scope, name):
    """Resolve one standard export through actual USE/host associations.

    Unknown wildcard modules, local procedures/storage and ambiguous exports
    cannot acquire intrinsic authority merely from an IEEE-looking spelling.
    """
    return _resolve_intrinsic_export(analysis, scope, name, _EXPORTS)


def _resolve_intrinsic_export(analysis, scope, name, reviewed_exports):
    """Share resolution mechanics; each native proof owns its finite exports."""
    name = str(name).lower()

    def canonical(identity):
        pieces = identity.split("::") if isinstance(identity, str) else ()
        return (reviewed_exports.get(pieces[1], {}).get(pieces[2])
                if len(pieces) == 3 and pieces[0] == "$intrinsic" else None)

    def resolve(owner, spelling, active):
        if owner is None:
            return set()
        key = id(owner), spelling
        if key in active or len(active) > analysis.depth_limit + analysis.procedure_limit:
            return {"$unresolved"}
        active = active | {key}
        if (spelling in owner.bindings or spelling in owner.procedure_arguments or spelling in owner.procedures
                or spelling in owner.externals or spelling in owner.generics or spelling in owner.ambiguous_imports):
            return {"$shadowed"}
        found = set()
        if spelling in owner.intrinsic_procedures:
            found.add(canonical(owner.intrinsic_procedures[spelling]) or "$unreviewed")
        for module in owner.intrinsic_wildcards:
            if spelling not in getattr(owner, "intrinsic_exclusions", {}).get(module, ()):
                value = reviewed_exports.get(module, {}).get(spelling)
                if value is not None:
                    found.add(value)
        sources = analysis.use_sources(owner, spelling)
        if sources is None:
            return {"$ambiguous"}
        for module, remote in sources:
            target = analysis.modules.get(module)
            if target is None:
                found.add("$unresolved")
            elif analysis._exported(target, remote):
                found.update(resolve(target, remote, active))
        return found if found else resolve(owner.parent, spelling, active)

    choices = resolve(scope, name, frozenset())
    return next(iter(choices)) if len(choices) == 1 and next(iter(choices)).startswith("$intrinsic::") else None
















