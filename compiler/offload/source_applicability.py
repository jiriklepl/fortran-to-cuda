"""Cost-only compiled-form metadata from an authenticated original region.

The record carries no execution, capture or definition authority. Its owner
keeps the existing prepared-region validation and binds these identities when
passing it to the generated worker's cost query.
"""

from dataclasses import dataclass

from compiler.offload.memory_applicability import POINTWISE_THREE_ARRAY


@dataclass(frozen=True)
class SourceComputeApplicability:
    native_compiled_form: str
    generated_compiled_form: str
    preparation_identity: str
    source_identity: str
    extraction_source_identity: str
    native_participation: str
    access_class: str | None = None

    def public(self):
        return {"schema_version": 1, "native_compiled_form": self.native_compiled_form,
                "generated_compiled_form": self.generated_compiled_form,
                "preparation_identity": self.preparation_identity, "source_identity": self.source_identity,
                "extraction_source_identity": self.extraction_source_identity,
                "native_participation": self.native_participation, "access_class": self.access_class,
                "authority": "cost applicability only; original prepared-region authority remains required"}


def source_compute_applicability(analysis, prepared, *, native_participation):
    """Classify one original direct loop; unsupported or stale costs decline.

    Surrounding source consumers still perform their hard authority checks.
    This optional classifier never substitutes for those checks or manufactures
    preparation authority for whole-procedure or independently lowered IR.
    """
    from compiler.frontend.numerical_preparation import PreparedOriginalRegion, _Budget, _walk
    from compiler.ir import CompilationError
    from compiler.offload.source_compute import fork_join_participation

    if type(prepared) is not PreparedOriginalRegion or native_participation not in {"serial", "fork_join"}:
        return None
    try:
        prepared.validate(analysis, prepared.procedure, prepared._selection)
        extraction = prepared.extraction
        if (extraction.operation_kind != "numerical_loop" or extraction.numerical_helpers
                or extraction.tile_domains or not prepared.completion.get("available")):
            return None
        has_openmp = prepared.completion.get("has_openmp_in_closure", False)
        expected = fork_join_participation(extraction.nodes) if has_openmp else "serial"
        if expected != native_participation:
            return None
        roots = tuple(node for node in extraction.nodes if type(node).__name__ != "Comment")
        if len(roots) != 1 or type(roots[0]).__name__ != "Block_Nonlabel_Do_Construct":
            return None
        loops = 0
        routine = analysis.routines[prepared.procedure]
        for node in _walk(extraction.nodes, _Budget(None)):
            kind = type(node).__name__
            if kind == "Block_Nonlabel_Do_Construct":
                loops += 1
            elif kind in {"Call_Stmt", "Function_Reference", "Structure_Constructor"}:
                return None
            elif kind == "Part_Ref":
                scope = analysis.source_scope_for(node, routine.scope)
                binding = analysis._binding(scope, str(node.items[0]).lower())
                if binding is None or not binding.rank:
                    return None
        if loops != 1:
            return None
        units = prepared.offload.units
        requirement = units[0].memory_access_requirement if len(units) == 1 else None
        access_class = (POINTWISE_THREE_ARRAY if requirement is not None and requirement.available
                        and requirement.class_id == POINTWISE_THREE_ARRAY else None)
        return SourceComputeApplicability("direct_loop", "direct_loop", prepared.identity,
            prepared._source_identity, extraction.source_identity, native_participation, access_class)
    except CompilationError:
        return None
