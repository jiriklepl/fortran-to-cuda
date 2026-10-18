"""Consume exact original-read preconditions at their reached numerical point."""

from compiler.ir import CompilationError
from compiler.scopes.access import build_native_accesses
from compiler.scopes.registration import RegistrationCode


def prepare_source_reads(builder, call):
    """Retain a source token before the ordered execution plan is sealed."""
    if getattr(builder, "_selection_activation_binding", None) is None:
        return None
    from compiler.frontend.reached_read_requirements import prove_reached_read_requirements

    if call.region is None:
        raise CompilationError("original read admission requires prepared original numerical source")
    inline = builder.inline_for(call.procedure)
    prepared = inline.prepared_regions.get(call.procedure)
    if prepared is None:
        raise CompilationError("original read admission lost its original numerical preparation")
    return prove_reached_read_requirements(builder.analysis, builder.entry.qualified,
                                          prepared._selection, prepared)


def validate_source_reads(builder, call, proof):
    """A valid same-procedure proof cannot authorize another original call."""
    if proof is None:
        return None
    from compiler.frontend.reached_read_requirements import ReachedReadRequirementsProof
    if type(proof) is not ReachedReadRequirementsProof:
        raise CompilationError("original read admission requires its registered typed source token")
    inline = builder.inline_for(call.procedure)
    if (call.region is None or inline.regions.get(call.procedure) is not call.region
            or inline.prepared_regions.get(call.procedure) is not proof.prepared):
        raise CompilationError("original read admission differs from its exact original numerical call")
    originals = inline.original_selection(call.node)
    if (len(originals) != len(proof.prepared._selection)
            or any(left is not right for left, right in zip(originals, proof.prepared._selection, strict=True))):
        raise CompilationError("original read admission differs from its exact original source selection")
    proof.validate(builder.analysis, builder.entry.qualified, proof.prepared._selection, proof.prepared)
    return proof


def build_source_reads(builder, proof, handles, values, prefix, *, call, on_error,
                       logical_lower_bounds=None):
    """Admit only missing exact reads after source guards, before local query.

    Original descriptors and canonical nonaliasing registration are separate
    prerequisites. Existing device-current points retain their authority.
    """
    if validate_source_reads(builder, call, proof) is None:
        return RegistrationCode((), ())
    return _build_read_sections(proof.sections, handles, values, prefix,
                                on_error=on_error, logical_lower_bounds=logical_lower_bounds)


def validate_native_source_reads(builder, operation):
    if operation.source_reads is None:
        return None
    from dataclasses import replace

    from compiler.frontend.reached_read_requirements import native_read_sections

    analysis = operation.proof_analysis
    if (analysis is None or analysis.session_identity != builder.analysis.session_identity
            or analysis.routines[builder.entry.qualified] is not builder.entry):
        raise CompilationError("native reached reads lost their original source session")
    sections = native_read_sections(analysis, builder.entry.qualified,
        tuple(builder.inline.original_selection(operation.nodes)), operation.source_reads)
    return replace(sections, resources=tuple(item for item in sections.resources
                                             if item.resource in operation.coherence_effects))


def build_native_source_reads(builder, operation, handles, values, prefix, *, on_error,
                              logical_lower_bounds=None):
    sections = validate_native_source_reads(builder, operation)
    if sections is None:
        return RegistrationCode((), ())
    parameters = dict(values)
    for item in sections.resources:
        for dependency in item.dependencies:
            if dependency.kind == "scalar_read" and dependency.resource not in parameters:
                parameters[dependency.resource] = builder.visible(builder.entry, dependency.resource)
    return _build_read_sections(sections, handles, parameters, prefix,
                                on_error=on_error, logical_lower_bounds=logical_lower_bounds)


def _build_read_sections(sections, handles, values, prefix, *, on_error, logical_lower_bounds):
    accesses = build_native_accesses(sections, handles, prefix, parameters=values,
        logical_lower_bounds=logical_lower_bounds, on_error=tuple(on_error))
    specification, prepare = [], []
    for access in accesses:
        specification.extend(access.specification)
        prepare.extend(access.prepare)
        prepare.extend([
            "fort_status = fort_scope_admit_source_reads_v1(fort_context, " + access.handle + ", &",
            f"{access.access_name}%reads, {access.access_name}%read_count)",
            "if (fort_status /= FORT_SCOPE_OK) then", *on_error, "endif"])
    return RegistrationCode(tuple(specification), tuple(prepare))
