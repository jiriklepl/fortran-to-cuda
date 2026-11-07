"""Pre-entry publication for independently proved original array bounds.

The first implementation deliberately observes the whole live domain. This
needs no guessed identity for private metadata and never inspects a descriptor
on a path that has not reached the original CALL. Observation preserves device
replicas; the original compiler alone evaluates the bounds, after publication.
"""
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.source_effects import _kind
from compiler.frontend.specification_reads import SpecificationReadsProof, prove_specification_reads
from compiler.ir import CompilationError
from compiler.scopes.segments import fortran_lines, statement_span


def prepare(member):
    if getattr(member.owner.builder, "_conditional_return_draft", None) is None:
        return
    if any(_kind(node) in {"Part_Ref", "Data_Ref", "Function_Reference", "Structure_Constructor"}
           for binding in member.routine.scope.bindings.values()
           for bound in (*binding.shape_nodes, *binding.lower_bound_nodes) for node in walk(bound)):
        proof = prove_specification_reads(member.builder.analysis, member.routine)
        if not proof.public()["array_resources"]:
            raise CompilationError("specification entry requires an original array-read projection")
        member.specification_reads = proof


def projection(member):
    proof = getattr(member, "specification_reads", None)
    if proof is not None:
        if type(proof) is not SpecificationReadsProof or member is member.owner:
            raise CompilationError("specification entry requires its checked original child")
        proof.validate(member.builder.analysis, member.routine)
    return proof


def publication(context):
    return ["block", "use iso_c_binding, only: c_int",
            "use fort_scoped_memory, only: fort_scope_observe_all_v1, FORT_SCOPE_OK",
            "integer(c_int) :: fort_specification_status",
            f"if ({context} /= 0) then",
            f"fort_specification_status = fort_scope_observe_all_v1({context})",
            "if (fort_specification_status /= FORT_SCOPE_OK) error stop 'original specification publication failed'",
            "endif", "end block"]


def preflight(parent, child):
    return publication(parent.context) if projection(child) is not None else []


def call_record(parent, call):
    proof = projection(call.callee)
    if proof is None:
        return None
    text = "\n".join(fortran_lines(publication(parent.context))) + "\n"
    marker = "call " + call.callee.entry + "("
    if call.source.count(text) != 1 or call.source.find(marker) <= call.source.find(text):
        raise CompilationError("specification entry lost publication before its original checked CALL")
    return {"version": 1, "proof_identity": proof.identity, "context": parent.context,
            "source_span": list(statement_span(call.node)), "publication_sha256": sha256(text.encode()).hexdigest(),
            "runtime": "fort_scope_observe_all_v1",
            "policy": "publish initialized domain regions before original argument and specification evaluation",
            "replicas": "preserve valid device replicas and allocation generations",
            "failure": "stop before child entry; never replay"}
