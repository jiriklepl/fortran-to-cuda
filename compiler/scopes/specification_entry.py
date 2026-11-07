"""Pre-entry publication for independently proved original array bounds.

Use original CALL handle forwarding when every bound array is already tracked.
Missing mappings or retired handles retain whole-domain observation. Neither
path inspects original descriptors; Fortran evaluates the bounds after reads
are coherent. Observation preserves device replicas and allocation identity.
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


def observation_bindings(parent, child, forwarding):
    proof = projection(child)
    if proof is None:
        return ()
    roots = proof.public()["array_resources"]
    if not all(root in child.resources for root in roots):
        return ()
    rows = []
    for root in roots:
        _binding, callee_slot = child.resources[root]
        caller_slot = forwarding.get(callee_slot)
        actuals = [binding for binding, slot in parent.resources.values() if slot == caller_slot]
        if len(actuals) != 1:
            raise CompilationError("specification entry lost its original metadata handle forwarding")
        rows.append({"resource": root, "callee_slot": callee_slot, "caller_slot": caller_slot,
                     "canonical_resource": parent.canonical(actuals[0]).root})
    return tuple(rows)


def publication(context, bindings=()):
    if bindings:
        slots = sorted({row["caller_slot"] for row in bindings})
        return ["block", "use iso_c_binding, only: c_int, c_int64_t, c_size_t, c_loc",
            "use fort_scoped_memory, only: fort_scope_observe_v1, fort_scope_observe_all_v1, &",
            "fort_scope_buffer_valid_v1, FORT_SCOPE_OK",
            "integer(c_int) :: fort_specification_status, fort_specification_valid",
            "integer :: fort_specification_index", "logical :: fort_specification_precise",
            f"integer(c_int64_t), target :: fort_specification_buffers({len(slots)})",
            f"if ({context} /= 0) then",
            *[f"fort_specification_buffers({index}) = fort_buffer_{slot}" for index, slot in enumerate(slots, 1)],
            "fort_specification_precise = .true.", f"do fort_specification_index = 1, {len(slots)}",
            f"fort_specification_status = fort_scope_buffer_valid_v1({context}, &",
            "fort_specification_buffers(fort_specification_index), fort_specification_valid)",
            "if (fort_specification_status /= FORT_SCOPE_OK) error stop 'original specification handle check failed'",
            "if (fort_specification_valid == 0) fort_specification_precise = .false.", "enddo",
            "if (fort_specification_precise) then",
            f"fort_specification_status = fort_scope_observe_v1({context}, &",
            f"c_loc(fort_specification_buffers), {len(slots)}_c_size_t)", "else",
            f"fort_specification_status = fort_scope_observe_all_v1({context})", "endif",
            "if (fort_specification_status /= FORT_SCOPE_OK) error stop 'original specification publication failed'",
            "endif", "end block"]
    return ["block", "use iso_c_binding, only: c_int",
            "use fort_scoped_memory, only: fort_scope_observe_all_v1, FORT_SCOPE_OK",
            "integer(c_int) :: fort_specification_status",
            f"if ({context} /= 0) then",
            f"fort_specification_status = fort_scope_observe_all_v1({context})",
            "if (fort_specification_status /= FORT_SCOPE_OK) error stop 'original specification publication failed'",
            "endif", "end block"]


def preflight(parent, child, forwarding):
    return (publication(parent.context, observation_bindings(parent, child, forwarding))
            if projection(child) is not None else [])


def call_record(parent, call):
    proof = projection(call.callee)
    if proof is None:
        return None
    bindings = observation_bindings(parent, call.callee, dict(call.handle_bindings))
    text = "\n".join(fortran_lines(publication(parent.context, bindings))) + "\n"
    marker = "call " + call.callee.entry + "("
    if call.source.count(text) != 1 or call.source.find(marker) <= call.source.find(text):
        raise CompilationError("specification entry lost publication before its original checked CALL")
    return {"version": 2, "proof_identity": proof.identity, "context": parent.context,
            "source_span": list(statement_span(call.node)), "publication_sha256": sha256(text.encode()).hexdigest(),
            "runtime": "fort_scope_observe_v1" if bindings else "fort_scope_observe_all_v1",
            "bindings": list(bindings),
            "policy": "publish mapped metadata; observe the domain when mappings or live handles are unavailable",
            "replicas": "preserve valid device replicas and allocation generations",
            "failure": "stop before child entry; never replay"}
