"""Bind CPU may-access acquisition to complete unchanged original work."""
from compiler.frontend.native_call_effects import NativeHostEffects
from compiler.ir import CompilationError


def validate(operation):
    proof = operation.call_effects
    if proof is None:
        return False
    if type(proof) is not NativeHostEffects:
        raise CompilationError("original host access requires registered complete CALL effects")
    proof.validate(operation.proof_analysis, proof.procedure, operation.nodes)
    expected = {binding.root: set(actions & {"read", "write"})
                for binding, role, actions in proof.resources
                if role == "managed" and actions & {"read", "write"}}
    if (not operation.preserve_original or operation.overwrites or operation.effects != expected
            or operation.sections is not proof.sections
            or operation.summary.get("demand_identity") != proof.identity
            or operation.summary.get("native_completion", {}).get("available") is not True):
        raise CompilationError("original host access differs from its unchanged CALL contract")
    return True


def public(operation):
    if not validate(operation):
        return None
    return {"version": 1, "begin": "fort_scope_original_host_begin_v1", "plan_kind": 6,
            "effects": "may-read and may-write", "definitions": "must-overwrites only",
            "proof": operation.call_effects.public()}
