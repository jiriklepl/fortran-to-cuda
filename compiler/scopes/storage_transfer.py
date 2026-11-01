"""Retain proved descriptor exchanges without publishing coherent payloads."""
from dataclasses import dataclass

from compiler.frontend.allocation_exchange import allocation_exchange
from compiler.ir import CompilationError
from compiler.scopes.call_registration import OriginalActual, register_original_actuals
from compiler.scopes.segments import fortran_lines, statement_span


@dataclass(frozen=True)
class StorageTransferPatch:
    proof: object
    node: object
    registration: tuple
    first: int
    last: int

    def render(self, member):
        self.proof.validate(member.builder.analysis, member.routine, self.node)
        if statement_span(self.node) != (self.first, self.last):
            raise CompilationError("allocation exchange original source span changed")
        bindings = self.proof.bindings
        names = tuple(str(item.actual) for item in self.proof._resolved.associations)
        handles = tuple("fort_buffer_" + str(member.resources[binding.root][1]) for binding in bindings)
        before = ["fort_swap_can = .true.", f"if ({member.enabled}) then"]
        for name, binding in zip(names, bindings, strict=True):
            before += [f"if (fort_swap_can) fort_swap_can = allocated({name})",
                       f"if (fort_swap_can) fort_swap_can = is_contiguous({name})"]
            before += [f"if (fort_swap_can) fort_swap_can = {condition}"
                       for condition in member.builder.original_bound_conditions(name, binding.rank)]
        before += ["if (fort_swap_can) then", *self.registration, "else",
                   *member.close(disable=False, prework=True, reason="unavailable allocation exchange descriptors",
                                 span=(self.first, self.last)), "endif", "endif"]
        # A checked child can reopen an epoch using an assumed-shape dummy's
        # rebased descriptor. That physical registration is valid for the child,
        # but it does not establish this caller's allocation coordinate origin.
        # Retire BEFORE the original mutation rather than weakening the exact
        # opposite-allocation postcondition or manufacturing caller provenance.
        before += [f"if ({member.context} /= 0) then"]
        for name, binding, handle in zip(names, bindings, handles, strict=True):
            from compiler.scopes.source import DTYPES

            _, enum, width = DTYPES[binding.signature()[:2]]
            before += [f"fort_swap_status = fort_scope_layout_get({member.context}, {handle}, fort_swap_registered)",
                       "if (fort_swap_status /= FORT_SCOPE_OK) error stop 'allocation exchange pre-mutation layout failed'",
                       f"if (fort_swap_registered%rank /= {binding.rank}) fort_swap_can = .false.",
                       f"if (fort_swap_registered%element_type /= {enum}) fort_swap_can = .false.",
                       f"if (fort_swap_registered%element_bytes /= {width}_c_size_t) fort_swap_can = .false.",
                       "if (fort_swap_registered%generation /= 1_c_int64_t) fort_swap_can = .false.",
                       "if (fort_swap_can) then",
                       f"call c_f_pointer(fort_swap_registered%extents, fort_swap_registered_extents, [{binding.rank}])",
                       f"call c_f_pointer(fort_swap_registered%lower_bounds, fort_swap_registered_lowers, [{binding.rank}])",
                       f"fort_swap_can = all(fort_swap_registered_extents == shape({name},kind=c_size_t))",
                       f"if (fort_swap_can) fort_swap_can = all(fort_swap_registered_lowers == lbound({name},kind=c_int64_t))",
                       "fort_swap_original_pointer = c_null_ptr",
                       f"if (all(shape({name},kind=c_size_t) > 0)) fort_swap_original_pointer = c_loc({name})",
                       "if (c_associated(fort_swap_original_pointer)) then",
                       "if (fort_swap_can) fort_swap_can = c_associated(fort_swap_registered%host, fort_swap_original_pointer)",
                       "else", "if (fort_swap_can) fort_swap_can = .not. c_associated(fort_swap_registered%host)",
                       "endif", "endif"]
        before += ["if (.not. fort_swap_can) then",
                   *member.close(disable=False, prework=True,
                       reason="allocation exchange registration coordinates differ from original descriptor",
                       span=(self.first, self.last)), "endif", "endif"]
        if member.control_guard:
            before = [f"if ({member.control_guard}) then", *before, "endif"]
        imports = ["use iso_c_binding", "use fort_scoped_memory"]
        prefix = ["block", *imports, "logical :: fort_swap_can", "integer(c_int) :: fort_swap_status",
                  "type(fort_scope_layout) :: fort_swap_registered", "type(c_ptr) :: fort_swap_original_pointer",
                  "integer(c_size_t), pointer :: fort_swap_registered_extents(:)",
                  "integer(c_int64_t), pointer :: fort_swap_registered_lowers(:)", *before, "end block"]
        rank = bindings[0].rank
        spec = ["integer(c_int) :: fort_swap_status", "type(fort_scope_layout) :: fort_swap_layout(2)",
                "type(c_ptr) :: fort_swap_pointer", "integer(c_int64_t) :: fort_swap_current(2)",
                f"integer(c_size_t), target :: fort_swap_extents({rank},2)",
                f"integer(c_int64_t), target :: fort_swap_lowers({rank},2)"]
        from compiler.scopes.source import DTYPES
        after = [f"if ({member.context} /= 0) then"]
        for index, (name, binding) in enumerate(zip(names, bindings, strict=True), 1):
            _, enum, width = DTYPES[binding.signature()[:2]]
            after += [f"fort_swap_extents(:,{index}) = shape({name},kind=c_size_t)",
                      f"fort_swap_lowers(:,{index}) = lbound({name},kind=c_int64_t)",
                      "fort_swap_pointer = c_null_ptr",
                      f"if (all(fort_swap_extents(:,{index}) > 0)) fort_swap_pointer = c_loc({name})",
                      f"fort_swap_layout({index}) = fort_scope_layout({rank}, {enum}, {width}_c_size_t, &",
                      f"fort_swap_pointer, c_loc(fort_swap_extents(:,{index})), &",
                      f"c_loc(fort_swap_lowers(:,{index})), 1_c_int64_t)"]
        after += [f"fort_swap_status = fort_scope_identity_swap_v1({member.context}, {handles[0]}, {handles[1]}, &",
                  "fort_swap_layout(1), fort_swap_layout(2), fort_swap_current(1), fort_swap_current(2))",
                  "if (fort_swap_status /= FORT_SCOPE_OK) error stop 'allocation exchange coherence failed after mutation'",
                  f"{handles[0]} = fort_swap_current(1)", f"{handles[1]} = fort_swap_current(2)",
                  f"{member.query_plan} = .false.", "endif"]
        if member.control_guard:
            after = [f"if ({member.control_guard}) then", *after, "endif"]
        suffix = ["block", *imports, *spec, *after, "end block"]
        return ("\n".join(fortran_lines(prefix)) + "\n" + member.original((self.node,))
                + "\n".join(fortran_lines(suffix)) + "\n")


def admit_storage_transfer(member, node):
    if member.builder.config.scope_execution != "graph":
        return False
    proof = allocation_exchange(member.builder.analysis, member.routine, node)
    if proof is None:
        return False
    for binding in proof.bindings:
        member.builder.capture(binding)
        member.declaration(binding)
        member.add_resource(binding)
    registration = register_original_actuals(member, tuple(
        OriginalActual(binding, str(item.actual)) for binding, item in
        zip(proof.bindings, proof._resolved.associations, strict=True)))
    first, last = statement_span(node)
    patch = StorageTransferPatch(proof, node, tuple(registration), first, last)
    member.storage_transfers.append(patch)
    member.patches.append((first, last, patch))
    return True
