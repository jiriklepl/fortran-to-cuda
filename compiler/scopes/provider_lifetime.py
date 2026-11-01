"""Emit explicit provider-only owners and original native call companions."""
from __future__ import annotations

from hashlib import sha256

from compiler.frontend.provider_lifetime import encoded, ending, header, retained_state_qualification, span
from compiler.frontend.source_effects import _children, _kind, _part
from compiler.ir import CompilationError
from compiler.scopes.segments import fortran_lines, statement_span

LIFE = "fort_provider_life"
PARENT = "fort_provider_parent"
BORROWED = "fort_provider_borrowed"


def selection(member):
    proof = getattr(member.builder, "provider_only_lifetime", None)
    return proof if proof is not None and member.routine is member.builder.entry else None


def module_name(proof):
    return "fort_provider_lifetime_" + proof.identity[:12]


def entry_name(proof, routine):
    return "fort_provider_entry_" + sha256((proof.identity + routine.qualified).encode()).hexdigest()[:12]



def context_creation(member, context, default):
    """Every selected lexical/native/external epoch belongs to the parent slot."""
    if selection(member) is None:
        return f"{default}(0_c_int, {context})"
    return f"fort_provider_borrow({LIFE},{context},{member.builder.device_budget}_c_size_t)"


def registration_generation(member):
    return f"{LIFE}%generation" if selection(member) is not None else "1_c_int64_t"


def rearm_condition(member, condition):
    return condition + (f" .and. {LIFE}%enabled" if selection(member) is not None else "")


def control(member):
    proof = selection(member)
    if proof is None:
        return None
    proof.validate()
    return ([f"use {module_name(proof)}"], [
        "type(fort_provider_lifetime), target :: fort_provider_local",
        f"type(fort_provider_lifetime), target, intent(inout) :: {PARENT}",
        f"type(fort_provider_lifetime), pointer :: {LIFE}", f"logical :: {BORROWED}"], [])


def initialize(member, code):
    proof = selection(member)
    if proof is None:
        return code
    args = [*member.routine.arguments, PARENT]
    return [f"{BORROWED} = .false.", "call fort_provider_reset(fort_provider_local)",
            f"{LIFE} => fort_provider_local", "goto 99999",
            f"entry {entry_name(proof, member.routine)}(" + ", ".join(args) + ")",
            f"{BORROWED} = .true.", f"{LIFE} => {PARENT}", "99999 continue", *code,
            f"{member.enabled} = {member.enabled} .and. {LIFE}%enabled"]


def close(member, *, disable, reason, span, prework, query_retirement):
    if selection(member) is None:
        return None
    from compiler.scopes.provenance import position, source_id
    boundary = source_id(member.builder, "boundary", span=span,
                         reason=reason or "owner cleanup or native fallback at the current source position")
    handles = [f"fort_buffer_{number}" for _binding, number in member.resources.values()]
    values = ",".join(handles)
    code = ["block", "use iso_c_binding, only: c_int, c_int64_t, c_int32_t, c_null_char, c_null_ptr",
            "use fort_scoped_memory, only: fort_scope_plan_abandon_remaining_v1, fort_scope_trace_set_v1, FORT_SCOPE_OK, &",
            "FORT_SCOPE_ARGUMENT, FORT_SCOPE_BOUNDARY, FORT_SCOPE_UNINITIALIZED, FORT_SCOPE_PLAN_MISMATCH",
            "integer(c_int) :: fort_close_status",
            f"integer(c_int64_t) :: fort_provider_fields({len(handles)})",
            *query_retirement, f"if ({member.context} /= 0) then",
            *position(member.builder, context=member.context, owner=member.trace_owner, boundary=boundary), "endif",
            f"fort_provider_fields = [{values}]",
            f"call fort_provider_publish({LIFE},{member.context},fort_provider_fields)",
            *[f"{handle} = 0" for handle in handles],
            f"{member.epoch_pristine} = .false.", f"{member.query_plan} = .false."]
    if prework:
        code += [f"call fort_provider_retire({LIFE})", f"{member.enabled} = .false."]
    elif reason == "original owning invocation end":
        code += [f"if (.not. {BORROWED}) call fort_provider_retire({LIFE})"]
    if disable:
        code.append(f"{member.enabled} = .false.")
    return [*code, "end block"]


def helper_source(proof, task, count):
    interface = task.proof.public()["provider"]["fortran"]
    # Scalar private plan state only. No host pointer, array identity, field
    # handle or replica is stored here; fields belong to each borrowed call.
    return f'''module {module_name(proof)}
use iso_c_binding
use fort_scoped_memory
use {interface['module']}, only: fort_provider_destroy => {interface['destroy']}
implicit none
private
public :: fort_provider_lifetime, fort_provider_reset, fort_provider_borrow, fort_provider_publish, fort_provider_retire
type :: fort_provider_lifetime
  integer(c_int64_t) :: domain, generation, words({count})
  type(c_ptr) :: provider
  logical :: enabled, active
end type
contains
subroutine fort_provider_reset(life)
  type(fort_provider_lifetime), intent(out) :: life
  life%domain = 0
  life%generation = 0
  life%words = 0
  life%provider = c_null_ptr
  life%enabled = .true.
  life%active = .false.
end subroutine
integer(c_int) function fort_provider_borrow(life,context,budget) result(status)
  type(fort_provider_lifetime), intent(inout) :: life
  integer(c_int64_t), intent(out) :: context
  integer(c_size_t), intent(in) :: budget
  context = 0
  status = FORT_SCOPE_BOUNDARY
  if (life%active) error stop 'provider-only lifetime reentry'
  if (.not. life%enabled) return
  if (fort_scope_serial_caller() == 0) return
  if (life%generation == huge(life%generation)) return
  if (life%domain == 0) status = fort_scope_domain_create_v1(0_c_int,life%domain)
  if (life%domain /= 0) status = FORT_SCOPE_OK
  if (status == FORT_SCOPE_OK) status = fort_scope_create_in_domain_v1(life%domain,context)
  if (status == FORT_SCOPE_OK) then
    ! Once attached, even a later budget rejection must publish/detach this owner.
    life%generation = life%generation + 1
    life%active = .true.
    status = fort_scope_set_device_budget(context,budget)
  endif
end function
subroutine fort_provider_publish(life,context,fields)
  type(fort_provider_lifetime), intent(inout) :: life
  integer(c_int64_t), intent(inout) :: context,fields(:)
  integer(c_int) :: status
  integer(c_int64_t) :: actual_domain
  integer :: i
  if (context /= 0) then
    if (.not. life%active .or. life%domain == 0 .or. life%generation == 0) &
      error stop 'provider-only publication lacks its active owner'
    actual_domain = 0
    status = fort_scope_domain_get_v1(context,actual_domain)
    if (status /= FORT_SCOPE_OK .or. actual_domain /= life%domain) &
      error stop 'provider-only publication has a foreign domain'
    ! Even read-only uploads require successful completion before unregister.
    status = fort_scope_wait(context)
    if (status /= FORT_SCOPE_OK) error stop 'provider-only completion failed; replay prohibited'
    do i=1,size(fields)
      if (fields(i) == 0) cycle
      status = fort_scope_unregister(context,fields(i))
      if (status /= FORT_SCOPE_OK) error stop 'provider-only field publication failed; replay prohibited'
      fields(i) = 0
    enddo
    status = fort_scope_close(context)
    if (status /= FORT_SCOPE_OK) error stop 'provider-only borrowed detach failed; replay prohibited'
    context = 0
  else
    if (any(fields /= 0)) error stop 'provider-only field lacks its borrowed context'
  endif
  life%active = .false.
end subroutine
subroutine fort_provider_retire(life)
  type(fort_provider_lifetime), intent(inout) :: life
  integer(c_int) :: status
  if (life%active) error stop 'provider-only retirement before field publication'
  life%enabled = .false.
  if (c_associated(life%provider)) then
    status = fort_provider_destroy(life%provider)
    if (status /= FORT_SCOPE_OK) error stop 'provider-only plan destruction failed'
    life%provider = c_null_ptr
  endif
  if (life%domain /= 0) then
    status = fort_scope_domain_close_v1(life%domain)
    if (status /= FORT_SCOPE_OK) error stop 'provider-only domain retirement failed'
    life%domain = 0
  endif
  life%words = 0
end subroutine
end module
'''


def _module_public(builder, routine, entry):
    module = routine.scope.parent
    contains = next(node for node in _children(_part(module.node, "Module_Subprogram_Part"))
                    if _kind(node) == "Contains_Stmt")
    line = span(contains)[0]
    builder.add_edit(module.path, line, line - 1, "public :: " + entry + "\n")


def _insert_import(builder, routine, imports):
    line = span(header(routine))[1] + 1
    builder.add_edit(routine.scope.path, line, line - 1, "\n".join(imports) + "\n")


def _render_call(proof, caller, node, resolved, slot):
    arguments = [*((item.keyword + " = " if item.keyword is not None else "") + str(item.actual)
                   for item in resolved.original_arguments), f"{PARENT}={slot}"]
    call = f"call {entry_name(proof, resolved.callee)}(" + ", ".join(arguments) + ")"
    parent = getattr(node, "parent", None)
    if type(parent).__name__ == "If_Stmt":
        return [f"if ({parent.items[0]}) then", call, "endif"]
    return [call]


def emit_spine(builder):
    proof = getattr(builder, "provider_only_lifetime", None)
    if proof is None or getattr(builder, "_provider_lifetime_emitted", False):
        return
    proof.validate()
    if encoded(builder.facts.get("provider_only_lifetime")) != proof._request:
        raise CompilationError("provider-only lifetime request changed")
    owners = [owner for owner in builder._provider_lifetime_members if owner.routine is builder.entry]
    if len(owners) != 1:
        raise CompilationError("provider-only lifetime requires one checked external leaf owner")
    member, = owners
    tasks = member.device_tasks
    if (len(tasks) != 1 or tasks[0].proof.public()["version"] != 2 or member.leaves
            or member.source_device_tasks or len(member.members) != 1):
        raise CompilationError("provider-only lifetime first slice requires one external v2 task and no other device work")
    task, = tasks
    task.proof.validate(builder.analysis, member.routine, task.node)
    provider = task.proof.public()["provider"]
    retained_state = retained_state_qualification(provider)
    from compiler.scopes.device_tasks import _word_count
    helper = helper_source(proof, task, max(1, _word_count(task)))
    helper_path = "provider_lifetimes/" + module_name(proof) + ".f90"
    builder.outputs[helper_path] = helper
    _module_public(builder, member.routine, entry_name(proof, member.routine))
    root = proof._routines[0]
    for routine in proof._routines[:-1]:
        imports = [f"use {module_name(proof)}"]
        children = {id(resolved.callee): resolved.callee for caller, _node, resolved in proof._calls if caller is routine}
        for child in children.values():
            if child.scope.parent is not routine.scope.parent:
                imports.append(f"use {child.scope.parent.module}, only: {entry_name(proof, child)}")
        _insert_import(builder, routine, imports)
        first = statement_span(next(iter(_children(routine.execution))))[0]
        if routine is root:
            prologue = ["type(fort_provider_lifetime), target :: fort_provider_owner",
                        "call fort_provider_reset(fort_provider_owner)"]
            internal = _part(routine.scope.node, "Internal_Subprogram_Part")
            last = (span(next(node for node in _children(internal) if _kind(node) == "Contains_Stmt"))[0]
                    if internal is not None else span(ending(routine))[0])
            builder.add_edit(routine.scope.path, last, last - 1,
                             "call fort_provider_retire(fort_provider_owner)\n")
        else:
            prologue = [f"type(fort_provider_lifetime), target, intent(inout) :: {PARENT}",
                "logical :: fort_provider_checked", "fort_provider_checked = .false.", "goto 99999",
                f"entry {entry_name(proof, routine)}(" + ", ".join([*routine.arguments, PARENT]) + ")",
                "fort_provider_checked = .true.", "99999 continue"]
            _module_public(builder, routine, entry_name(proof, routine))
        builder.add_edit(routine.scope.path, first, first - 1, "\n".join(fortran_lines(prologue)) + "\n", prepend=True)
    for caller, node, resolved in proof._calls:
        slot = "fort_provider_owner" if caller is root else PARENT
        lines = _render_call(proof, caller, node, resolved, slot)
        if caller is not root:
            # Keep an original one-line IF outside the dispatch so its source
            # condition is evaluated once, with the original short-circuit point.
            parent = getattr(node, "parent", None)
            guard = parent.items[0] if type(parent).__name__ == "If_Stmt" else None
            call = _render_call(proof, caller, node, resolved, slot)
            if guard is not None:
                call = call[1:-1]
            lines = ["if (fort_provider_checked) then", *call, "else", str(node), "endif"]
            if guard is not None:
                lines = [f"if ({guard}) then", *lines, "endif"]
        first, last = span(node)
        builder.add_edit(caller.scope.path, first, last, "\n".join(fortran_lines(lines)) + "\n")
    builder.provider_lifetime_record = {**proof.public(), "provider_slot": 1,
        "provider_source_proof_identity": task.proof.identity,
        "provider_manifest": task.proof.public()["contract"]["provider_manifest"],
        "canonical_resources": task.proof.public()["canonical_resources"],
        "provider_retained_state": {"qualification": provider["retained_state_qualification"], "record": retained_state},
        "configuration_words": _word_count(task), "helper": {"path": helper_path, "sha256": sha256(helper.encode()).hexdigest()},
        "checked_leaf_entry": entry_name(proof, member.routine),
        "generation": "fresh monotonically increasing owner-call generation; every field is unregistered before return"}
    builder._provider_lifetime_emitted = True
