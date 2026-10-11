"""Register original caller descriptors before a mapped dummy can rebase them."""

from compiler.ir import CompilationError
from compiler.scopes.source import DTYPES, _name


def register_original_actuals(owner, arrays):
    """Reserve a deferred patch; numerical artifacts are known after scanning."""
    key = _name("fort_original_actuals_", owner.routine.qualified + str(len(owner.patches)))
    registrations = getattr(owner, "original_actual_registrations", {})
    owner.original_actual_registrations = registrations
    for mapping in arrays:
        if str(mapping.actual).lower().startswith("fort_"):
            raise CompilationError("original mapped actual conflicts with compiler-private registration names")
        owner.builder.capture(mapping.binding)
        owner.declaration(mapping.binding)
        owner.add_resource(mapping.binding)
    registrations[key] = tuple(arrays)
    return ["!" + key]


def finish_original_registrations(owner, text):
    for key, arrays in getattr(owner, "original_actual_registrations", {}).items():
        if "!" + key not in text:
            continue
        imports = ["use iso_c_binding, only: c_int, c_ptr, c_size_t, c_int64_t, c_loc, c_null_ptr",
                   "use fort_scoped_memory"]
        spec = ["integer(c_int) :: fort_actual_status", "type(c_ptr) :: fort_actual_pointer",
                "type(fort_scope_layout) :: fort_actual_layout"]
        lines = ["fort_actual_status = FORT_SCOPE_OK", f"if ({owner.enabled}) then",
                 f"if ({owner.context} == 0) then",
                 f"fort_actual_status = fort_scope_create(0_c_int, {owner.context})",
                 "if (fort_actual_status == FORT_SCOPE_OK) &",
                 f"fort_actual_status = fort_scope_set_device_budget({owner.context}, {owner.builder.device_budget}_c_size_t)"]
        lines += owner.builder.owner_transfer_setup(owner.owner.leaves, imports, context=owner.context,
                                                    status="fort_actual_status")
        lines += ["endif"]
        for mapping in arrays:
            binding, name = mapping.binding, str(mapping.actual)
            number = owner.resources[binding.root][1]
            handle = f"fort_buffer_{number}"
            _, enum, width = DTYPES[binding.signature()[:2]]
            extents, lowers = f"fort_actual_extents_{number}", f"fort_actual_lowers_{number}"
            spec += [f"integer(c_size_t), target :: {extents}({binding.rank})",
                     f"integer(c_int64_t), target :: {lowers}({binding.rank})"]
            lines += [f"if (fort_actual_status == FORT_SCOPE_OK .and. {handle} == 0) then",
                      f"{extents} = shape({name}, kind=c_size_t)",
                      f"{lowers} = lbound({name}, kind=c_int64_t)", "fort_actual_pointer = c_null_ptr",
                      f"if (all({extents} > 0)) fort_actual_pointer = c_loc({name})",
                      f"fort_actual_layout = fort_scope_layout({binding.rank}, {enum}, {width}_c_size_t, &",
                      f"fort_actual_pointer, c_loc({extents}), c_loc({lowers}), 1_c_int64_t)"]
            fact = owner.builder.capture(binding)
            lines += [f"fort_actual_status = fort_scope_register({owner.context}, {number}_c_int64_t, &",
                      f"1_c_int64_t, fort_actual_layout, {int(fact['initialized'] == 'whole')}_c_int, {handle})", "endif"]
        lines += ["if (fort_actual_status /= FORT_SCOPE_OK) then", *owner.close(), "endif", "endif"]
        text = text.replace("!" + key, "\n".join(["block", *imports, *spec, *lines, "end block"]))
    return text
