"""Register original caller descriptors before a mapped dummy can rebase them."""

from dataclasses import dataclass

from compiler.ir import CompilationError
from compiler.scopes.registration import build_registration
from compiler.scopes.source import DTYPES, _name


@dataclass(frozen=True)
class OriginalActual:
    binding: object
    actual: str


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
        imports = ["use iso_c_binding",
                   "use fort_scoped_memory"]
        block = _name("fort_actuals_", key)
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
        failure = [*owner.close(), "exit " + block]
        for mapping in arrays:
            binding, name = mapping.binding, str(mapping.actual)
            number = owner.resources[binding.root][1]
            handle = f"fort_buffer_{number}"
            _, enum, width = DTYPES[binding.signature()[:2]]
            fact = owner.builder.capture(binding)
            if fact.get("requires_registered_handle"):
                lines += [f"if ({handle} == 0) then", "fort_actual_status = FORT_SCOPE_BOUNDARY", *failure, "endif"]
                continue
            extents, lowers = f"fort_actual_extents_{number}", f"fort_actual_lowers_{number}"
            spec += [f"integer(c_size_t), target :: {extents}({binding.rank})",
                     f"integer(c_int64_t), target :: {lowers}({binding.rank})"]
            lines += [f"if (fort_actual_status == FORT_SCOPE_OK .and. {handle} == 0) then",
                      f"{extents} = shape({name}, kind=c_size_t)",
                      f"{lowers} = lbound({name}, kind=c_int64_t)", "fort_actual_pointer = c_null_ptr",
                      f"if (all({extents} > 0)) fort_actual_pointer = c_loc({name})",
                      f"fort_actual_layout = fort_scope_layout({binding.rank}, {enum}, {width}_c_size_t, &",
                      f"fort_actual_pointer, c_loc({extents}), c_loc({lowers}), 1_c_int64_t)"]
            source = fact.get("source_sections")
            parameters = ({item.resource: owner.builder.visible(owner.routine, item.resource)
                           for item in source.dependencies if item.kind == "scalar_read"} if source else {})
            code = build_registration(fact, binding.rank, handle, f"fort_actual_defined_{number}",
                f"{number}_c_int64_t", "fort_actual_layout", context=owner.context, status="fort_actual_status",
                on_error=tuple(failure), parameters=parameters,
                logical_lower_bounds={binding.root: tuple(f"lbound({name},{axis},kind=c_int64_t)"
                                                          for axis in range(1, binding.rank+1))})
            spec += code.specification
            lines += [*code.prepare, "endif"]
        lines += ["if (fort_actual_status /= FORT_SCOPE_OK) then", *owner.close(), "endif", "endif"]
        text = text.replace("!" + key, "\n".join([block + ": block", *imports, *spec, *lines, "end block " + block]))
    return text
