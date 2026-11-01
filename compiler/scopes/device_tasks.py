"""Emit checked, reached external calls inside a caller-owned resident domain.

This first slice is an owning serial graph in forced sections mode. It keeps
the original call once as prework fallback and retires the provider before any
domain publication. Opaque provider work is deliberately unpriced.
"""
import json
from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from compiler.frontend.device_tasks import _KINDS, external_device_task
from compiler.ir import CompilationError
from compiler.scopes.call_registration import OriginalActual, register_original_actuals
from compiler.scopes.segments import fortran_lines, statement_span

_FORTRAN_KIND = {"int32": "c_int32_t", "int64": "c_int64_t", "size": "c_size_t", "real64": "c_double"}


def _pointer(patch):
    return "fort_provider_life%provider" if patch.provider_lifetime else "fort_provider_" + patch.proof.identity[:12]


def _words(patch):
    return "fort_provider_life%words" if patch.provider_lifetime else "fort_provider_words_" + patch.proof.identity[:12]


def _word_count(patch):
    public = patch.proof.public()
    return (sum(len(item["values"]) for item in public["contract"]["configuration"])
            + sum(item["count"] for item in public["provider"].get("metadata_query", {}).get("configuration_components", ())))


def owner_providers(member):
    """Every child that can retire a domain borrows all its provider slots."""
    return tuple(task for child in member.owner.members.values() for task in child.device_tasks
                 if task.proof.public()["version"] == 2)


def forwarded_control(member):
    return tuple(name for task in owner_providers(member) for name in (_pointer(task), _words(task)))


def validate_provider_mappings(owner):
    """The first v2 binding describes one canonical caller projection.

    Ordinary numeric helpers remain parameterized. A provider-bearing member
    and its checked callers must retain the represented original association;
    a different actual projection needs an explicit per-call task binding.
    """
    from compiler.scopes.source_task_bindings import call_projection

    pending = [member.routine.qualified for member in owner.members.values()
               if any(call_projection(member, task) is None for task in member.source_device_tasks)
               or any(task.proof.public()["version"] == 2 for task in member.device_tasks)]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        member = owner.members[name]
        expected = {root: binding.root for root, binding in member.canonical_bindings.items()}
        for call in getattr(member, "call_sites", ()):
            if call.get("resource_mappings") != expected:
                raise CompilationError("external provider member requires one canonical original association")
            if call["caller"] != owner.routine.qualified:
                pending.append(call["caller"])


def finish_controls(member, text):
    changed = False
    for marker, (callee, context) in getattr(member, "external_control_placeholders", {}).items():
        if marker in text:
            values = [f"{callee.context} = {context}", *[f"{name} = {name}" for name in forwarded_control(member)]]
            text = text.replace(marker, ", ".join(values))
            changed = True
    return "\n".join(fortran_lines(text.splitlines())) + "\n" if changed else text


def child_control(member):
    providers = owner_providers(member)
    if not providers:
        return [], [], []
    return (["use iso_c_binding, only: fort_external_pointer => c_ptr"],
            [line for task in providers for line in (
                f"type(fort_external_pointer), intent(inout) :: {_pointer(task)}",
                f"integer({member.kind}), intent(inout) :: {_words(task)}({max(1, _word_count(task))})")],
            list(forwarded_control(member)))


def control(member):
    """Invocation-local provider ownership; never SAVE or a global cache."""
    from compiler.scopes.provider_lifetime import control as lifetime_control
    lifetime = lifetime_control(member)
    if lifetime is not None:
        return lifetime
    providers = (*owner_providers(member), *(task for task in member.device_tasks if task.proof.public()["version"] == 1))
    if not providers:
        return [], [], []
    imports = ["use iso_c_binding, only: fort_external_pointer => c_ptr, fort_external_null => c_null_ptr"]
    specification, initialize = [], []
    for patch in providers:
        specification += [f"type(fort_external_pointer) :: {_pointer(patch)}",
                          f"integer({member.kind}) :: {_words(patch)}({max(1, _word_count(patch))})"]
        initialize += [f"{_pointer(patch)} = fort_external_null"]
    return imports, specification, initialize


def before_close(member):
    result = []
    providers = (*owner_providers(member), *(task for task in member.device_tasks if task.proof.public()["version"] == 1))
    for patch in providers:
        interface = patch.proof.public()["provider"]["fortran"]
        result += ["block", "use iso_c_binding, only: c_int, c_associated",
                   f"use {interface['module']}, only: fort_external_destroy => {interface['destroy']}",
                   "integer(c_int) :: fort_external_destroy_status",
                   f"if (c_associated({_pointer(patch)})) then",
                   f"fort_external_destroy_status = fort_external_destroy({_pointer(patch)})",
                   "if (fort_external_destroy_status /= 0) error stop 'external provider destruction failed'",
                   "endif", "end block"]
    return result


@dataclass(frozen=True)
class ExternalDevicePatch:
    proof: object
    node: object
    registration: tuple
    first: int
    last: int
    provider_lifetime: ClassVar[bool] = False

    def public(self, member):
        self.proof.validate(member.builder.analysis, member.routine, self.node)
        public = self.proof.public()
        if public["version"] == 1:
            return public
        public["source_proof_identity"] = public.pop("identity")
        public["execution_binding"] = {
            "owner": member.owner.routine.qualified,
            "canonical_resources": {formal: member.canonical(binding).root for formal, binding in zip(
                public["canonical_resources"], self.proof.bindings, strict=True)},
            "registration_slots": {formal: member.owner.resources[member.canonical(binding).root][1] for formal, binding in zip(
                public["canonical_resources"], self.proof.bindings, strict=True)},
            "provider_slot": next(index for index, task in enumerate(owner_providers(member), 1) if task is self),
            "configuration_words": _word_count(self),
            "ownership": "owning invocation controls forwarded by reference through every checked ENTRY"}
        if self.provider_lifetime:
            public["execution_binding"]["provider_only_lifetime"] = member.builder.provider_only_lifetime.identity
        return {"identity": sha256(json.dumps(public, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                **public}

    def render(self, member):
        self.proof.validate(member.builder.analysis, member.routine, self.node)
        if statement_span(self.node) != (self.first, self.last):
            raise CompilationError("external device CALL moved from its original source position")
        public = self.proof.public()
        contract, interface = public["contract"], public["provider"]["fortran"]
        names = {item.formal: str(item.actual) for item in self.proof._resolved.associations}
        bindings = {item.formal: item.binding for item in self.proof._resolved.associations
                    if item.binding is not None and item.binding.rank}
        handles = ["fort_buffer_" + str(member.resources[binding.root][1]) for binding in self.proof.bindings]
        pointer, words = _pointer(self), _words(self)
        block = "fort_external_" + self.proof.identity[:12]
        imports = ["use iso_c_binding", "use fort_scoped_memory",
                   f"use {interface['module']}, only: fort_external_config_type => {interface['config_type']}, &",
                   f"fort_external_create => {interface['create']}, fort_external_execute => {interface['execute']}, &",
                   f"fort_external_destroy => {interface['destroy']}"]
        query = public["provider"].get("metadata_query")
        if query is not None:
            imports += [f"use {query['module']}, only: fort_external_metadata => {query['procedure']}"]
        spec = ["type(fort_external_config_type) :: fort_external_config",
                "type(fort_scope_external_report_v1) :: fort_external_report",
                "integer(c_int) :: fort_external_status", "integer(c_int64_t) :: fort_external_domain",
                f"integer(c_int64_t) :: fort_external_words({max(1, _word_count(self))})",
                "logical :: fort_external_done, fort_external_can, fort_external_changed"]
        from compiler.scopes.provenance import position, source_id
        operation = source_id(member.builder, "external_device_task", span=(self.first, self.last),
                              proof_identity=self.proof.identity, callee=contract["source"]["callee"])
        implementation = source_id(member.builder, "implementation", provider=public["provider"],
                                   qualification_identity=public["provider"]["numerical_contract"]["identity"])
        lines = ["fort_external_done = .false.", f"if ({member.enabled}) then",
                 "fort_external_can = fort_scope_numerical_environment_supported() /= 0"]
        # Fortran logical expressions need not short circuit: ALLOCATED guards
        # therefore precede every descriptor inquiry in distinct statements.
        for formal, binding in bindings.items():
            name = names[formal]
            if "allocatable" in binding.attributes:
                lines += [f"if (fort_external_can) fort_external_can = allocated({name})"]
            lines += [f"if (fort_external_can) fort_external_can = is_contiguous({name})"]
            lines += [f"if (fort_external_can) fort_external_can = {condition}"
                      for condition in member.builder.original_bound_conditions(name, binding.rank)]
        lines += ["if (fort_external_can) then", *self.registration, f"if ({member.context} /= 0) then",
                  *position(member.builder, context=member.context, owner=member.trace_owner,
                            operation=operation, implementation=implementation),
                  "fort_external_config = fort_external_config_type()"]
        temporary_count = 0

        def expression(value):
            if type(value) is int:
                # Negative I64_MIN cannot be parsed as a positive literal
                # followed by unary minus in a standard Fortran expression.
                return ("(-9223372036854775807_c_int64_t-1_c_int64_t)" if value == -(1 << 63)
                        else f"{value}_c_int64_t")
            name = names[value["formal"]]
            descriptor = f"{value['descriptor']}({name},{value['axis']},kind=c_int64_t)"
            return f"({descriptor} + ({value['offset']}_c_int64_t))"

        def checked(value, lower=None, upper=None):
            nonlocal temporary_count
            temporary_count += 1
            name = "fort_external_value_" + str(temporary_count)
            spec.append(f"integer(c_int64_t) :: {name}")
            lines.append(f"{name} = {expression(value)}")
            if lower is not None:
                lines.append(f"if ({name} < {expression(lower)}) fort_external_can = .false.")
            if upper is not None:
                bound = expression(upper) if type(upper) is int else upper
                lines.append(f"if ({name} > {bound}) fort_external_can = .false.")
            return name

        word = 0
        for assignment in contract["configuration"]:
            values = []
            for value in assignment["values"]:
                word += 1
                if assignment["kind"] == "real64":
                    literal = repr(float(value)) + "_c_double"
                    values.append(literal)
                    lines.append(f"fort_external_words({word}) = transfer({literal},0_c_int64_t)")
                else:
                    lower, upper = _KINDS[assignment["kind"]]
                    temporary = checked(value, lower, upper)
                    values.append(f"int({temporary},{_FORTRAN_KIND[assignment['kind']]})")
                    lines.append(f"fort_external_words({word}) = {temporary}")
            rhs = values[0] if len(values) == 1 else "[" + ",".join(values) + "]"
            lines.append(f"if (fort_external_can) fort_external_config%{assignment['component']} = {rhs}")
        # Bounds are physical root coordinates, independent of Fortran dummy
        # rebasing. These checks grant no definitions and touch no payload.
        for resource in contract["resources"]:
            name = names[resource["formal"]]
            for effect in resource["effects"]:
                for axis, (lo, hi) in enumerate(zip(effect["lower"], effect["upper"], strict=True), 1):
                    lower = checked(lo, 0)
                    upper = checked(hi, 0, f"size({name},{axis},kind=c_int64_t)")
                    lines += [f"if ({upper} < {lower}) fort_external_can = .false."]
        if query is not None:
            lines += ["if (fort_external_can) then",
                      "fort_external_status = fort_external_metadata(fort_external_config," +
                      ",".join(names[formal] for formal in query["opaque_formals"]) + ")",
                      "fort_external_can = fort_external_status == FORT_SCOPE_OK", "endif"]
            for component in query["configuration_components"]:
                for index in range(1, component["count"] + 1):
                    word += 1
                    value = "fort_external_config%" + component["component"]
                    if component["count"] > 1:
                        value += f"({index})"
                    converted = (f"transfer({value},0_c_int64_t)" if component["kind"] == "real64"
                                 else f"int({value},c_int64_t)")
                    lines.append(f"if (fort_external_can) fort_external_words({word}) = {converted}")
        # Only a separately qualified must-read contract can admit missing
        # entry values. Provider may-read boxes and INTENT are insufficient.
        from compiler.frontend.external_entry_reads import external_entry_read_requirements
        entry_reads = external_entry_read_requirements(self.proof)
        entry_code = []
        if entry_reads is not None:
            entry_reads.validate(member.builder.analysis, member.routine, self.node)
            required = entry_reads.public()["qualification_record"]
            components = {item["component"]: item for item in query["configuration_components"]}
            for guard in required["layout_equalities"]:
                component = "fort_external_config%" + guard["component"]
                if components[guard["component"]]["count"] > 1:
                    component += f"({guard['index']})"
                value = checked(guard["descriptor"], 0)
                lines.append(f"if (fort_external_can) fort_external_can = int({component},c_int64_t) == {value}")
            for index, requirement in enumerate(required["required_reads"], 1):
                rank = len(requirement["lower"])
                lower = f"fort_entry_read_lower_{index}"
                upper = f"fort_entry_read_upper_{index}"
                section = f"fort_entry_read_section_{index}"
                spec += [f"integer(c_size_t), target :: {lower}({rank}), {upper}({rank})",
                         f"type(fort_scope_section), target :: {section}"]
                lows = [checked(value, 0) for value in requirement["lower"]]
                highs = [checked(value, 0, f"size({names[requirement['formal']]},{axis},kind=c_int64_t)")
                         for axis, value in enumerate(requirement["upper"], 1)]
                for low, high in zip(lows, highs, strict=True):
                    lines.append(f"if ({high} < {low}) fort_external_can = .false.")
                handle = "fort_buffer_" + str(member.resources[bindings[requirement["formal"]].root][1])
                entry_code += [lower + " = [" + ",".join(f"int({value},c_size_t)" for value in lows) + "]",
                               upper + " = [" + ",".join(f"int({value},c_size_t)" for value in highs) + "]",
                               f"{section} = fort_scope_section(c_loc({lower}),c_loc({upper}))",
                               f"fort_external_status = fort_scope_admit_source_reads_v1({member.context},{handle},c_loc({section}),1_c_size_t)",
                               "if (fort_external_status /= FORT_SCOPE_OK) error stop 'external entry-read admission failed'"]
        lines += ["if (fort_external_can) then", "fort_external_status = FORT_SCOPE_OK",
                  "fort_external_changed = .false.", f"if (c_associated({pointer})) then"]
        if word:
            lines.append(f"fort_external_changed = any(fort_external_words(1:{word}) /= {words}(1:{word}))")
        lines += ["if (fort_external_changed) then", f"fort_external_status = fort_external_destroy({pointer})",
                  "if (fort_external_status /= FORT_SCOPE_OK) error stop 'external provider reconfiguration failed'",
                  "endif", "endif", f"if (.not. c_associated({pointer})) then",
                  f"fort_external_status = fort_scope_domain_get_v1({member.context},fort_external_domain)",
                  "if (fort_external_status /= FORT_SCOPE_OK .or. fort_external_domain == 0) &",
                  "error stop 'external provider requires its checked resident domain'",
                  f"fort_external_status = fort_external_create(fort_external_domain,fort_external_config,{pointer})"]
        if word:
            lines += [f"if (fort_external_status == FORT_SCOPE_OK) {words}(1:{word}) = fort_external_words(1:{word})"]
        lines += ["endif", "if (fort_external_status == FORT_SCOPE_OK) then",
                  f"if (.not. c_associated({pointer})) error stop 'external provider returned an empty handle'",
                  "fort_external_report = fort_scope_external_report_v1()", *entry_code]
        for formal in public["source_effects"].get("entry_definition_kills", ()):
            handle = "fort_buffer_" + str(member.resources[bindings[formal].root][1])
            lines += [f"fort_external_status = fort_scope_forget_definition({member.context},{handle})",
                      "if (fort_external_status /= FORT_SCOPE_OK) error stop 'external original OUT definition failed'"]
        lines += [
                  f"fort_external_status = fort_external_execute({member.context},{pointer}," +
                  ",".join(handles) + ",fort_external_report)",
                  "if (fort_external_report%version /= 1 .or. fort_external_report%status /= fort_external_status) &",
                  "error stop 'external provider returned an invalid transaction report'",
                  "if (fort_external_status == FORT_SCOPE_OK) then",
                  "if (fort_external_report%started /= 1 .or. fort_external_report%committed /= 1) &",
                  "error stop 'external provider success lacks committed submission'",
                  "fort_external_done = .true."]
        if public["version"] == 2:
            lines += [f"fort_external_status = fort_scope_wait({member.context})",
                      "if (fort_external_status /= FORT_SCOPE_OK) error stop 'external completion failed; replay prohibited'"]
        lines += ["else",
                  "if (fort_external_report%started /= 0 .or. fort_external_report%committed /= 0) &",
                  "error stop 'external device work failed; replay prohibited'", "endif", "endif",
                  "if (fort_external_status /= FORT_SCOPE_OK) then",
                  "if (fort_external_status /= FORT_SCOPE_ARGUMENT .and. fort_external_status /= FORT_SCOPE_RESOURCE .and. &",
                  "fort_external_status /= FORT_SCOPE_BOUNDARY .and. fort_external_status /= FORT_SCOPE_UNINITIALIZED .and. &",
                  "fort_external_status /= FORT_SCOPE_PLAN_MISMATCH) error stop 'external provider failure is not recoverable'",
                  "endif", "endif", "endif", "endif", "if (.not. fort_external_done) then",
                  *member.close(prework=True, reason="external provider unavailable at original call", span=(self.first, self.last)),
                  "endif", "endif", "if (.not. fort_external_done) then",
                  *member.original((self.node,)).splitlines(), "endif"]
        emitted = [block + ": block", *imports, *spec, *lines, "end block " + block]
        if member.control_guard is not None:
            emitted = [f"if ({member.control_guard}) then", *emitted, "else",
                       *member.original((self.node,)).splitlines(), "endif"]
        return "\n".join(fortran_lines(emitted)) + "\n"


@dataclass(frozen=True)
class _ProviderLifetimePatch(ExternalDevicePatch):
    provider_lifetime: bool = True


def admit_external_device_task(member, node):
    proof = external_device_task(member.builder.analysis, member.routine, node)
    if proof is None:
        return False
    if (member.builder.config.scope_execution != "graph" or member.builder.config.policy != "sections"
            or proof.public()["version"] == 1 and (member is not member.owner or member.control_guard is not None)):
        raise CompilationError("external device first slice requires an owning serial forced-sections graph")
    associations = tuple(item for item in proof._resolved.associations if item.binding is not None and item.binding.rank)
    registration = register_original_actuals(member, tuple(OriginalActual(binding, str(item.actual))
        for binding, item in zip(proof.bindings, associations, strict=True)))
    first, last = statement_span(node)
    patch_type = (_ProviderLifetimePatch if getattr(member.builder, "provider_only_lifetime", None) is not None
                  else ExternalDevicePatch)
    patch = patch_type(proof, node, tuple(registration), first, last)
    member.device_tasks.append(patch)
    member.patches.append((first, last, patch))
    return True
