"""Emit a source-issued complete team on the ordinary residency transaction."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.resources import files
from typing import ClassVar

from compiler.frontend.source_device_tasks import source_device_task
from compiler.ir import CompilationError
from compiler.scopes.call_registration import OriginalActual, register_original_actuals
from compiler.scopes.segments import fortran_lines, statement_span


def _provider(proof, pricing=None, placement=None, *, transport="direct"):
    from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
    from compiler.frontend.line_snapshot_team import OriginalLineSnapshotTeamProof

    if type(transport) is not str or transport not in {"direct", "queued_single", "queued_team"}:
        raise CompilationError("unknown checked source transport")

    line_team = type(proof.team) is OriginalLineSnapshotTeamProof
    if line_team and (transport != "direct" or pricing is not None or placement is not None):
        raise CompilationError("line source tasks require unpriced direct thread transport")

    if transport == "queued_single":
        from compiler.scopes.queued_source_binding import checked_provider

        if pricing is not None or placement is not None:
            raise CompilationError("queued source transport cannot consume direct pricing or placement")
        return checked_provider(emit_tile_task_provider(proof.team, readable_scalars=proof,
                                                       access_envelopes=True, queued_single=True))
    if transport == "queued_team":
        from compiler.scopes.queued_team_source_binding import checked_provider

        if pricing is not None or placement is not None:
            raise CompilationError("queued whole-team transport cannot consume direct pricing or placement")
        return checked_provider(emit_tile_task_provider(proof.team, readable_scalars=proof,
                                                       access_envelopes=True, queued_team=True))
    if transport != "direct":
        raise CompilationError("unknown checked source transport")
    return emit_tile_task_provider(proof.team, readable_scalars=proof, pricing=pricing, placement=placement,
                                   mapping="thread" if line_team else "cooperative", access_envelopes=pricing is None)


def _provider_record(provider):
    return {"entry": provider.entry, "module": provider.module,
            "cuda_source_sha256": sha256(provider.cuda_source.encode()).hexdigest(),
            "fortran_source_sha256": sha256(provider.fortran_source.encode()).hexdigest(),
            "metadata": provider.public()}


@dataclass(frozen=True)
class SourceDevicePatch:
    proof: object
    nodes: tuple
    registration: tuple
    first: int
    last: int
    pricing: object = None
    placement: object = None
    _transport: ClassVar[str] = "direct"

    def __post_init__(self):
        if type(self._transport) is not str or self._transport not in {"direct", "queued_single", "queued_team"}:
            raise CompilationError("checked source task requires its exact transport")
        if self._transport != "direct" and (self.pricing is not None or self.placement is not None):
            raise CompilationError("queued source task cannot consume direct prices or placement")

    def public(self, member):
        self.proof.validate(member.builder.analysis, member.routine, self.nodes)
        binding = {
            "owner": member.owner.routine.qualified,
            "canonical_resources": {binding.root: member.canonical(binding).root for binding, _ in self.proof.fields},
            "registration_slots": {binding.root: member.owner.resources[member.canonical(binding).root][1]
                                   for binding, _ in self.proof.fields},
            "provider": _provider_record(_provider(self.proof, self.pricing, self.placement, transport=self._transport))}
        from compiler.scopes.source_task_bindings import call_projection

        projection = call_projection(member, self)
        if projection is not None:
            binding["call_projection"] = projection
        if self._transport == "queued_single":
            from compiler.scopes.queued_source_binding import execution_record

            return execution_record(self.proof, binding)
        if self._transport == "queued_team":
            from compiler.scopes.queued_team_source_binding import execution_record

            return execution_record(self.proof, binding)
        if self.pricing is None:
            record = self.proof.public()
            record["source_proof_identity"] = record.pop("identity")
            record["execution_binding"] = binding
        else:
            record = {"version": 2, "kind": "quoted_original_tile_private_team", "source_proof": self.proof.public(),
                      "pricing": self.pricing.public(), "execution_binding": binding}
            if self.pricing.native_live:
                record.update(version=3, kind="quoted_original_tile_private_team_live")
            if self.placement is not None:
                self.placement.validate(member)
                record.update(version=4, kind="scheduled_original_tile_private_team_live",
                              placement=self.placement.public())
        return {"identity": sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), **record}

    def render(self, member):
        from compiler.scopes.provenance import position, source_id

        self.proof.validate(member.builder.analysis, member.routine, self.nodes)
        if (statement_span(self.nodes[0])[0], statement_span(self.nodes[-1])[1]) != (self.first, self.last):
            raise CompilationError("source device task moved from its exact complete original team")
        if self.placement is not None:
            from compiler.scopes.graph import validate_scheduled_source_task

            validate_scheduled_source_task(member, self)
        from compiler.scopes.source_task_bindings import call_projection

        call_projection(member, self)  # Revalidate the exact original forwarding before emission.
        provider = _provider(self.proof, self.pricing, self.placement, transport=self._transport)
        # A speculative child builder can retain a detached outputs mapping.
        # Published execution artifacts belong to the sealed graph owner.
        outputs = member.owner.builder.outputs
        support = files("compiler.emission.common").joinpath("templates", "common_functions.cuh").read_text(encoding="utf-8")
        numeric = files("compiler.runtime").joinpath("numeric.hpp").read_text(encoding="utf-8")
        for name, text in ((provider.entry + ".cu", provider.cuda_source),
                           (provider.module + ".f90", provider.fortran_source),
                           ("common_functions.cuh", support), ("numeric.hpp", numeric)):
            old = outputs.setdefault(name, text)
            if old != text:
                raise CompilationError("source task provider artifact identity collision")
        if self.pricing is not None:
            record = self.pricing.public()
            text = json.dumps(self.pricing.profile(), sort_keys=True, separators=(",", ":"), allow_nan=False)
            if sha256(text.encode()).hexdigest() != record["profile_identity"]:
                raise CompilationError("source pricing canonical profile changed")
            old = outputs.setdefault(record["profile_artifact"], text)
            if old != text:
                raise CompilationError("source pricing profile artifact identity collision")
        block = "fort_source_task_" + self.proof.identity[:12]
        queued = self._transport in {"queued_single", "queued_team"}
        live = self.pricing is not None and self.pricing.native_live
        native_prepare = provider.public().get("native_prepare_entry")
        imports = ["use iso_c_binding", "use fort_scoped_memory", f"use {provider.module}, only: {provider.entry}" +
                   (", " + native_prepare if live else "")]
        spec = ["integer(c_int) :: fort_source_status", "type(fort_scope_external_report_v1) :: fort_source_report",
                "logical :: fort_source_done, fort_source_can"]
        if queued:
            imports[2] += ", fort_tile_queue_report_v1"
            spec = ["integer(c_int) :: fort_source_status, fort_source_cleanup",
                    "type(fort_tile_queue_report_v1) :: fort_source_report",
                    "character(kind=c_char,len=64),target :: fort_source_error_message",
                    "logical :: fort_source_done, fort_source_can, fort_source_fatal"]
        if live:
            spec += ["integer(c_int64_t) :: fort_source_native_token", "logical :: fort_source_registered"]
        if self.placement is not None:
            spec += ["integer(c_int) :: fort_source_selected"]
        arguments = [member.context, *["fort_buffer_" + str(member.resources[binding.root][1])
                                      for binding, _ in self.proof.fields]]
        snapshots = []
        for index, parameter in enumerate(provider.parameters):
            binding, role = parameter.binding, parameter.role
            actual = member.builder.visible(member.routine, binding.root)
            if role == "scalar":
                # The original scalar remains by reference. Do not introduce
                # a Fortran VALUE conversion or a speculative scalar snapshot.
                arguments.append(actual)
            elif role in {"external_lower", "scratch_lower", "scratch_upper"}:
                variable = "fort_source_descriptor_" + str(index)
                spec.append(f"integer({parameter.kind}) :: {variable}")
                inquiry = "ubound" if role == "scratch_upper" else "lbound"
                snapshots.append(f"{variable} = {inquiry}({actual},{parameter.axis},kind={parameter.kind})")
                arguments.append(variable)
            else:
                raise CompilationError("source task provider has an unknown original parameter role")
        if self.placement is not None:
            arguments.append("fort_source_selected")
        arguments.append("fort_source_report")
        operation = source_id(member.builder, "source_device_task", span=(self.first, self.last),
                              proof_identity=self.proof.identity)
        implementation = source_id(member.builder, "implementation", source_task_provider=_provider_record(provider))
        lines = ["fort_source_done = .false.", f"if ({member.enabled}) then",
                 "fort_source_can = fort_scope_numerical_environment_supported() /= 0",
                 "if (fort_source_can) then"]
        if live:
            lines.insert(1, "fort_source_registered = .false.")
        if self.placement is not None:
            lines.insert(1, "fort_source_selected = 0_c_int")
        from compiler.frontend.line_snapshot_team import OriginalLineSnapshotTeamProof

        line_team = type(self.proof.team) is OriginalLineSnapshotTeamProof
        domain = self.proof.team.public()["transverse_domain" if line_team else "domain"]
        domain_guards = 0
        for axis in domain:
            # Keep the original nested activation order. Fortran .AND. is not
            # a guard against touching a later-axis undefined descriptor/value.
            if not line_team:
                lines += [f"if (({axis['width']}) > 0) then"]
                domain_guards += 1
            lines += [f"if (({axis['lower']}) <= ({axis['upper']})) then"]
            domain_guards += 1
        for binding, _ in self.proof.fields:
            name = member.builder.visible(member.routine, binding.root)
            if "allocatable" in binding.attributes:
                lines.append(f"if (fort_source_can) fort_source_can = allocated({name})")
            lines.append(f"if (fort_source_can) fort_source_can = is_contiguous({name})")
            lines += [f"if (fort_source_can) fort_source_can = {condition}"
                      for condition in member.builder.original_bound_conditions(name, binding.rank)]
        lines += ["if (fort_source_can) then", *self.registration, f"if ({member.context} /= 0) then",
                  *(["fort_source_registered = .true."] if live else []), *snapshots,
                  *position(member.builder, context=member.context, owner=member.trace_owner,
                            operation=operation, implementation=implementation),
                  "fort_source_report = " + ("fort_tile_queue_report_v1(0_c_int32_t,0_c_int32_t,0_c_int32_t,0_c_int32_t)" if queued else "fort_scope_external_report_v1()"),
                  "fort_source_status = " + provider.entry + "(" + ",".join(arguments) + ")"]
        if queued:
            from compiler.scopes.queued_source_binding import report_lines

            lines += [*report_lines(member), "endif", "endif"]
        else:
            lines += ["if (fort_source_report%version /= 1 .or. fort_source_report%status /= fort_source_status) &",
                      "error stop 'source task returned an invalid transaction report'"]
        if not queued and self.placement is not None:
            lines += ["if (fort_source_selected /= 0 .and. fort_source_selected /= 1) &",
                      "error stop 'source graph returned an invalid backend'",
                      "if (fort_source_selected == 1) then",
                      "if (fort_source_status /= FORT_SCOPE_OK) error stop 'source GPU selection failed; replay prohibited'"]
        elif not queued:
            lines += ["if (fort_source_status == FORT_SCOPE_OK) then"]
        if not queued:
            lines += [
                  "if (fort_source_report%started /= 1 .or. fort_source_report%committed /= 1) &",
                  "error stop 'source task success lacks committed work'", "fort_source_done = .true.", "else",
                  "if (fort_source_report%started /= 0 .or. fort_source_report%committed /= 0) &",
                  "error stop 'source task attempted work; replay prohibited'",
                  "if (" + ("fort_source_status /= FORT_SCOPE_OK .and. " if self.placement is not None else "") +
                  "fort_source_status /= FORT_SCOPE_ARGUMENT .and. fort_source_status /= FORT_SCOPE_RESOURCE .and. &",
                  "fort_source_status /= FORT_SCOPE_BOUNDARY .and. fort_source_status /= FORT_SCOPE_UNINITIALIZED .and. &",
                  "fort_source_status /= FORT_SCOPE_PLAN_MISMATCH) error stop 'source task failed fatally'",
                  "endif", "endif", "endif"]
        lines += ["endif"] * domain_guards
        lines += ["endif"]
        if live:
            native_args = [member.context, *["fort_buffer_" + str(member.resources[binding.root][1])
                                          for binding, _ in self.proof.fields], "fort_source_native_token"]
            native_condition = "fort_source_selected == 0 .and. " if self.placement is not None else ".not. fort_source_done .and. "
            lines += ["if (" + native_condition + "fort_source_registered) then",
                      "fort_source_native_token = 0_c_int64_t",
                      "fort_source_status = " + native_prepare + "(" + ",".join(native_args) + ")",
                      "if (fort_source_status == FORT_SCOPE_OK) then",
                      f"fort_source_status = fort_scope_source_host_start_v1({member.context},fort_source_native_token)",
                      "if (fort_source_status /= FORT_SCOPE_OK) error stop 'source native start failed; replay prohibited'",
                      *member.original(self.nodes).splitlines(),
                      f"fort_source_status = fort_scope_source_host_complete_v1({member.context},fort_source_native_token)",
                      "if (fort_source_status /= FORT_SCOPE_OK) error stop 'source native completion failed; replay prohibited'",
                      "fort_source_done = .true.", "else",
                      "if (fort_source_status /= FORT_SCOPE_ARGUMENT .and. fort_source_status /= FORT_SCOPE_RESOURCE .and. &",
                      "fort_source_status /= FORT_SCOPE_BOUNDARY .and. fort_source_status /= FORT_SCOPE_UNINITIALIZED .and. &",
                      "fort_source_status /= FORT_SCOPE_PLAN_MISMATCH) error stop 'source native preparation failed fatally'",
                      "endif", "endif"]
        lines += ["if (.not. fort_source_done) then",
                  *member.close(prework=True, reason="source task declined before complete original team",
                                span=(self.first, self.last)), "endif", "endif", "if (.not. fort_source_done) then",
                  *member.original(self.nodes).splitlines(), "endif"]
        emitted = [block + ": block", *imports, *spec, *lines, "end block " + block]
        if member.control_guard is not None:
            emitted = [f"if ({member.control_guard}) then", *emitted, "else",
                       *member.original(self.nodes).splitlines(), "endif"]
        return "\n".join(fortran_lines(emitted)) + "\n"


@dataclass(frozen=True)
class _QueuedSourceDevicePatch(SourceDevicePatch):
    _transport: str = field(default="queued_single", init=False)


@dataclass(frozen=True)
class _QueuedTeamSourceDevicePatch(SourceDevicePatch):
    _transport: str = field(default="queued_team", init=False)


def admit_source_device_task(member, nodes):
    return _admit_source_device_task(member, nodes)


def _admit_source_device_task(member, nodes, *, transport="direct"):
    """Private checked-emission hook; regular policy keeps its existing backend."""
    config = member.builder.config
    if config.scope_execution != "graph" or config.policy not in {"sections", "auto"}:
        return False
    # Source quotes snapshot the domain's AUTO copy policy and its calibrated
    # physical-copy costs through the ordinary coherence simulator. Requiring
    # explicit direct transfers here silently removes otherwise proved teams
    # before that runtime choice can be evaluated.
    if config.policy == "auto" and (config.profile is None or config.scope_transfers not in {"direct", "coalesced", "auto"}):
        return False
    originals = member.builder.inline.original_selection(tuple(nodes))
    proof = source_device_task(member.builder.analysis, member.routine, originals, builder=member.builder)
    pricing = None
    if config.policy == "auto":
        from compiler.frontend.line_snapshot_team import OriginalLineSnapshotTeamProof

        if type(proof.team) is OriginalLineSnapshotTeamProof:
            return False  # The new line source work has no qualified cost model.
        from compiler.offload.source_task_pricing import source_task_pricing

        try:
            pricing = source_task_pricing(proof.team, config.profile, member.builder.runtime["runtime_id"],
                                          host_threads=config.host_threads,
                                          native_live="native_transactions" in config.profile.get("source_tasks", {}))
        except ValueError:
            return False  # Unsupported/missing prices retain the original native team.
    _provider(proof, pricing, transport=transport)  # Reject an unsupported numerical/ABI provider before registering storage.
    roots = {member.canonical(binding).root for binding, _ in proof.fields}
    if roots & (member.host_read_roots() | member.owner.scalar_element_resources.keys()):
        raise CompilationError("source task field conflicts with an opaque original storage reservation")
    # Validate every capture before registering any of them. A declined team
    # must not leave a partially classified canonical resource inventory.
    for binding, _written in proof.fields:
        member.builder.capture(binding)
    registration = register_original_actuals(member, tuple(OriginalActual(binding,
        member.builder.visible(member.routine, binding.root)) for binding, _ in proof.fields))
    first, last = statement_span(originals[0])[0], statement_span(originals[-1])[1]
    placement = None
    if pricing is not None and pricing.native_live:
        from compiler.frontend.source_placement import source_placement

        placement = source_placement(member, proof, pricing,
                                    f"{member.routine.qualified}:source_device:{len(member.source_device_tasks)}")
    patch_type = ({"direct": SourceDevicePatch, "queued_single": _QueuedSourceDevicePatch,
                   "queued_team": _QueuedTeamSourceDevicePatch})[transport]
    patch = patch_type(proof, originals, tuple(registration), first, last, pricing, placement)
    member.source_device_tasks.append(patch)
    member.patches.append((first, last, patch))
    return True
