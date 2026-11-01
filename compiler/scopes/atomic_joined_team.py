"""An indivisible CPU/GPU alternative for an original conditional NOWAIT team."""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files

from compiler.emission.cuda.atomic_team_provider import emit_atomic_team_provider
from compiler.frontend.atomic_worksharing_team import atomic_worksharing_alternatives, prove_atomic_worksharing_team
from compiler.frontend.numerical_effects import prove_numerical_effects
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.frontend.reached_read_requirements import prove_reached_read_requirements
from compiler.ir import CompilationError
from compiler.scopes.call_registration import OriginalActual, register_original_actuals
from compiler.scopes.segments import fortran_lines, statement_span
from compiler.scopes.source_device_tasks import SourceDevicePatch


@dataclass(frozen=True)
class AtomicAlternative:
    source: object
    prepared: object
    effects: object
    reads: object
    registration: tuple

    def provider(self):
        return emit_atomic_team_provider(self.source, self.prepared, self.effects, self.reads)


def _provider_record(provider):
    return {"entry": provider.entry, "module": provider.module,
            "cuda_source_sha256": sha256(provider.cuda_source.encode()).hexdigest(),
            "fortran_source_sha256": sha256(provider.fortran_source.encode()).hexdigest(),
            "metadata": provider.public()}


@dataclass(frozen=True)
class AtomicJoinedTeamPatch(SourceDevicePatch):
    alternatives: tuple = ()
    predicate_registration: tuple = ()

    def public(self, member):
        self.proof.validate(member.builder.analysis, member.routine, self.nodes)
        record = {"version": 3 if self.proof.public().get("conditional_affine_context_required") else 2,
            "kind": "original_atomic_conditional_joined_team",
            "source_proof": self.proof.public(),
            "alternatives": [{"source": item.source.public(), "preparation": item.prepared.public(),
                "physical_effects": item.effects.public(), "source_reads": item.reads.public(),
                "provider": _provider_record(item.provider())} for item in self.alternatives if item is not None],
            "execution_binding": {"owner": member.owner.routine.qualified,
                "canonical_resources": {binding.root: member.canonical(binding).root for binding, _ in self.proof.fields},
                "registration_slots": {binding.root: member.owner.resources[member.canonical(binding).root][1]
                                       for binding, _ in self.proof.fields}},
            "definitions_authorized": False, "placement_authorized": False, "motion_authorized": False,
            "pricing": "unpriced; explicit sections only"}
        from compiler.scopes.source_task_bindings import call_projection

        projection = call_projection(member, self)
        if projection is not None:
            record["version"] = 4
            record["execution_binding"]["call_projection"] = projection
        return {"identity": sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), **record}

    def render(self, member):
        from compiler.scopes.provenance import position, source_id
        from compiler.scopes.source_reads import _build_read_sections, admit_prepared_source_reads

        # Speculative child builders can retain detached output mappings;
        # providers must be published with their owning compilation artifacts.
        outputs = member.owner.builder.outputs
        self.proof.validate(member.builder.analysis, member.routine, self.nodes)
        from compiler.scopes.source_task_bindings import call_projection

        call_projection(member, self)
        if (statement_span(self.nodes[0])[0], statement_span(self.nodes[-1])[1]) != (self.first, self.last):
            raise CompilationError("atomic team moved from its complete original source span")
        public = self.proof.public()
        by_path = {tuple(path): index for index, path in enumerate(public["alternatives"])}
        predicates = self.proof.predicates

        imports = ["use iso_c_binding", "use fort_scoped_memory"]
        specification = ["integer :: fort_atomic_variant", "integer(c_int) :: fort_atomic_status, fort_status",
                         "logical :: fort_atomic_done, fort_atomic_can",
                         "type(fort_scope_external_report_v1) :: fort_atomic_report"]
        selection_block = "fort_atomic_select"
        conditions = {}

        def condition(index):
            from compiler.scopes.access import build_native_accesses, build_native_acquisition

            if index in conditions:
                return conditions[index]
            header = predicates[index][0]
            sections = self.proof.predicate_sections(index)
            if not sections.resources:
                return [], "IF (" + str(header.items[0]) + ") THEN"
            handles = {resource.resource: "fort_buffer_" + str(member.resources[resource.resource][1])
                       for resource in sections.resources}
            accesses = build_native_accesses(sections, handles, "fort_ap_" + str(index),
                on_error=("exit " + selection_block,))
            acquisition = build_native_acquisition(
                ((access.handle, access.access_name) for access in accesses), "fort_ap_" + str(index))
            temporary = "fort_atomic_predicate_" + str(index)
            specification.extend(["logical :: " + temporary,
                *[line for access in accesses for line in access.specification], *acquisition.specification])
            prepare = [*self.predicate_registration[index],
                f"if (.not. {member.enabled} .or. {member.context}==0) exit " + selection_block,
                *[line for access in accesses for line in access.prepare],
                *admit_prepared_source_reads(accesses, on_error=(
                    "if (fort_status/=FORT_SCOPE_ARGUMENT .and. fort_status/=FORT_SCOPE_RESOURCE .and. &",
                    "    fort_status/=FORT_SCOPE_BOUNDARY .and. fort_status/=FORT_SCOPE_UNINITIALIZED .and. &",
                    "    fort_status/=FORT_SCOPE_PLAN_MISMATCH) &",
                    "error stop 'atomic predicate source admission failed fatally'",
                    "exit " + selection_block)),
                *acquisition.prepare,
                "fort_status = " + acquisition.expression,
                "if (fort_status/=FORT_SCOPE_OK .and. fort_status/=FORT_SCOPE_ARGUMENT .and. &",
                "    fort_status/=FORT_SCOPE_RESOURCE .and. fort_status/=FORT_SCOPE_BOUNDARY .and. &",
                "    fort_status/=FORT_SCOPE_UNINITIALIZED .and. fort_status/=FORT_SCOPE_PLAN_MISMATCH) &",
                "error stop 'atomic predicate acquisition failed fatally'",
                "if (fort_status /= FORT_SCOPE_OK) exit " + selection_block,
                temporary + " = " + str(header.items[0])]
            for access in accesses:
                prepare += [f"fort_status=fort_scope_host_end(fort_context,{access.handle})",
                            "if (fort_status/=FORT_SCOPE_OK) error stop 'atomic predicate completion failed'"]
            conditions[index] = prepare, "if (" + temporary + ") then"
            return conditions[index]

        def select(items, prefix=()):
            if not items:
                return ["fort_atomic_variant = " + str(by_path[prefix])]
            first, rest = items[0], items[1:]
            if "unit" in first:
                return select(rest, (*prefix, first["unit"]))
            def branches(remaining):
                if not remaining:
                    return []
                branch, *following = remaining
                index = branch["predicate"]
                body = select([*branch["body"], *rest], prefix)
                if index is None:
                    return body
                prepare, header = condition(index)
                return [*prepare, header, *body, "else", *branches(following), "endif"]
            return branches(first["branches"])
        operation = source_id(member.builder, "source_device_task", span=(self.first, self.last),
                              proof_identity=self.proof.identity)
        lines = ["fort_atomic_done = .false.", f"if ({member.enabled}) then",
                 selection_block + ": block",
                 *select(public["tree"]), "select case(fort_atomic_variant)"]
        for index, item in enumerate(self.alternatives):
            if item is None:
                continue  # Empty original paths retain the complete native team.
            provider = item.provider()
            for name, text in ((provider.entry + ".cu", provider.cuda_source),
                               (provider.module + ".f90", provider.fortran_source)):
                previous = outputs.setdefault(name, text)
                if previous != text:
                    raise CompilationError("atomic team provider artifact identity collision")
            imports.append(f"use {provider.module}, only: {provider.entry}")
            implementation = source_id(member.builder, "implementation", source_task_provider=_provider_record(provider))
            handles = {binding.root: "fort_buffer_" + str(member.resources[binding.root][1]) for binding in provider.fields}
            values = {binding.root: member.builder.visible(member.routine, binding.root)
                      for binding in item.prepared.extraction.bindings}
            lowers = {binding.root: tuple(f"lbound({values[binding.root]},{axis},kind=c_int64_t)"
                                         for axis in range(1, binding.rank+1)) for binding in provider.fields}
            block = "fort_atomic_prepare_" + str(index)
            if item.prepared.conditional_context is None:
                admission = _build_read_sections(item.reads.sections, handles, values, "fort_ar_" + str(index),
                    on_error=("exit " + block,), logical_lower_bounds=lowers)
            else:
                # Only the provider can establish the conditional source-read
                # precondition, after all original metadata and legal guards.
                from compiler.scopes.registration import RegistrationCode
                admission = RegistrationCode((), ())
            specification += admission.specification
            arguments = [member.context, *[handles[binding.root] for binding in provider.fields]]
            snapshots = []
            for number, parameter in enumerate(provider.parameters):
                original = member.builder.visible(member.routine, parameter.binding.root)
                if parameter.axis is None:
                    arguments.append(original)
                else:
                    temporary = f"fort_atomic_lower_{index}_{number}"
                    specification.append(f"integer(c_int) :: {temporary}")
                    snapshots.append(f"{temporary}=lbound({original},{parameter.axis},kind=c_int)")
                    arguments.append(temporary)
            arguments.append("fort_atomic_report")
            lines += ["case(" + str(index) + ")", block + ": block",
                      "fort_atomic_can=fort_scope_numerical_environment_supported()/=0"]
            for binding in provider.fields:
                name = values[binding.root]
                if "allocatable" in binding.attributes:
                    lines.append(f"if (fort_atomic_can) fort_atomic_can=allocated({name})")
                lines.append(f"if (fort_atomic_can) fort_atomic_can=is_contiguous({name})")
                lines += [f"if (fort_atomic_can) fort_atomic_can={condition}" for condition in
                          member.builder.original_bound_conditions(name, binding.rank)]
            lines += ["if (.not. fort_atomic_can) exit " + block,
                      *item.registration, f"if ({member.context}==0) exit " + block,
                      *[f"if (fort_buffer_{member.resources[point['resource']][1]}=={handles[binding.root]}) exit " + block
                        for predicate in public["predicates"] for point in predicate["host_point_reads"]
                        for binding in provider.fields if binding.root in {root for unit in public["units"] for root in unit["writes"]}],
                      *snapshots, *admission.prepare,
                      *position(member.builder, context=member.context, owner=member.trace_owner,
                                operation=operation, implementation=implementation),
                      "fort_atomic_report=fort_scope_external_report_v1()",
                      "fort_atomic_status=" + provider.entry + "(" + ",".join(arguments) + ")",
                      "if (fort_atomic_report%version/=1 .or. fort_atomic_report%status/=fort_atomic_status) &",
                      "error stop 'atomic original team returned an invalid transaction report'",
                      "if (fort_atomic_status==FORT_SCOPE_OK .and. fort_atomic_report%started==1 .and. &",
                      "    fort_atomic_report%committed==1) then", "fort_atomic_done=.true.", "else",
                      "if (fort_atomic_report%started/=0 .or. fort_atomic_report%committed/=0) &",
                      "error stop 'atomic original team attempted work; replay prohibited'",
                      "if (fort_atomic_status/=FORT_SCOPE_OK .and. fort_atomic_status/=FORT_SCOPE_ARGUMENT .and. &",
                      "    fort_atomic_status/=FORT_SCOPE_RESOURCE .and. fort_atomic_status/=FORT_SCOPE_BOUNDARY .and. &",
                      "    fort_atomic_status/=FORT_SCOPE_UNINITIALIZED .and. fort_atomic_status/=FORT_SCOPE_PLAN_MISMATCH) &",
                      "error stop 'atomic original team failed fatally'", "endif", "end block " + block]
        lines += ["end select", "end block " + selection_block, "if (.not. fort_atomic_done) then",
                  *member.close(prework=True, reason="atomic original team declined before complete alternative",
                                span=(self.first, self.last)), "endif", "endif",
                  "if (.not. fort_atomic_done) then", *member.original(self.nodes).splitlines(), "endif"]
        for name, text in (("common_functions.cuh", files("compiler.emission.common").joinpath(
                "templates", "common_functions.cuh").read_text(encoding="utf-8")),
                           ("numeric.hpp", files("compiler.runtime").joinpath("numeric.hpp").read_text(encoding="utf-8"))):
            previous = outputs.setdefault(name, text)
            if previous != text:
                raise CompilationError("atomic team support artifact collision")
        emitted = ["fort_atomic_team_" + self.proof.identity[:12] + ": block", *imports,
                   *specification, f"associate(fort_context => {member.context})", *lines,
                   "end associate", "end block fort_atomic_team_" + self.proof.identity[:12]]
        if member.control_guard is not None:
            emitted = [f"if ({member.control_guard}) then", *emitted, "else",
                       *member.original(self.nodes).splitlines(), "endif"]
        return "\n".join(fortran_lines(emitted)) + "\n"


def admit_atomic_joined_team(member, nodes):
    config = member.builder.config
    # The existing cost protocol does not cover this new atomic alternative.
    # AUTO retains its original/native route rather than inventing a price.
    if config.scope_execution != "graph" or config.policy != "sections":
        return False
    originals = tuple(member.builder.inline.original_selection(tuple(nodes)))
    team = prove_atomic_worksharing_team(member.builder.analysis, member.routine, originals)
    public = team.public()
    if not public["predicates"] or not any(unit["nowait"] for unit in public["units"]):
        return False
    policy = NumericalPreparationProvider(member.builder.analysis, options=member.builder.options,
        config=config, environment_root=member.builder._numerical_environment_root,
        immutable_inputs=getattr(member.builder, "immutable_inputs", {}))
    conditional_context = None
    if public.get("conditional_affine_context_required"):
        from compiler.frontend.conditional_numerical import prove_conditional_numerical_context
        conditional_context = prove_conditional_numerical_context(policy, team)
    alternatives = []
    for source in atomic_worksharing_alternatives(team):
        if not source.loops:
            alternatives.append(None)
            continue
        prepared = policy.prepare_atomic_alternative(source, conditional_context=conditional_context)
        effects = prove_numerical_effects(member.builder.analysis, member.routine.qualified, originals, prepared)
        reads = prove_reached_read_requirements(member.builder.analysis, member.routine.qualified, originals, prepared)
        emit_atomic_team_provider(source, prepared, effects, reads)
        alternatives.append(AtomicAlternative(source, prepared, effects, reads, ()))
    for binding, _written in team.fields:
        member.builder.capture(binding)
    by_root = {binding.root: binding for binding, _written in team.fields}
    predicates = []
    for index in range(len(team.predicates)):
        sections = team.predicate_sections(index)
        registration = register_original_actuals(member, tuple(
            OriginalActual(by_root[item.resource], member.builder.visible(member.routine, item.resource))
            for item in sections.resources)) if sections.resources else ()
        predicates.append(tuple(registration))
    for index, item in enumerate(alternatives):
        if item is None:
            continue
        fields = item.provider().fields
        registration = register_original_actuals(member, tuple(OriginalActual(binding,
            member.builder.visible(member.routine, binding.root)) for binding in fields))
        alternatives[index] = AtomicAlternative(item.source, item.prepared, item.effects, item.reads, tuple(registration))
    first, last = statement_span(originals[0])[0], statement_span(originals[-1])[1]
    patch = AtomicJoinedTeamPatch(team, originals, (), first, last, alternatives=tuple(alternatives),
                                 predicate_registration=tuple(predicates))
    member.source_device_tasks.append(patch)
    member.patches.append((first, last, patch))
    return True
