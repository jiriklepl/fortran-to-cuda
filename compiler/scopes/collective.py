"""Additive source companions for specifically proved existing OpenMP teams.

The original owning procedure remains callable. Only source-bound full-team
calls enter this dispatcher; resource and definition checks precede computation.
This initial implementation accepts direct worksharing leaves, not nested call
graphs. Automatic placement requires matching offline collective calibration.
"""

from __future__ import annotations

import json
from dataclasses import replace

from fparser.two import Fortran2003 as F

from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources
from compiler.emission.fortran.formatting import _fortran_line, _fortran_list
from compiler.frontend import lower_file
from compiler.frontend.source_effects import _children, _kind, _part
from compiler.ir import CompilationError, SourceLocation
from compiler.scopes.numerical import resource_binding
from compiler.scopes.participation import verify_collective_participation
from compiler.scopes.source import DTYPES, ScopeBuilder, _call, _name, _span


class CollectiveScopeBuilder(ScopeBuilder):
    def __init__(self, paths, entry, *, facts, options, config, **kwargs):
        if not config.collective:
            raise CompilationError("schema-2 source scopes require explicit collective execution")
        serial_facts = {**facts, "schema_version": 1, "participation": "serial"}
        super().__init__(paths, entry, facts=serial_facts, options=options,
                         config=replace(config, collective=False), **kwargs)
        self.facts, self.config = facts, config
        self.participation = verify_collective_participation(
            self.analysis, self.entry.qualified, facts, host_threads=config.host_threads)
        self.roles = {}
        assertions = facts.get("native_participation", {})
        if not isinstance(assertions, dict):
            raise CompilationError("native participation assertions must be keyed by qualified procedure")
        for procedure, assertion in assertions.items():
            if procedure not in self.analysis.routines or not isinstance(assertion, dict):
                raise CompilationError("native participation assertion is unavailable")
            role = self.role(procedure)
            if not role["available"] or any(assertion.get(key) != role.get(key) for key in
                                            ("source", "source_sha256", "kind", "completion")):
                raise CompilationError("native participation assertion differs from source proof: " + procedure)

    def role(self, procedure):
        if procedure not in self.roles:
            from compiler.scopes.collective_roles import prove_existing_team_worksharing
            self.roles[procedure] = prove_existing_team_worksharing(self.analysis, procedure)
        return self.roles[procedure]

    def numerical(self, procedure):
        if procedure in self.generated:
            return self.generated[procedure]
        routine = self.analysis.routines[procedure]
        summary = self.analysis.summarize(procedure)
        role = self.role(procedure)
        result, reason = None, role["reason"]
        if role["available"]:
            try:
                self.check_numerical_capture_origins(procedure)
                self.check_query_specification(routine)
                if summary["persistent_state"] or summary.get("definition_diagnostics"):
                    raise CompilationError("numerical leaf has persistent state or unsupported definition effects")
                if any(binding.dtype == "logical" and binding.kind != 1
                       for name, binding in routine.scope.bindings.items() if name in routine.arguments):
                    raise CompilationError("source LOGICAL conversion requires guarded ABI handling")
                effects, _, _ = self.native_effects(procedure)
                package = self.packages.get(procedure)
                if package:
                    if not set(effects).issubset(package.arrays):
                        raise CompilationError("numerical source package omits original array effects")
                    function = lower_file(package.path, package.entry)
                else:
                    if any(not root.startswith("argument::") for root in effects):
                        raise CompilationError("hidden numerical arrays require a normalized source package")
                    function = lower_file(self.analysis.inputs.path(routine.scope.path), procedure)
                function, plan = prepare_function(function, options=self.options)
                if not plan.regions:
                    raise CompilationError("numerical source has no parallel region")
                sources = generate_sources(function, plan, offload_config=self.config, memory_model="scoped")
                if not sources.scoped or not sources.scoped.get("team", {}).get("available"):
                    raise CompilationError("numerical source has no supported full-team entry")
                result = sources
            except CompilationError as error:
                reason = str(error)
        self.generated[procedure] = result
        self.numerical_reasons[procedure] = None if result else reason
        return result

    def closure(self, procedure, active=()):
        if self.numerical(procedure):
            return {procedure}, {procedure}
        return set(), set()

    def check_native_definitions(self, call, actions, definitions, overwrites):
        role = self.role(call.procedure)
        if not role["available"]:
            raise CompilationError("native existing-team role is unsupported: " + role["reason"])
        super().check_native_definitions(call, actions, definitions, overwrites)

    def run(self):
        variant_checkpoint = self.variants.checkpoint()
        try:
            if any("allocatable" in self.entry.scope.bindings[formal].attributes
                   for formal in self.entry.arguments):
                # Even an unused allocatable OUT dummy deallocates its actual
                # on entry. The additive owner must not bypass that event.
                raise CompilationError("allocatable owning formals require original descriptor and allocation semantics")
            self.check_owner_namespace()
            calls = [self.resolve(self.entry, node) for node in _children(self.entry.execution)
                     if _kind(node) != "Comment"]
            if not 1 <= len(calls) <= 32:
                raise CompilationError("collective source scope requires between one and 32 direct leaves")
            for call in calls:
                if str(call.node.items[0]).lower().startswith("fort_"):
                    raise CompilationError("collective owner helper namespace conflicts with an original call")
                if not self.role(call.procedure)["available"]:
                    raise CompilationError("unsupported source worksharing role: " + self.role(call.procedure)["reason"])
            leaves = {call.procedure for call in calls if self.numerical(call.procedure)}
            if not leaves:
                raise CompilationError("collective source scope has no supported numerical leaf")
            arrays, scalars, written = self.owner_inputs(calls)
            if not arrays:
                raise CompilationError("collective source scope requires whole array captures")
            available, reason, _, _ = self.owner_query(calls, arrays, written)
            if not available:
                raise CompilationError("complete collective query is unavailable: " + reason)
            if self.config.policy == "auto":
                for procedure in sorted(leaves):
                    public = self.numerical(procedure).scoped
                    if not public.get("automatic_estimate_available"):
                        raise CompilationError(public.get("automatic_reason") or
                                               "collective synchronization calibration is unavailable; native execution")
            # Each captured owner root must be available at every qualified
            # caller, including hidden module state. Entry-local state cannot
            # be moved outside its original specification or initialization.
            for site in self.participation.sites:
                if not set((*arrays, *scalars)).issubset(site.bindings):
                    raise CompilationError("entry-local captures cannot enter a qualified companion")
            self.scopes.append(self.owner_team(calls, leaves, arrays, scalars, written))
        except CompilationError as error:
            # Unsupported execution is a successful unchanged native result.
            # Invalid asserted authority was rejected in the constructor.
            self.outputs.clear()
            self.edits.clear()
            self.variants.restore(variant_checkpoint)
            self.boundaries.append({"procedure": self.entry.qualified, "reason": str(error),
                                    "execution": "unchanged native source"})
        outputs, report = self.finish()
        report["participation"] = self.participation.public()
        report["collective_roles"] = [{"procedure": procedure, **role}
                                      for procedure, role in sorted(self.roles.items())]
        report["limits"]["collective"] = "direct source-backed worksharing leaves; bounded descriptor agreement"
        outputs["scope-manifest.json"] = json.dumps(report, indent=2) + "\n"
        return outputs, report

    def check_owner_namespace(self):
        """Copied USE associations cannot redefine generated checks or hooks."""
        reserved = {"size", "all", "any", "is_contiguous", "huge", "transfer", "int", "merge",
                    "c_loc", "c_null_ptr", "c_int", "c_int64_t", "c_intptr_t", "c_size_t",
                    "c_float", "c_double", "c_bool", "c_ptr", "omp_get_thread_num"}
        if any(name in reserved or name.startswith("fort_") for name in self.entry.scope.imports):
            raise CompilationError("collective owner helper namespace conflicts with an original USE association")

        def helper_exports(module, visited, depth=0):
            if module.module in visited:
                return set()
            if depth >= 8 or len(visited) >= 64:
                raise CompilationError("collective owner USE export closure exceeds its bounded proof")
            visited.add(module.module)
            names = {name for name in (*module.bindings, *module.imports, *module.generics)
                     if name.startswith("fort_")}
            names |= {name.split("::")[-1] for name in (*self.analysis.routines, *self.analysis.functions)
                      if name.startswith(module.module + "::fort_")}
            for child in module.wildcards:
                target = self.analysis.modules.get(child)
                if target is None:
                    raise CompilationError("collective owner requires bounded wildcard USE exports")
                names.update(helper_exports(target, visited, depth + 1))
            return names

        for module_name in self.entry.scope.wildcards:
            module = self.analysis.modules.get(module_name)
            if module is None:
                raise CompilationError("collective owner requires bounded wildcard USE exports")
            exported = reserved | helper_exports(module, set())
            for name in exported:
                if self.analysis._exported(module, name) and (
                        self.analysis._binding(module, name) or self.analysis._candidates(module, F.Name(name))):
                    raise CompilationError("collective owner helper namespace conflicts with an original USE export")

    def owner_team(self, calls, leaves, arrays, scalars, written):
        digest = self.entry.qualified + ":qualified-full-team"
        name, controller = _name("fort_team_owner_", digest), _name("fort_team_state_", digest)
        owner_variant = self.variants.register(
            self.entry.qualified, interface="source-team-v1", role="coordinator", name=name,
            summary_identity=self.analysis.summarize(self.entry.qualified)["summary_identity"],
            requirements=("proved source span and full-team caller participation", "one invocation owns buffer lifetime"),
            shared_artifacts=("sources/" + _name("source_", str(self.entry.scope.path)) + ".f90",))
        parameters = {root: "fort_capture_" + str(i) for i, root in enumerate((*arrays, *scalars))}
        views = {root: parameters[root] + "_view" if root in arrays else parameters[root]
                 for root in parameters}
        handles = {root: f"fort_state%handles({i})" for i, root in enumerate(arrays, 1)}
        origin_roots = self.runtime_origin_roots(leaves)
        count, rank, threads = len(arrays), max(binding.rank for binding in arrays.values()), self.config.host_threads
        if threads * count * (32 + 16 * rank) + (4 * threads if origin_roots else 0) > 1024 * 1024:
            raise CompilationError("collective descriptor metadata exceeds the one-MiB bound")
        module = self.entry.scope.parent
        if _kind(module.node) != "Module":
            raise CompilationError("qualified source companion requires a module owning entry")
        contains = next(node for node in _children(_part(module.node, "Module_Subprogram_Part"))
                        if _kind(node) == "Contains_Stmt")
        declaration = [f"type, public :: {controller}",
                       f"logical :: allocated({threads}), contiguous({count},{threads})",
                       *([f"logical :: bounds_fit({threads})"] if origin_roots else []),
                       f"integer(kind=8) :: extents({rank},{count},{threads}), lowers({rank},{count},{threads})",
                       f"integer(kind=8) :: addresses({count},{threads})",
                       f"integer(kind=8) :: context, handles({count})",
                       "integer :: status", "logical :: descriptor_fallback, fallback", f"end type {controller}", ""]
        # Source types use the ISO kinds of the generated owner below. This
        # controller stores only fixed-width integer metadata, never payload.
        # Require the established 64-bit source INTEGER ABI before publication.
        self.add_edit(module.path, _span(contains)[0], _span(contains)[0] - 1, "\n".join(declaration))

        imports = ["use iso_c_binding", "use fort_scoped_memory", "use fort_scoped_team_observer", "use omp_lib"]
        imports += [str(node) for node in _children(_part(self.entry.scope.node, "Specification_Part"))
                    if _kind(node) == "Use_Stmt"]
        specs = [f"type({controller}), pointer, intent(inout) :: fort_state",
                 "intrinsic :: size, all, any, is_contiguous, huge, transfer, int, merge",
                 "integer :: fort_tid, fort_i, fort_j", "integer(c_int) :: fort_cleanup",
                 "integer(c_intptr_t) :: fort_address", "type(c_ptr) :: fort_host_pointer",
                 "type(fort_scope_access) :: fort_access",
                 f"type(fort_scope_plan_binding), target :: fort_bindings({max(1,count)})",
                 "integer(c_size_t) :: fort_elements, fort_extent",
                 f"integer(c_int), parameter :: fort_mode = {2 if self.config.policy == 'auto' else 1}_c_int",
                 "type(fort_scope_plan_decision) :: fort_decision", "logical :: fort_observing"]
        for i, (root, binding) in enumerate(arrays.items(), 1):
            dtype, _, _ = DTYPES[binding.signature()[:2]]
            if binding.intent == "in" and root in written:
                raise CompilationError("scope writes an INTENT(IN) capture")
            intent = "inout" if root in written else "in"
            shape = ",".join(":" for _ in range(binding.rank))
            specs += _fortran_line(f"{dtype}, target, intent({intent}) :: {parameters[root]}({shape})", 0)
            specs += [f"{dtype}, pointer, contiguous :: {views[root]}({shape})",
                      f"integer(c_size_t), target :: fort_extents_{i}({binding.rank})",
                      f"integer(c_int64_t), target :: fort_lowers_{i}({binding.rank})",
                      f"type(fort_scope_layout) :: fort_layout_{i}"]
            fact = self.capture(binding)
            if fact["initialized"] == "sections" and fact["sections"]:
                specs += [f"type(fort_scope_section), target :: fort_defined_{i}({len(fact['sections'])})"]
                for j in range(len(fact["sections"])):
                    specs += [f"integer(c_size_t), target :: fort_defined_{i}_{j}_lower({binding.rank})",
                              f"integer(c_size_t), target :: fort_defined_{i}_{j}_upper({binding.rank})"]
        for root, binding in scalars.items():
            if binding.signature()[:2] not in DTYPES:
                raise CompilationError("collective scalar requires an interoperable immutable type")
            specs += [f"{DTYPES[binding.signature()[:2]][0]}, intent(in) :: {parameters[root]}"]

        def actuals(call, *, shared=True):
            selected = views if shared else parameters
            return [selected[binding.root] if
                    (binding := self.analysis._actual_binding(self.entry.scope, actual))
                    and binding.root in selected else str(actual) for actual in call.actuals]

        original = [line for call in calls for line in _call(str(call.node.items[0]), actuals(call, shared=False))]
        if any(str(call.node.items[0]).lower() in {"size", "all", "any", "is_contiguous", "huge", "transfer", "int", "merge"}
               for call in calls):
            raise CompilationError("collective owner intrinsic conflicts with an original call")
        postproof_native = [line for call in calls for line in _call(str(call.node.items[0]), actuals(call))]
        def observed_native(lines):
            return ["if (fort_observing) call fort_scope_team_observe_begin_v1(FORT_SCOPE_TEAM_OBSERVE_COMPUTE)",
                    *lines, "if (fort_observing) call fort_scope_team_observe_end_v1(FORT_SCOPE_TEAM_OBSERVE_COMPUTE)"]
        original, postproof_native = observed_native(original), observed_native(postproof_native)
        body = ["fort_tid = omp_get_thread_num() + 1",
                "fort_observing = fort_scope_team_observer_enabled_v1() /= 0"]
        # Allocation was agreed at the original caller. Only now may these
        # descriptors and empty-aware payload addresses be inspected.
        for i, (root, binding) in enumerate(arrays.items(), 1):
            visible = parameters[root]
            body += ["if (fort_observing) call fort_scope_team_observe_begin_v1(FORT_SCOPE_TEAM_OBSERVE_DESCRIPTOR)",
                     f"fort_state%contiguous({i},fort_tid) = is_contiguous({visible})",
                     f"fort_state%extents(:,{i},fort_tid) = 0_8"]
            body += _fortran_list(f"fort_state%extents(1:{binding.rank},{i},fort_tid) = [",
                                 [f"size({visible},{axis},kind=c_int64_t)" for axis in range(1,binding.rank+1)], "]", 0)
            body += ["fort_host_pointer = c_null_ptr",
                     f"if (all(fort_state%extents(1:{binding.rank},{i},fort_tid) > 0)) &",
                     f"  fort_host_pointer = c_loc({visible})",
                     "fort_address = transfer(fort_host_pointer, fort_address)",
                     f"fort_state%addresses({i},fort_tid) = int(fort_address,kind=8)",
                     "if (fort_observing) call fort_scope_team_observe_end_v1(FORT_SCOPE_TEAM_OBSERVE_DESCRIPTOR)"]
        # Participants can reach the descriptor branch at different times.
        # Keep this decision immutable while the coordinator records the plan;
        # reusing its flag for planning could divert a late peer into native
        # worksharing while the coordinator waits at the planning barrier.
        body += ["!$omp barrier", "!$omp master",
                 "fort_state%descriptor_fallback = .not. all(fort_state%contiguous)",
                 f"do fort_i = 2, {threads}",
                 "if (any(fort_state%extents(:,:,fort_i) /= fort_state%extents(:,:,1)) .or. &",
                 "    any(fort_state%lowers(:,:,fort_i) /= fort_state%lowers(:,:,1)) .or. &",
                 "    any(fort_state%addresses(:,fort_i) /= fort_state%addresses(:,1))) fort_state%descriptor_fallback = .true.",
                 "enddo", "!$omp end master", "!$omp barrier",
                 "if (fort_state%descriptor_fallback) then", *original, "return", "endif"]
        body += [f"{views[root]} => {parameters[root]}" for root in arrays]
        body += ["!$omp master", "fort_state%status = fort_scope_create(0_c_int, fort_state%context)",
                 "if (fort_state%status == FORT_SCOPE_OK) &",
                 f"  fort_state%status = fort_scope_set_device_budget(fort_state%context, {self.device_budget}_c_size_t)"]
        body += self.owner_transfer_setup(leaves, imports, context="fort_state%context", status="fort_state%status")
        if self.config.policy == "auto" and self.config.scope_transfers == "direct":
            public, _ = self.entry_artifacts(sorted(leaves)[0])
            imports.append(f"use {public['fortran_module']}, only: fort_configure_team => configure")
            body += ["if (fort_state%status == FORT_SCOPE_OK) &",
                     "  fort_state%status = fort_configure_team(fort_state%context)"]
        for i, (root, binding) in enumerate(arrays.items(), 1):
            _, enum, width = DTYPES[binding.signature()[:2]]
            visible = views[root]
            body += ["if (fort_state%status == FORT_SCOPE_OK) then",
                     "if (fort_observing) call fort_scope_team_observe_begin_v1(FORT_SCOPE_TEAM_OBSERVE_DESCRIPTOR)"]
            body += _fortran_list(f"fort_extents_{i} = [",
                                 [f"size({visible},{axis},kind=c_size_t)" for axis in range(1,binding.rank+1)], "]", 0)
            body += [f"fort_lowers_{i} = 1_c_int64_t", "fort_elements = 1_c_size_t",
                     f"do fort_j = 1, {binding.rank}", f"fort_extent = fort_extents_{i}(fort_j)",
                     "if (fort_extent /= 0) then",
                     f"if (fort_elements > huge(fort_elements) / {width}_c_size_t / fort_extent) &",
                     "  fort_state%status = FORT_SCOPE_RESOURCE", "endif",
                     "if (fort_state%status == FORT_SCOPE_OK) fort_elements = fort_elements * fort_extent", "enddo",
                     "fort_host_pointer = c_null_ptr",
                     f"if (all(fort_extents_{i} > 0)) fort_host_pointer = c_loc({visible})",
                     f"fort_layout_{i} = fort_scope_layout({binding.rank}, {enum}, {width}_c_size_t, &",
                     f"    fort_host_pointer, c_loc(fort_extents_{i}), c_loc(fort_lowers_{i}), 1_c_int64_t)"]
            fact = self.capture(binding)
            # Entering the replaced outer procedure also changes definitions.
            outer_out = root.startswith("argument::") and binding.intent == "out"
            if fact["initialized"] == "sections" and not outer_out:
                boxes = fact["sections"]
                for j, box in enumerate(boxes):
                    for bound in ("lower", "upper"):
                        body += _fortran_list(f"fort_defined_{i}_{j}_{bound} = [",
                                              [f"{value}_c_size_t" for value in box[bound]], "]", 0)
                        body += [f"fort_defined_{i}({j+1})%{bound} = c_loc(fort_defined_{i}_{j}_{bound})"]
                expression = f"fort_scope_register_sections(fort_state%context, {i}_c_int64_t, 1_c_int64_t, &"
                body += ["if (fort_state%status == FORT_SCOPE_OK) &", "  fort_state%status = " + expression,
                         f"    fort_layout_{i}, {'c_loc(fort_defined_'+str(i)+')' if boxes else 'c_null_ptr'}, &",
                         f"    {len(boxes)}_c_size_t, {handles[root]})"]
            else:
                initialized = int(fact["initialized"] == "whole" and not outer_out)
                body += ["if (fort_state%status == FORT_SCOPE_OK) &",
                         "  fort_state%status = fort_scope_register(fort_state%context, &",
                         f"    {i}_c_int64_t, 1_c_int64_t, fort_layout_{i}, {initialized}_c_int, {handles[root]})"]
            body += ["if (fort_observing) call fort_scope_team_observe_end_v1(FORT_SCOPE_TEAM_OBSERVE_DESCRIPTOR)", "endif"]
        body += ["if (fort_state%status == FORT_SCOPE_OK) &",
                 "  fort_state%status = fort_scope_plan_reset(fort_state%context)"]
        for index, call in enumerate(calls):
            mapping = {formal: binding.root for formal, binding in call.bindings.items()}
            native_handles = {formal: handles[root] for formal, root in mapping.items() if root in handles}
            native_handles.update({root: handle for root, handle in handles.items() if not root.startswith("argument::")})
            if call.procedure in leaves:
                public, _ = self.entry_artifacts(call.procedure)
                alias = f"fort_plan_{index}"
                imports.append(f"use {public['fortran_module']}, only: {alias} => {public['planning']['fortran_procedure']}")
                arguments = self.numerical_arguments(call, public, handles, views, query=True)
                body += self.package_forgets(call, handles, query=True)
                body += ["if (fort_state%status == FORT_SCOPE_OK) then"]
                body += _fortran_list(f"fort_state%status = {alias}(", ["fort_state%context", *arguments], ")", 0)
                body += ["endif"]
            else:
                body += self.checked("fort_scope_plan_team_native_call_v1(fort_state%context)")
                lines = self.native_plan(call, *self.native_effects(call.procedure), native_handles)
                block = f"fort_record_{index}"
                lines = [line.replace("fort_context", "fort_state%context").replace("fort_status", "fort_state%status")
                         for line in lines]
                lines = [("exit " + block if line.strip() == "return" else
                          line.replace("return", "exit " + block)) for line in lines]
                body += ["if (fort_state%status == FORT_SCOPE_OK) then", block + ": block", *lines,
                         "end block " + block, "endif"]
        body += ["if (fort_state%status == FORT_SCOPE_OK) &",
                 "  fort_state%status = fort_scope_plan_validate(fort_state%context)"]
        if self.config.policy == "auto":
            public, _ = self.entry_artifacts(sorted(leaves)[0])
            imports.append(f"use {public['fortran_module']}, only: fort_choose_team => choose")
            body += ["if (fort_state%status == FORT_SCOPE_OK) &",
                     "  fort_state%status = fort_choose_team(fort_state%context, fort_decision)",
                     "fort_state%fallback = fort_state%status /= FORT_SCOPE_OK",
                     "if (fort_state%status == FORT_SCOPE_OK) fort_state%fallback = fort_decision%gpu_units == 0"]
        else:
            body += ["fort_state%fallback = fort_state%status /= FORT_SCOPE_OK"]
        body += [
                 "if (fort_state%fallback .and. fort_state%context /= 0) then",
                 "fort_cleanup = fort_scope_close(fort_state%context)",
                 "if (fort_cleanup /= FORT_SCOPE_OK) error stop 'collective preflight cleanup failed'", "endif",
                 "!$omp end master", "!$omp barrier", "if (fort_state%fallback) then",
                 *postproof_native, "return", "endif"]
        for index, call in enumerate(calls):
            if call.procedure in leaves:
                public, _ = self.entry_artifacts(call.procedure)
                alias = f"fort_run_{index}"
                imports.append(f"use {public['fortran_module']}, only: {alias} => {public['team']['fortran_procedure']}")
                body += ["!$omp master", *self.package_forgets(call, handles, query=False),
                         "!$omp end master", "!$omp barrier", *self.check_status()]
                # All participants enter the same public worker. Its returned
                # status is uniform; only the coordinator publishes it.
                body += ["block", "integer(c_int) :: fort_returned"]
                body += _fortran_list(f"fort_returned = {alias}(",
                                     ["fort_state%context", "fort_mode",
                                      *self.numerical_arguments(call, public, handles, views, query=False)], ")", 0)
                body += ["!$omp master", "fort_state%status = fort_returned", "!$omp end master",
                         "!$omp barrier", "end block", *self.check_status()]
            else:
                actions, definitions, overwrites = self.roots_for(call)
                body += ["if (fort_observing) call fort_scope_team_observe_begin_v1(FORT_SCOPE_TEAM_OBSERVE_NATIVE_CALL)",
                         "!$omp master"]
                for root in sorted(definitions):
                    body += self.checked(f"fort_scope_forget_definition(fort_state%context, {handles[root]})")
                for root, kinds in sorted(actions.items()):
                    body += self.access_flags(kinds, root in overwrites)
                    body += self.checked(f"fort_scope_host_begin(fort_state%context, {handles[root]}, fort_access)")
                body += ["!$omp end master", "!$omp barrier", *self.check_status()]
                body += ["if (fort_observing) call fort_scope_team_observe_begin_v1(FORT_SCOPE_TEAM_OBSERVE_COMPUTE)"]
                body += _call(str(call.node.items[0]), actuals(call))
                body += ["if (fort_observing) call fort_scope_team_observe_end_v1(FORT_SCOPE_TEAM_OBSERVE_COMPUTE)"]
                body += ["!$omp barrier", "!$omp master"]
                for root in sorted(actions):
                    body += self.checked(f"fort_scope_host_end(fort_state%context, {handles[root]})")
                body += ["!$omp end master", "!$omp barrier", *self.check_status(),
                         "if (fort_observing) call fort_scope_team_observe_end_v1(FORT_SCOPE_TEAM_OBSERVE_NATIVE_CALL)"]
        body += ["!$omp master", "fort_state%status = fort_scope_close(fort_state%context)",
                 "!$omp end master", "!$omp barrier", *self.check_status()]
        header = _fortran_list("subroutine " + name + "(", [*parameters.values(), "fort_state"], ")", 0)
        text = "\n".join([*header, *dict.fromkeys(imports), "implicit none", *specs, *body,
                          "end subroutine " + name, ""])
        self.append_procedure(module, name, text)
        for site in self.participation.sites:
            self.add_edit(site.source, site.first_line, site.last_line,
                          self.caller_dispatch(site, name, controller, arrays, parameters, origin_roots=origin_roots))
        return {"owner": name, "path": str(self.entry.scope.path),
                "first_line": _span(calls[0].node)[0], "last_line": _span(calls[-1].node)[1],
                "parameters": [{"name": parameters[root], "resource": root,
                                "actual": self.visible(self.entry, root)} for root in parameters],
                "calls": [call.procedure for call in calls],
                "gpu_leaves": sorted(leaves), "mode": self.config.policy,
                "numerical_entries": [{"procedure": procedure, "public": self.numerical(procedure).scoped}
                                      for procedure in sorted(leaves)],
                "participation": "qualified_full_team", "estimate_available": self.config.policy == "auto",
                "planning_reason": None if self.config.policy == "auto" else "scoped automatic selection was not requested",
                "owner_variant": owner_variant.identity,
                "definition_preflight": {"abi_version": 1, "query_available": True, "reason": None,
                                         "position": "before numerical execution"},
                "allocation_preflight": {"position": "original qualified caller before owner association",
                                         "participation": "full team; allocation agreement then exact descriptors",
                                         **(self.bounds_preflight_public(origin_roots, "agreed original caller allocation descriptors")
                                            if origin_roots else {})},
                "resources": [{"resource": root, "registration_identity": i,
                               "allocation_generation": 1, "initialized": self.capture(binding)["initialized"]}
                              for i, (root, binding) in enumerate(arrays.items(), 1)],
                "placement": ("calibrated shared CPU/GPU workers; all-native decision executes original existing-team calls"
                              if self.config.policy == "auto" else
                              "forced shared GPU workers and original existing-team native worksharing"),
                "transfer_configuration": self.numerical(sorted(leaves)[0]).scoped["transfer_configuration"],
                "metadata_budget_bytes": 1024 * 1024, "host_threads": threads}

    @staticmethod
    def checked(expression):
        return ["if (fort_state%status == FORT_SCOPE_OK) then",
                "fort_state%status = " + expression, "endif"]

    @staticmethod
    def check_status():
        return ["if (fort_state%status /= FORT_SCOPE_OK) error stop 'collective scope execution failed; no replay'"]

    @staticmethod
    def access_flags(kinds, overwrite):
        flags = (["FORT_SCOPE_READ_ALL"] if "read" in kinds else [])
        flags += (["FORT_SCOPE_WRITE_ALL"] if "write" in kinds else [])
        flags += (["FORT_SCOPE_OVERWRITE_ALL"] if overwrite else [])
        return ["fort_access = fort_scope_access()", "fort_access%flags = " + " + ".join(flags)]

    def package_forgets(self, call, handles, *, query):
        if call.procedure not in self.packages:
            return []
        lines = []
        routine = self.analysis.routines[call.procedure]
        for formal in routine.arguments:
            binding = routine.scope.bindings[formal]
            if not binding.rank or binding.intent != "out":
                continue
            handle = handles[call.bindings["argument::" + formal].root]
            if query:
                lines += ["if (fort_state%status == FORT_SCOPE_OK) then",
                          "fort_bindings(1) = fort_scope_plan_binding()", f"fort_bindings(1)%buffer = {handle}",
                          "fort_state%status = fort_scope_plan_add(fort_state%context, FORT_SCOPE_PLAN_FORGET, &",
                          "    0_c_int64_t, c_loc(fort_bindings), 1_c_size_t, 0.0_c_double, 0.0_c_double, 0_c_int)", "endif"]
            else:
                lines += self.checked(f"fort_scope_forget_definition(fort_state%context, {handle})")
        return lines

    def numerical_arguments(self, call, public, handles, views, *, query):
        package = self.packages.get(call.procedure)
        normalized = {parameter.name: parameter for parameter in package.parameters} if package else {}
        routine = self.analysis.routines[call.procedure]

        def integer(value):
            # The minimum signed value cannot appear as an unsigned default
            # INTEGER source token; express it using representable operands.
            return "(-2147483647_c_int - 1_c_int)" if value == -(2**31) else str(value) + "_c_int"

        def argument(name):
            parameter = normalized.get(name.lower())
            resource = parameter.resource if parameter else "argument::" + name.lower()
            original = call.bindings.get(resource)
            root = original.root if original else resource
            if parameter and parameter.lower_bound_dimension is not None:
                if parameter.runtime_lower_bound:
                    index = list(handles).index(root) + 1
                    return f"int(fort_state%lowers({parameter.lower_bound_dimension},{index},1), kind=c_int)"
                binding = resource_binding(self.analysis, routine, resource)
                dimension = parameter.lower_bound_dimension
                lower = routine.scope.kinds.integer(F.Level_2_Expr(binding.lower_bounds[dimension - 1]),
                                                    SourceLocation(str(routine.scope.path)))
                return f"merge(1_c_int, {integer(lower)}, size({views[root]},{dimension}) == 0)"
            if root in handles:
                return handles[root]
            if root in views:
                return views[root]
            if parameter:
                binding = resource_binding(self.analysis, routine, resource)
                if "parameter" in binding.attributes and binding.signature() == ("integer", 4, 0):
                    value = routine.scope.kinds.integer(F.Name(binding.name), SourceLocation(str(routine.scope.path)))
                    return integer(value)
            if resource.startswith("argument::"):
                formal = resource.split("::", 1)[1]
                return str(call.actuals[routine.arguments.index(formal)])
            raise CompilationError("collective numerical parameter lacks a stable root: " + resource)

        order = public["planning"]["argument_order"][1:] if query else public["argument_order"][2:]
        return [argument(name) for name in order]

    def caller_dispatch(self, site, owner, controller, arrays, parameters, *, origin_roots=()):
        original = "".join(site.source.read_text().splitlines(keepends=True)[site.first_line - 1:site.last_line])
        if not original.endswith("\n"):
            original += "\n"
        prefix = _name("fort_dispatch_", self.entry.qualified + ":" + str(site.first_line))
        names = {root: self.visible(site.routine, site.bindings[root].root) for root in parameters}
        # Block declarations may not shadow actual names, including host and
        # use-associated objects. All generated names are individually checked.
        locals_ = [prefix + suffix for suffix in ("_state", "_status", "_tid", "_level", "_threads")]
        observing = prefix + "_observing"
        hooks = {name: prefix + "_" + name for name in ("begin", "end", "enabled", "owner", "descriptor")}
        fortran_identity = None
        if self.config.policy == "auto" and self.config.profile:
            fortran_identity = self.config.profile.get("scoped", {}).get("collective", {}).get("fortran")
        reserved = {*locals_, observing, *hooks.values(), prefix + "_id", owner, controller, "allocated", "lbound", "all"}
        reserved.update(prefix + suffix for suffix in ("_version", "_options", "_nul", "_compatible"))
        intrinsics = {"allocated", "lbound", "all"} | ({"ubound", "size"} if origin_roots else set())
        if fortran_identity:
            intrinsics.add("achar")
        reserved.update(intrinsics)
        if reserved & set(names.values()) or str(site.node.items[0]).lower() in intrinsics:
            raise CompilationError("collective caller helper namespace conflicts with an actual")
        state, status, tid, level, threads = locals_
        imports = [f"use omp_lib, only: {level} => omp_get_level, {threads} => omp_get_num_threads, &",
                   f"    {prefix}_id => omp_get_thread_num",
                   "use fort_scoped_team_observer, only: &",
                   f"  {hooks['begin']} => fort_scope_team_observe_begin_v1, &",
                   f"  {hooks['end']} => fort_scope_team_observe_end_v1, &",
                   f"  {hooks['enabled']} => fort_scope_team_observer_enabled_v1, &",
                   f"  {hooks['owner']} => FORT_SCOPE_TEAM_OBSERVE_OWNER, &",
                   f"  {hooks['descriptor']} => FORT_SCOPE_TEAM_OBSERVE_DESCRIPTOR"]
        if fortran_identity:
            imports += [f"use iso_fortran_env, only: {prefix}_version => compiler_version, &",
                        f"  {prefix}_options => compiler_options",
                        f"use iso_c_binding, only: {prefix}_nul => c_null_char",
                        f"use fort_scoped_team_observer, only: {prefix}_compatible => fort_scope_team_fortran_compatible_v1"]
        if site.routine.scope.module != self.entry.scope.module:
            imports += [f"use {self.entry.scope.module}, only: {owner}, {controller}"]
        lines = ["block", *imports, f"type({controller}), pointer :: {state}",
                 f"integer :: {status}, {tid}", f"logical :: {observing}",
                 "intrinsic :: allocated, lbound, all" + (", ubound, size" if origin_roots else "")
                 + (", achar" if fortran_identity else ""),
                 f"if ({level}() /= 1 .or. {threads}() /= {self.config.host_threads}) then",
                 original.rstrip(), "else", f"{observing} = {hooks['enabled']}() /= 0",
                 f"if ({observing}) call {hooks['begin']}({hooks['owner']})",
                 "!$omp barrier", "!$omp single", f"allocate({state}, stat={status})",
                 f"if ({status} == 0) then", f"{state}%context = 0_8", f"{state}%status = 0",
                 f"{state}%handles = 0_8", f"{state}%lowers = 0_8", f"{state}%fallback = .false.", "endif",
                 f"!$omp end single copyprivate({state},{status})", f"if ({status} /= 0) then",
                 original.rstrip(), "else", f"{tid} = {prefix}_id() + 1", f"{state}%allocated({tid}) = .true."]
        if fortran_identity:
            def literal(value):
                return ["'" + value[index:index+48].replace("'", "''") + "'"
                        for index in range(0, len(value), 48)] or ["''"]
            version = literal(fortran_identity["compiler_version"])
            semantic = []
            for token in fortran_identity["semantic_options"].split("\x1f"):
                if semantic:
                    semantic.append("achar(31)")
                semantic += literal(token)
            lines += [f"{state}%allocated({tid}) = {prefix}_compatible( &",
                      f"  {prefix}_version() // {prefix}_nul, &",
                      f"  {prefix}_options() // {prefix}_nul, &"]
            lines += ["  " + part + " // &" for part in version]
            lines += [f"  {prefix}_nul, &"]
            lines += ["  " + part + " // &" for part in semantic]
            lines += [f"  {prefix}_nul) /= 0"]
        for root in arrays:
            if "allocatable" in site.bindings[root].attributes:
                lines += [f"{state}%allocated({tid}) = &",
                          f"    {state}%allocated({tid}) .and. &", f"    allocated({names[root]})"]
        lines += ["!$omp barrier", f"if (all({state}%allocated)) then"]
        for i, (root, binding) in enumerate(arrays.items(), 1):
            lines += [f"if ({observing}) call {hooks['begin']}({hooks['descriptor']})"]
            lines += _fortran_list(f"{state}%lowers(1:{binding.rank},{i},{tid}) = [",
                                   [f"lbound({names[root]},{axis},kind=8)" for axis in range(1,binding.rank+1)], "]", 0)
            lines += [f"if ({observing}) call {hooks['end']}({hooks['descriptor']})"]
        if origin_roots:
            lines += [f"{state}%bounds_fit({tid}) = .true."]
            for root in origin_roots:
                for condition in self.original_bound_conditions(names[root], arrays[root].rank):
                    lines += [f"{state}%bounds_fit({tid}) = &",
                              f"    {state}%bounds_fit({tid}) .and. &", "    " + condition]
        lines += ["!$omp barrier"]
        if origin_roots:
            lines += [f"if (all({state}%bounds_fit)) then", *_call(owner, [*names.values(), state]),
                      "else", original.rstrip(), "endif"]
        else:
            lines += _call(owner, [*names.values(), state])
        lines += ["else", original.rstrip(), "endif",
                  "!$omp barrier", "!$omp single", f"deallocate({state})", "!$omp end single", "endif",
                  f"if ({observing}) call {hooks['end']}({hooks['owner']})", "endif", "end block", ""]
        # GNU otherwise associates a first BLOCK with the OpenMP construct's
        # structured-BLOCK form and rejects subsequent sibling dispatchers.
        # Keep the original team intact; CONTINUE performs no numerical work.
        if not any(other.source == site.source and other.team_first_line == site.team_first_line
                   and other.first_line < site.first_line for other in self.participation.sites):
            lines.insert(0, "continue")
        return "\n".join(lines)
