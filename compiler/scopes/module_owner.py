"""Bounded original-body companions for reached module continuations.

An additive Fortran entry shares the original declarations and body. Native
callers retain their ABI and never read absent compiler-control arguments.
"""

from copy import copy
from types import MethodType

from fparser.two.utils import walk

from compiler.frontend.source_effects import _children, _kind, _part
from compiler.ir import CompilationError
from compiler.scopes.lexical import LexicalOwner, optional_scalar_input
from compiler.scopes.reached_owner import MappedReachedControl
from compiler.scopes.registration import expected_descriptor_conditions
from compiler.scopes.segments import fortran_lines, statement_span
from compiler.scopes.source import _call, _name, _span


def _reusable_plan(value):
    """Aggregate validation work belongs to the call, not the worker body."""
    if isinstance(value, dict):
        return {key: _reusable_plan(item) for key, item in value.items() if key != "mapped_work"}
    if isinstance(value, list):
        return [_reusable_plan(item) for item in value]
    return value


def _caller_arrays(parent, resolved, child, immutable_inputs, reached):
    """Separate original associations from the child's retained handles."""
    associations = [mapping for mapping in resolved.mappings if mapping.formal_binding.rank
                    and mapping.formal_binding.root not in immutable_inputs]
    numbers = set(child.handle_numbers())
    parameterized = parent.builder.config.scope_execution == "graph" and reached is None
    native_roots = ({root for unit in child.units for operation in unit.scope.native
                     for root in operation.host_only_reads}
                    | set(getattr(child, "parameterized_native_bindings", {}))) if parameterized else set()
    native_arrays = [mapping for mapping in associations if mapping.formal_binding.root in native_roots]
    if native_arrays:
        from compiler.scopes.segments import _host_read_candidate
        for mapping in native_arrays:
            if mapping.formal_binding.intent != "in" or not _host_read_candidate(parent.builder, mapping.binding):
                raise CompilationError("native-only formal requires its original stable read-only module actual")
    arrays = [mapping for mapping in associations if mapping not in native_arrays
              if (("argument::" + mapping.formal in child.resources) if parameterized else
                  parent.owner.resources.get(parent.canonical(mapping.binding).root, (None, None))[1] in numbers)]
    preflight_arrays = list(associations)
    activation = (parent.builder.validate_selection_activation() if parent is parent.owner else None)
    if (reached is not None or activation is not None) and parent is parent.owner:
        from compiler.scopes.call_registration import OriginalActual

        if activation is not None:
            # Root identity is established before any dummy can rebase it.
            # Unknown prefix state grants no payload definitions here.
            partial = {state["resource"] for state in activation.public()["states"]
                       if state["initialized"] == "none"}
        else:
            record = parent.owner.builder.reached_plan_record()
            partial = {state["resource"] for state in record["states"] if state["initialized"] == "sections"}
        present = {parent.canonical(mapping.binding).root for mapping in arrays}
        for binding, number in parent.owner.resources.values():
            if number in numbers and binding.root in partial and binding.root not in present:
                actual = OriginalActual(binding, parent.owner.builder.visible(parent.routine, binding.root))
                arrays.append(actual)
                preflight_arrays.append(actual)
                present.add(binding.root)
    for mapping in preflight_arrays:
        name = str(mapping.actual).lower().partition("%")[0].partition("(")[0].strip()
        if name == "fort_actuals_contiguous":
            raise CompilationError("original module companion actual conflicts with descriptor guard name")
    conditions = [condition for mapping in arrays for condition in expected_descriptor_conditions(
        parent.builder.capture(mapping.binding), mapping.binding.rank, str(mapping.actual))]
    outputs = [mapping for mapping in associations if mapping.formal_binding.intent == "out"]
    # A reusable graph helper receives caller-owned identities. Register its
    # actuals before entering its rebased dummies, including IN/INOUT actuals;
    # its first instantiation's facts must never seed a different allocation.
    registrations = arrays if reached is not None or activation is not None or parameterized else outputs
    # Validate new caller storage while companion creation can still roll back.
    for mapping in registrations:
        if str(mapping.actual).lower().startswith("fort_"):
            raise CompilationError("original mapped actual conflicts with compiler-private registration names")
        parent.declaration(mapping.binding)
    for mapping in native_arrays:
        parent.declaration(mapping.binding)
    return arrays, preflight_arrays, registrations, conditions, native_arrays


def _parameterized_handles(parent, resolved, child):
    """Bind fixed helper handle formals to this original CALL's actual slots."""
    actuals = {"argument::" + mapping.formal: mapping.binding for mapping in resolved.mappings
               if mapping.formal_binding.rank and mapping.binding is not None}
    bindings = {number: None for number in child.handle_numbers()}
    roots = {number: root for root, (_binding, number) in parent.owner.resources.items()}
    for root, (_binding, slot) in child.resources.items():
        if root in actuals:
            bindings[slot] = actuals[root]
            roots[slot] = parent.canonical(actuals[root]).root
    if len(set(roots[number] for number in bindings)) != len(bindings):
        raise CompilationError("parameterized module companion has uncertain hidden-resource aliases")
    result = {}
    native_roots = ({root for unit in child.units for operation in unit.scope.native
                     for root in operation.host_only_reads}
                    | set(getattr(child, "parameterized_native_bindings", {})))
    native_slots = {slot for root, (_binding, slot) in child.resources.items() if root in native_roots}
    for slot, binding in bindings.items():
        if binding is None:
            result[slot] = slot
        else:
            parent.add_resource(binding)
            result[slot] = parent.resources[binding.root][1]
            attribute = "parameterized_native_bindings" if slot in native_slots else "parameterized_actual_bindings"
            retained = getattr(parent, attribute, {})
            setattr(parent, attribute, retained)
            retained[binding.root] = binding
    return result


def _body_input_intents(builder, routine):
    """Retain existing ordinary numeric definitions until actual body effects.

    OUT entry performs no physical write for these ordinary intrinsic arrays.
    Graph mode intentionally extends legacy programs that subsequently consume
    previously defined untouched elements: existing coverage survives, while
    absent coverage remains absent. Exact reached writes define new coverage.
    This policy cannot apply to allocatables, pointers, derived types or any
    entry action with finalization/association effects.
    """
    if builder.config.scope_execution != "graph":
        return frozenset(), ()
    outputs = {binding.root for name, binding in routine.scope.bindings.items()
               if name in routine.arguments and binding.rank and binding.intent == "out"
               and binding.dtype in {"real", "integer", "logical"}
               and not binding.attributes & {"allocatable", "pointer", "optional", "volatile", "asynchronous", "value"}}
    if not outputs:
        return frozenset(), ()
    return frozenset(outputs), tuple({"procedure": routine.qualified, "resource": root,
        "declared_intent": "out", "graph_body_intent": "body_effects",
        "policy": "ordinary numeric OUT entry preserves existing coverage; exact body effects define writes",
        "incoming_definition_authority": "existing caller runtime definitions only; absent coverage remains absent",
        "entry_definitions_inferred": False, "strict_fortran_undefinedness_extension": True}
        for root in sorted(outputs))


class ModuleOwner(MappedReachedControl, LexicalOwner):
    def __init__(self, parent, routine, mappings, immutable_inputs=None, reached=None):
        from compiler.scopes.region_dispatch import InlineRegions

        proxy = copy(parent.builder)
        proxy.entry = routine
        if reached is not None:
            proxy._mapped_reached_path, proxy._mapped_reached_child = reached
        proxy.immutable_inputs = dict(immutable_inputs or {})
        consumption = getattr(proxy, "_reached_consumption", None)
        if consumption is not None:
            consumption.bind_child(proxy)
        proxy.inline = InlineRegions(proxy)
        super().__init__(proxy, parent=parent)
        if any("save" in binding.attributes for binding in routine.scope.bindings.values()):
            raise CompilationError("original module companion saved local state requires a separate lifetime proof")
        self.canonical_bindings = mappings
        # Mapping values are exact original actuals retained from the resolved
        # source CALL. No capture fact or initialized coverage is inferred.
        proxy._native_host_array_actuals = dict(mappings) if proxy.config.scope_execution == "graph" else {}
        self.resources = {}
        self.forwarded_handles = set()
        self.entry = _name("fort_scope_entry_", routine.qualified)
        self.context = _name("fort_child_context_", routine.qualified)
        self.enabled = _name("fort_child_enabled_", routine.qualified)
        self.epoch_pristine = _name("fort_child_epoch_", routine.qualified)
        self.query_plan = _name("fort_child_query_", routine.qualified)
        self.kind = _name("fort_child_kind_", routine.qualified)
        self.control_guard = _name("fort_borrowed_", routine.qualified)
        self.call_sites = []
        self.body_input_intents, self.body_intent_diagnostics = _body_input_intents(proxy, routine)
        managed_roots = ({state["resource"] for state in reached[1].record["states"]}
                         if reached is not None else None)

        def capture(_proxy, binding):
            if managed_roots is not None and binding.rank and binding.root not in managed_roots:
                raise CompilationError("original module companion capture exceeds its reached managed projection: "
                                       + binding.root)
            fact = self.owner.builder.capture(self.canonical(binding))
            if binding.name in routine.arguments:
                # The original caller guards its actual before this dummy's
                # association can replace the original descriptor lower bounds.
                fact = {key: value for key, value in fact.items() if key != "expected_descriptor"}
            if (binding.name in routine.arguments and binding.intent == "out"
                    and binding.root not in self.body_input_intents):
                # The original OUT event invalidates caller coverage. A child
                # registration can never seed definitions from old values.
                return {**{key: value for key, value in fact.items()
                           if key not in {"source_sections", "sections", "requires_registered_handle"}},
                        "initialized": "none"}
            if "source_sections" in fact:
                # The original root caller must establish this coverage before
                # borrowing. Child coordinates cannot recreate root geometry.
                return {**{key: value for key, value in fact.items()
                           if key not in {"source_sections", "sections"}},
                        "initialized": "none", "requires_registered_handle": True}
            return fact

        proxy.capture = MethodType(capture, proxy)
        # Admitted reached units enroll their own resources. An unrelated
        # formal needs no payload facts merely to retain its original CALL
        # association, but OUT events precede every child branch and RETURN.
        for name in routine.arguments:
            binding = routine.scope.bindings[name]
            if binding.rank and binding.intent == "out" and binding.root not in proxy.immutable_inputs:
                self.add_resource(binding)
        self.owner.builder.region_owners[routine.qualified] = proxy.inline
        self.nodes = proxy.inline.prepare(_children(routine.execution))
        proxy.generated.update(proxy.inline.generated)
        proxy.numerical_ir.update(proxy.inline.ir)
        proxy.numerical_reasons.update({name: None for name in proxy.inline.regions})
        self.scan(self.nodes)
        if reached is not None:
            self.validate_mapped_plan()
        if len({number for _, number in self.resources.values()}) != len(self.resources):
            raise CompilationError("original module companion has uncertain hidden-resource aliases")

    def emit_companion(self):
        self.refresh()
        builder, routine = self.builder, self.routine
        header = _part(routine.scope.node, "Subroutine_Stmt")
        first = statement_span(self.nodes[0])[0]
        start = _span(header)[1]+1
        kind_import = f"use iso_c_binding, only: {self.kind} => c_int64_t"
        if start != first:
            builder.add_edit(routine.scope.path, start, start-1, kind_import + "\n")
        handles = [f"fort_buffer_{number}" for number in self.handle_numbers()]
        labels = {int(node.item.label) for node in walk(routine.execution)
                  if getattr(node, "item", None) is not None and getattr(node.item, "label", None)}
        label = next((number for number in range(99999, 99900, -1) if number not in labels), None)
        if label is None:
            raise CompilationError("original module companion has no bounded free statement label")
        graph_controls = ([self.epoch_pristine, self.query_plan]
                          if builder.config.scope_execution == "graph" else [])
        prologue = [*([kind_import] if start == first else []),
                    f"logical :: {self.control_guard}",
                    f"integer({self.kind}), intent(inout) :: " + ", ".join([self.context, *handles]),
                    "logical, intent(inout) :: " + ", ".join([self.enabled, *graph_controls]),
                    f"{self.control_guard} = .false.", f"goto {label}",
                    "entry " + self.entry + "(" + ", ".join([*routine.arguments, self.context, self.enabled, *graph_controls, *handles]) + ")",
                    f"{self.control_guard} = .true."]
        outputs = [binding for binding in routine.scope.bindings.values()
                   if (binding.name in routine.arguments and binding.rank and binding.intent == "out"
                       and binding.root not in self.body_input_intents)]
        if outputs:
            prologue += ["block", "use iso_c_binding, only: c_int", "use fort_scoped_memory",
                         "integer(c_int) :: fort_entry_status", f"if ({self.context} /= 0) then"]
            for binding in outputs:
                handle = "fort_buffer_" + str(self.resources[binding.root][1])
                prologue += [f"if ({handle} /= 0) then",
                             f"fort_entry_status = fort_scope_forget_definition({self.context}, {handle})",
                             "if (fort_entry_status /= FORT_SCOPE_OK) error stop 'original entry definition failed'", "endif"]
            prologue += ["endif", "end block"]
        prologue += [str(label) + " continue"]
        builder.add_edit(routine.scope.path, first, first-1, "\n".join(fortran_lines(prologue))+"\n", prepend=True)
        module = routine.scope.parent
        contains = next(node for node in _children(_part(module.node, "Module_Subprogram_Part"))
                        if _kind(node) == "Contains_Stmt")
        line = _span(contains)[0]
        builder.add_edit(module.path, line, line-1, "public :: " + self.entry + "\n")
        builder.variants.register(routine.qualified, interface="original_module_entry_v1", role="coordinator",
            name=self.entry, summary_identity=self.graph.identity,
            requirements=("original unsaved module body", "shared owning context control",
                          "per-call original actual handles" if builder.config.scope_execution == "graph"
                          else "one canonical actual mapping"),
            shared_artifacts=("sources/" + _name("source_", str(routine.scope.path)) + ".f90",))

    def public(self):
        return {"procedure": self.routine.qualified, "entry": self.entry,
                "interface": "original_module_entry_v1",
                "control_version": 3 if self.builder.config.scope_execution == "graph" else 1,
                "structured_summary_identity": self.graph.identity,
                "native_abi": "original entry; compiler-control arguments are never referenced",
                "storage": "one original body and declarations; no persistent-state copy",
                "resource_mappings": {root: binding.root for root, binding in self.canonical_bindings.items()},
                "immutable_array_inputs": [item.public() for item in self.builder.immutable_inputs.values()],
                "calls": self.call_sites, "boundaries": self.boundaries,
                "optional_scalar_formals": [name for name in self.routine.arguments
                    if optional_scalar_input(self.routine.scope.bindings[name])],
                "optional_scalar_association": "original OPTIONAL reference and PRESENT guards; no payload capture",
                "array_registration": "admitted reached resources; original OUT actuals registered before unconditional entry invalidation",
                "argument_rendering": "original source arguments; compiler controls appended by keyword",
                "planning_segments": [{"segment_id": index, "first_line": unit.scope.first,
                    "last_line": unit.scope.last, "operations": self.owner.execution_plan.operations(self, index)}
                    for index, unit in enumerate(self.owner.execution_plan.units(self))],
                "mapped_reached_plan": (self.builder._mapped_reached_child.public()
                    if hasattr(self.builder, "_mapped_reached_child") else None),
                "parameterized_handles": self.builder.config.scope_execution == "graph",
                "body_intent_diagnostics": list(self.body_intent_diagnostics),
                "limits": ("ordinary numeric formals and read-only optional scalars; per-call original actual handles; original saved locals remain native"
                           if self.builder.config.scope_execution == "graph" else
                           "ordinary numeric formals and read-only optional scalars; one canonical mapping; original saved locals remain native")}


def borrow_module(parent, node, *, reached=None):
    """Try one original module body after its complete borrowed worker failed."""
    if hasattr(node, "fort_inline_region"):
        return False
    resolved = parent.builder.execution_source_call(parent.routine, node)
    elements = tuple(mapping.scalar_element for mapping in resolved.mappings if mapping.scalar_element is not None)
    if elements:
        if reached is None:
            raise CompilationError("scalar element borrowing requires a registered mapped reached call")
        authorized = reached[1].coordinates.scalar_elements
        if (len(elements) != len(authorized) or any(
                element.actual is not original.actual or element.binding is not original.binding
                or element.public() != original.public()
                for element, original in zip(elements, authorized, strict=True))):
            raise CompilationError("scalar element borrowing lost original coordinate association authority")
        managed = {member.canonical(binding).root for member in parent.owner.members.values()
                   for unit in member.units for binding in unit.arrays.values()}
        if managed & {element.binding.root for element in elements}:
            raise CompilationError("scalar element backing storage must remain unmanaged")
    if reached is not None and any(mapping.formal_binding.rank and mapping.actual is not None
                                  and str(mapping.actual).lower().startswith("fort_")
                                  for mapping in resolved.mappings):
        raise CompilationError("original mapped actual conflicts with compiler-private registration names")
    routine = parent.builder.analysis.routines[resolved.procedure]
    if routine.source_kind != "module":
        return False
    from compiler.scopes.immutable_inputs import immutable_call_inputs
    immutable_inputs = immutable_call_inputs(parent.builder.analysis, parent.routine, node, resolved,
                                             getattr(parent.builder, "immutable_inputs", {}))
    parent.builder.check_whole_view_formals(routine, immutable_inputs=immutable_inputs)
    owner = parent.owner
    if routine.qualified in owner.active or len(owner.active) >= parent.builder.analysis.depth_limit:
        raise CompilationError("recursive or over-depth original module companion")
    if any(binding.dtype not in {"real", "integer", "logical"}
           or binding.attributes & {"allocatable", "pointer", "volatile", "asynchronous", "value"}
           or ("optional" in binding.attributes and not optional_scalar_input(binding))
           for name in routine.arguments for binding in (routine.scope.bindings[name],)):
        raise CompilationError("original module companion requires ordinary numeric arguments")
    if any("optional" in mapping.formal_binding.attributes
           and (mapping.presence not in {"supplied", "omitted", "forwarded_optional"}
                or (mapping.binding is not None and mapping.binding.attributes & {
                    "allocatable", "pointer", "volatile", "asynchronous", "value"}))
           for mapping in resolved.mappings):
        raise CompilationError("original module companion optional scalar requires original supplied or forwarded presence")
    mappings = {}
    for mapping in resolved.mappings:
        if not mapping.formal_binding.rank:
            continue
        if mapping.binding is None or mapping.section is not None:
            raise CompilationError("original module companion requires whole original array actuals")
        if mapping.binding.attributes & {"pointer", "optional", "volatile", "asynchronous"}:
            raise CompilationError("original module companion actual association requires a reached caller proof")
        mappings["argument::" + mapping.formal] = parent.canonical(mapping.binding)
    if len({binding.root for binding in mappings.values()}) != len(mappings):
        raise CompilationError("original module companion has uncertain actual aliases")
    existing = owner.members.get(routine.qualified)
    parameterized = parent.builder.config.scope_execution == "graph" and reached is None
    if existing is not None:
        same_mapping = (isinstance(existing, ModuleOwner)
            and {root: binding.root for root, binding in existing.canonical_bindings.items()}
            == {root: binding.root for root, binding in mappings.items()})
        same_immutable = (isinstance(existing, ModuleOwner)
            and {root: item.actual.root for root, item in existing.builder.immutable_inputs.items()}
            == {root: item.actual.root for root, item in immutable_inputs.items()})
        if not isinstance(existing, ModuleOwner) or not same_immutable or not same_mapping and not parameterized:
            raise CompilationError("original module companion requires one canonical mapping per owning invocation")
    if existing is not None and reached is not None:
        old = getattr(existing.builder, "_mapped_reached_child", None)
        if old is None or _reusable_plan(old.public()["plan"]) != _reusable_plan(reached[1].public()["plan"]):
            raise CompilationError("original module companion requires one reusable reached child plan")
    if existing is None:
        if len(owner.members) >= parent.builder.analysis.procedure_limit:
            raise CompilationError("original module companion procedure budget exhausted")
        checkpoint = owner.builder.scope_checkpoint()
        members, resources = dict(owner.members), dict(owner.resources)
        owner.active.append(routine.qualified)
        try:
            existing = ModuleOwner(parent, routine, mappings, immutable_inputs, reached)
            caller_arrays = _caller_arrays(parent, resolved, existing, immutable_inputs, reached)
            owner.members[routine.qualified] = existing
        except CompilationError:
            owner.builder.restore_scope_checkpoint(checkpoint)
            owner.members.clear()
            owner.members.update(members)
            owner.resources.clear()
            owner.resources.update(resources)
            consumption = getattr(parent.builder, "_reached_consumption", None)
            if consumption is not None:
                consumption.retain_children(member.builder for member in members.values())
            parent.refresh()
            raise
        finally:
            owner.active.pop()
    else:
        caller_arrays = _caller_arrays(parent, resolved, existing, immutable_inputs, reached)
    # Reached child admission can roll back rejected local candidates. Rebind
    # the caller's artifact maps before retaining its checked call patch.
    parent.refresh()
    first, last = statement_span(node)
    handle_bindings = (_parameterized_handles(parent, resolved, existing) if parameterized else
                       {number: number for number in existing.handle_numbers()})
    parent.forwarded_handles.update(handle_bindings.values())
    arguments = resolved.render_original_arguments()
    controls = [f"{existing.context} = {parent.context}", f"{existing.enabled} = {parent.enabled}",
                *([f"{existing.epoch_pristine} = {parent.epoch_pristine}",
                   f"{existing.query_plan} = {parent.query_plan}"]
                  if parent.builder.config.scope_execution == "graph" else []),
                *[f"fort_buffer_{slot} = fort_buffer_{number}" for slot, number in handle_bindings.items()]]
    # A CONTIGUOUS callee dummy can conceal a caller-created temporary. Check
    # actual storage before that association exists, otherwise a retained
    # handle might outlive temporary copyback at the original call return.
    arrays, preflight_arrays, registrations, descriptor_conditions, native_arrays = caller_arrays
    preflight = []
    if elements:
        # Only the original INTEGER selector is observed here, inside the
        # original CALL's guards. No table payload or child dummy is evaluated.
        preflight += ["block", "use iso_c_binding, only: c_int64_t"]
        for element in elements:
            lower, upper = element.bounds
            index = element.index.expression
            preflight += [f"if ({parent.enabled}) then",
                          f"if (int({index},kind=c_int64_t) < {lower}_c_int64_t .or. &",
                          f"int({index},kind=c_int64_t) > {upper}_c_int64_t) then", *parent.close(prework=True), "endif", "endif"]
        preflight += ["end block"]
    if preflight_arrays:
        preflight += ["block", *(["use iso_c_binding, only: c_int64_t"] if descriptor_conditions else []),
                      "logical :: fort_actuals_contiguous", "fort_actuals_contiguous = .true."]
        for mapping in preflight_arrays:
            name = str(mapping.actual)
            if "allocatable" in mapping.binding.attributes:
                preflight += [f"if (fort_actuals_contiguous) fort_actuals_contiguous = allocated({name})"]
            preflight += [f"if (fort_actuals_contiguous) fort_actuals_contiguous = is_contiguous({name})"]
            if reached is not None or descriptor_conditions:
                preflight += [f"if (fort_actuals_contiguous) fort_actuals_contiguous = {condition}"
                              for condition in parent.builder.original_bound_conditions(name, mapping.binding.rank)]
        preflight += ["if (.not. fort_actuals_contiguous) then", *parent.close(prework=True), "endif"]
        if descriptor_conditions:
            preflight += [f"if ({parent.enabled}) then",
                          *[f"if (fort_actuals_contiguous) fort_actuals_contiguous = {condition}"
                            for condition in descriptor_conditions],
                          "if (.not. fort_actuals_contiguous) then",
                          *parent.close(prework=True, reason="capture expected descriptor mismatch"), "endif", "endif"]
        preflight += ["end block"]
    if reached is not None or registrations or native_arrays:
        from compiler.scopes.call_registration import register_original_actuals
        # A zero handle must become an undefined tracked range at OUT entry,
        # rather than being freshly seeded from caller facts after return.
        preflight += register_original_actuals(parent, registrations,
                                              scalar_elements=elements, native_arrays=native_arrays)
    lines = [*parent.rearm_epoch(), f"if ({parent.enabled}) then", *preflight, "endif",
             f"if ({parent.enabled}) then", *_call(existing.entry, [*arguments, *controls]),
             "else", *parent.original((node,)).splitlines(), "endif"]
    if routine.scope.module != parent.routine.scope.module:
        # The import belongs to the checked replacement, just like its call.
        # Admission must not edit the original procedure: native placement or
        # capture discards this patch together with the companion definition.
        lines = ["block", f"use {routine.scope.module}, only: {existing.entry}", *lines, "end block"]
    if parent.control_guard:
        lines = [f"if ({parent.control_guard}) then", *lines, "else", *parent.original((node,)).splitlines(), "endif"]
    from compiler.scopes.provenance import borrowed
    lines = borrowed(parent, (node,), lines)
    parent.patches.append((first, last, "\n".join(fortran_lines(lines))+"\n"))
    if reached is not None:
        patches = getattr(parent, "mapped_call_patches", {})
        parent.mapped_call_patches = patches
        patches[first, last] = reached[1].node_id
    existing.call_sites.append({"caller": parent.routine.qualified, "first_line": first, "last_line": last,
                               **({"resource_mappings": {root: binding.root for root, binding in mappings.items()},
                                   "handle_bindings": {str(slot): number for slot, number in handle_bindings.items()}}
                                  if parameterized else {}),
                               "mapped_call_node_id": reached[1].node_id if reached is not None else None,
                               "coordinate_proof_identity": reached[1].coordinates.identity if reached is not None else None})
    parent.builder.resolved_calls[(parent.routine.qualified, (first, last))] = resolved.public()
    return True
