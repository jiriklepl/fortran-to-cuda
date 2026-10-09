"""Reached operations in an original lexical owner, with no prefix replay.

The original procedure owns its declarations and control flow. Short blocks
borrow its TARGET storage and a single invocation context. A reached boundary
publishes and closes that context before the unchanged source operation runs.
No synthetic array dummy remains associated across a native escape.
"""

from copy import copy
from dataclasses import dataclass

from fparser.two.utils import walk

from compiler.frontend.source_effects import _children, _kind, _part
from compiler.ir import CompilationError
from compiler.scopes.segments import StructuredScope, condition_fragment, fortran_lines, statement_span
from compiler.scopes.source import DTYPES, _name, _span


def check_structured_entry(routine):
    """A branch target must not bypass owner setup or enter a synthetic block."""
    if any(getattr(getattr(node, 'item', None), 'label', None) is not None
           for node in walk(routine.execution)):
        raise CompilationError('reached ownership requires structured entry; source statement labels remain native')


@dataclass
class Unit:
    nodes: tuple
    scope: object
    arrays: dict
    scalars: dict
    written: set
    leaves: set
    definition_summary: object
    source_guard: str | None = None
    native_source: str | None = None
    publication_only: bool = False
    host_metadata: object = None


class LexicalOwner:
    """One bounded source owner; arbitrary native continuation runs in place."""

    def __init__(self, builder, *, parent=None):
        self.builder = builder
        self.routine = builder.entry
        self.parent = parent
        self.owner = parent.owner if parent is not None else self
        if parent is None:
            self.members = {self.routine.qualified: self}
            self.active = []
        self.internal_calls = []
        self.graph = builder.analysis.structure(self.routine.qualified)
        if not self.graph.available:
            raise CompilationError("lexical owner requires an available bounded source graph")
        self.context = parent.context if parent else _name("fort_owner_context_", self.routine.qualified)
        self.enabled = parent.enabled if parent else _name("fort_owner_enabled_", self.routine.qualified)
        self.kind = parent.kind if parent else _name("fort_owner_kind_", self.routine.qualified)
        self.resources = parent.resources if parent else {}
        self.canonical_bindings = parent.canonical_bindings if parent else {}
        self.control_guard = parent.control_guard if parent else None
        self.forwarded_handles = parent.forwarded_handles if parent else set()
        self.units, self.boundaries, self.patches = [], [], []
        self.lines = self.routine.scope.path.read_text().splitlines(keepends=True)
        check_structured_entry(self.routine)
        statement = _part(self.routine.scope.node, "Subroutine_Stmt")
        if (statement is None or statement.items[3] is not None
                or any(str(item).lower() in {"pure", "elemental"} for item in _children(statement.items[0]))
                or any(_kind(item) == "Entry_Stmt" for part in
                       (_part(self.routine.scope.node, "Specification_Part"), self.routine.execution)
                       for item in walk(part))):
            raise CompilationError("lexical owner requires one ordinary impure subroutine entry")
        reserved = {"c_int", "c_int64_t", "c_size_t", "c_ptr", "c_loc", "c_null_ptr",
                    "c_double", "c_float", "c_bool", "is_contiguous", "lbound", "ubound", "size", "shape",
                    "allocated", "all", "int"}
        if any(name.startswith("fort_") for name in self.routine.scope.bindings):
            raise CompilationError("lexical owner conflicts with original compiler-control names")
        for name in reserved:
            if builder.analysis._binding(self.routine.scope, name) or builder.analysis._candidates(self.routine.scope, name):
                raise CompilationError("lexical owner requires unshadowed runtime intrinsic names: " + name)

    def refresh(self):
        if self.parent is not None:
            self.builder.inline.refresh(self.owner.builder)

    def canonical(self, binding):
        return self.canonical_bindings.get(binding.root, binding)

    def add_resource(self, binding):
        canonical = self.canonical(binding)
        _, number = self.owner.resources.setdefault(canonical.root, (canonical, len(self.owner.resources)+1))
        self.resources.setdefault(binding.root, (binding, number))

    def handle_numbers(self):
        return sorted(self.forwarded_handles | {number for _, number in self.resources.values()})

    def internal(self, node):
        """Instrument a direct lexical child in place, without copying state.

        The initial ABI has no child formals: all captures keep their original
        host-associated descriptor and binding identity. Other calls continue
        through ordinary mapped workers or a reached ownership boundary.
        """
        if hasattr(node, "fort_inline_region"):
            return False
        from compiler.scopes.native_guard import borrow
        if borrow(self, node):
            return True
        try:
            resolved = self.builder.execution_source_call(self.routine, node)
        except CompilationError:
            return False
        child = self.builder.analysis.routines[resolved.procedure]
        if child.source_kind != "internal" or child.scope.parent is not self.routine.scope:
            return False
        if child.arguments or resolved.actuals:
            raise CompilationError("lexical child formals require reached descriptor mappings")
        owner = self.owner
        if child.qualified in owner.active or len(owner.active) >= self.builder.analysis.depth_limit:
            raise CompilationError("recursive or over-depth lexical child execution")
        if child.qualified not in owner.members:
            if len(owner.members) >= self.builder.analysis.procedure_limit:
                raise CompilationError("lexical child procedure budget exhausted")
            from compiler.scopes.region_dispatch import InlineRegions
            proxy = copy(self.builder)
            proxy.entry = child
            proxy.inline = InlineRegions(proxy)
            nested = LexicalOwner(proxy, parent=self)
            owner.members[child.qualified] = nested
            self.builder.region_owners[child.qualified] = proxy.inline
            owner.active.append(child.qualified)
            try:
                nodes = proxy.inline.prepare(_children(child.execution))
                proxy.generated.update(proxy.inline.generated)
                proxy.numerical_ir.update(proxy.inline.ir)
                proxy.numerical_reasons.update({name: None for name in proxy.inline.regions})
                nested.scan(nodes)
            except CompilationError:
                # Nothing is emitted until every reached operation is either
                # proved or a local close. Remove incomplete child metadata.
                del owner.members[child.qualified]
                raise
            finally:
                owner.active.pop()
        span = statement_span(node)
        record = {**resolved.public(), "caller": self.routine.qualified, "source": str(self.routine.scope.path),
                  "first_line": span[0], "last_line": span[1],
                  "execution": "original internal procedure; host-associated owner control"}
        self.internal_calls.append(record)
        self.builder.resolved_calls[(self.routine.qualified, span)] = record
        return True

    def original(self, nodes):
        first, last = statement_span(nodes[0])[0], statement_span(nodes[-1])[1]
        return "".join(self.lines[first-1:last])

    def close(self, *, disable=True):
        lines = ["block", "use iso_c_binding, only: c_int", "use fort_scoped_memory, only: fort_scope_close, FORT_SCOPE_OK",
                 "integer(c_int) :: fort_close_status", f"if ({self.context} /= 0) then",
                 f"fort_close_status = fort_scope_close({self.context})",
                 "if (fort_close_status /= FORT_SCOPE_OK) error stop 'lexical owner publication failed'",
                 f"{self.context} = 0", "endif"]
        if disable:
            lines.append(f"{self.enabled} = .false.")
        lines += ["end block"]
        return [f"if ({self.control_guard}) then", *lines, "endif"] if self.control_guard else lines

    def boundary(self, nodes, reason):
        first, last = statement_span(nodes[0])[0], statement_span(nodes[-1])[1]
        reached_guard = None
        if len(nodes) == 1 and _kind(nodes[0]) == "If_Stmt":
            try:
                condition = condition_fragment(self.builder, nodes[0])
                if not condition.effects:
                    reached_guard = str(nodes[0].items[0])
            except CompilationError:
                pass
        self.boundaries.append({"first_line": first, "last_line": last, "reason": reason,
                                "execution": "publish and close when reached; original continuation once",
                                "conditional_close": reached_guard,
                                "reopen": False})
        if reached_guard is not None:
            replacement = "\n".join(fortran_lines([f"if ({reached_guard}) then", *self.close(),
                                                   str(nodes[0].items[1]), "endif"])) + "\n"
        else:
            replacement = "\n".join(fortran_lines(self.close())) + "\n" + self.original(nodes)
        self.patches.append((first, last, replacement))

    def admit(self, nodes, *, source_guard=None, native_source=None, boundary_nodes=None, condition_only=False):
        self.refresh()
        before = self.owner.builder.scope_checkpoint()
        try:
            scope = StructuredScope(self.builder, nodes, condition_only=condition_only,
                                    native_metadata=True, lexical_native=True)
            from compiler.scopes.views import view_call
            if any(view_call(call) for call in scope.calls):
                raise CompilationError("lexical rectangular calls require reached view preflight")
            arrays, scalars, written = self.builder.owner_inputs(scope.calls, reached_definitions=True)
            scope.inputs(arrays, scalars, written)
            # Descriptor lifetime, association, numeric type and initialization
            # evidence are checked before emitting any registration artifact.
            for binding in arrays.values():
                self.builder.capture(binding)
                self.declaration(binding)
            metadata = {root: binding for operation in scope.native for root, binding in operation.host_metadata.items()}
            if metadata:
                for name in ('storage_size', 'mod'):
                    if (self.builder.analysis._binding(self.routine.scope, name)
                            or self.builder.analysis._candidates(self.routine.scope, name)
                            or self.builder.analysis._unknown_exports(self.routine.scope)):
                        raise CompilationError('native metadata needs an unshadowed ' + name.upper() + ' intrinsic')
            for binding in metadata.values():
                self.declaration(binding)
            scope.prepare(arrays)
            from compiler.scopes.definitions import summarize
            definitions = summarize(scope)
            leaves = {leaf for call in scope.calls for leaf in self.builder.closure(call.procedure)[0]}
            if not arrays and not leaves and not metadata:
                return  # Original scalar operations need no runtime context.
            self.builder.scope_budget(scope.calls)
            unit = Unit(tuple(nodes), scope, arrays, scalars, written, leaves, definitions,
                        source_guard, "" if condition_only else native_source, condition_only, metadata)
            for root, binding in {**arrays, **metadata}.items():
                self.add_resource(binding)
            self.units.append(unit)
        except CompilationError as error:
            self.owner.builder.restore_scope_checkpoint(before)
            self.refresh()
            if condition_only:
                raise
            if source_guard is None and len(nodes) == 1 and _kind(nodes[0]) == "Call_Stmt":
                try:
                    from compiler.scopes.module_owner import borrow_module
                    if borrow_module(self, nodes[0]):
                        return
                except CompilationError as companion_error:
                    error = CompilationError(str(error) + "; original-body continuation: " + str(companion_error))
            self.boundary(tuple(boundary_nodes or nodes), str(error))

    def scan(self, nodes, depth=0):
        if depth > self.builder.analysis.depth_limit:
            self.boundary(tuple(nodes), "bounded lexical control depth exhausted")
            return
        for group in self.builder.inline.grouped_nodes(nodes):
            self.refresh()
            if isinstance(group, tuple):
                self.admit(group)
                continue
            kind = _kind(group)
            if kind == "Comment":
                continue
            if kind == "If_Construct":
                try:
                    publication = None
                    for header in group.content:
                        if _kind(header) in {"If_Then_Stmt", "Else_If_Stmt"}:
                            proof = condition_fragment(self.builder, header)
                            if proof.effects:
                                if _kind(header) == "Else_If_Stmt":
                                    raise CompilationError("ELSEIF payload publication requires nested original guards")
                                publication = header
                    if publication is not None:
                        self.admit((publication,), condition_only=True)
                except CompilationError as error:
                    self.boundary((group,), str(error))
                    continue
                body = []
                for item in group.content:
                    if _kind(item) in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                        if body:
                            self.scan(body, depth+1)
                        body = []
                    else:
                        body.append(item)
            elif kind == "Call_Stmt":
                try:
                    if not self.internal(group):
                        if not hasattr(group, "fort_inline_region"):
                            try:
                                call = self.builder.execution_source_call(self.routine, group)
                                child = self.builder.analysis.routines[call.procedure]
                                if (child.source_kind == "module" and any(_kind(item) == "Return_Stmt"
                                        for item in walk(child.execution))):
                                    from compiler.scopes.module_owner import borrow_module
                                    if borrow_module(self, group):
                                        continue
                            except CompilationError:
                                # Complete native effects may still provide a
                                # safe ordinary call when a companion cannot.
                                pass
                        self.admit((group,))
                except CompilationError as error:
                    self.boundary((group,), str(error))
            elif kind == "If_Stmt":
                try:
                    proof = condition_fragment(self.builder, group)
                    if proof.effects:
                        self.admit((group,), condition_only=True)
                    if (self.parent is not None and _kind(group.items[1]) == "Return_Stmt"
                            and str(group.items[1]).strip().lower() == "return"):
                        # Returning from a borrowed child ends no managed
                        # allocation lifetime. Its outer owner remains live.
                        continue
                    action = self.builder.inline.statement_projection(group.items[1], group)
                    self.admit((action,), source_guard=str(group.items[0]), native_source=str(group.items[1]),
                               boundary_nodes=(group,))
                except CompilationError as error:
                    self.boundary((group,), str(error))
            elif kind == "Associate_Construct":
                original = self.builder.inline.original_selection(group)
                record = (self.builder.analysis._associate_scopes.get(id(original[0]))
                          if len(original) == 1 else None)
                if record is None or not record.available:
                    self.boundary((group,), "lexical association requires its original selector proof")
                else:
                    # Retain the original association and edit individual
                    # reached operations inside it. A later failure cannot
                    # restart the association's earlier numerical prefix.
                    self.scan(group.content[1:-1], depth+1)
            elif kind in {"Assignment_Stmt", "Block_Nonlabel_Do_Construct"}:
                self.admit((group,))
            elif kind == "Return_Stmt" and self.parent is not None and str(group).strip().lower() == "return":
                continue
            else:
                self.boundary((group,), "unsupported reached source operation: " + kind)

    def declaration(self, binding):
        """Locate actual storage; adding TARGET never moves its ownership."""
        binding = getattr(binding, "component_object", binding)
        scopes = list(self.builder.analysis.modules.values())
        current = self.routine.scope
        while current is not None:
            scopes.append(current)
            current = current.parent
        owner = next((scope for scope in scopes if scope.bindings.get(binding.name) is binding), None)
        if owner is None:
            raise CompilationError("lexical storage declaration is unavailable in its original owner: " + binding.root)
        if (owner is self.routine.scope and self.parent is not None
                and binding.name not in self.routine.arguments):
            raise CompilationError("child-local array lifetime does not span the lexical owner: " + binding.root)
        if binding.attributes & {"parameter", "pointer", "optional", "volatile", "asynchronous", "value"}:
            raise CompilationError("lexical TARGET association is not proved: " + binding.root)
        for declaration in _children(_part(owner.node, "Specification_Part")):
            if _kind(declaration) == "Type_Declaration_Stmt" and any(
                    str(entity.items[0]).lower() == binding.name for entity in _children(declaration.items[2])):
                return owner, declaration
        raise CompilationError("lexical storage lacks an original declaration: " + binding.root)

    def emit_unit(self, unit, index):
        self.refresh()
        builder, arrays = self.builder, unit.arrays
        block = "fort_reached_" + str(index)
        handles = {root: "fort_buffer_" + str(self.resources[root][1]) for root in arrays}
        parameters = {root: builder.visible(self.routine, root) for root in (*arrays, *unit.scalars)}
        direct = {root for root, binding in arrays.items()
                  if "protected" in binding.attributes and root not in unit.written}
        # A use-associated PROTECTED object cannot be the target of Fortran
        # pointer assignment. Read-only workers can borrow the original object
        # directly; registration still uses its guarded original descriptor.
        views = {root: parameters[root] if root in direct else "fort_view_" + str(self.resources[root][1])
                 for root in arrays}
        values = {**parameters, **views}
        imports = ["use iso_c_binding", "use fort_scoped_memory"]
        spec = ["integer(c_int) :: fort_status", "logical :: fort_can, fort_branch, fort_numerical_guard",
                "type(c_ptr) :: fort_pointer", "type(fort_scope_layout) :: fort_layout",
                "type(fort_scope_access) :: fort_access", "type(fort_scope_plan_decision) :: fort_decision",
                f"type(fort_scope_plan_binding), target :: fort_bindings({max(1,len(arrays))})",
                f"integer(c_int), parameter :: fort_mode = {1 if builder.config.policy == 'sections' else 2}"]
        prepare = ["fort_status = FORT_SCOPE_OK", "fort_can = .true."]
        for root, binding in arrays.items():
            name, number = parameters[root], self.resources[root][1]
            dtype, enum, width = DTYPES[binding.signature()[:2]]
            spec += [f"integer(c_size_t), target :: fort_extents_{number}({binding.rank})",
                     f"integer(c_int64_t), target :: fort_lowers_{number}({binding.rank})"]
            if root not in direct:
                spec += [f"{dtype}, pointer, contiguous :: {views[root]}({','.join(':' for _ in range(binding.rank))})"]
            if "allocatable" in binding.attributes:
                prepare += [f"if (fort_can) fort_can = allocated({name})"]
            prepare += [f"if (fort_can) fort_can = is_contiguous({name})"]
            prepare += [f"if (fort_can) fort_can = {condition}" for condition in builder.original_bound_conditions(name,binding.rank)]
        prepare += ["if (.not. fort_can) then", *self.close(), "exit " + block, "endif",
                    "if (fort_context == 0) then", "fort_status = fort_scope_create(0_c_int, fort_context)",
                    "if (fort_status == FORT_SCOPE_OK) &",
                    f"fort_status = fort_scope_set_device_budget(fort_context, {builder.device_budget}_c_size_t)",
                    *builder.owner_transfer_setup(self.owner.leaves, imports), "endif"]
        for root, binding in arrays.items():
            name, number = parameters[root], self.resources[root][1]
            _, enum, width = DTYPES[binding.signature()[:2]]
            fact = builder.capture(binding)
            if root not in direct:
                prepare += [f"{views[root]} => {name}"]
            prepare += [f"if (fort_status == FORT_SCOPE_OK .and. {handles[root]} == 0) then",
                        f"fort_extents_{number} = shape({name}, kind=c_size_t)",
                        f"fort_lowers_{number} = lbound({name}, kind=c_int64_t)",
                        "fort_pointer = c_null_ptr",
                        f"if (all(fort_extents_{number} > 0)) fort_pointer = c_loc({name})",
                        f"fort_layout = fort_scope_layout({binding.rank}, {enum}, {width}_c_size_t, &",
                        f"fort_pointer, c_loc(fort_extents_{number}), c_loc(fort_lowers_{number}), 1_c_int64_t)"]
            if fact["initialized"] == "sections":
                sections = fact["sections"]
                spec += [f"type(fort_scope_section), target :: fort_defined_{number}({len(sections)})"]
                for ordinal, section in enumerate(sections, 1):
                    for bound in ("lower", "upper"):
                        variable = f"fort_defined_{number}_{ordinal}_{bound}"
                        spec += [f"integer(c_size_t), target :: {variable}({binding.rank})"]
                        prepare += [variable + " = [" + ",".join(f"{v}_c_size_t" for v in section[bound]) + "]",
                                    f"fort_defined_{number}({ordinal})%{bound} = c_loc({variable})"]
                pointer = f"c_loc(fort_defined_{number})" if sections else "c_null_ptr"
                prepare += [f"fort_status = fort_scope_register_sections(fort_context, {number}_c_int64_t, &",
                            f"1_c_int64_t, fort_layout, {pointer}, {len(sections)}_c_size_t, {handles[root]})"]
            else:
                prepare += [f"fort_status = fort_scope_register(fort_context, {number}_c_int64_t, &",
                            f"1_c_int64_t, fort_layout, {int(fact['initialized']=='whole')}_c_int, {handles[root]})"]
            prepare += ["endif"]
        failure = [*self.close(), "exit " + block]
        prepare += ["if (fort_status /= FORT_SCOPE_OK) then", *failure, "endif"]
        # Native derived-object fields keep their original representation.
        # An undefined opaque registration reserves the address range for
        # alias checking only: no host/device access or transfer is requested.
        for root, binding in (unit.host_metadata or {}).items():
            name, number = builder.visible(self.routine, root), self.resources[root][1]
            spec += [f"integer(c_size_t), target :: fort_meta_extents_{number}({binding.rank})",
                     f"integer(c_int64_t), target :: fort_meta_lowers_{number}({binding.rank})",
                     f"integer(c_size_t) :: fort_meta_bits_{number}"]
            if 'allocatable' in binding.attributes:
                prepare += [f"if (allocated({name})) then"]
            prepare += [f"if (fort_status == FORT_SCOPE_OK .and. fort_buffer_{number} == 0) then",
                        f"if (.not. is_contiguous({name})) then", *failure, "endif",
                        f"fort_meta_extents_{number} = shape({name}, kind=c_size_t)",
                        f"fort_meta_lowers_{number} = lbound({name}, kind=c_int64_t)",
                        f"fort_meta_bits_{number} = storage_size({name}, kind=c_size_t)",
                        f"if (fort_meta_bits_{number} <= 0 .or. mod(fort_meta_bits_{number}, 8_c_size_t) /= 0) then",
                        *failure, "endif", "fort_pointer = c_null_ptr",
                        f"if (all(fort_meta_extents_{number} > 0)) fort_pointer = c_loc({name})",
                        f"fort_layout = fort_scope_layout({binding.rank}, FORT_SCOPE_BYTES, &",
                        f"fort_meta_bits_{number}/8_c_size_t, fort_pointer, c_loc(fort_meta_extents_{number}), &",
                        f"c_loc(fort_meta_lowers_{number}), 1_c_int64_t)",
                        f"fort_status = fort_scope_register(fort_context, {number}_c_int64_t, &",
                        f"1_c_int64_t, fort_layout, 0_c_int, fort_buffer_{number})", "endif"]
            if 'allocatable' in binding.attributes:
                prepare += ['endif']
            prepare += ["if (fort_status /= FORT_SCOPE_OK) then", *failure, "endif"]
        required = sorted(unit.definition_summary.required_whole)
        if required:
            # Validate the child's entry obligations against reached coverage,
            # without publishing host data or using original capture freshness.
            # This runs before any operation in this unit: failure can close
            # and execute its original source once, never replay a GPU prefix.
            prepare += ["fort_status = fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)"]
            for index, root in enumerate(required, 1):
                prepare += [f"fort_bindings({index}) = fort_scope_plan_binding()",
                            f"fort_bindings({index})%buffer = {handles[root]}",
                            f"fort_bindings({index})%access%flags = FORT_SCOPE_READ_ALL"]
            prepare += ["if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_add( &",
                        "fort_context, FORT_SCOPE_PLAN_NATIVE, 0_c_int64_t, &",
                        f"c_loc(fort_bindings), {len(required)}_c_size_t, 0.0_c_double, 0.0_c_double, 0_c_int)",
                        "if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_validate(fort_context)",
                        "if (fort_status == FORT_SCOPE_OK) &",
                        "fort_status = fort_scope_plan_reset_mode(fort_context, FORT_SCOPE_PLAN_CONTINUE)",
                        "if (fort_status /= FORT_SCOPE_OK) then", *failure, "endif"]

        def actuals(call, *, shared=True):
            from compiler.scopes.segments import renamed
            return [renamed(builder, value, values) if value is not None else "None" for value in call.actuals]

        selector = None
        if builder.config.policy == "auto":
            selector, _ = builder.entry_artifacts(sorted(self.owner.leaves)[0])
            imports += [f"use {selector['fortran_module']}, only: fort_choose => {selector['planning']['fortran_selector']}"]
        preserve_original = any(operation.preserve_original for operation in unit.scope.native)
        execution = unit.scope.emit(handles, values, actuals, imports, selector=selector,
                                    terminal_owner=False, preflight_failure=failure,
                                    surround_native=preserve_original,
                                    logical_lower_bounds={root: tuple(
                                        f"lbound({parameters[root]}, {axis}, kind=c_int64_t)"
                                        for axis in range(1, binding.rank+1))
                                        for root, binding in arrays.items()})
        first, last = unit.scope.first, unit.scope.last
        if preserve_original:
            if unit.source_guard is not None or unit.publication_only:
                raise CompilationError("original native group cannot use a synthetic action guard")
            before, after = execution
            # The original native block remains in the file exactly once,
            # including CPP directives/includes. A failed preflight leaves the
            # wrapper, closes the owner, then reaches that original block. The
            # native ABI of a companion never reads absent control arguments.
            prefix = ["block", *dict.fromkeys(imports), *spec,
                      "logical :: fort_native_ready", "fort_native_ready = .false."]
            if self.control_guard:
                prefix += [f"if ({self.control_guard}) then"]
            prefix += [f"if ({self.enabled}) then", f"associate(fort_context => {self.context})",
                       block + ": block", *prepare, *before, "fort_native_ready = .true.",
                       "end block " + block, "end associate", "endif"]
            if self.control_guard:
                prefix += ["endif"]
            suffix = ["if (fort_native_ready) then", f"associate(fort_context => {self.context})",
                      *after, "end associate", "endif", "end block"]
            return [(first, first-1, "\n".join(fortran_lines(prefix))+"\n"),
                    (last+1, last, "\n".join(fortran_lines(suffix))+"\n")]
        original = unit.native_source if unit.native_source is not None else self.original(unit.nodes)
        lines = [f"if ({self.enabled}) then", f"associate(fort_context => {self.context})", block + ": block",
                 *dict.fromkeys(imports), *spec, *prepare, *execution, "end block " + block, "end associate", "endif",
                 f"if (.not. {self.enabled}) then", *original.splitlines(), "endif"]
        if unit.source_guard is not None:
            # Evaluate the original predicate once, before descriptor guards
            # and any fallback for its one original action.
            lines = [f"if ({unit.source_guard}) then", *lines, "endif"]
        if self.control_guard:
            lines = [f"if ({self.control_guard}) then", *lines,
                     *([] if unit.publication_only else ["else", *self.original(unit.nodes).splitlines()]), "endif"]
        return [(first, first-1 if unit.publication_only else last, "\n".join(fortran_lines(lines)) + "\n")]

    def run(self, nodes):
        builder = self.builder
        self.active.append(self.routine.qualified)
        self.scan(nodes)
        self.active.pop()
        self.leaves = {leaf for member in self.members.values() for unit in member.units for leaf in unit.leaves}
        if not self.leaves:
            return False
        # One shared implementation identity, independent of the count of
        # reached source segments. Original native control/storage stays here.
        variant = builder.variants.register(self.routine.qualified, interface="lexical_owner_v1", role="coordinator",
            name=self.context, summary_identity=self.graph.identity,
            requirements=("one original invocation", "reached descriptor guards", "native continuation without prefix replay"),
            shared_artifacts=("sources/" + _name("source_", str(self.routine.scope.path)) + ".f90",))
        target_declarations = {}
        for member in self.members.values():
            for unit in member.units:
                for binding in {**unit.arrays, **(unit.host_metadata or {})}.values():
                    owner, declaration = member.declaration(binding)
                    target_declarations[owner.path, _span(declaration)] = declaration
        for (path, (first,last)), declaration in target_declarations.items():
            dtype, attributes, entities = declaration.items
            flags = list(map(str, _children(attributes)))
            if any(flag.lower() == "target" for flag in flags):
                continue
            text = str(dtype) + ", " + ", ".join([*flags, "TARGET"]) + " :: " + str(entities)
            builder.add_edit(path, first,last, "\n".join(fortran_lines([text])) + "\n")
        for member in self.members.values():
            edits = [edit for index,unit in enumerate(member.units) for edit in member.emit_unit(unit,index)]
            for first,last,text in [*member.patches, *edits]:
                builder.add_edit(member.routine.scope.path, first,last,text)
            if member is not self:
                if hasattr(member, "emit_companion"):
                    member.emit_companion()
                    continue
                builder.variants.register(member.routine.qualified, interface="lexical_child_v1", role="coordinator",
                    name=member.routine.qualified, summary_identity=member.graph.identity,
                    requirements=("original internal procedure", "host-associated owning invocation control"),
                    shared_artifacts=("sources/" + _name("source_", str(member.routine.scope.path)) + ".f90",))
        start = _span(_part(self.routine.scope.node,"Subroutine_Stmt"))[1]+1
        first = statement_span(nodes[0])[0]
        last = statement_span(nodes[-1])[1]
        builder.add_edit(self.routine.scope.path, start,start-1,
                         f"use iso_c_binding, only: {self.kind} => c_int64_t\n")
        native_reason = builder.automatic_native_preflight(self.leaves)
        declarations = [f"integer({self.kind}) :: {self.context}", f"logical :: {self.enabled}",
                        *[f"integer({self.kind}) :: fort_buffer_{number}" for _binding, number in self.resources.values()]]
        initialize = [f"{self.context} = 0", *[f"fort_buffer_{number} = 0" for _binding,number in self.resources.values()],
                      "block", "use fort_scoped_memory, only: fort_scope_serial_caller",
                      f"{self.enabled} = " + (".false." if native_reason else "fort_scope_serial_caller() /= 0"), "end block"]
        builder.add_edit(self.routine.scope.path, first,first-1,
                         "\n".join(fortran_lines([*declarations,*initialize]))+"\n", prepend=True)
        builder.add_edit(self.routine.scope.path, last+1,last,"\n".join(fortran_lines(self.close(disable=False)))+"\n")
        builder.scopes.append({"owner": self.context, "owner_variant": variant.identity,
            "path": str(self.routine.scope.path), "first_line": first, "last_line": last,
            "gpu_leaves": sorted(self.leaves), "mode": builder.config.policy,
            "estimate_available": all(unit.scope.planning_status()[0] for member in self.members.values() for unit in member.units),
            "planning_reason": native_reason,
            "ownership": {"lifetime": "one original invocation", "planning_mode": "continuation",
                          "retained_resources": list(self.resources), "end_reason": "reached boundary or original invocation end"},
            "resource_bindings": {"schema_version": 1,
                "identity": "invocation-local canonical registration; matches runtime buffer trace identity",
                "resources": [{"resource": root, "registration_identity": number,
                               "descriptor": binding.public()}
                              for root, (binding, number) in self.resources.items()]},
            "source_authority": {"version": 1, "structured_summary_identity": self.graph.identity},
            "internal_coordinators": [{"procedure": member.routine.qualified,
                "structured_summary_identity": member.graph.identity,
                "control": "host association; original declarations and body",
                "calls": member.internal_calls, "boundaries": member.boundaries,
                "planning_segments": [{"segment_id": index, "first_line": unit.scope.first,
                    "last_line": unit.scope.last, "resources": list(unit.arrays),
                    "gpu_leaves": sorted(unit.leaves), "ordered_definitions": unit.definition_summary.public(),
                    "operations": unit.scope.public()}
                    for index, unit in enumerate(member.units)]}
                for member in self.members.values() if member is not self and not hasattr(member, "emit_companion")],
            "module_coordinators": [member.public() for member in self.members.values() if hasattr(member, "emit_companion")],
            "internal_calls": self.internal_calls,
            "planning_segments": [{"segment_id": index, "first_line": unit.scope.first, "last_line": unit.scope.last,
                                   "resources": list(unit.arrays), "gpu_leaves": sorted(unit.leaves),
                                   "ordered_definitions": unit.definition_summary.public(),
                                   "operations": unit.scope.public()} for index,unit in enumerate(self.units)],
            "boundaries": self.boundaries, "native_continuation": "original lexical scope; no replay and no synthetic argument association",
            "registration": "first reached use after original allocation and descriptor guards",
            "reopen_after_boundary": False})
        builder.scopes[-1]["transfer_configuration"] = builder.numerical(sorted(self.leaves)[0]).scoped["transfer_configuration"]
        builder.boundaries.extend(boundary for member in self.members.values() for boundary in member.boundaries)
        return True
