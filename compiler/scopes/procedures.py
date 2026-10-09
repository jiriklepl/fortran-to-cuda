"""Bounded source coordinators which borrow an existing ownership context.

The original native entry and storage remain untouched. A coordinator retains
the original control flow and delegates placement only when each segment is
reached. It never provides a speculative whole-procedure planning query.
"""

from copy import copy
from types import MethodType

from fparser.two import Fortran2003 as F

from compiler.frontend.source_effects import _children, _kind, _part
from compiler.ir import CompilationError


class BorrowedCoordinator:
    def __init__(self, builder, procedure, active):
        from compiler.scopes.region_dispatch import InlineRegions
        from compiler.scopes.segments import StructuredScope

        self.root = getattr(builder, "_coordinator_root", builder)
        self.routine = builder.analysis.routines[procedure]
        self.procedure = procedure
        self.summary = builder.analysis.summarize(procedure)
        if (not self.summary["complete"] or self.routine.source_kind != "module"
                or self.summary["persistent_state"]):
            raise CompilationError("borrowed source coordinator requires complete effects and original unsaved storage")
        if any(name.startswith("fort_") for name in self.routine.scope.bindings):
            raise CompilationError("borrowed coordinator conflicts with the generated helper namespace")
        if any(builder.analysis._binding(self.routine.scope, name) for name in ("c_int", "c_int64_t", "c_size_t")):
            raise CompilationError("borrowed coordinator conflicts with generated ISO C kind names")
        if (builder.analysis._binding(self.routine.scope, "lbound")
                or builder.analysis._candidates(self.routine.scope, F.Name("lbound"))):
            raise CompilationError("borrowed coordinator requires original LBOUND intrinsic authority")
        for name in self.routine.arguments:
            binding = self.routine.scope.bindings[name]
            if binding.dtype not in {"real", "integer", "logical"} or binding.attributes & {
                    "optional", "pointer", "volatile", "asynchronous", "value"}:
                raise CompilationError("borrowed coordinator requires fixed numeric original arguments")
        self.builder = copy(builder)
        self.builder.entry = self.routine
        self.builder._coordinator_root = self.root
        self.builder.inline = InlineRegions(self.builder)

        def borrowed_capture(proxy, binding):
            if binding.root.startswith("argument::") and binding.name in self.routine.arguments:
                # A borrowed handle supplies allocation identity and definition
                # coverage. This symbolic record is not an application lifetime
                # assertion; parent requirements are checked through actuals.
                return {"storage": "stable", "initialized": "whole", "escapes": False,
                        "allocation_changes": False, "borrowed_context": True}
            return self.root.capture(binding)

        self.builder.capture = MethodType(borrowed_capture, self.builder)
        nodes = tuple(_children(self.routine.execution))
        if not nodes:
            raise CompilationError("borrowed coordinator has no reached source operations")
        # This first coordinator reuses existing numerical callees. Inline
        # overlays in other procedures need their own dispatch authority and
        # remain an explicit later extension.
        self.scope = StructuredScope(self.builder, nodes)
        self.leaves = {leaf for call in self.scope.calls
                       for leaf in self.builder.closure(call.procedure, active)[0]}
        if not self.leaves:
            raise CompilationError("borrowed coordinator has no proved numerical callee")
        self.arrays, self.scalars, self.written = {}, {}, set()
        effects, _definitions, _overwrites = builder.native_effects(procedure)
        for name in self.routine.arguments:
            binding = self.routine.scope.bindings[name]
            if binding.rank:
                self.arrays[binding.root] = binding
            else:
                self.scalars[binding.root] = binding
        for root in effects:
            self.arrays[root] = self.builder.visible_binding(self.routine, root)
        self.scope.inputs(self.arrays, self.scalars, self.written)
        for binding in self.arrays.values():
            if binding.root.startswith(self.procedure + "::"):
                raise CompilationError("procedure-local array state requires registration in its original owner")
        self.scope.check_conservative_definitions(self.arrays)
        self.scope.prepare(self.arrays)
        # The current public numerical query publishes payload controls as
        # whole arrays. A borrowed handle supplies identity, not a freshness or
        # definition assertion; propagate those requirements to original actuals.
        self.required_whole = {root for segment in self.scope.segments for root in segment.payload}
        for operation in self.scope.native:
            if not operation.sections.available:
                self.required_whole.update(set(operation.effects) - operation.overwrites)
        for call in self.scope.calls:
            child = self.builder.coordinator(call.procedure)
            if child is not None:
                child.check_aliases(call)
                for root in child.required_whole:
                    self.required_whole.add(call.bindings[root].root if root in call.bindings else root)
            elif not self.builder.closure(call.procedure, active)[0]:
                actions, _definitions, overwrites = self.builder.roots_for(call)
                self.builder.check_native_definitions(call, *self.builder.call_effects(call))
                if not self.builder.analysis.native_sections(call.procedure).available:
                    self.required_whole.update(set(actions) - overwrites)
        entry_undefined = {self.routine.scope.bindings[name].root for name in self.routine.arguments
                           if self.routine.scope.bindings[name].rank and self.routine.scope.bindings[name].intent == "out"}
        if self.required_whole & entry_undefined:
            raise CompilationError("borrowed query or conservative native effects cannot preserve original INTENT(OUT) definitions")
        self.resources = tuple(self.arrays)

    def refresh(self):
        # Failed outer candidates roll back output dictionaries. Keep typed
        # source proofs, while always writing artifacts into the current root.
        for name in ("outputs", "edits", "clones", "queries", "generated", "numerical_reasons",
                     "numerical_ir", "batch_chains", "variants", "view_generated", "view_clones", "view_queries"):
            setattr(self.builder, name, getattr(self.root, name))

    def public(self):
        graph = self.builder.analysis.structure(self.procedure)
        scope = self.scope.public()
        return {"procedure": self.procedure, "role": "borrowed_reached_coordinator",
                "structured_summary_identity": graph.identity,
                "summary_identity": self.summary["summary_identity"],
                "resources": list(self.resources), "gpu_leaves": sorted(self.leaves),
                "requirements": {"original_numeric_arguments": True, "joined_native_operations": True,
                                 "unique_canonical_resource_mappings": True,
                                 "whole_initialized_resources": sorted(self.required_whole),
                                 "context": "borrowed from the owning invocation; never closed by this worker"},
                "planning": "reached segments only; no whole-procedure query or condition hoisting",
                "cost_estimate": "reached segment with eventual owner publication; complete child estimate unavailable",
                "ownership": {"lifetime": "borrowed owning invocation", "close_count": 0,
                              "retained_resources": list(self.resources)},
                "planning_segments": scope["planning_segments"],
                "native_operations": scope["native_operations"],
                "source_tree": scope["structured_tree"]}

    def check_actuals(self, call):
        self.check_aliases(call)
        mapping = {formal: binding for formal, binding in call.bindings.items()}
        for resource in self.required_whole:
            binding = mapping.get(resource, self.arrays[resource])
            if self.root.capture(binding)["initialized"] != "whole":
                raise CompilationError("borrowed queries or conservative native effects require whole initialized actual storage: " + binding.root)

    def check_aliases(self, call):
        mapped = [call.bindings[root].root if root in call.bindings else root for root in self.resources]
        if len(set(mapped)) != len(mapped):
            # Ordinary leaf hooks merge read-only roots. A reached native
            # fragment needs the same merge before two child formals can safely
            # acquire one canonical handle, even when no writes alias.
            raise CompilationError("borrowed coordinator aliases require merged reached native effects")

    def worker(self):
        from compiler.emission.fortran.formatting import _fortran_list
        from compiler.scopes.segments import fortran_lines
        from compiler.scopes.source import _checked, _name

        self.refresh()
        builder, routine = self.builder, self.routine
        if self.procedure in self.root.clones:
            return self.root.clones[self.procedure]
        name = _name("fort_scope_coordinator_", self.procedure)
        handles = {root: "fort_handle_" + str(index) for index, root in enumerate(self.resources)}
        self.root.clones[self.procedure] = name, list(self.resources)
        builder.variants.register(self.procedure, interface="borrowed_coordinator_v1", role="call_worker", name=name,
            summary_identity=self.summary["summary_identity"],
            requirements=("borrowed owning context", "reached source segments", "original native state retained"),
            shared_artifacts=("sources/" + _name("source_", str(routine.scope.path)) + ".f90",))
        header = _fortran_list("subroutine " + name + "(",
                               ["fort_context", "fort_mode", *routine.arguments, *handles.values()], ")", 0)
        imports = ["use iso_c_binding", "use fort_scoped_memory"]
        selector, selector_name = None, "fort_choose"
        if builder.config.policy == "auto":
            # This is a sibling of the owning helper, so its USE aliases do
            # not inherit that helper's calibrated selector import. Reuse the
            # same public protocol, with a private name checked in this scope.
            selector_name = _name("fort_choose_", self.procedure)
            if (builder.analysis._binding(routine.scope, selector_name)
                    or builder.analysis._candidates(routine.scope, F.Name(selector_name))):
                raise CompilationError("borrowed coordinator conflicts with the generated selector alias")
            selector, _ = builder.entry_artifacts(sorted(self.leaves)[0])
            imports.append(f"use {selector['fortran_module']}, only: {selector_name} => "
                           f"{selector['planning']['fortran_selector']}")
        specification = []
        for declaration in _children(_part(routine.scope.node, "Specification_Part")):
            if _kind(declaration) == "Use_Stmt":
                imports.append(str(declaration))
            elif _kind(declaration) != "Type_Declaration_Stmt":
                specification.append(str(declaration))
            else:
                dtype, attributes, entities = declaration.items
                for entity in _children(entities):
                    variable = str(entity.items[0]).lower()
                    flags = [str(attribute) for attribute in _children(attributes)]
                    if variable in routine.arguments and routine.scope.bindings[variable].rank:
                        if not any(flag.upper() == "TARGET" for flag in flags):
                            flags.append("TARGET")
                    specification += _fortran_list(str(dtype) + (", " + ", ".join(flags) if flags else "") + " ::",
                                                    [str(entity)], "", 0)
        specification += ["integer(c_int64_t), intent(in) :: fort_context",
                          "integer(c_int), intent(in) :: fort_mode",
                          *_fortran_list("integer(c_int64_t), intent(in) ::", list(handles.values()), "", 0),
                          "integer(c_int) :: fort_status", "logical :: fort_branch",
                          "type(fort_scope_access) :: fort_access", "type(fort_scope_plan_decision) :: fort_decision",
                          f"type(fort_scope_plan_binding), target :: fort_bindings({max(1, len(handles))})"]
        if self.scope.guarded:
            specification.append("logical :: fort_numerical_guard")
        # A dummy's bounds belong to its original entry descriptor. Snapshot
        # them here before native operations can change specification scalars;
        # the canonical registered allocation may have different logical bounds.
        dummy_lowers = {root: "fort_original_lower_" + str(index)
                        for index, (root, binding) in enumerate(self.arrays.items())
                        if binding.root.startswith("argument::") and "allocatable" not in binding.attributes}
        specification += [f"integer(c_int64_t) :: {value}({self.arrays[root].rank})"
                          for root, value in dummy_lowers.items()]
        parameters = {root: builder.visible(routine, root) for root in (*self.arrays, *self.scalars)}

        def actuals(call, *, shared=True):
            # These remain original local/formal references in this worker.
            # Original descriptors, lexical aliases and keyword order survive.
            from compiler.scopes.segments import renamed
            return [renamed(builder, actual, parameters) if actual is not None else "None" for actual in call.actuals]

        body = [f"{value} = lbound({builder.visible(routine, root)}, kind=c_int64_t)"
                for root, value in dummy_lowers.items()]
        for variable in routine.arguments:
            binding = routine.scope.bindings[variable]
            if binding.rank and binding.intent == "out":
                body += _checked(f"fort_scope_forget_definition(fort_context, {handles[binding.root]})")
        body += self.scope.emit(handles, parameters, actuals, imports, selector=selector,
                                selector_name=selector_name, execution_mode="fort_mode",
                                terminal_owner=False,
                                logical_lower_bounds={root: tuple(f"{value}({axis})"
                                                                 for axis in range(1, self.arrays[root].rank + 1))
                                                      for root, value in dummy_lowers.items()})
        builder.append_procedure(routine.scope.parent, name, "\n".join(fortran_lines([
            *header, *dict.fromkeys(imports), *specification, *body, "end subroutine " + name, ""])))
        return name, list(self.resources)
