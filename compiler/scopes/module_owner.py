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
from compiler.scopes.segments import fortran_lines, statement_span
from compiler.scopes.source import _call, _name, _span


class ModuleOwner(LexicalOwner):
    def __init__(self, parent, routine, mappings, immutable_inputs=None):
        from compiler.scopes.region_dispatch import InlineRegions

        proxy = copy(parent.builder)
        proxy.entry = routine
        proxy.immutable_inputs = dict(immutable_inputs or {})
        proxy.inline = InlineRegions(proxy)
        super().__init__(proxy, parent=parent)
        if any("save" in binding.attributes for binding in routine.scope.bindings.values()):
            raise CompilationError("original module companion saved local state requires a separate lifetime proof")
        self.canonical_bindings = mappings
        self.resources = {}
        self.forwarded_handles = set()
        self.entry = _name("fort_scope_entry_", routine.qualified)
        self.context = _name("fort_child_context_", routine.qualified)
        self.enabled = _name("fort_child_enabled_", routine.qualified)
        self.kind = _name("fort_child_kind_", routine.qualified)
        self.control_guard = _name("fort_borrowed_", routine.qualified)
        self.call_sites = []

        def capture(_proxy, binding):
            fact = self.owner.builder.capture(self.canonical(binding))
            if binding.name in routine.arguments and binding.intent == "out":
                # First registration can follow the original OUT event. No
                # handle existed at ENTRY in that case, so old caller values
                # cannot initialize this callee's definition coverage.
                return {**fact, "initialized": "none"}
            return fact

        proxy.capture = MethodType(capture, proxy)
        # OUT events also matter for formals unused by admitted numerical work.
        for name in routine.arguments:
            binding = routine.scope.bindings[name]
            if binding.rank and binding.root not in proxy.immutable_inputs:
                self.add_resource(binding)
        self.owner.builder.region_owners[routine.qualified] = proxy.inline
        self.nodes = proxy.inline.prepare(_children(routine.execution))
        proxy.generated.update(proxy.inline.generated)
        proxy.numerical_ir.update(proxy.inline.ir)
        proxy.numerical_reasons.update({name: None for name in proxy.inline.regions})
        self.scan(self.nodes)
        if len({number for _, number in self.resources.values()}) != len(self.resources):
            raise CompilationError("original module companion has uncertain hidden-resource aliases")

    def emit_companion(self):
        self.refresh()
        builder, routine = self.builder, self.routine
        header = _part(routine.scope.node, "Subroutine_Stmt")
        first = statement_span(self.nodes[0])[0]
        start = _span(header)[1]+1
        builder.add_edit(routine.scope.path, start, start-1,
                         f"use iso_c_binding, only: {self.kind} => c_int64_t\n")
        handles = [f"fort_buffer_{number}" for number in self.handle_numbers()]
        labels = {int(node.item.label) for node in walk(routine.execution)
                  if getattr(node, "item", None) is not None and getattr(node.item, "label", None)}
        label = next((number for number in range(99999, 99900, -1) if number not in labels), None)
        if label is None:
            raise CompilationError("original module companion has no bounded free statement label")
        prologue = [f"logical :: {self.control_guard}",
                    f"integer({self.kind}), intent(inout) :: " + ", ".join([self.context, *handles]),
                    f"logical, intent(inout) :: {self.enabled}",
                    f"{self.control_guard} = .false.", f"goto {label}",
                    "entry " + self.entry + "(" + ", ".join([*routine.arguments, self.context, self.enabled, *handles]) + ")",
                    f"{self.control_guard} = .true."]
        outputs = [binding for binding in routine.scope.bindings.values()
                   if binding.name in routine.arguments and binding.rank and binding.intent == "out"]
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
            requirements=("original unsaved module body", "shared owning context control", "one canonical actual mapping"),
            shared_artifacts=("sources/" + _name("source_", str(routine.scope.path)) + ".f90",))

    def public(self):
        return {"procedure": self.routine.qualified, "entry": self.entry,
                "interface": "original_module_entry_v1", "control_version": 1,
                "structured_summary_identity": self.graph.identity,
                "native_abi": "original entry; compiler-control arguments are never referenced",
                "storage": "one original body and declarations; no persistent-state copy",
                "resource_mappings": {root: binding.root for root, binding in self.canonical_bindings.items()},
                "immutable_array_inputs": [item.public() for item in self.builder.immutable_inputs.values()],
                "calls": self.call_sites, "boundaries": self.boundaries,
                "optional_scalar_formals": [name for name in self.routine.arguments
                    if optional_scalar_input(self.routine.scope.bindings[name])],
                "optional_scalar_association": "original OPTIONAL reference and PRESENT guards; no payload capture",
                "argument_rendering": "original source arguments; compiler controls appended by keyword",
                "planning_segments": [{"segment_id": index, "first_line": unit.scope.first,
                    "last_line": unit.scope.last, "operations": unit.scope.public()}
                    for index, unit in enumerate(self.units)],
                "limits": "ordinary numeric formals and read-only optional scalars; one canonical mapping; original saved locals remain native"}


def borrow_module(parent, node):
    """Try one original module body after its complete borrowed worker failed."""
    if hasattr(node, "fort_inline_region"):
        return False
    resolved = parent.builder.execution_source_call(parent.routine, node)
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
    if existing is not None and (not isinstance(existing, ModuleOwner)
            or {root: binding.root for root, binding in existing.canonical_bindings.items()}
            != {root: binding.root for root, binding in mappings.items()}
            or {root: item.actual.root for root, item in existing.builder.immutable_inputs.items()}
            != {root: item.actual.root for root, item in immutable_inputs.items()}):
        raise CompilationError("original module companion requires one canonical mapping per owning invocation")
    if existing is None:
        if len(owner.members) >= parent.builder.analysis.procedure_limit:
            raise CompilationError("original module companion procedure budget exhausted")
        checkpoint = owner.builder.scope_checkpoint()
        members, resources = dict(owner.members), dict(owner.resources)
        owner.active.append(routine.qualified)
        try:
            existing = ModuleOwner(parent, routine, mappings, immutable_inputs)
            owner.members[routine.qualified] = existing
        except CompilationError:
            owner.builder.restore_scope_checkpoint(checkpoint)
            owner.members.clear()
            owner.members.update(members)
            owner.resources.clear()
            owner.resources.update(resources)
            parent.refresh()
            raise
        finally:
            owner.active.pop()
    # Reached child admission can roll back rejected local candidates. Rebind
    # the caller's artifact maps before adding its companion import.
    parent.refresh()
    first, last = statement_span(node)
    if routine.scope.module != parent.routine.scope.module:
        header = _part(parent.routine.scope.node, "Subroutine_Stmt")
        line = _span(header)[1]+1
        key = (parent.routine.qualified, routine.qualified)
        imports = getattr(owner, "companion_imports", set())
        owner.companion_imports = imports
        if key not in imports:
            parent.builder.add_edit(parent.routine.scope.path, line, line-1,
                                    f"use {routine.scope.module}, only: {existing.entry}\n")
            imports.add(key)
    parent.forwarded_handles.update(existing.handle_numbers())
    handles = ["fort_buffer_"+str(number) for number in existing.handle_numbers()]
    arguments = resolved.render_original_arguments()
    controls = [f"{existing.context} = {parent.context}", f"{existing.enabled} = {parent.enabled}",
                *[f"{handle} = {handle}" for handle in handles]]
    # A CONTIGUOUS callee dummy can conceal a caller-created temporary. Check
    # actual storage before that association exists, otherwise a retained
    # handle might outlive temporary copyback at the original call return.
    arrays = [mapping for mapping in resolved.mappings if mapping.formal_binding.rank
              and mapping.formal_binding.root not in immutable_inputs]
    preflight = []
    if arrays:
        preflight = ["block", "logical :: fort_actuals_contiguous", "fort_actuals_contiguous = .true."]
        for mapping in arrays:
            name = str(mapping.actual)
            if "allocatable" in mapping.binding.attributes:
                preflight += [f"if (fort_actuals_contiguous) fort_actuals_contiguous = allocated({name})"]
            preflight += [f"if (fort_actuals_contiguous) fort_actuals_contiguous = is_contiguous({name})"]
        preflight += ["if (.not. fort_actuals_contiguous) then", *parent.close(), "endif", "end block"]
    lines = [f"if ({parent.enabled}) then", *preflight, "endif",
             f"if ({parent.enabled}) then", *_call(existing.entry, [*arguments, *controls]),
             "else", *parent.original((node,)).splitlines(), "endif"]
    if parent.control_guard:
        lines = [f"if ({parent.control_guard}) then", *lines, "else", *parent.original((node,)).splitlines(), "endif"]
    from compiler.scopes.provenance import borrowed
    lines = borrowed(parent, (node,), lines)
    parent.patches.append((first, last, "\n".join(fortran_lines(lines))+"\n"))
    existing.call_sites.append({"caller": parent.routine.qualified, "first_line": first, "last_line": last})
    parent.builder.resolved_calls[(parent.routine.qualified, (first, last))] = resolved.public()
    return True
