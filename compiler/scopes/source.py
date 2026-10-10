"""Compiler-owned, bounded source scopes through public build/source artifacts.

The capture document supplies storage lifetime/initialization facts. It does not
choose GPU calls, scopes, transfers, or numerical lowering. Those decisions live
here. Initial source scopes contain straight-line helper paths and serial callers;
control-flow around sibling spans remains in the original source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256

from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.emission.fortran.formatting import _fortran_line, _fortran_list
from compiler.frontend import lower_file
from compiler.frontend.call_bindings import resolve_source_call
from compiler.frontend.component_bindings import references
from compiler.frontend.source_effects import SourceEffects, _children, _kind, _part
from compiler.ir import CompilationError, SourceLocation
from compiler.ir.integers import integer_literal
from compiler.scopes.access import build_native_accesses, build_native_view_accesses
from compiler.scopes.numerical import load_numerical_sources, resource_binding
from compiler.scopes.variants import VariantRegistry
from compiler.scopes.views import build_view, forget_view, view_call
from compiler.scopes.views import check_call as check_view_call
from compiler.scopes.views import dependencies as view_dependencies

DTYPES = {
    ("real", 8): ("real(c_double)", "FORT_SCOPE_REAL64", 8),
    ("real", 4): ("real(c_float)", "FORT_SCOPE_REAL32", 4),
    ("integer", 4): ("integer(c_int)", "FORT_SCOPE_INTEGER32", 4),
    ("logical", 1): ("logical(c_bool)", "FORT_SCOPE_LOGICAL", 1),
}


def _name(prefix, identity):
    return prefix + sha256(identity.encode()).hexdigest()[:12]


def _call(name, arguments):
    return _fortran_list("call " + name + "(", list(arguments), ")", 0)


def _checked(expression):
    return ["fort_status = " + expression,
            "if (fort_status /= FORT_SCOPE_OK) error stop 'shared scope operation failed'"]


def _span(node):
    item = getattr(node, "item", None)
    if item is None:
        raise CompilationError("source scope requires exact statement spans")
    span = getattr(item, "fort_original_span", item.span)
    if span is None:
        raise CompilationError("source scope statement lacks an original edit span")
    return span


@dataclass
class Call:
    node: object
    procedure: str
    actuals: tuple
    bindings: dict
    summary: dict
    resolved: object = None
    borrowed: bool = False
    region: object = None

    def original_arguments(self, values=None):
        return self.resolved.render_original_arguments(values) if self.resolved else tuple(
            map(str, self.actuals if values is None else values))


class ScopeBuilder:
    def __init__(self, paths, entry, *, facts, options, config, contracts=None, numerical_sources=None,
                 analysis_sources=None, summary_cache=None):
        if not isinstance(facts, dict) or facts.get("schema_version") != 1:
            raise CompilationError("scope capture facts require schema_version 1")
        if facts.get("participation") != "serial":
            raise CompilationError("source scopes currently require proved serial caller participation")
        if not isinstance(facts.get("captures"), dict):
            raise CompilationError("scope captures must be keyed by canonical source resource")
        if config.policy not in {"sections", "auto"} or config.collective:
            raise CompilationError("source scopes require sections/auto and a serial coordinator")
        self.analysis = SourceEffects(paths, contracts=contracts, analysis_sources=analysis_sources,
                                      summary_cache=summary_cache)
        if facts.get("sources") != self.analysis.sources:
            raise CompilationError("scope capture facts do not match the supplied source hashes")
        names = [name for name in self.analysis.routines
                 if name == entry.lower() or ("::" not in entry and name.split("::")[-1] == entry.lower())]
        if len(names) != 1:
            raise CompilationError("source scope entry is unavailable or ambiguous")
        self.entry = self.analysis.routines[names[0]]
        self.facts, self.options, self.config = facts, options, config
        # Lifetime assertions authorize only exact defining-module roots. They
        # cannot bless a formal/local allocation or change a default effect
        # analysis. Source hashes and capture definitions have already passed
        # their public checks before any summary or package consumes this proof.
        stable_module_allocatables = set()
        for module in self.analysis.modules.values():
            for binding in module.bindings.values():
                if (binding.rank and "allocatable" in binding.attributes
                        and binding.root in facts["captures"]):
                    try:
                        self.capture(binding)
                    except CompilationError:
                        # An unrelated unsupported assertion cannot enlarge
                        # authority or prevent otherwise valid native spans.
                        continue
                    stable_module_allocatables.add(binding.root)
        self.analysis.authorize_stable_module_allocatables(stable_module_allocatables)
        self.packages = load_numerical_sources(numerical_sources, self.analysis)
        self.device_budget = facts.get("device_budget_bytes", 256*1024*1024)
        if type(self.device_budget) is not int or not 0 <= self.device_budget < 2**63:
            raise CompilationError("scope device budget must be a nonnegative signed-64-bit byte count")
        self.outputs, self.edits, self.clones, self.generated = {}, {}, {}, {}
        self.queries = {}
        self.numerical_ir = {}
        self.view_generated = {}
        self.view_clones = {}
        self.view_queries = {}
        self.batch_chains = {}
        self.numerical_reasons = {}
        self.boundaries, self.scopes = [], []
        self.resolved_calls = {}
        self.variants = VariantRegistry()
        self.runtime_outputs, self.runtime = read_scoped_runtime()
        self.visiting = set()
        from compiler.scopes.region_dispatch import InlineRegions
        self.inline = InlineRegions(self)
        # Artifact discovery is shared, but each registry authenticates nodes
        # against its own original procedure. Child projections must never be
        # admitted through the caller's lexical scope or source spans.
        self.region_owners = {self.entry.qualified: self.inline}
        self.coordinators, self.coordinator_reasons = {}, {}
        self.coordinator_visiting = []

    def inline_for(self, procedure):
        owner = self.region_owners.get(procedure.partition("#region")[0])
        return owner if owner is not None and procedure in owner.regions else None

    def stable_descriptors(self, procedure):
        return {item["resource"] for item in self.analysis.descriptor_stability(procedure)["resources"]
                if item["stable"] and item["original_descriptor_required"]}

    def check_whole_view_formals(self, callee):
        """Whole-root hooks require a complete, unreduced assumed-shape view."""
        for declaration in _children(_part(callee.scope.node, "Specification_Part")):
            if _kind(declaration) != "Type_Declaration_Stmt":
                continue
            _, attributes, entities = declaration.items
            dimension = next((a.items[1] for a in _children(attributes)
                              if _kind(a) == "Dimension_Attr_Spec"), None)
            for entity in _children(entities):
                name = str(entity.items[0]).lower()
                if name not in callee.arguments:
                    continue
                shape = entity.items[1] if entity.items[1] is not None else dimension
                binding = callee.scope.bindings[name]
                permitted = {"Assumed_Shape_Spec"}
                if ("allocatable" in binding.attributes
                        and binding.root in self.stable_descriptors(callee.qualified)):
                    permitted.add("Deferred_Shape_Spec")
                if any(_kind(axis) not in permitted for axis in _children(shape)):
                    raise CompilationError("explicit dummy extents require a proven whole-storage shape mapping")

    def resolve(self, routine, node):
        if hasattr(node, "fort_inline_region"):
            if self.inline.nodes.get(node.fort_inline_region) is not node:
                raise CompilationError("inline numerical call lacks compiler-owned source authority")
            return self.inline.call(node)
        resolved = self.execution_source_call(routine, node)
        procedure, actuals, bindings = resolved.procedure, resolved.actuals, resolved.bindings
        span = getattr(node.item, "fort_original_span", node.item.span)
        self.resolved_calls[(routine.qualified, span)] = {
            **resolved.public(), "caller": routine.qualified, "source": str(routine.scope.path),
            "first_line": span[0], "last_line": span[1]}
        callee = self.analysis.routines[procedure]
        if callee.source_kind != "module":
            raise CompilationError("external source entry execution requires standalone procedure variants: " + procedure)
        if any(mapping.presence not in {"supplied", "omitted", "allocation_dependent"}
               for mapping in resolved.mappings):
            raise CompilationError("optional source-call arguments require presence-preserving execution variants")
        if any("allocatable" in mapping.formal_binding.attributes
               and mapping.formal_binding.root not in self.stable_descriptors(procedure)
               for mapping in resolved.mappings):
            raise CompilationError("allocatable callee formals require a source-proven stable original descriptor")
        self.check_whole_view_formals(callee)
        coordinator = self.coordinator(procedure) if self.config.scope_execution == "reached" else None
        summary = coordinator.summary if coordinator is not None else self.analysis.summarize(procedure)
        if coordinator is None and not summary["complete"]:
            raise CompilationError("native effects incomplete: " + "; ".join(summary["reasons"]))
        if summary.get("definition_diagnostics"):
            diagnostic = summary["definition_diagnostics"][0]
            raise CompilationError(diagnostic["reason"]+": "+diagnostic["procedure"]+" "+diagnostic["resource"])
        borrowed = (any(mapping.section is not None for mapping in resolved.mappings)
                    or self.view_wrapper_supported(procedure))
        effects, definitions, _ = self.native_effects(
            procedure, {"argument::" + item.formal: None for item in resolved.mappings if item.actual is None},
            allow_views=borrowed)
        aliases = {}
        for formal, binding in bindings.items():
            if binding.rank:
                aliases.setdefault(binding.root, []).append(formal)
        for root in effects:
            if not root.startswith("argument::"):
                aliases.setdefault(root, []).append(root)
        uses_views = borrowed
        if not uses_views and any(len(formals) > 1 and any("write" in effects.get(formal, set()) or
                                        formal in definitions for formal in formals)
               for formals in aliases.values()):
            raise CompilationError("writable source call arguments alias")
        if not uses_views and any(len(formals) > 1 for formals in aliases.values()) and self.numerical(procedure):
            raise CompilationError("numerical source aliases require merged entry access descriptors")
        call = Call(node, procedure, actuals, bindings, summary, resolved, borrowed)
        check_view_call(self, call)
        return call

    def execution_source_call(self, routine, node):
        """Keep source actual arithmetic at its original native evaluation point.

        Type-only effect composition can describe an expression without proving
        that a planning preview may evaluate it.  Until an execution variant has
        checked, reached snapshots, only original scalar storage and literals
        may cross a generated procedure boundary.
        """
        resolved = resolve_source_call(self.analysis, self.analysis.source_scope_for(node, routine.scope), node)
        for mapping in resolved.mappings:
            if mapping.actual is None or mapping.formal_binding.rank or mapping.binding is not None:
                continue
            actual = mapping.actual
            # A signed literal is still a literal: it has no mutable reads or
            # potentially trapping intermediate arithmetic to hoist.
            if _kind(actual) in {"Level_2_Unary_Expr", "Signed_Int_Literal_Constant",
                                  "Signed_Real_Literal_Constant"}:
                items = _children(actual)
                if len(items) == 2 and str(items[0]) in {"+", "-"}:
                    actual = items[1]
            if _kind(actual) not in {"Int_Literal_Constant", "Real_Literal_Constant",
                                     "Logical_Literal_Constant"}:
                raise CompilationError("scalar-expression actual requires original-position checked evaluation: "
                                       + str(mapping.actual))
        return resolved

    def view_wrapper_supported(self, procedure, active=(), *, require_borrowed=True):
        """Recognize bounded call-only numerical paths needing composed views."""
        if procedure in active or len(active) >= 8:
            return False
        routine = self.analysis.routines[procedure]
        summary = self.analysis.summarize(procedure)
        nodes = [node for node in _children(routine.execution) if _kind(node) != "Comment"]
        if (routine.source_kind != "module" or not summary["complete"] or not summary["cloneable"]
                or not nodes or any(_kind(node) != "Call_Stmt" for node in nodes)
                or any(routine.scope.bindings[name].attributes & {"optional", "allocatable", "contiguous"}
                       for name in routine.arguments)):
            return False
        borrowed = False
        try:
            for node in nodes:
                child = self.execution_source_call(routine, node)
                if child.procedure in self.packages:
                    return False
                if (not self.numerical(child.procedure) and not self.view_native_supported(child.procedure)
                        and not self.view_wrapper_supported(child.procedure, active + (procedure,), require_borrowed=False)):
                    return False
                borrowed |= any(mapping.section is not None for mapping in child.mappings)
                borrowed |= self.view_wrapper_supported(child.procedure, active + (procedure,))
            return borrowed if require_borrowed else True
        except CompilationError:
            return False

    def view_native_supported(self, procedure):
        routine = self.analysis.routines[procedure]
        summary = self.analysis.summarize(procedure)
        if (not summary["complete"] or not summary["native_completion"]["available"]
                or any(_kind(node) == "Call_Stmt" for node in walk(routine.execution))
                or any(routine.scope.bindings[name].attributes & {"optional", "allocatable", "contiguous"}
                       for name in routine.arguments)):
            return False
        sections = self.analysis.native_sections(procedure)
        return sections.available and all(resource.resource.startswith("argument::") for resource in sections.resources)

    def numerical(self, procedure):
        regional = self.inline_for(procedure)
        if regional is not None:
            # A rejected outer candidate rolls back emitted artifacts, not
            # immutable source proofs discovered in a reusable child.
            self.generated[procedure] = regional.generated[procedure]
            self.numerical_ir[procedure] = regional.ir[procedure]
            self.numerical_reasons[procedure] = None
            return self.generated[procedure]
        if procedure in self.generated:
            return self.generated[procedure]
        routine = self.analysis.routines[procedure]
        leaf = not any(_kind(node) == "Call_Stmt" for node in walk(routine.execution))
        if self.config.scope_execution == "reached" and not leaf:
            self.generated[procedure] = None
            self.numerical_reasons[procedure] = "source calls require their reached workers"
            return None
        summary = self.analysis.summarize(procedure)
        result = None
        # Keep procedure-entry definition events at their original positions.
        # Numerical helper inlining currently discards those source events;
        # wrappers therefore get explicit context/handle clones instead.
        package = self.packages.get(procedure)
        reason = ("source effect closure is incomplete" if not summary["complete"] else
                  "numerical leaves require explicit source call-position workers" if not leaf else
                  "persistent state or OpenMP participation requires original native execution")
        if summary.get("definition_diagnostics"):
            reason = summary["definition_diagnostics"][0]["reason"]
        elif any(routine.scope.bindings[name].attributes & {"optional", "allocatable"} for name in routine.arguments):
            reason = "optional or allocatable numerical formals require descriptor-aware numerical variants"
        elif (summary["cloneable"] or package and not summary["persistent_state"]) and summary["complete"] and leaf:
            try:
                self.check_numerical_capture_origins(procedure)
                if any(b.dtype == "logical" and b.kind != 1 for b in routine.scope.bindings.values()
                       if b.name in routine.arguments):
                    raise CompilationError("source LOGICAL conversion requires guarded ABI handling")
                if package:
                    effects, _, _ = self.native_effects(procedure)
                    if not set(effects).issubset(package.arrays):
                        raise CompilationError("numerical source package omits original array effects")
                    function = lower_file(package.path, package.entry)
                else:
                    effects, _, _ = self.native_effects(procedure)
                    if any(not root.startswith("argument::") for root in effects):
                        raise CompilationError("hidden numerical arrays require a normalized source package")
                    function = lower_file(self.analysis.inputs.path(routine.scope.path), procedure)
                function, plan = prepare_function(function, options=self.options)
                if plan.regions:
                    from compiler.offload.source_compute import native_participation
                    participation = native_participation(self.analysis, procedure)
                    sources = generate_sources(function, plan,
                        offload_config=replace(self.config, native_participation=participation), memory_model="scoped")
                    if sources.scoped:
                        result = sources
                        self.numerical_ir[procedure] = (function, plan)
                    else:
                        reason = "numerical source has no supported shared entry"
                else:
                    reason = "numerical source has no parallel region"
            except CompilationError as error:
                reason = str(error) # Supported native effects do not imply GPU eligibility.
        self.generated[procedure] = result
        self.numerical_reasons[procedure] = None if result else reason
        return result

    def check_numerical_capture_origins(self, procedure):
        """Require complete original-descriptor mappings for dynamic origins."""
        summary = self.analysis.summarize(procedure)
        resources = {operation.get("resource") for operation in summary["operations"]}
        package = self.packages.get(procedure)
        if package:
            resources.update(parameter.resource for parameter in package.parameters)
        dynamic = resources & self.analysis.stable_module_allocatables
        if dynamic and (not package or not dynamic.issubset(package.runtime_origins)):
            raise CompilationError("module allocatable numerical origins require original runtime lower bounds; native execution")
        if dynamic:
            routine = self.analysis.routines[procedure]
            for intrinsic in ("int", "lbound"):
                if (self.analysis._binding(routine.scope, intrinsic)
                        or self.analysis._candidates(routine.scope, F.Name(intrinsic))):
                    raise CompilationError("numerical allocation origin conversion conflicts with an original "
                                           + intrinsic.upper() + " binding")

    @staticmethod
    def lower_bound_actual(parameter, visible):
        if parameter.runtime_lower_bound:
            return (f"int(lbound({visible}, {parameter.lower_bound_dimension}, kind=c_int64_t), "
                    "kind=c_int)")
        return f"lbound({visible}, {parameter.lower_bound_dimension})"

    def runtime_origin_roots(self, leaves):
        return sorted({root for procedure in leaves if procedure in self.packages
                       for root in self.packages[procedure].runtime_origins}
                      | {binding.root for procedure in leaves if procedure in self.inline.regions
                         for binding in self.inline.regions[procedure].bindings if binding.rank})

    @staticmethod
    def original_bound_conditions(visible, rank):
        """Inspect original descriptors only after allocation/presence checks."""
        minimum, maximum = "(-2147483647_8 - 1_8)", "2147483647_8"
        return [condition for axis in range(1, rank + 1)
                for condition in (f"lbound({visible}, {axis}, kind=8) >= {minimum}",
                                  f"lbound({visible}, {axis}, kind=8) <= {maximum}",
                                  f"ubound({visible}, {axis}, kind=8) >= {minimum}",
                                  f"ubound({visible}, {axis}, kind=8) <= {maximum}",
                                  f"size({visible}, {axis}, kind=8) <= {maximum}")]

    @staticmethod
    def bounds_preflight_public(resources, origin_source):
        return {"bounds_guard": {"resources": list(resources), "inquiry_kind": 8,
                                 "integer_abi_bits": 32, "inquiries": ["lbound", "ubound", "size"],
                                 "position": "original caller after allocation checks; before owner association",
                                 "fallback": "unchanged original source span"},
                "origin_source": origin_source}

    def closure(self, procedure, active=()):
        """Return GPU leaves and cloneable call-only wrappers, or a boundary."""
        if self.inline_for(procedure) is not None:
            return {procedure}, {procedure}
        if procedure in active or len(active) >= 8:
            raise CompilationError("recursive or over-depth source scope")
        routine = self.analysis.routines[procedure]
        if self.numerical(procedure):
            return {procedure}, {procedure}
        coordinator = self.coordinator(procedure)
        if coordinator is not None:
            return set(coordinator.leaves), {procedure}
        summary = self.analysis.summarize(procedure)
        if not summary["complete"]:
            raise CompilationError("incomplete source effect closure")
        children = [n for n in _children(routine.execution) if _kind(n) != "Comment"]
        if not children or any(_kind(n) != "Call_Stmt" for n in children):
            return set(), set()
        if not summary["cloneable"]:
            return set(), set()
        if any("optional" in routine.scope.bindings[name].attributes for name in routine.arguments):
            # Original native forwarding retains absence without associating
            # the optional argument with a generated ordinary array dummy.
            return set(), set()
        leaves, wrappers = set(), {procedure}
        for node in children:
            call = self.resolve(routine, node)
            own_leaves, own_wrappers = self.closure(call.procedure, active + (procedure,))
            leaves.update(own_leaves)
            wrappers.update(own_wrappers)
        return leaves, wrappers if leaves else set()

    def coordinator(self, procedure):
        """Build a reusable reached worker; availability never blesses opacity."""
        if self.config.collective:
            # Borrowing an original joined team requires its coordination
            # protocol, not the serial coordinator's call convention.
            return None
        if procedure in self.coordinators:
            return self.coordinators[procedure]
        if self.inline_for(procedure) is not None or procedure in self.coordinator_visiting:
            return None
        if len(self.coordinator_visiting) >= 8 or len(self.coordinators) >= self.analysis.procedure_limit:
            return None
        routine = self.analysis.routines[procedure]
        if self.numerical(procedure):
            self.coordinators[procedure] = None
            return None
        self.coordinator_visiting.append(procedure)
        try:
            children = [node for node in _children(routine.execution) if _kind(node) != "Comment"]
            if (self.config.scope_execution != "reached" and children
                    and all(_kind(node) == "Call_Stmt" for node in children)):
                # Existing straight wrappers retain their shared query unless a
                # descendant explicitly owns reached source decisions.
                if not any(self.coordinator(self.execution_source_call(routine, node).procedure)
                           for node in children):
                    self.coordinators[procedure] = None
                    return None
            from compiler.scopes.procedures import BorrowedCoordinator
            coordinator = BorrowedCoordinator(self, procedure, tuple(self.coordinator_visiting))
            self.coordinators[procedure] = coordinator
            self.coordinator_reasons[procedure] = None
            return coordinator
        except CompilationError as error:
            self.coordinators[procedure] = None
            self.coordinator_reasons[procedure] = str(error)
            return None
        finally:
            self.coordinator_visiting.pop()

    def native_effects(self, procedure, mapping=None, active=(), *, allow_views=False):
        """Flatten bounded effects with formal-to-root identities, never IR."""
        if self.config.scope_execution == "reached":
            # Classification creates the coordinator. Numerical leaf probing
            # also asks for native effects and must not recursively classify
            # the same unfinished procedure here.
            coordinator = self.coordinators.get(procedure)
            if coordinator is not None:
                mapping = {} if mapping is None else mapping
                effects = {}
                for root, actions in coordinator.effects.items():
                    mapped = mapping.get(root, root)
                    if mapped is not None:
                        effects.setdefault(mapped, set()).update(actions)
                definitions = {mapping.get(root, root) for root in coordinator.definitions} - {None}
                return effects, definitions, set()
        if procedure in active:
            raise CompilationError("recursive native effects")
        if self.analysis.routines[procedure].source_kind != "module":
            raise CompilationError("external source entry execution requires standalone procedure variants: " + procedure)
        routine = self.analysis.routines[procedure]
        for node in walk(routine.execution):
            if _kind(node) == "Call_Stmt":
                self.execution_source_call(routine, node)
        summary = self.analysis.summarize(procedure)
        mapping = {} if mapping is None else mapping
        effects, definitions, overwrites = {}, set(), set()

        def mapped(root):
            return mapping.get(root, root)

        definitions.update(mapped(root) for root in summary["definition_changes"] if mapped(root) is not None)
        overwrites.update(mapped(root) for root in summary["guaranteed_whole_overwrites"] if mapped(root) is not None)
        for operation in summary["operations"]:
            kind = operation["kind"]
            if kind in {"read", "write", "overwrite"} and operation["rank"]:
                root = mapped(operation["resource"])
                if root is None:
                    continue
                effects.setdefault(root, set()).add("read" if kind == "read" else "write")
            elif kind == "call":
                # Summary composition can describe sections and descriptor
                # forwarding before executable root views are implemented.
                # Never turn a child's partial OUT/overwrite into a whole-root
                # coherence claim through this legacy flattening path.
                for actual in operation.get("resource_mappings", ()):
                    descriptor = actual["formal_descriptor"]
                    if (actual["storage"] == "rectangle" and not allow_views
                            or descriptor["rank"] and actual["storage"] not in {"whole", "omitted", "rectangle"}):
                        raise CompilationError("nested rectangular source-call mapping requires canonical root views: "
                                               + operation["procedure"])
                    if actual["presence"] not in {"supplied", "omitted", "forwarded_optional", "allocation_dependent"}:
                        raise CompilationError("nested optional source-call arguments require presence-preserving execution variants: "
                                               + operation["procedure"])
                    if ("allocatable" in descriptor["attributes"]
                            and descriptor["resource"] not in self.stable_descriptors(operation["procedure"])):
                        raise CompilationError("nested allocatable callee formals require a source-proven stable original descriptor: "
                                               + operation["procedure"])
                self.check_whole_view_formals(self.analysis.routines[operation["procedure"]])
                child_mapping = {formal: mapped(actual) for formal, actual in operation["resource_mapping"].items()}
                child_mapping.update({actual["formal_resource"]: None for actual in operation.get("resource_mappings", ())
                                      if actual["storage"] == "omitted"})
                child_effects, child_definitions, child_overwrites = self.native_effects(
                    operation["procedure"], child_mapping, active + (procedure,), allow_views=allow_views)
                for root, actions in child_effects.items():
                    effects.setdefault(root, set()).update(actions)
                definitions.update(child_definitions)
                # Only unconditional calls supply entry-wide must overwrites.
                if not operation["guard"]:
                    overwrites.update(child_overwrites)
            elif kind == "native_contract":
                for effect in operation["effects"]:
                    if not effect.get("rank", 1):
                        continue
                    root = mapped(effect["resource"])
                    if root is None:
                        continue
                    effects.setdefault(root, set()).add("read" if effect["kind"] == "read" else "write")
                    if effect["kind"] == "overwrite" and not operation["guard"]:
                        overwrites.add(root)
        return effects, definitions, overwrites

    def roots_for(self, call):
        if call.region is not None:
            return self.inline.effects(call, mapped=True)
        mapping = {formal: binding.root for formal, binding in call.bindings.items()}
        mapping.update({root: None for root in self.omitted_resources(call)})
        return self.native_effects(call.procedure, mapping, allow_views=view_call(call))

    @staticmethod
    def omitted_resources(call):
        if call.region is not None:
            return set()
        return {"argument::" + item.formal for item in call.resolved.mappings if item.actual is None}

    def call_effects(self, call):
        """Absent optional actuals have no payload or definition events."""
        if call.region is not None:
            return self.inline.effects(call)
        return self.native_effects(call.procedure, {root: None for root in self.omitted_resources(call)},
                                   allow_views=view_call(call))

    def visible(self, routine, root):
        if "%" in root:
            object_root, components = root.split("%", 1)
            return self.visible(routine, object_root) + "%" + components
        candidates = set(routine.scope.bindings)
        visited = set()

        def imported(module):
            target = self.analysis.modules.get(module)
            if target is None or module in visited:
                return
            visited.add(module)
            candidates.update(target.bindings)
            candidates.update(target.imports)
            for name in target.wildcards:
                imported(name)

        current = routine.scope
        while current:
            candidates.update(current.bindings)
            candidates.update(current.imports)
            for module in current.wildcards:
                imported(module)
            current = current.parent
        found = [name for name in sorted(candidates)
                 if (binding := self.analysis._binding(routine.scope, name)) and binding.root == root]
        if not found:
            raise CompilationError("hidden resource is unavailable at owning scope: " + root)
        return found[0]

    def visible_binding(self, routine, root):
        name = self.visible(routine, root)
        return self.analysis._binding(routine.scope, F.Data_Ref(name) if "%" in name else name)

    def capture(self, binding):
        if (binding.dtype, binding.kind) not in DTYPES:
            raise CompilationError("unsupported scoped capture type: " + binding.name)
        if {"pointer", "optional", "volatile", "asynchronous", "value"} & binding.attributes:
            raise CompilationError("capture association or participation is uncertain: " + binding.name)
        fact = self.facts["captures"].get(binding.root)
        if (not isinstance(fact, dict) or fact.get("storage") != "stable"
                or fact.get("escapes") is not False or fact.get("allocation_changes") is not False
                or not isinstance(fact.get("initialized"),str)
                or fact.get("initialized") not in {"whole", "none", "sections"}):
            raise CompilationError("missing stable storage/definition facts: " + binding.root)
        if fact["initialized"] == "sections":
            boxes = fact.get("sections")
            if not isinstance(boxes,list) or len(boxes) > 32:
                raise CompilationError("initialized capture sections exceed the bounded interface")
            for box in boxes:
                if not isinstance(box,dict) or set(box) != {"lower","upper"}:
                    raise CompilationError("initialized capture section requires lower/upper physical coordinates")
                lower,upper=box["lower"],box["upper"]
                if (not isinstance(lower,list) or not isinstance(upper,list)
                        or len(lower) != binding.rank or len(upper) != binding.rank
                        or any(type(v) is not int or not 0 <= v < 2**63 for v in [*lower,*upper])
                        or any(lo > hi for lo,hi in zip(lower,upper,strict=True))):
                    raise CompilationError("invalid initialized physical capture section")
        return fact

    def add_edit(self, path, first, last, replacement, *, prepend=False):
        edits = self.edits.setdefault(path, [])
        edit = (first, last, replacement)
        if prepend:
            if last >= first:
                raise CompilationError("only source insertions can precede other edits")
            edits.insert(0, edit)
        else:
            edits.append(edit)

    def entry_artifacts(self, procedure, *, views=False):
        sources = self.numerical(procedure)
        if sources is None:
            raise CompilationError("requested source leaf has no shared numerical entry")
        regional = self.inline_for(procedure)
        directory = "entries/" + _name("entry_", regional.entries[procedure] if regional else procedure)
        if views:
            if self.config.collective or procedure not in self.numerical_ir:
                raise CompilationError("borrowed numerical views require a prepared serial source leaf")
            if procedure not in self.view_generated:
                function, plan = self.numerical_ir[procedure]
                participation = sources.scoped["compute_estimates"]["native_participation"]
                self.view_generated[procedure] = generate_scoped(
                    function, plan, replace(self.config, native_participation=participation),
                    "common_functions.cuh", runtime_id=self.runtime["runtime_id"],
                    root_views=True, root_view_abi=2)
            companion = self.view_generated[procedure]
            public = companion.report
            artifacts = {"shared_entry.cu": companion.cuda, "shared_interface.f90": companion.fortran}
            directory += "_views"
        else:
            public, artifacts = sources.scoped, sources.artifacts
        if regional is None:
            self.variants.register(
                procedure, interface="root_view_v2" if views else "full_root_entry_v1", role="numerical_entry",
                name=public["fortran_module"] + "::" + public["fortran_procedure"],
                summary_identity=self.analysis.summarize(procedure)["summary_identity"],
                requirements=("compiler-approved numerical entry",
                              "borrowed canonical root view descriptors" if views else "canonical full-root handles"),
                shared_artifacts=(directory + "/shared_entry.cu", directory + "/shared_interface.f90"))
        if directory + "/shared_entry.cu" not in self.outputs:
            for name in ("shared_entry.cu", "shared_interface.f90"):
                self.outputs[directory + "/" + name] = artifacts[name]
            self.outputs[directory + "/common_functions.cuh"] = read_common_header()
            for name in ("scoped_entry.hpp", "scoped_runtime.h", "scoped_team_observer.hpp"):
                self.outputs[directory + "/" + name] = self.runtime_outputs[name]
            if views:
                for name in ("view_entry.hpp", "scoped_regions.hpp"):
                    self.outputs[directory + "/" + name] = self.runtime_outputs[name]
        return public, directory

    def view_clone(self, procedure, *, query=False):
        """Reuse one original-state wrapper with mode-bearing borrowed views."""
        cache = self.view_queries if query else self.view_clones
        if procedure in cache:
            return cache[procedure]
        routine = self.analysis.routines[procedure]
        if not self.view_wrapper_supported(procedure, require_borrowed=False):
            raise CompilationError("source wrapper requires complete view-composable numerical calls: " + procedure)
        if any(name.startswith("fort_") for name in routine.scope.bindings):
            raise CompilationError("source names conflict with the view helper namespace")
        arrays = [name for name in routine.arguments if routine.scope.bindings[name].rank]
        roots = ["argument::" + name for name in arrays]
        descriptors = {root: "fort_view_" + str(index) for index, root in enumerate(roots)}
        handles = {root: descriptors[root] + "%buffer" for root in roots}
        name = _name("fort_scope_view_query_" if query else "fort_scope_view_clone_", procedure)
        cache[procedure] = name, roots
        if not query:
            self.variants.register(
                procedure, interface="source_root_view_v2", role="call_worker", name=name,
                summary_identity=self.analysis.summarize(procedure)["summary_identity"],
                requirements=("mode-bearing CPU/GPU execution", "composed borrowed source views", "original storage retained"),
                shared_artifacts=("sources/" + _name("source_", str(routine.scope.path)) + ".f90",))
        arguments = ["fort_context", *([] if query else ["fort_mode"]), *routine.arguments,
                     *descriptors.values(), *(["fort_status"] if query else [])]
        header = _fortran_list("subroutine " + name + "(", arguments, ")", 0)
        imports = ["use iso_c_binding", "use fort_scoped_memory"]
        spec = []
        for declaration in _children(_part(routine.scope.node, "Specification_Part")):
            if _kind(declaration) != "Type_Declaration_Stmt":
                spec.append(str(declaration))
                continue
            dtype, attributes, entities = declaration.items
            for entity in _children(entities):
                formal = str(entity.items[0]).lower()
                attrs = [str(attribute) for attribute in _children(attributes)]
                if formal in routine.arguments:
                    if query:
                        attrs = [attribute for attribute in attrs if not attribute.upper().startswith("INTENT")
                                 and attribute.upper() != "VALUE"] + ["INTENT(IN)"]
                    if formal in arrays and "TARGET" not in attrs:
                        attrs.append("TARGET")
                spec += _fortran_list(str(dtype) + (", " + ", ".join(attrs) if attrs else "") + " ::",
                                      [str(entity)], "", 0)
        spec += ["integer(c_int64_t), intent(in) :: fort_context",
                 *( [] if query else ["integer(c_int), intent(in) :: fort_mode"]),
                 "integer(c_int)" + (", intent(out)" if query else "") + " :: fort_status",
                 *[f"type(fort_scope_view_v2), intent(in) :: {descriptor}" for descriptor in descriptors.values()]]
        if query:
            spec.append(f"type(fort_scope_plan_binding), target :: fort_bindings({max(1, len(roots))})")
        original_lowers = {root: "fort_lower_" + str(index) for index, root in enumerate(roots)}
        for root, lower in original_lowers.items():
            spec.append(f"integer(c_int64_t) :: {lower}({routine.scope.bindings[root.split('::')[-1]].rank})")
        body = ["fort_status = FORT_SCOPE_OK"]
        on_error = ("return",) if query else ("error stop 'composed source view failed after preflight'",)
        for root, lower in original_lowers.items():
            binding = routine.scope.bindings[root.split("::")[-1]]
            body += _fortran_list(lower + " = [", [f"lbound({binding.name},{axis},kind=c_int64_t)"
                                                   for axis in range(1, binding.rank + 1)], "]", 0)
        children = []
        for index, node in enumerate(_children(routine.execution)):
            if _kind(node) == "Comment":
                continue
            call = self.resolve(routine, node)
            codes = {}
            for formal_index, mapping in enumerate(call.resolved.mappings):
                if not mapping.formal_binding.rank:
                    continue
                if mapping.resource not in descriptors:
                    raise CompilationError("hidden source wrapper arrays require an explicit parent view: " + mapping.resource)
                values = {dependency.root: self.visible(routine, dependency.root)
                          for dependency in view_dependencies(call).values()}
                code = build_view(mapping, handles[mapping.resource], original_lowers[mapping.resource], values,
                                  f"fort_child_{index}_{formal_index}", on_error=on_error,
                                  parent_view=descriptors[mapping.resource])
                spec.extend(code.specification)
                body.extend(code.prepare)
                codes[mapping.formal] = code
            children.append((call, codes))
        for formal in arrays:
            binding = routine.scope.bindings[formal]
            if binding.intent == "out":
                body += forget_view(descriptors["argument::" + formal], binding.rank, query=query, on_error=on_error)
        for call, codes in children:
            if self.closure(call.procedure)[0]:
                body += self.view_numerical_call(call, codes, list(map(str, call.actuals)), imports, query=query,
                                                caller_module=routine.scope.module)
            else:
                mapped_handles = {"argument::" + formal: code.name + "%buffer" for formal, code in codes.items()}
                operation = self.native_plan if query else self.native_call
                body += operation(call, *self.call_effects(call), mapped_handles, view_codes=codes)
            if query:
                body += ["if (fort_status /= FORT_SCOPE_OK) return"]
        from compiler.scopes.segments import fortran_lines
        self.append_procedure(routine.scope.parent, name, "\n".join(fortran_lines([
            *header, *dict.fromkeys(imports), *spec, *body, "end subroutine " + name, ""])))
        return cache[procedure]

    def owner_transfer_setup(self, leaves, imports, *, context="fort_context", status="fort_status"):
        """Configure nondefault transfers once, before recording any work."""
        if self.config.scope_transfers == "direct":
            return []
        public, _ = self.entry_artifacts(sorted(leaves)[0])
        configuration = public["transfer_configuration"]
        imports.append(f"use {public['fortran_module']}, only: fort_configure => {configuration['fortran_procedure']}")
        return [f"if ({status} == FORT_SCOPE_OK) &", f"  {status} = fort_configure({context})"]

    def clone(self, procedure):
        if procedure in self.clones:
            return self.clones[procedure]
        coordinator = self.coordinator(procedure)
        if coordinator is not None:
            return coordinator.worker()
        routine = self.analysis.routines[procedure]
        if any(name.startswith("fort_") for name in routine.scope.bindings):
            raise CompilationError("source names conflict with the initial scope helper namespace")
        summary = self.analysis.summarize(procedure)
        numeric = self.numerical(procedure)
        if not summary["cloneable"] and not (numeric and procedure in self.packages):
            raise CompilationError("scope helper cannot be cloned: " + procedure)
        _, wrappers = self.closure(procedure)
        array_names = [a for a in routine.arguments if routine.scope.bindings[a].rank]
        effects, _, _ = self.native_effects(procedure)
        array_roots = ["argument::" + a for a in array_names]
        array_roots += sorted(set(effects) - set(array_roots))
        name = _name("fort_scope_clone_", procedure)
        self.variants.register(
            procedure, interface="root_handles_v1", role="call_worker", name=name,
            summary_identity=summary["summary_identity"],
            requirements=("source-proven complete effects", "mode-bearing CPU/GPU execution"),
            shared_artifacts=("sources/" + _name("source_", str(routine.scope.path)) + ".f90",))
        handles = {root: "fort_handle_" + str(i) for i, root in enumerate(array_roots)}
        self.clones[procedure] = (name, array_roots)
        header = _fortran_list("subroutine " + name + "(",
                               ["fort_context", "fort_mode", *routine.arguments, *handles.values()], ")", 0)
        uses = ["use iso_c_binding", "use fort_scoped_memory"]
        spec = []
        for node in _children(_part(routine.scope.node, "Specification_Part")):
            if _kind(node) != "Type_Declaration_Stmt":
                spec.append(str(node))
                continue
            dtype, attributes, entities = node.items
            attrs = [str(a) for a in _children(attributes)]
            captured = [e for e in _children(entities) if str(e.items[0]).lower() in array_names]
            others = [e for e in _children(entities) if e not in captured]
            if not captured or "TARGET" in attrs:
                spec.append(str(node))
                continue
            # Coherence hooks can modify borrowed storage through its registered
            # address while these dummies are live. TARGET makes that association
            # explicit instead of relying on an optimizer's alias assumptions.
            for declarations, qualifiers in ((captured, [*attrs, "TARGET"]), (others, attrs)):
                if declarations:
                    spec += _fortran_list(str(dtype) + (", " + ", ".join(qualifiers) if qualifiers else "") + " ::",
                                          list(map(str, declarations)), "", 0)
        if numeric and procedure in self.packages:
            # fparser can place comments in an Implicit_Part as well as directly
            # in the specification. The extraction contract covers all removed
            # directives; retain the original declarations and ordinary comments.
            spec = [line for node in spec for line in node.splitlines()
                    if not line.lstrip().lower().startswith("!$omp")]
        body = []
        if numeric:
            public, _ = self.entry_artifacts(procedure)
            uses += [f"use {public['fortran_module']}, only: fort_run => {public['fortran_procedure']}"]
            package = self.packages.get(procedure)
            if package:
                bindings = {p.name:p for p in package.parameters}
                arrays = [handles[bindings[p["name"].lower()].resource] for p in public["array_parameters"]]
                scalars = []
                for parameter in public["scalar_parameters"]:
                    binding = bindings[parameter["name"].lower()]
                    visible = self.visible(routine, binding.resource)
                    scalars.append(self.lower_bound_actual(binding, visible)
                                   if binding.lower_bound_dimension is not None else visible)
                # Normalized access intents do not replace source definitions.
                for array in array_names:
                    if routine.scope.bindings[array].intent == "out":
                        body += _checked(f"fort_scope_forget_definition(fort_context, {handles['argument::'+array]})")
            else:
                arrays = [handles["argument::"+p["name"].lower()] for p in public["array_parameters"]]
                scalars = [p["name"] for p in public["scalar_parameters"]]
            body += _fortran_list("fort_status = fort_run(",
                                   ["fort_context", "fort_mode", *arrays, *scalars], ")", 0)
            body += ["if (fort_status /= FORT_SCOPE_OK) error stop 'shared numerical entry failed'"]
        else:
            for array in array_names:
                if routine.scope.bindings[array].intent == "out":
                    body += _checked(f"fort_scope_forget_definition(fort_context, {handles['argument::'+array]})")
            for node in _children(routine.execution):
                if _kind(node) == "Comment":
                    body.append(str(node))
                    continue
                call = self.resolve(routine, node)
                leaves, child_wrappers = self.closure(call.procedure)
                root_handles = {formal: handles[b.root] for formal, b in call.bindings.items() if b.rank}
                root_handles.update({root:handle for root,handle in handles.items() if not root.startswith("argument::")})
                if leaves:
                    child_name, child_arrays = self.clone(call.procedure)
                    module = self.analysis.routines[call.procedure].scope.module
                    if module != routine.scope.module:
                        uses.append(f"use {module}, only: {child_name}")
                    body += _call(child_name, ["fort_context", "fort_mode",
                                              *map(str, call.actuals),
                                              *[root_handles[root] for root in child_arrays]])
                else:
                    actions, definitions, overwrites = self.call_effects(call)
                    body += self.native_call(call, actions, definitions, overwrites, root_handles)
        # USE statements precede the original specification, including IMPLICIT.
        extra = ["integer(c_int64_t), intent(in) :: fort_context",
                 "integer(c_int), intent(in) :: fort_mode",
                 *_fortran_list("integer(c_int64_t), intent(in) ::", list(handles.values()), "", 0),
                 "integer(c_int) :: fort_status", "type(fort_scope_access) :: fort_access"]
        text = "\n".join([*header, *dict.fromkeys(uses), *spec, *extra, *body,
                          "end subroutine " + name, ""])
        self.append_procedure(routine.scope.parent, name, text)
        return name, array_roots

    def scalar_writes(self, procedure, mapping=None):
        """Map scalar effects through the same bounded source closure."""
        if self.inline_for(procedure) is not None:
            return set()  # Extraction proved every loop-written scalar dead/private.
        mapping = {} if mapping is None else mapping
        if self.config.scope_execution == "reached":
            coordinator = self.coordinator(procedure)
            if coordinator is not None:
                return {mapping.get(root, root) for root in coordinator.scalar_changes} - {None}
        routine = self.analysis.routines[procedure]
        result = {mapping.get(binding.root, binding.root) for binding in routine.scope.bindings.values()
                  if not binding.rank and binding.intent == "out" and binding.name in routine.arguments}
        for operation in self.analysis.summarize(procedure)["operations"]:
            if operation["kind"] in {"write", "overwrite"} and not operation["rank"]:
                result.add(mapping.get(operation["resource"], operation["resource"]))
            elif operation["kind"] == "call":
                child = {formal: mapping.get(actual, actual)
                         for formal, actual in operation["resource_mapping"].items()}
                result.update(self.scalar_writes(operation["procedure"], child))
            elif operation["kind"] == "native_contract":
                result.update(mapping.get(effect["resource"], effect["resource"])
                              for effect in operation["effects"]
                              if not effect.get("rank", 1) and effect["kind"] in {"write", "overwrite"})
        return result

    def check_query_specification(self, routine):
        """Do not execute dynamic specification expressions ahead of source work."""
        for node in _children(_part(routine.scope.node, "Specification_Part")):
            if _kind(node) != "Type_Declaration_Stmt":
                continue
            dtype, attributes, entities = node.items
            if _kind(dtype) != "Intrinsic_Type_Spec" or str(dtype.items[0]).lower() not in {
                "real", "integer", "logical", "double precision"
            }:
                raise CompilationError("planning query requires numeric declarations without dynamic type parameters")
            dimension = next((a.items[1] for a in _children(attributes)
                              if _kind(a) == "Dimension_Attr_Spec"), None)
            for entity in _children(entities):
                shape = entity.items[1] if entity.items[1] is not None else dimension
                for axis in _children(shape):
                    for bound in axis.items:
                        if bound is None:
                            continue
                        try:
                            routine.scope.kinds.integer(bound, SourceLocation(str(routine.scope.path)))
                        except CompilationError as error:
                            raise CompilationError("planning query requires static or assumed-shape declarations; "
                                                   "dynamic specification bound: " + str(bound)) from error

    def planning_inputs(self, procedure, mapping=None, *, require_estimates=True):
        """Read public numerical metadata; never infer costs from generated code."""
        mapping = {} if mapping is None else mapping
        regional = self.inline_for(procedure)
        if regional is not None:
            return regional.inputs(procedure, mapping, estimates=require_estimates)
        if self.coordinator(procedure) is not None:
            raise CompilationError("source child coordinator plans only its reached segments")
        routine = self.analysis.routines[procedure]
        self.check_query_specification(routine)
        numeric = self.numerical(procedure)
        if numeric:
            public = numeric.scoped
            planning = public["planning"]
            if require_estimates:
                if not planning["available"] or not planning["profile_available"]:
                    raise CompilationError(planning["reason"] or planning["profile_reason"])
            elif not planning["query_available"]:
                raise CompilationError(planning["query_reason"])
            package = self.packages.get(procedure)
            parameters = {p.name: p for p in package.parameters} if package else {}

            def resource(name):
                parameter = parameters.get(name.lower())
                root = parameter.resource if parameter else "argument::" + name.lower()
                # Lower-bound parameters depend on a descriptor, not its payload.
                if parameter and parameter.lower_bound_dimension is not None:
                    return None
                if parameter and not parameter.rank:
                    binding = resource_binding(self.analysis, routine, root)
                    if (routine.scope.bindings.get(binding.name) is binding and
                            binding.root == routine.qualified + "::" + binding.name and
                            "parameter" in binding.attributes):
                        if binding.signature() != ("integer", 4, 0) or binding.attributes != {"parameter"}:
                            raise CompilationError("planning local PARAMETER requires a scalar default INTEGER: " + root)
                        location = SourceLocation(str(routine.scope.path))
                        value = routine.scope.kinds.integer(F.Name(binding.name), location)
                        integer_literal(str(value), location)
                        # Its declaration and actual remain inside the original
                        # leaf clones; the owner need not capture this constant.
                        return None
                return mapping.get(root, root)

            payload = {resource(name) for name in planning["payload_arrays"]} - {None}
            scalars = {resource(name) for name in planning["scalar_inputs"]} - {None}
            return payload, scalars
        payload, scalars = set(), set()
        for node in _children(routine.execution):
            if _kind(node) == "Comment":
                continue
            call = self.resolve(routine, node)
            leaves, _ = self.closure(call.procedure)
            if not leaves:
                continue
            child = {formal: mapping.get(binding.root, binding.root) for formal, binding in call.bindings.items()}
            callee = self.analysis.routines[call.procedure]
            for formal in callee.arguments:
                child.setdefault("argument::" + formal, None)  # Literal actuals carry no mutable resource.
            own_payload, own_scalars = self.planning_inputs(call.procedure, child, require_estimates=require_estimates)
            payload.update(own_payload)
            scalars.update(own_scalars)
            scalars.update(mapping.get(root, root) for root in view_dependencies(call))
        return payload, scalars

    def native_accesses(self, call, handles, *, query, view_codes=None, actuals=None):
        """Use one checked original-coordinate mapping for query and hooks."""
        sections = self.analysis.native_sections(call.procedure)
        omitted = self.omitted_resources(call)
        if omitted:
            sections = replace(sections, resources=tuple(item for item in sections.resources if item.resource not in omitted))
        if not sections.available:
            if view_codes is not None:
                raise CompilationError("native rectangular source-call mapping lacks exact projected effects")
            return None

        prefix = _name("fort_native_", call.procedure + ":" + str(_span(call.node)))
        parameters = {}
        values = call.actuals if actuals is None else actuals
        for mapping, value in zip(call.resolved.mappings, values, strict=True):
            if value is not None and not mapping.formal_binding.rank and mapping.formal_binding.dtype == "integer":
                parameters["argument::" + mapping.formal] = str(value)
        try:
            for resource in sections.resources:
                for dependency in resource.dependencies:
                    if dependency.kind == "scalar_read" and not dependency.resource.startswith("argument::"):
                        parameters[dependency.resource] = self.visible(self.entry, dependency.resource)
            if view_codes is not None:
                return build_native_view_accesses(
                    sections, {"argument::" + formal: code.name for formal, code in view_codes.items()}, handles, prefix,
                    parameters=parameters,
                    on_error=("return",) if query else ("error stop 'projected native access preparation failed'",))
            return build_native_accesses(sections, handles, prefix,
                                         parameters=parameters,
                                         on_error=("return",) if query else
                                         ("error stop 'shared native access preparation failed'",))
        except CompilationError:
            if view_codes is not None:
                raise
            # Read-only aliases and unsupported physical mappings keep the
            # existing conservative effects, which the preflight must validate.
            return None

    def native_bound_bindings(self, call):
        """Original scalar reads used by reached physical native footprints."""
        if call.region is not None:
            return {}
        if self.config.scope_execution == "reached" and self.coordinator(call.procedure) is not None:
            return {}  # Child native bounds are read only in its reached worker.
        sections = self.analysis.native_sections(call.procedure)
        if not sections.available:
            return {}
        actuals = {"argument::" + item.formal: item.actual for item in call.resolved.mappings}
        bindings = {}
        for resource in sections.resources:
            for dependency in resource.dependencies:
                if dependency.kind != "scalar_read":
                    continue
                if dependency.resource in actuals:
                    actual = actuals[dependency.resource]
                    if actual is not None:
                        bindings.update((binding.root, binding) for binding, _ in
                                        references(self.analysis, self.entry.scope, actual) if not binding.rank)
                else:
                    bindings[dependency.resource] = dependency.binding
        return bindings

    def native_plan(self, call, actions, definitions, overwrites, handles, *, view_codes=None, actuals=None):
        """Record fixed host effects at their source positions without calling them."""
        outer = set(self.analysis.summarize(call.procedure)["definition_changes"])
        if (definitions - outer) & handles.keys():
            raise CompilationError("nested native definition changes require original-position planning hooks")
        lines = []
        for root in sorted(definitions & handles.keys()):
            if view_codes is not None:
                formal = root.split("::")[-1]
                lines += forget_view(view_codes[formal].name, call.resolved.mappings[call.resolved.formals.index(formal)].formal_binding.rank,
                                     query=True, on_error=("return",))
                continue
            lines += ["fort_bindings(1) = fort_scope_plan_binding()",
                      f"fort_bindings(1)%buffer = {handles[root]}",
                      "fort_status = fort_scope_plan_add(fort_context, FORT_SCOPE_PLAN_FORGET, 0_c_int64_t, &",
                      "    c_loc(fort_bindings), 1_c_size_t, 0.0_c_double, 0.0_c_double, 0_c_int)",
                      "if (fort_status /= FORT_SCOPE_OK) return"]
        refined = self.native_accesses(call, handles, query=True, view_codes=view_codes, actuals=actuals)
        if refined is not None:
            lines += ["block", *[line for access in refined for line in access.specification]]
            for i, access in enumerate(refined, 1):
                lines += [*access.prepare,
                          f"fort_bindings({i}) = fort_scope_plan_binding()",
                          f"fort_bindings({i})%buffer = {access.handle}",
                          f"fort_bindings({i})%access = {access.access_name}"]
            lines += ["fort_status = fort_scope_plan_add(fort_context, FORT_SCOPE_PLAN_NATIVE, 0_c_int64_t, &",
                      f"    {'c_loc(fort_bindings)' if refined else 'c_null_ptr'}, {len(refined)}_c_size_t, &",
                      "    0.0_c_double, 0.0_c_double, 0_c_int)",
                      "if (fort_status /= FORT_SCOPE_OK) return", "end block"]
            return lines
        merged = {}
        for root, kinds in sorted(actions.items()):
            if root not in handles:
                raise CompilationError("native planning needs unavailable buffer mapping: " + root)
            handle = handles[root]
            old, overwrite = merged.get(handle, (set(), False))
            merged[handle] = (old | kinds, overwrite or root in overwrites)
        for i, (handle, (kinds, overwrite)) in enumerate(merged.items(), 1):
            flags = (["FORT_SCOPE_READ_ALL"] if "read" in kinds else [])
            flags += (["FORT_SCOPE_WRITE_ALL"] if "write" in kinds else [])
            flags += (["FORT_SCOPE_OVERWRITE_ALL"] if overwrite else [])
            lines += [f"fort_bindings({i}) = fort_scope_plan_binding()",
                      f"fort_bindings({i})%buffer = {handle}",
                      f"fort_bindings({i})%access%flags = " + " + ".join(flags)]
        lines += ["fort_status = fort_scope_plan_add(fort_context, FORT_SCOPE_PLAN_NATIVE, 0_c_int64_t, &",
                  f"    {'c_loc(fort_bindings)' if merged else 'c_null_ptr'}, {len(merged)}_c_size_t, &",
                  "    0.0_c_double, 0.0_c_double, 0_c_int)",
                  "if (fort_status /= FORT_SCOPE_OK) return"]
        return lines

    def query_clone(self, procedure):
        """Project call-only wrappers into side-effect-free planning queries."""
        if procedure in self.queries:
            return self.queries[procedure]
        routine = self.analysis.routines[procedure]
        numeric = self.numerical(procedure)
        array_names = [a for a in routine.arguments if routine.scope.bindings[a].rank]
        effects, _, _ = self.native_effects(procedure)
        roots = ["argument::" + a for a in array_names]
        roots += sorted(set(effects) - set(roots))
        handles = {root: "fort_handle_" + str(i) for i, root in enumerate(roots)}
        name = _name("fort_scope_query_", procedure)
        self.queries[procedure] = (name, roots)
        header = _fortran_list("subroutine " + name + "(",
                               ["fort_context", *routine.arguments, *handles.values(), "fort_status"], ")", 0)
        uses = ["use iso_c_binding", "use fort_scoped_memory"]
        spec = []
        for node in _children(_part(routine.scope.node, "Specification_Part")):
            if _kind(node) != "Type_Declaration_Stmt":
                spec.append(str(node))
                continue
            dtype, attributes, entities = node.items
            attrs = [str(a) for a in _children(attributes)]
            arguments = [e for e in _children(entities) if str(e.items[0]).lower() in routine.arguments]
            others = [e for e in _children(entities) if e not in arguments]
            for declarations, qualifiers in ((arguments, [*[a for a in attrs if not a.upper().startswith("INTENT") and a.upper() != "VALUE"],
                                                        "INTENT(IN)"]), (others, attrs)):
                if declarations:
                    spec += _fortran_list(str(dtype) + (", " + ", ".join(qualifiers) if qualifiers else "") + " ::",
                                          list(map(str, declarations)), "", 0)
        if numeric and procedure in self.packages:
            spec = [line for node in spec for line in node.splitlines()
                    if not line.lstrip().lower().startswith("!$omp")]
        body = ["fort_status = FORT_SCOPE_OK"]
        # Normalized entries do not own original INTENT(OUT) source events.
        if not numeric or procedure in self.packages:
            for array in array_names:
                if routine.scope.bindings[array].intent == "out":
                    root = "argument::" + array
                    body += ["fort_bindings(1) = fort_scope_plan_binding()",
                             f"fort_bindings(1)%buffer = {handles[root]}",
                             "fort_status = fort_scope_plan_add(fort_context, FORT_SCOPE_PLAN_FORGET, 0_c_int64_t, &",
                             "    c_loc(fort_bindings), 1_c_size_t, 0.0_c_double, 0.0_c_double, 0_c_int)",
                             "if (fort_status /= FORT_SCOPE_OK) return"]
        if numeric:
            public, _ = self.entry_artifacts(procedure)
            uses.append(f"use {public['fortran_module']}, only: fort_plan => {public['planning']['fortran_procedure']}")
            package = self.packages.get(procedure)
            if package:
                parameters = {p.name: p for p in package.parameters}
                arrays = [handles[parameters[p["name"].lower()].resource] for p in public["array_parameters"]]
                scalars = []
                for parameter in public["scalar_parameters"]:
                    if parameter["name"] not in public["planning"]["scalar_inputs"]:
                        continue
                    binding = parameters[parameter["name"].lower()]
                    visible = self.visible(routine, binding.resource)
                    if binding.lower_bound_dimension is not None:
                        scalars.append(self.lower_bound_actual(binding, visible))
                    elif parameter["dtype"] == "logical":
                        scalars.append(f"logical({visible}, kind=c_bool)")
                    else:
                        scalars.append(visible)
            else:
                arrays = [handles["argument::" + p["name"].lower()] for p in public["array_parameters"]]
                scalars = [p["name"] for p in public["scalar_parameters"]
                           if p["name"] in public["planning"]["scalar_inputs"]]
            body += _fortran_list("fort_status = fort_plan(", ["fort_context", *arrays, *scalars], ")", 0)
        else:
            for node in _children(routine.execution):
                if _kind(node) == "Comment":
                    continue
                call = self.resolve(routine, node)
                leaves, _ = self.closure(call.procedure)
                root_handles = {formal: handles[b.root] for formal, b in call.bindings.items() if b.rank}
                root_handles.update({root: handle for root, handle in handles.items() if not root.startswith("argument::")})
                if leaves:
                    child, child_roots = self.query_clone(call.procedure)
                    module = self.analysis.routines[call.procedure].scope.module
                    if module != routine.scope.module:
                        uses.append(f"use {module}, only: {child}")
                    body += _call(child, ["fort_context", *map(str, call.actuals),
                                          *[root_handles[root] for root in child_roots], "fort_status"])
                    body += ["if (fort_status /= FORT_SCOPE_OK) return"]
                else:
                    actions, definitions, overwrites = self.call_effects(call)
                    body += self.native_plan(call, actions, definitions, overwrites, root_handles)
        extra = ["integer(c_int64_t), intent(in) :: fort_context",
                 *_fortran_list("integer(c_int64_t), intent(in) ::", list(handles.values()), "", 0),
                 "integer(c_int), intent(out) :: fort_status",
                 f"type(fort_scope_plan_binding), target :: fort_bindings({max(1, len(roots))})"]
        self.append_procedure(routine.scope.parent, name,
                              "\n".join([*header, *dict.fromkeys(uses), *spec, *extra, *body,
                                         "end subroutine " + name, ""]))
        return name, roots

    def check_native_definitions(self, call, actions, definitions, overwrites):
        # A native call must finish its effects before host_end commits them.
        # Memory summaries alone do not prove participation or completion of
        # worksharing, tasks or target operations, including hidden callees.
        completion = self.analysis.summarize(call.procedure)["native_completion"]
        if not completion["available"]:
            raise CompilationError("native OpenMP participation and completion require source proof: "
                                   + call.procedure + ": " + completion["reason"])
        # These effects surround one original native call. They cannot prepare
        # a read of values defined later inside that call, or preserve undefined
        # holes in a conservative whole-resource write. Numerical workers have
        # their own physical effects and original-position definition events.
        for operation in self.analysis.summarize(call.procedure)["operations"]:
            if operation["kind"] == "call":
                child_definitions = self.native_effects(operation["procedure"])[1]
                mapping = operation["resource_mapping"]
                if any(mapping.get(root, root) in actions for root in child_definitions):
                    raise CompilationError("nested native definition changes require original-position hooks")
        for root in sorted(definitions & actions.keys()):
            kinds = actions[root]
            physical = self.analysis.native_sections(call.procedure)
            exact = next((item for item in physical.resources if item.resource == root), None) if physical.available else None
            partial_overwrite = exact is not None and not exact.reads and exact.writes == exact.overwrites
            if "read" in kinds or ("write" in kinds and root not in overwrites and not partial_overwrite):
                raise CompilationError("native INTENT(OUT) effects require original-position definition hooks: "
                                       + call.procedure + " " + root)

    def check_native_calls(self, call, *, reached_definitions=False):
        if call.region is not None:
            return
        coordinator = self.coordinator(call.procedure)
        if coordinator is not None:
            if self.config.scope_execution == "reached" and reached_definitions:
                # Reached owners validate these obligations against current
                # coverage before entering the child, after earlier producers.
                coordinator.check_aliases(call)
            else:
                coordinator.check_actuals(call)
            return
        leaves, _ = self.closure(call.procedure)
        if not leaves:
            self.check_native_definitions(call, *self.call_effects(call))
        elif not self.numerical(call.procedure):
            routine = self.analysis.routines[call.procedure]
            for node in _children(routine.execution):
                if _kind(node) == "Call_Stmt":
                    self.check_native_calls(self.resolve(routine, node), reached_definitions=reached_definitions)

    def native_call(self, call, actions, definitions, overwrites, handles, *, actuals=None, view_codes=None):
        self.check_native_definitions(call, actions, definitions, overwrites)
        lines = []
        # Only outer dummy definition changes can be performed before the call.
        # A nested INTENT(OUT) event must stay at its original source position.
        outer = set(self.analysis.summarize(call.procedure)["definition_changes"])
        if (definitions - outer) & handles.keys():
            raise CompilationError("nested native definition changes require original-position hooks")
        for root in sorted(definitions):
            if root in handles:
                if view_codes is not None:
                    formal = root.split("::")[-1]
                    lines += forget_view(view_codes[formal].name, call.resolved.mappings[call.resolved.formals.index(formal)].formal_binding.rank,
                                         query=False, on_error=("error stop 'native view definition failed'",))
                    continue
                lines += _checked(f"fort_scope_forget_definition(fort_context, {handles[root]})")
        refined = self.native_accesses(call, handles, query=False, view_codes=view_codes, actuals=actuals)
        if refined is not None:
            lines += ["block", *[line for access in refined for line in access.specification]]
            for access in refined:
                lines += [*access.prepare]
                lines += _checked(f"fort_scope_host_begin(fort_context, {access.handle}, {access.access_name})")
            lines += _call(str(call.node.items[0]), call.original_arguments(actuals))
            for access in refined:
                lines += _checked(f"fort_scope_host_end(fort_context, {access.handle})")
            lines += ["end block"]
            return lines
        # Multiple read-only formals may identify one canonical root. Prepare
        # that handle once; the runtime forbids overlapping prepared accesses.
        accesses = {}
        for root, kinds in sorted(actions.items()):
            if root not in handles:
                raise CompilationError("native operation needs unavailable buffer mapping: " + root)
            handle = handles[root]
            merged, overwrite = accesses.get(handle, (set(), False))
            accesses[handle] = (merged | kinds, overwrite or root in overwrites)
        for handle, (kinds, overwrite) in accesses.items():
            flags = []
            if "read" in kinds:
                flags.append("FORT_SCOPE_READ_ALL")
            if "write" in kinds:
                flags.append("FORT_SCOPE_WRITE_ALL")
            if overwrite:
                flags.append("FORT_SCOPE_OVERWRITE_ALL")
            lines += ["fort_access = fort_scope_access()",
                      "fort_access%flags = " + " + ".join(flags)]
            lines += _checked(f"fort_scope_host_begin(fort_context, {handle}, fort_access)")
        lines += _call(str(call.node.items[0]), call.original_arguments(actuals))
        for handle in accesses:
            lines += _checked(f"fort_scope_host_end(fort_context, {handle})")
        return lines

    def append_procedure(self, module, name, code):
        end = _part(module.node, "End_Module_Stmt")
        contains = next(n for n in _children(_part(module.node, "Module_Subprogram_Part"))
                        if _kind(n) == "Contains_Stmt")
        self.add_edit(module.path, _span(end)[0], _span(end)[0]-1, code)
        self.add_edit(module.path, _span(contains)[0], _span(contains)[0]-1, "public :: " + name + "\n")

    def owner_inputs(self, calls, *, reached_definitions=False):
        """Resolve checked root captures shared by serial and team owners."""
        routine = self.entry
        if any(name.startswith("fort_") for name in routine.scope.bindings):
            raise CompilationError("source names conflict with the initial scope owner namespace")
        arrays, scalars, written = {}, {}, set()
        for call in calls:
            if call.region is not None:
                for binding in call.region.bindings:
                    if binding.rank:
                        self.capture(binding)
                        arrays[binding.root] = binding
                    else:
                        scalars[binding.root] = binding
                written.update(call.region.written_resources)
                continue
            self.check_native_calls(call, reached_definitions=reached_definitions)
            actions, definitions, overwrites = self.roots_for(call)
            written.update(root for root,kinds in actions.items() if "write" in kinds)
            _, formal_definitions, _ = self.call_effects(call)
            for formal,binding in call.bindings.items():
                if binding.rank:
                    self.capture(binding)
                    arrays[binding.root] = binding
                    if formal in formal_definitions:
                        written.add(binding.root)
            for root in actions:
                visible = self.visible(routine, root)
                binding = self.visible_binding(routine, root)
                self.capture(binding)
                arrays[root] = binding
            child_leaves, _ = self.closure(call.procedure)
            for root in self.runtime_origin_roots(child_leaves):
                binding = self.visible_binding(routine, root)
                self.capture(binding)
                arrays[root] = binding
            for actual in call.actuals:
                binding = self.analysis._actual_binding(self.analysis.source_scope_for(actual, routine.scope), actual)
                if binding and not binding.rank:
                    if "allocatable" in binding.attributes:
                        raise CompilationError("allocatable scalar captures require original descriptor semantics")
                    if (binding.dtype not in {"real", "integer", "logical"}
                            or binding.kind not in ({1, 4} if binding.dtype == "logical" else {4, 8})):
                        raise CompilationError("unsupported scalar source capture")
                    scalars[binding.root] = binding
            scalars.update(view_dependencies(call))
            scalars.update(self.native_bound_bindings(call))
        return arrays, scalars, written

    def owner_query(self, calls, arrays, written):
        """Prove immutable, safely available query inputs before execution."""
        routine = self.entry
        planning_available = False
        planning_reason = "scoped automatic selection was not requested"
        query_available, query_reason = False, None
        try:
            payload, controls, changed_scalars = set(), set(), set()
            for call in calls:
                mapping = {formal: binding.root for formal, binding in call.bindings.items()}
                if call.region is None:
                    callee = self.analysis.routines[call.procedure]
                    for formal in callee.arguments:
                        mapping.setdefault("argument::" + formal, None)
                child_leaves, _ = self.closure(call.procedure)
                if child_leaves:
                    own_payload, own_controls = self.planning_inputs(call.procedure, mapping, require_estimates=False)
                    payload.update(own_payload)
                    controls.update(own_controls)
                controls.update(view_dependencies(call))
                controls.update(self.native_bound_bindings(call))
                changed_scalars.update(self.scalar_writes(call.procedure, mapping))
            if payload & written:
                raise CompilationError("planning payload arrays change inside the complete source scope")
            if controls & changed_scalars:
                raise CompilationError("planning scalar inputs change inside the complete source scope")
            for root in sorted(controls):
                binding = self.visible_binding(routine, root)
                if binding.attributes & {"volatile", "asynchronous", "optional", "pointer", "allocatable"}:
                    raise CompilationError("planning control scalar association or participation is uncertain: " + root)
            if any(root not in arrays or self.capture(arrays[root])["initialized"] != "whole" for root in sorted(payload)):
                raise CompilationError("planning payload arrays require whole initialized host storage")
            query_available = True
        except CompilationError as error:
            query_reason = str(error)
        if self.config.policy == "auto":
            planning_reason = query_reason
            if query_available:
                try:
                    for call in calls:
                        child_leaves, _ = self.closure(call.procedure)
                        if child_leaves:
                            self.planning_inputs(call.procedure)
                    planning_available = True
                except CompilationError as error:
                    planning_reason = str(error)
        return query_available, query_reason, planning_available, planning_reason

    def check_module_array_aliases(self, calls, arrays, written):
        """Do not introduce an illegal host/use alias to a new target dummy."""
        visited = set()

        def visit(procedure):
            if procedure in visited or self.inline_for(procedure) is not None or self.numerical(procedure):
                return
            visited.add(procedure)
            coordinator = self.coordinator(procedure)
            if coordinator is not None:
                for operation in coordinator.scope.native:
                    for root in operation.effects:
                        if (root in arrays and root in written and not root.startswith("argument::")
                                and "target" not in arrays[root].attributes):
                            raise CompilationError("native host/use array alias requires original TARGET capture: " + procedure + " " + root)
                for call in coordinator.scope.calls:
                    visit(call.procedure)
                return
            leaves, _ = self.closure(procedure)
            if leaves:
                routine = self.analysis.routines[procedure]
                for node in _children(routine.execution):
                    if _kind(node) == "Call_Stmt":
                        visit(self.resolve(routine, node).procedure)
                return
            effects, _, _ = self.native_effects(procedure)
            for root in effects:
                if (root in arrays and root in written and not root.startswith("argument::")
                        and "target" not in arrays[root].attributes):
                    raise CompilationError("native host/use array alias requires original TARGET capture: "
                                           + procedure + " " + root)
        for call in calls:
            visit(call.procedure)

    def automatic_native_preflight(self, leaves):
        """A static missing-estimate proof needs no owning runtime context.

        One estimated leaf keeps the reached planner available for mixed
        placement. Unknown dimensions and protected scalar inputs are never
        inspected here: this decision consumes compiler-produced metadata only.
        """
        if self.config.policy != "auto" or not leaves:
            return None
        reasons = []
        for procedure in sorted(leaves):
            sources = self.numerical(procedure)
            if sources is None:
                return None
            planning = sources.scoped["planning"]
            if planning["available"] and planning["profile_available"]:
                return None
            reasons.append(planning["reason"] or planning["profile_reason"] or "offline estimate unavailable")
        return "all numerical alternatives lack compatible offline estimates: " + "; ".join(dict.fromkeys(reasons))

    def owner(self, calls, leaves, *, structure=None):
        routine = self.entry
        first, last = ((structure.first, structure.last) if structure else
                       (_span(calls[0].node)[0], _span(calls[-1].node)[1]))
        digest = f"{routine.qualified}:{first}:{last}"
        name = _name("fort_scope_owner_", digest)
        caller_kind = _name("fort_scope_i64_", digest)
        if (self.analysis._binding(routine.scope, caller_kind)
                or self.analysis._candidates(routine.scope, F.Name(caller_kind))):
            raise CompilationError("original source conflicts with descriptor guard INTEGER kind alias")
        owner_variant = self.variants.register(
            routine.qualified, interface=f"source_span_v1:{first}:{last}:{bool(structure)}",
            role="coordinator", name=name,
            summary_identity=self.analysis.summarize(routine.qualified)["summary_identity"],
            requirements=("proved source span and caller participation", "one invocation owns buffer lifetime"),
            shared_artifacts=("sources/" + _name("source_", str(routine.scope.path)) + ".f90",))
        arrays, scalars, written = self.owner_inputs(calls)
        borrowed = any(view_call(call) for call in calls)
        regional = any(call.region is not None for call in calls)
        tile_guards = tuple(dict.fromkeys(guard for call in calls if call.region is not None
                                         for guard in call.region.runtime_guards))
        numerical_environment = any(call.region is not None and call.region.requires_numerical_environment for call in calls)
        runtime_guards = (*tile_guards, *(["fort_scope_numerical_environment_supported() /= 0"] if numerical_environment else []))
        if tile_guards and len(calls) != 1 and not structure:
            raise CompilationError("tile runtime guards initially require one original reached numerical region")
        if borrowed and structure:
            raise CompilationError("reached rectangular source views require per-segment descriptor preflight")
        if structure:
            structure.inputs(arrays, scalars, written)
            structure.check_conservative_definitions(arrays)
            if any(binding.attributes & {"pointer", "optional", "volatile", "asynchronous", "allocatable", "value"}
                   for binding in scalars.values()):
                raise CompilationError("structured scalar association or participation is uncertain")
            structure.prepare(arrays)
            query_available, query_reason = True, None
            planning_available, planning_reason = structure.planning_status()
        else:
            query_available, query_reason, planning_available, planning_reason = self.owner_query(calls, arrays, written)
        self.check_module_array_aliases(calls, arrays, written)
        caller_fallback = any(call.region is not None and call.region.numerical_helpers for call in calls) or any(
            parameter.rank and parameter.resource in written and parameter.resource in arrays
            and not parameter.resource.startswith("argument::")
            and "target" not in arrays[parameter.resource].attributes
            for leaf in leaves if (package := self.packages.get(leaf))
            for parameter in package.parameters)
        original_scalars = {root: binding for root, binding in scalars.items()
                            if len(root.split("::")) == 2 and root.split("::")[0] in self.analysis.modules}
        if any(self.visible(routine, root).startswith("fort_") for root in original_scalars):
            raise CompilationError("original module scalar conflicts with scope owner namespace")
        # Native children may update module state by host/use association.
        # Preserve those bindings directly, avoiding a new dummy association
        # whose alias rules could hide an update from an optimizing compiler.
        scalars = {root: binding for root, binding in scalars.items() if root not in original_scalars}
        names = [self.visible(routine, root) for root in (*arrays, *scalars)]
        if any(n.startswith("fort_") for n in names):
            raise CompilationError("capture names conflict with the initial scope owner namespace")
        allocated_roots = [root for root, binding in arrays.items() if "allocatable" in binding.attributes]
        origin_roots = self.runtime_origin_roots(leaves)
        if structure or borrowed or regional:
            # All these paths retain original logical descriptor origins.
            # Check their index ABI before association or any earlier segment
            # can modify numerical data, including purely native child arrays.
            origin_roots = sorted(set(origin_roots) | arrays.keys())
        serial = _name("fort_scope_serial_", digest)
        if allocated_roots and ("allocated" in names or
                                any(str(call.node.items[0]).lower() == "allocated" for call in calls)):
            raise CompilationError("allocation guard intrinsic conflicts with an original capture or call: allocated")
        if allocated_roots and any(str(call.node.items[0]).lower() == serial for call in calls):
            raise CompilationError("allocation guard coordinator conflicts with an original call: " + serial)
        bound_intrinsics = {"lbound", "ubound", "size"}
        if origin_roots and (bound_intrinsics & set(names)
                             or any(str(call.node.items[0]).lower() in bound_intrinsics for call in calls)):
            raise CompilationError("allocation bounds guard intrinsic conflicts with an original capture or call")
        # Captured module fields can be re-exported by the original USE list.
        # A dummy with the same spelling conflicts with use association, even
        # when it would legally shadow host association. Keep synthetic formals
        # private to this owner and map whole-variable actuals by root identity.
        parameters = {root: _name("fort_capture_", digest) + "_" + str(i)
                      for i, root in enumerate((*arrays, *scalars))}
        formal_names = list(parameters.values())
        bounds = {root: parameters[root] + "_lower" for root in arrays} if structure or borrowed or regional else {}
        formal_names += list(bounds.values())
        if caller_fallback:
            formal_names.append("fort_native_required")
        views = {root: parameters[root] + "_view" if root in arrays else parameters[root]
                 for root in parameters}

        def actuals(call, *, shared=True):
            if call.region is not None:
                return [views[binding.root] if shared else parameters[binding.root] for binding in call.region.bindings]
            values = []
            for mapping, actual in zip(call.resolved.mappings, call.actuals, strict=True):
                binding = self.analysis._actual_binding(self.analysis.source_scope_for(actual, routine.scope), actual)
                storage = parameters if not shared or "allocatable" in mapping.formal_binding.attributes else views
                if binding and binding.root in parameters:
                    values.append(storage[binding.root])
                elif actual is not None:
                    from compiler.scopes.segments import renamed
                    values.append(renamed(self, actual, storage))
                else:
                    values.append(str(actual))
            return values

        header = _fortran_list("subroutine " + name + "(", formal_names, ")", 0)
        imports = [str(n) for n in _children(_part(routine.scope.node, "Specification_Part"))
                   if _kind(n) == "Use_Stmt"]
        spec = []
        handles, layouts, flags = {}, {}, {}
        if structure or borrowed or regional:
            spec += [f"integer(c_int64_t), intent(in) :: {bounds[root]}({binding.rank})"
                     for root, binding in arrays.items()]
        if structure:
            spec += ["logical :: fort_branch"]
            if structure.guarded:
                spec += ["logical :: fort_numerical_guard"]
        if caller_fallback:
            spec += ["logical, intent(out) :: fort_native_required"]
        for i, (root, binding) in enumerate(arrays.items()):
            visible = parameters[root]
            dtype, enum, width = DTYPES[binding.signature()[:2]]
            handles[root] = "fort_handle_" + str(i)
            layouts[root] = "fort_layout_" + str(i)
            flags[root] = int(self.capture(binding)["initialized"] == "whole")
            if binding.intent == "in" and root in written:
                raise CompilationError("scope writes an INTENT(IN) capture")
            intent = "inout" if root in written else "in"
            descriptor = "allocatable" in binding.attributes
            shape = ','.join(f"{bounds[root]}({axis}):" for axis in range(1, binding.rank + 1)) if (structure or borrowed or regional) and not descriptor else ','.join(':' for _ in range(binding.rank))
            attributes = "allocatable, target" if descriptor else "target"
            spec += [*_fortran_line(f"{dtype}, {attributes}, intent({intent}) :: {visible}({shape})", 0),
                     *_fortran_line(f"{dtype}, pointer, contiguous :: {views[root]}({','.join(':' for _ in range(binding.rank))})", 0),
                     f"integer(c_size_t), target :: fort_extents_{i}({binding.rank})",
                     f"integer(c_int64_t), target :: fort_lowers_{i}({binding.rank})",
                     f"type(fort_scope_layout) :: {layouts[root]}"]
            fact=self.capture(binding)
            if fact["initialized"] == "sections" and fact["sections"]:
                spec += [f"type(fort_scope_section), target :: fort_defined_{i}({len(fact['sections'])})"]
                for j,_box in enumerate(fact["sections"]):
                    for bound in ["lower","upper"]:
                        spec += [f"integer(c_size_t), target :: fort_defined_{i}_{j}_{bound}({binding.rank})"]
        for root, binding in scalars.items():
            if binding.signature()[:2] not in DTYPES and binding.signature()[:2] != ("logical",4):
                raise CompilationError("unsupported scalar capture width")
            dtype = "logical" if binding.signature()[:2] == ("logical",4) else DTYPES[binding.signature()[:2]][0]
            intent = "in" if binding.intent == "in" or "parameter" in binding.attributes else "inout"
            spec += [f"{dtype}, intent({intent}) :: {parameters[root]}"]
        private = {name for call in calls if call.region is not None
                   for name in (*call.region.private_scalars, *call.region.private_arrays)}
        if structure:
            private.update(binding.name for operation in structure.native
                           for binding in operation.private_bindings.values())
        for private_name in sorted(private):
            binding = routine.scope.bindings[private_name]
            from compiler.scopes.regions import _fixed_shape
            shape = "(" + ",".join(_fixed_shape(routine, binding)) + ")" if binding.rank else ""
            signature = binding.signature()[:2]
            dtype = (DTYPES[signature][0] if signature in DTYPES else
                     "integer(c_int64_t)" if signature == ("integer", 8) else
                     "logical" if signature == ("logical", 4) else None)
            if dtype is None:
                raise CompilationError("native private storage has unsupported numeric type: " + binding.root)
            spec.append(f"{dtype} :: {private_name}{shape}")
        view_codes = {}
        view_block = _name("fort_views_", digest)
        scalar_values = {**parameters, **{root: self.visible(routine, root) for root in original_scalars}}
        if borrowed:
            for call_index, call in enumerate(calls):
                if not view_call(call):
                    continue
                codes = {}
                for mapping_index, mapping in enumerate(call.resolved.mappings):
                    if not mapping.formal_binding.rank:
                        continue
                    code = build_view(mapping, handles[mapping.resource], bounds[mapping.resource], scalar_values,
                                      f"fort_v{call_index}_{mapping_index}", on_error=("exit " + view_block,))
                    spec.extend(code.specification)
                    codes[mapping.formal] = code
                view_codes[id(call.node)] = codes
        spec += ["integer(c_int64_t) :: fort_context = 0",
                 *_fortran_list("integer(c_int64_t) ::", list(handles.values()), "", 0),
                 "integer(c_int) :: fort_status, fort_cleanup",
                 "type(c_ptr) :: fort_host_pointer",
                 f"integer(c_int), parameter :: fort_mode = {1 if self.config.policy == 'sections' else 2}",
                 "type(fort_scope_access) :: fort_access",
                 "type(fort_scope_plan_decision) :: fort_decision",
                 f"type(fort_scope_plan_binding), target :: fort_bindings({max(1, len(arrays))})"]
        # Explicit initialization on a local declaration implies SAVE. Assign at
        # entry instead: no scope-local context or state persists across calls.
        spec = [s.replace("fort_context = 0", "fort_context") for s in spec]
        original = structure.original(parameters) if structure else self.original_calls(calls, parameters, actuals, shared=False)
        # An original numerical procedure may access its hidden module arrays
        # directly. A non-TARGET actual cannot be updated through that alias
        # while the new owner dummy is associated. Return before any numerical
        # work and run the original span after the helper association ends.
        native = ["return"] if caller_fallback else [*original, "return"]
        body = [*(["fort_native_required = .true."] if caller_fallback else []), "fort_context = 0"]
        native_preflight = None
        static_native_reason = self.automatic_native_preflight(leaves)
        if static_native_reason:
            body += ["! Successful static native selection: " + static_native_reason, *native]
        elif not structure and (not query_available or (self.config.policy == "auto" and not planning_available)):
            body += ["! Whole-span native selection: " + (query_reason or planning_reason), *native]
        else:
            body += ["if (fort_scope_serial_caller() == 0) then", *native, "endif",
                     *_fortran_list("if (any([", [".not. is_contiguous(" + n + ")"
                                                 for n in formal_names[:len(arrays)]], "])) then", 0),
                     *native, "endif"]
            # Associate only after the original storage passed its checks. These
            # contiguous pointers are simply contiguous actuals, so subsequent
            # CONTIGUOUS helper dummies cannot acquire copy-in/out temporaries
            # that differ from the registered host storage. Empty views retain
            # their descriptors without accessing or manufacturing a payload.
            body += [f"{views[root]} => {parameters[root]}" for root in arrays]
            # These views have now proved the same contiguous original storage.
            # Use them on later native fallbacks too, so CONTIGUOUS callees do
            # not acquire whole-array temporaries from synthetic owner dummies.
            if not caller_fallback:
                native = (structure.original(views) if structure else
                          self.original_calls(calls, views, actuals)) + ["return"]
            if structure is None and self.config.policy == "auto":
                from compiler.scopes.native_preflight import source_native_preflight
                native_preflight = source_native_preflight(self, calls, leaves, scalar_values,
                                                          original_descriptors=bool(regional))
                if native_preflight.expression is not None:
                    imports += list(native_preflight.imports)
                    body += ["if (" + native_preflight.expression + ") then", *native, "endif"]
            body += ["fort_status = fort_scope_create(0_c_int, fort_context)"]
            body += ["if (fort_status == FORT_SCOPE_OK) &",
                     f"  fort_status = fort_scope_set_device_budget(fort_context, {self.device_budget}_c_size_t)"]
            body += self.owner_transfer_setup(leaves, imports)
            for i, (root, binding) in enumerate(arrays.items()):
                visible = parameters[root]
                _, enum, width = DTYPES[binding.signature()[:2]]
                body += ["if (fort_status == FORT_SCOPE_OK) then"]
                body += _fortran_list(f"fort_extents_{i} = [",
                                     [f"size({visible},{axis},kind=c_size_t)" for axis in range(1,binding.rank+1)], "]", 0)
                body += [f"fort_lowers_{i} = {bounds[root] if root in bounds else '1_c_int64_t'}",
                         "fort_host_pointer = c_null_ptr",
                         f"if (all(fort_extents_{i} > 0)) fort_host_pointer = c_loc({visible})",
                         f"{layouts[root]} = fort_scope_layout({binding.rank}, {enum}, {width}_c_size_t, &",
                         f"    fort_host_pointer, c_loc(fort_extents_{i}), c_loc(fort_lowers_{i}), 1_c_int64_t)"]
                fact=self.capture(binding)
                if fact["initialized"] == "sections":
                    boxes=fact["sections"]
                    for j,box in enumerate(boxes):
                        for bound in ["lower","upper"]:
                            body += _fortran_list(f"fort_defined_{i}_{j}_{bound} = [",
                                                  [f"{v}_c_size_t" for v in box[bound]], "]", 0)
                            body += [f"fort_defined_{i}({j+1})%{bound} = c_loc(fort_defined_{i}_{j}_{bound})"]
                    body += _fortran_list("fort_status = fort_scope_register_sections(",
                                          ["fort_context", f"{i+1}_c_int64_t", "1_c_int64_t", layouts[root],
                                           f"c_loc(fort_defined_{i})" if boxes else "c_null_ptr",
                                           f"{len(boxes)}_c_size_t", handles[root]], ")", 0)
                else:
                    body += _fortran_list("fort_status = fort_scope_register(",
                                         ["fort_context", f"{i+1}_c_int64_t", "1_c_int64_t",
                                          layouts[root], f"{flags[root]}_c_int", handles[root]], ")", 0)
                body += ["endif"]
            if borrowed:
                body += ["if (fort_status == FORT_SCOPE_OK) then", view_block + ": block"]
                body += [line for codes in view_codes.values() for code in codes.values() for line in code.prepare]
                body += ["end block " + view_block, "endif"]
            body += ["if (fort_status /= FORT_SCOPE_OK) then",
                     "if (fort_context /= 0) then",
                     "fort_cleanup = fort_scope_close(fort_context)",
                     "if (fort_cleanup /= FORT_SCOPE_OK) error stop 'shared scope preflight cleanup failed'",
                     "endif", *native, "endif"]
            if self.config.policy == "auto":
                selector = (self.entry_artifacts(sorted(leaves)[0], views=True)[0]
                            if borrowed else self.numerical(sorted(leaves)[0]).scoped)
                imports.append(f"use {selector['fortran_module']}, only: fort_choose => {selector['planning']['fortran_selector']}")
            if structure:
                body += structure.emit(handles, views, actuals, imports,
                                       selector=selector if self.config.policy == "auto" else None,
                                       logical_lower_bounds={root: tuple(f"{value}({axis})"
                                                                        for axis in range(1, arrays[root].rank + 1))
                                                             for root, value in bounds.items()})
                body += _checked("fort_scope_close(fort_context)")
            else:
                body += self.owner_complete_plan(calls, leaves, handles, actuals, imports, native, parameters=views,
                                                 view_codes=view_codes)
        if caller_fallback:
            body += ["fort_native_required = .false."]
        lines = [*header, "use iso_c_binding", "use fort_scoped_memory",
                 *dict.fromkeys(imports), "implicit none", *spec, *body, "end subroutine " + name, ""]
        if structure or borrowed or regional:
            from compiler.scopes.segments import fortran_lines
            lines = fortran_lines(lines)
        text = "\n".join(lines)
        self.append_procedure(routine.scope.parent, name, text)
        owner_actuals = [*names, *["[" + ",".join(f"lbound({self.visible(routine, root)},{axis},kind={caller_kind})"
                                                  for axis in range(1, arrays[root].rank + 1)) + "]" for root in bounds]]
        fallback_flag = _name("fort_native_", digest)
        if caller_fallback:
            owner_actuals.append(fallback_flag)
        replacement = "\n".join(_call(name,owner_actuals)) + "\n"
        if caller_fallback:
            original_source = "".join(routine.scope.path.read_text().splitlines(keepends=True)[first-1:last])
            replacement = ("block\nlogical :: " + fallback_flag + "\n" + replacement
                           + "if (" + fallback_flag + ") then\n" + original_source
                           + "endif\nend block\n")
        if structure or borrowed or regional:
            # Metadata is obtained from the original descriptor. The intrinsic
            # declaration also prevents an original local name from shadowing
            # the generated inquiry, before owner dummy association rebases it.
            if allocated_roots:
                replacement = "block\nuse iso_c_binding, only: " + caller_kind + " => c_int64_t\nintrinsic :: lbound\n" + replacement + "end block\n"
            else:
                original_source = "".join(routine.scope.path.read_text().splitlines(keepends=True)[first-1:last])
                replacement = ("block\nuse iso_c_binding, only: " + caller_kind + " => c_int64_t\n"
                               "use fort_scoped_memory, only: " + serial + " => fort_scope_serial_caller\n"
                               "intrinsic :: lbound\nif (" + serial + "() == 0) then\n" + original_source
                               + "else\n" + replacement + "endif\nend block\n")
        caller_guards = () if structure else runtime_guards
        if allocated_roots or origin_roots or caller_guards:
            replacement = self.guard_owner_allocation(routine, first, last, replacement, serial,
                                                      allocated_roots, origin_roots, arrays, caller_guards)
        if structure or borrowed or regional:
            replacement = "\n".join(fortran_lines(replacement.splitlines())) + "\n"
        compute_identity = None
        if not static_native_reason and self.config.policy == "auto":
            from compiler.scopes.compute_identity import compute_identity_guard
            compute_identity = compute_identity_guard(self, leaves, caller=routine)
            if not compute_identity.available:
                planning_available = False
                planning_reason = compute_identity.reason
            if compute_identity.required:
                from compiler.scopes.segments import fortran_lines
                original_source = "".join(routine.scope.path.read_text().splitlines(keepends=True)[first-1:last])
                replacement = "\n".join(fortran_lines([
                    "block", *compute_identity.imports, "if (" + compute_identity.expression + ") then",
                    *replacement.splitlines(), "else", *original_source.splitlines(), "endif", "end block"])) + "\n"
        if static_native_reason:
            # A compile-time all-native decision consumes no caller facts.
            # Keep the original statement bytes, rather than evaluating GPU
            # descriptor/IEEE guards or duplicating the native loop in arms
            # which can never reach a device worker. The unused owning helper
            # retains its public identity and is not called by this source.
            replacement = "".join(routine.scope.path.read_text().splitlines(keepends=True)[first-1:last])
        self.add_edit(routine.scope.path, first, last, replacement)
        scope = {"owner": name, "owner_variant": owner_variant.identity,
                "path": str(routine.scope.path), "first_line": first, "last_line": last,
                "parameters": [{"name": parameters[root], "resource": root, "actual": visible}
                               for root, visible in zip(parameters, names, strict=True)],
                "calls": [c.procedure for c in calls], "gpu_leaves": sorted(leaves),
                "resources": [{"resource": root, "visible": self.visible(routine,root),
                               "registration_identity": i+1, "allocation_generation": 1,
                               "initialized": self.capture(binding)["initialized"],
                               "initialized_sections": self.capture(binding).get("sections",[])}
                              for i,(root,binding) in enumerate(arrays.items())],
                "estimate_available": planning_available,
                "planning_reason": planning_reason,
                "definition_preflight": {"abi_version": 1, "query_available": query_available,
                                         "reason": query_reason, "position": "when segment is reached" if structure else "before numerical execution"},
                "cost_estimates": "separate original Fortran, coordinated CPU and GPU compute where calibrated; runtime costs charged separately",
                "native_estimate": ("original native body; no runtime coherence setup" if static_native_reason else
                                    "original Fortran counterfactual for fresh native fallback; coherent host execution for continuations"),
                "placement": "forced scoped GPU with native helpers" if self.config.policy == "sections" else
                             "calibrated coherent source scope" if planning_available else
                             "native; " + str(planning_reason),
                **({"allocation_preflight": {"resources": allocated_roots, "position": "original caller before owner association",
                                               "participation": "serial before allocation inquiries",
                                               "fallback": "unchanged original source span",
                                               **(self.bounds_preflight_public(origin_roots, "original module allocation descriptor")
                                                  if origin_roots else {})}} if allocated_roots else {}),
                "mode": self.config.policy, "participation": "serial"}
        if compute_identity is not None:
            scope["compute_identity"] = compute_identity.public()
        if native_preflight is not None:
            scope["native_preflight"] = native_preflight.public
        if static_native_reason:
            scope["automatic_preflight"] = {
                "selection": "native", "successful": True, "reason": static_native_reason,
                "authority": "compiler-produced offline estimate availability for every numerical alternative",
                "position": "before context creation, registration and query construction",
                "runtime_decision_inputs": [], "contexts_created": 0,
                "registrations": 0, "queries_constructed": 0,
                "caller_source_unchanged": True, "caller_guards_evaluated": False,
                "generated_owner_invoked": False,
                "execution": "unchanged original source span"}
        if runtime_guards:
            scope["runtime_preflight"] = {
                "conditions": list(runtime_guards),
                "position": "original reached segment after allocation and bounds guards",
                "evaluation_order": "source-ordered checks; a failed prerequisite prevents later evaluation",
                "fallback": ("original reached native operation with coherence hooks; earlier work retained" if structure
                             else "unchanged original source span")}
        scope["transfer_configuration"] = self.numerical(sorted(leaves)[0]).scoped["transfer_configuration"]
        if borrowed:
            scope["borrowed_views"] = {
                "abi_version": 2, "preflight": "all reached straight-span descriptors before numerical execution",
                "resource_identity": "canonical root registered once; child views borrow its generation and lifetime",
                "calls": [{"procedure": call.procedure, "first_line": _span(call.node)[0],
                           "last_line": _span(call.node)[1],
                           "mappings": [code.public for code in view_codes[id(call.node)].values()]}
                          for call in calls if view_call(call)],
                "boundaries": ["mutable or reached section bounds", "normalized package sections",
                               "native effects or aliases without exact projected proofs",
                               "optional or allocatable descriptor forwarding through view wrappers"]}
        chains = [item[3] for item in self.batch_chains.values()
                  if first <= item[3]["first_line"] <= item[3]["last_line"] <= last]
        if chains:
            scope["batch_subchains"] = chains
        if structure:
            scope.update(structure.public())
            scope["ownership"]["retained_resources"] = list(arrays)
        if original_scalars:
            scope["original_scalar_bindings"] = [{"resource": root, "visible": self.visible(routine, root),
                                                "storage_owner": "original defining module", "association": "host/use"}
                                               for root in original_scalars]
        if caller_fallback:
            scope["native_fallback"] = {"position": "original caller after owning helper returns",
                                        "phase": "before numerical execution only",
                                        "reason": "original hidden numerical array access requires ended dummy association"}
        return scope

    def owner_complete_plan(self, calls, leaves, handles, actuals, imports, native, *, parameters=None, view_codes=None):
        """Emit the unchanged whole-span plan used by legacy straight owners."""
        routine = self.entry
        body = ["fort_status = fort_scope_plan_reset(fort_context)"]
        for call in calls:
            child_leaves, _ = self.closure(call.procedure)
            mapping = {formal: binding.root for formal, binding in call.bindings.items()}
            native_handles = {formal: handles[root] for formal, root in mapping.items() if root in handles}
            native_handles.update({root: handle for root, handle in handles.items() if not root.startswith("argument::")})
            body += ["if (fort_status == FORT_SCOPE_OK) then"]
            if call.region is not None:
                body += self.inline.emit_call(call, handles, parameters or {}, imports, query=True)
            elif view_call(call) and child_leaves:
                body += self.view_numerical_call(call, view_codes[id(call.node)], actuals(call), imports, query=True)
            elif child_leaves:
                query, query_roots = self.query_clone(call.procedure)
                module = self.analysis.routines[call.procedure].scope.module
                if module != routine.scope.module:
                    imports.append(f"use {module}, only: {query}")
                body += _call(query, ["fort_context", *actuals(call),
                                      *[native_handles[root] for root in query_roots], "fort_status"])
            else:
                # This owner may fall back; RETURN in query glue belongs
                # to the query helper rather than the executing owner.
                query_lines = self.native_plan(call, *self.call_effects(call), native_handles,
                                               view_codes=view_codes.get(id(call.node)) if view_codes else None,
                                               actuals=actuals(call))
                block = _name("fort_record_", str(_span(call.node)))
                body += [block + ": block", *[("exit " + block if line.strip() == "return" else
                                              line.replace("if (fort_status /= FORT_SCOPE_OK) return",
                                                           "if (fort_status /= FORT_SCOPE_OK) exit " + block))
                                            for line in query_lines], "end block " + block]
            body += ["endif"]
        body += ["if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_validate(fort_context)"]
        fallback_condition = "fort_status /= FORT_SCOPE_OK"
        if self.config.policy == "auto":
            body += ["if (fort_status == FORT_SCOPE_OK) fort_status = fort_choose(fort_context, fort_decision)"]
            fallback_condition += " .or. fort_decision%gpu_units == 0"
        body += ["if (" + fallback_condition + ") then",
                 "fort_cleanup = fort_scope_close(fort_context)",
                 "if (fort_cleanup /= FORT_SCOPE_OK) error stop 'shared scope planning cleanup failed'",
                 *native, "endif"]
        body += self.owner_calls(calls, "fort_mode", handles, parameters or {}, actuals, imports, terminal=True,
                                 view_codes=view_codes)
        body += _checked("fort_scope_close(fort_context)")
        return body

    def view_numerical_call(self, call, codes, actuals, imports, *, query=False, mode="fort_mode", caller_module=None):
        if not self.numerical(call.procedure):
            name, roots = self.view_clone(call.procedure, query=query)
            module = self.analysis.routines[call.procedure].scope.module
            if module != (caller_module or self.entry.scope.module):
                imports.append(f"use {module}, only: {name}")
            return _call(name, ["fort_context", *([] if query else [mode]), *actuals,
                                *[codes[root.split("::")[-1]].name for root in roots],
                                *(["fort_status"] if query else [])])
        public, _ = self.entry_artifacts(call.procedure, views=True)
        function = public["planning"]["fortran_procedure"] if query else public["fortran_procedure"]
        alias = _name("fort_view_plan_" if query else "fort_view_run_", call.procedure)
        imports.append(f"use {public['fortran_module']}, only: {alias} => {function}")
        values = dict(zip(call.resolved.formals, actuals, strict=True))
        arguments = ["fort_context", *([] if query else [mode]),
                     *[codes[parameter["name"].lower()].name for parameter in public["array_parameters"]],
                     *[values[parameter["name"].lower()] for parameter in public["scalar_parameters"]
                       if not query or parameter["name"] in public["planning"]["scalar_inputs"]]]
        lines = _fortran_list("fort_status = " + alias + "(", arguments, ")", 0)
        if not query:
            lines += ["if (fort_status /= FORT_SCOPE_OK) error stop 'borrowed numerical entry failed'"]
        return lines

    def owner_calls(self, calls, mode, handles, parameters, actuals, imports, *, batch=True, terminal=False,
                    view_codes=None):
        """Run original approved calls, or one generic direct-leaf batch chain."""
        if batch and not any(view_call(call) or call.region is not None for call in calls) and self.config.scope_transfers in {"pipelined", "auto"}:
            from compiler.scopes.batch import execute_calls
            return execute_calls(self, calls, mode, handles, parameters, actuals, imports,
                                 lambda group: self.owner_calls(group, mode, handles, parameters, actuals, imports, batch=False),
                                 terminal=terminal)
        routine = self.entry
        body = []
        for call in calls:
            child_leaves, _ = self.closure(call.procedure)
            if call.region is not None:
                body += self.inline.emit_call(call, handles, parameters, imports, query=False, mode=mode)
            elif view_call(call) and child_leaves:
                body += self.view_numerical_call(call, view_codes[id(call.node)], actuals(call), imports, mode=mode)
            elif child_leaves:
                clone, arrays_ = self.clone(call.procedure)
                module = self.analysis.routines[call.procedure].scope.module
                if module != routine.scope.module:
                    imports.append(f"use {module}, only: {clone}")
                body += _call(clone, ["fort_context", mode, *actuals(call),
                                     *[handles[call.bindings[root].root if root.startswith("argument::") else root]
                                       for root in arrays_]])
            else:
                mapping = {formal: binding.root for formal, binding in call.bindings.items()}
                actions, definitions, overwrites = self.call_effects(call)
                native_handles = {formal: handles[root] for formal, root in mapping.items() if root in handles}
                # Hidden visible roots retain their canonical identity.
                native_handles.update({root: handle for root,handle in handles.items() if not root.startswith("argument::")})
                body += self.native_call(call, actions, definitions, overwrites, native_handles,
                                         actuals=actuals(call),
                                         view_codes=view_codes.get(id(call.node)) if view_codes else None)
        return body

    def original_calls(self, calls, parameters, actuals, *, shared=True):
        result = []
        for call in calls:
            if call.region is not None:
                result += self.inline.original(call, parameters)
            else:
                result += _call(str(call.node.items[0]), call.original_arguments(actuals(call, shared=shared)))
        return result

    def guard_owner_allocation(self, routine, first, last, replacement, serial, allocated_roots, origin_roots, arrays,
                               runtime_guards=()):
        # Passing an unallocated actual to this ordinary assumed-shape
        # owner is already too early. Inspect only allocation state here,
        # at the original caller, after proving serial participation.
        original = "".join(routine.scope.path.read_text().splitlines(keepends=True)[first-1:last])
        if not original.endswith("\n"):
            original += "\n"
        conditions = ["allocated(" + self.visible(routine, root) + ")" for root in allocated_roots]
        guard = (["if ( &", *[condition + " .and. &" for condition in conditions[:-1]],
                  conditions[-1] + " &", ") then"] if conditions else [])
        guard_intrinsics = "allocated" if conditions else ""
        kind_import = ""
        guard_digest = f"{routine.qualified}:{first}:{last}"
        environment_name = _name("fort_scope_environment_", guard_digest)
        check_name = _name("fort_scope_guard_", guard_digest)
        check_declaration = ""
        if runtime_guards:
            if self.analysis._binding(routine.scope, check_name) or self.analysis._candidates(routine.scope, F.Name(check_name)):
                raise CompilationError("original source conflicts with numerical guard variable")
            check_declaration = "logical :: " + check_name + "\n"
            if any("kind=c_int64_t" in condition for condition in runtime_guards):
                guard_intrinsics += (", " if guard_intrinsics else "") + "int"
                kind_name = _name("fort_scope_guard_i64_", guard_digest)
                if (self.analysis._binding(routine.scope, kind_name)
                        or self.analysis._candidates(routine.scope, F.Name(kind_name))):
                    raise CompilationError("original source conflicts with numerical guard INTEGER kind alias")
                kind_import = "use iso_c_binding, only: " + kind_name + " => c_int64_t\n"
                runtime_guards = tuple(condition.replace("c_int64_t", kind_name) for condition in runtime_guards)
            if any("fort_scope_numerical_environment_supported" in condition for condition in runtime_guards):
                if (self.analysis._binding(routine.scope, environment_name)
                        or self.analysis._candidates(routine.scope, F.Name(environment_name))):
                    raise CompilationError("original source conflicts with numerical environment guard alias")
                runtime_guards = tuple(condition.replace("fort_scope_numerical_environment_supported", environment_name)
                                       for condition in runtime_guards)
            checks = [check_name + " = .true.",
                      *["if (" + check_name + ") " + check_name + " = " + condition for condition in runtime_guards],
                      "if (" + check_name + ") then"]
            replacement = "\n".join(checks) + "\n" + replacement + "else\n" + original + "endif\n"
        if origin_roots:
            conditions = [condition for root in origin_roots
                          for condition in self.original_bound_conditions(self.visible(routine, root), arrays[root].rank)]
            bounds = ["if ( &", *[condition + " .and. &" for condition in conditions[:-1]],
                      conditions[-1] + " &", ") then"]
            replacement = "\n".join(bounds) + "\n" + replacement + "else\n" + original + "endif\n"
            guard_intrinsics += (", " if guard_intrinsics else "") + "lbound, ubound, size"
        environment_import = (", " + environment_name + " => fort_scope_numerical_environment_supported"
                              if any(environment_name in guard for guard in runtime_guards) else "")
        replacement = ("block\n" + kind_import + "use fort_scoped_memory, only: " + serial + " => fort_scope_serial_caller" + environment_import + "\n"
                       + ("intrinsic :: " + guard_intrinsics + "\n" if guard_intrinsics else "")
                       + check_declaration
                       + "if (" + serial + "() == 0) then\n" + original + "else\n"
                       + ("\n".join(guard) + "\n" + replacement + "else\n" + original + "endif\n" if guard else replacement)
                       + "endif\nend block\n")
        return replacement

    def build_span(self, nodes):
        if not nodes:
            return
        before = self.scope_checkpoint()
        try:
            calls = [self.resolve(self.entry,node) for node in nodes]
            delegated = any(self.coordinator(call.procedure) is not None for call in calls if call.region is None)
            if len(nodes) < 2 and not delegated and not any(call.region is not None for call in calls):
                return
            budget = self.scope_budget(calls)
            leaves = set()
            for call in calls:
                own, _ = self.closure(call.procedure)
                leaves.update(own)
            if not leaves:
                return
            arrays, _scalars, written = self.owner_inputs(calls)
            query = self.owner_query(calls, arrays, written)
            if delegated or not query[0] and query[1] == "planning scalar inputs change inside the complete source scope":
                from compiler.scopes.segments import StructuredScope
                scope = self.owner(calls, leaves, structure=StructuredScope(self, nodes))
            else:
                scope = self.owner(calls,leaves)
            scope["effect_closure"] = budget
            self.scopes.append(scope)
        except CompilationError as error:
            self.restore_scope_checkpoint(before)
            self.boundaries.append({"first_line":_span(nodes[0])[0], "last_line":_span(nodes[-1])[1],
                                    "reason":str(error)})

    def scope_budget(self, calls):
        # Reusable source graphs are bounded per procedure. Reached segments
        # have their own expansion budget; mutually exclusive branches must not
        # spend one eagerly flattened transitive closure budget together.
        if len(calls) > 32:
            raise CompilationError("bounded source scope exceeds call generation budget")
        summaries, graphs, depth = {}, {}, 0
        for call in calls:
            if call.region is not None:
                original = self.inline.original_selection(call.node)
                operations = sum(_kind(node) in {"Assignment_Stmt", "Block_Nonlabel_Do_Construct", "If_Stmt", "If_Construct"}
                                 for root in original for node in walk(root))
                summaries[call.procedure] = operations
                continue
            summary = call.summary
            coordinator = self.coordinator(call.procedure) if self.config.scope_execution == "reached" else None
            if coordinator is None and not summary["complete"]:
                raise CompilationError("reached source-call effects incomplete")
            graph = self.analysis.structure(call.procedure)
            if not graph.available:
                raise CompilationError("bounded source structure unavailable: " + "; ".join(graph.reasons))
            graphs[call.procedure] = graph.identity
            summaries[call.procedure] = (len(graph.nodes) - 2 if coordinator is not None else
                                        len(summary.get("ordered_effects", summary["operations"])))
            depth = max(depth, summary.get("closure_depth", 1))
        operations = max(summaries.values(), default=0)
        if operations > self.analysis.operation_limit or len(summaries) > self.analysis.procedure_limit:
            raise CompilationError("reached source demand budget exhausted")
        return {"procedures": len(summaries), "operations": operations, "depth": depth,
                "budget_role": "maximum independently proved reached demand; source graphs are reused",
                "structured_summaries": graphs,
                "reached_demands": [{"procedure": procedure, "operations": count}
                                    for procedure, count in summaries.items()]}

    def scope_checkpoint(self):
        return (dict(self.outputs), {path:list(edits) for path,edits in self.edits.items()},
                dict(self.clones), dict(self.queries), dict(self.generated), dict(self.numerical_reasons),
                dict(self.numerical_ir), dict(self.batch_chains), self.variants.checkpoint(), dict(self.view_generated),
                dict(self.view_clones), dict(self.view_queries),
                {procedure: set(registry.used) for procedure, registry in self.region_owners.items()})

    def restore_scope_checkpoint(self, checkpoint):
        (self.outputs, self.edits, self.clones, self.queries,
         self.generated, self.numerical_reasons, self.numerical_ir, self.batch_chains, variants, self.view_generated,
         self.view_clones, self.view_queries, used) = checkpoint
        self.variants.restore(variants)
        for procedure, registry in self.region_owners.items():
            registry.used = set(used.get(procedure, ()))

    def scan(self, nodes):
        from compiler.scopes.segments import StructuredScope, statement_span
        # A structured candidate is bounded and source-backed before edits are
        # emitted. Rejected candidates retain the previous straight-span scan.
        if any(_kind(node) in {"If_Construct", "If_Stmt", "Associate_Construct", "Assignment_Stmt", "Block_Nonlabel_Do_Construct"}
               for node in nodes):
            candidate = []

            def finish_candidate():
                if not candidate:
                    return
                before = self.scope_checkpoint()
                try:
                    structure = StructuredScope(self, candidate)
                    calls = structure.calls
                    leaves = {leaf for call in calls for leaf in self.closure(call.procedure)[0]}
                    has_outer_work = any(_kind(node) in {"Call_Stmt", "Assignment_Stmt", "Block_Nonlabel_Do_Construct"}
                                         for node in candidate)
                    if leaves and len(calls) >= 2 and has_outer_work:
                        budget = self.scope_budget(calls)
                        scope = self.owner(calls, leaves, structure=structure)
                        scope["effect_closure"] = budget
                        self.scopes.append(scope)
                        return
                except CompilationError as error:
                    self.restore_scope_checkpoint(before)
                    self.boundaries.append({"first_line":statement_span(candidate[0])[0],
                                            "last_line":statement_span(candidate[-1])[1], "reason":str(error)})
                self.scan_straight(candidate)

            for node in nodes:
                nested_boundaries = {"Return_Stmt", "Exit_Stmt", "Cycle_Stmt", "Allocate_Stmt", "Deallocate_Stmt",
                                     "Pointer_Assignment_Stmt", "Nullify_Stmt", "Stop_Stmt", "Error_Stop_Stmt"}
                if (_kind(node) in {"If_Stmt", "If_Construct", "Associate_Construct", "Block_Nonlabel_Do_Construct"}
                        and any(_kind(item) in nested_boundaries for item in walk(node))):
                    finish_candidate()
                    candidate = []
                    first, last = statement_span(node)
                    self.boundaries.append({"first_line":first, "last_line":last,
                                            "reason":"unsupported exit or lifetime boundary in original guarded operation"})
                    if _kind(node) == "If_Construct":
                        self.scan_straight((node,))
                    continue
                if _kind(node) == "Call_Stmt":
                    try:
                        self.resolve(self.entry, node)
                    except CompilationError:
                        finish_candidate()
                        candidate = []
                        self.scan_straight((node,))
                        continue
                if _kind(node) in {"Call_Stmt", "If_Construct", "If_Stmt", "Associate_Construct", "Assignment_Stmt",
                                   "Block_Nonlabel_Do_Construct", "Comment"}:
                    candidate.append(node)
                else:
                    finish_candidate()
                    candidate = []
                    first, last = statement_span(node)
                    reason = ("storage lifetime/association boundary" if _kind(node) in {
                        "Allocate_Stmt", "Deallocate_Stmt", "Pointer_Assignment_Stmt", "Nullify_Stmt"}
                              else "unsupported exit or native ordering boundary: " + _kind(node))
                    self.boundaries.append({"first_line":first, "last_line":last, "reason":reason})
            finish_candidate()
            return
        self.scan_straight(nodes)

    def scan_straight(self, nodes):
        run = []
        for node in nodes:
            if _kind(node) == "Comment" and not str(node).lower().lstrip().startswith("!$omp"):
                continue
            if _kind(node) == "Call_Stmt":
                try:
                    first,last = _span(node)
                except CompilationError as error:
                    self.build_span(run)
                    run = []
                    self.boundaries.append({"analysis_first_line":node.item.span[0],"reason":str(error)})
                    continue
                original = self.entry.scope.path.read_text().splitlines()
                text = original[first-1:last]
                if run and any(line.lstrip().startswith("#") for line in original[_span(run[-1])[1]:first-1]):
                    self.build_span(run)
                    run = []
                    self.boundaries.append({"first_line":first,"last_line":last,
                                            "reason":"preprocessor control boundary between source calls"})
                if (len(run) < 32 and text and (text[0].lstrip().lower().startswith("call ") or hasattr(node, "fort_inline_region"))
                        and all(";" not in line for line in text)):
                    try:
                        self.resolve(self.entry,node)
                    except CompilationError as error:
                        self.build_span(run)
                        run = []
                        self.boundaries.append({"first_line":first,"last_line":last,"reason":str(error)})
                    else:
                        run.append(node)
                    continue
            self.build_span(run)
            run = []
            if _kind(node) == "If_Construct":
                self.scan(node.content)
        self.build_span(run)

    def run(self):
        if self.entry.source_kind != "module":
            first = _span(_part(self.entry.scope.node, "Subroutine_Stmt"))[0]
            last = _span(_part(self.entry.scope.node, "End_Subroutine_Stmt"))[1]
            self.boundaries.append({"first_line": first, "last_line": last,
                                   "reason": "external source entry execution requires standalone procedure variants"})
            return self.finish()
        if self.config.scope_execution == "reached":
            from compiler.scopes.lexical import check_structured_entry
            try:
                check_structured_entry(self.entry)
            except CompilationError as error:
                # Do not fall through to outlined scopes: a source label can
                # enter an outlined statement or bypass its coherence setup.
                self.boundaries.append({"procedure": self.entry.qualified, "reason": str(error),
                                        "execution": "unchanged native source"})
                return self.finish()
        nodes = self.inline.prepare(_children(self.entry.execution)) if not self.config.collective else _children(self.entry.execution)
        self.generated.update(self.inline.generated)
        self.numerical_ir.update(self.inline.ir)
        self.numerical_reasons.update({procedure: None for procedure in self.inline.regions})
        if self.config.scope_execution == "reached" and not self.config.collective:
            from compiler.scopes.lexical import LexicalOwner
            before = self.scope_checkpoint()
            try:
                if LexicalOwner(self).run(nodes):
                    return self.finish()
            except CompilationError as error:
                self.restore_scope_checkpoint(before)
                self.boundaries.append({"reason": "lexical owner unavailable: " + str(error),
                                        "fallback": "existing bounded source scopes"})
        self.scan(nodes)
        return self.finish()

    def finish(self):
        """Publish source edits and build artifacts after approved scope work."""
        self.analysis.inputs.verify()
        if any(sha256(package.path.read_bytes()).hexdigest() != package.digest for package in self.packages.values()):
            raise CompilationError("normalized source changed while scope artifacts were being prepared")
        for registry in self.region_owners.values():
            if registry.used:
                registry.refresh(self)
                registry.finish()
        provenance = {}
        patches = []
        for path, edits in self.edits.items():
            source = path.read_text()
            lines = source.splitlines(keepends=True)
            # Group insertions at the same boundary; preserve discovery order.
            grouped = {}
            for first,last,replacement in edits:
                grouped.setdefault((first,last),[]).append(replacement)
            ordered = sorted(grouped.items(), reverse=True)
            previous = len(lines)+1
            for (first,last),values in ordered:
                if last >= previous:
                    raise CompilationError("overlapping compiler source edits")
                lines[first-1:last] = ["".join(values)]
                previous = first if last >= first else previous
                patches.append({"source":str(path), "source_sha256":self.analysis.sources[str(path)],
                                "first_line":first, "last_line":last, "replacement":"".join(values)})
            destination = "sources/" + _name("source_",str(path)) + ".f90"
            self.outputs[destination] = "".join(lines)
            provenance[str(path)] = {"sha256":self.analysis.sources[str(path)], "replacement":destination}
        if self.scopes:
            self.outputs.update(self.runtime_outputs)
            self.outputs["scoped-runtime.json"] = json.dumps(self.runtime,indent=2)+"\n"
        report = {
            "schema_version":1, "entry":self.entry.qualified,
            "scope_execution": self.config.scope_execution,
            **({"reached_plan_schema_version": 1} if self.config.scope_execution == "reached" else {}),
            "automatic_scope_available":bool(self.scopes), "scope_count":len(self.scopes),
            "scopes":self.scopes, "boundaries":self.boundaries, "source_edits":patches,
            "sources":provenance, "runtime":self.runtime if self.scopes else None,
            "native_effects":self.analysis.report(self.entry.qualified,
                materialize=self.config.scope_execution != "reached"),
            "resolved_calls": [self.resolved_calls[key] for key in sorted(self.resolved_calls)],
            "implementation_variants": self.variants.public(),
            "inline_numerical_regions": self.inline.public(),
            "child_numerical_regions": [{"procedure": procedure, **registry.public()}
                                        for procedure, registry in self.region_owners.items()
                                        if procedure != self.entry.qualified and registry.used],
            "automatic_estimate_available":any(scope["estimate_available"] for scope in self.scopes),
            "capture_facts_sha256":sha256(json.dumps(self.facts,sort_keys=True).encode()).hexdigest(),
            "source_inputs":self.analysis.sources,
            "analysis_sources":self.analysis.inputs.public(),
            "numerical_sources":[{"procedure":p.procedure,"path":str(p.path),"entry":p.entry,"sha256":p.digest}
                                 for p in sorted(self.packages.values(), key=lambda p:p.procedure)],
            "numerical_decisions":[{"procedure":procedure,"supported":bool(self.generated[procedure]),
                                    "reason":self.numerical_reasons[procedure],
                                    **({"batch_execution": self.generated[procedure].scoped["batch_execution"]}
                                       if self.generated[procedure] else {})}
                                   for procedure in sorted(self.generated)],
            "borrowed_source_coordinators": [coordinator.public() for procedure, coordinator in self.coordinators.items()
                                              if coordinator is not None and procedure in self.clones],
            "source_coordinator_boundaries": [{"procedure": procedure, "reason": reason}
                                               for procedure, reason in sorted(self.coordinator_reasons.items()) if reason],
            "artifacts_sha256":{name:sha256(content.encode()).hexdigest() for name,content in self.outputs.items()},
            "device_budget_bytes":self.device_budget,
            "limits":{"calls_per_span":32, "closure_depth":8,
                      "physical_native_sections":"bounded original-coordinate rectangles; conservative whole-resource fallback",
                      "native_rectangles_per_resource":32},
            "build_sources":[
                {"path":name,"language":"cuda" if name.endswith(".cu") else "fortran",
                 "role":"common_runtime" if name in self.runtime_outputs else
                        "original_source" if name.startswith("sources/") else "shared_entry"}
                for name in self.outputs if name.endswith((".cu",".f90"))
            ],
        }
        self.outputs["scope-manifest.json"] = json.dumps(report,indent=2)+"\n"
        return self.outputs, report


def form_source_scopes(paths, entry, *, facts, options, config, contracts=None, numerical_sources=None,
                       analysis_sources=None, summary_cache=None):
    if isinstance(facts, dict) and facts.get("schema_version") == 2:
        from compiler.scopes.collective import CollectiveScopeBuilder
        return CollectiveScopeBuilder(paths, entry, facts=facts, options=options, config=config,
                                      contracts=contracts, numerical_sources=numerical_sources,
                                      analysis_sources=analysis_sources, summary_cache=summary_cache).run()
    return ScopeBuilder(paths,entry,facts=facts,options=options,config=config,contracts=contracts,
                        numerical_sources=numerical_sources, analysis_sources=analysis_sources, summary_cache=summary_cache).run()
