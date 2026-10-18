"""One mode-bearing dispatcher for borrowed inline regions of an original owner."""

import copy
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
from types import SimpleNamespace

from fparser.two import Fortran2003 as F
from fparser.two.utils import Base, walk

from compiler.emission import generate_sources
from compiler.emission.fortran.formatting import _fortran_list
from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError
from compiler.scopes.regions import _exception_observers, extract_region


class InlineRegions:
    def __init__(self, builder):
        self.builder = builder
        self.regions, self.nodes, self.generated, self.ir, self.entries = {}, {}, {}, {}, {}
        self.used = set()
        self.bindings = {}
        self.element_operands = {}
        self.definition_proofs = {}
        self.prepared_regions = {}
        self.compute_applicability = {}
        self.original_numerical_regions = {}
        self.collective_regions = set()
        self.computations = {}
        previous = getattr(builder, "region_owners", {}).get(builder.entry.qualified)
        self.attempts = previous.attempts if previous is not None else 0
        self.operations = previous.operations if previous is not None else 0
        # These identities authorize projections made here, never an arbitrary
        # AST carrying a plausible span or fort_inline_region attribute.
        self.source_selections = {}
        self.facade_proofs = {}
        self._native_environment_required = None
        self.name = "fort_regions_" + sha256(builder.entry.qualified.encode()).hexdigest()[:12]
        self.path = "regions/" + self.name + ".f90"

    def native_environment_required(self):
        """Guard original invocation descendants that can use IEEE state.

        A parent's regions can be outlined before a borrowed child is scanned.
        Resolve bounded original direct calls from the configured entry up
        front. This only restricts execution, never issues native authority.
        """
        if self._native_environment_required is None:
            from compiler.frontend.native_environment import intrinsic_export
            analysis = self.builder.analysis
            root = self.builder._numerical_environment_root
            pending = [root]
            # This traversal only reads original calls and namespace records.
            # Share syntax checks within this demand, never across builders.
            syntax = analysis._source_syntax.checkpoint()
            visited, calls = set(), 0
            required = False
            while pending and not required:
                procedure = pending.pop()
                if procedure in visited:
                    continue
                if len(visited) >= analysis.procedure_limit:
                    required = True
                    break
                analysis._require_original(procedure, _syntax=syntax)
                routine = analysis.routines[procedure]
                visited.add(procedure)
                for call in walk(routine.execution, F.Call_Stmt):
                    calls += 1
                    if calls > analysis.operation_limit:
                        required = True
                        break
                    if _kind(call.items[0]) != "Name":
                        continue  # Unknown calls already end reached ownership.
                    scope = analysis.source_scope_for(call, routine.scope)
                    if intrinsic_export(analysis, scope, call.items[0]) in {
                            "$intrinsic::ieee_exceptions::ieee_get_halting_mode",
                            "$intrinsic::ieee_exceptions::ieee_set_halting_mode"}:
                        required = True
                        break
                    # Traverse all source-backed generic candidates; unknown
                    # calls grant no borrowed/native continuation authority.
                    pending.extend(target for target in analysis._candidates(scope, call.items[0])
                                   if target in analysis.routines and target not in visited)
            # Recheck syntax and every visited namespace before publishing.
            # This second checkpoint is fresh even when a scan stopped early.
            syntax = analysis._source_syntax.checkpoint()
            for procedure in sorted(visited | {root}):
                analysis._require_original(procedure, _syntax=syntax)
            self._native_environment_required = required
        return self._native_environment_required

    def refresh(self, root):
        """Retain lexical authority while rebinding rolled-back artifact maps."""
        for name in ("outputs", "edits", "clones", "queries", "generated", "numerical_reasons",
                     "numerical_ir", "batch_chains", "variants", "view_generated", "view_clones", "view_queries"):
            setattr(self.builder, name, getattr(root, name))

    def _register_projection(self, node, sources):
        from compiler.scopes.segments import statement_span
        self.source_selections[id(node)] = (node, tuple(sources), str(node), statement_span(node))

    def statement_projection(self, action, header):
        """Attach an original single-line IF span without changing its AST."""
        from compiler.scopes.segments import statement_span
        self.original_selection(header)
        if _kind(header) != "If_Stmt" or header.items[1] is not action:
            raise CompilationError("statement projection requires the original IF action")
        projected = copy.copy(action)
        span = statement_span(header)
        projected.item = SimpleNamespace(span=span, fort_original_span=span, label=None, name=None)
        self._register_projection(projected, (action,))
        return projected

    def grouped_nodes(self, nodes):
        """Group approved nodes and register any peeled directive projections."""
        from compiler.scopes.segments import grouped_nodes
        nodes = tuple(nodes)
        for node in nodes:
            self.original_selection(node)
        groups = grouped_nodes(nodes)
        for group in groups:
            for node in group if isinstance(group, tuple) else (group,):
                if id(node) in self.source_selections or any(node is original for original in nodes):
                    continue
                source = next((candidate for candidate in nodes
                               if _kind(candidate) in {"Block_Nonlabel_Do_Construct", "If_Construct", "Associate_Construct"}
                               and _kind(node) == _kind(candidate)
                               and node.content and len(candidate.content) >= len(node.content)
                               and all(left is right for left, right in
                                       zip(candidate.content[-len(node.content):], node.content, strict=True))), None)
                if source is None:
                    # Exposed prefix comments remain original descendants.
                    self.original_selection(node)
                else:
                    self._register_projection(node, self.original_selection(source))
        return groups

    def prepare(self, nodes, *, closed_selections=()):
        """Outline bounded numerical work, including reached structured branches."""
        from compiler.scopes.segments import grouped_nodes, statement_span
        original = tuple(nodes)
        authoritative = {id(node): node for node in walk(self.builder.entry.execution)}
        retained = {tuple(map(id, selection)): selection for selection in closed_selections}
        for selection in closed_selections:
            self.builder.analysis._selected_source(self.builder.entry.qualified, selection)

        def register_projection(node, sources):
            self._register_projection(node, sources)

        def original_group(group, sequence):
            selected = []
            for node in group:
                if id(node) in authoritative:
                    selected.append(node)
                    continue
                # grouped_nodes peels the original directive prefix off a DO.
                # Recover only that compiler-created projection, by exact
                # child identity, while we hold its authoritative input list.
                source = next((candidate for candidate in sequence
                               if _kind(candidate) in {"Block_Nonlabel_Do_Construct", "If_Construct", "Associate_Construct"}
                               and _kind(node) == _kind(candidate)
                               and node.content and len(candidate.content) >= len(node.content)
                               and all(left is right for left, right in
                                       zip(candidate.content[-len(node.content):], node.content, strict=True))), None)
                if source is None:
                    raise CompilationError("inline grouping lost original source authority")
                register_projection(node, (source,))
                selected.append(source)
            # A peeled directive can also be contained in its selected DO.
            # Materialize the original construct once, including its prefix.
            descendants = {id(child) for node in selected for child in walk(node) if child is not node}
            unique = {id(node): node for node in selected}
            return tuple(node for key, node in unique.items() if key not in descendants)
        lifetime = {"Return_Stmt", "Exit_Stmt", "Cycle_Stmt", "Allocate_Stmt", "Deallocate_Stmt",
                    "Pointer_Assignment_Stmt", "Nullify_Stmt", "Stop_Stmt", "Error_Stop_Stmt"}

        def context(sources):
            from compiler.scopes.region_context import original_region_context
            original = original_region_context(self.builder.entry, sources)
            return original.preceding, original.following, original.cyclic_following

        def prepare_sequence(sequence, depth=0):
            result = []
            try:
                groups = grouped_nodes(sequence)
            except CompilationError:
                return tuple(sequence)
            for grouped in groups:
                group = tuple(grouped) if isinstance(grouped, tuple) else (grouped,)
                sources = original_group(group, sequence)
                node = group[0]
                if tuple(map(id, sources)) in retained:
                    from compiler.scopes.closed_selection import ClosedSelectionBoundary

                    result.append(ClosedSelectionBoundary(sources))
                    continue
                if len(group) == 1 and _kind(node) in {"If_Construct", "Associate_Construct"}:
                    # Lifetimes/exits remain original boundaries. In particular
                    # do not borrow a region through a possibly reallocating
                    # branch and then pretend that its owner is unchanged.
                    association = self.builder.analysis._associate_scopes.get(id(node)) if _kind(node) == "Associate_Construct" else None
                    if (depth < 8 and (_kind(node) != "Associate_Construct" or association is not None and association.available)
                            and not any(_kind(item) in lifetime for item in walk(node))):
                        clone = copy.copy(node)
                        clone.content = list(prepare_sequence(node.content, depth + 1))
                        register_projection(clone, sources)
                        result.append(clone)
                    else:
                        result.append(node)
                    continue
                array_operation = len(group) == 1 and _kind(node) == "Assignment_Stmt"
                if array_operation:
                    from compiler.frontend.component_bindings import source_scope_for
                    original_target = node.items[0]
                    base = original_target.items[0] if _kind(original_target) == "Part_Ref" else original_target
                    target = self.builder.analysis._binding(source_scope_for(self.builder.analysis, original_target,
                                                                            self.builder.entry.scope), base)
                    array_operation = target is not None and bool(target.rank)
                if not array_operation and not any(_kind(item) == "Block_Nonlabel_Do_Construct" for item in group):
                    result.extend(group)
                    continue
                if len(group) == 1 and _kind(node) == "Block_Nonlabel_Do_Construct" and depth < 8:
                    from compiler.scopes.counted_loops import has_coordinator_call, reached_counted_body
                    if has_coordinator_call(self.builder, node):
                        try:
                            body = reached_counted_body(self.builder, node)
                        except CompilationError:
                            pass
                        else:
                            # A host coordinator is not an eagerly expanded
                            # numerical candidate. Charge its reached children
                            # individually under the unchanged shared budget.
                            clone = copy.copy(node)
                            start = next(index for index, child in enumerate(node.content)
                                         if _kind(child) == "Nonlabel_Do_Stmt")
                            clone.content = [*node.content[:start+1], *prepare_sequence(body.nodes, depth+1),
                                             node.content[-1]]
                            register_projection(clone, sources)
                            result.append(clone)
                            continue
                first, last = statement_span(group[0])[0], statement_span(group[-1])[1]
                before, after, cyclic = context(sources)
                try:
                    self._reserve(group)
                    if array_operation:
                        from compiler.scopes.array_operations import extract_array_operation
                        extraction = extract_array_operation(self.builder.analysis, self.builder.entry, node,
                                                             preceding=before, following=after)
                    else:
                        extraction = extract_region(self.builder.analysis, self.builder.entry,
                                                    sources if len(sources) > 1 else sources[0],
                                                    preceding=before, following=after,
                                                    cyclic_following=cyclic,
                                                    immutable_inputs=getattr(self.builder, "immutable_inputs", {}))
                    result.append(self._outline(extraction, sources, collective=self.builder.config.collective))
                except CompilationError as error:
                    self.builder.boundaries.append({"first_line": first, "last_line": last,
                                                    "reason": "inline numerical boundary: " + str(error),
                                                    "kind": "candidate_rejection",
                                                    "phase": "inline_numerical_extraction"})
                    if len(group) == 1 and _kind(node) == "Block_Nonlabel_Do_Construct" and depth < 8:
                        try:
                            from compiler.scopes.counted_loops import reached_counted_body
                            body = reached_counted_body(self.builder, node)
                        except CompilationError:
                            result.extend(group)
                        else:
                            clone = copy.copy(node)
                            start = next(index for index, child in enumerate(node.content)
                                         if _kind(child) == "Nonlabel_Do_Stmt")
                            clone.content = [*node.content[:start+1], *prepare_sequence(body.nodes, depth+1),
                                             node.content[-1]]
                            register_projection(clone, sources)
                            result.append(clone)
                    else:
                        result.extend(group)
            return tuple(result)

        return prepare_sequence(original)

    def _reserve(self, nodes):
        """Charge attempts before extraction, across both participation modes."""
        operations = sum(_kind(item) in {"Assignment_Stmt", "Block_Nonlabel_Do_Construct", "If_Stmt", "If_Construct"}
                         for node in nodes for item in walk(node))
        if self.attempts >= 32 or self.operations + operations > 256:
            raise CompilationError("bounded inline region/operation budget exhausted")
        self.attempts += 1
        self.operations += operations

    def prepare_worksharing(self, loop, proof, *, preceding=(), following=(), prepared=None):
        """Register one original DO for execution by its complete original team.

        The caller retains the original team and native effects. This method
        authenticates participation and numerical legality only; its facade
        cannot be executed through the serial dispatcher.
        """
        from compiler.frontend.numerical_preparation import NumericalPreparationProvider, PreparedOriginalRegion

        sources = self.original_selection(loop)
        if len(sources) != 1:
            raise CompilationError("worksharing preparation requires one exact original loop")
        if prepared is None:
            provider = NumericalPreparationProvider(self.builder.analysis, options=self.builder.options,
                config=self.builder.config, environment_root=self.builder._numerical_environment_root,
                immutable_inputs=getattr(self.builder, "immutable_inputs", {}))
            prepared = provider.prepare_worksharing(self.builder.analysis, self.builder.entry.qualified,
                sources[0], proof, preceding=preceding, following=following)
        if type(prepared) is not PreparedOriginalRegion or prepared.completion_proof is not proof:
            raise CompilationError("worksharing preparation requires its exact original-team token")
        return self._outline(prepared.extraction, sources, collective=True, reservation=sources, prepared=prepared)

    def prepare_combined_team(self, nodes):
        """Consume one retained whole numerical group through its original team.

        This is not a worksharing cut or native-effect proof. The distinct
        combined-team consumer retains the complete original join and checks
        every participant before any reached registration or query.
        """
        from compiler.scopes.segments import original_roots

        sources = original_roots(self.original_selection(nodes))
        region = self.builder._consume_numerical_region(
            self.builder.analysis, self.builder.entry.qualified, sources)
        if region is None:
            region = self.prepare_numerical_region(sources)
        retained = region.prepared
        if retained.completion_proof is None:
            raise CompilationError("combined numerical team requires complete original numerical authority")
        # Preparation validates the exact registered completion type: simple
        # helper-free groups and pure-helper closures have distinct token ABIs.
        return self._outline(retained.extraction, sources, collective=True, reservation=sources)

    def resolved_numerical_region(self, sources):
        """Reauthenticate a pair issued by this original lexical inline owner."""
        from compiler.frontend.numerical_effects import NumericalEffectProof
        from compiler.frontend.numerical_preparation import PreparedOriginalRegion
        from compiler.frontend.reached_windows import _ResolvedNumericalRegion

        sources = tuple(sources)
        region = self.original_numerical_regions.get(tuple(id(node) for node in sources))
        if region is None:
            return None
        builder = self.builder
        if (type(region) is not _ResolvedNumericalRegion or region.procedure != builder.entry.qualified
                or len(sources) != len(region.selection)
                or any(left is not right for left, right in zip(sources, region.selection, strict=True))
                or type(region.prepared) is not PreparedOriginalRegion
                or type(region.effects) is not NumericalEffectProof):
            raise CompilationError("ordinary numerical region lost its exact original pair")
        region.prepared.validate_configuration(options=builder.options, config=builder.config,
            environment_root=builder._numerical_environment_root,
            immutable_inputs=getattr(builder, "immutable_inputs", {}))
        region.effects.validate(builder.analysis, builder.entry.qualified, sources, region.prepared)
        return region

    def prepare_numerical_region(self, sources):
        """Issue source computation/effect evidence without capture or emission.

        A bound reached plan remains the sole authority for its conditional
        region. Ordinary lexical owners retain only the same source-issued
        pair; storage admission and reached registration stay independent.
        """
        from compiler.frontend.numerical_effects import prove_numerical_effects
        from compiler.frontend.numerical_preparation import NumericalPreparationProvider
        from compiler.frontend.reached_windows import _ResolvedNumericalRegion

        builder = self.builder
        sources = tuple(sources)
        region = builder.resolved_numerical_region(builder.analysis, builder.entry.qualified, sources)
        if region is not None:
            return region
        if builder.validate_reached_window() is not None:
            raise CompilationError("combined numerical team requires its registered reached preparation")
        if len(self.original_numerical_regions) >= min(32, builder.analysis.operation_limit):
            raise CompilationError("ordinary numerical region registry budget exhausted")
        provider = NumericalPreparationProvider(builder.analysis, options=builder.options, config=builder.config,
            environment_root=builder._numerical_environment_root,
            immutable_inputs=getattr(builder, "immutable_inputs", {}))
        prepared = provider.prepare(builder.analysis, builder.entry.qualified, sources)
        effects = prove_numerical_effects(builder.analysis, builder.entry.qualified, sources, prepared)
        region = _ResolvedNumericalRegion(builder.entry.qualified, sources, prepared, effects)
        self.original_numerical_regions[tuple(id(node) for node in sources)] = region
        try:
            region = self.resolved_numerical_region(sources)
            consumer = builder._ordinary_pair_state.active
            if consumer is not None:
                consumer.enroll(self, region)
            return region
        except BaseException:
            del self.original_numerical_regions[tuple(id(node) for node in sources)]
            raise

    def validate_numerical_regions(self):
        """Check retained local capabilities again before source publication."""
        from compiler.frontend.reached_windows import _ResolvedNumericalRegion

        if len(self.original_numerical_regions) > min(32, self.builder.analysis.operation_limit):
            raise CompilationError("ordinary numerical region registry budget exhausted")
        for key, region in self.original_numerical_regions.items():
            if type(region) is not _ResolvedNumericalRegion or key != tuple(id(node) for node in region.selection):
                raise CompilationError("ordinary numerical region registry selection changed")
            self.resolved_numerical_region(region.selection)

    def _outline(self, extraction, sources, *, collective, reservation=None, prepared=None):
        """Lower an authenticated extraction and register its private facade."""
        from compiler.frontend.numerical_preparation import (
            NumericalPreparationProvider,
            PreparedOriginalRegion,
            _prepare,
        )
        if prepared is not None:
            if type(prepared) is not PreparedOriginalRegion or extraction is not prepared.extraction:
                raise CompilationError("inline preparation requires its exact original extraction")
            prepared.validate_configuration(options=self.builder.options, config=self.builder.config,
                environment_root=self.builder._numerical_environment_root,
                immutable_inputs=getattr(self.builder, "immutable_inputs", {}))
            prepared.validate(self.builder.analysis, self.builder.entry.qualified, sources)
        if collective and extraction.scalar_element_captures:
            raise CompilationError("immutable element preparation requires a serial reached preflight; collective execution remains native")
        # Publishing arrays cannot reconstruct CPU exception flags produced by
        # device arithmetic. Apply the same configured-source observer boundary
        # as array operations before acquiring any capture or artifact authority.
        if observers := _exception_observers(self.builder.analysis):
            raise CompilationError("source-observable floating-point exception flags prevent GPU numerical closure: "
                                   + "; ".join(observers))
        native_environment = self.native_environment_required()
        if native_environment and not collective and extraction.completion.get("has_openmp_in_closure"):
            # A serial caller's mask cannot establish the original workers'
            # masks. Keep the team for JoinedTeam's uniform reached preflight.
            raise CompilationError("original IEEE environment requires an original-team numerical preflight")
        region = self.builder._consume_numerical_region(
            self.builder.analysis, self.builder.entry.qualified, sources)
        if region is not None:
            retained = region.prepared
            if prepared is not None and retained is not prepared:
                raise CompilationError("inline preparation differs from its registered numerical effects")
            # Source preparation retains the exact original full context. The
            # extraction below is only enriched by reached descriptor guards.
            extraction = retained.extraction
        elif prepared is not None:
            retained = prepared
        else:
            provider = NumericalPreparationProvider(self.builder.analysis, options=self.builder.options,
                config=self.builder.config, environment_root=self.builder._numerical_environment_root,
                immutable_inputs=getattr(self.builder, "immutable_inputs", {}))
            retained = _prepare(self.builder.analysis, self.builder.entry.qualified, sources, provider,
                                extraction=extraction, require_effects=False)
            extraction = retained.extraction
        for binding in extraction.bindings:
            if binding.rank:
                self.builder.capture(binding)
        if hasattr(self.builder, "_coordinator_root"):
            # Canonical storage bounds checked by the parent do not prove a
            # child's rebased logical INTEGER ABI. Read that descriptor only
            # at the reached original region, after its allocation guards.
            for intrinsic in ("lbound", "ubound", "size"):
                if (self.builder.analysis._binding(self.builder.entry.scope, intrinsic)
                        or self.builder.analysis._candidates(self.builder.entry.scope, F.Name(intrinsic))):
                    raise CompilationError("child region bounds conflict with original " + intrinsic.upper())
            guards = tuple(condition for binding in extraction.bindings if binding.rank
                           for condition in self.builder.original_bound_conditions(
                               self.builder.visible(self.builder.entry, binding.root), binding.rank))
            extraction = replace(extraction, runtime_guards=(*guards, *extraction.runtime_guards))
        source_name = str(self.builder.entry.scope.path) + "#inline:" + extraction.source_identity
        # Capture is independently authorized before reusing prepared IR.
        function, plan = retained.function, retained.plan
        if reservation is not None:
            # Original parent/unit analysis is bounded separately. Native
            # corrections that fail complete numerical legality must not use
            # the generation budget of a later supported worksharing unit.
            self._reserve(reservation)
        completion = extraction.completion
        if collective:
            native_participation = "existing_team"
        elif completion.get("has_openmp_in_closure"):
            from compiler.offload.source_compute import fork_join_participation
            native_participation = fork_join_participation(extraction.nodes)
        else:
            native_participation = "serial" if completion.get("available") else "unknown"
        from compiler.offload.dependency_source import current_source_bundle_requested
        from compiler.offload.source_applicability import source_compute_applicability
        applicability = (source_compute_applicability(self.builder.analysis, retained,
                                                     native_participation=native_participation)
                         if isinstance(self.builder.config.profile, dict)
                         and current_source_bundle_requested(self.builder.config.profile) else None)
        # Classify the exact original preparation before rewriting locations
        # for artifact reuse. Lowered arithmetic alone cannot prove the native
        # source's compiled form or participation protocol.
        function, plan = (self.artifact_ir(value, source_name, extraction.source_identity)
                          for value in (function, plan))
        # Observations and placement belong to the checked original owner,
        # never a separately generated numerical worker.
        config = replace(self.builder.config, collective=collective, native_participation=native_participation,
                         source_applicability=applicability, scope_measurement=None, exact_window=None)
        generated = generate_sources(function, plan, offload_config=config, memory_model="scoped")
        if generated.scoped is None:
            raise CompilationError("inline numerical candidate has no shared numerical entry")
        if any(item.name in generated.scoped["planning"]["scalar_inputs"]
               for item in extraction.scalar_element_captures):
            raise CompilationError("immutable element values cannot affect numerical query metadata")
        procedure = self.builder.entry.qualified + "#region" + str(len(self.regions) + 1)
        # The generated artifact includes its original native counterfactual.
        # Identical arithmetic cannot share that artifact across serial,
        # fork/join and existing-team participation contracts.
        computation = sha256("\n".join(extraction.source.splitlines()[1:-1]).encode()).hexdigest()
        cost_contract = ((applicability.native_compiled_form, applicability.generated_compiled_form,
                          applicability.native_participation, applicability.access_class)
                         if applicability is not None else None)
        canonical = self.computations.setdefault((collective, native_participation, computation, cost_contract), procedure)
        if canonical != procedure:
            generated, function, plan = self.generated[canonical], *self.ir[canonical]
        self.regions[procedure], self.generated[procedure] = extraction, generated
        self.prepared_regions[procedure] = retained
        # Each source region keeps its own provenance, even when identical
        # work reuses an artifact issued for a different original region.
        self.compute_applicability[procedure] = applicability
        self.ir[procedure], self.entries[procedure] = (function, plan), canonical
        if collective:
            self.collective_regions.add(procedure)
        self.bindings.update((binding.root, binding) for binding in extraction.bindings)
        self.element_operands.update((item.resource, item) for item in extraction.scalar_element_captures)
        facade = F.Call_Stmt("call " + self.name + "()")
        facade.item = SimpleNamespace(span=extraction.span, fort_original_span=extraction.span, label=None, name=None)
        facade.fort_inline_region = procedure
        self.nodes[procedure] = facade
        self._register_projection(facade, sources)
        self.facade_proofs[procedure] = extraction
        return facade

    def original_selection(self, node):
        """Recover authoritative original nodes from approved preparations.

        Source summaries use these nodes rather than interpreting synthetic
        calls, their arguments or normalized numerical INTENT declarations.
        """
        analysis, routine = self.builder.analysis, self.builder.entry
        session = getattr(analysis, "_source_session", None)
        if session is None:
            raise CompilationError("inline selection requires compilation-session source authority")
        originals = session.execution_nodes(analysis, routine)
        return self._original_selection(node, originals)

    def _original_selection(self, node, originals):
        if isinstance(node, (tuple, list)):
            return tuple(source for child in node for source in self._original_selection(child, originals))
        registered = self.source_selections.get(id(node))
        if registered is not None:
            from compiler.scopes.segments import statement_span
            original_node, sources, text, span = registered
            if node is not original_node or str(node) != text or statement_span(node) != span:
                raise CompilationError("inline source projection was changed after registration")
            if hasattr(node, "fort_inline_region"):
                procedure = node.fort_inline_region
                if (self.nodes.get(procedure) is not node
                        or self.regions.get(procedure) is not self.facade_proofs.get(procedure)):
                    raise CompilationError("inline numerical facade lacks its registered proof")
            return sources
        if id(node) in originals and originals[id(node)] is node:
            return (node,)
        raise CompilationError("source selection is not an original node or approved inline projection")

    @staticmethod
    def artifact_ir(value, source_name, identity):
        """Canonicalize private locations and source identities for artifacts.

        Exact original paths remain on the registered preparation and source
        manifest. A private entry's provenance hashes must use its stable
        extraction identity rather than the caller's temporary directory.
        """
        if isinstance(value, (tuple, list)):
            return type(value)(InlineRegions.artifact_ir(child, source_name, identity) for child in value)
        if isinstance(value, str):
            return value.replace(source_name, "inline:" + identity)
        if is_dataclass(value) and not isinstance(value, type):
            return replace(value, **{field.name: InlineRegions.artifact_ir(getattr(value, field.name), source_name, identity)
                                     for field in fields(value)})
        return value

    def call(self, node):
        from compiler.scopes.source import Call
        self.original_selection(node)
        procedure = node.fort_inline_region
        region = self.regions[procedure]
        bindings = {"argument::" + parameter.name: self.bindings[parameter.resource]
                    for parameter in region.parameters if parameter.lower_bound_dimension is None
                    and parameter.resource not in self.element_operands}
        return Call(node, procedure, tuple(F.Name(binding.name) for binding in bindings.values()), bindings,
                    {"complete": True, "summary_identity": region.source_identity}, region=region)

    def restore(self, node):
        """Expand private CALL facades before emitting any native fallback."""
        def visit(value):
            if isinstance(value, (tuple, list)):
                result = []
                for child in value:
                    if hasattr(child, "fort_inline_region"):
                        self.original_selection(child)
                        result.extend(self.regions[child.fort_inline_region].nodes)
                    else:
                        result.append(visit(child))
                return type(value)(result)
            if not isinstance(value, Base):
                return value
            if hasattr(value, "fort_inline_region"):
                self.original_selection(value)
                return self.regions[value.fort_inline_region].nodes
            # fparser comments do not implement Base's pickle/copy protocol
            # (they have no ``string``). Keep their source and directive
            # metadata intact while restoring neighbouring private facades.
            if _kind(value) == "Comment":
                from compiler.frontend.sealed_syntax import mutable_copy
                result = mutable_copy(value)
            else:
                result = copy.copy(value)
            for attribute in ("content", "items"):
                if hasattr(value, attribute):
                    setattr(result, attribute, visit(getattr(value, attribute)))
            return result
        restored = visit(node)
        return restored if isinstance(restored, tuple) else (restored,)

    def original(self, call, parameters):
        from compiler.scopes.segments import renamed
        return [line for node in call.region.nodes for line in renamed(self.builder, node, parameters).splitlines()]

    def effects(self, call, *, mapped=False):
        result = {}
        for parameter in call.region.parameters:
            if not parameter.rank:
                continue
            root = parameter.resource if mapped else "argument::" + parameter.name
            result[root] = {"read", "write"} if parameter.resource in call.region.written_resources else {"read"}
        return result, set(), set()

    def whole_definitions(self, call):
        """Must-write coverage authenticated against the original region."""
        if self.builder._consume_call_numerical_effect(call) is not None:
            # Exact partial definition transfer belongs to the retained typed
            # effects proof. A bounding box never promotes a whole overwrite.
            return set()
        if call.procedure not in self.definition_proofs:
            summary = self.builder.segment_summary(
                self.builder.analysis, self.builder.entry.qualified, call.region.nodes, capture_locals=True)
            self.definition_proofs[call.procedure] = frozenset(
                summary["guaranteed_whole_overwrites"] if summary["complete"] else ())
        return set(self.definition_proofs[call.procedure])

    def inputs(self, procedure, mapping, *, estimates):
        if estimates and self.regions[procedure].scalar_element_captures:
            raise CompilationError("immutable element preparation is uncalibrated")
        public = self.generated[procedure].scoped
        planning = public["planning"]
        if estimates and (not planning["available"] or not planning["profile_available"]):
            raise CompilationError(planning["reason"] or planning["profile_reason"])
        if not estimates and not planning["query_available"]:
            raise CompilationError(planning["query_reason"])
        parameters = {parameter.name: parameter for parameter in self.regions[procedure].parameters}
        payload = {parameters[name.lower()].resource for name in planning["payload_arrays"]}
        scalars = {parameters[name.lower()].resource for name in planning["scalar_inputs"]
                   if parameters[name.lower()].lower_bound_dimension is None}
        return {mapping.get(root, root) for root in payload}, {mapping.get(root, root) for root in scalars}

    @property
    def arrays(self):
        return {root: binding for root, binding in sorted(self.bindings.items()) if binding.rank}

    @property
    def scalars(self):
        return {root: binding for root, binding in sorted(self.bindings.items()) if not binding.rank}

    def parameters(self):
        arrays = {root: "fort_buffer_" + str(index) for index, root in enumerate(self.arrays)}
        scalars = {root: "fort_scalar_" + str(index) for index, root in enumerate(self.scalars)}
        scalars.update((root, "fort_element_" + str(index))
                       for index, root in enumerate(sorted(self.element_operands)))
        lowers = {(root, axis): "fort_lower_" + str(index) + "_" + str(axis)
                  for index, (root, binding) in enumerate(self.arrays.items()) for axis in range(1, binding.rank + 1)}
        return arrays, scalars, lowers

    def emit_call(self, call, handles, values, imports, *, query, mode="fort_mode", collective=False,
                  status="fort_status", check_status=True):
        from compiler.scopes.source import DTYPES
        procedure, region = call.procedure, call.region
        if not query and collective != (procedure in self.collective_regions):
            raise CompilationError("inline numerical execution requires its proven serial or original-team participation")
        self.builder.entry_artifacts(procedure)
        self.used.add(procedure)
        self.builder.variants.register(
            self.builder.entry.qualified, interface="source_inline_dispatch_v1", role="region_dispatcher",
            name=self.name + "::run", summary_identity=(
                self.builder.analysis.structure(self.builder.entry.qualified).identity
                if self.builder.config.scope_execution == "reached" else
                self.builder.analysis.summarize(self.builder.entry.qualified)["summary_identity"]),
            requirements=("bounded original region IDs", "original saved storage and guards retained", "mode-bearing CPU/GPU execution"),
            shared_artifacts=(self.path,))
        target = "query" if query else "run_team" if collective else "run"
        alias = "fort_inline_query" if query else "fort_inline_team_run" if collective else "fort_inline_run"
        imports.append(f"use {self.name}, only: {alias} => {target}")
        active = {binding.root for binding in region.bindings}
        arguments = ["fort_context", *([] if query else [mode]), str(list(self.regions).index(procedure) + 1) + "_c_int"]
        arguments += [handles[root] if root in active else "0_c_int64_t" for root in self.arrays]
        zero = {"real(c_double)": "0.0_c_double", "real(c_float)": "0.0_c_float", "integer(c_int)": "0_c_int",
                "logical(c_bool)": ".false._c_bool"}
        arguments += [(values[root] if root in values else self.builder.visible(self.builder.entry, root))
                      if root in active else zero[DTYPES[binding.signature()[:2]][0]]
                      for root, binding in self.scalars.items()]
        captured = {item.resource: item for item in region.scalar_element_captures}
        arguments += [(captured[root].name if not query and root in captured else
                       zero[DTYPES[(item.dtype, item.kind)][0]])
                      for root, item in sorted(self.element_operands.items())]
        arguments += [f"int(lbound({values[root]},{axis},kind=c_int64_t),c_int)" if root in active else "0_c_int"
                      for root, binding in self.arrays.items() for axis in range(1, binding.rank + 1)]
        lines = _fortran_list(status + " = " + alias + "(", arguments, ")", 0)
        if not query and captured:
            # Operand preparation is reached work, never part of a future
            # query. Nested IFs preserve dormant inner bounds and indices;
            # Fortran logical expressions do not promise short-circuiting.
            prefix = ["block"]
            for item in captured.values():
                dtype = DTYPES[(item.dtype, item.kind)][0]
                prefix += [f"{dtype} :: {item.name}"]
            prefix += [f"{status} = FORT_SCOPE_OK"]
            for item in captured.values():
                dtype = DTYPES[(item.dtype, item.kind)][0]
                prefix += [f"{item.name} = {zero[dtype]}", f"if ({status} == FORT_SCOPE_OK) then"]
                prefix += [f"if ({guard}) then" for guard in item.activation_guards]
                prefix += [f"if ({item.index_guard}) then", f"{item.name} = {item.expression}", "else",
                           f"{status} = FORT_SCOPE_BOUNDARY", "endif"]
                prefix += ["endif"] * (len(item.activation_guards) + 1)
            lines = [*prefix, f"if ({status} == FORT_SCOPE_OK) then", *lines, "endif", "end block"]
        if not query and check_status:
            lines += [f"if ({status} /= FORT_SCOPE_OK) error stop 'inline numerical region failed'"]
        return lines

    def finish(self):
        if not self.used:
            return
        from compiler.scopes.segments import fortran_lines
        from compiler.scopes.source import DTYPES
        arrays, scalars, lowers = self.parameters()
        imports, cases = [], {"query": [], "run": [], "run_team": []}
        for procedure in self.regions:
            if procedure not in self.used:
                continue
            public, _ = self.builder.entry_artifacts(procedure)
            region = self.regions[procedure]
            region_id = list(self.regions).index(procedure) + 1
            mapping = {parameter.name: parameter for parameter in region.parameters}
            targets = {"query": public["planning"]["fortran_procedure"]}
            if procedure in self.collective_regions:
                targets["run_team"] = public["team"]["fortran_procedure"]
            else:
                targets["run"] = public["fortran_procedure"]
            for operation, target in targets.items():
                query = operation == "query"
                alias = "fort_entry_" + operation + "_" + str(region_id)
                imports.append(f"use {public['fortran_module']}, only: {alias} => {target}")
                arguments = ["context", *([] if query else ["mode"])]
                arguments += [arrays[mapping[item["name"].lower()].resource] for item in public["array_parameters"]]
                for item in public["scalar_parameters"]:
                    if query and item["name"] not in public["planning"]["scalar_inputs"]:
                        continue
                    parameter = mapping[item["name"].lower()]
                    arguments.append(lowers[parameter.resource, parameter.lower_bound_dimension]
                                     if parameter.lower_bound_dimension else scalars[parameter.resource])
                cases[operation] += [f"case ({region_id})", *_fortran_list("status = " + alias + "(", arguments, ")", 0)]
            self.builder.outputs["regions/" + region.source_identity[:16] + ".source"] = region.source
        lines = ["module " + self.name, "use iso_c_binding", "use fort_scoped_memory", *dict.fromkeys(imports),
                 "implicit none", "private", "public :: run, query" + (", run_team" if cases["run_team"] else ""), "contains"]
        scalar_types = {root: DTYPES[(self.element_operands[root].dtype, self.element_operands[root].kind)
                                    if root in self.element_operands else self.scalars[root].signature()[:2]][0]
                        for root in scalars}
        for name in ("query", "run", *(("run_team",) if cases["run_team"] else ())):
            query = name == "query"
            parameters = ["context", *([] if query else ["mode"]), "region", *arrays.values(), *scalars.values(), *lowers.values()]
            lines += _fortran_list("function " + name + "(", parameters, ") result(status)", 0)
            lines += ["integer(c_int64_t), intent(in) :: context", "integer(c_int), intent(in) :: region",
                      *([] if query else ["integer(c_int), intent(in) :: mode"]), "integer(c_int) :: status"]
            lines += [f"integer(c_int64_t), intent(in) :: {name}" for name in arrays.values()]
            lines += [f"{scalar_types[root]}, intent(in) :: {name}"
                      for root, name in scalars.items()]
            lines += [f"integer(c_int), intent(in) :: {name}" for name in lowers.values()]
            lines += ["select case (region)", *cases[name], "case default", "status = FORT_SCOPE_BOUNDARY",
                      "end select", "end function " + name]
        lines += ["end module " + self.name, ""]
        self.builder.outputs[self.path] = "\n".join(fortran_lines(lines))

    def public(self):
        return {"dispatcher": self.name + "::run" if self.used else None, "variant_interface": "source_inline_dispatch_v1",
                "team_dispatcher": self.name + "::run_team" if self.used & self.collective_regions else None,
                "limits": {"regions": 32, "operations": 256},
                "regions": [{**region.public(), "region_id": index + 1, "used": procedure in self.used,
                             "execution_participation": "qualified_original_team" if procedure in self.collective_regions else "serial_coordinator",
                             "source_compute_applicability": (self.compute_applicability[procedure].public()
                                 if self.compute_applicability[procedure] is not None else None),
                             "shared_compute_applicability": self.generated[procedure].scoped["compute_estimates"]["source_applicability"],
                             "shared_computation": self.entries[procedure]} for index, (procedure, region) in enumerate(self.regions.items())],
                "boundaries": ["unknown effects and allocation changes",
                               "saved or live scalar outputs", "unsupported numerical computation"]}
