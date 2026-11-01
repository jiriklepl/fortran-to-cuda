"""Search original storage intervals around connected numerical preparation.

This module only orders candidates. Syntactic lifetime hints and prepared
numerics never grant descriptor, capture, boundary or execution authority.
Every original candidate remains eligible through the complete fallback order.
"""

from collections import deque

from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_original_call
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError
from compiler.scopes.segments import statement_span


def original_candidates(graph):
    """The complete original longest-prefix order, without a work-count cap."""
    for sequence in graph.nodes.values():
        if sequence.kind != "sequence":
            continue
        children = tuple(sequence.children)
        for start in range(len(children) - 1):
            if graph.nodes[children[start]].kind not in {"call", "branch"}:
                continue
            end = start
            while end < len(children) - 1 and graph.nodes[children[end]].kind in {"call", "branch"}:
                end += 1
            for stop in range(end, start, -1):
                yield children[start:stop], children[stop]


class _Priority:
    def __init__(self, owner, roots, diagnostics):
        self.owner, self.builder, self.graph = owner, owner.builder, owner.graph
        self.analysis, self.roots, self.diagnostics = owner.builder.analysis, set(roots), diagnostics
        self.calls, self.frontiers, self.preparations = {}, {}, {}
        self.provider = None

    def call(self, identity):
        if identity not in self.calls:
            (original,) = self.graph.source_nodes(identity)
            try:
                call = resolve_original_call(
                    self.analysis, self.analysis.source_scope_for(original, self.owner.routine.scope), original
                )
            except CompilationError as error:
                self.diagnostics["source_call_declines"].append({"node_id": identity, "reason": str(error)})
                call = None
            self.calls[identity] = call
            self.diagnostics["resolved_call_sites"] += 1
        return self.calls[identity]

    def frontier(self, identity):
        """A direct-callee syntax hint, never a transitive lifetime conclusion."""
        originals = self.graph.source_nodes(identity)
        if len(originals) != 1 or _kind(originals[0]) != "Call_Stmt":
            return None  # A branch keeps its existing control-local partition.
        call = self.call(identity)
        if call is None:
            return None
        if call.procedure not in self.frontiers:
            routine = self.analysis.routines[call.procedure]
            self.analysis._require_original(routine.qualified)
            self.diagnostics["direct_callee_syntax_scans"] += 1
            reason = None
            for argument in routine.arguments:
                binding = routine.scope.bindings[argument]
                if "allocatable" in binding.attributes and binding.intent == "out":
                    reason = "direct callee has allocatable OUT storage"
                    break
            for node in walk(routine.execution):
                if _kind(node) in {"Allocate_Stmt", "Deallocate_Stmt", "Pointer_Assignment_Stmt", "Nullify_Stmt"}:
                    reason = "direct callee contains " + _kind(node)
                    break
                if _kind(node) == "Call_Stmt" and str(node.items[0]).lower() == "move_alloc":
                    # Even a shadowed spelling may lower search priority. The
                    # real lifetime issuer resolves it before authorizing a cut.
                    reason = "direct callee contains a move_alloc candidate"
                    break
            self.frontiers[call.procedure] = reason
        return self.frontiers[call.procedure]

    def proposals(self):
        for sequence in self.graph.nodes.values():
            if sequence.kind != "sequence":
                continue
            current = []
            for identity in sequence.children:
                node = self.graph.nodes[identity]
                reason = self.frontier(identity)
                if reason is not None:
                    if current:
                        yield tuple(current), identity, reason
                    current = []
                elif node.kind in {"call", "branch"}:
                    # A priority group activates at a reached call. Initial
                    # branches may allocate its storage; they remain in the
                    # original candidate inventory, not this numerical seed.
                    if current or node.kind == "call":
                        current.append(identity)
                else:
                    current = []

    def numerical(self, procedure):
        """Find one prepared original numerical task, without admitting work."""
        if procedure in self.preparations:
            self.diagnostics["preparation_cache_hits"] += 1
            return self.preparations[procedure]
        if self.provider is None:
            self.provider = NumericalPreparationProvider(
                self.analysis,
                options=self.builder.options,
                config=self.builder.config,
                environment_root=self.builder._numerical_environment_root,
                immutable_inputs=getattr(self.builder, "immutable_inputs", {}),
            )
        pending, seen = deque(((procedure, 0),)), set()
        result = None
        while pending:
            name, depth = pending.popleft()
            if name in seen:
                continue
            if depth > self.analysis.depth_limit or len(seen) >= self.analysis.procedure_limit:
                self.diagnostics["preparation_search_limited"] = True
                break  # Only this priority search ends; original graph remains.
            self.analysis._require_original(name)
            seen.add(name)
            self.diagnostics["numerical_procedure_visits"] += 1
            routine = self.analysis.routines[name]
            peeled = {}

            def groups(nodes, peeled=peeled):
                for item in grouped_original_sequence(nodes, peeled):
                    if isinstance(item, NativeGroupCandidate):
                        yield item.original_nodes, True
                    elif _kind(item) == "Block_Nonlabel_Do_Construct":
                        yield (item,), False
                    elif getattr(item, "content", None) is not None:
                        yield from groups(_children(item))

            for originals, team in groups(_children(routine.execution)):
                row = {"procedure": name, "span": [statement_span(originals[0])[0], statement_span(originals[-1])[1]]}
                self.diagnostics["numerical_preparations"].append(row)
                try:
                    prepared = self.provider.prepare(self.analysis, name, originals)
                except CompilationError as error:
                    row.update(prepared=False, reason=str(error))
                    if not team:
                        continue
                    from compiler.scopes.worksharing_inventory import prepare_worksharing_inventory

                    try:
                        _parent, candidates = prepare_worksharing_inventory(
                            self.builder, name, originals, provider=self.provider
                        )
                    except CompilationError:
                        continue
                    prepared = next(
                        (candidate.prepared for candidate in candidates if candidate.prepared is not None), None
                    )
                    if prepared is None:
                        continue
                row.update(prepared=True, preparation_identity=prepared.identity)
                result = {"procedure": name, "preparation_identity": prepared.identity}
                break
            if result is not None:
                break
            for node in walk(routine.execution):
                if _kind(node) != "Call_Stmt":
                    continue
                try:
                    resolved = resolve_original_call(
                        self.analysis, self.analysis.source_scope_for(node, routine.scope), node
                    )
                except CompilationError:
                    continue
                if resolved.procedure not in seen:
                    pending.append((resolved.procedure, depth + 1))
        self.preparations[procedure] = result
        return result

    def useful(self, selected):
        from compiler.scopes.graph_epochs import _call_nodes

        try:
            nodes = _call_nodes(self.owner, selected)
        except CompilationError:
            return None
        calls = []
        for identity in nodes:
            call = self.call(identity)
            if call is None:
                continue
            roots = {
                self.owner.canonical(mapping.binding).root
                for mapping in call.associations
                if mapping.binding is not None and mapping.formal_binding.rank
            }
            roots &= self.roots
            if roots:
                calls.append((identity, call.procedure, roots))
        # Exact shared capture connectivity is a search signal, not alias or
        # payload-effect authority. Native readers can connect numerical tasks.
        components = []
        for identity, procedure, roots in calls:
            linked = [component for component in components if component[0] & roots]
            members = [(identity, procedure)]
            for component in linked:
                roots |= component[0]
                members += component[1]
                components.remove(component)
            components.append((roots, members))
        order = {identity: index for index, (identity, _procedure, _roots) in enumerate(calls)}
        for _roots, members in components:
            if len(members) < 2:
                continue
            witnesses = []
            for identity, procedure in sorted(members, key=lambda member: order[member[0]]):
                if witness := self.numerical(procedure):
                    witnesses.append({"call_node_id": identity, **witness})
                    if len(witnesses) == 2:
                        return witnesses  # Rank the complete connected span.
        return None


def candidates(owner, roots):
    """Prefer connected prepared work before explicit source lifetime hints."""
    diagnostics = owner.position_epoch_search = {
        "policy": "connected numerical preparation before direct lifetime hints",
        "execution_authority": False,
        "all_original_candidates_retained": True,
        "numerical_preparations": [],
        "preparation_search_limited": False,
        "attempts": [],
        "source_proposals": [],
        "source_call_declines": [],
        "resolved_call_sites": 0,
        "direct_callee_syntax_scans": 0,
        "numerical_procedure_visits": 0,
        "preparation_cache_hits": 0,
    }
    priority, seen = _Priority(owner, roots, diagnostics), set()
    try:
        for selected, cut, reason in priority.proposals():
            proposal = {
                "selected_node_ids": list(selected),
                "cut_before": cut,
                "reason": reason,
                "numerical_priority": "searching",
            }
            diagnostics["source_proposals"].append(proposal)
            try:
                witnesses = priority.useful(selected)
            except CompilationError:
                witnesses = None  # A heuristic failure cannot suppress a candidate.
            proposal["numerical_priority"] = witnesses is not None
            if witnesses is None:
                continue
            seen.add((selected, cut))
            yield (
                selected,
                cut,
                {"priority": "connected prepared calls", "reason": reason, "numerical_witnesses": witnesses},
            )
    except CompilationError as error:
        diagnostics["priority_search_decline"] = str(error)
    for selected, cut in original_candidates(owner.graph):
        if (selected, cut) not in seen:
            yield selected, cut, {"priority": "original complete search", "reason": None}
