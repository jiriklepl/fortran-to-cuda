"""Checked metadata-only lookahead over entered source control arms.

A query records cost alternatives; existing numerical and native effect proofs still authorize
every operation at its original execution point. No future read admission,
floating-point preparation, host publication, or device work occurs here.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, replace

from fparser.two.utils import walk

from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError
from compiler.ir.integers import INTEGER_MAX, INTEGER_MIN
from compiler.scopes.execution_plan import _configuration, _digest
from compiler.scopes.segments import Segment, fortran_lines, statement_span

# Signatures contain identities/primitive data only. Weak attachment must not
# retain every transitive application source graph across compiler requests.
_ISSUED = {}
_FORBIDDEN = {"optional", "pointer", "volatile", "asynchronous", "value"}


@dataclass(frozen=True)
class MetadataValue:
    kind: str
    value: object = None
    binding: object = None
    children: tuple = ()
    source: object = None


@dataclass(frozen=True)
class QueryCallStep:
    member: object
    child: object
    source_identity: str
    node: object
    span: tuple
    site: object
    mappings: tuple
    site_identity: str


@dataclass(frozen=True)
class QueryStep:
    member: object
    unit: object
    call: object
    mappings: tuple
    call_path: tuple
    # Native preparation stays at its original point. Metadata recipes only
    # snapshot INTEGER values needed by later query geometry.
    recipes: tuple = ()
    scalar_writes: tuple = ()

    @property
    def scope(self):
        return self.unit.scope if hasattr(self.unit, "scope") else self.unit[1]

    @property
    def nodes(self):
        return self.unit.nodes if hasattr(self.unit, "nodes") else self.unit[0]


def _binding_signature(binding):
    return (id(binding), binding.root, binding.signature(), tuple(sorted(binding.attributes)),
            tuple(str(value) for value in binding.lower_bounds))


def _value_signature(value, memo=None):
    if isinstance(value, MetadataValue):
        memo = {} if memo is None else memo
        if id(value) not in memo:
            # SSA substitutions form a DAG (x=x+x can share both operands).
            # Hash child identities once rather than expanding repeated paths.
            memo[id(value)] = ("INTEGER recipe", _digest((value.kind, value.value,
                _binding_signature(value.binding) if value.binding else None,
                tuple(_value_signature(child, memo) for child in value.children), id(value.source))))
        return memo[id(value)]
    return _binding_signature(value)


def _worker_signature(step):
    """Count existing issued physical workers, never reconstructed source loops."""
    if step.call is None:
        return None, None, ()
    from compiler.frontend.numerical_preparation import PreparedOriginalRegion

    builder, call = step.member.builder, step.call
    inline = builder.inline_for(call.procedure)
    prepared = inline.prepared_regions.get(call.procedure) if inline is not None else None
    if (type(prepared) is not PreparedOriginalRegion
            or inline.regions.get(call.procedure) is not call.region
            or getattr(builder.analysis, "_numerical_preparations", {}).get(prepared.identity) is not prepared):
        raise CompilationError("query worker count lost its registered original numerical preparation")
    return id(prepared), prepared.identity, tuple(id(unit) for unit in prepared.offload.units)


def _signature(proof):
    owner = proof.owner
    return (id(owner), id(proof.anchor), proof.identity, proof.configuration, proof.first, proof.last,
            id(owner.execution_plan), tuple((id(member), id(member.execution_plan),
                tuple(id(unit) for unit in member.units), tuple(id(unit) for unit in member.graph_native_tasks))
                for member in owner.members.values()),
            tuple((id(step.member), id(step.unit), id(step.call), _worker_signature(step),
                tuple((resource, _value_signature(value)) for resource, value in step.mappings),
                tuple((id(part.member), id(part.child), part.source_identity, id(part.node), part.span,
                    id(part.site), tuple(sorted((key, repr(value)) for key, value in part.site.items())),
                    part.mappings, getattr(part.member, "mapped_call_patches", {}).get(part.span))
                    for part in step.call_path),
                tuple((root, _value_signature(value)) for root, value in step.recipes), step.scalar_writes)
                for step in proof.steps),
            tuple((root, _binding_signature(binding)) for root, binding in proof.arrays),
            tuple(proof.table_reads), tuple(proof.metadata_inputs))


@dataclass(frozen=True)
class QueryRegionProof:
    owner: object
    anchor: object
    identity: str
    configuration: str
    first: int
    last: int
    steps: tuple
    arrays: tuple
    table_reads: tuple
    metadata_inputs: tuple

    def validate(self):
        issued = _ISSUED.get(id(self))
        if issued is None or issued[0]() is not self or _signature(self) != issued[1]:
            raise CompilationError("query region changed its issued bindings, order, mapping or child authority")
        if not any(proof is self for proof in self.owner.emission.query_regions):
            raise CompilationError("query region lacks exact issued attachment")
        if self.configuration != _configuration(self.owner.builder):
            raise CompilationError("query region configuration changed")
        self.owner.execution_plan.validate()
        for step in self.steps:
            if not any(member is step.member for member in self.owner.execution_plan.members):
                raise CompilationError("query region member lost checked execution authority")
            if (not any(unit is step.unit for unit in self.owner.execution_plan.units(step.member))
                    and not any(unit is step.unit for unit in step.member.graph_native_tasks)):
                raise CompilationError("query region lost its admitted original unit")
            if step.call is not None and not any(call is step.call for call in step.scope.calls):
                raise CompilationError("query region lost its exact original numerical call")
            for part in step.call_path:
                originals = part.member.graph.source_nodes(part.source_identity)
                if (len(originals) != 1 or originals[0] is not part.node
                        or self.owner.members.get(part.child.routine.qualified) is not part.child
                        or not any(site is part.site for site in part.child.call_sites)):
                    raise CompilationError("query region lost its exact mapped source call path")
        return self

    def public(self):
        return {"identity": self.identity, "procedure": self.anchor.routine.qualified,
                "span": [self.first, self.last], "workers": sum(len(_worker_signature(step)[2]) for step in self.steps),
                "source_calls": sum(step.call is not None for step in self.steps),
                "worker_count_semantics": "registered compiled workers; empty domains and source branches may record fewer",
                "source_steps": len(self.steps), "helpers": list(dict.fromkeys(
                    step.member.routine.qualified for step in self.steps)),
                "metadata_inputs": list(self.metadata_inputs), "metadata_table_reads": list(self.table_reads),
                "tasks": [{"procedure": step.member.routine.qualified,
                    "span": [step.scope.first, step.scope.last],
                    "kind": "numerical" if step.call is not None else "original native preparation",
                    "mappings": {resource: (value.binding.root if isinstance(value, MetadataValue) and value.binding
                        else "INTEGER shadow" if isinstance(value, MetadataValue) else value.root)
                        for resource, value in step.mappings},
                    "recipe_roots": [root for root, _value in step.recipes]} for step in self.steps],
                "entry_definitions": "reviewed invocation facts only before the first close",
                "floating_point_preparation": "original reached source; unpriced in graph estimate",
                "future_source_read_admission": False, "early_payload_transfers": False}


class _Discovery:
    def __init__(self, owner):
        self.owner = owner
        self.declines = []
        self.active = set()
        self.unit_maps = {}
        for member in owner.members.values():
            units, skipped = {}, set()
            for unit in member.units:
                graph, selected = member.builder.analysis._selected_source(member.routine.qualified,
                    member.builder.inline.original_selection(unit.nodes))
                if graph is not member.graph or not selected:
                    raise CompilationError("query region lost original selection")
                units[selected[0]] = unit
                skipped.update(selected[1:])
            for unit in member.graph_native_tasks:
                graph, selected = member.builder.analysis._selected_source(member.routine.qualified,
                    member.builder.inline.original_selection(unit[0]))
                if graph is not member.graph or not selected:
                    raise CompilationError("query retained native preparation lost original authority")
                units[selected[0]] = unit
                skipped.update(selected[1:])
            self.unit_maps[id(member)] = units, skipped

    def resolve(self, member, resource, environment):
        if resource in environment:
            return environment[resource]
        binding = member.builder.visible_binding(member.routine, resource)
        if binding.attributes & _FORBIDDEN:
            raise CompilationError("query metadata lacks stable ordinary storage: " + resource)
        # A helper-local variable may be defined later. Only an independently
        # reconstructed total INTEGER slice can supply such an ahead value.
        root_binding = self.owner.builder.visible_binding(self.owner.routine, binding.root)
        if root_binding is not binding:
            raise CompilationError("query metadata changed its exact visible binding")
        if binding.rank:
            fact = self.owner.builder.capture(binding)
            if fact.get("storage") != "stable" or fact.get("allocation_changes") is not False:
                raise CompilationError("query array lacks an invocation-stable descriptor: " + resource)
            if fact.get("source_sections") is not None or fact.get("initialized") not in {"whole", "none", "sections"}:
                raise CompilationError("query array needs reviewed existing anchor coverage: " + resource)
        return binding

    def expression(self, member, node, environment):
        kind = _kind(node)
        if kind == "Int_Literal_Constant":
            value = int(str(node).split("_")[0])
            if not INTEGER_MIN <= value <= INTEGER_MAX:
                raise CompilationError("query INTEGER literal exceeds original ABI")
            return MetadataValue("literal", value, source=node)
        if kind == "Name":
            scope = member.builder.analysis.source_scope_for(node, member.routine.scope)
            binding = member.builder.analysis._binding(scope, node)
            if binding is None or binding.rank or binding.dtype != "integer" or binding.kind != 4:
                raise CompilationError("query shadow requires default INTEGER inputs")
            if "parameter" in binding.attributes:
                value = scope.kinds.integer(str(node), None)
                return MetadataValue("literal", value, source=node)
            value = self.resolve(member, binding.root, environment)
            return value if isinstance(value, MetadataValue) else MetadataValue("scalar", binding=value, source=node)
        if kind == "Part_Ref":
            scope = member.builder.analysis.source_scope_for(node, member.routine.scope)
            binding = member.builder.analysis._binding(scope, node.items[0])
            if (binding is None or binding.dtype != "integer" or binding.kind != 4 or binding.rank != 1
                    or "protected" not in binding.attributes):
                raise CompilationError("query table snapshot requires a protected INTEGER vector")
            binding = self.resolve(member, binding.root, environment)
            indices = tuple(getattr(node.items[1], "items", ()))
            if len(indices) != 1:
                raise CompilationError("query metadata table requires one exact index")
            fact = self.owner.builder.capture(binding)
            if fact.get("initialized") != "whole":
                raise CompilationError("query table snapshot lacks existing whole input definitions")
            return MetadataValue("element", binding=binding,
                                 children=(self.expression(member, indices[0], environment),), source=node)
        items = tuple(getattr(node, "items", ()))
        if kind == "Parenthesis" and len(items) == 3:
            return self.expression(member, items[1], environment)
        if len(items) == 2 and items[0] in {"+", "-"}:
            return MetadataValue("unary", items[0], children=(self.expression(member, items[1], environment),), source=node)
        if len(items) == 3 and items[1] in {"+", "-", "*"}:
            return MetadataValue("binary", items[1], children=(self.expression(member, items[0], environment),
                self.expression(member, items[2], environment)), source=node)
        raise CompilationError("query INTEGER recipe excludes calls, division and indirect expressions")

    def scalar_preparation(self, member, node):
        """Recognize scalar arithmetic that will still execute when reached."""
        kind = _kind(node)
        if kind in {"Int_Literal_Constant", "Real_Literal_Constant"}:
            return True
        if kind == "Name":
            scope = member.builder.analysis.source_scope_for(node, member.routine.scope)
            binding = member.builder.analysis._binding(scope, node)
            return binding is not None and not binding.rank and not binding.attributes & _FORBIDDEN
        items = tuple(getattr(node, "items", ()))
        if kind == "Parenthesis" and len(items) == 3:
            return self.scalar_preparation(member, items[1])
        if len(items) == 2 and items[0] in {"+", "-"}:
            return self.scalar_preparation(member, items[1])
        return (len(items) == 3 and items[1] in {"+", "-", "*", "/", "**"}
                and self.scalar_preparation(member, items[0]) and self.scalar_preparation(member, items[2]))

    def native(self, member, unit, environment, path):
        scope = unit.scope if hasattr(unit, "scope") else unit[1]
        nodes = unit.nodes if hasattr(unit, "nodes") else unit[0]
        if (getattr(unit, "source_guard", None) is not None or getattr(unit, "publication_only", False)
                or getattr(unit, "host_metadata", None)
                or any(operation.preserve_original or operation.native_environment for operation in scope.native)):
            raise CompilationError("query stops at native control, lifetime or participation work")
        original = member.builder.inline.original_selection(nodes)
        recipes = []

        for node in original:
            if _kind(node) == "Comment":
                continue
            if _kind(node) != "Assignment_Stmt" or _kind(node.items[0]) != "Name":
                raise CompilationError("query native preparation must be original scalar assignments")
            scope = member.builder.analysis.source_scope_for(node, member.routine.scope)
            binding = member.builder.analysis._binding(scope, node.items[0])
            if (binding is None or binding.rank or binding.attributes & (_FORBIDDEN | {"save", "allocatable"})
                    or binding.name in member.routine.arguments or binding.declaring_scope is not member.routine.scope):
                raise CompilationError("query preparation may only update exact unsaved local scalars")
            if binding.dtype == "integer" and binding.kind == 4:
                value = self.expression(member, node.items[2], environment)
                environment[binding.root] = value
                recipes.append((binding.root, value))
            elif binding.dtype == "real":
                # Existing independent native effects prove what actually
                # executes. No FP value is copied into the ahead query.
                if not self.scalar_preparation(member, node.items[2]):
                    raise CompilationError("query stops at FP preparation with payload reads or calls")
                environment[binding.root] = None
            else:
                raise CompilationError("query native preparation has unsupported scalar storage")
        return QueryStep(member, unit, None, (), path, tuple(recipes))

    def numerical(self, member, unit, environment, path):
        from compiler.scopes.joined_team import JoinedTeam
        if isinstance(unit.scope, JoinedTeam):
            raise CompilationError("query stops at a conditional joined-team coordinator")
        if (len(unit.scope.calls) != 1 or len(unit.scope.tree) != 1
                or not isinstance(unit.scope.tree[0], Segment)
                or unit.source_guard is not None or unit.publication_only or unit.host_metadata
                or unit.definition_summary.changes
                or any(op.kind != "guarded original numerical fallback" for op in unit.scope.native)):
            raise CompilationError("query requires one exact original numerical operation")
        call, = unit.scope.calls
        if call.region is None or call.region.scalar_element_captures:
            raise CompilationError("query excludes reached scalar payload captures")
        public, _ = member.builder.entry_artifacts(call.procedure)
        planning = public["planning"]
        if not planning["query_available"] or planning["payload_arrays"]:
            raise CompilationError("query requires descriptor/INTEGER-only planning inputs")
        parameters = {p.name.lower(): p for p in call.region.parameters}
        needed = {p.resource for p in call.region.parameters if p.rank}
        needed.update(parameters[name.lower()].resource for name in planning["scalar_inputs"]
                      if parameters[name.lower()].lower_bound_dimension is None)
        mappings = []
        for resource in sorted(needed):
            value = self.resolve(member, resource, environment)
            if value is None:
                raise CompilationError("query requires an unavailable FP preparation value for geometry")
            if not isinstance(value, MetadataValue) and not value.rank and value.dtype not in {"integer", "logical"}:
                raise CompilationError("query cannot anticipate floating-point geometry")
            mappings.append((resource, value))
        for parameter in parameters.values():
            if parameter.lower_bound_dimension is not None:
                binding = member.builder.visible_binding(member.routine, parameter.resource)
                if not parameter.resource.startswith("argument::"):
                    # A module allocatable's declaration has no current
                    # allocation lower bound. Its guarded original descriptor
                    # supplies that value at this query point.
                    continue
                try:
                    int(str(binding.lower_bounds[parameter.lower_bound_dimension - 1]).replace(" ", ""))
                except ValueError:
                    raise CompilationError("query requires literal original dummy lower bounds") from None
        scalar_writes = set()
        original = member.builder.inline.original_selection(unit.nodes)
        for node in walk(original):
            target = node.items[0] if _kind(node) == "Assignment_Stmt" else None
            if _kind(node) == "Loop_Control" and node.items[1]:
                target = node.items[1][0]
            if _kind(target) != "Name":
                continue
            lexical = member.builder.analysis.source_scope_for(target, member.routine.scope)
            binding = member.builder.analysis._binding(lexical, target)
            if binding is not None and not binding.rank:
                value = environment.get(binding.root, binding)
                if value is not None and not isinstance(value, MetadataValue):
                    scalar_writes.add(value.root)
        return QueryStep(member, unit, call, tuple(mappings), path, scalar_writes=tuple(sorted(scalar_writes)))

    def helper(self, member, identity, environment, path):
        originals = member.graph.source_nodes(identity)
        if len(originals) != 1 or _kind(originals[0]) != "Call_Stmt":
            raise CompilationError("query stops at original control or unsupported native work")
        node = originals[0]
        resolved = member.builder.execution_source_call(member.routine, node)
        child = self.owner.members.get(resolved.procedure)
        span = statement_span(node)
        site = next((site for site in getattr(child, "call_sites", ()) if site["caller"] == member.routine.qualified
                     and (site["first_line"], site["last_line"]) == span), None) if child else None
        if child is None or site is None:
            raise CompilationError("query helper lacks exact admitted original call authority")
        if id(child) in self.active:
            raise CompilationError("query helper recursion is unsupported")
        child_environment = {}
        for mapping in resolved.mappings:
            if mapping.binding is not None and mapping.section is None and mapping.scalar_element is None and _kind(mapping.actual) == "Name":
                child_environment["argument::" + mapping.formal] = self.resolve(member, mapping.binding.root, environment)
            elif mapping.actual is not None and _kind(mapping.actual) == "Int_Literal_Constant":
                child_environment["argument::" + mapping.formal] = self.expression(member, mapping.actual, environment)
            elif (mapping.actual is not None and not mapping.formal_binding.rank
                    and (self.scalar_preparation(member, mapping.actual)
                         or _kind(mapping.actual) == "Logical_Literal_Constant")):
                # Unused data-arithmetic literals are never evaluated early.
                # If queried as geometry, resolve() explicitly rejects None.
                child_environment["argument::" + mapping.formal] = None
            else:
                raise CompilationError("query cannot anticipate original expression/section association")
        part = QueryCallStep(member, child, identity, node, span, site,
            tuple((mapping.formal, id(mapping.binding), mapping.binding.root if mapping.binding else None,
                   id(mapping.actual), str(mapping.actual)) for mapping in resolved.mappings), _digest(site))
        self.active.add(id(child))
        try:
            steps = self.complete(child, child.graph.root_id, child_environment, (*path, part))
        finally:
            self.active.remove(id(child))
        return steps

    def complete(self, member, identity, environment, path):
        units, skipped = self.unit_maps[id(member)]
        source = member.graph.nodes[identity]
        if identity in skipped or source.kind == "entry":
            return []
        if identity in units:
            unit = units[identity]
            return [self.numerical(member, unit, environment, path) if getattr(unit, "leaves", ())
                    else self.native(member, unit, environment, path)]
        if source.kind == "sequence":
            return [step for child in source.children for step in self.complete(member, child, environment, path)]
        return self.helper(member, identity, environment, path)

    def entered(self, member, identity):
        """Find maximal straight lines only after their original controls ran."""
        source = member.graph.nodes[identity]
        units, skipped = self.unit_maps[id(member)]
        if identity in skipped:
            return
        if source.kind == "sequence":
            pending, first, last, environment = [], None, None, {}
            for child in source.children:
                node = member.graph.nodes[child]
                before = dict(environment)
                try:
                    steps = self.complete(member, child, environment, ())
                except CompilationError as error:
                    if pending:
                        yield member, first, last, tuple(pending)
                    pending, first, last, environment = [], None, None, {}
                    self.declines.append({"procedure": member.routine.qualified, "span": list(node.span), "reason": str(error)})
                    if child not in units:
                        yield from self.entered(member, child)
                else:
                    if steps:
                        selected = units.get(child)
                        scope = (selected.scope if hasattr(selected, "scope") else selected[1]) if selected is not None else None
                        span = (scope.first, scope.last) if scope is not None else node.span
                        if last is not None and span[0] <= last:
                            raise CompilationError("query source tasks share an original physical statement line")
                        pending += steps
                        first = span[0] if first is None else first
                        last = span[-1]
                    else:
                        environment = before
            if pending:
                yield member, first, last, tuple(pending)
        elif source.kind == "branch":
            originals = member.graph.source_nodes(identity)
            if len(originals) == 1 and _kind(originals[0]) == "If_Construct":
                for _condition, body in source.alternatives:
                    yield from self.entered(member, body)
        elif source.kind == "loop":
            originals = member.graph.source_nodes(identity)
            if len(originals) == 1 and _kind(originals[0]) == "Block_Nonlabel_Do_Construct":
                # Only the body sequence is entered; never its header or a
                # source-attached OpenMP group lacking an admitted whole unit.
                for child in source.children:
                    if member.graph.nodes[child].kind == "sequence":
                        yield from self.entered(member, child)


def _values(value, seen=None):
    if isinstance(value, MetadataValue):
        seen = set() if seen is None else seen
        if id(value) in seen:
            return
        seen.add(id(value))
        if value.binding is not None:
            yield value.kind, value.binding
        for child in value.children:
            yield from _values(child, seen)
    else:
        yield "array" if value.rank else "scalar", value


def discover_query_regions(owner):
    """Collect draft source slices before declaration and plan sealing."""
    owner.emission.query_drafts = ()
    owner.emission.query_regions = ()
    owner.emission.query_declines = []
    owner.emission.query_resources = ()
    if owner.parent is not None or owner.builder.config.policy != "auto":
        return
    discovery = _Discovery(owner)
    drafts, resources = [], {}
    for anchor, first, last, steps in discovery.entered(owner, owner.graph.root_id):
        arrays, tables, scalars = {}, set(), set()
        try:
            if sum(len(_worker_signature(step)[2]) for step in steps) < 2:
                continue
            for step in steps:
                for value in (*[value for _root, value in step.mappings], *[value for _root, value in step.recipes]):
                    for kind, binding in _values(value):
                        if binding.rank:
                            arrays[binding.root] = binding
                            if kind == "element":
                                tables.add(binding.root)
                        else:
                            scalars.add(binding.root)
            if any(set(step.scalar_writes) & scalars for step in steps):
                raise CompilationError("query geometry input is modified by an original numerical task")
            # An ahead payload snapshot requires its exact storage unchanged
            # across this whole query region, including mapped native helpers.
            for step in steps:
                for binding in getattr(step.unit, "arrays", {}).values():
                    canonical = next((value for root, value in step.mappings if root == binding.root),
                                     step.member.canonical(binding))
                    if not isinstance(canonical, MetadataValue) and canonical.root in tables and binding.root in step.unit.written:
                        raise CompilationError("query metadata table is written inside this source region")
            for root, binding in arrays.items():
                owner.add_resource(binding)
                resources[root] = binding
            drafts.append((anchor, first, last, steps, tuple(sorted(arrays.items())), tuple(sorted(tables)), tuple(sorted(scalars))))
        except CompilationError as error:
            discovery.declines.append({"procedure": anchor.routine.qualified, "span": [first, last], "reason": str(error)})
    owner.emission.query_drafts = tuple(drafts)
    owner.emission.query_resources = tuple(resources.values())
    owner.emission.query_declines = discovery.declines


def seal_query_regions(owner, motions):
    proofs = []
    config = _configuration(owner.builder)
    for anchor, first, last, steps, arrays, tables, scalars in owner.emission.query_drafts:
        # The checked execution issuer freezes owned lists/dicts into fresh
        # immutable containers. Reattach only equivalent exact objects from
        # that sealed inventory; the independently retained draft digest must
        # still match the original source call site.
        sealed = []
        for step in steps:
            path = []
            for part in step.call_path:
                site = next((site for site in part.child.call_sites if site["caller"] == part.member.routine.qualified
                    and (site["first_line"], site["last_line"]) == part.span), None)
                if site is None or _digest(site) != part.site_identity:
                    raise CompilationError("query source call changed before checked plan publication")
                path.append(replace(part, site=site))
            unit = step.unit
            if not hasattr(unit, "scope"):
                unit = next((item for item in step.member.graph_native_tasks if item[1] is step.scope
                    and tuple(id(node) for node in item[0]) == tuple(id(node) for node in step.nodes)), None)
                if unit is None:
                    raise CompilationError("query retained native preparation changed before publication")
            sealed.append(replace(step, unit=unit, call_path=tuple(path)))
        steps = tuple(sealed)
        if any(motion["procedure"] == anchor.routine.qualified and first <= motion["last_line"]
               and motion["first_line"] <= last for motion in motions):
            owner.emission.query_declines.append({"procedure": anchor.routine.qualified, "span": [first, last],
                "reason": "query source anchors overlap checked emitted task motion"})
            continue
        identity = _digest({"source": anchor.graph.identity, "span": [first, last], "config": config,
            "steps": [(step.member.routine.qualified, step.scope.first, step.scope.last,
                       step.call.procedure if step.call else None) for step in steps]})
        proof = QueryRegionProof(owner, anchor, identity, config, first, last, steps, arrays, tables, scalars)
        proofs.append(proof)
        key = id(proof)
        _ISSUED[key] = weakref.ref(proof, lambda _value, key=key: _ISSUED.pop(key, None)), _signature(proof)
    owner.emission.query_regions = tuple(proofs)
    return proofs


def emit_query_regions(owner, record, motions):
    """Create one metadata-only query and replay choices at reached workers."""
    proofs = seal_query_regions(owner, motions)
    record["query_regions"] = [proof.public() for proof in proofs]
    record["query_region_declines"] = owner.emission.query_declines
    for ordinal, proof in enumerate(proofs):
        _emit_region(proof, ordinal)


def _emit_region(proof, ordinal):
    proof.validate()
    owner, anchor = proof.owner, proof.anchor
    builder = owner.builder
    tag = proof.identity[:12]
    block, own = "fort_query_" + tag, "fort_query_owned_" + tag
    imports = ["use iso_c_binding", "use fort_scoped_memory"]
    spec = ["integer(c_int) :: fort_status, fort_live", "logical :: fort_can", "type(c_ptr) :: fort_pointer",
            "type(fort_scope_layout) :: fort_layout", "type(fort_scope_plan_decision) :: fort_decision"]
    code = ["fort_status = FORT_SCOPE_OK", "fort_can = .true.",
            f"if (.not. {anchor.epoch_pristine}) exit {block}"]
    leaves = {step.call.procedure for step in proof.steps if step.call is not None}
    configuration = builder.owner_transfer_setup(leaves, imports, context=anchor.context)
    from compiler.scopes.source import DTYPES, _fortran_list
    for root, binding in proof.arrays:
        name = builder.visible(anchor.routine, root)
        if "allocatable" in binding.attributes:
            code += [f"if (fort_can) fort_can = allocated({name})"]
        code += [f"if (fort_can) fort_can = is_contiguous({name})"]
        code += [f"if (fort_can) fort_can = {condition}"
                 for condition in builder.original_bound_conditions(name, binding.rank)]
        from compiler.scopes.registration import expected_descriptor_conditions
        code += [f"if (fort_can) fort_can = {condition}" for condition in
                 expected_descriptor_conditions(builder.capture(binding), binding.rank, name)]
    code += [f"if (.not. fort_can) exit {block}", f"if ({anchor.context} == 0) then",
        f"fort_status = fort_scope_create(0_c_int, {anchor.context})",
        f"if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_set_device_budget({anchor.context}, &",
        f"{builder.device_budget}_c_size_t)", *configuration, "endif"]
    for index, (root, binding) in enumerate(proof.arrays):
        name = builder.visible(anchor.routine, root)
        handle = "fort_buffer_" + str(owner.resources[root][1])
        extent, lower = f"fort_query_extents_{index}", f"fort_query_lowers_{index}"
        spec += [f"integer(c_size_t), target :: {extent}({binding.rank})",
                 f"integer(c_int64_t), target :: {lower}({binding.rank})"]
        _, enum, width = DTYPES[binding.signature()[:2]]
        code += ["fort_live = 0", "if (fort_status == FORT_SCOPE_OK) &",
            f"fort_status = fort_scope_buffer_valid_v1({anchor.context}, {handle}, fort_live)",
            f"if (fort_status == FORT_SCOPE_OK .and. fort_live == 0) {handle} = 0",
            f"if (fort_status == FORT_SCOPE_OK .and. {handle} == 0) then",
            f"{extent} = shape({name},kind=c_size_t)", f"{lower} = lbound({name},kind=c_int64_t)",
            "fort_pointer = c_null_ptr", f"if (all({extent} > 0)) fort_pointer = c_loc({name})",
            f"fort_layout = fort_scope_layout({binding.rank}, {enum}, {width}_c_size_t, &",
            f"fort_pointer, c_loc({extent}), c_loc({lower}), 1_c_int64_t)"]
        from compiler.scopes.registration import build_registration
        registration = build_registration(builder.capture(binding), binding.rank, handle,
            f"fort_query_defined_{index}", f"{owner.resources[root][1]}_c_int64_t", "fort_layout",
            context=anchor.context, on_error=(f"exit {block}",), entry_guard=anchor.epoch_pristine)
        spec += registration.specification
        code += [*registration.prepare, "endif"]
    code += [f"if (fort_status /= FORT_SCOPE_OK) exit {block}"]
    for root in proof.table_reads:
        handle = "fort_buffer_" + str(owner.resources[root][1])
        code += [f"if (fort_scope_plan_host_current({anchor.context}, {handle}) /= FORT_SCOPE_OK) exit {block}"]
    temporaries = {}

    def expression(value):
        if not isinstance(value, MetadataValue):
            return builder.visible(anchor.routine, value.root)
        key = _value_signature(value)
        if key in temporaries:
            return f"int({temporaries[key]},c_int)"
        if value.kind == "literal":
            return str(value.value) + "_c_int"
        if value.kind == "scalar":
            return builder.visible(anchor.routine, value.binding.root)
        args = [expression(child) for child in value.children]
        name = "fort_query_integer_" + str(len(temporaries))
        temporaries[key] = name
        spec.append(f"integer(c_int64_t) :: {name}")
        if value.kind == "element":
            table = builder.visible(anchor.routine, value.binding.root)
            code.extend([f"if (int({args[0]},c_int64_t) < lbound({table},1,kind=c_int64_t)) exit {block}",
                f"if (int({args[0]},c_int64_t) > ubound({table},1,kind=c_int64_t)) exit {block}",
                f"{name} = int({table}({args[0]}),c_int64_t)"])
        elif value.kind == "unary":
            code.append(f"{name} = {value.value}(int({args[0]},c_int64_t))")
        elif value.kind == "binary":
            code.append(f"{name} = int({args[0]},c_int64_t) {value.value} int({args[1]},c_int64_t)")
        else:
            raise CompilationError("unknown issued query metadata recipe")
        code.extend([f"if ({name} < ({INTEGER_MIN}_c_int64_t)) exit {block}",
                     f"if ({name} > {INTEGER_MAX}_c_int64_t) exit {block}"])
        return f"int({name},c_int)"

    # Compute every needed shadow before opening a planning transaction. A
    # failed shadow never leaves an unfinished schedule or admits future data.
    prepared = {}
    for step in proof.steps:
        if step.call is not None:
            for resource, value in step.mappings:
                if isinstance(value, MetadataValue) or not value.rank:
                    prepared[id(step), resource] = expression(value)
    code += [f"fort_status = fort_scope_plan_reset_mode({anchor.context}, FORT_SCOPE_PLAN_CONTINUE)"]
    for index, step in enumerate(proof.steps):
        if step.call is None:
            # Local scalar preparation has no managed writes. Its protected
            # table reads are already current; original hooks execute later.
            code += ["if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_add( &",
                f"{anchor.context}, FORT_SCOPE_PLAN_NATIVE, 0_c_int64_t, c_null_ptr, 0_c_size_t, &",
                "0.0_c_double, 0.0_c_double, 0_c_int)"]
            continue
        call = step.call
        public, _ = step.member.builder.entry_artifacts(call.procedure)
        alias = "fort_region_query_" + str(index)
        imports.append(f"use {public['fortran_module']}, only: {alias} => {public['planning']['fortran_procedure']}")
        parameters = {parameter.name.lower(): parameter for parameter in call.region.parameters}
        mappings = dict(step.mappings)
        args = [anchor.context]
        for array in public["array_parameters"]:
            parameter = parameters[array["name"].lower()]
            args.append("fort_buffer_" + str(owner.resources[mappings[parameter.resource].root][1]))
        for scalar in public["scalar_parameters"]:
            if scalar["name"] not in public["planning"]["scalar_inputs"]:
                continue
            parameter = parameters[scalar["name"].lower()]
            if parameter.lower_bound_dimension is not None:
                binding = step.member.builder.visible_binding(step.member.routine, parameter.resource)
                if parameter.resource.startswith("argument::"):
                    args.append(str(int(str(binding.lower_bounds[parameter.lower_bound_dimension - 1]).replace(" ", ""))) + "_c_int")
                else:
                    name = builder.visible(anchor.routine, mappings[parameter.resource].root)
                    args.append(f"int(lbound({name},{parameter.lower_bound_dimension},kind=c_int64_t),c_int)")
            else:
                args.append(prepared[id(step), parameter.resource])
        code += ["if (fort_status == FORT_SCOPE_OK) then",
                 *_fortran_list("fort_status = " + alias + "(", args, ")", 0), "endif"]
    from compiler.scopes.compute_identity import compute_identity_guard
    guard = compute_identity_guard(builder, sorted(leaves), caller=anchor.routine)
    imports += list(guard.imports)
    selector_step = next(step for step in proof.steps if step.call is not None)
    selector, _ = selector_step.member.builder.entry_artifacts(selector_step.call.procedure)
    imports.append(f"use {selector['fortran_module']}, only: fort_region_choose => {selector['planning']['fortran_selector']}")
    code += [f"if (.not. ({guard.expression})) fort_status = FORT_SCOPE_BOUNDARY",
        f"if (fort_status == FORT_SCOPE_OK) fort_status = fort_scope_plan_validate({anchor.context})",
        f"if (fort_status == FORT_SCOPE_OK) fort_status = fort_region_choose({anchor.context}, fort_decision)",
        f"if (fort_status /= FORT_SCOPE_OK) exit {block}", f"{anchor.query_plan} = .true.", f"{own} = .true."]
    activation = [f"if ({anchor.enabled}) then", f"if (.not. {anchor.query_plan}) then",
        block + ": block", *dict.fromkeys(imports), *spec, *code, "end block " + block, "endif", "endif"]
    if anchor.control_guard:
        activation = [f"if ({anchor.control_guard}) then", *activation, "endif"]
    prefix = ["block", f"logical :: {own}", f"{own} = .false.", *activation]
    suffix = [f"if ({own}) {anchor.query_plan} = .false.", "end block"]

    def insert_boundary(step, first, last, lines, *, after):
        path = anchor.routine.scope.path
        text = "\n".join(fortran_lines(lines)) + "\n"
        retained = ()
        if step.member is anchor and any(unit is step.unit for unit in anchor.units):
            emitted = getattr(anchor.unit_emission(step.unit), "source_edits", None)
            if emitted is None:
                raise CompilationError("query boundary lost its emitted original unit edits")
            retained = tuple(edit for edit in emitted if edit[:2] == (first, last))
        if retained:
            edits = builder.edits.get(path, [])
            matches = [index for index, edit in enumerate(edits)
                       if edit[:2] == (first, last)
                       and any(edit[2] is source[2] for source in retained)]
            if len(matches) != len(retained):
                raise CompilationError("query boundary changed its exact emitted unit insertion")
            position = max(matches) + 1 if after else min(matches)
            edits.insert(position, (first, last, text))
        else:
            # A replaced source CALL has no task insertion at this boundary.
            # Keep its root prologue outside, and its owner cleanup afterward.
            builder.add_edit(path, first, last, text, prepend=after)

    insert_boundary(proof.steps[0], proof.first, proof.first - 1, prefix, after=False)
    insert_boundary(proof.steps[-1], proof.last + 1, proof.last, suffix, after=True)
