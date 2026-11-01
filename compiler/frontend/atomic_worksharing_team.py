"""One conditional original worksharing team with completion only at its join.

This is source participation and private-state authority.  It never converts a
NOWAIT DO into an independently completed unit, proves numerical IR, evaluates
a predicate, or supplies storage/definition/alias authority.  A consumer must
prepare the entire reached alternative before attempting any device work.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.component_bindings import references
from compiler.frontend.native_completion import _children, _directive, _joined_completion_facts, _kind
from compiler.frontend.native_group_structure import original_sequence
from compiler.ir import CompilationError

_ISSUER = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fail(message):
    raise CompilationError("atomic original team: " + message)


@dataclass(frozen=True, slots=True)
class AtomicWorksharingUnit:
    opening: object
    loop: object
    ending: object
    reads: tuple
    writes: tuple
    unit_kind: str = "worksharing_do"
    group_opening: object = None


def _derive(analysis, routine, nodes):
    graph, selected = analysis._selected_source(routine.qualified, nodes)
    if analysis.routines.get(routine.qualified) is not routine:
        _fail("requires its exact original routine")
    parent = analysis.joined_completion(routine.qualified, nodes)
    inventory, uniform_sites, section_inventory = [], [], []
    section_team = parent.public().get("native_sections_contract") == "original-explicit-counted-sections-v1"
    if section_team:
        _joined_completion_facts(analysis, routine.qualified, nodes, section_units=section_inventory,
                                 uniform_sites=uniform_sites)
        directives = [_directive(node) for node in walk(nodes)]
        if any(text and (text == "do" or text.startswith("do ")) for text in directives):
            _fail("conditional SECTION first slice does not mix original worksharing DO units")
        if parent.public().get("native_environment_operations") or parent.public().get("original_terminal_calls"):
            _fail("conditional SECTION first slice has other native-only participation effects")
        inventory = [(opening, loop, ending) for opening, loop, ending, _group in section_inventory]
    else:
        _joined_completion_facts(analysis, routine.qualified, nodes, worksharing=inventory, uniform_sites=uniform_sites)
    if not inventory or (parent.public().get("native_only") and not section_team):
        _fail("requires one complete loop-only original PARALLEL team")
    # The parent source scanner admits calls only through separate native
    # proofs.  That does not authorize numerical substitution of those calls.
    for node in walk(nodes):
        if _kind(node) == "Call_Stmt":
            _fail("does not substitute original calls or environment operations")
    roots, iterators, units, bindings = {}, set(), [], {}
    for opening, loop, ending in inventory:
        reads, writes = {}, {}
        for node in walk(loop):
            kind = _kind(node)
            scope = analysis.source_scope_for(node, routine.scope)
            if (kind.endswith("_Stmt") and kind not in {
                    "Nonlabel_Do_Stmt", "End_Do_Stmt", "Assignment_Stmt", "If_Stmt", "If_Then_Stmt",
                    "Else_If_Stmt", "Else_Stmt", "End_If_Stmt", "Continue_Stmt"}):
                _fail("requires loop-only original arithmetic without lifetime, I/O or exit statements")
            if getattr(getattr(node, "item", None), "label", None) is not None:
                _fail("does not duplicate original procedure-scope labels")
            if kind == "Loop_Control":
                if node.items[1] is None:
                    _fail("requires original counted loop control")
                binding = analysis._binding(scope, node.items[1][0])
                if binding is None:
                    _fail("has an unresolved original induction variable")
                iterators.add(binding.root)
            elif kind == "Assignment_Stmt":
                target = node.items[0]
                if _kind(target) != "Part_Ref" or _kind(target.items[0]) != "Name":
                    _fail("requires point writes to original array storage; no scalar carries")
                binding = analysis._binding(scope, target.items[0])
                if binding is None or not binding.rank:
                    _fail("has an unresolved original array write")
                writes[binding.root] = binding
                for value in (node.items[2], target.items[1]):
                    for source, _reference in references(analysis, scope, value):
                        if source.rank:
                            reads[source.root] = source
            elif kind in {"Function_Reference", "Structure_Constructor", "Data_Ref"}:
                _fail("requires simple original numeric storage and intrinsic-only loop expressions")
        for binding, _reference in references(analysis, routine.scope, loop):
            boundary = analysis.resource_identity_boundary(binding)
            if boundary:
                _fail(boundary)
            bindings[binding.root] = binding
            if binding.rank:
                if (binding.dtype not in {"real", "integer"} or binding.kind not in {4, 8}
                        or binding.attributes & {"pointer", "optional", "volatile", "asynchronous", "value"}):
                    _fail("requires stable ordinary original numeric array associations")
                roots[binding.root] = binding
                # Conservatively include condition/header/subscript reads as
                # well as assignment RHS reads.  A write target may therefore
                # also appear here; this is dependency rejection only, never
                # a physical upload or initialized-input requirement.
                reads[binding.root] = binding
        # Numerical lowering still checks all index, arithmetic, dependence,
        # IEEE and intrinsic semantics.  Only original array effects are used
        # for the conservative cross-unit check below.
        group = next((group for marker, original, _ending, group in section_inventory
                      if marker is opening and original is loop), None)
        units.append(AtomicWorksharingUnit(opening, loop, ending,
                                          tuple(sorted(reads)), tuple(sorted(writes)),
                                          "section" if section_team else "worksharing_do", group))
    if set(parent.private_roots) - iterators:
        _fail("private state across units is unsupported; only original induction variables are allowed")
    by_loop = {id(unit.loop): index for index, unit in enumerate(units)}
    peeled, predicates, predicate_points = {}, [], []
    point_keys = ("resource", "source_reference", "subscript", "source_subscript", "parameter_resource",
                  "original_lower_bound", "original_upper_bound")

    def predicate(header):
        expression = header.items[0]
        scope = analysis.source_scope_for(header, routine.scope)
        sites = [facts for original, facts in uniform_sites if original is header]
        if len(sites) != 1:
            _fail("requires the exact parent uniform-condition site")
        points = []
        for node in walk(expression):
            if _kind(node) == "Part_Ref":
                binding = analysis._binding(scope, node.items[0])
                matches = [item for item in sites[0]
                           if binding is not None and item["resource"] == binding.root
                           and item["source_reference"] == str(node)]
                if (binding is None or (binding.dtype, binding.kind) != ("integer", 4)
                        or not matches or any(item.get("storage") != "fixed_explicit_shape" for item in matches)):
                    _fail("requires a parent-proved fixed INTEGER4 predicate point")
                # The original joined proof owns bounds, shared/unwritten state
                # and parameter resolution. Reuse its exact point inventory;
                # the runtime still checks initialization and physical aliases.
                point = {key: matches[0][key] for key in point_keys}
                if point not in points:
                    points.append(point)
                bindings[binding.root] = binding
                continue
            if _kind(node) in {"Data_Ref", "Intrinsic_Function_Reference",
                               "Function_Reference", "Structure_Constructor", "Real_Literal_Constant"}:
                _fail("requires original INTEGER/LOGICAL scalar predicates without calls or payload reads")
        inputs = []
        for binding, _reference in references(analysis, scope, expression):
            if binding.rank and any(item["resource"] == binding.root for item in points):
                continue
            if (binding.rank or (binding.dtype, binding.kind) not in {("integer", 4), ("logical", 1), ("logical", 4)}
                    or binding.root in set(parent.private_roots)
                    or binding.attributes & {"pointer", "optional", "target", "volatile", "asynchronous", "value"}):
                _fail("requires unchanged shared original scalar predicates")
            boundary = analysis.resource_identity_boundary(binding)
            if boundary:
                _fail(boundary)
            bindings[binding.root] = binding
            inputs.append(binding.root)
        index = len(predicates)
        predicates.append((header, tuple(sorted(inputs))))
        predicate_points.append(points)
        return index

    def sequence(items):
        result = []
        for node in original_sequence(items, peeled):
            kind = _kind(node)
            if kind == "Comment":
                continue  # Exact whole parent proof already checked directives.
            if id(node) in by_loop:
                result.append({"unit": by_loop[id(node)]})
            elif kind == "If_Construct":
                branches, body, condition = [], [], None
                for child in peeled.get(id(node), _children(node)):
                    label = _kind(child)
                    if label in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                        if condition is not None or body:
                            branches.append({"predicate": condition, "body": sequence(body)})
                        body = []
                        condition = predicate(child) if label in {"If_Then_Stmt", "Else_If_Stmt"} else None
                    else:
                        body.append(child)
                if not branches or branches[0]["predicate"] is None:
                    _fail("lost its exact original conditional branch")
                if branches[-1]["predicate"] is not None:
                    branches.append({"predicate": None, "body": []})
                result.append({"branches": branches})
            else:
                _fail("requires only complete original DO units and uniform IF constructs")
        return result

    tree = sequence(nodes)
    # Enumerate source alternatives, not runtime work units or speculative
    # predicate values.  Every conditional node is retained in the tree.
    def paths(items):
        alternatives = [()]
        for item in items:
            if "unit" in item:
                choices = [(item["unit"],)]
            else:
                choices = [path for branch in item["branches"] for path in paths(branch["body"])]
            if len(alternatives) * len(choices) > analysis.operation_limit:
                _fail("conditional source-alternative proof budget exhausted")
            alternatives = [(*prefix, *suffix) for prefix in alternatives for suffix in choices]
        return alternatives

    alternatives = tuple(paths(tree))
    for path in alternatives:
        for position, index in enumerate(path):
            unit = units[index]
            for other_index in path[:position]:
                other = units[other_index]
                if not section_team and (set(unit.writes) & set((*other.reads, *other.writes))
                        or set(other.writes) & set(unit.reads)):
                    _fail("cross-unit original array dependencies require a separate grouped dependence proof")
    unit_ids = [graph.node_id(unit.loop) for unit in units]
    if len(uniform_sites) != len(predicates) or any(
            site is not header for (site, _facts), (header, _inputs) in zip(uniform_sites, predicates, strict=True)):
        _fail("conditional tree differs from the parent uniform-site order")
    record = {"version": 3 if section_team else 2, "kind": "atomic_original_conditional_joined_team", "procedure": routine.qualified,
              "source_graph": graph.identity, "source_inputs": analysis.inputs.identity(),
              "selected_node_ids": list(selected), "parent_completion": parent.public(),
              "units": [{"source_node_id": identity, "reads": list(unit.reads), "writes": list(unit.writes),
                         "nowait": _directive(unit.ending) in {"end do nowait", "end sections nowait"},
                         **({"unit_kind": "section", "section_directive_node_id": graph.node_id(unit.opening),
                             "section_group_node_id": graph.node_id(unit.group_opening)} if section_team else {})}
                        for identity, unit in zip(unit_ids, units, strict=True)],
              "predicates": [{"source_node_id": graph.node_id(header, "condition"), "original_source": str(header),
                              "scalar_resources": list(inputs), "host_point_reads": points}
                             for (header, inputs), points in zip(predicates, predicate_points, strict=True)],
              "predicate_read_sites": [{"source_node_id": graph.node_id(header, "condition"),
                  "original_source": str(header), "expression": str(header.items[0]),
                  "host_point_reads": [{key: point[key] for key in point_keys} for point in facts]}
                  for header, facts in uniform_sites],
              "tree": tree, "alternatives": [list(path) for path in alternatives],
              "private_resources": list(parent.private_roots),
              "analysis_limits": [analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit],
              "bindings": [binding.public() for binding in (bindings[root] for root in sorted(bindings))],
              "physical_alias_requirement": "every distinct writable/read root must be checked disjoint at reached preparation",
              "completion": "whole parent PARALLEL join only; no independent NOWAIT unit completion",
              "preparation": "reached original predicates admit only missing exact read regions before host acquisition; existing device replicas retained; no writes or whole-entry definitions; entire reached alternative before first device submission",
              "fallback": "complete original team once before submission; fatal after any attempt; no individual native replay",
              "execution_authorized": False, "gpu_legality_established": False,
              "definitions_authorized": False, "capture_authorized": False,
              "alias_authorized": False, "motion_authorized": False}
    if section_team:
        record["conditional_affine_context_required"] = True
    return parent, tuple(units), tuple(predicates), tuple(bindings[root] for root in sorted(bindings)), record


@dataclass(frozen=True, slots=True)
class AtomicWorksharingTeamProof:
    identity: str
    parent: object = field(repr=False, compare=False)
    units: tuple = field(repr=False, compare=False)
    predicates: tuple = field(repr=False, compare=False)
    bindings: tuple = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _nodes: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    @property
    def fields(self):
        writes = {root for unit in self.units for root in unit.writes}
        return tuple((binding, binding.root in writes) for binding in self.bindings if binding.rank)

    @property
    def scalars(self):
        return tuple(binding for binding in self.bindings
                     if not binding.rank and binding.root not in self.parent.private_roots)

    def predicate_sections(self, index):
        """Exact host point geometry consumed only after this team's validation."""
        from compiler.frontend.native_sections import (
            NativeAxis,
            NativeBound,
            NativeRectangle,
            NativeResourceSections,
            NativeSections,
        )

        self.validate()
        if type(index) is not int or not 0 <= index < len(self.predicates):
            _fail("predicate index is outside the exact original tree")
        header = self.predicates[index][0]
        resources = {}
        for point in self.public()["predicates"][index]["host_point_reads"]:
            references = [node for node in walk(header.items[0])
                          if _kind(node) == "Part_Ref" and str(node) == point["source_reference"]]
            if not references:
                _fail("predicate point lost its exact original reference")
            coordinate = NativeBound("literal", value=point["subscript"])
            rectangle = NativeRectangle((NativeAxis(coordinate, coordinate, point=True),), references[0])
            root = point["resource"]
            resources.setdefault(root, (point["original_lower_bound"], []))[1].append(rectangle)
        return NativeSections(True, resources=tuple(
            NativeResourceSections(root, 1, (lower,), tuple(boxes), (), ())
            for root, (lower, boxes) in sorted(resources.items())))

    @property
    def team(self):
        """Common graph-resource interface; this token itself is the team."""
        return self

    def validate(self, analysis=None, routine=None, nodes=None):
        analysis = self._analysis if analysis is None else analysis
        routine = self._routine if routine is None else routine
        nodes = self._nodes if nodes is None else tuple(nodes)
        if (self._issuer is not _ISSUER or analysis is not self._analysis or routine is not self._routine
                or len(nodes) != len(self._nodes) or any(a is not b for a, b in zip(nodes, self._nodes, strict=True))
                or getattr(analysis, "_atomic_worksharing_teams", {}).get(self.identity) is not self
                or sha256(self._record.encode()).hexdigest() != self.identity):
            _fail("requires its exact registered original team")
        # Consumption cannot silently reissue a retired original completion.
        self.parent.validate(analysis, routine.qualified, self.parent.selected_node_ids)
        parent, units, predicates, bindings, record = _derive(analysis, routine, nodes)
        if (parent is not self.parent or _encoded(record) != self._record
                or len(bindings) != len(self.bindings) or any(a is not b for a, b in zip(bindings, self.bindings, strict=True))
                or len(units) != len(self.units) or any(
                    any(getattr(a, name) is not getattr(b, name) for name in ("opening", "loop", "ending", "group_opening"))
                    or a.unit_kind != b.unit_kind or a.reads != b.reads or a.writes != b.writes
                    for a, b in zip(units, self.units, strict=True))
                or len(predicates) != len(self.predicates) or any(a is not b or x != y
                    for (a, x), (b, y) in zip(predicates, self.predicates, strict=True))):
            _fail("original source, bindings, parent join or private state changed")
        return self


def prove_atomic_worksharing_team(analysis, routine, original_nodes):
    nodes = tuple(original_nodes)
    registry = getattr(analysis, "_atomic_worksharing_teams", None)
    if registry is None or getattr(analysis, "_atomic_worksharing_owner", None) is not analysis:
        registry = analysis._atomic_worksharing_teams = {}
        analysis._atomic_worksharing_owner = analysis
    for proof in registry.values():
        if len(nodes) == len(proof._nodes) and all(a is b for a, b in zip(nodes, proof._nodes, strict=True)):
            return proof.validate(analysis, routine, nodes)
    parent, units, predicates, bindings, record = _derive(analysis, routine, nodes)
    encoded = _encoded(record)
    proof = AtomicWorksharingTeamProof(sha256(encoded.encode()).hexdigest(), parent, units, predicates, bindings,
                                     analysis, routine, nodes, encoded, _ISSUER)
    if len(registry) >= analysis.operation_limit * analysis.procedure_limit:
        _fail("original-team registry budget exhausted")
    registry[proof.identity] = proof
    return proof.validate()


@dataclass(frozen=True, slots=True)
class AtomicWorksharingAlternative:
    """Exact loop sequence usable only in its whole atomic parent's path.

    This token supplies a distinct extraction context, never independently
    completed DO participation.  The source consumer must choose the matching
    original conditional path before invoking its prepared numerical worker.
    """
    identity: str
    team: AtomicWorksharingTeamProof = field(repr=False, compare=False)
    index: int
    loops: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    @property
    def private_roots(self):
        return self.team.parent.private_roots

    def public(self):
        return {"proof_identity": self.identity, **json.loads(self._record)}

    def validate(self, analysis, procedure, selection):
        self.team.validate(analysis)
        if (self._issuer is not _ISSUER or procedure != self.team._routine.qualified
                or getattr(analysis, "_atomic_worksharing_alternatives", {}).get(self.identity) is not self
                or len(selection) != len(self.team._nodes)
                or any(a is not b for a, b in zip(selection, self.team._nodes, strict=True))):
            _fail("requires its registered exact whole-team alternative")
        path = self.team.public()["alternatives"][self.index]
        if (len(path) != len(self.loops) or any(self.team.units[index].loop is not loop
                for index, loop in zip(path, self.loops, strict=True))
                or self._record != _alternative_record(self.team, self.index)):
            _fail("whole-team alternative source order changed")
        return self


def _alternative_record(team, index):
    public = team.public()
    return _encoded({"schema_version": 1, "proof_role": "atomic_original_team_alternative",
        "team_identity": team.identity, "alternative": index,
        "selected_original_units": public["alternatives"][index],
        "parent_completion_identity": team.parent.identity,
        "available": True, "has_openmp_in_closure": True,
        "caller_contract": "selected original conditional path inside one atomic whole-team replacement",
        "join": "original parent PARALLEL completion only",
        "private_resources": list(team.parent.private_roots),
        "standalone_execution_authority": False, "individual_unit_completion_authorized": False,
        "native_effects_authority": False, "gpu_legality_established": False})


def atomic_worksharing_alternatives(team):
    team.validate()
    analysis = team._analysis
    registry = getattr(analysis, "_atomic_worksharing_alternatives", None)
    if registry is None or getattr(analysis, "_atomic_worksharing_alternative_owner", None) is not analysis:
        registry = analysis._atomic_worksharing_alternatives = {}
        analysis._atomic_worksharing_alternative_owner = analysis
    result = []
    for index, path in enumerate(team.public()["alternatives"]):
        record = _alternative_record(team, index)
        identity = sha256(record.encode()).hexdigest()
        token = registry.get(identity)
        if token is None:
            token = AtomicWorksharingAlternative(identity, team, index,
                tuple(team.units[number].loop for number in path), record, _ISSUER)
            registry[identity] = token
        token.validate(analysis, team._routine.qualified, team._nodes)
        result.append(token)
    return tuple(result)
