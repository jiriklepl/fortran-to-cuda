"""Source-issued coverage for ordered tile-local producer/consumer scratch.

No source expression is substituted or recomputed. This first slice proves
rectangular coverage and disjoint tile outputs, retaining all original loops,
assignments, branches, evaluation counts and phase order for ordinary lowering.
It grants no placement, initialized host storage or physical alias authority.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.native_private import original_private_array
from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError

_ISSUER = object()


def _fail(message):
    raise CompilationError("tile-private scratch " + message)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _linear(node, environment=None):
    """Only integer affine syntax; never evaluate payload or a descriptor."""
    env = environment or {}
    try:
        tree = ast.parse(str(node).strip().lower(), mode="eval").body
    except SyntaxError:
        _fail("requires affine integer coordinates")

    def add(a, b, scale=1):
        terms = dict(a[1])
        for name, value in b[1].items():
            terms[name] = terms.get(name, 0) + scale * value
        return a[0] + scale * b[0], {k: v for k, v in terms.items() if v}

    def visit(value):
        if isinstance(value, ast.Constant) and type(value.value) is int:
            return value.value, {}
        if isinstance(value, ast.Name):
            return env.get(value.id, (0, {value.id: 1}))
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
            result = visit(value.operand)
            return result if isinstance(value.op, ast.UAdd) else (-result[0], {k: -v for k, v in result[1].items()})
        if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Sub)):
            return add(visit(value.left), visit(value.right), -1 if isinstance(value.op, ast.Sub) else 1)
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
            a, b = visit(value.left), visit(value.right)
            if a[1] and b[1]:
                _fail("requires affine integer coordinates")
            scalar, other = (a[0], b) if not a[1] else (b[0], a)
            return scalar * other[0], {k: scalar * v for k, v in other[1].items() if scalar * v}
        _fail("requires affine integer coordinates")

    return visit(tree)


def _body(loop):
    return tuple(x for x in _children(loop) if _kind(x) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"})


def _loop(loop):
    if _kind(loop) != "Block_Nonlabel_Do_Construct":
        _fail("requires original counted DO nests")
    opening = next((x for x in _children(loop) if _kind(x) == "Nonlabel_Do_Stmt"), None)
    controls = opening.items[1] if opening is not None else None
    counted = controls.items[1] if controls is not None else None
    if counted is None:
        _fail("requires original counted DO nests")
    name, bounds = str(counted[0]).lower(), tuple(counted[1])
    return name, bounds


def _nest(loop, rank):
    chain, current = [], loop
    for index in range(rank):
        name, bounds = _loop(current)
        chain.append((current, name, bounds))
        if index + 1 != rank:
            body = _body(current)
            if len(body) != 1:
                _fail("requires a perfect nest within each phase")
            (current,) = body
    return chain, _body(current)


def _clipped(expression, base, width, upper):
    """Keep MIN exactly; prove a lower bound relative to the consumer ceiling."""
    try:
        value = ast.parse(str(expression).lower(), mode="eval").body
    except SyntaxError:
        _fail("requires original MIN-clipped tile bounds")
    offset = 0
    if (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, (ast.Add, ast.Sub))
        and isinstance(value.right, ast.Constant)
    ):
        if type(value.right.value) is not int:
            _fail("requires integer tile clipping")
        offset = value.right.value * (-1 if isinstance(value.op, ast.Sub) else 1)
        value = value.left
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "min"
        and len(value.args) == 2
        and not value.keywords
    ):
        _fail("requires original MIN-clipped tile bounds")
    args = [_linear(ast.unparse(x)) for x in value.args]
    endpoint = _linear(upper)
    if args[1] == endpoint:
        inside = args[0]
    elif args[0] == endpoint:
        inside = args[1]
    else:
        _fail("clipping lost the original domain upper bound")
    expected = _linear(base + "+" + str(width))
    if inside[1] != expected[1]:
        _fail("clipping lost the original tile width")
    # MIN(base+width+k,upper)+p >= MIN(base+width-1,upper)+p+min(k+1,0).
    return inside[0] - expected[0], offset


def _checked_item_span(item, *, original):
    span = getattr(item, "fort_original_span", item.span) if original else item.span
    if (not isinstance(span, tuple) or len(span) != 2 or any(type(value) is not int for value in span)
            or not 1 <= span[0] <= span[1]):
        _fail("requires authenticated original source coordinates")
    return span


def _loop_item(node):
    for item in walk(node):
        if _kind(item) == "Nonlabel_Do_Stmt":
            return item.item
    _fail("lost original source location")


def _span(node):
    return _checked_item_span(_loop_item(node), original=True)[0]


def _private_group(analysis, routine, node):
    """Read original lexical PRIVATE clauses; never trust supplied names."""
    if analysis._require_original(routine.qualified) is not routine:
        _fail("requires the exact authenticated original routine")
    original_lines = analysis.source_text(routine.scope.path).splitlines()
    # AST item.span belongs to the configured parser input. Only published edit
    # coordinates belong to the original file; scanning that file with prepared
    # offsets can both crash and borrow an inactive CPP PRIVATE directive.
    lines = analysis.source_text(analysis.inputs.path(routine.scope.path)).splitlines()
    target = _checked_item_span(_loop_item(node), original=False)[0]
    opening, ending = routine.scope.node.content[0].item, routine.scope.node.content[-1].item
    first, last = _checked_item_span(opening, original=False)[0], _checked_item_span(ending, original=False)[1]
    original_first = _checked_item_span(opening, original=True)[0]
    original_last = _checked_item_span(ending, original=True)[1]
    if (not 1 <= first <= target <= last <= len(lines)
            or not 1 <= original_first <= _span(node) <= original_last <= len(original_lines)):
        _fail("configured source coordinates leave their authenticated routine")
    stack, active_do = [], set()
    for number in range(first, target):
        match = re.match(r"\s*!\$omp\s+(.*)", lines[number - 1], re.I)
        if not match:
            continue
        text = match.group(1).strip().lower()
        if text.startswith("end parallel"):
            if stack:
                stack.pop()
            active_do = set()
        elif re.match(r"parallel(?:\s|$)", text):
            if "&" in text:
                _fail("requires complete original PRIVATE clauses")
            names = {n.strip() for clause in re.findall(r"\bprivate\s*\(([^()]*)\)", text) for n in clause.split(",")}
            stack.append((number, names))
        elif re.match(r"do(?:\s|$)", text):
            active_do = {
                n.strip() for clause in re.findall(r"\bprivate\s*\(([^()]*)\)", text) for n in clause.split(",")
            }
        elif text.startswith("end do"):
            active_do = set()
    if len(stack) != 1:
        _fail("requires one original OpenMP PRIVATE owner")
    start, names = stack[0]
    stop = next(
        (
            i
            for i in range(target, last + 1)
            if re.match(r"\s*!\$omp\s+end\s+parallel(?:\s|$)", lines[i - 1], re.I)
        ),
        None,
    )
    if stop is None:
        _fail("requires the complete original OpenMP owner")
    configured = analysis.inputs.entries.get(str(routine.scope.path))
    if configured is not None:
        mapping = configured["line_map"]
        if len(mapping) != len(lines):
            _fail("configured PRIVATE owner lacks a complete original line map")
        original_start, original_stop = mapping[start - 1], mapping[stop - 1]
    else:
        original_start, original_stop = start, stop
    if (type(original_start) is not int or type(original_stop) is not int
            or not original_first <= original_start < _span(node) < original_stop <= original_last
            or not re.match(r"\s*!\$omp\s+parallel(?:\s|$)", original_lines[original_start - 1], re.I)
            or not re.match(r"\s*!\$omp\s+end\s+parallel(?:\s|$)", original_lines[original_stop - 1], re.I)):
        _fail("PRIVATE owner lacks authenticated original source coordinates")
    return names | active_do, (original_start, original_stop)


def _access(node):
    if _kind(node) != "Part_Ref":
        return None
    return str(node.items[0]).lower(), tuple(_children(node.items[1]))


def _derive_unit(analysis, routine, node, private):
    # Discover scratch from actual accesses and original automatic declarations.
    bindings = {}
    for part in walk(node):
        access = _access(part)
        if access and access[0] in private:
            binding = analysis._binding(routine.scope, access[0])
            if binding is not None and binding.rank:
                original_private_array(routine, binding)
                if binding.dtype != "real" or binding.kind not in (4, 8):
                    _fail("requires REAL4 or REAL8 automatic private arrays")
                bindings[binding.name] = binding
    if not bindings:
        _fail("requires original indexed PRIVATE scratch")
    ranks = {x.rank for x in bindings.values()}
    if len(ranks) != 1 or not 1 <= next(iter(ranks)) <= 3:
        _fail("requires one to three matching scratch/tile axes")
    rank = next(iter(ranks))
    outer, phases = _nest(node, rank)
    if len(phases) != 2:
        _fail("requires exactly one producer followed by one consumer")
    producer, producer_body = _nest(phases[0], rank)
    consumer, consumer_body = _nest(phases[1], rank)
    if analysis._binding(routine.scope, "min") is not None or analysis._candidates(routine.scope, "min"):
        _fail("requires the unshadowed MIN clipping intrinsic")
    axes = []
    for (_, tile, bounds), (_, pi, pb), (_, ci, cb) in zip(outer, producer, consumer, strict=True):
        if (
            len(bounds) != 3
            or len(pb) not in (2, 3)
            or len(cb) not in (2, 3)
            or pi != ci
            or pi not in private
            or len(pb) == 3
            and _linear(pb[2]) != (1, {})
            or len(cb) == 3
            and _linear(cb[2]) != (1, {})
        ):
            _fail("requires PRIVATE phase coordinates with unit positive stride")
        lower, upper, width = bounds
        if _linear(cb[0]) != _linear(tile) or _clipped(cb[1], tile, width, upper) != (-1, 0):
            _fail("requires the original clipped consumer domain")
        pl = _linear(pb[0])
        t = _linear(tile)
        if pl[1] != t[1]:
            _fail("producer lower bound lost its tile coordinate")
        k, p = _clipped(pb[1], tile, width, upper)
        axes.append(
            {
                "tile": tile,
                "point": pi,
                "lower": str(lower),
                "upper": str(upper),
                "width": str(width),
                "producer_lower_offset": pl[0] - t[0],
                "scratch_index_lower": pl[0] - t[0] + 1,
                "scratch_index_upper_width_offset": k + p + 1,
                "producer_upper_guarantee": p + min(k + 1, 0),
            }
        )
    if len({a["tile"] for a in axes}) != rank or len({a["point"] for a in axes}) != rank:
        _fail("requires distinct original tile and point axes")
    coordinates = {axis[role] for axis in axes for role in ("tile", "point")}
    if any(set(_linear(bound)[1]) & coordinates for _, _, bounds in outer for bound in bounds):
        _fail("requires rectangular outer tile domains independent of tile and point coordinates")
    layouts, written, output = {}, set(), {}
    array_reads = []
    for phase, body in (("producer", producer_body), ("consumer", consumer_body)):
        environment = {}
        conditional_assignments = {
            id(value)
            for root in body
            for condition in walk(root)
            if _kind(condition) in {"If_Construct", "If_Stmt"}
            for value in walk(condition)
            if _kind(value) == "Assignment_Stmt"
        }

        def check_reads(expression, phase=phase, environment=environment):
            for value in walk(expression):
                read = _access(value)
                if not read:
                    continue
                name, indices = read
                if name in bindings:
                    if phase != "consumer" or name not in layouts or len(indices) != rank:
                        _fail("reads scratch without a complete prior producer")
                    for index, axis in zip(indices, layouts[name], strict=True):
                        constant, terms = _linear(index, environment)
                        shift = constant - 1
                        if (
                            terms != {axis["point"]: 1, axis["tile"]: -1}
                            or shift < axis["producer_lower_offset"]
                            or shift > axis["producer_upper_guarantee"]
                        ):
                            _fail("shifted scratch read lacks unconditional producer coverage")
                else:
                    binding = analysis._binding(routine.scope, name)
                    if binding is None or not binding.rank:
                        _fail("cannot infer an unknown function or storage effect")
                    array_reads.append(
                        (
                            binding.root,
                            phase,
                            tuple(_linear(x, environment) for x in indices)
                            if all(_kind(x) != "Subscript_Triplet" for x in indices)
                            else None,
                        )
                    )

        for part in (item for root in body for item in walk(root)):
            kind = _kind(part)
            if kind.endswith("_Stmt") and kind not in {
                "Assignment_Stmt",
                "If_Then_Stmt",
                "Else_If_Stmt",
                "Else_Stmt",
                "End_If_Stmt",
                "If_Stmt",
            }:
                _fail("cannot move calls, storage changes or control effects inside a tile")
            if kind == "Block_Nonlabel_Do_Construct" or phase == "producer" and kind in {"If_Construct", "If_Stmt"}:
                _fail("requires unconditional rectangular producers and no extra nested loops")
            if kind == "Function_Reference":
                _fail("cannot infer effects of an original helper function")
            if kind in {"If_Then_Stmt", "Else_If_Stmt", "If_Stmt"}:
                check_reads(part.items[0])
            if kind != "Assignment_Stmt":
                continue
            lhs, _, rhs = part.items
            target = _access(lhs)
            if _kind(lhs) == "Name":
                name = str(lhs).lower()
                binding = analysis._binding(routine.scope, name)
                if name not in private or binding is None or binding.rank:
                    _fail("cannot modify shared scalar state")
                if binding.dtype == "integer" and id(part) in conditional_assignments:
                    _fail("requires unconditional original integer coordinate aliases")
            elif target is None:
                _fail("requires ordinary scalar or indexed array assignments")
            elif target[0] not in bindings:
                if phase != "consumer":
                    _fail("producer may write only private scratch")
                binding = analysis._binding(routine.scope, target[0])
                if binding is None or binding.rank != rank or len(target[1]) != rank:
                    _fail("requires exact point-local output arrays")
                coords = [_linear(x, environment) for x in target[1]]
                if {
                    next(iter(x[1])) for x in coords if len(x[1]) == 1 and next(iter(x[1].values())) == 1
                } != {a["point"] for a in axes}:
                    _fail("outputs can overlap a different logical tile")
                if binding.root in output and output[binding.root] != tuple(coords):
                    _fail("output coordinates change within the tile")
                output[binding.root] = tuple(coords)
            else:
                if phase != "producer" or target[0] in written:
                    _fail("requires one unconditional producer definition per scratch cell")
                if len(target[1]) != rank:
                    _fail("scratch producer rank differs")
                order = []
                for index in target[1]:
                    constant, terms = _linear(index, environment)
                    matches = [axis for axis in axes if terms == {axis["point"]: 1, axis["tile"]: -1} and constant == 1]
                    if len(matches) != 1:
                        _fail("requires original tile-relative producer coordinates")
                    order.append(matches[0])
                if len({x["tile"] for x in order}) != rank:
                    _fail("scratch coordinates do not span the tile")
                layouts[target[0]] = order
                written.add(target[0])
            check_reads(rhs)
            if _kind(lhs) == "Name":
                name = str(lhs).lower()
                try:
                    environment[name] = _linear(rhs, environment)
                except CompilationError:
                    environment.pop(name, None)
    if written != set(bindings) or not output:
        _fail("requires complete scratch production and a disjoint consumer output")
    # A written external array may only be read at this same consumer point.
    for resource, phase, coords in array_reads:
        if resource in output and (phase != "consumer" or coords != output[resource]):
            _fail("external payload dependence can cross logical tiles")
    return (
        {
            "axes": axes,
            "scratch": [x.public() for x in sorted(bindings.values(), key=lambda x: x.name)],
            "scratch_axis_order": {name: [axis["tile"] for axis in order] for name, order in sorted(layouts.items())},
            "outputs": sorted(output),
            "producer": str(phases[0]),
            "consumer": str(phases[1]),
        },
        tuple(bindings.values()),
        phases,
    )


def _derive(analysis, routine, node):
    analysis.verify_source()
    graph, selected = analysis._selected_source(routine.qualified, (node,))
    private, owner_span = _private_group(analysis, routine, node)
    record, bindings, phases = _derive_unit(analysis, routine, node, private)
    # Every use of these original PRIVATE arrays in their lexical owner must
    # belong to a complete independently covered tile unit. No cross-unit live
    # scratch or opaque observer is silently discarded.
    covered = set()
    for candidate in walk(routine.execution):
        if _kind(candidate) != "Block_Nonlabel_Do_Construct":
            continue
        line = _span(candidate)
        if not owner_span[0] < line < owner_span[1]:
            continue
        try:
            other, _, _ = _derive_unit(analysis, routine, candidate, private)
        except CompilationError:
            continue
        if set(x["name"] for x in other["scratch"]) & {x.name for x in bindings}:
            covered.update(id(x) for x in walk(candidate))
    for value in walk(routine.execution):
        access = _access(value)
        if access and access[0] in {x.name for x in bindings} and id(value) not in covered:
            _fail("has scratch use outside completely proved tile units")
    private_scalars = set()
    for value in walk(node):
        if _kind(value) == "Name" and str(value).lower() in private:
            binding = analysis._binding(routine.scope, str(value).lower())
            if binding is not None and not binding.rank:
                private_scalars.add(binding.name)
    for candidate in walk(routine.execution):
        item = getattr(candidate, "item", None)
        if id(candidate) in covered or item is None or not _kind(candidate).endswith("_Stmt"):
            continue
        if owner_span[0] < _checked_item_span(item, original=True)[0] < owner_span[1] and any(
            _kind(value) == "Name" and str(value).lower() in private_scalars for value in walk(candidate)
        ):
            _fail("private scalar has an observer outside completely proved tile units")
    record.update(
        version=1,
        procedure=routine.qualified,
        source_graph=graph.identity,
        selected_node_ids=list(selected),
        source_inputs=analysis.inputs.identity(),
        original_tile=str(node),
        owner_span=list(owner_span),
        original_evaluation_order=True,
        recomputation_authorized=False,
        definitions_authorized=False,
        placement_authorized=False,
        requires_existing_alias_guard=True,
        descriptor_policy="original allocated bounds; reached nonzero domain only; never reevaluate specification expressions",
        phase_barrier="single thread owns one tile; original producer completes before consumer",
        pricing="unpriced; forced sections only",
        runtime_requirements=[
            "positive original tile widths and nonzero source strides",
            "guard inner tile-bound snapshots by prior-axis positive extent",
            "original allocated scratch descriptors cover every producer/read coordinate",
            "checked integer addresses and size_t products; bounded scratch lease",
            "ordinary source-array bounds, definition, physical-alias and floating-environment guards",
            "all batches ordered on one actual context stream or explicit scratch dependencies",
        ],
    )
    return record, bindings, phases


@dataclass(frozen=True, slots=True)
class TilePrivateProof:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _node: object = field(repr=False, compare=False)
    bindings: tuple = field(repr=False, compare=False)
    phases: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def validate(self, analysis=None, routine=None, node=None):
        analysis = self._analysis if analysis is None else analysis
        routine = self._routine if routine is None else routine
        node = self._node if node is None else node
        if (
            self._issuer is not _ISSUER
            or analysis is not self._analysis
            or routine is not self._routine
            or node is not self._node
            or getattr(analysis, "_tile_private_proofs", {}).get(self.identity) is not self
        ):
            _fail("requires its exact compiler-issued original proof")
        from compiler.frontend.tile_validation import reuse_validated_unit

        if reuse_validated_unit(self):
            return self
        record, bindings, phases = _derive(analysis, routine, node)
        if (
            _encoded(record) != self._record
            or any(a is not b for a, b in zip(bindings, self.bindings, strict=True))
            or any(a is not b for a, b in zip(phases, self.phases, strict=True))
        ):
            _fail("original source or storage identity changed")
        return self


def prove_tile_private(analysis, routine, original_outer_loop):
    for proof in getattr(analysis, "_tile_private_proofs", {}).values():
        if proof._node is original_outer_loop:
            return proof.validate(analysis, routine, original_outer_loop)
    record, bindings, phases = _derive(analysis, routine, original_outer_loop)
    encoded = _encoded(record)
    proof = TilePrivateProof(
        sha256(encoded.encode()).hexdigest(), analysis, routine, original_outer_loop, bindings, phases, encoded, _ISSUER
    )
    registry = getattr(analysis, "_tile_private_proofs", None)
    if registry is None:
        registry = analysis._tile_private_proofs = {}
    registry[proof.identity] = proof
    return proof.validate(analysis, routine, original_outer_loop)


def _cooperative_derive(proof, region_id):
    """Prove per-point scalar independence from the original lowered statements."""
    from compiler.analysis.effects import header_reads, upward_reads
    from compiler.frontend.lowering import lower_tile_private
    from compiler.ir import Assignment, If, Loop

    proof.validate()
    lowered = lower_tile_private(proof, region_id=region_id)
    region = lowered.region
    allowed = set(region.parameters) | {loop.iterator for loop in region.outer_loops}
    summaries = []
    for name, phase in (("producer", region.producer), ("consumer", region.consumer)):
        body = phase
        coordinates = []
        for _ in region.outer_loops:
            if len(body.statements) != 1 or not isinstance(body.statements[0], Loop):
                _fail("cooperative mapping requires the original perfect phase nest")
            loop = body.statements[0]
            if header_reads(loop) - allowed:
                _fail("cooperative phase bounds depend on a point-private value")
            coordinates.append(loop.iterator)
            body = loop.body
        missing = {symbol for symbol in upward_reads(body) - allowed - set(coordinates) if not symbol.rank}
        if missing:
            _fail("cooperative point body carries private scalar state: " + ", ".join(sorted(x.name for x in missing)))

        def ordinary(block):
            for statement in block.statements:
                if isinstance(statement, If):
                    ordinary(statement.then_body)
                    ordinary(statement.else_body)
                elif not isinstance(statement, Assignment):
                    _fail("cooperative point body requires ordinary assignments and branches")

        ordinary(body)
        summaries.append({"phase": name, "coordinates": [x.name for x in coordinates],
                          "private_scalar_inputs": [], "point_evaluation_order": "original lowered body unchanged"})
    # Original PRIVATE terminal/last-iteration values must have no hidden native
    # observer after or between replaced tile units. Reject rather than invent
    # a scalar live-out or a relation to an arbitrary OpenMP worker schedule.
    analysis, routine = proof._analysis, proof._routine
    private, span = _private_group(analysis, routine, proof._node)
    names = private & {x.name for x in region.private_symbols}
    covered = set()
    for candidate in walk(routine.execution):
        if _kind(candidate) != "Block_Nonlabel_Do_Construct" or not span[0] < _span(candidate) < span[1]:
            continue
        try:
            _derive_unit(analysis, routine, candidate, private)
        except CompilationError:
            continue
        covered.update(id(x) for x in walk(candidate))
    for candidate in walk(routine.execution):
        item = getattr(candidate, "item", None)
        if id(candidate) in covered or item is None or not _kind(candidate).endswith("_Stmt"):
            continue
        if span[0] < item.span[0] < span[1] and any(
            _kind(value) == "Name" and str(value).lower() in names for value in walk(candidate)
        ):
            _fail("cooperative private scalar has an observer outside proved tile units")
    full_points = len(region.outer_loops) > 1
    record = {"version": 2 if full_points else 1, "source_proof_identity": proof.identity, "region_id": region_id,
              "procedure": routine.qualified, "source_graph": proof.public()["source_graph"],
              "phases": summaries, "mapping": ("one CTA per tile; thread-strided rectangular points without a point product"
                                                if full_points else "one CTA per tile; thread-strided innermost points"),
              "phase_barriers": "unconditional after producer and consumer, including before scratch reuse",
              "scalar_liveout_authorized": False, "definitions_authorized": False, "placement_authorized": False,
              "pricing": "unpriced; explicit forced sections only"}
    if full_points:
        record["point_distribution"] = {
            "order": "original lexicographic coordinates; innermost coordinate varies first",
            "advance": "mixed-radix carry by blockDim.x; no multidimensional size product",
            "bounds": "uniform nested nonempty guards retain original bound reachability",
            "arithmetic": "32-bit source extents and at most one block-width carry in 64-bit size_t",
        }
    return record, lowered


@dataclass(frozen=True, slots=True)
class CooperativeTilePrivateProof:
    """Separate issued permission to distribute independent original points."""

    identity: str
    source: TilePrivateProof = field(repr=False, compare=False)
    lowered: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def validate(self, region=None):
        if self._issuer is not _ISSUER or getattr(self.source._analysis, "_cooperative_tile_private_proofs", {}).get(self.identity) is not self:
            _fail("cooperative mapping requires its exact compiler-issued proof")
        record, lowered = _cooperative_derive(self.source, self.lowered.region.id)
        if _encoded(record) != self._record or lowered.region != self.lowered.region or lowered.addressing != self.lowered.addressing:
            _fail("cooperative source lowering or point independence changed")
        if region is not None and region != lowered.region:
            _fail("cooperative mapping received substitute lowered work")
        return self


def prove_cooperative_tile_private(proof, *, region_id=0):
    if type(proof) is not TilePrivateProof or type(region_id) is not int or region_id < 0:
        _fail("cooperative mapping requires an issued original tile proof")
    record, lowered = _cooperative_derive(proof, region_id)
    text = _encoded(record)
    identity = sha256(text.encode()).hexdigest()
    analysis = proof._analysis
    if not hasattr(analysis, "_cooperative_tile_private_proofs"):
        analysis._cooperative_tile_private_proofs = {}
    registry = analysis._cooperative_tile_private_proofs
    if identity not in registry:
        registry[identity] = CooperativeTilePrivateProof(identity, proof, lowered, text, _ISSUER)
    return registry[identity].validate()


def _derive_tile_team(analysis, routine, nodes):
    from compiler.frontend.lowering import lower_tile_private
    from compiler.scopes.segments import directive, statement_span

    analysis.verify_source()
    if not isinstance(nodes, tuple) or len(nodes) < 3:
        _fail("team requires an exact complete original node tuple")

    def contiguous(value):
        children = tuple(_children(value))
        if any(len(children[index:index + len(nodes)]) == len(nodes) and all(
            actual is expected for actual, expected in zip(children[index:index + len(nodes)], nodes, strict=True)
        ) for index in range(len(children))):
            return True
        return any(contiguous(child) for child in children)

    if not contiguous(routine.execution):
        _fail("team nodes are not one unchanged contiguous original selection")
    start, end = directive(nodes[0]), directive(nodes[-1])
    if start is None or end != "end parallel" or not re.fullmatch(r"parallel(?:\s+private\s*\([^()]+\))*\s*", start):
        _fail("team requires the entire original plain PRIVATE parallel owner")
    span = statement_span(nodes[0])[0], statement_span(nodes[-1])[1]
    units, fields, scalars, scratch = [], {}, {}, {}
    in_do = False
    do_has_loop = False
    for node in nodes[1:-1]:
        text = directive(node)
        if text is not None:
            if re.match(r"do(?:\s|$)", text) and not in_do:
                rest = re.sub(r"\bschedule\s*\([^()]*\)|\bcollapse\s*\(\s*[123]\s*\)", "", text[2:]).strip()
                if rest:
                    _fail("team has unsupported original worksharing control")
                in_do = True
                do_has_loop = False
            elif text == "end do" and in_do and do_has_loop:
                in_do = False
            else:
                _fail("team has extra directives, NOWAIT or nested owners")
            continue
        if _kind(node) == "Comment":
            continue
        if not in_do or _kind(node) != "Block_Nonlabel_Do_Construct":
            _fail("team contains observable work outside proved original tile units")
        # A worksharing directive must select one whole tile nest, never two.
        if do_has_loop:
            _fail("team worksharing interval contains several roots")
        proof = prove_tile_private(analysis, routine, node)
        if tuple(proof.public()["owner_span"]) != span:
            _fail("team proof does not cover its whole original PRIVATE lifetime")
        lowered = lower_tile_private(proof, region_id=len(units))
        written = set(proof.public()["outputs"])
        for binding, symbol in lowered.bindings:
            if symbol not in lowered.region.parameters:
                continue
            if symbol.rank:
                previous = fields.get(binding.root)
                fields[binding.root] = (binding, binding.root in written or previous is not None and previous[1])
            else:
                scalars[binding.root] = binding
        for binding in proof.bindings:
            scratch[binding.root] = binding
        units.append(proof)
        do_has_loop = True
    if in_do or not units:
        _fail("team lacks complete original worksharing intervals")
    domains = [tuple((axis["lower"].lower(), axis["upper"].lower(), axis["width"].lower())
                     for axis in unit.public()["axes"]) for unit in units]
    if any(domain != domains[0] for domain in domains[1:]):
        _fail("team first slice requires one common original rectangular domain")
    graph, selected = analysis._selected_source(routine.qualified, nodes)
    fields = tuple(fields[key] for key in sorted(fields))
    scalars = tuple(scalars[key] for key in sorted(scalars))
    scratch = tuple(scratch[key] for key in sorted(scratch))
    record = {"version": 1, "procedure": routine.qualified, "source_graph": graph.identity,
              "selected_node_ids": list(selected), "source_inputs": analysis.inputs.identity(),
              "span": list(span), "original_team": "\n".join(map(str, nodes)),
              "unit_proof_identities": [unit.identity for unit in units],
              "domain": [{key: axis[key] for key in ("lower", "upper", "width")} for axis in units[0].public()["axes"]],
              "fields": [{"binding": binding.public(), "written": written} for binding, written in fields],
              "scalars": [binding.public() for binding in scalars], "scratch": [binding.public() for binding in scratch],
              "source_order": "all original tile units and phase evaluations retained",
              "fallback": "unchanged entire original OpenMP team once; prework only",
              "definitions_authorized": False, "placement_authorized": False,
              "requires_existing_capture_alias_bounds_definition_and_FP_guards": True,
              "pricing": "unpriced; forced sections only"}
    return record, tuple(units), fields, scalars, scratch


@dataclass(frozen=True, slots=True)
class TilePrivateTeamProof:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _nodes: tuple = field(repr=False, compare=False)
    units: tuple = field(repr=False, compare=False)
    fields: tuple = field(repr=False, compare=False)
    scalars: tuple = field(repr=False, compare=False)
    scratch_bindings: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def validate(self, analysis=None, routine=None, nodes=None):
        analysis = self._analysis if analysis is None else analysis
        routine = self._routine if routine is None else routine
        nodes = self._nodes if nodes is None else nodes
        if (self._issuer is not _ISSUER or analysis is not self._analysis or routine is not self._routine
                or len(nodes) != len(self._nodes) or any(a is not b for a, b in zip(nodes, self._nodes, strict=True))
                or getattr(analysis, "_tile_private_team_proofs", {}).get(self.identity) is not self):
            _fail("team requires its exact compiler-issued whole source proof")
        from compiler.frontend.tile_validation import remember_validated_team, reuse_validated_team

        if reuse_validated_team(self):
            return self
        record, units, fields, scalars, scratch = _derive_tile_team(analysis, routine, self._nodes)
        if (_encoded(record) != self._record or any(a is not b for a, b in zip(units, self.units, strict=True))
                or len(fields) != len(self.fields) or any(a is not b or x != y for (a, x), (b, y) in zip(fields, self.fields, strict=True))
                or any(a is not b for a, b in zip(scalars, self.scalars, strict=True))
                or any(a is not b for a, b in zip(scratch, self.scratch_bindings, strict=True))):
            _fail("team source, complete units or original bindings changed")
        remember_validated_team(self)
        return self


def prove_tile_private_team(analysis, routine, nodes):
    nodes = tuple(nodes)
    for proof in getattr(analysis, "_tile_private_team_proofs", {}).values():
        if (proof._analysis is analysis and proof._routine is routine and len(proof._nodes) == len(nodes)
                and all(a is b for a, b in zip(proof._nodes, nodes, strict=True))):
            return proof.validate(analysis, routine, nodes)
    record, units, fields, scalars, scratch = _derive_tile_team(analysis, routine, nodes)
    text = _encoded(record)
    identity = sha256(text.encode()).hexdigest()
    if not hasattr(analysis, "_tile_private_team_proofs"):
        analysis._tile_private_team_proofs = {}
    registry = analysis._tile_private_team_proofs
    if identity not in registry:
        registry[identity] = TilePrivateTeamProof(identity, analysis, routine, nodes, units, fields, scalars, scratch, text, _ISSUER)
    return registry[identity].validate()
