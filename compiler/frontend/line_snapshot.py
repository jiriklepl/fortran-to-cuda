"""Original complete PRIVATE line snapshots; no placement or execution authority.

This issuer recognizes disjoint transverse lines and ordered private production.
Descriptor conformance, integer/address ranges, initialized external regions and
physical alias checks remain mandatory reached prework obligations. It supplies
no invented tile/MIN geometry and does not alter the original OpenMP owner.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.native_private import original_private_array
from compiler.frontend.source_effects import _children, _kind
from compiler.frontend.tile_private import _linear, _loop
from compiler.ir import CompilationError
from compiler.scopes.segments import directive, statement_span

_ISSUER = object()


def _fail(message):
    raise CompilationError("original line snapshot " + message)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _body(node):
    return tuple(x for x in _children(node) if _kind(x) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"})


def _affine(value):
    constant, terms = _linear(value)
    return {"constant": constant, "terms": dict(sorted(terms.items()))}


def _sum(left, right, coefficient=1):
    terms = dict(left[1])
    for name, value in right[1].items():
        terms[name] = terms.get(name, 0) + coefficient * value
    return left[0] + coefficient * right[0], {k: v for k, v in terms.items() if v}


def _interval(node, environment):
    constant, terms = _linear(node)
    lower, upper = (constant, {}), (constant, {})
    for name, coefficient in terms.items():
        lo, hi = environment.get(name, ((0, {name: 1}), (0, {name: 1})))
        if coefficient < 0:
            lo, hi = hi, lo
        lower, upper = _sum(lower, lo, coefficient), _sum(upper, hi, coefficient)
    return lower, upper


def _public_interval(value):
    return {
        key: {"constant": part[0], "terms": dict(sorted(part[1].items()))}
        for key, part in zip(("lower", "upper"), value, strict=True)
    }


def _relational_interval(node, loops):
    """Substitute innermost original affine iterator relations before envelopes.

    This retains ii=i+offset correlation while checking w(ii-i). It does not
    authorize integer arithmetic: original bound/index/terminal validity remains
    a mandatory prework obligation of any future numerical consumer.
    """
    lower = upper = _linear(node)
    for loop in reversed(loops):
        name, bounds = _loop(loop)
        lo, hi = _linear(bounds[0]), _linear(bounds[1])
        result = []
        for expression, choose_lower in ((lower, True), (upper, False)):
            constant, terms = expression
            terms = dict(terms)
            coefficient = terms.pop(name, 0)
            chosen = lo if (coefficient >= 0) == choose_lower else hi
            result.append(_sum((constant, terms), chosen, coefficient))
        lower, upper = result
    return lower, upper


def _derive(analysis, routine, nodes, *, completion=None):
    analysis.verify_source()
    if analysis._require_original(routine.qualified) is not routine:
        _fail("requires the exact original routine")
    if not isinstance(nodes, tuple) or not nodes:
        _fail("requires an exact nonempty original selection")
    graph, selected = analysis._selected_source(routine.qualified, nodes)
    if completion is None:
        completion = analysis.joined_completion(routine.qualified, nodes)
    completion.validate(analysis, routine.qualified, selected)
    # Grouping may attach the opening directive to its original DO. Normalize
    # only the view, retaining every source object and the complete issuer token.
    normalized = []
    for node in nodes:
        if _kind(node) == "Block_Nonlabel_Do_Construct":
            normalized.extend(x for x in _children(node) if _kind(x) == "Comment" and directive(x))
        normalized.append(node)
    directives = [directive(x) for x in normalized if directive(x)]
    if (
        len(directives) != 2
        or not re.match(r"parallel\s+do(?:\s|$)", directives[0])
        or directives[-1] != "end parallel do"
    ):
        _fail("requires one complete original combined PARALLEL DO")
    roots = [x for x in nodes if _kind(x) != "Comment"]
    if len(roots) != 1 or _kind(roots[0]) != "Block_Nonlabel_Do_Construct":
        _fail("requires one complete original associated loop")
    outer = roots[0]
    if routine.issues:
        _fail("requires an exact ordinary numeric specification")
    private = set(completion.private_roots)
    transverse, current = [], outer
    while True:
        name, bounds = _loop(current)
        binding = analysis._binding(routine.scope, name)
        if (
            binding is None
            or binding.root not in private
            or binding.dtype != "integer"
            or binding.kind != 4
            or len(bounds) == 3
            and _linear(bounds[2]) != (1, {})
        ):
            _fail("requires original private INTEGER transverse loops with unit positive stride")
        transverse.append((current, binding, bounds))
        body = _body(current)
        if len(body) == 1 and _kind(body[0]) == "Block_Nonlabel_Do_Construct":
            current = body[0]
            if len(transverse) >= 2:
                _fail("requires one or two transverse axes")
            continue
        break
    if len(body) != 2 or _kind(body[0]) != "Assignment_Stmt" or _kind(body[1]) != "Block_Nonlabel_Do_Construct":
        _fail("requires one unconditional snapshot followed by its original serial consumer")
    producer, consumer = body
    target, _, source = producer.items
    if _kind(target) != "Name" or _kind(source) != "Part_Ref":
        _fail("requires a whole private vector copied from one original array line")
    scratch = analysis._binding(routine.scope, target)
    field_binding = analysis._binding(routine.scope, source.items[0])
    if scratch is None or scratch.root not in private or scratch.rank != 1 or scratch.dtype != "real":
        _fail("requires one original rank-one REAL private snapshot")
    original_private_array(routine, scratch)
    if (
        field_binding is None
        or field_binding.rank != len(transverse) + 1
        or field_binding.dtype != scratch.dtype
        or field_binding.kind != scratch.kind
    ):
        _fail("requires conforming original numeric line storage")
    indices = tuple(_children(source.items[1]))
    slices = [i for i, x in enumerate(indices) if _kind(x) == "Subscript_Triplet"]
    if len(indices) != field_binding.rank or len(slices) != 1 or any(x is not None for x in indices[slices[0]].items):
        _fail("first slice requires one whole original line axis")
    line_axis = slices[0]
    transverse_names = {binding.name for _, binding, _ in transverse}
    if len(transverse_names) != len(transverse):
        _fail("transverse loop coordinates must be distinct")
    fixed = {i: _linear(value) for i, value in enumerate(indices) if i != line_axis}
    if {next(iter(terms)) for _, terms in fixed.values() if len(terms) == 1} != transverse_names or any(
        constant or len(terms) != 1 or next(iter(terms.values())) != 1 for constant, terms in fixed.values()
    ):
        _fail("snapshot must preserve each independent transverse coordinate exactly")
    fields = {field_binding.root: (field_binding, False)}
    scalars, constants, scratch_reads, array_accesses = {}, {}, [], []
    private_bindings, active = {scratch.root: scratch}, set()
    active_loops, constant_accesses, constant_records = [], [], []
    environment = {}

    def binding_for(node):
        binding = analysis._binding(analysis.source_scope_for(node, routine.scope), node)
        if binding is None or binding.attributes & {
            "pointer",
            "target",
            "optional",
            "volatile",
            "asynchronous",
            "value",
        }:
            _fail("requires ordinary source-bound numeric storage")
        if binding.dtype not in {"real", "integer", "logical"} or binding.kind not in {1, 4, 8}:
            _fail("requires ordinary numeric kinds")
        return binding

    if binding_for(source.items[0]) is not field_binding:
        _fail("snapshot source identity changed")

    def scalar_read(node, defined):
        binding = binding_for(node)
        if binding.rank:
            _fail("consumer cannot read whole arrays or unproved sections")
        if binding.root in private:
            if binding.root not in defined:
                _fail("private scalar lacks a preceding same-line definition: " + binding.name)
            private_bindings[binding.root] = binding
        else:
            scalars[binding.root] = binding

    def coordinate(node, defined):
        _linear(node)
        for name in walk(node):
            if _kind(name) == "Name":
                scalar_read(name, defined)
                if binding_for(name).dtype != "integer":
                    _fail("line coordinates require original integer expressions")

    def access(node, defined, env, *, writing=False):
        binding = binding_for(node.items[0])
        coords = tuple(_children(node.items[1]))
        if len(coords) != binding.rank or any(_kind(x) == "Subscript_Triplet" for x in coords):
            _fail("consumer requires ordinary point references")
        for item in coords:
            coordinate(item, defined)
        if binding is scratch:
            if writing:
                _fail("consumer cannot overwrite its initialized private snapshot")
            scratch_reads.append(
                {"source_reference": str(node), "interval": _public_interval(_interval(coords[0], env))}
            )
            return
        if binding.root in private:
            _fail("additional private array storage is unsupported")
        if "parameter" in binding.attributes:
            if writing:
                _fail("cannot write original PARAMETER storage")
            if binding.rank != 1:
                _fail("first slice requires rank-one immutable constant arrays")
            constants[binding.root] = binding
            constant_accesses.append((node, binding, tuple(active_loops)))
            constant_records.append(
                {
                    "resource": binding.root,
                    "source_reference": str(node),
                    "index_expression": str(coords[0]),
                    "original_loop_relations": [
                        {
                            "iterator": _loop(loop)[0],
                            "lower": str(_loop(loop)[1][0]),
                            "upper": str(_loop(loop)[1][1]),
                            "step": 1,
                        }
                        for loop in active_loops
                    ],
                    "affine_index_interval": _public_interval(_relational_interval(coords[0], active_loops)),
                    "integer_range_and_lookup_execution_authorized": False,
                }
            )
            return
        if binding.rank != field_binding.rank or any(_linear(coords[i]) != fixed[i] for i in fixed):
            _fail("external payload crosses independently scheduled lines")
        if writing and binding.intent == "in":
            _fail("cannot write original INTENT(IN) storage")
        previous = fields.get(binding.root)
        fields[binding.root] = (binding, writing or bool(previous and previous[1]))
        array_accesses.append(
            {
                "resource": binding.root,
                "effect": "write" if writing else "read",
                "source_reference": str(node),
                "line_interval": _public_interval(_interval(coords[line_axis], env)),
            }
        )

    def expression(node, defined, env):
        kind = _kind(node)
        if kind.endswith("Literal_Constant") or node is None or isinstance(node, str):
            return
        if kind == "Name":
            scalar_read(node, defined)
            return
        if kind == "Part_Ref":
            access(node, defined, env)
            return
        if kind in {"Function_Reference", "Structure_Constructor", "Data_Ref"}:
            _fail("unknown function, component or call effects are unsupported")
        if kind == "Intrinsic_Function_Reference":
            name = str(node.items[0]).lower()
            if name != "abs" or analysis._binding(routine.scope, name) or analysis._candidates(routine.scope, name):
                _fail("consumer permits only the unshadowed ABS intrinsic")
            for arg in _children(node.items[1]):
                expression(arg, defined, env)
            return
        for child in _children(node):
            expression(child, defined, env)

    def statements(sequence, defined, env):
        defined = set(defined)
        for node in sequence:
            kind = _kind(node)
            if kind == "Comment":
                if directive(node):
                    _fail("nested directives are unsupported")
                continue
            if kind == "Assignment_Stmt":
                lhs, _, rhs = node.items
                expression(rhs, defined, env)
                if _kind(lhs) == "Name":
                    binding = binding_for(lhs)
                    if binding.rank or binding.root not in private or binding.root in active:
                        _fail("consumer may assign only original private non-induction scalars")
                    if binding.dtype == "integer":
                        _fail("first slice does not infer private integer coordinate assignments")
                    defined.add(binding.root)
                    private_bindings[binding.root] = binding
                elif _kind(lhs) == "Part_Ref":
                    access(lhs, defined, env, writing=True)
                else:
                    _fail("unsupported original consumer assignment target")
            elif kind == "Block_Nonlabel_Do_Construct":
                name, bounds = _loop(node)
                binding = binding_for(name)
                if (
                    binding.root not in private
                    or binding.root in active
                    or binding.dtype != "integer"
                    or binding.kind != 4
                    or len(bounds) == 3
                    and _linear(bounds[2]) != (1, {})
                ):
                    _fail("consumer loops require distinct private INTEGER unit-stride coordinates")
                for bound in bounds:
                    coordinate(bound, defined)
                lo = _interval(bounds[0], env)[0]
                hi = _interval(bounds[1], env)[1]
                active.add(binding.root)
                active_loops.append(node)
                private_bindings[binding.root] = binding
                statements(_body(node), defined | {binding.root}, {**env, name: (lo, hi)})
                active.remove(binding.root)
                active_loops.pop()
                # DO's induction variable is assigned even on a zero-trip path;
                # other body definitions cannot escape a potentially empty loop.
                defined.add(binding.root)
            elif kind == "If_Stmt":
                expression(node.items[0], defined, env)
                statements((node.items[1],), defined, env)
            elif kind == "If_Construct":
                arms, arm, has_else, started = [], [], False, False
                for item in node.content:
                    label = _kind(item)
                    if label in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                        if started:
                            arms.append(statements(arm, defined, env))
                        arm = []
                        started = label != "End_If_Stmt"
                        if label in {"If_Then_Stmt", "Else_If_Stmt"}:
                            expression(item.items[0], defined, env)
                        if label == "Else_Stmt":
                            has_else = True
                    else:
                        arm.append(item)
                if has_else and arms:
                    defined |= set.intersection(*arms)
            else:
                _fail("consumer cannot move calls, storage changes or control effects: " + kind)
        return defined

    for node, binding, bounds in transverse:
        for bound in bounds:
            if set(_linear(bound)[1]) & transverse_names:
                _fail("transverse domains must be independent rectangles")
            coordinate(bound, set())
        active.add(binding.root)
        active_loops.append(node)
        private_bindings[binding.root] = binding
        environment[binding.name] = _interval(bounds[0], {})[0], _interval(bounds[1], {})[1]
    statements((consumer,), set(active), environment)
    if not scratch_reads or not any(written for _, written in fields.values()):
        _fail("requires initialized snapshot consumption and a disjoint line output")
    domain = [
        {
            "iterator": binding.name,
            "resource": binding.root,
            "lower": str(bounds[0]),
            "upper": str(bounds[1]),
            "step": 1,
        }
        for _, binding, bounds in transverse
    ]
    record = {
        "version": 1,
        "kind": "original_combined_private_line_snapshot",
        "procedure": routine.qualified,
        "source_graph": graph.identity,
        "selected_node_ids": list(selected),
        "source_inputs": analysis.inputs.identity(),
        "span": [min(statement_span(n)[0] for n in nodes), max(statement_span(n)[1] for n in nodes)],
        "original_team": "\n".join(map(str, nodes)),
        "completion": completion.public(),
        "transverse_domain": domain,
        "line_axis": line_axis + 1,
        "snapshot": {
            "scratch": scratch.public(),
            "source": field_binding.public(),
            "original_assignment": str(producer),
            "rank": 1,
            "fixed_coordinates": {str(i + 1): str(indices[i]) for i in fixed},
            "copy_mapping": "whole original line to whole private vector in Fortran element order",
            "physical_stride": "selected source descriptor stride, including Y/Z strided lines",
            "descriptor_source": "existing reached original bounds; never re-evaluate automatic specification",
        },
        "scratch_reads": scratch_reads,
        "external_point_accesses": array_accesses,
        "fields": [{"binding": b.public(), "written": w} for b, w in sorted(fields.values(), key=lambda x: x[0].root)],
        "scalars": [b.public() for b in sorted(scalars.values(), key=lambda b: b.root)],
        "private_scalars": [b.public() for b in sorted(private_bindings.values(), key=lambda b: b.root) if not b.rank],
        "constant_array_accesses": constant_records,
        "immutable_constant_arrays": [b.public() for b in sorted(constants.values(), key=lambda b: b.root)],
        "source_order": "complete snapshot before serial original consumer; all branches and arithmetic unchanged",
        "runtime_obligations": [
            "positive transverse domains before later operand inspection",
            "conforming source-line and original scratch extents and logical origins",
            "every scratch-read interval lies within the fully initialized original snapshot",
            "checked original integer operations, external addresses and size_t scratch products",
            "initialized external read regions, physical alias and numerical environment checks",
            "bounded same-stream scratch lease; prework rejection only, no replay after submission",
        ],
        "definitions_authorized": False,
        "placement_authorized": False,
        "execution_authorized": False,
        "arithmetic_reassociation_authorized": False,
        "cooperative_mapping_authorized": False,
    }
    return (
        record,
        outer,
        producer,
        consumer,
        scratch,
        field_binding,
        tuple(x[0] for x in transverse),
        tuple(sorted(fields.values(), key=lambda x: x[0].root)),
        tuple(sorted(scalars.values(), key=lambda b: b.root)),
        tuple(constant_accesses),
        completion,
    )


@dataclass(frozen=True, slots=True)
class OriginalLineSnapshotProof:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _nodes: tuple = field(repr=False, compare=False)
    _node: object = field(repr=False, compare=False)
    _producer: object = field(repr=False, compare=False)
    _consumer: object = field(repr=False, compare=False)
    scratch_binding: object = field(repr=False, compare=False)
    source_binding: object = field(repr=False, compare=False)
    transverse_loops: tuple = field(repr=False, compare=False)
    fields: tuple = field(repr=False, compare=False)
    scalars: tuple = field(repr=False, compare=False)
    constant_accesses: tuple = field(repr=False, compare=False)
    _completion: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def validate(self, analysis=None, routine=None, nodes=None):
        analysis = self._analysis if analysis is None else analysis
        routine = self._routine if routine is None else routine
        nodes = self._nodes if nodes is None else tuple(nodes)
        if (
            self.identity != sha256(self._record.encode()).hexdigest()
            or self._issuer is not _ISSUER
            or analysis is not self._analysis
            or routine is not self._routine
            or len(nodes) != len(self._nodes)
            or any(a is not b for a, b in zip(nodes, self._nodes, strict=True))
            or getattr(analysis, "_line_snapshot_proofs", {}).get(self.identity) is not self
        ):
            _fail("requires its exact compiler-issued original proof")
        record, outer, producer, consumer, scratch, source, transverse, fields, scalars, constants, completion = (
            _derive(analysis, routine, nodes, completion=self._completion)
        )
        if (
            _encoded(record) != self._record
            or outer is not self._node
            or producer is not self._producer
            or consumer is not self._consumer
            or scratch is not self.scratch_binding
            or source is not self.source_binding
            or len(transverse) != len(self.transverse_loops)
            or any(a is not b for a, b in zip(transverse, self.transverse_loops, strict=True))
            or len(fields) != len(self.fields)
            or any(a is not b or x != y for (a, x), (b, y) in zip(fields, self.fields, strict=True))
            or completion is not self._completion
            or len(constants) != len(self.constant_accesses)
            or any(
                node is not old_node
                or binding is not old_binding
                or len(loops) != len(old_loops)
                or any(a is not b for a, b in zip(loops, old_loops, strict=True))
                for (node, binding, loops), (old_node, old_binding, old_loops) in zip(
                    constants, self.constant_accesses, strict=True
                )
            )
            or len(scalars) != len(self.scalars)
            or any(a is not b for a, b in zip(scalars, self.scalars, strict=True))
        ):
            _fail("original source, ordering or storage identity changed")
        return self


def prove_line_snapshot(analysis, routine, original_nodes):
    nodes = tuple(original_nodes)
    for proof in getattr(analysis, "_line_snapshot_proofs", {}).values():
        if len(nodes) == len(proof._nodes) and all(a is b for a, b in zip(nodes, proof._nodes, strict=True)):
            return proof.validate(analysis, routine, nodes)
    record, outer, producer, consumer, scratch, source, transverse, fields, scalars, constants, completion = _derive(
        analysis, routine, nodes
    )
    encoded = _encoded(record)
    proof = OriginalLineSnapshotProof(
        sha256(encoded.encode()).hexdigest(),
        analysis,
        routine,
        nodes,
        outer,
        producer,
        consumer,
        scratch,
        source,
        transverse,
        fields,
        scalars,
        constants,
        completion,
        encoded,
        _ISSUER,
    )
    if not hasattr(analysis, "_line_snapshot_proofs"):
        analysis._line_snapshot_proofs = {}
    analysis._line_snapshot_proofs[proof.identity] = proof
    return proof.validate()
