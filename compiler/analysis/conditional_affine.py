"""Bounded conditional affine models, without source or execution authority.

One exact event model supplies both parallel-independence and incoming-read
conditions. The source issuer and guarded provider must still bind this result
to the original whole team before any consumer can execute a conditional plan.
Ordinary planning never consumes this module's unaccepted regions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json

import islpy as isl

from compiler.analysis.dependence import Affine, _region_model
from compiler.analysis.effects import mapped_prefix
from compiler.analysis.semantics import validate_function
from compiler.ir import ArrayAccess, Assignment, Binary, Block, CompilationError, IntrinsicCall, Loop, ScalarType, walk_expr


class ConditionalAffineUnsupported(CompilationError):
    """Decline the entire optional conditional model, never a partial domain."""


@dataclass
class _Budget:
    remaining: int

    def charge(self, count=1):
        self.remaining -= count
        if self.remaining < 0:
            raise ConditionalAffineUnsupported("conditional affine proof effort exhausted")


@dataclass(frozen=True)
class AffineCondition:
    constant: int
    coefficients: tuple[tuple[str, int], ...]
    equality: bool

    def evaluate(self, values):
        value = self.constant + sum(coefficient * values[name] for name, coefficient in self.coefficients)
        return value == 0 if self.equality else value >= 0

    def public(self):
        return {"constant": self.constant, "coefficients": [list(row) for row in self.coefficients],
                "equality": self.equality}


@dataclass(frozen=True)
class ConditionalAffineClause:
    """One exact projected legal set in a path's conjunction."""
    alternatives: tuple[tuple[AffineCondition, ...], ...]
    legal_set: str
    forbidden_set: str

    def public(self):
        record = {"legal_set": self.legal_set, "forbidden_set": self.forbidden_set,
                  "constraints": [[condition.public() for condition in arm] for arm in self.alternatives]}
        return {"identity": sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), **record}


@dataclass(frozen=True)
class ConditionalAffineDomain:
    """Analysis data only: exact intersection, no source/execution authority."""

    parameters: tuple
    clauses: tuple[ConditionalAffineClause, ...]
    conflicts: tuple[tuple[str, str], ...]
    _models: tuple = field(repr=False, compare=False)

    @property
    def legal_set(self):
        return " AND ".join("("+clause.legal_set+")" for clause in self.clauses) or "true"

    @property
    def forbidden_set(self):
        return " OR ".join("("+clause.forbidden_set+")" for clause in self.clauses) or "false"

    def accepts(self, values):
        names = {name for name, _symbol in self.parameters}
        if set(values) != names or any(type(value) is not int or not -(2**31) <= value < 2**31
                                      for value in values.values()):
            return False
        return all(any(all(condition.evaluate(values) for condition in arm) for arm in clause.alternatives)
                   for clause in self.clauses)

    def public(self):
        return {"version": 2, "kind": "unaccepted_conditional_affine_domain",
                "parameters": [{"model_name": name, "symbol_id": symbol.id, "name": symbol.name,
                                "dtype": symbol.dtype.value} for name, symbol in self.parameters],
                "units": len(self._models),
                "guard_conjunction": [clause.public() for clause in self.clauses],
                "domain": "intersection of exact projected legal sets over original INTEGER4 inputs",
                "forbidden_relations": dict(self.conflicts),
                "source_authority": False, "execution_authorized": False,
                "incoming_read_admission_authorized": False, "alias_authorized": False,
                "requires_original_integer_prework": True,
                "requires_exact_source_binding_and_guard_before_admission": True}


def _slice(function, budget):
    validate_function(function)
    loops = function.body.statements
    if not loops:
        raise ConditionalAffineUnsupported("conditional affine model requires a nonempty original alternative")
    for loop in loops:
        budget.charge()
        if not isinstance(loop, Loop):
            raise ConditionalAffineUnsupported("conditional affine first slice requires only original counted nests")
        mapped, body = mapped_prefix(loop)
        budget.charge(len(mapped))
        if (len(body.statements) != 1 or not isinstance(body.statements[0], Assignment)
                or not isinstance(body.statements[0].target, ArrayAccess)):
            raise ConditionalAffineUnsupported("conditional affine first slice requires one unconditional array assignment per nest")
        for expression in (body.statements[0].target, body.statements[0].value):
            for node in walk_expr(expression):
                budget.charge()
                if (isinstance(node, IntrinsicCall) or isinstance(node, Binary)
                        and node.operator.lower() in {".and.", ".or.", "and", "or"}):
                    raise ConditionalAffineUnsupported("conditional affine first slice does not grant incoming reads through intrinsics or conditional operands")
    return tuple(loops)


def _renamed(model, ordinal, context, space, budget):
    """Each unit retains distinct original statement-instance tuple identities."""
    def rename(value, outputs=False):
        text = str(value)
        budget.charge(1 + len(text)//4096)
        original = isl.UnionMap.read_from_str(context, text).align_params(space)
        result = isl.UnionMap.empty(space)
        maps = original.get_map_list()
        for index in range(maps.n_map()):
            budget.charge()
            mapping = maps.get_at(index)
            name = mapping.get_tuple_name(isl.dim_type.in_)
            mapping = mapping.set_tuple_name(isl.dim_type.in_, f"U{ordinal}_{name}")
            if outputs:
                name = mapping.get_tuple_name(isl.dim_type.out)
                mapping = mapping.set_tuple_name(isl.dim_type.out, f"U{ordinal}_{name}")
            result = result.union(isl.UnionMap.from_map(mapping))
        return result
    return (rename(model.reads), rename(model.writes), rename(model.parallel),
            tuple((kind, rename(relation, outputs=True)) for kind, relation in model.conflicts))


def _conditions(domain, parameters, budget):
    arms = []
    sets = domain.get_basic_set_list()
    for index in range(sets.n_basic_set()):
        budget.charge()
        basic = sets.get_at(index)
        if basic.dim(isl.dim_type.div):
            raise ConditionalAffineUnsupported("conditional affine guard has unsupported local divisions")
        constraints = basic.get_constraint_list()
        arm = []
        for number in range(constraints.n_constraint()):
            budget.charge()
            constraint = constraints.get_at(number)
            constant = constraint.get_constant_val().to_python()
            coefficients = []
            for position, name in enumerate(parameters):
                budget.charge()
                value = constraint.get_coefficient_val(isl.dim_type.param, position).to_python()
                if value:
                    coefficients.append((name, value))
            if any(type(value) is not int or not -(2**63) <= value < 2**63
                   for value in [constant, *(coefficient for _name, coefficient in coefficients)]):
                raise ConditionalAffineUnsupported("conditional affine guard coefficient exceeds checked integer representation")
            arm.append(AffineCondition(constant, tuple(coefficients), constraint.is_equality()))
        arms.append(tuple(arm))
    return tuple(arms)


@dataclass(frozen=True)
class ConditionalAffineModels:
    """Unaccepted unit models sharing source-work and private ISL limits."""

    function: object = field(repr=False, compare=False)
    _models: tuple = field(repr=False, compare=False)
    _context: object = field(repr=False, compare=False)
    _budget: object = field(repr=False, compare=False)
    _space: object = field(repr=False, compare=False)
    _parameters: tuple = field(repr=False, compare=False)
    _unit_cache: dict = field(default_factory=dict, repr=False, compare=False)
    _pair_cache: dict = field(default_factory=dict, repr=False, compare=False)
    _path_cache: dict = field(default_factory=dict, repr=False, compare=False)
    _guard_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def project(self, indices):
        if (not isinstance(indices, tuple) or not indices
                or any(type(index) is not int or not 0 <= index < len(self._models) for index in indices)
                or tuple(sorted(set(indices))) != indices):
            raise ConditionalAffineUnsupported("conditional affine path requires distinct original units in source order")
        self._budget.charge(len(indices))
        if indices not in self._path_cache:
            self._path_cache[indices] = _project(self, indices)
        return self._path_cache[indices]


def _unit_model(bank, index):
    bank._budget.charge()
    if index not in bank._unit_cache:
        reads, writes, parallel, conflicts = _renamed(bank._models[index], index, bank._context,
                                                     bank._space, bank._budget)
        same = parallel.apply_range(parallel.reverse())
        forbidden, relations = isl.Set.empty(bank._space), []
        for kind, relation in conflicts:
            bank._budget.charge()
            exact = relation.subtract(same)
            forbidden = forbidden.union(exact.params())
            relations.append((f"unit{index}:{kind}", str(exact)))
        bank._unit_cache[index] = reads, writes, forbidden.coalesce(), tuple(relations)
    return bank._unit_cache[index]


def _pair_model(bank, first, second):
    bank._budget.charge()
    key = first, second
    if key not in bank._pair_cache:
        reads, writes, *_ = _unit_model(bank, first)
        other_reads, other_writes, *_ = _unit_model(bank, second)
        forbidden, relations = isl.Set.empty(bank._space), []
        for kind, relation in (("RAW", writes.apply_range(other_reads.reverse())),
                               ("WAR", reads.apply_range(other_writes.reverse())),
                               ("WAW", writes.apply_range(other_writes.reverse()))):
            bank._budget.charge()
            forbidden = forbidden.union(relation.params())
            relations.append((f"unit{first}->unit{second}:{kind}", str(relation)))
        bank._pair_cache[key] = forbidden.coalesce(), tuple(relations)
    return bank._pair_cache[key]


def _guard_model(bank, forbidden):
    key = str(forbidden)
    bank._budget.charge(1 + len(key)//4096)
    if key not in bank._guard_cache:
        if forbidden.is_empty():
            result = None
        else:
            # Complement each distinct projected predicate once. A path is
            # their conjunction (De Morgan), never a new coalesced crossproduct.
            # INTEGER4 input validity stays in accepts()/original ABI/prework.
            legal = forbidden.complement().coalesce()
            arms = _conditions(legal, [name for name, _symbol in bank._parameters], bank._budget)
            result = ConditionalAffineClause(arms, str(legal), key)
        bank._guard_cache[key] = result
    return key, bank._guard_cache[key]


def _project(bank, indices):
    models = tuple(bank._models[index] for index in indices)
    relations, clauses, seen = [], [], set()
    try:
        def add(forbidden):
            key, clause = _guard_model(bank, forbidden)
            if key not in seen:
                seen.add(key)
                if clause is not None:
                    clauses.append(clause)
        for position, index in enumerate(indices):
            _reads, _writes, bad, records = _unit_model(bank, index)
            add(bad)
            relations.extend(records)
            for other in indices[position + 1:]:
                bad, records = _pair_model(bank, index, other)
                add(bad)
                relations.extend(records)
        return ConditionalAffineDomain(bank._parameters, tuple(clauses), tuple(relations), tuple(models))
    except isl.Error as error:
        raise ConditionalAffineUnsupported("conditional affine exact projection unavailable: " + str(error)) from error


def collect_conditional_affine(function, *, operation_budget=4096, isl_operation_budget=1000000):
    """Collect each exact unit once; all later paths share one bounded context."""
    if (type(operation_budget) is not int or operation_budget <= 0
            or type(isl_operation_budget) is not int or isl_operation_budget <= 0):
        raise ConditionalAffineUnsupported("conditional affine budgets must be positive integers")
    budget = _Budget(operation_budget)
    loops = _slice(function, budget)
    environment = {symbol: Affine(terms=((f"p{symbol.id}", 1),)) for symbol in function.parameters
                   if not symbol.rank and symbol.dtype is ScalarType.INTEGER}
    defined = {symbol for symbol in function.parameters if not symbol.rank}
    # A private ISL context bounds projection work and cannot change the
    # operation limits/error state of ordinary unconditional compiler queries.
    context = isl.Context()
    context.set_max_operations(isl_operation_budget)
    try:
        models = []
        for ordinal, loop in enumerate(loops):
            budget.charge()
            model = _region_model(ordinal, loop, environment, defined, Block(loops[ordinal + 1:]),
                                  context=context, consume=budget.charge)
            if not model.physical_affine:
                raise ConditionalAffineUnsupported("conditional affine model rejects conservative or surrogate access maps")
            models.append(model)
        names = sorted({model.reads.get_space().get_dim_name(isl.dim_type.param, index)
                        for model in models for index in range(model.reads.dim(isl.dim_type.param))}
                       | {model.writes.get_space().get_dim_name(isl.dim_type.param, index)
                          for model in models for index in range(model.writes.dim(isl.dim_type.param))})
        parameters = {f"p{symbol.id}": symbol for symbol in environment}
        if any(name not in parameters for name in names):
            raise ConditionalAffineUnsupported("conditional affine guard requires original default-INTEGER scalar parameters")
        space = isl.Space.params_alloc(context, len(names))
        for index, name in enumerate(names):
            budget.charge()
            space = space.set_dim_name(isl.dim_type.param, index, name)
        return ConditionalAffineModels(function, tuple(models), context, budget, space,
                                       tuple((name, parameters[name]) for name in names))
    except isl.Error as error:
        raise ConditionalAffineUnsupported("conditional affine exact model unavailable: " + str(error)) from error


def analyze_conditional_affine(function, *, operation_budget=4096, isl_operation_budget=1000000):
    models = collect_conditional_affine(function, operation_budget=operation_budget,
                                       isl_operation_budget=isl_operation_budget)
    return models.project(tuple(range(len(models._models))))
