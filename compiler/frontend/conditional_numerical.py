"""Source-owned conditional numerical context for a complete original team.

The affine model remains data. This subordinate context authenticates its exact
original SECTION inventory, extraction and typed parameters. It never admits
reads or submits work; its only consumer must emit the same checked legal-domain
guard before both operations. Ordinary numerical preparation cannot consume it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from hashlib import sha256

from compiler.analysis.conditional_affine import collect_conditional_affine
from compiler.analysis.effects import mapped_prefix
from compiler.frontend.atomic_worksharing_team import AtomicWorksharingAlternative, AtomicWorksharingTeamProof, atomic_worksharing_alternatives
from compiler.frontend.numerical_preparation import (
    NumericalPreparationUnsupported, _Budget, _SCAN_LIMIT, _digest, _extraction_identity, _source_authority,
)
from compiler.ir import ExecutionPlan, ScalarType, SourceLocation

_ISSUER = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fail(message):
    raise NumericalPreparationUnsupported("conditional original numerical context: " + message)


def _shape(value):
    """Only compare an exact already-bound original unit across extractions.

    Locations differ because outlined alternatives have different source text.
    They remain authenticated by source/extraction identities, never discarded
    from the source proof. All numerical, type, symbol and layout fields remain.
    """
    if isinstance(value, SourceLocation):
        return None
    if isinstance(value, Enum):
        return (type(value).__name__, value.value)
    if is_dataclass(value):
        return (type(value).__name__, tuple((part.name, _shape(getattr(value, part.name)))
                                            for part in fields(value)))
    if isinstance(value, (tuple, list)):
        return tuple(_shape(child) for child in value)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    _fail("unsupported numerical structural comparison")


def _extraction(provider, alternative):
    from compiler.frontend import lower_source
    from compiler.scopes.region_context import original_region_context
    from compiler.scopes.regions import extract_region

    team = alternative.team
    analysis, routine, selection = team._analysis, team._routine, team._nodes
    alternative.validate(analysis, routine.qualified, selection)
    context = original_region_context(routine, selection, analysis=analysis)
    extraction = extract_region(analysis, routine, selection, preceding=context.preceding,
        following=context.following, cyclic_following=context.cyclic_following,
        atomic_alternative=alternative, immutable_inputs=provider.immutable_inputs)
    function = lower_source(extraction.source, extraction.entry,
                            source_name=str(routine.scope.path) + "#inline:" + extraction.source_identity)
    return context, extraction, function


def _parameter_record(extraction, function):
    parameters = {item.name: item for item in extraction.parameters}
    bindings = {item.root: item for item in extraction.bindings}
    result = []
    if len(parameters) != len(function.parameters):
        _fail("lowered parameter inventory differs from exact extraction")
    for symbol in function.parameters:
        parameter = parameters.get(symbol.name)
        if parameter is None or parameter.rank != symbol.rank:
            _fail("lowered parameter lost its original association")
        binding = bindings.get(parameter.resource)
        if binding is None:
            _fail("parameter lacks exact original binding")
        if not symbol.rank and symbol.dtype is not ScalarType.INTEGER:
            _fail("first slice requires original default-INTEGER numerical inputs")
        if (not symbol.rank and parameter.lower_bound_dimension is None
                and (binding.dtype, binding.kind, binding.rank) != ("integer", 4, 0)):
            _fail("conditional parameter is not original default INTEGER")
        if parameter.lower_bound_dimension is not None and (
                not parameter.runtime_lower_bound or not 1 <= parameter.lower_bound_dimension <= binding.rank):
            _fail("conditional descriptor bound lost its original axis")
        result.append({"symbol_id": symbol.id, "name": symbol.name, "dtype": symbol.dtype.value,
                       "rank": symbol.rank, "resource": parameter.resource,
                       "lower_bound_dimension": parameter.lower_bound_dimension,
                       "runtime_lower_bound": parameter.runtime_lower_bound,
                       "binding": binding.public()})
    return json.loads(_encoded(result))


def _domain_record(domain):
    if domain is None:
        return None
    raw = domain.public()
    # Keep the complete internal conflict evidence bound without repeating
    # every source-instance relation in each public prepared alternative.
    identity = sha256(_encoded(raw).encode()).hexdigest()
    return {"model_identity": identity, **{key: value for key, value in raw.items()
                                           if key != "forbidden_relations"}}


def _record_for(team, provider, extraction, function, domains, graph, identities, source, payload_identity):
    analysis, routine = team._analysis, team._routine
    budget = _Budget()
    paths = team.public()["alternatives"]
    parameters = _parameter_record(extraction, function)
    return {"version": 1, "kind": "original_conditional_affine_numerical_context",
        "team_identity": team.identity, "provider_identity": provider.identity,
        "procedure": routine.qualified, "source_graph": graph.identity,
        "selected_node_ids": list(identities), "source_identity": source,
        "source_inputs": analysis.inputs.identity(), "extraction_identity": _extraction_identity(extraction, budget), "payload_identity": payload_identity,
        "parameters": parameters,
        "paths": [{"original_units": path, "domain": _domain_record(domain)}
                  for path, domain in zip(paths, domains, strict=True)],
        "model_units": len(team.units), "source_work_budget": _SCAN_LIMIT,
        "isl_operation_budget": 1000000,
        "guard": "same checked INTEGER64 legal domain before source-read admission and device submission",
        "incoming_reads": "perfect mapped nests; one unconditional assignment; no retained loops, intrinsics or conditional operands",
        "original_integer_prework_required": True, "physical_bounds_and_alias_guards_required": True,
        "independent_section_completion_authorized": False, "definition_transfer_authorized": False,
        "execution_authorized": False, "source_read_admission_authorized": False}


@dataclass(frozen=True, slots=True)
class ConditionalNumericalContext:
    identity: str
    team: object = field(repr=False, compare=False)
    provider: object = field(repr=False, compare=False)
    _function: object = field(repr=False, compare=False)
    _extraction: object = field(repr=False, compare=False)
    _bank: object = field(repr=False, compare=False)
    _domains: tuple = field(repr=False, compare=False)
    _source_identity: str = field(repr=False, compare=False)
    _payload_identity: str = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def _payload(self):
        return (self._function, self._extraction.source, self._extraction.parameters,
                tuple(model.region for model in self._bank._models),
                tuple(None if domain is None else domain.public() for domain in self._domains))

    def validate(self, provider=None):
        analysis, routine = self.team._analysis, self.team._routine
        if (self._issuer is not _ISSUER or provider is not None and provider is not self.provider
                or getattr(analysis, "_conditional_numerical_owner", None) is not analysis
                or getattr(analysis, "_conditional_numerical_contexts", {}).get(self.identity) is not self
                or sha256(self._record.encode()).hexdigest() != self.identity):
            _fail("requires its registered exact original context")
        self.team.validate()
        self.provider.validate(analysis)
        budget = _Budget()
        _routine, _graph, _ids, source = _source_authority(analysis, routine.qualified, self.team._nodes,
                                                        self._extraction.numerical_helpers, budget)
        record = self.public()
        expected = _record_for(self.team, self.provider, self._extraction, self._function, self._domains,
                               _graph, _ids, source, self._payload_identity)
        if (_encoded(expected) != self._record or source != self._source_identity or source != record["source_identity"]
                or record["source_graph"] != _graph.identity or record["selected_node_ids"] != list(_ids)
                or record["source_inputs"] != analysis.inputs.identity() or record["procedure"] != routine.qualified
                or self._payload_identity != record["payload_identity"] or self._bank.function is not self._function
                or _digest(self._payload(), budget) != self._payload_identity
                or _extraction_identity(self._extraction, budget) != record["extraction_identity"]
                or _parameter_record(self._extraction, self._function) != record["parameters"]
                or record["team_identity"] != self.team.identity
                or record["provider_identity"] != self.provider.identity
                or len(self._domains) != len(self.team.public()["alternatives"])
                or record["paths"] != [{"original_units": path, "domain": _domain_record(domain)}
                    for path, domain in zip(self.team.public()["alternatives"], self._domains, strict=True)]):
            _fail("source, immutable model, typed parameters or legal domains changed")
        return self

    def domain(self, alternative):
        self.validate()
        if type(alternative) is not AtomicWorksharingAlternative:
            _fail("requires its exact original alternative type")
        alternative.validate(self.team._analysis, self.team._routine.qualified, self.team._nodes)
        if alternative.team is not self.team:
            _fail("alternative belongs to another original team")
        return self._domains[alternative.index]

    def prepare_plan(self, provider, alternative, extraction, function):
        """Rebind only a checked original unit, preserving its exact arithmetic."""
        from compiler.addressing import plan_addressing
        from compiler.scheduling import schedule_plan

        self.validate(provider)
        domain = self.domain(alternative)
        if domain is None:
            _fail("empty original paths retain native completion")
        path = self.team.public()["alternatives"][alternative.index]
        if (_parameter_record(extraction, function) != self.public()["parameters"]
                or _shape(function.parameters) != _shape(self._function.parameters)
                or len(function.body.statements) != len(path)):
            _fail("alternative does not share the exact typed original parameter inventory")
        regions = []
        for position, (index, loop) in enumerate(zip(path, function.body.statements, strict=True)):
            if _shape(loop) != _shape(self._function.body.statements[index]):
                _fail("alternative changes an original unit's numerical operations")
            original = self._bank._models[index].region
            mapped, body = mapped_prefix(loop)
            regions.append(replace(original, id=position, loops=tuple(mapped), body=body,
                                   assignments=tuple(body.statements)))
        plan = ExecutionPlan(tuple(regions), reports=("conditional original SECTION legality requires its exact pre-admission guard",))
        # No fusion, scalar motion or source reordering is licensed by this
        # first slice. Scheduling/addressing reuse the ordinary typed backend.
        return plan_addressing(schedule_plan(function, plan, options=provider.options), options=provider.options)


def prove_conditional_numerical_context(provider, team):
    if type(team) is not AtomicWorksharingTeamProof or not team.public().get("conditional_affine_context_required"):
        _fail("requires the exact complete original SECTION team")
    analysis, routine = team._analysis, team._routine
    provider.validate(analysis)
    team.validate()
    registry = getattr(analysis, "_conditional_numerical_contexts", None)
    if registry is None:
        registry = analysis._conditional_numerical_contexts = {}
        analysis._conditional_numerical_owner = analysis
    if getattr(analysis, "_conditional_numerical_owner", None) is not analysis:
        _fail("cannot borrow another analysis context registry")
    for token in registry.values():
        if token.team is team and token.provider is provider:
            return token.validate(provider)
    alternatives = atomic_worksharing_alternatives(team)
    complete = [item for item in alternatives if item.public()["selected_original_units"] == list(range(len(team.units)))]
    if not complete:
        _fail("first slice requires an original path containing the complete unit inventory")
    _context, extraction, function = _extraction(provider, complete[0])
    budget = _Budget()
    _routine, graph, identities, source = _source_authority(analysis, routine.qualified, team._nodes,
                                                          extraction.numerical_helpers, budget)
    parameters = _parameter_record(extraction, function)
    bank = collect_conditional_affine(function, operation_budget=_SCAN_LIMIT)
    paths = team.public()["alternatives"]
    domains = tuple(bank.project(tuple(path)) if path else None for path in paths)
    payload = (function, extraction.source, extraction.parameters, tuple(model.region for model in bank._models),
               tuple(None if domain is None else domain.public() for domain in domains))
    record = _record_for(team, provider, extraction, function, domains, graph, identities, source,
                         _digest(payload, budget))
    encoded = _encoded(record)
    token = ConditionalNumericalContext(sha256(encoded.encode()).hexdigest(), team, provider, function, extraction,
        bank, domains, source, record["payload_identity"], encoded, _ISSUER)
    if len(registry) >= analysis.operation_limit * analysis.procedure_limit:
        _fail("context registry budget exhausted")
    registry[token.identity] = token
    try:
        return token.validate(provider)
    except BaseException:
        if registry.get(token.identity) is token:
            del registry[token.identity]
        raise
