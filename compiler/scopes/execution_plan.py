"""One checked scan result for placement, emission and discovery.

This is the owning interval's ordered program, not another source analyser.
The existing source, numerical, definition and completion proofs admit its
units. A plan retains those units and their original control structure once;
placement and emission cannot silently select different inventories. Public
JSON is a diagnostic view and cannot be imported as execution authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
from types import MappingProxyType, SimpleNamespace

from compiler.ir import CompilationError

VERSION = 1
_ISSUER = object()


class _ProjectedProvenance:
    """Seal synthetic facade spans without changing original parser items."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return sha256(_json(value).encode()).hexdigest()


def _configuration(builder):
    from compiler.frontend.numerical_preparation import _Budget
    from compiler.frontend.numerical_preparation import _configuration as configuration

    # Observations select an already defined alternative. They cannot change
    # its identity, otherwise a calibration could never select that same plan.
    config = replace(builder.config, exact_window=None, scope_measurement=None)
    return configuration(builder.options, config, builder._numerical_environment_root,
                         getattr(builder, "immutable_inputs", {}), _Budget())


def _portable_numerical_references(analysis):
    """Describe registered numerical work without exporting Python identities.

    Tokens still authenticate exact parser/binding objects in this process.
    Calibration instead binds the source selection, checked numerical payload,
    context and policy. Only known registered token references are substituted;
    arbitrary diagnostic fields and emitted source text are never discarded.
    """
    references = {}

    def portable(value):
        if isinstance(value, str):
            return references.get(value, value)
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [portable(item) for item in value]
        return value

    for proof in getattr(analysis, "_numerical_preparations", {}).values():
        public = proof.public()
        public.pop("preparation_identity")
        references[proof.identity] = _digest({
            "portable_numerical_preparation_version": 1,
            "preparation": public,
            "extraction": proof.extraction.public(),
            "extracted_source": proof.extraction.source,
            "checked_numerical_payload_identity": proof._payload_identity,
            "original_context_identity": proof._context_identity,
        })
    for proof in getattr(analysis, "_numerical_effect_proofs", {}).values():
        public = proof.public()
        public.pop("proof_identity")
        references[proof.identity] = _digest({"portable_numerical_effect_version": 1,
                                               "effects": portable(public)})
    return portable


def _unit_row(member, unit, ordinal):
    originals = member.builder.inline.original_selection(unit.nodes)
    graph, selected = member.builder.analysis._selected_source(member.routine.qualified, originals)
    if graph is not member.graph:
        # Equal serialized graphs do not grant a foreign source object access.
        raise CompilationError("execution plan requires the retained original source graph")
    return {
        "ordinal": ordinal,
        "selected_node_ids": list(selected),
        "first_line": unit.scope.first,
        "last_line": unit.scope.last,
        "resources": {root: member.canonical(binding).root for root, binding in unit.arrays.items()},
        "ordered_definitions": unit.definition_summary.public(),
        "operations": unit.scope.public(),
    }


def _registered_authorities(owner, owned_types):
    """Retain only existing source/proof authorities from execution services.

    This classifies graph ownership; existing consumers still validate each
    proof with its original source selection and analysis. A method called
    ``validate`` cannot turn arbitrary frontend metadata into an authority.
    """
    from fparser.two.utils import Base

    from compiler.frontend.descriptor_windows import DescriptorWindowProof
    from compiler.frontend.native_call_coordinates import NativeCallCoordinatesProof
    from compiler.frontend.native_completion import NativeCompletionProof
    from compiler.frontend.native_definitions import NativeDefinitionProof
    from compiler.frontend.native_demand import SourceBoundNativeDemandEffects
    from compiler.frontend.native_environment import NativeEnvironmentProof, NativeLocalStateProof
    from compiler.frontend.native_intrinsics import NativePredicateProof
    from compiler.frontend.native_prefix import NativePrefixProof
    from compiler.frontend.numerical_completion import NumericalCompletionProof
    from compiler.frontend.numerical_effects import NumericalEffectProof
    from compiler.frontend.numerical_preparation import NumericalPreparationProvider, PreparedOriginalRegion
    from compiler.frontend.omp_reductions import OpenMPReductionProof
    from compiler.frontend.reached_windows import ReachedWindowProof
    from compiler.frontend.reductions import ReductionProof
    from compiler.frontend.source_effects import SourceEffects
    from compiler.frontend.source_functions import NativeSourceFunctionProof
    from compiler.frontend.structured_effects import StructuredSummary
    from compiler.frontend.worksharing_completion import WorksharingCompletionProof, WorksharingNativeCompletionProof
    from compiler.scopes.lexical import OwnerEmission
    from compiler.scopes.resource_demand import IntervalResourceDemand
    from compiler.scopes.source import ScopeBuilder

    registries = {
        DescriptorWindowProof: "_descriptor_window_proofs",
        NativeCallCoordinatesProof: "_native_definition_proofs",
        NativeCompletionProof: "_joined_completions",
        NativeDefinitionProof: "_native_definition_proofs",
        SourceBoundNativeDemandEffects: "_native_demand_effects",
        NativeEnvironmentProof: "_native_environments",
        NativeLocalStateProof: "_native_environment_states",
        NativePredicateProof: "_native_predicates",
        NativePrefixProof: "_native_prefix_proofs",
        NumericalCompletionProof: "_numerical_completions",
        NumericalEffectProof: "_numerical_effect_proofs",
        NumericalPreparationProvider: "_numerical_preparation_providers",
        PreparedOriginalRegion: "_numerical_preparations",
        OpenMPReductionProof: "_omp_reduction_proofs",
        ReachedWindowProof: "_reached_window_proofs",
        ReductionProof: "_reduction_proofs",
        NativeSourceFunctionProof: "_native_functions",
        WorksharingCompletionProof: "_worksharing_completions",
        WorksharingNativeCompletionProof: "_worksharing_native_completions",
    }
    session = owner.builder.analysis._source_session
    pending, seen, analyses, builders = [owner], set(), {}, {}
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, SourceEffects):
            analyses[id(value)] = value
            origin = getattr(value, "_descriptor_source", None)
            if origin is not None:
                pending.append(origin)
        elif isinstance(value, ScopeBuilder):
            builders[id(value)] = value
            pending.append(value.analysis)
        elif (isinstance(value, (Base, OwnerEmission, StructuredSummary))
              or type(value) in registries or type(value) is IntervalResourceDemand
              or session is not None and session.contains(value)):
            continue
        elif isinstance(value, (dict, MappingProxyType)):
            pending.extend(value.values())
            pending.extend(value.keys())
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
        elif isinstance(value, owned_types) or is_dataclass(value):
            pending.extend(vars(value).values() if hasattr(value, "__dict__") else
                           (getattr(value, part.name) for part in fields(value)))
    authorities = {id(member.graph): member.graph for member in owner.members.values()}
    for builder in builders.values():
        for value in getattr(builder, "_resource_demands", {}).values():
            if (type(value) is IntervalResourceDemand
                    and builder._resource_demands.get(value.identity) is value):
                authorities[id(value)] = value
    for analysis in analyses.values():
        for kind, registry in registries.items():
            for value in getattr(analysis, registry, {}).values():
                key = id(value) if kind is NumericalPreparationProvider else getattr(value, "identity", None)
                if type(value) is kind and getattr(analysis, registry).get(key) is value:
                    authorities[id(value)] = value
        for graph in analysis._structures.values():
            if type(graph) is StructuredSummary:
                authorities[id(graph)] = graph
    return authorities, (*registries, StructuredSummary, IntervalResourceDemand)


def _seal_execution(owner):
    """Freeze admitted data; builders and registered proof services stay separate.

    Every field on an owned object participates automatically. An unknown
    mutable object is an error, rather than a silent hole in a field whitelist.
    Source/proof authorities retain their existing identities and validation.
    """
    from fparser.two.utils import Base, walk

    from compiler.frontend.immutable_graph import GraphFreezer, original_type
    from compiler.frontend.sealed_syntax import seal as seal_syntax
    from compiler.frontend.source_effects import Binding, Routine, Scope, SourceEffects
    from compiler.scopes.lexical import LexicalOwner, OwnerEmission, Unit
    from compiler.scopes.procedures import BorrowedCoordinator
    from compiler.scopes.segments import Association, Branch, Delegated, Native, Segment, StructuredScope
    from compiler.scopes.source import Call, ScopeBuilder

    session = owner.builder.analysis._source_session
    owned_types = (LexicalOwner, Unit, StructuredScope, Native, Segment, Branch, Association,
                   Delegated, Call, BorrowedCoordinator)
    authorities, authority_types = _registered_authorities(owner, owned_types)
    syntax = {}
    portable_numerical = _portable_numerical_references(owner.builder.analysis)
    # Prepared extractions belong to registered numerical authority. They share
    # identity-sensitive completion/precondition objects with their proof;
    # changing those aliases would invalidate rather than strengthen the proof.
    extractions = {id(proof.extraction): proof for proof in
                   getattr(owner.builder.analysis, "_numerical_preparations", {}).values()}
    numerical_authorities = {}

    def borrowed(value):
        if isinstance(value, Base):
            descendants = tuple(walk(value))
            for node in descendants:
                item = getattr(node, "item", None)
                if type(item) is SimpleNamespace:
                    replacement = _ProjectedProvenance()
                    vars(replacement).update(vars(item))
                    node.item = replacement
            seal_syntax((value,))
            syntax.update((id(node), node) for node in descendants)
            return True
        if session is not None and session.contains(value):
            return True
        if id(value) in extractions and extractions[id(value)].extraction is value:
            proof = extractions[id(value)]
            numerical_authorities[proof.identity] = proof
            return True
        if authorities.get(id(value)) is value:
            return True
        if type(value) in authority_types:
            raise CompilationError("execution plan requires an exact registered source/proof authority")
        return False

    external = (ScopeBuilder, SourceEffects, OwnerEmission)
    freezer = GraphFreezer(
        owned_types=owned_types,
        own_if=is_dataclass, borrowed_types=external, borrow_if=borrowed)
    freezer.freeze(owner)

    def record(root):
        seen = {}

        def visit(value):
            if value is None or isinstance(value, (str, bool, float)):
                return value
            if isinstance(value, int):
                return value
            if isinstance(value, Base):
                return {"syntax": type(value).__name__, "text": str(value),
                        "span": getattr(getattr(value, "item", None), "span", None)}
            if isinstance(value, ScopeBuilder):
                return {"service": "builder", "entry": value.entry.qualified}
            if isinstance(value, SourceEffects):
                return {"service": "source_analysis", "sources": dict(value.sources)}
            if isinstance(value, OwnerEmission):
                return {"result": "separate emission metadata"}
            if isinstance(value, Binding) and session is not None and session.contains(value):
                return {"binding": value.public()}
            if isinstance(value, (Routine, Scope)) and session is not None and session.contains(value):
                scope = value.scope if isinstance(value, Routine) else value
                return {"source_role": getattr(value, "qualified", scope.module), "path": str(scope.path)}
            if id(value) in extractions and extractions[id(value)].extraction is value:
                return {"authority": "registered numerical extraction",
                        "preparation_identity": portable_numerical(extractions[id(value)].identity),
                        "extraction": portable_numerical(value.public())}
            if borrowed(value):
                return {"authority": original_type(value).__name__, "proof": portable_numerical(value.public())}
            if id(value) in seen:
                return {"reference": seen[id(value)]}
            seen[id(value)] = len(seen)
            if isinstance(value, (dict, MappingProxyType)):
                # Guard and loop indexes use parser identities internally;
                # their stable source syntax is the portable key.
                pairs = [(visit(syntax[key]) if type(key) is int and key in syntax else visit(key), item)
                         for key, item in value.items()]
                return {"map": [[key, visit(item)] for key, item in sorted(pairs, key=lambda pair: _json(pair[0]))]}
            if isinstance(value, (list, tuple)):
                return [visit(item) for item in value]
            if isinstance(value, (set, frozenset)):
                return {"set": sorted((visit(item) for item in value), key=_json)}
            if freezer.objects.get(id(value)) is value:
                return {"type": original_type(value).__module__ + "." + original_type(value).__name__,
                        "fields": {name: visit(item) for name, item in sorted(value.__getstate__().items())
                                   if name != "execution_plan"}}
            # Paths/enums are immutable values admitted by the generic helper.
            from enum import Enum
            from pathlib import PurePath
            if isinstance(value, (Enum, PurePath)):
                return {"type": type(value).__name__, "value": str(value)}
            raise CompilationError("execution plan cannot identify admitted metadata: " + type(value).__name__)

        return _digest(visit(root))

    return ({member.routine.qualified: record(member) for member in owner.members.values()},
            tuple(numerical_authorities.values()))


def _cost_requirements(members):
    requirements = []
    for member in members:
        for ordinal, unit in enumerate(member.units):
            kinds = set()
            for operation in unit.scope.native:
                if operation.host_only_reads:
                    kinds.add("host_only_ranges")
                if operation.native_environment:
                    kinds.add("native_environment")
                if getattr(operation, "indirect_sections", None) is not None:
                    kinds.add("indirect_inspector")
                if getattr(operation, "demand_effects", None) is not None:
                    kinds.add("source_native_observations")
            if any(call.region is not None and call.region.scalar_element_captures for call in unit.scope.calls):
                kinds.add("immutable_elements")
            reasons = {
                "host_only_ranges": "host-only native range preparation lacks compatible offline cost estimates",
                "native_environment": "original IEEE environment coordination lacks compatible offline cost estimates",
                "immutable_elements": "immutable element preparation lacks compatible offline cost estimates",
                "indirect_inspector": "native indirect inspector preparation lacks compatible offline cost estimates",
                "source_native_observations": "native interval costs require compatible source-bound observations",
            }
            for kind in reasons:
                if kind in kinds:
                    requirements.append({"kind": kind, "reason": reasons[kind],
                                         "procedure": member.routine.qualified, "unit": ordinal})
    return requirements


def _invocations(owner, members):
    """Count original call sites, including repeated mapped child invocations."""
    direct = {member.routine.qualified: sum(bool(member.builder.closure(call.procedure)[0])
              for unit in member.units for call in unit.scope.calls) for member in members}
    if not any(getattr(member.builder, "_mapped_reached_child", None) is not None for member in members):
        return sum(direct.values())
    edges, seen, visits = {}, set(), 0
    for member in members:
        for site in getattr(member, "call_sites", ()):
            identity = site.get("mapped_call_node_id")
            if identity is None:
                continue
            caller = owner.members.get(site["caller"])
            span = site["first_line"], site["last_line"]
            if caller is None or getattr(caller, "mapped_call_patches", {}).get(span) != identity:
                raise CompilationError("numerical invocation count lacks its retained original call site")
            key = site["caller"], identity
            if key in seen:
                raise CompilationError("numerical invocation count repeats one original call site")
            seen.add(key)
            visits += 1
            if visits > owner.builder.analysis.procedure_limit:
                raise CompilationError("numerical invocation call traversal budget exhausted")
            edges.setdefault(site["caller"], []).append(member.routine.qualified)
    counts = {}

    def count(name, active=()):
        if name in active or len(active) >= owner.builder.analysis.depth_limit:
            raise CompilationError("numerical invocation call depth or recursion budget exhausted")
        if name not in counts:
            counts[name] = direct[name] + sum(count(child, (*active, name)) for child in edges.get(name, ()))
        return counts[name]

    return count(owner.routine.qualified)


@dataclass(frozen=True, slots=True)
class _ProcedurePlan:
    member: object
    units: tuple
    patches: tuple
    record: str


@dataclass(frozen=True, slots=True)
class CheckedExecutionPlan:
    identity: str
    execution_identity: str
    numerical_invocations: int
    leaves: frozenset
    _owner: object
    _procedures: tuple
    _contexts: tuple
    _numerical_authorities: tuple
    _record: str
    _issuer: object

    @property
    def members(self):
        return tuple(item.member for item in self._procedures)

    def units(self, member):
        self._require_owner()
        for item in self._procedures:
            if item.member is member:
                return item.units
        raise CompilationError("execution plan does not own this procedure")

    def _require_owner(self):
        if self._issuer is not _ISSUER or self._owner.execution_plan is not self:
            raise CompilationError("execution plan requires its issuing lexical owner")

    def patches(self, member):
        self._require_owner()
        for item in self._procedures:
            if item.member is member:
                return item.patches
        raise CompilationError("execution plan does not own this procedure")

    def validate(self):
        """One emission checkpoint; never another discovery or proof search."""
        self._require_owner()
        self.exact_window_placement()
        for builder, analysis, routine, configuration, facts in self._contexts:
            if (builder.analysis is not analysis or builder.entry is not routine
                    or _configuration(builder) != configuration or _digest(builder.facts) != facts):
                raise CompilationError("execution plan source context, configuration or capture facts changed after placement")
            analysis.verify_publication()
        for proof in self._numerical_authorities:
            proof.validate(proof._provider._analysis, proof.procedure, proof._selection)
        return self

    def operations(self, member, ordinal):
        """Detached diagnostic view of exactly the planned operation tree."""
        self._require_owner()
        for item in self._procedures:
            if item.member is member:
                return json.loads(item.record)["units"][ordinal]["operations"]
        raise CompilationError("execution plan does not own this procedure")

    def exact_window_placement(self):
        """Return only the exact issuer-registered attachment for this plan."""
        self._require_owner()
        from compiler.offload.exact_window import ExactWindowPlacement

        placement = self._owner.emission.exact_window
        if placement is None:
            if self.identity in getattr(self._owner.builder, "_exact_window_placements", {}):
                raise CompilationError("checked execution plan lost its issued placement attachment")
            return None
        if type(placement) is not ExactWindowPlacement:
            raise CompilationError("checked execution plan requires an issued exact-window placement")
        return placement.validate(self)

    def resource_demand(self):
        """Share the sealed whole-interval inventory with native consumers.

        This inventory includes later numerical uses and their dependencies.
        It supplies no effects, alias, completion, or placement authority.
        """
        self._require_owner()
        from compiler.scopes.resource_demand import IntervalResourceDemand, resource_demand_from_plan

        for demand in getattr(self._owner.builder, "_resource_demands", {}).values():
            if type(demand) is not IntervalResourceDemand:
                raise CompilationError("execution plan requires an issued interval resource demand")
            if demand._plan is self:
                return demand.validate()
        return resource_demand_from_plan(self)

    def automatic_native_reason(self):
        placement = self.exact_window_placement()
        if self._owner.builder.config.policy != "auto":
            return None
        if placement is not None:
            return placement.native_reason
        reasons = self.public()["cost_requirements"]
        # Keep the existing conservative policy. A complete-window observation
        # can cover these costs only after its independent admission checks.
        if reasons:
            return reasons[-1]["reason"]
        return self._owner.builder.automatic_native_preflight(self.leaves)

    def public(self):
        self._require_owner()
        return json.loads(self._record)


def _procedure_row(member, units, emission_identity):
    path = str(member.routine.scope.path)
    return {
        "procedure": member.routine.qualified,
        "source": path,
        "source_sha256": member.builder.analysis.sources[path],
        "structured_summary_identity": member.graph.identity,
        "emission_identity": emission_identity,
        "units": [_unit_row(member, unit, ordinal) for ordinal, unit in enumerate(units)],
        "boundaries": member.boundaries,
        "calls": [*member.internal_calls, *getattr(member, "call_sites", ())],
    }


def build_execution_plan(owner):
    """Seal an admitted scan, after exact source-position validation."""
    if owner.parent is not None or owner.active:
        raise CompilationError("execution plan requires a completed owning source scan")
    members = tuple(owner.members.values())
    if not members or members[0] is not owner:
        raise CompilationError("execution plan requires its original root procedure")
    emission_identities, numerical_authorities = _seal_execution(owner)
    procedures = []
    for member in members:
        units = tuple(member.units)
        procedures.append(_ProcedurePlan(member, units, tuple(member.patches),
                                         _json(_procedure_row(member, units, emission_identities[member.routine.qualified]))))
    leaves = frozenset(leaf for member in members for unit in member.units for leaf in unit.leaves)
    config = _configuration(owner.builder)
    facts = _digest(owner.builder.facts)
    invocations = _invocations(owner, members)
    record = {
        "schema_version": VERSION,
        "entry": owner.routine.qualified,
        "sources": dict(owner.builder.analysis.sources),
        "source_inputs_identity": _digest(owner.builder.analysis.inputs.identity()),
        "configuration_identity": config,
        "capture_facts_identity": facts,
        "procedures": [json.loads(item.record) for item in procedures],
        "resources": [{"resource": root, "registration_identity": number, "descriptor": binding.public()}
                      for root, (binding, number) in owner.resources.items()],
        "numerical_invocations": invocations,
        "gpu_leaves": sorted(leaves),
        "policy": {
            "order": "original source order within each procedure; original call sites and guards retained",
            "lifetime": "one-shot reached interval" if hasattr(owner, "window_proof") else "original invocation",
            "guards": "original reached evaluation points; no hoisting across native effects",
            "fallback": "publish and continue the unexecuted suffix once; execution failure never authorizes replay",
            "authority": "existing source, numerical, definition, native-effect and completion proofs",
            "public_record": "diagnostics only; cannot be imported as execution authority",
            "transfers": {"prediction": "unavailable until compatible placement estimates and runtime sections",
                          "observed": False,
                          "coverage_required": ["entry", "native interruptions", "exit", "cleanup"]},
        },
        "cost_requirements": _cost_requirements(members),
    }
    # The diagnostic record retains exact per-compilation proof identities.
    # The executable fingerprint binds the same immutable operation/control
    # graph through its semantic digests, using portable numerical references.
    # Current complete-window admission excludes reached-window names derived
    # from private proof tokens; no portability claim is made for that path.
    execution_identity = _digest({
        "execution_identity_version": 1,
        "entry": record["entry"], "sources": record["sources"],
        "source_inputs_identity": record["source_inputs_identity"],
        "configuration_identity": config, "capture_facts_identity": facts,
        "procedures": [{"procedure": member.routine.qualified,
                        "emission_identity": emission_identities[member.routine.qualified]} for member in members],
        "resources": record["resources"], "numerical_invocations": invocations,
        "gpu_leaves": record["gpu_leaves"],
    })
    record["execution_identity"] = execution_identity
    identity = _digest(record)
    record["identity"] = identity
    contexts = tuple((member.builder, member.builder.analysis, member.routine, _configuration(member.builder),
                      _digest(member.builder.facts)) for member in members)
    result = CheckedExecutionPlan(identity, execution_identity, invocations, leaves, owner, tuple(procedures), contexts,
                                  numerical_authorities, _json(record), _ISSUER)
    # The only publication of authority occurs after the entire graph is sealed.
    object.__setattr__(owner, "execution_plan", result)
    # A native proof made against a prospective reached inventory must cover
    # every final member, including resources first used by later workers.
    # Validate before any source emission, then retain one inventory for the
    # complete checked interval. A prefix inventory cannot grant admission.
    from compiler.scopes.resource_demand import IntervalResourceDemand

    for demand in tuple(getattr(owner.builder, "_resource_demands", {}).values()):
        if type(demand) is not IntervalResourceDemand:
            raise CompilationError("execution plan requires an issued interval resource demand")
        if demand._plan is None:
            demand.validate_plan(result)
    result.resource_demand()
    return result
