"""Registered, option-bound preparation of original numerical source.

Preparation performs extraction, lowering and physical analysis only. It never
captures storage, emits a facade, reserves an artifact, or authorizes execution.
The ordinary outlining consumer still checks storage and participation first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from hashlib import sha256

from compiler.ir import CompilationError

NUMERICAL_PREPARATION_VERSION = 2
_SCAN_LIMIT = 65536
_IDENTITY_TEXT_LIMIT = 4 * 1024 * 1024
_PROFILE_TEXT_LIMIT = 32 * 1024 * 1024


class NumericalPreparationUnsupported(CompilationError):
    """A checked numerical support limit, never source or budget authority."""


class _Budget:
    def __init__(self, consume=None):
        self.consume, self.visits = consume, 0

    def tick(self, _role=None):
        self.visits += 1
        if self.consume is not None:
            self.consume(1)
        if self.visits > _SCAN_LIMIT:
            raise CompilationError("original numerical preparation source-scan budget exhausted")


def _digest(value, budget):
    """Bounded structural fingerprint, including normally non-comparing fields."""
    memo, active = {}, set()

    def visit(item, depth=0):
        budget.tick()
        if depth > 256:
            raise CompilationError("original numerical preparation identity depth exhausted")
        if item is None or isinstance(item, (bool, int, float, str)):
            if isinstance(item, str) and len(item) > _IDENTITY_TEXT_LIMIT:
                raise CompilationError("original numerical preparation identity text budget exhausted")
            return item
        if isinstance(item, Enum):
            return [type(item).__module__, type(item).__name__, item.value]
        key = id(item)
        if key in active:
            raise CompilationError("original numerical preparation identity is cyclic")
        if key in memo:
            return {"shared_digest": memo[key]}
        active.add(key)
        if is_dataclass(item) and not isinstance(item, type):
            result = [type(item).__module__, type(item).__name__,
                      [[part.name, visit(getattr(item, part.name), depth + 1)] for part in fields(item)]]
        elif isinstance(item, dict):
            if any(not isinstance(key, (str, int)) for key in item):
                raise CompilationError("original numerical preparation identity has unsupported map keys")
            result = [[str(key), visit(child, depth + 1)] for key, child in sorted(item.items(), key=lambda pair: str(pair[0]))]
        elif isinstance(item, (tuple, list, set, frozenset)):
            result = [visit(child, depth + 1) for child in item]
            if isinstance(item, (set, frozenset)):
                result.sort(key=lambda child: json.dumps(child, sort_keys=True))
        else:
            raise CompilationError("unsupported original numerical preparation identity value: " + type(item).__name__)
        active.remove(key)
        digest = sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        memo[key] = digest
        return {"shared_digest": digest}

    try:
        return sha256(json.dumps(visit(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    except (ValueError, TypeError, OverflowError) as error:
        raise CompilationError("invalid original numerical preparation identity") from error


def _registry(analysis, name):
    value = getattr(analysis, name, None)
    if value is None:
        value = {}
        setattr(analysis, name, value)
    return value


def _profile_identity(profile, budget):
    """Hash retained calibration data separately from executable source nodes."""
    if profile is None:
        return None
    digest, size, charged = sha256(), 0, 0
    budget.tick()
    try:
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
        for chunk in encoder.iterencode(profile):
            encoded = chunk.encode("ascii")
            size += len(encoded)
            if size > _PROFILE_TEXT_LIMIT:
                raise CompilationError("original numerical preparation profile text budget exhausted")
            blocks = size // 4096
            while charged < blocks:
                budget.tick()
                charged += 1
            digest.update(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise CompilationError("invalid original numerical preparation profile identity") from error
    return {"private_identity_kind": "calibration_profile_sha256_v1", "bytes": size, "sha256": digest.hexdigest()}


def _configuration(options, config, environment_root, immutable_inputs, budget):
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    if type(options) is not CompilerOptions or type(config) is not OffloadConfig:
        raise CompilationError("original numerical preparation requires validated compiler options and offload configuration")
    if not isinstance(environment_root, str):
        raise CompilationError("original numerical preparation requires an original environment root")
    immutable = []
    for resource, token in sorted(immutable_inputs.items()):
        budget.tick()
        if not isinstance(resource, str) or not hasattr(token, "validate") or not isinstance(getattr(token, "identity", None), str):
            raise CompilationError("original numerical preparation requires typed immutable-input associations")
        immutable.append((resource, token.identity))
    # Observations remain fully authenticated on every demand, but their
    # JSON fields are data, not additional executable statements. No profile
    # acceptance flag or retained hash can substitute for this fresh digest.
    profile = _profile_identity(config.profile, budget)
    # Complete-window observations choose between already admitted alternatives
    # at the owning interval. They are not an input to numerical lowering or
    # effects. Including them here would change every prepared proof when a
    # calibration is later consumed for that exact same checked program.
    # Diagnostic caller timing likewise observes the same admitted program.
    structural = _digest((options, replace(config, profile=None, exact_window=None, scope_measurement=None),
                          environment_root, tuple(immutable)), budget)
    # Budget consumers may run compiler callbacks. Rehash after their final
    # invocation without external callbacks, including previously encoded rows.
    if _profile_identity(config.profile, _Budget()) != profile:
        raise CompilationError("calibration profile changed while accounting original numerical preparation")
    return _digest((structural, profile), _Budget())


def _walk(node, budget):
    from fparser.two.utils import Base
    stack = [iter((node,))]
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        budget.tick()
        if isinstance(item, (tuple, list)):
            stack.append(iter(item))
        elif isinstance(item, Base):
            yield item
            stack.append(iter(getattr(item, "content", getattr(item, "items", ())) or ()))


def _source_calls(node, budget):
    """Visit original CALL statements without expanding expression payloads.

    Lexical constructs, including labeled/shared DO terminals, use content.
    A single-line IF is the only item-based statement containing an action;
    WHERE and FORALL actions are assignments, and expressions cannot hold CALL.
    """
    stack = [iter((node,))]
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        budget.tick()
        if isinstance(item, (tuple, list)):
            stack.append(iter(item))
        elif type(item).__name__ == "Call_Stmt":
            yield item
        elif type(item).__name__ == "If_Stmt":
            stack.append(iter((item.items[1],)))
        else:
            content = getattr(item, "content", None)
            if content is not None:
                stack.append(iter(content))


def _input_identity(analysis, budget):
    """Fingerprint line-map data without charging it as executable syntax."""
    session = getattr(analysis, "_source_session", None)
    if session is not None:
        session.verify(analysis)
        budget.tick()
        return {"compilation_session_version": 1, "identity": session.identity}
    identity = analysis.inputs.identity()
    for entry in identity["entries"]:
        budget.tick()
        mapping = entry.get("line_map") if isinstance(entry, dict) else None
        if type(mapping) is not list:
            raise CompilationError("original numerical preparation line map requires an INTEGER-or-null list")
        if len(mapping) > _IDENTITY_TEXT_LIMIT // 2:
            raise CompilationError("original numerical preparation line-map text budget exhausted")
        if any(value is not None and type(value) is not int for value in mapping):
            raise CompilationError("original numerical preparation line map requires an INTEGER-or-null list")
        digest, size = sha256(), 0
        try:
            for chunk in json.JSONEncoder(separators=(",", ":"), allow_nan=False).iterencode(mapping):
                encoded = chunk.encode("ascii")
                size += len(encoded)
                if size > _IDENTITY_TEXT_LIMIT:
                    raise CompilationError("original numerical preparation line-map text budget exhausted")
                digest.update(encoded)
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            raise CompilationError("invalid original numerical preparation line-map identity") from error
        entry["line_map"] = {"private_identity_kind": "configured_line_map_sha256_v1",
                             "count": len(mapping), "sha256": digest.hexdigest()}
    return identity


def _source_authority(analysis, procedure, selection, helpers, budget):
    """Authenticate consulted source namespaces and exact original bindings."""
    from fparser.two.utils import Base

    from compiler.frontend.source_functions import (
        _original,
        _scope_roles,
        _validate_resolution,
        _validate_scope,
        _validate_scope_roles,
    )
    scope_roles = _scope_roles(analysis)
    if not selection:
        raise CompilationError("original numerical preparation requires exact original selected nodes")
    for node in selection:
        budget.tick()
        if not isinstance(node, Base):
            raise CompilationError("original numerical preparation requires exact original selected nodes")
    # The fresh structured-source lookup authenticates exact original objects,
    # including joined groups, lexical aliases and accompanying comments. Its
    # string-ID interface is deliberately excluded above: this consumer requires
    # parser objects. Walking every unrelated expression in the owner again
    # adds no membership authority and can exhaust a small region's budget.
    graph, identities = analysis._selected_source(procedure, selection)
    # Selection authenticates this original routine and its source namespaces.
    # Read that same routine without starting an adjacent full-source demand.
    routine = analysis.routines[procedure]
    scope_ids, consulted_scopes = set(), {}

    def resolution_scope_authority(scope):
        _validate_scope(analysis, scope)
        consulted_scopes[id(scope)] = scope

    def scope_authority(scope):
        while scope is not None and id(scope) not in scope_ids:
            budget.tick()
            for _binding in scope.bindings:
                budget.tick()
            for _name in scope.imports:
                budget.tick()
            _validate_scope(analysis, scope, track=False)
            scope_ids.add(id(scope))
            consulted_scopes[id(scope)] = scope
            scope = scope.parent

    owners = [(routine, selection)]
    if len(helpers) > analysis.procedure_limit:
        raise CompilationError("original numerical preparation helper budget exhausted")
    for name in helpers:
        helper = _original(analysis, name)
        owners.append((helper, (helper.execution,)))
    source = []
    for owner, body in owners:
        scope_authority(owner.scope)
        nodes = tuple(_walk(body, budget))
        # Preserve role-aware component selectors; component authority is
        # checked separately below, never treated as a lexical procedure name.
        omitted = {id(part) for node in nodes if type(node).__name__ == "Data_Ref"
                   for part in node.items[1:] if type(part).__name__ in {"Name", "Part_Ref"}}
        omitted.update(id(part.items[0]) for node in nodes if type(node).__name__ == "Data_Ref"
                       for part in node.items[1:] if type(part).__name__ == "Part_Ref")
        _validate_resolution(analysis, owner.scope, (node for node in nodes if id(node) not in omitted), tick=budget.tick,
                             _scope_validator=resolution_scope_authority)
        for node in nodes:
            if type(node).__name__ != "Data_Ref":
                continue
            from compiler.frontend.component_bindings import ComponentAccessUnsupported, _typename, component_access
            from compiler.frontend.source_objects import _TypeBuilder
            try:
                access = component_access(analysis, owner.scope, node)
            except ComponentAccessUnsupported as error:
                # Parent budget callbacks ran after the initial source check.
                # Reauthenticate without callbacks before treating this known
                # syntax limit as a support-only numerical decline.
                analysis._selected_source(procedure, selection)
                for helper in helpers:
                    _original(analysis, helper)
                for scope in consulted_scopes.values():
                    _validate_scope(analysis, scope, track=False)
                _validate_scope_roles(analysis, scope_roles, consulted_scopes)
                raise NumericalPreparationUnsupported(str(error)) from error
            if access is None:
                raise CompilationError("original numerical preparation component authority is unavailable")
            authority = _TypeBuilder(analysis)
            schema = authority.binding(access.root_binding)
            for index, name in enumerate(access.path):
                budget.tick()
                if index >= analysis.depth_limit:
                    raise CompilationError("original numerical preparation component depth exhausted")
                for _member in schema.members:
                    budget.tick()
                member = schema.members.get(name)
                if member is None:
                    raise CompilationError("original numerical preparation component declaration changed")
                if index + 1 < len(access.path):
                    nested = _typename(member[0])
                    schema = authority.resolve(schema.scope, nested) if nested else None
                    if schema is None:
                        raise CompilationError("original numerical preparation component type changed")
            for _visit in range(authority.visits):
                budget.tick()
        session = getattr(analysis, "_source_session", None)
        if session is not None:
            if not session.contains(owner):
                raise CompilationError("original numerical preparation requires original source authority")
            source.append((owner.qualified, session.identity))
        else:
            source.append((owner.qualified, analysis._routine_signature(owner), str(owner.scope.node)))
    return routine, graph, identities, _digest((_input_identity(analysis, budget), tuple(source)), budget)


def _environment_required(analysis, root, budget):
    from compiler.frontend.native_environment import intrinsic_export
    from compiler.frontend.source_functions import _validate_scope
    checked = {}
    syntax = analysis._require_original_global()

    def scope_authority(scope):
        while scope is not None and id(scope) not in checked:
            budget.tick()
            for _binding in scope.bindings:
                budget.tick()
            _validate_scope(analysis, scope, track=False)
            checked[id(scope)] = scope
            scope = scope.parent

    # Environment lookup may consult original reexports. Authenticate their
    # namespaces without demanding numerical legality/resolution of unrelated
    # opaque calls elsewhere in the configured invocation.
    for scope in analysis.modules.values():
        scope_authority(scope)
    pending, visited, calls = [root], set(), 0

    def publish(required):
        # A budget callback can mutate syntax after its containing module was
        # checked. Recheck before every return without retaining either memo.
        for scope in checked.values():
            _validate_scope(analysis, scope, track=False)
        final_syntax = analysis._require_original_global()
        for procedure in sorted(visited | {root}):
            analysis._require_original_routine(procedure, _syntax=final_syntax)
        return required

    while pending:
        name = pending.pop()
        if name in visited:
            continue
        if len(visited) >= analysis.procedure_limit:
            return publish(True)
        routine = analysis._require_original_routine(name, _syntax=syntax)
        scope_authority(routine.scope)
        visited.add(name)
        for node in _source_calls(routine.execution, budget):
            calls += 1
            if calls > analysis.operation_limit:
                return publish(True)
            if type(node.items[0]).__name__ != "Name":
                continue
            scope = analysis.source_scope_for(node, routine.scope)
            if intrinsic_export(analysis, scope, node.items[0]) in {
                    "$intrinsic::ieee_exceptions::ieee_get_halting_mode",
                    "$intrinsic::ieee_exceptions::ieee_set_halting_mode"}:
                return publish(True)
            pending.extend(target for target in analysis._candidates(scope, node.items[0])
                           if target in analysis.routines and target not in visited)
    return publish(False)


def _extraction_identity(extraction, budget):
    def binding_identity(binding):
        original = getattr(binding, "component_object", None)
        if original is not None:
            # ComponentAccess produces a new view Binding on every lookup.
            # Authority remains the exact original object and selector, not
            # the ephemeral view object's address.
            return ("original_component", id(original), id(getattr(binding, "component_selector", None)),
                    binding.public())
        return ("original_binding", id(binding), id(binding.declaring_scope), binding.public())

    return _digest((extraction.public(), extraction.source,
                    tuple((id(node), str(node)) for node in extraction.nodes),
                    tuple(binding_identity(binding) for binding in extraction.bindings),
                    tuple((id(node), str(node)) for guard in extraction.preconditions for node in guard._source_nodes)), budget)


@dataclass(frozen=True, init=False)
class NumericalPreparationProvider:
    """An explicit source-only policy capability, bound to one analysis."""
    identity: str
    options: object = field(repr=False)
    config: object = field(repr=False)
    environment_root: str
    _analysis: object = field(repr=False, compare=False)
    _immutable_inputs: tuple = field(repr=False, compare=False)

    def __init__(self, analysis, *, options, config, environment_root, immutable_inputs=None):
        budget = _Budget()
        immutable = dict(immutable_inputs or {})
        analysis._require_original(environment_root)
        identity = _configuration(options, config, environment_root, immutable, budget)
        for token in immutable.values():
            token.validate(analysis)
        for name, value in (("identity", identity), ("options", options), ("config", config),
                            ("environment_root", environment_root), ("_analysis", analysis),
                            ("_immutable_inputs", tuple(sorted(immutable.items())))):
            object.__setattr__(self, name, value)
        registry = _registry(analysis, "_numerical_preparation_providers")
        if len(registry) >= analysis.operation_limit * analysis.procedure_limit:
            raise CompilationError("numerical preparation provider registry budget exhausted")
        registry[id(self)] = self

    @property
    def immutable_inputs(self):
        return dict(self._immutable_inputs)

    def validate(self, analysis, *, consume=None):
        budget = _Budget(consume)
        if (analysis is not self._analysis
                or _registry(analysis, "_numerical_preparation_providers").get(id(self)) is not self
                or _configuration(self.options, self.config, self.environment_root, self.immutable_inputs, budget) != self.identity):
            raise CompilationError("numerical preparation provider lacks registered original policy authority")
        analysis._require_original(self.environment_root)
        for token in self.immutable_inputs.values():
            budget.tick()
            token.validate(analysis)
        return self

    def public(self):
        return {"schema_version": NUMERICAL_PREPARATION_VERSION, "proof_role": "numerical_preparation_provider",
                "provider_identity": self.identity, "environment_root": self.environment_root,
                "source_only": True, "execution_authorized": False}

    def validate_configuration(self, *, options, config, environment_root, immutable_inputs=None):
        if _configuration(options, config, environment_root, immutable_inputs or {}, _Budget()) != self.identity:
            raise CompilationError("prepared original numerical region compiler policy changed")
        self.validate(self._analysis)
        return self

    def prepare(self, analysis, procedure, exact_original_selection, *, consume=None):
        self.validate(analysis, consume=consume)
        return _prepare(analysis, procedure, exact_original_selection, self, consume=consume)

    def prepare_worksharing(self, analysis, procedure, exact_original_loop, completion, *,
                            preceding=(), following=(), consume=None):
        """Prepare an exact original-team DO without capture or emission.

        Participation and suffix liveness come from the registered completion
        proof. Ordinary lowering retains reached guards that need not supply
        source effect authority; consumers must check that authority separately.
        """
        from compiler.frontend.worksharing_completion import WorksharingCompletionProof
        from compiler.scopes.regions import extract_region

        self.validate(analysis, consume=consume)
        selection = (exact_original_loop,)
        if type(completion) is not WorksharingCompletionProof:
            raise CompilationError("worksharing preparation requires a registered original-team proof")
        completion.validate(analysis, procedure, selection)
        extraction = extract_region(analysis, analysis.routines[procedure], exact_original_loop,
            preceding=preceding, following=following, worksharing=completion,
            immutable_inputs=self.immutable_inputs)
        return _prepare(analysis, procedure, selection, self, consume=consume,
                        extraction=extraction, require_effects=False)


@dataclass(frozen=True)
class PreparedOriginalRegion:
    procedure: str
    identity: str
    extraction: object = field(repr=False, compare=False)
    function: object = field(repr=False, compare=False)
    plan: object = field(repr=False, compare=False)
    offload: object = field(repr=False, compare=False)
    context: object = field(repr=False, compare=False)
    preconditions: tuple
    scalar_changes: tuple[str, ...]
    completion: object = field(repr=False, compare=False)
    completion_proof: object = field(repr=False, compare=False)
    requires_original_team_environment_preflight: bool
    _provider: NumericalPreparationProvider = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _selected_node_ids: tuple[str, ...]
    _source_identity: str = field(repr=False)
    _payload_identity: str = field(repr=False)
    _extraction_identity: str = field(repr=False)
    _context_identity: str = field(repr=False)
    _native_environment_required: bool = field(repr=False)

    @property
    def original_storage_effects(self):
        available = self.extraction.operation_kind == "numerical_loop"
        return {"available": available, "descriptor_changes": False if available else None,
                "escapes": False if available else None,
                "source": "complete original extraction and stable required helper declarations",
                "current_allocation_authorized": False, "alias_authorized": False,
                "entry_definition_authorized": False, "execution_authorized": False}

    def public(self):
        return {"schema_version": NUMERICAL_PREPARATION_VERSION,
                "proof_role": "prepared_original_numerical_region", "preparation_identity": self.identity,
                "provider_identity": self._provider.identity, "procedure": self.procedure,
                "selected_node_ids": list(self._selected_node_ids), "source_identity": self._source_identity,
                "extraction_source_identity": self.extraction.source_identity,
                "preconditions": [item.public() for item in self.preconditions],
                "scalar_changes": list(self.scalar_changes), "completion": dict(self.completion),
                "physical_analysis_available": self.offload.available,
                "physical_analysis_reason": self.offload.reason,
                "physical_backend": "scoped_pointer_abi",
                "all_runtime_guards_typed": self.all_runtime_guards_typed,
                "original_storage_effects": self.original_storage_effects,
                "requires_original_team_environment_preflight": self.requires_original_team_environment_preflight,
                "capture_authorized": False, "artifacts_emitted": False, "execution_authorized": False,
                "definition_transfer": "conditional on all original typed preconditions; separate effects proof required"}

    @property
    def all_runtime_guards_typed(self):
        # The original extraction emits exactly two string predicates per
        # typed tile pair. Other guards (for example indexed sections) retain
        # their ordinary emitter path without gaining definition authority.
        return self.extraction.runtime_guards == self.covered_runtime_guards

    @property
    def covered_runtime_guards(self):
        return tuple(dict.fromkeys(item.runtime_expression for item in self.preconditions))

    def validate_configuration(self, *, options, config, environment_root, immutable_inputs=None):
        self._provider.validate_configuration(options=options, config=config, environment_root=environment_root,
                                              immutable_inputs=immutable_inputs)
        return self

    def validate(self, analysis, procedure, exact_original_selection, *, consume=None):
        budget = _Budget(consume)
        self._provider.validate(analysis, consume=budget.tick)
        selection = tuple(exact_original_selection)
        if (procedure != self.procedure or len(selection) != len(self._selection)
                or any(left is not right for left, right in zip(selection, self._selection, strict=True))
                or _registry(analysis, "_numerical_preparations").get(self.identity) is not self):
            raise CompilationError("prepared numerical region lacks registered exact original selection authority")
        routine, graph, identities, source = _source_authority(analysis, procedure, selection,
                                                              self.extraction.numerical_helpers, budget)
        from compiler.scopes.region_context import original_region_context
        context = original_region_context(routine, selection, consume=budget.tick)
        _validate_completion(analysis, procedure, selection, identities, self.completion, self.completion_proof)
        # The configured caller can reach IEEE imports through namespaces
        # outside this selected region's helper closure. Reauthenticate the
        # same bounded original scan on every reuse, including its reexports.
        native_environment = _environment_required(analysis, self._provider.environment_root, budget)
        team_guard = (native_environment and not self._provider.config.collective
                      and bool(self.completion.get("has_openmp_in_closure")))
        if (native_environment != self._native_environment_required
                or native_environment and (not self.function.requires_numerical_environment
                                           or not self.extraction.numerical_environment_required)
                or team_guard != self.requires_original_team_environment_preflight):
            raise CompilationError("prepared numerical original environment authority changed")
        if (identities != self._selected_node_ids or source != self._source_identity
                or _extraction_identity(self.extraction, budget) != self._extraction_identity
                or _digest((self.function, self.plan, self.offload), budget) != self._payload_identity
                or _digest(context.public(), budget) != self._context_identity
                or any(len(getattr(context, name)) != len(getattr(self.context, name))
                       or any(left is not right for left, right in zip(getattr(context, name), getattr(self.context, name), strict=True))
                       for name in ("preceding", "following", "cyclic_following"))
                or self.preconditions is not self.extraction.preconditions
                or self.completion is not self.extraction.completion
                or _digest((self._provider.identity, graph.identity, identities, source,
                            self._payload_identity, self._extraction_identity, self._context_identity,
                            self.scalar_changes, self.requires_original_team_environment_preflight,
                            native_environment), budget) != self.identity):
            raise CompilationError("prepared numerical source, context or checked IR authority changed")
        return self


def _validate_completion(analysis, procedure, selection, identities, record, proof):
    from compiler.frontend.native_completion import NativeCompletionProof
    from compiler.frontend.numerical_completion import NumericalCompletionProof
    from compiler.frontend.worksharing_completion import WorksharingCompletionProof
    if proof is None:
        if record.get("proof_identity") or record.get("has_openmp_in_closure"):
            raise CompilationError("original numerical preparation lacks registered completion authority")
    elif type(proof) is NativeCompletionProof:
        proof.validate(analysis, procedure, identities)
    elif type(proof) is NumericalCompletionProof:
        proof.validate(analysis)
        if proof.procedure != procedure or proof.selected_node_ids != identities:
            raise CompilationError("original numerical preparation completion selection changed")
    elif type(proof) is WorksharingCompletionProof:
        proof.validate(analysis, procedure, selection)
    else:
        raise CompilationError("original numerical preparation completion token is unsupported")
    if proof is not None and record != proof.public():
        raise CompilationError("original numerical preparation completion record changed")


def _prepare(analysis, procedure, exact_original_selection, provider, *, consume=None, extraction=None, context=None,
             require_effects=True):
    from compiler.driver.pipeline import prepare_function
    from compiler.frontend import lower_source
    from compiler.ir import ParallelRegion
    from compiler.offload.analysis import OffloadAnalysis
    from compiler.offload.preparation import prepare_offload
    from compiler.scopes.region_context import original_region_context
    from compiler.scopes.regions import _exception_observers, extract_region
    budget = _Budget(consume)
    selection = tuple(exact_original_selection)
    routine, graph, identities, _source = _source_authority(analysis, procedure, selection, (), budget)
    context = context or original_region_context(routine, selection, consume=budget.tick)
    if extraction is None:
        # A reached owner also visits complete native teams, such as original
        # IEEE setup. Authenticate their source first, then decline numerical
        # preparation without granting native effects or swallowing a failed
        # numerical proof. The ordinary native analysis remains responsible.
        if not any(type(node).__name__ == "Block_Nonlabel_Do_Construct"
                   for node in _walk(selection, budget)):
            raise NumericalPreparationUnsupported("original numerical candidate contains no counted loop")
        extraction = extract_region(analysis, routine, selection, preceding=context.preceding,
                                    following=context.following, cyclic_following=context.cyclic_following,
                                    immutable_inputs=provider.immutable_inputs)
    routine, graph, identities, source = _source_authority(analysis, procedure, selection,
                                                        extraction.numerical_helpers, budget)
    if observers := _exception_observers(analysis):
        raise NumericalPreparationUnsupported("source-observable floating-point exception flags prevent GPU numerical closure: "
                                               + "; ".join(observers))
    native_environment = _environment_required(analysis, provider.environment_root, budget)
    source_name = str(routine.scope.path) + "#inline:" + extraction.source_identity
    function, plan = prepare_function(lower_source(extraction.source, extraction.entry, source_name=source_name),
                                      options=provider.options)
    if native_environment:
        function = replace(function, requires_numerical_environment=True)
    if function.requires_numerical_environment:
        extraction = replace(extraction, numerical_environment_required=True)
    if not plan.regions:
        raise NumericalPreparationUnsupported("inline numerical candidate has no proven parallel region")
    # Source-scoped workers use a pointer scalar ABI. Reuse that emitter's
    # physical preparation, including protected-input and safe-query checks;
    # the ordinary by-value analyzer deliberately rejects unused tile scalars.
    # Keep the initial numerical-effects contract flat and all-device.
    offload = (prepare_offload(function, plan).analysis
               if plan.steps and all(isinstance(step, ParallelRegion) for step in plan.steps)
               else OffloadAnalysis(False, "original numerical effects require a nonempty flat all-device execution plan"))
    if require_effects and not offload.available:
        raise NumericalPreparationUnsupported(offload.reason)
    if require_effects and extraction.runtime_guards != tuple(dict.fromkeys(
            item._runtime_expression for item in extraction.preconditions)):
        raise NumericalPreparationUnsupported("original numerical geometry has runtime guards without typed source preconditions")
    completion_proof = None
    completion_id = extraction.completion.get("proof_identity")
    if completion_id is not None:
        for registry in ("_numerical_completions", "_joined_completions", "_worksharing_completions"):
            completion_proof = getattr(analysis, registry, {}).get(completion_id)
            if completion_proof is not None:
                break
    _validate_completion(analysis, procedure, selection, identities, extraction.completion, completion_proof)
    serial = not extraction.completion.get("has_openmp_in_closure")
    scalar_changes = tuple(sorted({routine.scope.bindings[name].root for name in extraction.private_scalars
                                   if name in routine.scope.bindings})) if serial else ()
    payload, extracted, context_id = (_digest((function, plan, offload), budget),
                                     _extraction_identity(extraction, budget), _digest(context.public(), budget))
    team_guard = native_environment and not provider.config.collective and not serial
    identity = _digest((provider.identity, graph.identity, identities, source, payload, extracted, context_id,
                        scalar_changes, team_guard, native_environment), budget)
    token = PreparedOriginalRegion(procedure, identity, extraction, function, plan, offload, context,
                                   extraction.preconditions, scalar_changes, extraction.completion, completion_proof, team_guard,
                                   provider, selection, identities, source, payload, extracted, context_id, native_environment)
    registry = _registry(analysis, "_numerical_preparations")
    budget.tick()
    existing = registry.get(identity)
    if existing is not None:
        if (type(existing) is not PreparedOriginalRegion or existing._provider.identity != provider.identity
                or existing.procedure != procedure or len(existing._selection) != len(selection)
                or any(left is not right for left, right in zip(existing._selection, selection, strict=True))):
            raise CompilationError("original numerical preparation registry identity collision")
        existing.validate(analysis, procedure, selection, consume=budget.tick)
        return existing
    if len(registry) >= analysis.operation_limit * analysis.procedure_limit:
        raise CompilationError("original numerical preparation registry budget exhausted")
    registry[identity] = token
    try:
        # Cold and reused tokens perform the same complete payload/source
        # validation. A cached identity never avoids bounded original work.
        token.validate(analysis, procedure, selection, consume=budget.tick)
    except BaseException:
        if registry.get(identity) is token:
            del registry[identity]
        raise
    return token


def prepare_original_region(analysis, procedure, exact_original_selection, *, options, config, environment_root,
                            immutable_inputs=None, consume=None):
    """Demand one original candidate without capture, artifact or runtime work."""
    provider = NumericalPreparationProvider(analysis, options=options, config=config, environment_root=environment_root,
                                            immutable_inputs=immutable_inputs)
    return provider.prepare(analysis, procedure, exact_original_selection, consume=consume)
