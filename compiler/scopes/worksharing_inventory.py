"""Original-team numerical candidates shared by discovery and admission.

This inventory performs source preparation only. Native effects, storage,
definitions and execution still require their independent checked-plan proofs.
An unsupported unit stays in the ordered inventory with its original DO.
"""

from dataclasses import dataclass, field

from fparser.two.utils import walk

from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.frontend.worksharing_completion import ordered_worksharing_inventory
from compiler.ir import CompilationError
from compiler.scopes.segments import statement_span


def prepared_span(node):
    spans = [item.item.span for item in walk(node) if getattr(item, "item", None)]
    if not spans or any(span is None for span in spans):
        raise CompilationError("original team unit lacks prepared source order")
    return min(span[0] for span in spans), max(span[1] for span in spans)


def unit_span(unit):
    opening, loop, ending = unit
    first = statement_span(opening)[0]
    last = statement_span(ending)[1] if ending is not None else statement_span(loop.content[-1])[1]
    return first, last


@dataclass(frozen=True)
class WorksharingCandidate:
    unit: tuple = field(repr=False)
    completion: object = field(repr=False)
    prepared: object = field(repr=False)
    span: tuple | None
    reason: str | None


@dataclass(frozen=True)
class PreparedWorksharingInventory:
    """Complete ordered candidates under one exact numerical policy.

    Declines are retained as source-only results, never native-effect authority.
    Reuse checks every prepared payload before admission can outline a worker.
    """
    source: object = field(repr=False)
    provider: object = field(repr=False)
    candidates: tuple[WorksharingCandidate, ...]
    _key: tuple = field(repr=False)

    def validate(self, analysis, procedure, originals, provider):
        self.source.validate(analysis, procedure, originals)
        provider.validate(analysis)
        self.provider.validate(analysis)
        if (provider.identity != self.provider.identity
                or _prepared_registry(analysis).get(self._key) is not self):
            raise CompilationError("prepared worksharing inventory lacks registered source and policy authority")
        for candidate in self.candidates:
            if candidate.completion is not None:
                candidate.completion.validate(analysis, procedure, (candidate.unit[1],))
            if candidate.prepared is not None:
                candidate.prepared.validate(analysis, procedure, (candidate.unit[1],))
        return self


def _prepared_registry(analysis):
    registry = getattr(analysis, "_prepared_worksharing_inventories", None)
    if registry is None or getattr(analysis, "_prepared_worksharing_registry_owner", None) is not analysis:
        registry = {}
        analysis._prepared_worksharing_inventories = registry
        analysis._prepared_worksharing_registry_owner = analysis
    return registry


def prepare_worksharing_inventory(builder, procedure, originals, *, provider=None):
    """Prepare one complete ordered team once for discovery and admission.

    The registered source inventory checks the complete team before numerical
    preparation. Reuse authenticates the same source, original completion and
    numerical policy; native effects, capture and execution are never reused.
    """
    analysis = builder.analysis
    originals = tuple(originals)
    source = ordered_worksharing_inventory(analysis, procedure, originals)
    provider = provider or NumericalPreparationProvider(analysis, options=builder.options, config=builder.config,
        environment_root=builder._numerical_environment_root,
        immutable_inputs=getattr(builder, "immutable_inputs", {}))
    provider.validate(analysis)
    key = (id(analysis), source.identity, provider.identity)
    registry = _prepared_registry(analysis)
    if key in registry:
        inventory = registry[key].validate(analysis, procedure, originals, provider)
        return inventory.source.parent, inventory.candidates
    candidates = []
    for unit in source.units:
        span, completion, prepared, reason = None, None, None, None
        try:
            # Prepared include coordinates are not original editable spans.
            span = unit_span(unit)
            completion = analysis.worksharing_completion(procedure, originals, (unit[1],))
            prepared = provider.prepare_worksharing(analysis, procedure, unit[1], completion)
        except CompilationError as error:
            reason = str(error)
        candidates.append(WorksharingCandidate(tuple(unit), completion, prepared, span, reason))
    inventory = PreparedWorksharingInventory(source, provider, tuple(candidates), key)
    if len(registry) >= analysis.operation_limit * analysis.procedure_limit:
        raise CompilationError("prepared worksharing inventory registry budget exhausted")
    registry[key] = inventory
    try:
        inventory.validate(analysis, procedure, originals, provider)
    except BaseException:
        if registry.get(key) is inventory:
            del registry[key]
        raise
    return source.parent, inventory.candidates
