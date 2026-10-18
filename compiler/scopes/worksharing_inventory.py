"""Original-team numerical candidates shared by discovery and admission.

This inventory performs source preparation only. Native effects, storage,
definitions and execution still require their independent checked-plan proofs.
An unsupported unit stays in the ordered inventory with its original DO.
"""

from dataclasses import dataclass, field

from fparser.two.utils import walk

from compiler.frontend.native_completion import _joined_completion_facts
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
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


def prepare_worksharing_inventory(builder, procedure, originals, *, provider=None):
    """Keep every synchronized original DO, including declined native work.

    The parent and first native token check the complete team before any
    numerical subregion is prepared. NOWAIT in an unrelated sibling therefore
    cannot be hidden by selecting only a numerical loop.
    """
    analysis = builder.analysis
    originals = tuple(originals)
    parent = analysis.joined_completion(procedure, originals)
    units = []
    _joined_completion_facts(analysis, procedure, originals, worksharing=units)
    if not units:
        raise CompilationError("mixed execution requires separate original PARALLEL and synchronized DOs")
    analysis.worksharing_native_completion(procedure, originals, (units[0][1],))
    provider = provider or NumericalPreparationProvider(analysis, options=builder.options, config=builder.config,
        environment_root=builder._numerical_environment_root,
        immutable_inputs=getattr(builder, "immutable_inputs", {}))
    candidates = []
    for unit in units:
        span, completion, prepared, reason = None, None, None, None
        try:
            # Prepared include coordinates are not original editable spans.
            span = unit_span(unit)
            completion = analysis.worksharing_completion(procedure, originals, (unit[1],))
            prepared = provider.prepare_worksharing(analysis, procedure, unit[1], completion)
        except CompilationError as error:
            reason = str(error)
        candidates.append(WorksharingCandidate(tuple(unit), completion, prepared, span, reason))
    return parent, tuple(candidates)
