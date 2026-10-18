"""Original lexical liveness context shared by preparation and emission."""

from collections import OrderedDict
from dataclasses import dataclass
from weakref import ref

from fparser.two.utils import Base

from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError


@dataclass(frozen=True)
class OriginalRegionContext:
    preceding: tuple
    following: tuple
    cyclic_following: tuple

    def public(self):
        return {name: [str(node) for node in getattr(self, name)]
                for name in ("preceding", "following", "cyclic_following")}


@dataclass(frozen=True)
class _SessionContext:
    session: object
    owner: object
    routine: object
    sources: tuple
    context: OriginalRegionContext
    visits: int


# Only completed queries over a sealed original graph enter this private memo.
# A weak analysis reference releases entries with their compilation owner; the
# LRU also bounds simultaneous selections. Exact objects prevent recycled IDs
# from identifying a different source. Mutable standalone analyses never enter.
_SESSION_CONTEXTS = OrderedDict()
_SESSION_CONTEXT_LIMIT = 128


def original_region_context(routine, sources, *, consume=None, analysis=None):
    """Reuse exact immutable-session contexts without reducing proof budgets.

    Source/configuration and prepared-IR authentication remain with the caller.
    A memo hit replays every logical scan debit, including callback failures;
    it only avoids reconstructing the same lexical result from immutable ASTs.
    """
    session = getattr(analysis, "_source_session", None)
    if session is None:
        return _derive_original_region_context(routine, sources, consume=consume)
    sources = tuple(sources)

    def authenticate():
        if (getattr(analysis, "_source_session", None) is not session
                or session.routine(analysis, routine.qualified) is not routine):
            raise CompilationError("original numerical context requires exact compilation-session source")

    authenticate()
    key = id(session), id(routine), tuple(map(id, sources))
    saved = _SESSION_CONTEXTS.get(key)
    if (saved is not None and saved.session is session and saved.routine is routine
            and len(saved.sources) == len(sources)
            and all(left is right for left, right in zip(saved.sources, sources, strict=True))):
        if consume is not None:
            for _visit in range(saved.visits):
                consume(1)
        # A budget consumer can invoke compiler callbacks that replace a
        # registry/projection after the initial authentication.
        authenticate()
        if _SESSION_CONTEXTS.get(key) is saved:
            _SESSION_CONTEXTS.move_to_end(key)
        return saved.context

    # Validate exact original membership once before retaining this selection.
    # Session sealing then prevents its syntax, provenance or order changing.
    analysis._selected_source(routine.qualified, sources)
    visits = 0

    def account(amount):
        nonlocal visits
        visits += amount
        if consume is not None:
            consume(amount)

    context = _derive_original_region_context(routine, sources, consume=account)
    authenticate()

    def expire(owner_ref):
        current = _SESSION_CONTEXTS.get(key)
        if current is not None and current.owner is owner_ref:
            del _SESSION_CONTEXTS[key]

    _SESSION_CONTEXTS[key] = _SessionContext(session, ref(analysis, expire), routine, sources, context, visits)
    _SESSION_CONTEXTS.move_to_end(key)
    while len(_SESSION_CONTEXTS) > _SESSION_CONTEXT_LIMIT:
        _SESSION_CONTEXTS.popitem(last=False)
    return context


def _derive_original_region_context(routine, sources, *, consume=None):
    """Keep excluded suffixes and backedges as distinct original paths.

    Prepared coordinates establish order, including included source that has no
    editable original span. This helper grants no definition or capture facts.
    """
    visits = 0

    def tick():
        nonlocal visits
        visits += 1
        if consume is not None:
            consume(1)
        if visits > 65536:
            raise CompilationError("original numerical context source-scan budget exhausted")

    # Only this ordering query owns the memo. Retain each object as well as
    # its aggregate, so recycled IDs cannot identify a different parser node.
    # An existing item with no span poisons the aggregate; an absent item does
    # not. Expression descendants remain part of the original full-walk proof.
    spans_by_node = {}

    def children(node):
        if isinstance(node, Base):
            values = node.children
            return tuple(values) if isinstance(values, (list, tuple)) else (values,)
        if isinstance(node, tuple):
            # fparser.walk calls walk(component) for each tuple component;
            # list/tuple arguments are normalized at that call's entry.
            return tuple(child for component in node
                         for child in (component if isinstance(component, (list, tuple)) else (component,)))
        return ()

    def summary(node):
        pending, active = [(node, None)], set()
        while pending:
            current, descendants = pending.pop()
            key = id(current)
            if descendants is None:
                tick()  # A memo lookup is still charged to both work budgets.
                if key in spans_by_node:
                    continue
                if key in active:
                    raise CompilationError("original numerical context source is cyclic")
                active.add(key)
                descendants = children(current)
                pending.append((current, descendants))
                pending.extend((child, None) for child in reversed(descendants))
                continue
            item = getattr(current, "item", None)
            span, poisoned = (item.span, item.span is None) if item is not None else (None, False)
            for child in descendants:
                _original, own_span, own_poisoned = spans_by_node[id(child)]
                poisoned |= own_poisoned
                if own_span is not None:
                    span = own_span if span is None else (min(span[0], own_span[0]), max(span[1], own_span[1]))
            spans_by_node[key] = current, span, poisoned
            active.remove(key)
        return spans_by_node[id(node)][1:]

    def prepared_span(node, *, optional=False):
        span, poisoned = None, False
        # Match walk's normalization of its root argument, including unknown
        # wrappers. A list encountered as a child is not itself descended.
        for root in node if isinstance(node, (list, tuple)) else (node,):
            own_span, own_poisoned = summary(root)
            poisoned |= own_poisoned
            if own_span is not None:
                span = own_span if span is None else (min(span[0], own_span[0]), max(span[1], own_span[1]))
        if span is None or poisoned:
            if optional:
                return None
            raise CompilationError("source context has no prepared ordering span")
        return span

    spans = [prepared_span(node) for node in sources]
    if not spans:
        raise CompilationError("original numerical context requires a nonempty selection")
    first, last = min(span[0] for span in spans), max(span[1] for span in spans)
    before, after, cyclic = [], [], []

    def visit(node):
        span = prepared_span(node, optional=True)
        if span is None:
            return
        low, high = span
        if high < first:
            before.append(node)
        elif low > last:
            after.append(node)
        elif _kind(node) in {"If_Construct", "Block_Nonlabel_Do_Construct"}:
            for child in node.content:
                visit(child)
            if _kind(node) == "Block_Nonlabel_Do_Construct":
                for child in node.content:
                    low, high = prepared_span(child)
                    if high < first and _kind(child) not in {"Nonlabel_Do_Stmt", "Comment"}:
                        cyclic.append(child)

    for node in _children(routine.execution):
        visit(node)
    return OriginalRegionContext(tuple(before), tuple(after), tuple(cyclic))
