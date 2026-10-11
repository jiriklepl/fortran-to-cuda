"""Original lexical liveness context shared by preparation and emission."""

from dataclasses import dataclass

from fparser.two.utils import walk

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


def original_region_context(routine, sources, *, consume=None):
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

    def prepared_span(node, *, optional=False):
        spans = []
        for item in walk(node):
            tick()
            if getattr(item, "item", None) is not None:
                spans.append(item.item.span)
        if not spans or any(span is None for span in spans):
            if optional:
                return None
            raise CompilationError("source context has no prepared ordering span")
        return min(span[0] for span in spans), max(span[1] for span in spans)

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
