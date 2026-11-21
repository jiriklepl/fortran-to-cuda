"""Bounded structured owners, with queries evaluated at reached source points.

This is source scheduling, not numerical lowering. Inline source operations stay
native and retain their original OpenMP team. Numerical calls reuse the existing
mode-bearing workers and public planning interface.
"""

from __future__ import annotations

import copy
import re

from fparser.two.utils import walk

from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError


def fortran_lines(lines):
    """Wrap generated free-form syntax, including continued OpenMP comments."""
    result = []
    for original in lines:
        line = original
        openmp = line.lstrip().lower().startswith("!$omp")
        if line.lstrip().startswith("!") and not openmp:
            result.append(line)
            continue
        while len(line) > 112:
            quote, boundary = None, None
            for index, character in enumerate(line[:112]):
                if character in {"'", '"'}:
                    quote = None if quote == character else character if quote is None else quote
                elif quote is None and character in {" ", ","} and index > 5:
                    boundary = index + (character == ",")
            if boundary is None:
                raise CompilationError("structured source has an overlong indivisible token")
            result.append(line[:boundary].rstrip() + " &")
            line = ("!$omp& " if openmp else "  & ") + line[boundary:].lstrip()
        result.append(line)
    return result


def directive(node):
    if _kind(node) != "Comment":
        return None
    text = str(node).lstrip().lower()
    return text[5:].strip() if text.startswith("!$omp") else None




def grouped_nodes(nodes):
    """Keep whole joined groups as one native scheduling operation."""
    expanded = []
    for node in nodes:
        if _kind(node) in {"Block_Nonlabel_Do_Construct", "If_Construct", "Associate_Construct"}:
            prefix = []
            for item in node.content:
                if _kind(item) != "Comment":
                    break
                prefix.append(item)
            if prefix:
                # fparser attaches directives before the DO header to its
                # construct. Their source positions precede the loop, so expose
                # them to the group parser without mutating the original AST.
                clone = copy.copy(node)
                clone.content = list(node.content[len(prefix):])
                expanded += [*prefix, clone]
                continue
        expanded.append(node)
    nodes = expanded
    result, index = [], 0
    while index < len(nodes):
        start = directive(nodes[index])
        if start is not None and re.match(r"parallel(?:\s|$)", start):
            combined = bool(re.match(r"parallel\s+do(?:\s|$)", start))
            ending = "end parallel do" if combined else "end parallel"
            if combined:
                loop = index + 1
                while loop < len(nodes) and _kind(nodes[loop]) == "Comment" and directive(nodes[loop]) is None:
                    loop += 1
                if loop >= len(nodes) or _kind(nodes[loop]) != "Block_Nonlabel_Do_Construct":
                    raise CompilationError("combined PARALLEL DO requires its original associated loop")
                end = loop + 1
                while end < len(nodes) and _kind(nodes[end]) == "Comment" and directive(nodes[end]) is None:
                    end += 1
                if end < len(nodes) and directive(nodes[end]) == ending:
                    result.append(tuple(nodes[index:end + 1]))
                    index = end + 1
                else:
                    if end < len(nodes) and (directive(nodes[end]) or "").startswith("end parallel"):
                        raise CompilationError("combined PARALLEL DO has a mismatched original join")
                    # Fortran's END PARALLEL DO is optional. The complete DO
                    # itself supplies the join; never absorb following work.
                    result.append(tuple(nodes[index:loop + 1]))
                    index = loop + 1
                continue
            end = index + 1
            while end < len(nodes) and directive(nodes[end]) != ending:
                end += 1
            if end == len(nodes):
                raise CompilationError("native PARALLEL group has no proven join")
            result.append(tuple(nodes[index:end + 1]))
            index = end + 1
        else:
            result.append(nodes[index])
            index += 1
    return result


def statement_span(node):
    items = [item for item in walk(node) if getattr(item, "item", None)]
    if not items:
        raise CompilationError("structured source operation lacks an exact edit span")
    spans = [item.item.span for item in items]
    return min(span[0] for span in spans), max(span[1] for span in spans)




















