"""Bounded structured owners, with queries evaluated at reached source points.

This is source scheduling, not numerical lowering. Inline source operations stay
native and retain their original OpenMP team. Numerical calls reuse the existing
mode-bearing workers and public planning interface.
"""

from __future__ import annotations

import copy
import re

from fparser.two.utils import walk

from compiler.frontend.syntax import kind_of
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
    """The text of an OpenMP directive line after ``!$omp``, or None for other nodes."""
    if kind_of(node) != "Comment":
        return None
    text = str(node).lstrip().lower()
    return text[5:].strip() if text.startswith("!$omp") else None


# A ``!$fort`` line of the compiler's: a residency directive, or what the preparation keeps of an OpenMP directive.
FORT_LINE = re.compile(r"^(?P<indent>[ \t]*)!\$fort\b[ \t]*(?P<body>.*?)[ \t]*$", re.IGNORECASE)
REDUCTION_LINE = re.compile(r"^[ \t]*!\$fort[ \t]+reduction\b(?P<body>.*?)[ \t]*$", re.IGNORECASE)
_CLAUSE = re.compile(r"\b(reduction|private)\s*\(([^()]*)\)", re.IGNORECASE)
_PRIVATE = re.compile(r"!\$fort\s+private\s*\(([^)]*)\)", re.IGNORECASE)


def parse_license(text):
    """The reduction operators and private names a ``!$fort reduction`` line or an OpenMP directive states."""
    operators, private = {}, set()
    for clause, body in _CLAUSE.findall(text):
        if clause.lower() == "private":
            private.update(name.strip().lower() for name in body.split(",") if name.strip())
            continue
        operator, _, names = body.partition(":")
        for name in names.split(","):
            if name.strip():
                operators[name.strip().lower()] = operator.strip().lower()
    return operators, private


def license_line(text):
    """The ``!$fort reduction`` line for an OpenMP directive, or None when it has no reduction clause."""
    operators, private = parse_license(text)
    if not operators:
        return None
    grouped = {}
    for name, operator in operators.items():
        grouped.setdefault(operator, []).append(name)
    line = "!$fort " + " ".join(f"reduction({operator}: {', '.join(names)})" for operator, names in grouped.items())
    return line + (f" private({', '.join(sorted(private))})" if private else "")


def private_marker(names):
    """The comment that says, at the end of a DO statement, which scalars have no value when an iteration starts."""
    return "!$fort private(" + ", ".join(sorted(names)) + ")"


def private_names(text):
    """The names of the ``!$fort private`` comment in ``text``; empty if it has none."""
    match = _PRIVATE.search(text)
    return {name.strip().lower() for name in match.group(1).split(",") if name.strip()} if match else set()


def is_residency_directive(node):
    """A ``!$fort`` line, other than the ``private`` one a loop's OpenMP directive left, which says nothing of residency."""
    text = str(node).lstrip().lower()
    return text.startswith("!$fort") and not re.match(r"!\$fort\s+private\b", text)


def grouped_nodes(nodes):
    """Keep whole joined groups as one native scheduling operation."""
    expanded = []
    for node in nodes:
        if kind_of(node) in {"Block_Nonlabel_Do_Construct", "If_Construct", "Associate_Construct"}:
            prefix = []
            for item in node.content:
                if kind_of(item) != "Comment":
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
                while loop < len(nodes) and kind_of(nodes[loop]) == "Comment" and directive(nodes[loop]) is None:
                    loop += 1
                if loop >= len(nodes) or kind_of(nodes[loop]) != "Block_Nonlabel_Do_Construct":
                    raise CompilationError("combined PARALLEL DO requires its original associated loop")
                end = loop + 1
                while end < len(nodes) and kind_of(nodes[end]) == "Comment" and directive(nodes[end]) is None:
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


class Spans:
    """``statement_span`` of the nodes of trees that do not change meanwhile, each computed once."""

    def __init__(self):
        # Each node is kept with its span, so that its id is not taken by another node.
        self._known = {}

    def __call__(self, node):
        known = self._known.get(id(node))
        if known is None or known[0] is not node:
            known = self._known[id(node)] = (node, statement_span(node))
        return known[1]
