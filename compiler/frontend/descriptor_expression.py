"""Iterative, source-budgeted descriptor expression traversal.

Call recursion is bounded separately by the complete descriptor query.
Each original expression node still passes the existing classifier and charged
source traversal; the worklist supplies no evaluation or effect authority.
"""
from __future__ import annotations


def visit_descriptor_expression(check, routine, original, *, descriptor=False):
    """Resume classifier frames in the exact former depth-first visit order."""
    frames = [iter(((original, descriptor),))]
    try:
        while frames:
            try:
                node, is_descriptor = next(frames[-1])
            except StopIteration:
                frames.pop()
                continue
            frames.append(iter(check.expression_children(routine, node, descriptor=is_descriptor)))
    finally:
        # A typed source-function demand aborts this complete consumer attempt;
        # no suspended classifier/AST frame survives into the frontend retry.
        for frame in reversed(frames):
            close = getattr(frame, "close", None)
            if close is not None:
                close()
