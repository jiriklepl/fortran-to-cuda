#!/usr/bin/env python3
"""Use Loki's inliner/loop fusion on the three unmodified advection sources.

This is deliberately a transformation recipe for CDU/CDV/CDW, not a general
parallelizing compiler. The arrays read by the stencil are separate from its
pointwise output; this fact makes fusing the four helper loops legal.
"""

from __future__ import annotations

import importlib.metadata

from loki import (
    FP,
    Assignment,
    CallStatement,
    Comment,
    CommentBlock,
    FindNodes,
    FindVariables,
    Loop,
    Pragma,
    Sourcefile,
    Subroutine,
    Transformer,
)
from loki.transformations.inline.procedures import inline_subroutine_calls
from loki.transformations.transform_loop import do_loop_fusion

LOKI_COMMIT = "be67f4ae348c15d88413111590f1b42a058e21c3"
LOKI_URL = "https://github.com/ecmwf-ifs/loki"


def _loop_nest(loop):
    nest = [loop]
    for _ in range(2):
        children = [n for n in nest[-1].body if not isinstance(n, (Comment, CommentBlock))]
        if len(children) != 1 or not isinstance(children[0], Loop):
            raise ValueError("Expected a perfect three-dimensional loop nest")
        nest.append(children[0])
    return nest


def _check_and_hoist(routine, output_name):
    """Check the benchmark's pointwise-output contract before enabling fusion.

    Loki's pragma-directed fusion does not prove arbitrary dependence safety.
    Reject other shapes instead of accidentally applying this recipe to them.
    """
    loops = [n for n in routine.body.body if isinstance(n, Loop)]
    if len(loops) != 4:
        raise ValueError("Expected exactly four helper loop nests")
    top_assignments = [n for n in routine.body.body if isinstance(n, Assignment)]
    scalar_writes = set()
    first_bounds = None
    for loop in loops:
        nest = _loop_nest(loop)
        bounds = tuple(str(n.bounds).lower() for n in nest)
        if first_bounds is None:
            first_bounds = bounds
        elif bounds != first_bounds:
            raise ValueError("The four helper loops must have identical bounds")
        indices = tuple(str(n.variable).lower() for n in reversed(nest))
        for statement in nest[-1].body:
            if isinstance(statement, (Comment, CommentBlock)):
                continue
            if not isinstance(statement, Assignment):
                raise ValueError("Only straight-line assignment bodies are supported")
            lhs = statement.lhs
            if lhs.type.shape:
                if lhs.name.lower() != output_name:
                    raise ValueError("Only the output array may be written")
            else:
                scalar_writes.add(lhs.name.lower())
            for variable in FindVariables().visit(statement):
                if (
                    variable.name.lower() == output_name
                    and tuple(str(d).lower() for d in variable.dimensions) != indices
                ):
                    raise ValueError("The output must only be accessed pointwise")
    for assignment in top_assignments:
        if assignment.lhs.type.shape or assignment.lhs.name.lower() in scalar_writes:
            raise ValueError("Only independent scalar setup can precede fused loops")
        if any(v.type.shape or v.name.lower() in scalar_writes for v in FindVariables().visit(assignment.rhs)):
            raise ValueError("Scalar setup must not depend on loop results")
    permitted = (Assignment, Loop, Comment, CommentBlock, Pragma)
    if any(not isinstance(node, permitted) for node in routine.body.body):
        raise ValueError("Unsupported statement between helper loops")
    # Fusion inserts at the first loop. Place all loop-invariant coefficients
    # before it, retaining their original order and Loki's disambiguated names.
    routine.body._update(
        body=tuple(top_assignments) + tuple(node for node in routine.body.body if node not in top_assignments)
    )
    return loops


def transform(source: str, case: str, *, target="openacc", fuse=True, vector_length=128) -> tuple[str, dict]:
    if case not in ("CDU", "CDV", "CDW") or target not in ("cpu", "openacc"):
        raise ValueError("supported cases: CDU/CDV/CDW; targets: cpu/openacc")
    if vector_length < 1:
        raise ValueError("vector_length must be positive")
    installed_version = importlib.metadata.version("loki")
    if not installed_version.endswith("+g" + LOKI_COMMIT[:9]):
        raise RuntimeError(f"Install pinned Loki {LOKI_COMMIT}; found {installed_version}")
    source = Sourcefile.from_source(source, frontend=FP)
    module = source["MomentumAdvection"]
    routine = module[case]
    routine.enrich(module.subroutines)
    calls = FindNodes(CallStatement).visit(routine.body)
    expected = ["set", f"{case.lower()}div", f"{case.lower()}adv", "multiply"]
    if [str(c.name).lower() for c in calls] != expected:
        raise ValueError(f"Expected entry calls {expected}")
    for call in calls:
        inline_subroutine_calls(routine, [call], call.routine, allowed_aliases=("i", "j", "k"))
    arrays = [arg for arg in routine.arguments if arg.type.shape]
    output_array = arrays[0]
    if (
        len(arrays) != 4
        or output_array.type.intent.lower() != "out"
        or any(arg.type.intent.lower() != "in" for arg in arrays[1:])
    ):
        raise ValueError("Expected one output and three input arrays")
    loops = _check_and_hoist(routine, output_array.name.lower())
    if fuse:
        routine.body = Transformer(
            {loop: (Pragma(keyword="loki", content="loop-fusion collapse(3)"), loop) for loop in loops}
        ).visit(routine.body)
        do_loop_fusion(routine)
    # Leave the original public ABI and empty timing hooks in place. Helpers
    # have been expanded using Loki's argument and symbol substitution.
    keep = {case.lower(), "start_hot", "finish_hot"}
    module.contains._update(
        body=tuple(
            node for node in module.contains.body if not isinstance(node, Subroutine) or node.name.lower() in keep
        )
    )
    loops = [node for node in routine.body.body if isinstance(node, Loop)]
    if target == "openacc":
        array_names = ", ".join(arg.name for arg in arrays)
        loop_map = {}
        for loop in loops:
            # Thread-local scalar work variables assigned inside the stencil.
            private = sorted({a.lhs.name for a in FindNodes(Assignment).visit(loop) if not a.lhs.type.shape})
            clauses = f"parallel loop gang vector collapse(3) vector_length({vector_length}) present({array_names})"
            if private:
                clauses += f" private({', '.join(private)})"
            loop_map[loop] = (
                Pragma(keyword="acc", content=clauses),
                loop,
                Pragma(keyword="acc", content="end parallel loop"),
            )
        routine.body = Transformer(loop_map).visit(routine.body)
        data = f"data copyin({', '.join(a.name for a in arrays[1:])}) copyout({output_array.name})"
        routine.body.prepend(Pragma(keyword="acc", content=data))
        routine.body.append(Pragma(keyword="acc", content="end data"))
    metadata = {
        "tool": "loki",
        "upstream": LOKI_URL,
        "commit": LOKI_COMMIT,
        "installed_version": installed_version,
        "fused": fuse,
        "loop_nests": len(loops),
        "vector_length": vector_length if target == "openacc" else None,
        "recipe": ["Loki FP frontend", "inline_subroutine_calls", "hoist independent scalar setup"]
        + (["do_loop_fusion collapse(3)"] if fuse else [])
        + (["OpenACC data and parallel-loop pragmas"] if target == "openacc" else []),
    }
    return source.to_fortran() + "\n", metadata
