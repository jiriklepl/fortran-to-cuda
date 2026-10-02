#!/usr/bin/env python3
"""PSyclone transformation recipe for the existing CDU/CDV/CDW modules.

The input is never edited. The only parser accommodation temporarily removes
CONTIGUOUS, unsupported by PSyclone's typed PSyIR; emitted declarations restore
it. All physics comes from the original source, through actual PSyclone
inlining, code-motion, fusion and accelerator-data transformations.
"""

from __future__ import annotations

import importlib.metadata
import re

VERSION = "3.3.1"


def transform(source: str, case: str, *, target="openacc", fuse=True, vector_length=128) -> tuple[str, dict]:
    """Transform this benchmark family, retaining its public module interface."""
    from psyclone.core import SymbolicMaths
    from psyclone.line_length import FortLineLength
    from psyclone.psyir.backend.fortran import FortranWriter
    from psyclone.psyir.frontend.fortran import FortranReader
    from psyclone.psyir.nodes import (
        ArrayReference,
        Assignment,
        Call,
        IntrinsicCall,
        Literal,
        Loop,
        Reference,
        Routine,
        Schedule,
    )
    from psyclone.psyir.symbols import ArrayType, ScalarType
    from psyclone.psyir.transformations import (
        ACCLoopTrans,
        InlineTrans,
        LoopFuseTrans,
        MoveTrans,
    )
    from psyclone.transformations import ACCDataTrans, ACCParallelTrans

    installed = importlib.metadata.version("psyclone")
    if installed != VERSION:
        raise ValueError(f"recipe requires psyclone=={VERSION}; found {installed}")
    if case not in ("CDU", "CDV", "CDW") or target not in ("cpu", "openacc"):
        raise ValueError("supported cases: CDU/CDV/CDW; targets: cpu/openacc")
    if vector_length != 128:
        raise ValueError("vector_length is only configurable in the Loki recipe")

    # This accommodation does not weaken the generated Fortran's contiguity
    # contract: record exactly which declarations need restoring after writing.
    contiguous: dict[str, set[str]] = {}
    routine_name = None
    for line in source.splitlines():
        match = re.match(r"\s*subroutine\s+(\w+)\s*\(", line, re.I)
        if match:
            routine_name = match[1].lower()
        if re.search(r",\s*contiguous\b", line, re.I):
            if routine_name is None or "::" not in line:
                raise ValueError("unsupported CONTIGUOUS declaration")
            # Only this family's plain rank-three assumed-shape arrays.
            declaration = line.split("::", 1)[1]
            names = re.findall(r"(\w+)\s*\(\s*:\s*,\s*:\s*,\s*:\s*\)", declaration)
            remainder = re.sub(r"\w+\s*\(\s*:\s*,\s*:\s*,\s*:\s*\)", "", declaration)
            if not names or remainder.strip(" ,"):
                raise ValueError("expected plain rank-three assumed-shape arrays")
            contiguous.setdefault(routine_name, set()).update(n.lower() for n in names)
    tree = FortranReader().psyir_from_source(re.sub(r",\s*contiguous\b", "", source, flags=re.I))
    entry = next(r for r in tree.walk(Routine) if r.name.lower() == case.lower())
    helpers = [c for c in entry.walk(Call) if not isinstance(c, IntrinsicCall)]
    if [c.routine.name.lower() for c in helpers] != ["set", case.lower() + "div", case.lower() + "adv", "multiply"]:
        raise ValueError("expected the benchmark's set/div/adv/multiply call sequence")
    for call in helpers:
        InlineTrans().apply(call)

    # InlineTrans correctly accounts for actual-argument bounds. For these
    # nonallocatable assumed-shape dummy arguments (:,:,:), each lower bound
    # inside the entry routine is exactly one, regardless of the caller's bounds.
    folded_bounds = 0
    for call in entry.walk(IntrinsicCall):
        if call.intrinsic != IntrinsicCall.Intrinsic.LBOUND:
            continue
        array = call.arguments[0]
        if (
            not isinstance(array, Reference)
            or array.symbol not in entry.symbol_table.argument_list
            or not isinstance(array.symbol.datatype, ArrayType)
            or any(extent != ArrayType.Extent.ATTRIBUTE for extent in array.symbol.datatype.shape)
        ):
            raise ValueError("cannot prove that the inlined actual argument has lower bound one")
        call.replace_with(Literal("1", ScalarType.integer_type()))
        folded_bounds += 1
    # Integer-only simplification removes inlining's i - 1 + 1 offsets. Do not
    # simplify/reassociate any floating-point stencil expression.
    for array in entry.walk(ArrayReference):
        for index in list(array.indices):
            SymbolicMaths.expand(index)

    # InlineTrans renames the two helpers' coefficient variables. MoveTrans
    # checks dependencies while hoisting those scalar initializations, making
    # the four iteration spaces adjacent without duplicating floating work.
    moved_assignments = 0
    for statement in list(entry.children):
        if isinstance(statement, Assignment):
            earlier_loops = [n for n in entry.children[: statement.position] if isinstance(n, Loop)]
            if earlier_loops:
                MoveTrans().apply(statement, earlier_loops[0])
                moved_assignments += 1
    fusions = 0
    if fuse:
        # Traversal visits the surviving outer loop's schedule after outer
        # fusion, so nested j/i loops can then be fused with the same operation.
        for schedule in entry.walk(Schedule):
            while True:
                pair = next(
                    (
                        (a, b)
                        for a, b in zip(schedule.children, schedule.children[1:], strict=False)
                        if isinstance(a, Loop) and isinstance(b, Loop)
                    ),
                    None,
                )
                if pair is None:
                    break
                LoopFuseTrans().apply(*pair)  # No force/dependency override.
                fusions += 1
    loops = [n for n in entry.children if isinstance(n, Loop)]
    if len(loops) != (1 if fuse else 4):
        raise ValueError("unexpected top-level loop structure after transformation")
    if target == "openacc":
        for loop in loops:
            ACCLoopTrans().apply(loop, {"collapse": 3, "gang": True, "vector": True})
            # Enclose this loop's newly created ACCLoopDirective.
            ACCParallelTrans().apply(loop.parent.parent, {"default_present": False})
        parallel_regions = [loop.parent.parent.parent.parent for loop in loops]
        # PSyclone derives copyin(u,v,w) and copyout(output) from its access
        # analysis. Nested use inside a caller data region reuses present data.
        ACCDataTrans().apply(parallel_regions)

    generated = FortranWriter()(tree)
    lines = []
    routine_name = None
    restored = set()
    for line in generated.splitlines():
        match = re.match(r"\s*subroutine\s+(\w+)\s*\(", line, re.I)
        if match:
            routine_name = match[1].lower()
        declaration = re.match(r"(\s*real\(kind=knd\), dimension\(:,:,:\))(.*::\s*)(\w+)\s*$", line, re.I)
        if declaration and declaration[3].lower() in contiguous.get(routine_name, set()):
            line = f"{declaration[1]}, contiguous{declaration[2]}{declaration[3]}"
            restored.add((routine_name, declaration[3].lower()))
        lines.append(line)
    expected = {(routine, name) for routine, names in contiguous.items() for name in names}
    if restored != expected:
        raise ValueError(f"could not restore CONTIGUOUS attributes: {expected - restored}")
    generated = FortLineLength().process("\n".join(lines) + "\n")
    manifest = {
        "tool": "PSyclone",
        "version": installed,
        "recipe": ["InlineTrans", "fold proven assumed-shape LBOUND=1", "simplify integer array indices", "MoveTrans"]
        + (["LoopFuseTrans with dependency analysis"] if fuse else [])
        + (
            ["ACCLoopTrans collapse(3) gang vector", "ACCParallelTrans", "ACCDataTrans automatic copyin/copyout"]
            if target == "openacc"
            else []
        ),
        "compatibility": "temporarily remove CONTIGUOUS while parsing, restore in generated declarations",
        "inlined_calls": len(helpers),
        "folded_lower_bounds": folded_bounds,
        "moved_coefficient_assignments": moved_assignments,
        "loop_fusions": fusions,
        "loop_nests": len(loops),
        "fused": fuse,
        "source": "https://psyclone.readthedocs.io/en/stable/user_guide/transformations.html",
        "generated_code_control": "editable Fortran/OpenACC; transformation script exposes loop mapping, fusion and data regions",
    }
    return generated, manifest
