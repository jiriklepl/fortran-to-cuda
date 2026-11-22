"""Rewrite loop forms written for a CPU cache into forms whose points are independent.

Both rewrites keep every arithmetic expression and its evaluation order. They
change only where intermediate values live, so that each output point can be
computed on its own.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from fparser.two.utils import walk

from compiler.frontend.syntax import parse_fortran

from .anchors import _line_char_offsets, _split_top_level_args, find_routine_span


def _do_loop_info(loop_node) -> tuple[str, list]:
    header = next(
        (item for item in loop_node.content if type(item).__name__ == "Nonlabel_Do_Stmt"),
        None,
    )
    if header is None or len(header.items) < 2 or header.items[1] is None:
        raise ValueError("Expected counted DO construct")
    counted = header.items[1].items[1]
    if not counted:
        raise ValueError("Expected counted DO loop control")
    var = str(counted[0]).lower()
    bounds = list(counted[1])
    return var, bounds


def _loop_statements(node) -> list:
    return [
        item
        for item in node.content
        if type(item).__name__ not in {"Comment", "Nonlabel_Do_Stmt", "End_Do_Stmt"}
    ]


def _extract_consumer_upper_bound(bound_expr: str, outer_var: str, outer_hi: str) -> str:
    text = re.sub(r"\s+", "", bound_expr)
    m = re.fullmatch(r"min\((.+)\)", text, re.IGNORECASE)
    if m:
        args = _split_top_level_args(m.group(1))
        for arg in args:
            if not re.search(rf"\b{re.escape(outer_var)}\b", arg, re.IGNORECASE):
                return arg
    if not re.search(rf"\b{re.escape(outer_var)}\b", text, re.IGNORECASE):
        return bound_expr.strip()
    return outer_hi.strip()


def _parse_local_index_assignment(lhs_str: str, rhs_str: str, inner_vars: set[str], outer_vars: set[str]):
    """Match ``lvar = inner_var - outer_var (+ int)?``."""
    if "(" in lhs_str:
        return None
    clean_rhs = re.sub(r"\s+", "", rhs_str).lower()
    for ivar in inner_vars:
        for ovar in outer_vars:
            m = re.fullmatch(
                rf"{re.escape(ivar)}-{re.escape(ovar)}([+-]\d+)?",
                clean_rhs,
            )
            if m:
                offset = int(m.group(1)) if m.group(1) else 0
                return lhs_str.strip().lower(), ivar, ovar, offset
    return None


def _shift_vars(expression: str, var_map: dict[str, str], offsets: Sequence[int], ordered_axes: Sequence[str]) -> str:
    """Shift loop variables in ``expression`` in a single pass."""
    mapping = {}
    for axis_idx, pvar in enumerate(ordered_axes):
        cvar = var_map[pvar]
        off = offsets[axis_idx]
        mapping[pvar] = f"({cvar}+({off}))" if off != 0 else cvar
    if not mapping:
        return expression
    pattern = re.compile(
        r"(?<![.%_])\b(" + "|".join(re.escape(v) for v in mapping) + r")\b",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: mapping[m.group(1).lower()], expression)


def _unpack_3d_nest(nest_node):
    """Unpack a 3-level loop nest, allowing hoisted assignments between loop headers."""
    v2, b2 = _do_loop_info(nest_node)
    stmts2 = _loop_statements(nest_node)
    assigns2 = [s for s in stmts2 if type(s).__name__ == "Assignment_Stmt"]
    loops2 = [s for s in stmts2 if type(s).__name__ == "Block_Nonlabel_Do_Construct"]
    if len(loops2) != 1 or len(assigns2) + 1 != len(stmts2):
        raise ValueError("Expected single nested loop at level 2")
    v1, b1 = _do_loop_info(loops2[0])
    stmts1 = _loop_statements(loops2[0])
    assigns1 = [s for s in stmts1 if type(s).__name__ == "Assignment_Stmt"]
    loops1 = [s for s in stmts1 if type(s).__name__ == "Block_Nonlabel_Do_Construct"]
    if len(loops1) != 1 or len(assigns1) + 1 != len(stmts1):
        raise ValueError("Expected single nested loop at level 1")
    v0, b0 = _do_loop_info(loops1[0])
    return v2, b2, v1, b1, v0, b0, assigns2 + assigns1 + _loop_statements(loops1[0])


def inline_tile_scratch_nests(source_text: str) -> tuple[str, list[dict]]:
    """Inline 3D producer-consumer private tile scratch arrays into pointwise loops."""
    lines = source_text.splitlines(keepends=True)
    tree = parse_fortran(source_text)
    changes: list[tuple[int, int, str, dict]] = []

    for routine in (n for n in walk(tree) if type(n).__name__ == "Subroutine_Subprogram"):
        header = next(n for n in routine.content if type(n).__name__ == "Subroutine_Stmt")
        rname = str(header.items[1]).lower()
        execution = next((n for n in routine.content if type(n).__name__ == "Execution_Part"), None)
        if execution is None:
            continue
        tiles = [n for n in execution.content if type(n).__name__ == "Block_Nonlabel_Do_Construct"]
        for tile in tiles:
            # Verify 3-level outer tile nest containing [producer, consumer]
            try:
                o_var2, o_b2 = _do_loop_info(tile)
                c2 = _loop_statements(tile)
                if len(c2) != 1 or type(c2[0]).__name__ != "Block_Nonlabel_Do_Construct":
                    continue
                mid = c2[0]
                o_var1, o_b1 = _do_loop_info(mid)
                c1 = _loop_statements(mid)
                if len(c1) != 1 or type(c1[0]).__name__ != "Block_Nonlabel_Do_Construct":
                    continue
                inner = c1[0]
                o_var0, o_b0 = _do_loop_info(inner)
                pair = _loop_statements(inner)
                if len(pair) != 2 or any(type(n).__name__ != "Block_Nonlabel_Do_Construct" for n in pair):
                    continue
                producer, consumer = pair
                # Unpack 3-level producer and consumer nests
                p_var2, _, p_var1, _, p_var0, _, prod_stmts = _unpack_3d_nest(producer)
                c_var2, c_b2, c_var1, c_b1, c_var0, c_b0, cons_stmts = _unpack_3d_nest(consumer)
            except ValueError:
                continue

            outer_vars = {o_var0, o_var1, o_var2}
            prod_vars = {p_var0, p_var1, p_var2}
            cons_vars = {c_var0, c_var1, c_var2}

            prod_local_idx: dict[str, tuple[str, int]] = {}  # lvar -> (p_loop_var, const_offset)
            scalar_defs: dict[str, str] = {}
            scratch_defs: dict[tuple[str, tuple[str, str, str]], str] = {}
            straight_line = True
            for stmt in prod_stmts:
                if type(stmt).__name__ != "Assignment_Stmt":
                    straight_line = False
                    break
                lhs, _, rhs = stmt.items
                lhs_str = str(lhs).strip()
                parsed_idx = _parse_local_index_assignment(lhs_str, str(rhs), prod_vars, outer_vars)
                if parsed_idx is not None:
                    lvar, ivar, _ovar, off = parsed_idx
                    prod_local_idx[lvar] = (ivar, off)
                    continue
                expr = str(rhs)
                if scalar_defs:
                    pat = re.compile(
                        r"(?<![.%_])\b(" + "|".join(re.escape(k) for k in scalar_defs) + r")\b",
                        re.IGNORECASE,
                    )
                    expr = pat.sub(lambda m, defs=scalar_defs: f"({defs[m.group(1).lower()]})", expr)
                if type(lhs).__name__ == "Part_Ref":
                    arr_name = str(lhs.items[0]).lower()
                    idx_items = tuple(re.sub(r"\s+", "", str(x)).lower() for x in lhs.items[1].items)
                    if len(idx_items) != 3 or not all(idx in prod_local_idx for idx in idx_items):
                        straight_line = False
                        break
                    scratch_defs[(arr_name, idx_items)] = expr
                else:
                    scalar_defs[lhs_str.lower()] = expr

            if not straight_line or not scratch_defs:
                continue

            scratch_names = {arr for arr, _ in scratch_defs}
            # Determine consumer local index variables and map producer->consumer axes
            cons_local_idx: dict[str, tuple[str, int]] = {}
            cons_body_stmts = []
            for stmt in cons_stmts:
                if type(stmt).__name__ == "Assignment_Stmt":
                    lhs_str = str(stmt.items[0]).strip()
                    parsed_idx = _parse_local_index_assignment(lhs_str, str(stmt.items[2]), cons_vars, outer_vars)
                    if parsed_idx is not None:
                        lvar, ivar, _ovar, off = parsed_idx
                        cons_local_idx[lvar] = (ivar, off)
                        continue
                cons_body_stmts.append(stmt)

            prod_to_cons_var = {p_var0: c_var0, p_var1: c_var1, p_var2: c_var2}
            ordered_p_axes = [p_var0, p_var1, p_var2]

            rendered_body: list[str] = []
            valid_consumer = True
            for stmt in cons_body_stmts:
                rendered = str(stmt)
                for ref in sorted(
                    (
                        n
                        for n in walk(stmt)
                        if type(n).__name__ == "Part_Ref" and str(n.items[0]).lower() in scratch_names
                    ),
                    key=lambda n: len(str(n)),
                    reverse=True,
                ):
                    arr_name = str(ref.items[0]).lower()
                    ref_indices = getattr(ref.items[1], "items", ())
                    matching_keys = [k for k in scratch_defs if k[0] == arr_name]
                    if len(matching_keys) != 1 or len(ref_indices) != 3:
                        valid_consumer = False
                        break
                    _, def_idx_tuple = matching_keys[0]
                    offsets_by_pvar: dict[str, int] = {}
                    for def_lvar, idx_expr in zip(def_idx_tuple, ref_indices, strict=False):
                        p_var, p_off = prod_local_idx[def_lvar]
                        clean_idx = re.sub(r"\s+", "", str(idx_expr)).lower()
                        # Find matching consumer local index variable
                        matched_off = None
                        for clvar, (_cvar, c_off) in cons_local_idx.items():
                            m = re.fullmatch(rf"{re.escape(clvar)}(?:([+-])(\d+))?", clean_idx)
                            if m:
                                ref_off = 0 if m.group(1) is None else int(m.group(1) + m.group(2))
                                matched_off = ref_off + (c_off - p_off)
                                break
                        if matched_off is None:
                            valid_consumer = False
                            break
                        offsets_by_pvar[p_var] = matched_off
                    if not valid_consumer:
                        break
                    axis_offsets = [offsets_by_pvar[pvar] for pvar in ordered_p_axes]
                    expanded = _shift_vars(
                        scratch_defs[(arr_name, def_idx_tuple)],
                        prod_to_cons_var,
                        axis_offsets,
                        ordered_p_axes,
                    )
                    rendered = rendered.replace(str(ref), f"({expanded})")
                if not valid_consumer:
                    break
                rendered_body.append(rendered)

            if not valid_consumer or not rendered_body:
                continue

            lo2 = str(o_b2[0]).strip()
            hi2 = _extract_consumer_upper_bound(str(c_b2[1]), o_var2, str(o_b2[1]))
            lo1 = str(o_b1[0]).strip()
            hi1 = _extract_consumer_upper_bound(str(c_b1[1]), o_var1, str(o_b1[1]))
            lo0 = str(o_b0[0]).strip()
            hi0 = _extract_consumer_upper_bound(str(c_b0[1]), o_var0, str(o_b0[1]))
            replacement = (
                f"    do {c_var2}={lo2},{hi2}\n"
                f"      do {c_var1}={lo1},{hi1}\n"
                f"        do {c_var0}={lo0},{hi0}\n"
                + "\n".join(rendered_body)
                + "\n        enddo\n      enddo\n    enddo\n"
            )
            first = next(n for n in tile.content if type(n).__name__ == "Nonlabel_Do_Stmt").item.span[0]
            last = tile.content[-1].item.span[1]
            changes.append(
                (
                    first,
                    last,
                    replacement,
                    {"routine": rname, "scratch_arrays": sorted(scratch_names)},
                )
            )

    for first, last, replacement, _meta in sorted(changes, key=lambda item: item[0], reverse=True):
        lines[first - 1 : last] = [replacement]
    metadata = [meta for _, _, _, meta in sorted(changes, key=lambda item: item[0])]
    return "".join(lines), metadata


def promote_line_snapshots(source_text: str) -> tuple[str, list[dict]]:
    """Lift 1D line-copy buffers (``tmp = arr(:,j,k)``) into 3D snapshots.

    The snapshot is a local allocatable that the procedure allocates at its
    start and deallocates at its end: a temporary whose storage the domain
    lends (see ``storage.lend_storage``).
    """
    tree = parse_fortran(source_text)
    edits: list[dict] = []
    text = source_text

    candidate_routines = []
    for routine in (n for n in walk(tree) if type(n).__name__ == "Subroutine_Subprogram"):
        header = next(n for n in routine.content if type(n).__name__ == "Subroutine_Stmt")
        rname = str(header.items[1]).lower()
        # Look for 1D declaration using ubound(<arr>, <dim>)
        found = False
        for decl in walk(routine):
            if type(decl).__name__ != "Type_Declaration_Stmt":
                continue
            dtype, attrs, entities = decl.items
            if attrs is not None and "allocatable" in str(attrs).lower():
                continue
            for entity in entities.items:
                shape = entity.items[1]
                if shape is None or len(getattr(shape, "items", ())) != 1:
                    continue
                dim_spec = shape.items[0]
                hi = str(getattr(dim_spec, "items", (None, None))[1] or "")
                m = re.fullmatch(
                    r"ubound\(\s*([a-z_][a-z0-9_]*)\s*,\s*[a-z_][a-z0-9_]*\s*\)",
                    re.sub(r"\s+", "", hi),
                    re.I,
                )
                if m:
                    lo_expr = re.sub(r"\s+", "", str(dim_spec.items[0]))
                    tmp_name = str(entity.items[0])
                    arr_name = m.group(1)
                    candidate_routines.append((rname, str(dtype), tmp_name, arr_name, lo_expr))
                    found = True
                    break
            if found:
                break

    for rname, dtype_str, tmp_name, arr_name, lo_expr in reversed(candidate_routines):
        r_lo, r_hi = find_routine_span(text, rname)
        body = text[r_lo:r_hi]

        # 1. Replace 1D declaration with 3D allocatable declaration via AST span.
        sub_tree = parse_fortran(body)
        decl_span = None
        for decl in walk(sub_tree):
            if type(decl).__name__ != "Type_Declaration_Stmt":
                continue
            for entity in decl.items[2].items:
                if str(entity.items[0]).lower() == tmp_name.lower() and entity.items[1] is not None:
                    decl_span = decl.item.span
                    break
            if decl_span is not None:
                break
        if decl_span is None:
            raise ValueError(f"Missing 1D snapshot declaration for {tmp_name} in {rname}")
        body_offsets = _line_char_offsets(body)
        d_lo = body_offsets[decl_span[0]][0]
        d_hi = body_offsets[decl_span[1]][1]
        body = body[:d_lo] + f"    {dtype_str}, allocatable :: {tmp_name}(:,:,:)" + body[d_hi:]

        # 2. Insert 3D allocation & copy before the first executable assignment / control statement.
        sub_tree = parse_fortran(body)
        sub_exec = next(n for n in walk(sub_tree) if type(n).__name__ == "Execution_Part")
        first_exec = next((n for n in sub_exec.content if type(n).__name__ != "Comment"), None)
        if first_exec is None:
            raise ValueError(f"Empty execution body in {rname}")
        body_offsets = _line_char_offsets(body)
        insert_line = first_exec.item.span[0]
        insert_pos = body_offsets[insert_line][0]
        alloc_block = (
            f"    allocate({tmp_name}({lo_expr}:ubound({arr_name},1),{lo_expr}:ubound({arr_name},2),{lo_expr}:ubound({arr_name},3)))\n"
            f"    do k={lo_expr},ubound({arr_name},3)\n"
            f"      do j={lo_expr},ubound({arr_name},2)\n"
            f"        do i={lo_expr},ubound({arr_name},1)\n"
            f"          {tmp_name}(i,j,k)={arr_name}(i,j,k)\n"
            f"        enddo\n"
            f"      enddo\n"
            f"    enddo\n\n"
        )
        body = body[:insert_pos] + alloc_block + body[insert_pos:]

        # 3. Replace line copies tmp = U(:,j,k) / U(i,:,k) / U(i,j,:) and 1D tmp(idx) reads.
        line_patterns = [
            (rf"(?im)^[ \t]*{re.escape(tmp_name)}\s*=\s*{re.escape(arr_name)}\s*\(\s*:\s*,\s*([a-z_]\w*)\s*,\s*([a-z_]\w*)\s*\)[ \t]*$", 0),
            (rf"(?im)^[ \t]*{re.escape(tmp_name)}\s*=\s*{re.escape(arr_name)}\s*\(\s*([a-z_]\w*)\s*,\s*:\s*,\s*([a-z_]\w*)\s*\)[ \t]*$", 1),
            (rf"(?im)^[ \t]*{re.escape(tmp_name)}\s*=\s*{re.escape(arr_name)}\s*\(\s*([a-z_]\w*)\s*,\s*([a-z_]\w*)\s*,\s*:\s*\)[ \t]*$", 2),
        ]
        # Replace single-axis reads inside each branch (or by matching index variables ii/jj/kk)
        for pat_str, axis in line_patterns:
            m_line = re.search(pat_str, body)
            if m_line is None:
                raise ValueError(f"Missing line snapshot assignment for axis {axis} in {rname}")
            v1, v2 = m_line.group(1), m_line.group(2)
            body = body[: m_line.start()] + "! Input line is in the complete snapshot." + body[m_line.end() :]
            idx_name = ("ii", "jj", "kk")[axis]
            if not re.search(rf"\b{re.escape(tmp_name)}\s*\(\s*{idx_name}\s*\)", body, re.IGNORECASE):
                # Fallback: infer 1D index name from inner do-loop in the surrounding construct
                inner_do = re.search(
                    r"(?im)do\s+([a-z_]\w*)\s*=\s*[a-z_]\w*\s*-\s*(\d+)\s*,\s*[a-z_]\w*\s*\+\s*\2\b",
                    body[m_line.start() :],
                )
                if inner_do:
                    idx_name = inner_do.group(1)
            coords = [v1, v2]
            coords.insert(axis, idx_name)
            coord_str = ",".join(coords)
            read_pat = re.compile(rf"\b{re.escape(tmp_name)}\s*\(\s*{re.escape(idx_name)}\s*\)", re.IGNORECASE)
            if not read_pat.search(body):
                raise ValueError(f"Missing 1D snapshot read {tmp_name}({idx_name}) in {rname}")
            body = read_pat.sub(f"{tmp_name}({coord_str})", body)

        # 4. Express fixed symmetric stencil offsets directly: do ii = i-R, i+R -> do ii = -R, R
        stencil_pat = re.compile(
            r"(?ims)^([ \t]*)do\s+([a-z_]\w*)\s*=\s*([a-z_]\w*)\s*-\s*(\d+)\s*,\s*\3\s*\+\s*\4\b(.*?)^\s*end\s*do\b"
        )
        found_stencils = list(stencil_pat.finditer(body))
        if not found_stencils:
            raise ValueError(f"Missing original symmetric filter accumulation in {rname}")
        for m_loop in reversed(found_stencils):
            idx_v, axis_v, radius = m_loop.group(2), m_loop.group(3), m_loop.group(4)
            original_loop = m_loop.group(0)
            shifted = re.sub(
                rf"\b{re.escape(idx_v)}\b",
                f"({axis_v}+{idx_v})",
                original_loop,
                flags=re.IGNORECASE,
            )
            shifted = re.sub(
                rf"\(\s*{re.escape(axis_v)}\s*\+\s*{re.escape(idx_v)}\s*\)\s*-\s*{re.escape(axis_v)}\b",
                idx_v,
                shifted,
                flags=re.IGNORECASE,
            )
            shifted = re.sub(
                rf"(?i)do\s+\(\s*{re.escape(axis_v)}\s*\+\s*{re.escape(idx_v)}\s*\)\s*=\s*{re.escape(axis_v)}\s*-\s*{radius}\s*,\s*{re.escape(axis_v)}\s*\+\s*{radius}\b",
                f"do {idx_v} = -{radius}, {radius}",
                shifted,
                count=1,
            )
            body = body[: m_loop.start()] + shifted + body[m_loop.end() :]

        # 5. Release snapshot before end subroutine
        ending = re.search(rf"(?im)^[ \t]*end\s*subroutine(?:\s+{re.escape(rname)})?\b", body)
        if ending is None:
            raise ValueError(f"Missing procedure end in {rname}")
        release_block = f"\n    deallocate({tmp_name})\n"
        body = body[: ending.start()] + release_block + body[ending.start() :]
        text = text[:r_lo] + body + text[r_hi:]
        edits.append({"routine": rname, "snapshot": tmp_name, "source_array": arr_name})

    return text, edits
