"""Find places in Fortran source text: anchors, procedures and calls.

An anchor is a piece of the original text. It is matched exactly when it
occurs once, and otherwise by its tokens, so reformatting and comment edits
do not lose it. Every lookup must identify exactly one place.
"""
from __future__ import annotations

import re

from fparser.two.utils import walk

from compiler.frontend import syntax


def _sanitize_for_parse(text: str, undefined: tuple[str, ...] = ()) -> str:
    """Blank preprocessor lines and the blocks of ``undefined`` macros, keeping line numbers."""
    inactive = "|".join(re.escape(name.lower()) for name in undefined)
    lines = []
    skip_inactive = False
    depth = 0
    inactive_depth = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            directive = stripped[1:].strip().lower()
            if directive.startswith("if"):
                depth += 1
                if not skip_inactive and inactive and re.match(rf"ifdef\s+(?:{inactive})\b", directive):
                    skip_inactive = True
                    inactive_depth = depth
            elif directive.startswith("else") and skip_inactive and depth == inactive_depth:
                skip_inactive = False
            elif directive.startswith("endif"):
                if skip_inactive and depth == inactive_depth:
                    skip_inactive = False
                    inactive_depth = None
                depth = max(0, depth - 1)
            lines.append("\n" if line.endswith("\n") else "")
            continue
        if skip_inactive:
            lines.append("\n" if line.endswith("\n") else "")
            continue
        cleaned = line.replace("__FILE__", "'__FILE__'").replace("__LINE__", "0")
        lines.append(cleaned)
    return "".join(lines)


def parse_fortran(text: str, *, undefined: tuple[str, ...] = ()):
    """The tree of Fortran text that may not have been preprocessed (see ``syntax.parse_fortran``).

    Text that still carries preprocessor lines is retried with those lines
    blanked; the ``#ifdef`` blocks of the ``undefined`` macros are dropped.
    """
    try:
        return syntax.parse_fortran(text)
    except Exception:
        return syntax.parse_fortran(_sanitize_for_parse(text, tuple(undefined)))


def _strip_inline_comment(line: str) -> tuple[str, str]:
    """Split a Fortran line into (code, comment) respecting quotes and !$omp."""
    in_single = False
    in_double = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "'" and not in_double:
            if in_single and i + 1 < len(line) and line[i + 1] == "'":
                i += 2
                continue
            in_single = not in_single
        elif ch == '"' and not in_single:
            if in_double and i + 1 < len(line) and line[i + 1] == '"':
                i += 2
                continue
            in_double = not in_double
        elif ch == "!" and not in_single and not in_double:
            if line[i:].lower().startswith("!$omp"):
                return line, ""
            return line[:i], line[i:]
        i += 1
    return line, ""


def _logical_lines(text: str) -> list[str]:
    """Join Fortran '&' line continuations into logical lines."""
    raw_lines = text.splitlines()
    logical: list[str] = []
    current = ""
    continuing = False
    for raw in raw_lines:
        code, comment = _strip_inline_comment(raw)
        stripped = code.rstrip()
        if continuing:
            lead = stripped.lstrip()
            if lead.startswith("&"):
                lead = lead[1:]
            piece = lead
        else:
            piece = stripped
        has_cont = piece.endswith("&")
        if has_cont:
            piece = piece[:-1].rstrip()
        if continuing:
            current = f"{current} {piece}".strip() if current and piece else (current or piece)
        else:
            current = piece + (comment if not has_cont and comment else "")
        if has_cont:
            continuing = True
        else:
            logical.append(current)
            current = ""
            continuing = False
    if continuing or current:
        logical.append(current)
    return logical


_TOKEN_SPEC = re.compile(
    r"""
    (?P<string>'(?:''|[^'])*'|"(?:""|[^"])*")
    | (?P<omp>!\$omp\b[^\n]*)
    | (?P<comment>![^\n]*)
    | (?P<dotop>\.[a-z_]+\.)
    | (?P<two>::|/=|==|<=|>=|=>|\*\*|//)
    | (?P<word>[a-z_][a-z0-9_]*)
    | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[ed][+-]?\d+)?(?:_[a-z0-9_]+)?)
    | (?P<punct>[()\[\],=+\-*/%:])
    | (?P<ws>\s+)
    | (?P<other>.)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_COMPOUND_NEXT = {
    "end": {"do", "if", "subroutine", "function", "module", "block", "select", "associate", "type"},
    "else": {"if"},
}


def _tokenize_line(line: str, *, preserve_comments: bool) -> list[tuple[str, str]]:
    """Tokenize one logical Fortran line into (kind, raw_text) pairs."""
    tokens: list[tuple[str, str]] = []
    raw_tokens: list[tuple[str, str]] = []
    for match in _TOKEN_SPEC.finditer(line):
        kind = match.lastgroup
        value = match.group()
        if kind == "ws":
            continue
        if kind == "comment" and not preserve_comments:
            continue
        raw_tokens.append((kind, value))

    i = 0
    while i < len(raw_tokens):
        kind, value = raw_tokens[i]
        if kind == "word" and value.lower() in _COMPOUND_NEXT and i + 1 < len(raw_tokens):
            nkind, nval = raw_tokens[i + 1]
            if nkind == "word" and nval.lower() in _COMPOUND_NEXT[value.lower()]:
                tokens.append(("compound", f"{value.lower()}:{nval.lower()}"))
                i += 2
                continue
        if kind == "word" and value.lower() in {
            "enddo", "endif", "endsubroutine", "endfunction", "endmodule",
            "endblock", "endselect", "endassociate", "endtype", "elseif",
        }:
            prefix = "else" if value.lower().startswith("else") else "end"
            suffix = value.lower()[len(prefix):]
            tokens.append(("compound", f"{prefix}:{suffix}"))
            i += 1
            continue
        tokens.append((kind, value))
        i += 1
    return tokens


def _token_to_regex(kind: str, value: str) -> str:
    if kind == "compound":
        prefix, suffix = value.split(":", 1)
        return rf"\b{re.escape(prefix)}\s*{re.escape(suffix)}\b"
    if kind == "omp":
        sub = [_token_to_regex(k, v) for k, v in _tokenize_line(value[5:], preserve_comments=False)]
        return r"!\$omp" + (r"\s+" + r"\s*".join(sub) if sub else r"")
    if kind == "comment":
        body = " ".join(value[1:].split())
        if not body:
            return r"![^\n]*"
        words = [re.escape(part) for part in body.split()]
        return r"!\s*" + r"\s+".join(words)
    if kind in {"word", "number"}:
        return rf"\b{re.escape(value)}\b"
    return re.escape(value)


def _build_line_regex(tokens: list[tuple[str, str]]) -> str:
    if not tokens:
        return ""
    parts = [_token_to_regex(tokens[0][0], tokens[0][1])]
    wordlike = {"word", "number", "compound", "dotop"}
    sep_opt = r"(?:\s|&\s*(?:![^\n]*)?\r?\n\s*&?)*"
    sep_req = r"(?:\s|&\s*(?:![^\n]*)?\r?\n\s*&?)+"
    for (pk, _), (ck, cv) in zip(tokens, tokens[1:], strict=False):
        sep = sep_req if (pk in wordlike and ck in wordlike) else sep_opt
        parts.append(sep + _token_to_regex(ck, cv))
    return "".join(parts)


def _comment_mask(text: str) -> list[bool]:
    """Compute per-character boolean mask indicating characters inside non-OMP comments."""
    mask = [False] * len(text)
    in_single = False
    in_double = False
    in_comment = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            in_single = False
            in_double = False
            in_comment = False
            i += 1
            continue
        if in_comment:
            mask[i] = True
            i += 1
            continue
        if ch == "'" and not in_double:
            if in_single and i + 1 < n and text[i + 1] == "'":
                i += 2
                continue
            in_single = not in_single
        elif ch == '"' and not in_single:
            if in_double and i + 1 < n and text[i + 1] == '"':
                i += 2
                continue
            in_double = not in_double
        elif ch == "!" and not in_single and not in_double:
            if not text[i:].lower().startswith("!$omp"):
                in_comment = True
                mask[i] = True
        i += 1
    return mask


def _find_token_matches(text: str, anchor: str, *, preserve_comments: bool) -> list[tuple[int, int]]:
    logical = _logical_lines(anchor)
    stmt_regexes: list[str] = []
    for line in logical:
        tokens = _tokenize_line(line, preserve_comments=preserve_comments)
        if tokens:
            stmt_regexes.append(_build_line_regex(tokens))
    if not stmt_regexes:
        return []

    has_nl = "\n" in anchor
    whole_line = has_nl or anchor[:1] in {" ", "\t"}
    if len(stmt_regexes) == 1:
        body = stmt_regexes[0]
    else:
        between = r"[ \t]*(?:![^\n]*)?\r?\n(?:\s*(?:![^\n]*)?\r?\n)*[ \t]*"
        body = between.join(stmt_regexes)

    if whole_line:
        prefix = r"(?im)^[ \t]*"
        suffix = r"[ \t]*(?:![^\n]*)?\r?\n" if anchor.endswith("\n") else r""
    else:
        prefix = r"(?im)"
        suffix = r""

    pattern = re.compile(prefix + body + suffix)
    anchor_has_comment = anchor.lstrip().startswith("!") and not anchor.lstrip().lower().startswith("!$omp")
    mask = None if anchor_has_comment else _comment_mask(text)

    matches: list[tuple[int, int]] = []
    for m in pattern.finditer(text):
        start, end = m.span()
        if mask is not None and start < len(mask) and mask[start]:
            continue
        # Ensure inline expression anchors do not match starting inside a comment.
        if mask is not None and not whole_line and any(mask[pos] for pos in range(start, end)):
            continue
        matches.append((start, end))
    return matches


def find_unique_span(text: str, anchor: str, *, message: str | None = None) -> tuple[int, int]:
    """Locate ``anchor`` uniquely in ``text``, tolerating formatting and comment edits."""
    exact_count = text.count(anchor)
    if exact_count == 1:
        idx = text.index(anchor)
        return idx, idx + len(anchor)
    if exact_count > 1:
        raise ValueError(message or f"Expected exactly one occurrence of the anchor: {anchor!r}")

    has_plain_comment = any(
        _strip_inline_comment(line)[1].strip() for line in anchor.splitlines()
    )
    if has_plain_comment:
        matches = _find_token_matches(text, anchor, preserve_comments=True)
        if len(matches) == 1:
            return matches[0]
    matches = _find_token_matches(text, anchor, preserve_comments=False)
    if len(matches) == 1:
        return matches[0]

    # Allow optional trailing procedure/module name on END SUBROUTINE / END MODULE anchors.
    end_match = re.fullmatch(
        r"\s*end\s*(subroutine|function|module)\s+[a-z_][a-z0-9_]*\s*\n?",
        anchor,
        re.IGNORECASE,
    )
    if end_match:
        stripped_anchor = re.sub(
            r"(end\s*(?:subroutine|function|module))\s+[a-z_][a-z0-9_]*",
            r"\1",
            anchor,
            flags=re.IGNORECASE,
        )
        matches = _find_token_matches(text, stripped_anchor, preserve_comments=False)
        if len(matches) == 1:
            return matches[0]

    raise ValueError(message or f"Expected exactly one occurrence of the anchor: {anchor!r}")


def find_once(text: str, anchor: str, *, message: str | None = None) -> int:
    """Return the starting character offset of a unique anchor in ``text``."""
    return find_unique_span(text, anchor, message=message)[0]


def replace_once(
    text: str,
    anchor: str,
    replacement: str,
    *,
    after: bool = False,
    insert: bool = False,
    message: str | None = None,
) -> str:
    """Replace or insert relative to a unique anchor in ``text``."""
    start, end = find_unique_span(text, anchor, message=message)
    if insert:
        pos = end if after else start
        return text[:pos] + replacement + text[pos:]
    return text[:start] + replacement + text[end:]


def _line_char_offsets(text: str) -> list[tuple[int, int]]:
    """Return 1-indexed line offsets ``[(line_start, line_end_no_nl), ...]``."""
    offsets: list[tuple[int, int]] = [(0, 0)]  # dummy index 0
    pos = 0
    for line in text.splitlines(keepends=True):
        no_nl = line.rstrip("\r\n")
        offsets.append((pos, pos + len(no_nl)))
        pos += len(line)
    return offsets


def find_routine_span(text: str, name: str, *, kind: str = "subroutine",
                      undefined: tuple[str, ...] = ()) -> tuple[int, int]:
    """Return ``(start_char, end_char)`` for the named Fortran subroutine or function."""
    target = name.lower()
    sub_kind = "Subroutine_Subprogram" if kind == "subroutine" else "Function_Subprogram"
    stmt_kind = "Subroutine_Stmt" if kind == "subroutine" else "Function_Stmt"
    end_kind = "End_Subroutine_Stmt" if kind == "subroutine" else "End_Function_Stmt"

    try:
        tree = parse_fortran(text, undefined=undefined)
    except Exception:
        tree = None

    if tree is not None:
        top_matches: list[tuple[int, int]] = []
        all_matches: list[tuple[int, int]] = []
        offsets = _line_char_offsets(text)
        for node in walk(tree):
            if type(node).__name__ != sub_kind:
                continue
            stmt = next((c for c in node.content if type(c).__name__ == stmt_kind), None)
            end_stmt = next((c for c in reversed(node.content) if type(c).__name__ == end_kind), None)
            if stmt is None or end_stmt is None or len(stmt.items) < 2:
                continue
            if str(stmt.items[1]).lower() != target:
                continue
            s_line = stmt.item.span[0]
            e_line = end_stmt.item.span[1]
            char_span = (offsets[s_line][0], offsets[e_line][1])
            all_matches.append(char_span)
            parent_type = type(getattr(node, "parent", None)).__name__
            if parent_type not in {"Internal_Subprogram_Part"}:
                top_matches.append(char_span)
        chosen = top_matches if top_matches else all_matches
        if len(chosen) == 1:
            return chosen[0]
        if len(chosen) > 1:
            raise ValueError(f"Expected one complete original {name} procedure")

    # Fallback line scanner if raw syntax contains unsupported preprocessor blocks.
    start_pat = re.compile(
        rf"(?im)^[ \t]*(?:(?:pure|elemental|recursive|impure|module)\s+)*{kind}\s+{re.escape(name)}\b(?:\s*\(|[ \t]*(?:!.*)?$)"
    )
    starts = list(start_pat.finditer(text))
    if len(starts) == 1:
        named_end = re.search(
            rf"(?im)^[ \t]*end\s*{kind}\s+{re.escape(name)}\b[^\n]*",
            text[starts[0].end():],
        )
        end = named_end or re.search(
            rf"(?im)^[ \t]*end\s*{kind}[ \t]*(?:!.*)?$",
            text[starts[0].end():],
        )
        if end is not None:
            return starts[0].start(), starts[0].end() + end.end()
    raise ValueError(f"Expected one complete original {name} procedure")


def _split_top_level_args(arg_str: str) -> list[str]:
    args: list[str] = []
    cur: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    i = 0
    while i < len(arg_str):
        ch = arg_str[i]
        if ch == "'" and not in_double:
            in_single = not in_single
            cur.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            cur.append(ch)
        elif not in_single and not in_double:
            if ch in "([":
                depth += 1
                cur.append(ch)
            elif ch in ")]":
                depth = max(0, depth - 1)
                cur.append(ch)
            elif ch == "&":
                # Skip continuation marker and following newline/comment/leading &
                rest = arg_str[i + 1 :]
                nl = rest.find("\n")
                if nl != -1:
                    i += 1 + nl
                    after = arg_str[i + 1 :].lstrip(" \t")
                    skip = len(arg_str[i + 1 :]) - len(after)
                    if after.startswith("&"):
                        skip += 1
                    i += skip
            elif ch == "," and depth == 0:
                args.append(" ".join("".join(cur).split()))
                cur = []
                i += 1
                continue
            else:
                cur.append(ch)
        else:
            cur.append(ch)
        i += 1
    tail = " ".join("".join(cur).split())
    if tail or args:
        args.append(tail)
    return args


def find_all_calls(text: str, callee_name: str) -> list[tuple[int, int, str, list[str]]]:
    """Find all non-comment ``call <callee_name>(...)`` occurrences in ``text``."""
    mask = _comment_mask(text)
    pattern = re.compile(rf"(?im)(?:^[ \t]*|(?<=\))[ \t]*)call\s+{re.escape(callee_name)}\b")
    results: list[tuple[int, int, str, list[str]]] = []
    n = len(text)
    for m in pattern.finditer(text):
        call_start = m.start()
        if call_start < len(mask) and mask[call_start]:
            continue
        pos = m.end()
        # Skip whitespace and line continuations between callee name and '('
        while pos < n:
            if text[pos] in " \t":
                pos += 1
            elif text[pos] == "&":
                nl = text.find("\n", pos)
                if nl == -1:
                    pos = n
                else:
                    pos = nl + 1
                    while pos < n and text[pos] in " \t&":
                        pos += 1
            else:
                break
        if pos >= n or text[pos] != "(":
            # Argumentless CALL
            end_line = text.find("\n", m.end())
            stmt_end = n if end_line == -1 else end_line
            results.append((call_start, stmt_end, text[call_start:stmt_end], []))
            continue
        depth = 0
        in_single = False
        in_double = False
        in_comment = False
        arg_start = pos + 1
        end_pos = None
        i = pos
        while i < n:
            ch = text[i]
            if ch == "\n":
                in_comment = False
                i += 1
                continue
            if in_comment:
                i += 1
                continue
            if ch == "'" and not in_double:
                if in_single and i + 1 < n and text[i + 1] == "'":
                    i += 2
                    continue
                in_single = not in_single
            elif ch == '"' and not in_single:
                if in_double and i + 1 < n and text[i + 1] == '"':
                    i += 2
                    continue
                in_double = not in_double
            elif not in_single and not in_double:
                if ch == "!":
                    in_comment = True
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break
            i += 1
        if end_pos is None:
            continue
        args = _split_top_level_args(text[arg_start : end_pos - 1])
        results.append((call_start, end_pos, text[call_start:end_pos], args))
    return results


def find_call_span(
    text: str,
    callee_name: str,
    *,
    within_routine: str | None = None,
) -> tuple[int, int, str, list[str]]:
    """Locate a unique ``call <callee_name>(...)`` statement."""
    if within_routine is not None:
        r_lo, r_hi = find_routine_span(text, within_routine)
        sub = find_all_calls(text[r_lo:r_hi], callee_name)
        if len(sub) != 1:
            raise ValueError(f"Expected one call to {callee_name} in {within_routine}")
        s, e, raw, args = sub[0]
        return r_lo + s, r_lo + e, raw, args
    matches = find_all_calls(text, callee_name)
    if len(matches) != 1:
        raise ValueError(f"Expected one call to {callee_name}")
    return matches[0]
