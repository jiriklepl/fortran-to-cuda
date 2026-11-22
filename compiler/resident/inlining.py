"""Replace calls to a small procedure by its statements, so that its loop nests join the caller's.

A procedure that is called several times in a row with different arrays, such
as a halo update per field, gives one group of entries per call. Its nests
touch different arrays in each call and could run in the same launches, but
entries are formed per procedure. With the calls replaced by the procedure's
statements the nests stand next to each other in the caller, where the usual
grouping and the dependence proof decide what runs together.

The replacement is exact text substitution, so it is made only where that is
the whole story of the call:

* caller and callee are procedures of the same module, and the callee has no
  USE statements, internal procedures, RETURN or labelled statements;
* the callee declares only its dummies and scalar locals without attributes;
* each array dummy gets a whole array whose declared bounds in the caller are
  those of the dummy, so every subscript means the same element;
* each scalar dummy is INTENT(IN), is not assigned, and gets a name, a
  constant or an expression that the callee's statements do not change.

The callee's OpenMP directives are dropped as those of the caller are, with
their reduction clauses kept as ``!$fort reduction`` lines. A constant actual
argument decides the callee's branches on that dummy, which
are folded. A local that is assigned once, at the top of what remains, from
values the statements do not change is replaced by that value, so no scalar
assignment separates the nests. Other locals get names of their own in the
caller.
"""
import re
from fnmatch import fnmatchcase

from fparser.two.utils import walk

from compiler.frontend.syntax import kind_of, parse_fortran, part_of

from .anchors import _strip_inline_comment
from .configure import serialize_openmp
from .reductions import _heading, _names, _procedures, _rename, _span, holds_selected
from .specialize import _Conditions, _fold

_LITERAL = re.compile(r"[+-]?\d+\Z")


def _normal(text):
    return re.sub(r"\s+", "", str(text)).lower()


def _declarations(node):
    """name -> (type text, attribute text, array bounds text or None) for the entities a procedure declares."""
    found = {}
    specification = part_of(node, "Specification_Part")
    for statement in (specification.content if specification is not None else ()):
        if kind_of(statement) != "Type_Declaration_Stmt":
            continue
        dtype, attributes, entities = statement.items
        attrs = str(attributes).lower() if attributes is not None else ""
        shared = re.search(r"dimension\s*\((.*)\)", attrs)
        for entity in entities.items:
            bounds = entity.items[1] if entity.items[1] is not None else (shared.group(1) if shared else None)
            found[str(entity.items[0]).lower()] = (
                str(dtype), re.sub(r"dimension\s*\(.*\)", "", attrs), None if bounds is None else _normal(bounds),
                entity.items[3] is not None)
    return found


class _Callee:
    """What a procedure looks like to a caller that takes its statements; ``reason`` says why it cannot."""

    def __init__(self, scope, lines):
        self.scope, self.reason = scope, None
        node = scope.node
        statement = part_of(node, "Subroutine_Stmt")
        self.execution = part_of(node, "Execution_Part")
        if statement is None or self.execution is None:
            self.reason = "it is not a subroutine with statements"
            return
        self.dummies = [str(item).lower() for item in getattr(statement.items[2], "items", ())]
        self.declared = _declarations(node)
        specification = part_of(node, "Specification_Part")
        kinds = {kind_of(item) for item in (specification.content if specification is not None else ())}
        if part_of(node, "Internal_Subprogram_Part") is not None or kinds - {
                "Type_Declaration_Stmt", "Comment", "Implicit_Part"}:
            self.reason = "it has more than declarations and statements"
            return
        for name, (_dtype, attributes, bounds, initialized) in self.declared.items():
            if name in self.dummies:
                if bounds is None and "intent(in)" not in _normal(attributes):
                    self.reason = f"its scalar argument {name} is not INTENT(IN)"
            elif bounds is not None or initialized or attributes.replace(",", "").strip():
                self.reason = f"its local {name} is more than a plain scalar"
        if set(self.dummies) - set(self.declared):
            self.reason = "it does not declare all its arguments"
        assigned = set()
        for item in walk(self.execution):
            kind = kind_of(item)
            if kind in {"Return_Stmt", "Stop_Stmt", "Goto_Stmt", "Cycle_Stmt", "Exit_Stmt"}:
                self.reason = "it has a " + kind.removesuffix("_Stmt").upper() + " statement"
            if getattr(getattr(item, "item", None), "label", None) is not None:
                self.reason = "it has a labelled statement"
            if kind == "Assignment_Stmt" and kind_of(item.items[0]) == "Name":
                assigned.add(str(item.items[0]).lower())
            if kind == "Nonlabel_Do_Stmt" and item.items[1] is not None and item.items[1].items[1] is not None:
                assigned.add(str(item.items[1].items[1][0]).lower())
            if kind == "Call_Stmt":
                # What a call changes is not visible here; the arguments could be the dummies.
                assigned.update(_names(item.items[1]))
        self.assigned = assigned
        scalar_dummies = {name for name in self.dummies if self.declared.get(name, (None, None, None))[2] is None}
        if self.reason is None and scalar_dummies & assigned:
            self.reason = "it changes a scalar argument"
        self.locals = {name: dtype for name, (dtype, _a, _b, _i) in self.declared.items() if name not in self.dummies}
        self.lines = lines


def _substituted(lines, callee, indent):
    """Replace locals that are assigned once, at the top, from values the statements leave alone."""
    code = [_strip_inline_comment(line.rstrip("\n"))[0] for line in lines]
    base = min((len(line) - len(line.lstrip()) for line in code if line.strip()), default=0)
    replaced, kept, depth = {}, [], 0
    for index, line in enumerate(code):
        # Only a statement outside every construct runs each time; folding leaves branch bodies indented.
        statement = line.strip().lower()
        closes = bool(re.match(r"end\s*(if|do|select|block|where|associate)\b", statement))
        opens = bool(re.match(r"(\w+\s*:\s*)?(do\b|select\b|block\b|associate\b|where\s*\(.*\)\s*$)", statement)
                     or re.match(r"(\w+\s*:\s*)?if\s*\(.*\)\s*then$", statement))
        depth -= closes
        match = re.match(r"\s*([A-Za-z]\w*)\s*=(?!=)\s*(.+?)\s*$", line)
        name = match.group(1).lower() if match else None
        if (match and depth == 0 and name in callee.locals
                and sum(bool(re.match(rf"(?i)\s*(?:do\s+)?{re.escape(name)}\s*=(?!=)", other)) for other in code) == 1
                and not any(re.search(rf"(?i)(?<![\w%]){re.escape(name)}(?!\w)", other) for other in code[:index])
                and not {item.lower() for item in re.findall(r"[A-Za-z]\w*", match.group(2))} & callee.assigned):
            value = match.group(2)
            replaced[name] = value if re.fullmatch(r"\w+", value) else f"({value})"
        else:
            kept.append(lines[index])
        depth += opens
    kept = _rename(kept, replaced)
    return [indent + line[base:] if line.strip() else line for line in kept], set(replaced)


def inline_calls(text, callers, callees, constants=None):
    """Replace calls to ``callees`` in the procedures matching ``callers`` by the callees' statements.

    Both are ``module::name`` values (``callers`` may be globs). ``constants``
    maps named integer constants to their values, for folding the branches a
    constant argument decides. Returns the new text and one record per call:
    ``caller``, ``callee``, ``line`` and either ``inlined`` or the ``reason``
    the call stays.
    """
    wanted = {name.lower() for name in callees}
    if not holds_selected(text, wanted):
        return text, []
    lines = text.splitlines(keepends=True)
    tree = parse_fortran(text)
    scopes = list(_procedures(tree))
    available = {scope.qualified: _Callee(scope, lines) for scope in scopes if scope.qualified in wanted}
    patterns = [pattern.lower() for pattern in callers]
    edits, records, number = [], [], 0
    for scope in scopes:
        if not any(fnmatchcase(scope.qualified, pattern) for pattern in patterns) or scope.qualified in available:
            continue
        execution = part_of(scope.node, "Execution_Part")
        if execution is None:
            continue
        module = scope.qualified.split("::", 1)[0]
        # A module's own arrays are whole arrays too; the caller's declarations come first.
        declared, added = {**_declarations(scope.hosts[-1]), **_declarations(scope.node)}, []
        for call in walk(execution):
            if kind_of(call) != "Call_Stmt":
                continue
            callee = available.get(module + "::" + str(call.items[0]).lower())
            if callee is None:
                continue
            first, last = _span(call)
            record = {"caller": scope.qualified, "callee": callee.scope.qualified, "line": first}
            arguments = list(getattr(call.items[1], "items", ())) if call.items[1] is not None else []
            reason = callee.reason
            if reason is None and (not re.match(r"(?i)\s*call\b", lines[first - 1])
                                   or _strip_inline_comment(lines[last - 1].rstrip("\n"))[0].rstrip()[-1:] != ")"):
                reason = "the call is not a statement of its own"
            if reason is None and (len(arguments) != len(callee.dummies)
                                   or any(kind_of(item) == "Actual_Arg_Spec" for item in arguments)):
                reason = "the arguments are not all given, by position"
            mapping, facts = {}, []
            for dummy, actual in zip(callee.dummies, arguments, strict=False):
                if reason is not None:
                    break
                bounds = callee.declared[dummy][2]
                value = " ".join(str(actual).split())
                if bounds is not None:
                    mine = declared.get(value.lower())
                    if kind_of(actual) != "Name" or mine is None or mine[2] != bounds:
                        reason = f"{value} is not a whole array declared with the bounds of {dummy}"
                    mapping[dummy] = value
                elif {name for name in _names(actual)} & callee.assigned:
                    reason = f"the statements change what the argument {value} names"
                else:
                    mapping[dummy] = value if re.fullmatch(r"[\w.+-]+", value) else f"({value})"
                    if _LITERAL.match(value):
                        facts.append(f"{dummy} == {value}")
            if reason is not None:
                records.append({**record, "reason": reason})
                continue
            number += 1
            body = list(callee.execution.content)
            if facts:
                body_lines = _fold(body, lines, _Conditions(facts, constants or {}), [0])
            else:
                start, stop = _span(callee.execution)
                body_lines = lines[start - 1:stop]
            indent = lines[first - 1][:len(lines[first - 1]) - len(lines[first - 1].lstrip())]
            # The caller is a selected procedure: what it takes in runs serially like its own statements.
            body_lines = list(body_lines)
            serialize_openmp(body_lines, 0, len(body_lines))
            body_lines, gone = _substituted(body_lines, callee, indent)
            fresh = {name: f"fort_inl{number}_{name}" for name in callee.locals if name not in gone}
            used = {name for name in fresh if any(re.search(rf"(?i)(?<![\w%]){re.escape(name)}(?!\w)", line)
                                                  for line in body_lines)}
            body_lines = _rename(body_lines, {**{name: fresh[name] for name in used}, **mapping})
            added += [f"{callee.locals[name]} :: {fresh[name]}" for name in sorted(used)]
            edits.append((first, last, body_lines))
            records.append({**record, "inlined": True})
        if added:
            heading = _span(_heading(scope.node))[1]
            specification = part_of(scope.node, "Specification_Part")
            ends = [_span(item)[1] for item in (specification.content if specification is not None else ())
                    if kind_of(item) != "Comment" and _span(item) is not None]
            after = max([heading, *ends])
            margin = " " * (len(lines[heading - 1]) - len(lines[heading - 1].lstrip()) + 2)
            edits.append((after + 1, after, [margin + item + "\n" for item in added]))
    for first, last, replacement in sorted(edits, key=lambda edit: (edit[0], edit[1]), reverse=True):
        lines[first - 1:last] = replacement
    return "".join(lines), records
