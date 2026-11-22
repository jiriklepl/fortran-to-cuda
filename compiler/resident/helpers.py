"""The procedures that selected procedures call, written in a form a loop nest can take in.

A loop nest that calls a procedure can only be an entry when the procedure's
statements can be inlined into it: the same statements, in a form the device
code is made from. Three things in such a procedure are rewritten, in the
prepared source and only there:

* A call that leaves out optional arguments gets a copy of the procedure of
  its own, without the absent dummies. ``present`` of each dummy is the
  constant it is at that call, and the branches those constants decide are
  folded away::

      call WMFlatStress(ustar, dist, vel, wall, tan_vel, distance)

  calls ``wmflatstress_fort_1``, which has no ``tan_vect`` and in which
  ``if (present(tan_vect)) tan_vect = tan_vect_loc`` is gone. The copy stands
  right after the procedure. It is made only where nothing names an absent
  dummy once the branches are folded.
* A RETURN that leaves before the end becomes the end of a branch whose ELSE
  holds the statements that follow. That is done where it is exact: each
  RETURN is the last statement of an IF branch, or a one-line IF, outside
  loops, and what follows is needed in one branch only.
* A loop without control that counts its iterations up to a cap, such as an
  iteration to convergence that gives up after fifty tries, becomes a counted
  loop around its body, which runs while a flag says the loop has not been
  left; each EXIT clears the flag and the statements after it are guarded by
  it. The two run the same statements in the same order.

The calls followed are those in the selected procedures and in the procedures
of the same source they call, to procedures of that source; the selected
procedures themselves keep their statements. What cannot be rewritten stays as
it is, with the reason in its record.
"""
import re
from fnmatch import fnmatchcase

from fparser.two.utils import walk

from .anchors import _strip_inline_comment, parse_fortran
from .reductions import _heading, _kind, _part, _procedures, _span
from .specialize import _fold

_PRESENT = re.compile(r"(?i)\bpresent\s*\(\s*([a-z]\w*)\s*\)")
# Each rewrite takes one step per round; a procedure has few of them.
_ROUNDS = 64


class _Literals:
    """Decides conditions made of the logical constants that ``present`` became."""

    def decide(self, node):
        kind = _kind(node)
        if kind == "Logical_Literal_Constant":
            return str(node.items[0]).upper() == ".TRUE."
        if kind == "Parenthesis":
            return self.decide(node.items[1])
        if kind == "And_Operand" and str(node.items[0]).upper() == ".NOT.":
            value = self.decide(node.items[1])
            return None if value is None else not value
        if kind in {"Or_Operand", "Equiv_Operand"} and str(node.items[1]).upper() in {".AND.", ".OR."}:
            both = str(node.items[1]).upper() == ".AND."
            sides = [self.decide(node.items[0]), self.decide(node.items[2])]
            if (not both) in sides:
                return not both
            return both if sides == [both, both] else None
        return None


def _optional_dummies(scope):
    """The dummies of a procedure, in order, and those of them that are OPTIONAL."""
    heading = _heading(scope.node)
    dummies = [str(name).lower() for name in walk(heading.items[2], type(heading.items[1]))] \
        if len(heading.items) > 2 and heading.items[2] is not None else []
    optional = set()
    specification = _part(scope.node, "Specification_Part")
    for statement in (specification.content if specification is not None else ()):
        if _kind(statement) == "Type_Declaration_Stmt" and statement.items[1] is not None \
                and any(str(item).upper() == "OPTIONAL" for item in statement.items[1].items):
            optional |= {str(entity.items[0]).lower() for entity in statement.items[2].items}
        elif _kind(statement) == "Optional_Stmt":
            optional |= {str(name).lower() for name in walk(statement, type(heading.items[1]))}
    return dummies, optional & set(dummies)


def _resolve(scope, name, procedures):
    """The procedure of this source a call in ``scope`` names, innermost scope first."""
    qualified = scope.qualified
    while qualified:
        found = procedures.get(qualified + "::" + name)
        if found is not None:
            return found
        qualified = qualified.rpartition("::")[0]
    return None


def _given(call, dummies):
    """The dummies a call gives an actual argument to, or None for an association that is not plain."""
    arguments = list(getattr(call.items[1], "items", ())) if call.items[1] is not None else []
    given, keywords = set(), False
    for position, argument in enumerate(arguments):
        if _kind(argument) == "Actual_Arg_Spec":
            keywords = True
            given.add(str(argument.items[0]).lower())
        elif keywords or position >= len(dummies):
            return None
        else:
            given.add(dummies[position])
    return given if given <= set(dummies) else None


def _copy(scope, lines, name, absent, optional):
    """The text of the procedure as a version without ``absent``, called ``name``; None where that is not exact."""
    first, last = _span(scope.node)
    heading = _heading(scope.node)
    head_first, head_last = _span(heading)
    original = str(heading.items[1])
    text = "".join(lines[head_first - 1:head_last])
    pattern = re.compile(r"(?is)(\b(?:subroutine|function)\s+)" + re.escape(original) + r"(\s*\()([^)]*)(\))")
    match = pattern.search(text)
    if match is None:
        return None
    kept = [item.strip() for item in match.group(3).replace("&", " ").split(",") if item.strip()
            and item.strip().lower() not in absent]
    header = text[:match.start()] + match.group(1) + name + match.group(2) + ", ".join(kept) + match.group(4) \
        + text[match.end():]
    output = [header if header.endswith("\n") else header + "\n"]
    specification = _part(scope.node, "Specification_Part")
    statements = list(specification.content) if specification is not None else []
    covered = head_last
    for statement in statements:
        span = _span(statement) if _kind(statement) != "Comment" else None
        if span is None:
            continue
        output += lines[covered:span[0] - 1]
        covered = span[1]
        if _kind(statement) == "Optional_Stmt":
            continue
        if _kind(statement) != "Type_Declaration_Stmt" or not (
                {str(entity.items[0]).lower() for entity in statement.items[2].items} & (absent | optional)):
            output += lines[span[0] - 1:span[1]]
            continue
        entities = [entity for entity in statement.items[2].items if str(entity.items[0]).lower() not in absent]
        if not entities:
            continue
        attributes = [str(item) for item in (statement.items[1].items if statement.items[1] is not None else ())
                      if str(item).upper() != "OPTIONAL"]
        indent = lines[span[0] - 1][:len(lines[span[0] - 1]) - len(lines[span[0] - 1].lstrip())]
        output.append(indent + str(statement.items[0]) + "".join(", " + item for item in attributes)
                      + " :: " + ", ".join(str(entity) for entity in entities) + "\n")
    # The statements, with present() of each dummy the constant it is here, and what that decides folded away.
    execution = _part(scope.node, "Execution_Part")
    rest = lines[covered:last]
    if execution is not None:
        start, stop = _span(execution)
        body = [_PRESENT.sub(lambda found: (".false." if found.group(1).lower() in absent else ".true.")
                             if found.group(1).lower() in optional else found.group(0), line)
                for line in lines[start - 1:stop]]
        wrapped = parse_fortran("subroutine fort_fold\n" + "".join(body) + "end subroutine\n")
        folded_execution = _part(next(node for node in walk(wrapped) if _kind(node) == "Subroutine_Subprogram"),
                                 "Execution_Part")
        folded = _fold(list(folded_execution.content), ["\n", *body], _Literals(), [0], one_line=True) \
            if folded_execution is not None else []
        rest = [*lines[covered:start - 1], *folded, *lines[stop:last]]
    ending = rest[-1]
    rest[-1] = re.sub(r"(?i)(\bend\s*(?:subroutine|function)\s+)" + re.escape(original) + r"\b",
                      lambda found: found.group(1) + name, ending)
    output += rest
    used = re.compile(r"(?i)(?<![\w%])(" + "|".join(map(re.escape, sorted(absent))) + r")(?!\w)")
    if any(used.search(_strip_inline_comment(line.rstrip("\n"))[0]) for line in output[1:]):
        return None
    return output


def _returns(node):
    return sum(1 for item in walk(node) if _kind(item) == "Return_Stmt")


def _branches(construct):
    """The parts of an IF construct: (opening statement, body), and the END IF statement."""
    branches, end = [], None
    for item in construct.content:
        if _kind(item) in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt"}:
            branches.append([item, []])
        elif _kind(item) == "End_If_Stmt":
            end = item
        elif branches and _kind(item) != "Comment":
            branches[-1][1].append(item)
    return branches, end


def _return_edit(statements, lines):
    """One edit that takes a RETURN out of a statement list that ends where the procedure does, or None.

    Returns (first, last, replacement lines), or a string with the reason when a
    RETURN stands where it cannot be taken out.
    """
    statements = [item for item in statements if _kind(item) != "Comment"]
    for position, statement in enumerate(statements):
        if not _returns(statement):
            continue
        last = position == len(statements) - 1
        first_line, last_line = _span(statement)
        if _kind(statement) == "Return_Stmt":
            # Nothing after it runs: it and what follows go.
            return first_line, _span(statements[-1])[1], []
        if _kind(statement) == "If_Construct":
            branches, end = _branches(statement)
            if last:
                # Each branch ends where the procedure does.
                for _header, body in branches:
                    edit = _return_edit(body, lines)
                    if edit is not None:
                        return edit
                return "a RETURN in a branch cannot be taken out"
            returning, staying = [], []
            for header, body in branches:
                if not _returns(header) and body and _kind(body[-1]) == "Return_Stmt" and _returns_list(body) == 1:
                    returning.append((header, body))
                elif _returns_list(body):
                    return "a RETURN inside a branch is not its last statement"
                else:
                    staying.append((header, body))
            explicit_else = any(_kind(header) == "Else_Stmt" for header, _ in branches)
            if len(staying) + (not explicit_else) != 1:
                return "the statements after an early RETURN would be needed in two branches"
            rest = lines[last_line:_span(statements[-1])[1]]
            replacement = []
            cursor = first_line
            ending = _span(end)[0]
            for header, body in branches:
                if (header, body) in returning:
                    stop = _span(body[-1])
                    replacement += lines[cursor - 1:stop[0] - 1]
                    cursor = stop[1] + 1
                elif (header, body) in staying:
                    following = next((_span(other)[0] for other, _ in branches if _span(other)[0] > _span(header)[0]),
                                     ending)
                    replacement += lines[cursor - 1:following - 1] + rest
                    cursor = following
            if not explicit_else:
                indent = lines[ending - 1][:len(lines[ending - 1]) - len(lines[ending - 1].lstrip())]
                replacement += lines[cursor - 1:ending - 1] + [indent + "else\n"] + rest
                cursor = ending
            replacement += lines[cursor - 1:last_line]
            return first_line, _span(statements[-1])[1], replacement
        if _kind(statement) == "If_Stmt" and _kind(statement.items[1]) == "Return_Stmt":
            indent = lines[first_line - 1][:len(lines[first_line - 1]) - len(lines[first_line - 1].lstrip())]
            rest = lines[last_line:_span(statements[-1])[1]]
            return first_line, _span(statements[-1])[1], [
                f"{indent}if (.not. ({statement.items[0]})) then\n", *rest, f"{indent}end if\n"]
        return "a RETURN inside a loop or another construct cannot be taken out"
    return None


def _flagged(statements, lines, flag):
    """The lines of ``statements`` with each RETURN setting ``flag``, and what may follow one guarded by it.

    None where a RETURN stands in a loop or in another construct than IF.
    """
    output = []
    statements = [item for item in statements if _kind(item) != "Comment"]
    for position, statement in enumerate(statements):
        first, last = _span(statement)
        indent = lines[first - 1][:len(lines[first - 1]) - len(lines[first - 1].lstrip())]
        if not _returns(statement):
            output += lines[first - 1:last]
            continue
        if _kind(statement) == "Return_Stmt":
            # Nothing after it in this list runs.
            return output + [f"{indent}{flag} = .true.\n"]
        if _kind(statement) == "If_Stmt" and _kind(statement.items[1]) == "Return_Stmt":
            output.append(f"{indent}if ({statement.items[0]}) {flag} = .true.\n")
        elif _kind(statement) == "If_Construct":
            branches, end = _branches(statement)
            for header, body in branches:
                inner = _flagged(body, lines, flag)
                if inner is None:
                    return None
                output += lines[_span(header)[0] - 1:_span(header)[1]] + inner
            output += lines[_span(end)[0] - 1:_span(end)[1]]
        else:
            return None
        if position < len(statements) - 1:
            rest = _flagged(statements[position + 1:], lines, flag)
            if rest is None:
                return None
            return output + [f"{indent}if (.not. {flag}) then\n", *rest, f"{indent}end if\n"]
    return output


def _returns_list(statements):
    return sum(_returns(item) for item in statements)


_RELATION = {">=": 0, ".GE.": 0, ">": 1, ".GT.": 1, "==": 0, ".EQ.": 0}


def _exits(node):
    """EXIT statements that leave the loop ``node`` belongs to: those outside loops nested in it."""
    found = []
    for item in getattr(node, "content", None) or getattr(node, "items", None) or ():
        if _kind(item) == "Exit_Stmt":
            found.append(item)
        elif _kind(item) not in {"Block_Nonlabel_Do_Construct", "Block_Label_Do_Construct"} and not isinstance(item, str):
            found += _exits(item)
    return found


def _trip_bound(statements, loop, body):
    """How many iterations a loop without control can make at most; None if that is not known.

    The loop counts: its body adds one to an INTEGER variable at the top
    level, the last statement before the loop that sets that variable sets it
    to a constant, and a statement at the top level of the body, after the
    addition, leaves the loop once the variable reaches a constant.
    """
    before = [item for item in statements[:statements.index(loop)] if _kind(item) != "Comment"]
    for increment in body:
        if _kind(increment) != "Assignment_Stmt" or _kind(increment.items[0]) != "Name":
            continue
        counter = str(increment.items[0]).lower()
        if str(increment.items[2]).replace(" ", "").lower() not in {f"{counter}+1", f"1+{counter}"}:
            continue
        if sum(1 for item in walk(body) if _kind(item) == "Assignment_Stmt"
               and str(item.items[0]).lower() == counter) != 1:
            continue
        setting = [item for item in before if any(
            _kind(node) == "Assignment_Stmt" and str(node.items[0]).lower() == counter for node in walk(item))]
        if not setting or _kind(setting[-1]) != "Assignment_Stmt" \
                or any(_kind(node) == "Call_Stmt" for item in before[before.index(setting[-1]):] for node in walk(item)):
            continue
        start = str(setting[-1].items[2]).replace(" ", "")
        if not re.fullmatch(r"-?\d+", start):
            continue
        for item in body[body.index(increment) + 1:]:
            condition = None
            if _kind(item) == "If_Stmt" and _kind(item.items[1]) == "Exit_Stmt":
                condition = item.items[0]
            elif _kind(item) == "If_Construct":
                branches, _end = _branches(item)
                if len(branches) == 1 and branches[0][1] and _kind(branches[0][1][-1]) == "Exit_Stmt":
                    condition = branches[0][0].items[0]
            cap = _cap(condition, counter) if condition is not None else None
            if cap is not None:
                trips = cap - int(start)
                return trips if trips >= 1 else None
    return None


def _cap(condition, counter):
    """The value of ``counter`` at which ``condition`` is surely true, or None: ``i >= 50``, ``i >= 50 .or. x > 1``."""
    while _kind(condition) == "Parenthesis":
        condition = condition.items[1]
    if _kind(condition) in {"Or_Operand", "Equiv_Operand"} and str(condition.items[1]).upper() == ".OR.":
        found = [_cap(side, counter) for side in (condition.items[0], condition.items[2])]
        found = [value for value in found if value is not None]
        return min(found) if found else None
    if _kind(condition) == "Level_4_Expr" and str(condition.items[0]).lower() == counter \
            and str(condition.items[1]).upper() in _RELATION \
            and re.fullmatch(r"-?\d+", str(condition.items[2]).replace(" ", "")):
        return int(str(condition.items[2])) + _RELATION[str(condition.items[1]).upper()]
    return None


def _counted_loop(statements, lines, number):
    """One edit that writes a loop without control, that counts its iterations up to a cap, as a counted loop."""
    for statement in statements:
        if _kind(statement) not in {"If_Construct", "Block_Nonlabel_Do_Construct"}:
            continue
        if _kind(statement) == "If_Construct":
            for _header, body in _branches(statement)[0]:
                edit = _counted_loop(body, lines, number)
                if edit is not None:
                    return edit
            continue
        opening = next(item for item in statement.content if _kind(item) == "Nonlabel_Do_Stmt")
        inside = [item for item in statement.content
                  if _kind(item) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"}]
        if opening.items[1] is not None:
            edit = _counted_loop(inside, lines, number)
            if edit is not None:
                return edit
            continue
        body = inside
        trips = _trip_bound(statements, statement, body)
        if trips is None or any(_kind(item) in {"Cycle_Stmt", "Goto_Stmt", "Return_Stmt"} for item in walk(statement)):
            return "a loop without control whose iterations are not counted"
        flag, trip = f"fort_go_{number}", f"fort_trip_{number}"
        indent = lines[_span(opening)[0] - 1][:len(lines[_span(opening)[0] - 1]) - len(lines[_span(opening)[0] - 1].lstrip())]
        inner = []
        open_guards = 0
        for position, item in enumerate(body):
            first, last = _span(item)
            text = lines[first - 1:last]
            leaves = _exits(item)
            if leaves:
                if _kind(item) == "Exit_Stmt":
                    text = [f"{indent}    {flag} = .false.\n"]
                elif _kind(item) == "If_Stmt" and _kind(item.items[1]) == "Exit_Stmt":
                    text = [f"{indent}    if ({item.items[0]}) {flag} = .false.\n"]
                elif _kind(item) == "If_Construct" and all(
                        not _exits_list(body_part) or (_kind(body_part[-1]) == "Exit_Stmt"
                                                       and len(_exits_list(body_part)) == 1)
                        for _header, body_part in _branches(item)[0]):
                    text = list(text)
                    for leave in leaves:
                        line = _span(leave)[0] - first
                        text[line] = re.sub(r"(?i)\bexit\b", f"{flag} = .false.", text[line], count=1)
                else:
                    return "an EXIT that is not the last statement of a branch"
            inner += text
            if leaves and position < len(body) - 1:
                inner.append(f"{indent}    if ({flag}) then\n")
                open_guards += 1
        inner += [f"{indent}    end if\n"] * open_guards
        replacement = [f"{indent}{flag} = .true.\n", f"{indent}do {trip} = 1, {trips}\n",
                       f"{indent}  if ({flag}) then\n", *inner, f"{indent}  end if\n", f"{indent}end do\n"]
        return _span(statement)[0], _span(statement)[1], replacement, (flag, trip)
    return None


def _exits_list(statements):
    return [leave for item in statements for leave in ([item] if _kind(item) == "Exit_Stmt" else _exits(item))]


def _reached(text, patterns):
    """The parsed text, its procedures by qualified name, and those the selected procedures reach by calls."""
    tree = parse_fortran(text)
    scopes = {scope.qualified: scope for scope in _procedures(tree)}
    reached, pending = set(), [scope for name, scope in scopes.items() if any(fnmatchcase(name, p) for p in patterns)]
    while pending:
        scope = pending.pop()
        if scope.qualified in reached:
            continue
        reached.add(scope.qualified)
        execution = _part(scope.node, "Execution_Part")
        for call in walk(execution) if execution is not None else ():
            if _kind(call) == "Call_Stmt" and _kind(call.items[0]) == "Name":
                callee = _resolve(scope, str(call.items[0]).lower(), scopes)
                if callee is not None:
                    pending.append(callee)
    return scopes, reached


def _optional_versions(text, patterns, versions):
    """One round of copies for calls that leave out optional arguments: the new text and the records."""
    lines = text.splitlines(keepends=True)
    scopes, reached = _reached(text, patterns)
    edits, records = [], []
    for qualified in sorted(reached):
        scope = scopes[qualified]
        execution = _part(scope.node, "Execution_Part")
        for call in walk(execution) if execution is not None else ():
            if _kind(call) != "Call_Stmt" or _kind(call.items[0]) != "Name":
                continue
            callee = _resolve(scope, str(call.items[0]).lower(), scopes)
            if callee is None:
                continue
            dummies, optional = _optional_dummies(callee)
            given = _given(call, dummies)
            if not optional or given is None or optional <= given:
                continue
            absent = frozenset(optional - given)
            first, _last = _span(call)
            record = {"caller": qualified, "callee": callee.qualified, "line": first, "absent": sorted(absent)}
            name = str(call.items[0])
            if not re.match(r"(?i)\s*call\s+" + re.escape(name) + r"\b", lines[first - 1]):
                records.append({**record, "reason": "the call is not a statement of its own"})
                continue
            key = (callee.qualified, absent)
            if key not in versions:
                base = callee.qualified.rsplit("::", 1)[-1][:50]
                taken = {item.rsplit("::", 1)[-1] for item in scopes} | {value for value in versions.values()}
                number = 1
                while f"{base}_fort_{number}" in taken:
                    number += 1
                copy = _copy(callee, lines, f"{base}_fort_{number}", absent, optional)
                if copy is None:
                    records.append({**record, "reason": "the statements name an absent dummy"})
                    continue
                versions[key] = f"{base}_fort_{number}"
                edits.append((_span(callee.node)[1] + 1, _span(callee.node)[1], ["\n", *copy]))
            changed = re.sub(r"(?i)(\s*call\s+)" + re.escape(name) + r"\b",
                             lambda found, version=versions[key]: found.group(1) + version, lines[first - 1], count=1)
            edits.append((first, first, [changed]))
            records.append({**record, "version": versions[key]})
    for first, last, replacement in sorted(edits, key=lambda edit: (edit[0], edit[1]), reverse=True):
        lines[first - 1:last] = replacement
    return "".join(lines), records


def _declare(scope, lines, names):
    """An edit that declares ``names`` (Fortran declaration texts) after the procedure's specifications."""
    heading = _span(_heading(scope.node))[1]
    specification = _part(scope.node, "Specification_Part")
    ends = [_span(item)[1] for item in (specification.content if specification is not None else ())
            if _kind(item) != "Comment" and _span(item) is not None]
    after = max([heading, *ends])
    margin = " " * (len(lines[heading - 1]) - len(lines[heading - 1].lstrip()) + 2)
    return after + 1, after, [margin + item + "\n" for item in names]


def prepare_helpers(text, procedures):
    """Rewrite what the selected procedures call so that their nests can take it in.

    ``procedures`` are the selected ``module::name`` patterns. Returns the new
    text and one record per rewrite: optional versions (``caller``,
    ``callee``, ``line``, ``absent`` and the ``version`` called now), RETURN
    statements taken out (``procedure``, ``returns``) and counted loops
    (``procedure``, ``loop``), each with a ``reason`` instead where the
    rewrite cannot be made.
    """
    patterns = [pattern.lower() for pattern in procedures]
    if not any(pattern.split("::", 1)[0] in text.lower() for pattern in patterns):
        return text, []
    records, versions = {}, {}
    for _round in range(_ROUNDS):
        text, found = _optional_versions(text, patterns, versions)
        # A call that stays is seen again in every round, at a line that may have moved.
        records.update(((record["caller"], record["callee"], tuple(record["absent"]), "version" in record, record.get(
            "version") or 0), record) for record in found)
        if not any("version" in record for record in found):
            break
    records = list(records.values())
    refused = set()
    for _round in range(_ROUNDS):
        lines = text.splitlines(keepends=True)
        scopes, reached = _reached(text, patterns)
        edits = None
        for qualified in sorted(reached):
            scope = scopes[qualified]
            if any(fnmatchcase(qualified, pattern) for pattern in patterns) or qualified in refused:
                continue
            execution = _part(scope.node, "Execution_Part")
            if execution is None:
                continue
            if _returns(execution):
                found = _return_edit(list(execution.content), lines)
                if isinstance(found, str) or found is None:
                    # Where the statements after a RETURN cannot simply become a branch, a flag says it was reached.
                    taken = re.findall(r"(?i)\bfort_left_(\d+)\b", text)
                    flag = f"fort_left_{1 + max((int(item) for item in taken), default=0)}"
                    flagged = _flagged(list(execution.content), lines, flag)
                    if flagged is None:
                        records.append({"procedure": qualified, "reason": found or "a RETURN stays"})
                        refused.add(qualified)
                        continue
                    start, stop = _span(execution)
                    indent = lines[start - 1][:len(lines[start - 1]) - len(lines[start - 1].lstrip())]
                    found = (start, stop, [f"{indent}{flag} = .false.\n", *flagged])
                    edits = [found, _declare(scope, lines, [f"logical :: {flag}"])]
                else:
                    edits = [found]
                records.append({"procedure": qualified, "returns": True})
                break
            number = 1 + max((int(found) for found in re.findall(r"(?i)\bfort_trip_(\d+)\b", text)), default=0)
            found = _counted_loop(list(execution.content), lines, number)
            if isinstance(found, str):
                records.append({"procedure": qualified, "reason": found})
                refused.add(qualified)
                continue
            if found is not None:
                first, last, replacement, (flag, trip) = found
                edits = [(first, last, replacement), _declare(scope, lines, [f"logical :: {flag}", f"integer :: {trip}"])]
                records.append({"procedure": qualified, "loop": first})
                break
        if edits is None:
            break
        for first, last, replacement in sorted(edits, reverse=True):
            lines[first - 1:last] = replacement
        text = "".join(lines)
    return text, records
