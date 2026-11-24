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
* A local pointer that the first statements of a loop's body point at the
  element the loop variable selects, ``p => points(i)``, and that is used
  only for its fields, is replaced by that element: ``p%ustar`` becomes
  ``points(i)%ustar``. That is done in the selected procedures as well.
* A SELECT CASE on a variable becomes an IF construct with one condition per
  CASE and the CASE DEFAULT as its ELSE, in these procedures and in the
  selected ones.
* A loop without control that counts its iterations up to a cap, such as an
  iteration to convergence that gives up after fifty tries, becomes its body,
  for the first iteration, which always runs, followed by a counted loop over
  the others, whose body runs while a flag says the loop has not been left;
  each EXIT clears the flag and the statements after it are guarded by it.
  The two run the same statements in the same order.

The calls followed are those in the selected procedures and in the procedures
of the same source they call, to procedures of that source; the selected
procedures themselves keep their statements. What cannot be rewritten stays as
it is, with the reason in its record.
"""
import re
from fnmatch import fnmatchcase

from fparser.two.utils import walk

from compiler.frontend.syntax import children_of, kind_of, parse_fortran, part_of
from compiler.resident.procedures import declarations_end, heading_of, holds_selected, line_span, procedures_of

from .anchors import _strip_inline_comment
from .specialize import _fold

_PRESENT = re.compile(r"(?i)\bpresent\s*\(\s*([a-z]\w*)\s*\)")
# Each rewrite takes one step per round; a procedure has few of them.
_ROUNDS = 64


class _Literals:
    """Decides conditions made of the logical constants that ``present`` became."""

    def decide(self, node):
        kind = kind_of(node)
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
    heading = heading_of(scope.node)
    dummies = [str(name).lower() for name in walk(heading.items[2], type(heading.items[1]))] \
        if len(heading.items) > 2 and heading.items[2] is not None else []
    optional = set()
    specification = part_of(scope.node, "Specification_Part")
    for statement in (specification.content if specification is not None else ()):
        if kind_of(statement) == "Type_Declaration_Stmt" and statement.items[1] is not None \
                and any(str(item).upper() == "OPTIONAL" for item in statement.items[1].items):
            optional |= {str(entity.items[0]).lower() for entity in statement.items[2].items}
        elif kind_of(statement) == "Optional_Stmt":
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
        if kind_of(argument) == "Actual_Arg_Spec":
            keywords = True
            given.add(str(argument.items[0]).lower())
        elif keywords or position >= len(dummies):
            return None
        else:
            given.add(dummies[position])
    return given if given <= set(dummies) else None


def _copy(scope, lines, name, absent, optional):
    """The text of the procedure as a version without ``absent``, called ``name``; None where that is not exact."""
    first, last = line_span(scope.node)
    heading = heading_of(scope.node)
    head_first, head_last = line_span(heading)
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
    specification = part_of(scope.node, "Specification_Part")
    statements = list(specification.content) if specification is not None else []
    covered = head_last
    for statement in statements:
        span = line_span(statement) if kind_of(statement) != "Comment" else None
        if span is None:
            continue
        output += lines[covered:span[0] - 1]
        covered = span[1]
        if kind_of(statement) == "Optional_Stmt":
            continue
        if kind_of(statement) != "Type_Declaration_Stmt" or not (
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
    execution = part_of(scope.node, "Execution_Part")
    rest = lines[covered:last]
    if execution is not None:
        start, stop = line_span(execution)
        body = [_PRESENT.sub(lambda found: (".false." if found.group(1).lower() in absent else ".true.")
                             if found.group(1).lower() in optional else found.group(0), line)
                for line in lines[start - 1:stop]]
        wrapped = parse_fortran("subroutine fort_fold\n" + "".join(body) + "end subroutine\n")
        folded_execution = part_of(next(node for node in walk(wrapped) if kind_of(node) == "Subroutine_Subprogram"),
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
    return sum(1 for item in walk(node) if kind_of(item) == "Return_Stmt")


def _branches(construct):
    """The parts of an IF construct: (opening statement, body), and the END IF statement."""
    branches, end = [], None
    for item in construct.content:
        if kind_of(item) in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt"}:
            branches.append([item, []])
        elif kind_of(item) == "End_If_Stmt":
            end = item
        elif branches and kind_of(item) != "Comment":
            branches[-1][1].append(item)
    return branches, end


def _return_edit(statements, lines):
    """One edit that takes a RETURN out of a statement list that ends where the procedure does, or None.

    Returns (first, last, replacement lines), or a string with the reason when a
    RETURN stands where it cannot be taken out.
    """
    statements = [item for item in statements if kind_of(item) != "Comment"]
    for position, statement in enumerate(statements):
        if not _returns(statement):
            continue
        last = position == len(statements) - 1
        first_line, last_line = line_span(statement)
        if kind_of(statement) == "Return_Stmt":
            # Nothing after it runs: it and what follows go.
            return first_line, line_span(statements[-1])[1], []
        if kind_of(statement) == "If_Construct":
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
                if not _returns(header) and body and kind_of(body[-1]) == "Return_Stmt" and _returns_list(body) == 1:
                    returning.append((header, body))
                elif _returns_list(body):
                    return "a RETURN inside a branch is not its last statement"
                else:
                    staying.append((header, body))
            explicit_else = any(kind_of(header) == "Else_Stmt" for header, _ in branches)
            if len(staying) + (not explicit_else) != 1:
                return "the statements after an early RETURN would be needed in two branches"
            rest = lines[last_line:line_span(statements[-1])[1]]
            replacement = []
            cursor = first_line
            ending = line_span(end)[0]
            for header, body in branches:
                if (header, body) in returning:
                    stop = line_span(body[-1])
                    replacement += lines[cursor - 1:stop[0] - 1]
                    cursor = stop[1] + 1
                elif (header, body) in staying:
                    following = next((line_span(other)[0] for other, _ in branches if line_span(other)[0] > line_span(header)[0]),
                                     ending)
                    replacement += lines[cursor - 1:following - 1] + rest
                    cursor = following
            if not explicit_else:
                indent = lines[ending - 1][:len(lines[ending - 1]) - len(lines[ending - 1].lstrip())]
                replacement += lines[cursor - 1:ending - 1] + [indent + "else\n"] + rest
                cursor = ending
            replacement += lines[cursor - 1:last_line]
            return first_line, line_span(statements[-1])[1], replacement
        if kind_of(statement) == "If_Stmt" and kind_of(statement.items[1]) == "Return_Stmt":
            indent = lines[first_line - 1][:len(lines[first_line - 1]) - len(lines[first_line - 1].lstrip())]
            rest = lines[last_line:line_span(statements[-1])[1]]
            return first_line, line_span(statements[-1])[1], [
                f"{indent}if (.not. ({statement.items[0]})) then\n", *rest, f"{indent}end if\n"]
        return "a RETURN inside a loop or another construct cannot be taken out"
    return None


def _flagged(statements, lines, flag):
    """The lines of ``statements`` with each RETURN setting ``flag``, and what may follow one guarded by it.

    None where a RETURN stands in a loop or in another construct than IF.
    """
    output = []
    statements = [item for item in statements if kind_of(item) != "Comment"]
    for position, statement in enumerate(statements):
        first, last = line_span(statement)
        indent = lines[first - 1][:len(lines[first - 1]) - len(lines[first - 1].lstrip())]
        if not _returns(statement):
            output += lines[first - 1:last]
            continue
        if kind_of(statement) == "Return_Stmt":
            # Nothing after it in this list runs.
            return output + [f"{indent}{flag} = .true.\n"]
        if kind_of(statement) == "If_Stmt" and kind_of(statement.items[1]) == "Return_Stmt":
            output.append(f"{indent}if ({statement.items[0]}) {flag} = .true.\n")
        elif kind_of(statement) == "If_Construct":
            branches, end = _branches(statement)
            for header, body in branches:
                inner = _flagged(body, lines, flag)
                if inner is None:
                    return None
                output += lines[line_span(header)[0] - 1:line_span(header)[1]] + inner
            output += lines[line_span(end)[0] - 1:line_span(end)[1]]
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
        if kind_of(item) == "Exit_Stmt":
            found.append(item)
        elif kind_of(item) not in {"Block_Nonlabel_Do_Construct", "Block_Label_Do_Construct"} and not isinstance(item, str):
            found += _exits(item)
    return found


def _trip_bound(statements, loop, body):
    """How many iterations a loop without control can make at most; None if that is not known.

    The loop counts: its body adds one to an INTEGER variable at the top
    level, the last statement before the loop that sets that variable sets it
    to a constant, and a statement at the top level of the body, after the
    addition, leaves the loop once the variable reaches a constant.
    """
    before = [item for item in statements[:statements.index(loop)] if kind_of(item) != "Comment"]
    for increment in body:
        if kind_of(increment) != "Assignment_Stmt" or kind_of(increment.items[0]) != "Name":
            continue
        counter = str(increment.items[0]).lower()
        if str(increment.items[2]).replace(" ", "").lower() not in {f"{counter}+1", f"1+{counter}"}:
            continue
        if sum(1 for item in walk(body) if kind_of(item) == "Assignment_Stmt"
               and str(item.items[0]).lower() == counter) != 1:
            continue
        setting = [item for item in before if any(
            kind_of(node) == "Assignment_Stmt" and str(node.items[0]).lower() == counter for node in walk(item))]
        if not setting or kind_of(setting[-1]) != "Assignment_Stmt" \
                or any(kind_of(node) == "Call_Stmt" for item in before[before.index(setting[-1]):] for node in walk(item)):
            continue
        start = str(setting[-1].items[2]).replace(" ", "")
        if not re.fullmatch(r"-?\d+", start):
            continue
        for item in body[body.index(increment) + 1:]:
            condition = None
            if kind_of(item) == "If_Stmt" and kind_of(item.items[1]) == "Exit_Stmt":
                condition = item.items[0]
            elif kind_of(item) == "If_Construct":
                branches, _end = _branches(item)
                if len(branches) == 1 and branches[0][1] and kind_of(branches[0][1][-1]) == "Exit_Stmt":
                    condition = branches[0][0].items[0]
            cap = _cap(condition, counter) if condition is not None else None
            if cap is not None:
                trips = cap - int(start)
                return trips if trips >= 1 else None
    return None


def _cap(condition, counter):
    """The value of ``counter`` at which ``condition`` is surely true, or None: ``i >= 50``, ``i >= 50 .or. x > 1``."""
    while kind_of(condition) == "Parenthesis":
        condition = condition.items[1]
    if kind_of(condition) in {"Or_Operand", "Equiv_Operand"} and str(condition.items[1]).upper() == ".OR.":
        found = [_cap(side, counter) for side in (condition.items[0], condition.items[2])]
        found = [value for value in found if value is not None]
        return min(found) if found else None
    if kind_of(condition) == "Level_4_Expr" and str(condition.items[0]).lower() == counter \
            and str(condition.items[1]).upper() in _RELATION \
            and re.fullmatch(r"-?\d+", str(condition.items[2]).replace(" ", "")):
        return int(str(condition.items[2])) + _RELATION[str(condition.items[1]).upper()]
    return None


def _counted_loop(statements, lines, number):
    """One edit that writes a loop without control, that counts its iterations up to a cap, as a counted loop."""
    for statement in statements:
        if kind_of(statement) not in {"If_Construct", "Block_Nonlabel_Do_Construct"}:
            continue
        if kind_of(statement) == "If_Construct":
            for _header, body in _branches(statement)[0]:
                edit = _counted_loop(body, lines, number)
                if edit is not None:
                    return edit
            continue
        opening = next(item for item in statement.content if kind_of(item) == "Nonlabel_Do_Stmt")
        inside = [item for item in statement.content
                  if kind_of(item) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"}]
        if opening.items[1] is not None:
            edit = _counted_loop(inside, lines, number)
            if edit is not None:
                return edit
            continue
        body = inside
        trips = _trip_bound(statements, statement, body)
        if trips is None or any(kind_of(item) in {"Cycle_Stmt", "Goto_Stmt", "Return_Stmt"} for item in walk(statement)):
            return "a loop without control whose iterations are not counted"
        flag, trip = f"fort_go_{number}", f"fort_trip_{number}"
        indent = lines[line_span(opening)[0] - 1][:len(lines[line_span(opening)[0] - 1]) - len(lines[line_span(opening)[0] - 1].lstrip())]
        inner = []
        open_guards = 0
        for position, item in enumerate(body):
            first, last = line_span(item)
            text = lines[first - 1:last]
            leaves = _exits(item)
            if leaves:
                if kind_of(item) == "Exit_Stmt":
                    text = [f"{indent}    {flag} = .false.\n"]
                elif kind_of(item) == "If_Stmt" and kind_of(item.items[1]) == "Exit_Stmt":
                    text = [f"{indent}    if ({item.items[0]}) {flag} = .false.\n"]
                elif kind_of(item) == "If_Construct" and all(
                        not _exits_list(body_part) or (kind_of(body_part[-1]) == "Exit_Stmt"
                                                       and len(_exits_list(body_part)) == 1)
                        for _header, body_part in _branches(item)[0]):
                    text = list(text)
                    for leave in leaves:
                        line = line_span(leave)[0] - first
                        text[line] = re.sub(r"(?i)\bexit\b", f"{flag} = .false.", text[line], count=1)
                else:
                    return "an EXIT that is not the last statement of a branch"
            inner += text
            if leaves and position < len(body) - 1:
                inner.append(f"{indent}    if ({flag}) then\n")
                open_guards += 1
        inner += [f"{indent}    end if\n"] * open_guards
        # The first iteration always runs: written ahead of the loop, what it sets is set after the loop too.
        replacement = [f"{indent}{flag} = .true.\n", *inner]
        if trips > 1:
            replacement += [f"{indent}do {trip} = 2, {trips}\n", f"{indent}  if ({flag}) then\n", *inner,
                            f"{indent}  end if\n", f"{indent}end do\n"]
        return line_span(statement)[0], line_span(statement)[1], replacement, (flag, trip)
    return None


def _exits_list(statements):
    return [leave for item in statements for leave in ([item] if kind_of(item) == "Exit_Stmt" else _exits(item))]


def _reached(text, patterns):
    """The parsed text, its procedures by qualified name, and those the selected procedures reach by calls."""
    tree = parse_fortran(text)
    scopes = {scope.qualified: scope for scope in procedures_of(tree)}
    reached, pending = set(), [scope for name, scope in scopes.items() if any(fnmatchcase(name, p) for p in patterns)]
    while pending:
        scope = pending.pop()
        if scope.qualified in reached:
            continue
        reached.add(scope.qualified)
        execution = part_of(scope.node, "Execution_Part")
        for call in walk(execution) if execution is not None else ():
            # Calls, and references to functions of the source.
            if kind_of(call) in {"Call_Stmt", "Part_Ref", "Function_Reference"} and kind_of(call.items[0]) == "Name":
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
        execution = part_of(scope.node, "Execution_Part")
        for call in walk(execution) if execution is not None else ():
            if kind_of(call) != "Call_Stmt" or kind_of(call.items[0]) != "Name":
                continue
            callee = _resolve(scope, str(call.items[0]).lower(), scopes)
            if callee is None:
                continue
            dummies, optional = _optional_dummies(callee)
            given = _given(call, dummies)
            if not optional or given is None or optional <= given:
                continue
            absent = frozenset(optional - given)
            first, _last = line_span(call)
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
                edits.append((line_span(callee.node)[1] + 1, line_span(callee.node)[1], ["\n", *copy]))
            changed = re.sub(r"(?i)(\s*call\s+)" + re.escape(name) + r"\b",
                             lambda found, version=versions[key]: found.group(1) + version, lines[first - 1], count=1)
            edits.append((first, first, [changed]))
            records.append({**record, "version": versions[key]})
    for first, last, replacement in sorted(edits, key=lambda edit: (edit[0], edit[1]), reverse=True):
        lines[first - 1:last] = replacement
    return "".join(lines), records


def _element_pointer(scope, lines):
    """An edit that writes ``p%field`` as ``points(i)%field`` where ``p => points(i)`` opens the body of i's loop.

    The pointer is a local scalar pointer that is used for nothing but its
    fields, and only in loops whose body it opens by such a statement, at the
    top level, pointing at the element the loop variable selects. One loop is
    rewritten per call. Returns (first, last, lines) or None.
    """
    from .procedures import _CHARACTER_LITERAL, _strip_inline_comment, dummy_names

    execution = part_of(scope.node, "Execution_Part")
    specification = part_of(scope.node, "Specification_Part")
    dummies = set(dummy_names(scope.node))
    pointers = set()
    for statement in (specification.content if specification is not None else ()):
        if kind_of(statement) == "Type_Declaration_Stmt" and statement.items[1] is not None:
            attr_names = {str(item).upper() for item in statement.items[1].items}
            if "POINTER" in attr_names and "SAVE" not in attr_names:
                pointers |= {str(entity.items[0]).lower() for entity in statement.items[2].items
                             if entity.items[1] is None and str(entity.items[0]).lower() not in dummies}
    if execution is None or not pointers:
        return None
    opened = {}
    for loop in walk(execution):
        if kind_of(loop) != "Block_Nonlabel_Do_Construct":
            continue
        opening = next(item for item in loop.content if kind_of(item) == "Nonlabel_Do_Stmt")
        if opening.items[1] is None:
            continue
        variable = str(opening.items[1].items[1][0]).lower()
        for statement in loop.content:
            if kind_of(statement) == "Pointer_Assignment_Stmt" and str(statement.items[0]).lower() in pointers:
                target = statement.items[2]
                if kind_of(target) == "Part_Ref" and kind_of(target.items[0]) == "Name" \
                        and [str(item).lower() for item in getattr(target.items[1], "items", ())] == [variable]:
                    opened.setdefault(str(statement.items[0]).lower(), []).append((loop, statement, variable))
    for name, places in sorted(opened.items()):
        assignments = [item for item in walk(execution)
                       if kind_of(item) == "Pointer_Assignment_Stmt" and str(item.items[0]).lower() == name]
        uses = [item for item in walk(execution) if kind_of(item) == "Name" and str(item).lower() == name
                and not any(item is assignment.items[0] for assignment in assignments)]
        fields = [item for item in walk(execution) if kind_of(item) == "Data_Ref" and kind_of(item.items[0]) == "Name"
                  and str(item.items[0]).lower() == name]
        spans = [(line_span(statement)[1], line_span(loop)[1]) for loop, statement, _variable in places]
        if len(assignments) != len(places) or len(uses) != len(fields) \
                or not all(any(after < _line(item) <= stop for after, stop in spans) for item in fields):
            continue
        loop, statement, variable = places[0]
        first, last = line_span(statement)
        stop = line_span(loop)[1]
        if any(other is not statement and first <= line_span(other)[0] <= stop for other in assignments):
            continue
        element = f"{statement.items[2].items[0]}({variable})"
        pattern = re.compile(r"(?i)(?<![\w%])" + re.escape(name) + r"(\s*%)")

        def replace_outside_literals(line, pattern=pattern, element=element):
            code, comment = _strip_inline_comment(line.rstrip("\n"))
            parts, cursor = [], 0
            for literal in _CHARACTER_LITERAL.finditer(code):
                parts += [pattern.sub(lambda found: element + found.group(1), code[cursor:literal.start()]),
                          literal.group(0)]
                cursor = literal.end()
            return "".join(parts) + pattern.sub(lambda found: element + found.group(1), code[cursor:]) + comment + "\n"

        rewritten = [replace_outside_literals(line) for line in lines[last:stop]]
        return first, stop, rewritten
    return None


def _select_case(scope, lines):
    """An edit that writes one SELECT CASE construct of the procedure as an IF construct, or None.

    Each CASE becomes a condition on the selector, which is evaluated once
    per condition: that is done where the selector is a variable. A CASE
    DEFAULT becomes the ELSE, wherever it stands, since no two cases share a
    value. A selector compared with LOGICAL values is left as it is.
    """
    execution = part_of(scope.node, "Execution_Part")
    for construct in walk(execution) if execution is not None else ():
        if kind_of(construct) != "Case_Construct":
            continue
        opening = construct.content[0]
        if kind_of(opening) != "Select_Case_Stmt" or kind_of(opening.items[0]) != "Name" \
                or getattr(opening.item, "name", None):
            continue
        selector = str(opening.items[0])
        cases, default, current = [], None, None
        for item in construct.content[1:-1]:
            if kind_of(item) == "Case_Stmt":
                current = (item, [])
                if item.items[0].items[0] is None:
                    default = current
                else:
                    cases.append(current)
            elif current is not None and not (kind_of(item) == "Comment" and item.item.span[0] == line_span(current[0])[0]):
                current[1].append(item)
        if any("logical" in kind_of(value).lower() for case, _body in cases for value in walk(case)):
            continue

        def condition(case, selector=selector):
            terms = []
            for value in case.items[0].items[0].items:
                if kind_of(value) == "Case_Value_Range":
                    low, high = value.items
                    parts = ([f"{selector} >= ({low})"] if low is not None else []) + \
                            ([f"{selector} <= ({high})"] if high is not None else [])
                    terms.append("(" + " .and. ".join(parts) + ")")
                else:
                    terms.append(f"{selector} == ({value})")
            return " .or. ".join(terms)

        def emit_body(items):
            emitted = []
            seen = set()
            for item in items:
                span = line_span(item)
                if span not in seen:
                    seen.add(span)
                    emitted += lines[span[0] - 1:span[1]]
            return emitted

        first, last = line_span(construct)
        indent = lines[first - 1][:len(lines[first - 1]) - len(lines[first - 1].lstrip())]
        output = []
        for position, (case, body) in enumerate(cases):
            output.append(f"{indent}{'if' if position == 0 else 'else if'} ({condition(case)}) then\n")
            output += emit_body(body) if body else []
        if default is not None:
            body = emit_body(default[1])
            output += ([f"{indent}else\n"] if cases else []) + body
        if cases:
            output.append(f"{indent}end if\n")
        return first, last, output
    return None


def _line(node):
    """The source line a part of a statement stands on."""
    while getattr(node, "item", None) is None and getattr(node, "parent", None) is not None:
        node = node.parent
    return node.item.span[0]


def _declare(scope, lines, names):
    """An edit that declares ``names`` (Fortran declaration texts) after the procedure's specifications."""
    heading = line_span(heading_of(scope.node))[1]
    after = declarations_end(scope.node)
    margin = " " * (len(lines[heading - 1]) - len(lines[heading - 1].lstrip()) + 2)
    return after + 1, after, [margin + item + "\n" for item in names]


def _moved(line, moves):
    """Where a line of the text is after the edits ``(first, last, added lines)`` made since, in their order."""
    for first, last, added in moves:
        if line > last:
            line += added
        elif line >= first:
            raise ValueError("A rewrite changed lines of another procedure")
    return line


def _alone(lines, first, last):
    """The node of the procedure on lines ``first`` to ``last``, parsed on its own at the same line numbers."""
    tree = parse_fortran("\n" * (first - 1) + "".join(lines[first - 1:last]))
    node, = (item for item in children_of(tree) if kind_of(item) in {"Subroutine_Subprogram", "Function_Subprogram"})
    return node


def _rewrite(node, qualified, lines, text, selected):
    """The next rewrite of a procedure: its edits and record; no edits and maybe a record where there is none.

    A pointer to the element of a list and a SELECT CASE are rewritten in any
    procedure; the RETURN statements and capped loops only in those the
    selected ones call. A record without edits says why a rewrite cannot be
    made; the procedure gets no more of them then.
    """
    scope = _Node(node)
    found = _element_pointer(scope, lines)
    if found is not None:
        return [found], {"procedure": qualified, "pointer": found[0]}
    found = _select_case(scope, lines)
    if found is not None:
        return [found], {"procedure": qualified, "cases": found[0]}
    execution = part_of(node, "Execution_Part")
    if selected or execution is None:
        return [], None
    if _returns(execution):
        found = _return_edit(list(execution.content), lines)
        if found is not None and not isinstance(found, str):
            return [found], {"procedure": qualified, "returns": True}
        # Where the statements after a RETURN cannot simply become a branch, a flag says it was reached.
        taken = re.findall(r"(?i)\bfort_left_(\d+)\b", text)
        flag = f"fort_left_{1 + max((int(item) for item in taken), default=0)}"
        flagged = _flagged(list(execution.content), lines, flag)
        if flagged is None:
            return [], {"procedure": qualified, "reason": found or "a RETURN stays"}
        start, stop = line_span(execution)
        indent = lines[start - 1][:len(lines[start - 1]) - len(lines[start - 1].lstrip())]
        return ([(start, stop, [f"{indent}{flag} = .false.\n", *flagged]), _declare(scope, lines, [f"logical :: {flag}"])],
                {"procedure": qualified, "returns": True})
    number = 1 + max((int(found) for found in re.findall(r"(?i)\bfort_trip_(\d+)\b", text)), default=0)
    found = _counted_loop(list(execution.content), lines, number)
    if isinstance(found, str):
        return [], {"procedure": qualified, "reason": found}
    if found is None:
        return [], None
    first, last, replacement, (flag, trip) = found
    return ([(first, last, replacement), _declare(scope, lines, [f"logical :: {flag}", f"integer :: {trip}"])],
            {"procedure": qualified, "loop": first})


class _Node:
    """A procedure as the rewrites see it: its node."""

    def __init__(self, node):
        self.node = node


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
    if not holds_selected(text, patterns):
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
    # The rewrites of a procedure change its own lines only, and do not change what it calls: one procedure is
    # rewritten after the other, and parsed alone again after each rewrite.
    scopes, reached = _reached(text, patterns)
    moves = []
    for qualified in sorted(reached):
        selected = any(fnmatchcase(qualified, pattern) for pattern in patterns)
        first, last = (_moved(line, moves) for line in line_span(scopes[qualified].node))
        node = scopes[qualified].node if not moves else None
        for _round in range(_ROUNDS):
            lines = text.splitlines(keepends=True)
            if node is None:
                node = _alone(lines, first, last)
            edits, record = _rewrite(node, qualified, lines, text, selected)
            if record is not None:
                records.append(record)
            if not edits:
                break
            for edit_first, edit_last, replacement in sorted(edits, reverse=True):
                lines[edit_first - 1:edit_last] = replacement
                moves.append((edit_first, edit_last, len(replacement) - (edit_last - edit_first + 1)))
                last += len(replacement) - (edit_last - edit_first + 1)
            text, node = "".join(lines), None
        else:
            raise ValueError(f"The procedures that {qualified} calls take more than {_ROUNDS} rewrites")
    return text, records
