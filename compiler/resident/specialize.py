"""A second version of a procedure's body for the case that declared conditions hold.

Branches on run-time configuration keep loop nests apart that one launch could
run together: each branch is its own entry, and the host decides between them
at every call. When the application can say which values its configuration has
while its fields stay on the device, the branches those values decide are
folded away in a copy of the body, and the nests that remain end up next to
each other. The original body stays for every other case::

    if (<the conditions>) then
      <body with the decided branches folded>
    else
      <original body>
    end if
"""
import operator
import re

from fparser.two import Fortran2003
from fparser.two.utils import walk

from compiler.frontend.syntax import parse_fortran, parser
from compiler.resident.procedures import _CHARACTER_LITERAL, normalized, procedures_of
from compiler.scopes.segments import statement_span

from .anchors import _logical_lines, _split_top_level_args, _strip_inline_comment

_PARAMETERS = re.compile(r"(?i)^\s*integer\b[^!:]*\bparameter\b[^!:]*::(.*)$")
_LITERAL = re.compile(r"[+-]?\d+\Z")
# Every comparison of two known integers is decided, not only the equalities a condition may state.
_RELATIONS = {"==": operator.eq, ".EQ.": operator.eq, "/=": operator.ne, ".NE.": operator.ne,
              "<": operator.lt, ".LT.": operator.lt, "<=": operator.le, ".LE.": operator.le,
              ">": operator.gt, ".GT.": operator.gt, ">=": operator.ge, ".GE.": operator.ge}


def integer_parameters(texts):
    """Integer named constants with one literal value throughout ``texts``, by lower-case name.

    A name that is given different values in different places is left out:
    without scoping, its value at a use is not known.
    """
    values, conflicting = {}, set()
    for text in texts:
        for line in _logical_lines(text):
            match = _PARAMETERS.match(line)
            if match is None:
                continue
            for entity in _split_top_level_args(match.group(1)):
                name, equals, value = (part.strip() for part in entity.partition("="))
                value = value.replace(" ", "")
                if not equals or not re.fullmatch(r"[A-Za-z]\w*", name) or not _LITERAL.match(value):
                    continue
                name = name.lower()
                if values.setdefault(name, int(value)) != int(value):
                    conflicting.add(name)
    return {name: value for name, value in values.items() if name not in conflicting}


_SYMBOLS = {".EQ.": "==", ".NE.": "/=", ".LT.": "<", ".LE.": "<=", ".GT.": ">", ".GE.": ">="}
_MIRRORED = {"==": "==", "/=": "/=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}


class _Conditions:
    """What a set of conditions ``expression <relation> integer constant`` decides about other conditions.

    An equality gives the expression its value. Another relation bounds it:
    ``viscosity > 0`` decides ``viscosity <= 0``, for instance. A LOGICAL
    variable, or ``.not.`` one, gives the variable its value.
    """

    def __init__(self, conditions, constants):
        parser()  # the expression classes exist once a parser has been created
        self.constants = constants
        self.known = {}
        self.bounds = {}
        self.names = set()
        self.truths = {}
        for text in conditions:
            node = Fortran2003.Expr(text)
            # A LOGICAL variable, or its negation, is a value of its own.
            negated = type(node).__name__ == "And_Operand" and str(node.items[0]).upper() == ".NOT."
            variable = node.items[1] if negated else node
            if type(variable).__name__ in {"Name", "Data_Ref"}:
                self.truths[normalized(variable)] = not negated
                self.names |= {str(name).lower() for name in walk(variable, Fortran2003.Name)}
                continue
            relation = _SYMBOLS.get(str(node.items[1]).upper(), str(node.items[1])) \
                if type(node).__name__ == "Level_4_Expr" else None
            if relation not in _MIRRORED or relation == "/=":
                raise ValueError("An assumed condition has the form <expression> <relation> <integer constant>: " + text)
            value = self._value(node.items[2])
            if value is None:
                raise ValueError("The constant of an assumed condition has no known integer value: " + text)
            key = normalized(node.items[0])
            if relation == "==":
                if self.known.setdefault(key, value) != value:
                    raise ValueError("Assumed conditions contradict each other: " + text)
            else:
                self.bounds.setdefault(key, []).append((relation, value))
            self.names |= {str(name).lower() for name in walk(node.items[0], Fortran2003.Name)}

    def _implied(self, key, relation, value):
        """Whether the bounds known for ``key`` make ``key <relation> value`` true or false; None if neither."""
        lower, upper = (None, False), (None, False)
        for bound, constant in self.bounds.get(key, ()):
            if bound in {">", ">="} and (lower[0] is None or constant > lower[0] or (constant == lower[0] and bound == ">")):
                lower = (constant, bound == ">")
            if bound in {"<", "<="} and (upper[0] is None or constant < upper[0] or (constant == upper[0] and bound == "<")):
                upper = (constant, bound == "<")
        above = lower[0] is not None and (lower[0] > value or (lower[0] == value and lower[1]))   # every x > value
        at_least = lower[0] is not None and lower[0] >= value                                  # every x >= value
        below = upper[0] is not None and (upper[0] < value or (upper[0] == value and upper[1]))  # every x < value
        at_most = upper[0] is not None and upper[0] <= value                                   # every x <= value
        truth = {">": (above, at_most), ">=": (at_least, below), "<": (below, at_least), "<=": (at_most, above),
                 "==": (False, above or below), "/=": (above or below, False)}[relation]
        return True if truth[0] else False if truth[1] else None

    def _value(self, node):
        if type(node).__name__ == "Parenthesis":
            return self._value(node.items[1])
        key = normalized(node)
        if key in self.known:
            return self.known[key]
        if _LITERAL.match(key):
            return int(key)
        return self.constants.get(key) if type(node).__name__ == "Name" else None

    def decide(self, node):
        """True, False, or None when the conditions do not decide ``node``."""
        kind = type(node).__name__
        if kind == "Parenthesis":
            return self.decide(node.items[1])
        if kind in {"Name", "Data_Ref"} and normalized(node) in self.truths:
            return self.truths[normalized(node)]
        if kind == "And_Operand" and str(node.items[0]).upper() == ".NOT.":
            value = self.decide(node.items[1])
            return None if value is None else not value
        if kind in {"Or_Operand", "Equiv_Operand"} and str(node.items[1]).upper() in {".AND.", ".OR."}:
            both = str(node.items[1]).upper() == ".AND."
            sides = [self.decide(node.items[0]), self.decide(node.items[2])]
            if (not both) in sides:
                return not both
            return both if sides == [both, both] else None
        if kind == "Level_4_Expr" and str(node.items[1]).upper() in _RELATIONS:
            left, right = self._value(node.items[0]), self._value(node.items[2])
            if left is not None and right is not None:
                return _RELATIONS[str(node.items[1]).upper()](left, right)
            relation = _SYMBOLS.get(str(node.items[1]).upper(), str(node.items[1]))
            if right is not None and normalized(node.items[0]) in self.bounds:
                return self._implied(normalized(node.items[0]), relation, right)
            if left is not None and normalized(node.items[2]) in self.bounds:
                return self._implied(normalized(node.items[2]), _MIRRORED[relation], left)
        return None


def _decides(node, conditions, one_line):
    """Whether the conditions decide an IF somewhere in ``node``."""
    kinds = {"If_Then_Stmt", "Else_If_Stmt", *(("If_Stmt",) if one_line else ())}
    return any(type(item).__name__ in kinds and conditions.decide(item.items[0]) is not None for item in walk(node))


def _leading(comments, lines):
    """The lines of comments, and blank lines, the parser put in front of a construct that are lines of their own.

    A comment at the end of the line before belongs to that line, which is
    written with its statement.
    """
    found = []
    for item in comments:
        if getattr(item, "item", None) is None:
            continue
        text = lines[item.item.span[0] - 1:item.item.span[1]]
        if all(not line.strip() or line.lstrip().startswith("!") for line in text):
            found += text
    return found


def _shares_a_line(first, last, lines):
    """Whether one of lines ``first`` to ``last`` holds another statement, after a semicolon."""
    for line in lines[first - 1:last]:
        code, _comment = _strip_inline_comment(line.rstrip("\n"))
        if ";" in _CHARACTER_LITERAL.sub("", code):
            return True
    return False


def _fold(nodes, lines, conditions, counter, one_line=False, covered=None):
    """The source lines of ``nodes`` with every decided IF construct folded, and with ``one_line`` every IF statement.

    Lines are written whole and once: a statement on a line already written,
    the second of two on a line or the comment after a statement, is written
    with it. ``covered`` holds the last line written or folded away. A decided
    IF must have its first and last lines to itself: ValueError otherwise.
    """
    covered = [0] if covered is None else covered
    output = []

    def emit(first, last):
        if last > covered[0]:
            output.extend(lines[max(first, covered[0] + 1) - 1:last])
            covered[0] = last

    def alone(first, last):
        if _shares_a_line(first, first, lines) or _shares_a_line(last, last, lines):
            raise ValueError("A decided IF shares a line with another statement: " + lines[first - 1].strip())

    for node in nodes:
        first, last = statement_span(node)
        if one_line and type(node).__name__ == "If_Stmt":
            value = conditions.decide(node.items[0])
            if value is not None:
                alone(first, last)
                counter[0] += 1
                if value:
                    indent = lines[first - 1][:len(lines[first - 1]) - len(lines[first - 1].lstrip())]
                    output.append(indent + str(node.items[1]) + "\n")
                covered[0] = max(covered[0], last)
                continue
        if type(node).__name__ == "Block_Nonlabel_Do_Construct" and _decides(node, conditions, one_line):
            # The branches in a loop's body are decided like any others; what lies between its
            # statements stays as it is.
            position = next(index for index, item in enumerate(node.content)
                            if type(item).__name__ == "Nonlabel_Do_Stmt")
            opening, ending = node.content[position], node.content[-1]
            output += _leading(node.content[:position], lines)
            emit(*opening.item.span)
            for item in node.content[position + 1:-1]:
                if type(item).__name__ == "Comment":
                    continue
                emit(covered[0] + 1, statement_span(item)[0] - 1)
                output += _fold([item], lines, conditions, counter, one_line, covered)
            emit(covered[0] + 1, ending.item.span[1])
            continue
        if type(node).__name__ != "If_Construct":
            emit(first, last)
            continue
        alone(first, last)
        branches = []
        for item in node.content:
            if type(item).__name__ in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt"}:
                branches.append([item, []])
            elif type(item).__name__ == "End_If_Stmt":
                break
            elif branches:
                branches[-1][1].append(item)
            else:
                # A comment the parser attached in front of the construct's first statement.
                output += _leading([item], lines)
        kept = []
        for header, body in branches:
            value = True if type(header).__name__ == "Else_Stmt" else conditions.decide(header.items[0])
            if value is False:
                counter[0] += 1
                continue
            kept.append((header, body, value))
            if value:
                counter[0] += type(header).__name__ != "Else_Stmt"
                break
        if kept and kept[0][2]:
            covered[0] = max(covered[0], kept[0][0].item.span[1])
            output += _fold(kept[0][1], lines, conditions, counter, one_line, covered)
        elif kept:
            for position, (header, body, value) in enumerate(kept):
                start, stop = header.item.span
                text = lines[start - 1:stop]
                indent = text[0][:len(text[0]) - len(text[0].lstrip())]
                if value:
                    text = [indent + "else\n"]
                elif position == 0 and type(header).__name__ == "Else_If_Stmt":
                    text = [re.sub(r"(?i)^(\s*)else\s*if\b", r"\1if", text[0], count=1), *text[1:]]
                output += text
                covered[0] = max(covered[0], stop)
                output += _fold(body, lines, conditions, counter, one_line, covered)
            end = node.content[-1].item.span
            output += lines[end[0] - 1:end[1]]
        # The construct's lines are written or folded away, the comment after its END IF with them.
        covered[0] = max(covered[0], last)
    return output

def specialize(text, procedures, conditions, constants):
    """Give each of ``procedures`` a version of its body for the case that ``conditions`` hold.

    ``procedures`` are ``module::name`` names of procedures of ``text`` (see
    ``reductions._procedures``). ``conditions`` are Fortran comparisons
    ``<expression> <relation> <integer constant>``, or LOGICAL variables and
    their negations, that the procedures can evaluate; ``constants`` maps named
    constants to their values (see ``integer_parameters``). An IF branch is
    folded when the conditions decide it. A procedure tests the conditions once
    at entry and runs the original body when they do not hold, so nothing it
    calls may change what they test. Raises ValueError when the conditions
    decide no branch of a procedure, when a procedure itself assigns to a
    variable they name or passes one to a call, or when its body cannot be
    written twice.
    """
    from .procedures import declarations_of

    scopes = {scope.qualified: scope for scope in procedures_of(parse_fortran(text))}
    lines = text.splitlines(keepends=True)
    edits = []
    for name in procedures:
        scope = scopes.get(name.lower())
        if scope is None:
            raise ValueError(f"Expected the procedure {name} to specialize")
        # A variable of the procedure or of a host, the innermost declaration first, is not a constant.
        declared = {}
        for host in (*reversed(scope.hosts), scope.node):
            declared.update(declarations_of(host))
        shadowed = {var for var, (_dtype, attrs, _bounds, _init) in declared.items() if "parameter" not in attrs}
        scoped_constants = {k: v for k, v in constants.items() if k not in shadowed}
        known = _Conditions(conditions, scoped_constants)
        edits.append(_specialized(scope.node, name, lines, conditions, known))
    for first, last, replacement in sorted(edits, reverse=True):
        lines[first - 1:last] = replacement
    return "".join(lines)


def _specialized(procedure, name, lines, conditions, known):
    """The edit ``(first, last, lines)`` that gives one procedure its version for the case that ``known`` holds."""
    execution = next((part for part in procedure.content if type(part).__name__ == "Execution_Part"), None)
    if execution is None:
        raise ValueError(f"Procedure {name} has no executable statements")
    for node in walk(execution):
        item = getattr(node, "item", None)
        if item is not None and (getattr(item, "label", None) is not None or getattr(item, "name", None)):
            raise ValueError(f"Procedure {name} has a labelled or named statement; its body cannot be written twice")
        # The conditions are tested once, at entry: the body must not be able to change them.
        changed = ()
        if type(node).__name__ == "Assignment_Stmt":
            changed = [name for name in walk(node.items[0], Fortran2003.Name)][:1]
        elif type(node).__name__ == "Call_Stmt" and node.items[1] is not None:
            changed = walk(node.items[1], Fortran2003.Name)
        if any(str(item).lower() in known.names for item in changed):
            raise ValueError(f"Procedure {name} may change what an assumed condition tests: "
                             + " ".join(str(node).split()))
    first, last = statement_span(execution)
    counter = [0]
    folded = _fold(execution.content, lines, known, counter)
    if not counter[0]:
        raise ValueError(f"The assumed conditions decide no branch of {name}")
    original = lines[first - 1:last]
    indent = original[0][:len(original[0]) - len(original[0].lstrip())]
    test = (" .and. &\n" + indent + "    ").join("(" + condition.strip() + ")" for condition in conditions)
    return first, last, [f"{indent}if ({test}) then\n", *folded, f"{indent}else\n", *original, f"{indent}end if\n"]
