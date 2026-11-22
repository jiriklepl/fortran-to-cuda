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

from .anchors import _get_parser, _logical_lines, _split_top_level_args, parse_fortran

_PARAMETERS = re.compile(r"(?i)^\s*integer\b[^!:]*\bparameter\b[^!:]*::(.*)$")
_LITERAL = re.compile(r"[+-]?\d+\Z")
_COMPARISONS = {"==": True, ".EQ.": True, "/=": False, ".NE.": False}
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


def _key(node):
    return re.sub(r"\s+", "", str(node)).lower()


class _Conditions:
    """What a set of ``expression == constant`` conditions decides about other conditions."""

    def __init__(self, conditions, constants):
        _get_parser()  # the expression classes exist once a parser has been created
        self.constants = constants
        self.known = {}
        self.names = set()
        for text in conditions:
            node = Fortran2003.Expr(text)
            if type(node).__name__ != "Level_4_Expr" or not _COMPARISONS.get(str(node.items[1]).upper()):
                raise ValueError("An assumed condition has the form <expression> == <integer constant>: " + text)
            value = self._value(node.items[2])
            if value is None:
                raise ValueError("The constant of an assumed condition has no known integer value: " + text)
            if self.known.setdefault(_key(node.items[0]), value) != value:
                raise ValueError("Assumed conditions contradict each other: " + text)
            self.names |= {str(name).lower() for name in walk(node.items[0], Fortran2003.Name)}

    def _value(self, node):
        if type(node).__name__ == "Parenthesis":
            return self._value(node.items[1])
        key = _key(node)
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
            if left is None or right is None:
                return None
            return _RELATIONS[str(node.items[1]).upper()](left, right)
        return None


def _span(node):
    spans = [item.item.span for item in walk(node) if getattr(item, "item", None) is not None]
    return min(span[0] for span in spans), max(span[1] for span in spans)


def _fold(nodes, lines, conditions, counter, one_line=False):
    """The source lines of ``nodes`` with every decided IF construct folded, and with ``one_line`` every IF statement."""
    output = []
    for node in nodes:
        first, last = _span(node)
        if one_line and type(node).__name__ == "If_Stmt":
            value = conditions.decide(node.items[0])
            if value is not None:
                counter[0] += 1
                if value:
                    indent = lines[first - 1][:len(lines[first - 1]) - len(lines[first - 1].lstrip())]
                    output.append(indent + str(node.items[1]) + "\n")
                continue
        if type(node).__name__ != "If_Construct":
            output += lines[first - 1:last]
            continue
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
                output += lines[item.item.span[0] - 1:item.item.span[1]]
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
        if not kept:
            continue
        if kept[0][2]:
            output += _fold(kept[0][1], lines, conditions, counter, one_line)
            continue
        for position, (header, body, value) in enumerate(kept):
            start, stop = header.item.span
            text = lines[start - 1:stop]
            indent = text[0][:len(text[0]) - len(text[0].lstrip())]
            if value:
                text = [indent + "else\n"]
            elif position == 0 and type(header).__name__ == "Else_If_Stmt":
                text = [re.sub(r"(?i)^(\s*)else\s*if\b", r"\1if", text[0], count=1), *text[1:]]
            output += text + _fold(body, lines, conditions, counter, one_line)
        end = node.content[-1].item.span
        output += lines[end[0] - 1:end[1]]
    return output


def specialize(text, routine, conditions, constants):
    """Give ``routine`` a version of its body for the case that ``conditions`` hold.

    ``conditions`` are Fortran comparisons ``<expression> == <integer constant>``
    that the procedure can evaluate; ``constants`` maps named constants to
    their values (see ``integer_parameters``). An IF branch is folded when the
    conditions decide it. The procedure tests the conditions once at entry and
    runs the original body when they do not hold, so nothing it calls may
    change what they test. Raises ValueError when the conditions decide no
    branch of the procedure, when the procedure itself assigns to a variable
    they name or passes one to a call, or when its body cannot be written
    twice.
    """
    known = _Conditions(conditions, constants)
    tree = parse_fortran(text)
    matches = []
    for node in walk(tree):
        if type(node).__name__ in {"Subroutine_Subprogram", "Function_Subprogram"}:
            header = next((item for item in node.content
                           if type(item).__name__ in {"Subroutine_Stmt", "Function_Stmt"}), None)
            if header is not None and len(header.items) > 1 and str(header.items[1]).lower() == routine.lower():
                matches.append(node)
    if len(matches) != 1:
        raise ValueError(f"Expected one procedure {routine} to specialize, found {len(matches)}")
    execution = next((part for part in matches[0].content if type(part).__name__ == "Execution_Part"), None)
    if execution is None:
        raise ValueError(f"Procedure {routine} has no executable statements")
    for node in walk(execution):
        item = getattr(node, "item", None)
        if item is not None and (getattr(item, "label", None) is not None or getattr(item, "name", None)):
            raise ValueError(f"Procedure {routine} has a labelled or named statement; its body cannot be written twice")
        # The conditions are tested once, at entry: the body must not be able to change them.
        changed = ()
        if type(node).__name__ == "Assignment_Stmt":
            changed = [name for name in walk(node.items[0], Fortran2003.Name)][:1]
        elif type(node).__name__ == "Call_Stmt" and node.items[1] is not None:
            changed = walk(node.items[1], Fortran2003.Name)
        if any(str(name).lower() in known.names for name in changed):
            raise ValueError(f"Procedure {routine} may change what an assumed condition tests: "
                             + " ".join(str(node).split()))
    lines = text.splitlines(keepends=True)
    first, last = _span(execution)
    counter = [0]
    folded = _fold(execution.content, lines, known, counter)
    if not counter[0]:
        raise ValueError(f"The assumed conditions decide no branch of {routine}")
    original = lines[first - 1:last]
    indent = original[0][:len(original[0]) - len(original[0].lstrip())]
    test = (" .and. &\n" + indent + "    ").join("(" + condition.strip() + ")" for condition in conditions)
    return "".join([*lines[:first - 1], f"{indent}if ({test}) then\n", *folded, f"{indent}else\n", *original,
                    f"{indent}end if\n", *lines[last:]])
