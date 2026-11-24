"""The procedures of a source, and what the passes read of their trees and write into their text.

Each function takes a parsed tree (see ``compiler.frontend.syntax``) or the
lines of the text it was parsed from; none changes a tree.
"""
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase

from fparser.two.utils import walk

from compiler.frontend.syntax import kind_of, part_of
from compiler.resident.anchors import _strip_inline_comment


def normalized(node):
    """A node's text without spaces, in lower case: equal for equal Fortran."""
    return re.sub(r"\s+", "", str(node)).lower()


def line_span(node):
    """First and last source line of a statement or construct."""
    item = getattr(node, "item", None)
    if item is not None:
        return item.span
    spans = [span for span in (line_span(child) for child in node.content if kind_of(child) != "Comment")
             if span is not None]
    # A part that holds nothing but comments has no statement to place it by.
    return (min(first for first, _ in spans), max(last for _, last in spans)) if spans else None


def names_in(node):
    """Lower-case names in ``node``, in source order, without the fields selected from them."""
    if node is None or isinstance(node, str):
        return []
    if kind_of(node) == "Name":
        return [str(node).lower()]
    if kind_of(node) == "Data_Ref":
        # object(i)%field: the object and its subscripts are names of the scope; the field is not.
        return names_in(node.items[0]) + [name for part in node.items[1:] if kind_of(part) != "Name"
                                        for name in names_in(getattr(part, "items", (None, None))[1])]
    children = getattr(node, "items", None)
    if children is None:
        children = getattr(node, "content", ())
    return [name for child in children for name in names_in(child)]


@dataclass
class Procedure:
    """A procedure with the scopes its names can come from, innermost first."""

    qualified: str
    node: object
    hosts: tuple = ()


def heading_of(node):
    """The statement that opens a procedure or a main program."""
    return part_of(node, "Subroutine_Stmt") or part_of(node, "Function_Stmt") or part_of(node, "Program_Stmt")


def dummy_names(node):
    """Lower-case dummy argument names of a subroutine or function node."""
    heading = heading_of(node)
    if heading is None or kind_of(heading) not in {"Subroutine_Stmt", "Function_Stmt"} or heading.items[2] is None:
        return ()
    return tuple(str(item).lower() for item in getattr(heading.items[2], "items", ()))


def escaping_names(node):
    """Lower-case names through which a procedure's values leave it: its dummies and, for a function, its result."""
    names = set(dummy_names(node))
    heading = heading_of(node)
    if heading is not None and kind_of(heading) == "Function_Stmt":
        result = str(heading.items[1]).lower()
        for item in getattr(heading.items[3], "items", None) or ():
            if item is not None and kind_of(item) == "Name":
                result = str(item).lower()
        names.add(result)
    return names


def unit_names(text):
    """The first parts of the names procedures of ``text`` can be selected by.

    A module's procedures are ``module::name``, an external procedure is
    ``$external::name`` and a named main program ``$program::name``; internal
    procedures add ``::name`` to their host's.
    """
    return {"$external", "$program",
            *(match.group(1).lower() for match in re.finditer(r"(?im)^\s*module\s+([a-z_]\w*)\b", text))}


def holds_selected(text, patterns):
    """Whether ``text`` may hold procedures that the ``module::name`` patterns select (see ``unit_names``)."""
    return any(fnmatchcase(unit, pattern.split("::", 1)[0]) for unit in unit_names(text) for pattern in patterns)


def procedures_of(tree):
    def subprograms(nodes, prefix, hosts):
        for node in nodes:
            statement = heading_of(node) if kind_of(node) in {"Subroutine_Subprogram", "Function_Subprogram"} else None
            if statement is None:
                continue
            scope = Procedure(prefix + "::" + str(statement.items[1]).lower(), node, hosts)
            yield scope
            yield from internal(scope)

    def internal(scope):
        container = part_of(scope.node, "Internal_Subprogram_Part")
        yield from subprograms(container.content if container is not None else (), scope.qualified,
                               (scope.node, *scope.hosts))

    for module in (node for node in walk(tree) if kind_of(node) == "Module"):
        name = str(part_of(module, "Module_Stmt").items[1]).lower()
        container = part_of(module, "Module_Subprogram_Part")
        yield from subprograms(container.content if container is not None else (), name, (module,))
    # Program units of their own: external procedures and the main program.
    yield from subprograms(tree.content, "$external", ())
    for program in (node for node in tree.content if kind_of(node) == "Main_Program"):
        if part_of(program, "Program_Stmt") is not None:
            scope = Procedure("$program::" + str(part_of(program, "Program_Stmt").items[1]).lower(), program, ())
            yield scope
            yield from internal(scope)


# A character literal, to the end of the line where it is continued on the next.
_CHARACTER_LITERAL = re.compile(r"'(?:[^']|'')*(?:'|$)|\"(?:[^\"]|\"\")*(?:\"|$)")
# What code is read as when names are replaced: a dotted operator or logical constant (.and., .eq., .true.), a
# number (1, 1.5, 1.e3, .5d0, 2_8; the dot of 1.eq.n is the operator's), or a name.
_TOKENS = re.compile(r"\.[a-z]+\.|(?:\d+(?:\.(?![a-z]+\.)\d*)?|\.\d+)(?:[edq][+-]?\d+)?(?:_\w+)?|[a-z_]\w*",
                     re.IGNORECASE)


def renamed_lines(lines, mapping):
    """Replace whole names outside comments and character literals; a name after ``%`` is a field and stays."""
    if not mapping:
        return list(lines)

    def rename(match, code):
        token = match.group(0)
        replacement = mapping.get(token.lower()) if token[0].isalpha() or token[0] == "_" else None
        if replacement is None or code[:match.start()].rstrip().endswith("%"):
            return token
        return replacement

    def names(code):
        return _TOKENS.sub(lambda match: rename(match, code), code)

    renamed = []
    for line in lines:
        code, comment = _strip_inline_comment(line.rstrip("\n"))
        parts, cursor = [], 0
        for literal in _CHARACTER_LITERAL.finditer(code):
            parts += [names(code[cursor:literal.start()]), literal.group(0)]
            cursor = literal.end()
        renamed.append("".join(parts) + names(code[cursor:]) + comment + "\n")
    return renamed


def _extract_dimension(attrs):
    """Return ``(bounds_text_or_None, attrs_without_dimension)`` using balanced parentheses."""
    match = re.search(r"dimension\s*\(", attrs)
    if match is None:
        return None, attrs
    start = match.end()
    depth = 1
    index = start
    while index < len(attrs) and depth > 0:
        if attrs[index] == "(":
            depth += 1
        elif attrs[index] == ")":
            depth -= 1
        index += 1
    if depth != 0:
        return None, attrs
    bounds = attrs[start:index - 1]
    stripped = attrs[:match.start()] + attrs[index:]
    return bounds, stripped


def integer_variables(texts):
    """Integer scalar variables of the modules in ``texts``: lower-case name -> type text without spaces.

    A name that modules declare with different types is left out: without
    scoping, which one a use means is not known.
    """
    from compiler.frontend.syntax import parse_fortran

    types, conflicting = {}, set()
    for text in texts:
        if not re.search(r"(?im)^\s*module\s+(?!procedure\b)\w", text):
            continue
        for module in (node for node in parse_fortran(text).content if kind_of(node) == "Module"):
            for name, (dtype, attributes, bounds, _initialized) in declarations_of(module).items():
                dtype = normalized(dtype)
                if bounds is None and "parameter" not in attributes and types.setdefault(name, dtype) != dtype:
                    conflicting.add(name)
    return {name: dtype for name, dtype in types.items() if dtype.startswith("integer") and name not in conflicting}


def declarations_of(node):
    """name -> (type text, attribute text, array bounds text or None) for the entities a procedure declares."""
    found = {}
    specification = part_of(node, "Specification_Part")
    for statement in (specification.content if specification is not None else ()):
        if kind_of(statement) != "Type_Declaration_Stmt":
            continue
        dtype, attributes, entities = statement.items
        attrs = str(attributes).lower() if attributes is not None else ""
        shared, stripped = _extract_dimension(attrs)
        for entity in entities.items:
            bounds = entity.items[1] if entity.items[1] is not None else shared
            found[str(entity.items[0]).lower()] = (
                str(dtype), stripped, None if bounds is None else normalized(bounds),
                entity.items[3] is not None)
    return found


def declarations_end(node):
    """The last line of a procedure's heading and specifications: what is declared anew goes after it."""
    specification = part_of(node, "Specification_Part")
    ends = [line_span(item)[1] for item in (specification.content if specification is not None else ())
            if kind_of(item) != "Comment" and line_span(item) is not None]
    return max([line_span(heading_of(node))[1], *ends])
