"""Residency directives: comment lines that place the domain's epoch and its host access.

Compiled entries keep array values on the device. The directives say what the
compiler cannot decide from the loop nests alone: where the application starts
and stops keeping values there, and where code that stays on the host needs
them. They are comments, so the same source builds natively unchanged.

    !$fort enter budget(<bytes>) [if(<condition>)] [device(<ordinal>)]
    !$fort leave
    !$fort close
    !$fort publish [if(<condition>)]
    !$fort observe(<array>[, <array>...]) [if(<condition>)]
    !$fort release(<array>[, <array>...])

``enter`` starts keeping values on the device unless that is already the case;
``leave`` brings every value back and stops. ``close`` also frees the device
state and must run before the bound arrays are deallocated at program end.
``publish`` brings every value back to the host and forgets the device copies:
put it before host code whose accesses are not known. ``observe`` brings back
only the named arrays and keeps their device copies, for host code that reads
them. ``release`` retires a temporary's device copy without bringing it back,
before the temporary is deallocated. Every directive except ``enter`` and
``close`` does nothing while values are not on the device.
"""
import re

_LINE = re.compile(r"^(?P<indent>[ \t]*)!\$fort\b[ \t]*(?P<body>.*?)[ \t]*$", re.IGNORECASE)
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:%[A-Za-z][A-Za-z0-9_]*)*\Z")
_CLAUSES = {"enter": {"budget", "if", "device"}, "leave": set(), "close": set(), "publish": {"if"},
            "observe": {"if"}, "release": set()}
_ARGUMENTS = {"observe", "release"}


def _clauses(text, line):
    """Split ``name(args) clause(value) ...`` into the name, its arguments and its clauses."""
    words, position = [], 0
    while position < len(text):
        match = re.compile(r"\s*([A-Za-z_]\w*)\s*").match(text, position)
        if match is None:
            raise ValueError(f"Malformed residency directive on line {line}: {text!r}")
        word, position, value = match.group(1).lower(), match.end(), None
        if position < len(text) and text[position] == "(":
            depth, start = 0, position
            while position < len(text):
                depth += (text[position] == "(") - (text[position] == ")")
                position += 1
                if depth == 0:
                    break
            if depth:
                raise ValueError(f"Unbalanced parentheses in the residency directive on line {line}")
            value = text[start + 1:position - 1].strip()
        words.append((word, value))
    if not words:
        raise ValueError(f"Empty residency directive on line {line}")
    (name, arguments), clauses = words[0], dict(words[1:])
    if name not in _CLAUSES:
        raise ValueError(f"Unknown residency directive on line {line}: {name}")
    if len(clauses) != len(words) - 1 or set(clauses) - _CLAUSES[name] or any(value is None for value in clauses.values()):
        raise ValueError(f"Unsupported clause in the residency directive on line {line}: {text!r}")
    if (arguments is not None) != (name in _ARGUMENTS):
        raise ValueError(f"The {name} directive on line {line} "
                         + ("needs a list of arrays" if name in _ARGUMENTS else "takes no arguments"))
    names = []
    if arguments is not None:
        names = [item.strip() for item in arguments.split(",")]
        if not all(_NAME.match(item) for item in names):
            raise ValueError(f"The {name} directive on line {line} takes array names, not expressions")
    return name, names, clauses


def _block(indent, domain, body, *, binding=False):
    lines = ["block"]
    if binding:
        lines.append("  use, intrinsic :: iso_c_binding")
    lines += [f"  use {domain.module}, only: {domain.object}", *("  " + line for line in body), "end block"]
    return "".join(indent + line + "\n" for line in lines)


def _generate(name, names, clauses, indent, domain, origin):
    this = domain.object
    condition = clauses.get("if")
    active = f"{this}%active" + (f" .and. ({condition})" if condition else "")
    if name == "enter":
        if "budget" not in clauses:
            raise ValueError("The enter directive needs budget(<bytes>): the device memory the domain may use")
        start = (f"call {this}%begin_epoch(int({clauses.get('device', '0')}, c_int), "
                 f"int({clauses['budget']}, c_size_t))")
        guarded = [f"if ({condition}) {start}"] if condition else [start]
        return _block(indent, domain, [f"if (.not. {this}%active) then", *("  " + line for line in guarded), "endif"],
                      binding=True)
    if name == "leave":
        return _block(indent, domain, [f"if ({this}%active) call {this}%end_epoch()"])
    if name == "close":
        return _block(indent, domain, [f"call {this}%close()"])
    if name == "publish":
        return _block(indent, domain, [f"if ({active}) call {this}%publish()"])
    if name == "release":
        return _block(indent, domain, [f"if ({this}%active) then",
                                       *(f"  call {this}%release_real({item})" for item in names), "endif"])
    handles = f"{domain.prefix}_observed"
    binds = [f"  {handles}({index}) = {this}%bind({item}, lbound({item}), .true., origin='{origin}::{item.lower()}')"
             for index, item in enumerate(names, 1)]
    return _block(indent, domain, [f"integer(c_int64_t) :: {handles}({len(names)})", f"if ({active}) then", *binds,
                                   f"  call {this}%observe({handles})", "endif"], binding=True)


def expand_directives(text, domain, *, origin="source"):
    """Replace every residency directive in ``text`` by the code it stands for.

    ``domain`` is the ``ResidentDomain`` the application's entries use;
    ``origin`` labels the bindings an ``observe`` makes, for the runtime's
    diagnostics. Returns the new text and one record per directive.
    """
    output, records = [], []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        match = _LINE.match(line.rstrip("\r\n"))
        if match is None:
            output.append(line)
            continue
        name, names, clauses = _clauses(match.group("body"), number)
        try:
            output.append(_generate(name, names, clauses, match.group("indent"), domain, origin))
        except ValueError as error:
            raise ValueError(f"{error} (line {number})") from error
        records.append({"line": number, "directive": name, "arrays": names, "clauses": clauses})
    return "".join(output), records
