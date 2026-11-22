"""What the resident build changes in the declarations of an application's sources.

``lend_storage`` lets the domain lend the storage of temporary arrays;
``addressable_arrays`` gives the arrays a domain can bind the TARGET attribute.
"""
import re
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.syntax import parse_fortran
from compiler.resident.procedures import procedures_of


def lend_storage(text, domain, temporaries):
    """Let the domain lend the storage of a procedure's temporary arrays.

    ``temporaries`` maps a procedure (``module::name``, as the export names
    it) to the names of arrays it allocates and deallocates once per call (see
    ``resident_entries._temporaries``). Allocated anew in each call, such an
    array costs a device allocation and, when it is deallocated, a wait for
    everything queued that still uses it. Lent by the domain it keeps its host
    and device storage from one call to the next while values are kept on the
    device, and is an ordinary allocation otherwise. The array becomes a
    contiguous pointer, its ALLOCATE and DEALLOCATE become calls to the domain.
    Returns the new text.
    """
    wanted = {procedure.lower(): {name.lower() for name in names} for procedure, names in temporaries.items()}
    lines = text.splitlines(keepends=True)
    changes, done = [], {}

    def span(node):
        return node.item.span

    def margin(node):
        line = lines[span(node)[0] - 1]
        return line[:len(line) - len(line.lstrip())]

    def block(node, calls):
        indent = margin(node)
        body = [f"{indent}block", f"{indent}  use, intrinsic :: iso_c_binding, only: c_int64_t",
                f"{indent}  use {domain.module}, only: {domain.object}",
                *(f"{indent}  {call}" for call in calls), f"{indent}end block"]
        return "".join(line + "\n" for line in body)

    for scope in procedures_of(parse_fortran(text)):
        names = wanted.get(scope.qualified)
        if not names:
            continue
        found = {"declared": set(), "allocated": set(), "deallocated": set()}
        local = []
        for node in walk(scope.node):
            kind = type(node).__name__
            if kind == "Type_Declaration_Stmt":
                dtype, attributes, entities = node.items
                mine = [entity for entity in entities.items if str(entity.items[0]).lower() in names]
                if not mine:
                    continue
                attrs = [item.strip() for item in re.split(r",(?![^()]*\))", str(attributes))] if attributes else []
                kept = ", ".join([str(dtype), *attrs])
                lent = ", ".join([str(dtype), *(item for item in attrs if item.lower() not in {"allocatable", "target"}),
                                  "POINTER", "CONTIGUOUS"])
                others = [str(entity) for entity in entities.items if entity not in mine]
                indent = margin(node)
                replacement = [f"{indent}{kept} :: {', '.join(others)}\n"] if others else []
                replacement.append(f"{indent}{lent} :: {', '.join(str(entity) for entity in mine)}\n")
                local.append((*span(node), "".join(replacement)))
                found["declared"].update(str(entity.items[0]).lower() for entity in mine)
            elif kind == "Allocate_Stmt":
                objects = [item for item in walk(node) if type(item).__name__ == "Allocation"]
                if not any(str(item.items[0]).lower() in names for item in objects):
                    continue
                calls = []
                for item in objects:
                    name = str(item.items[0])
                    bounds = [(str(axis.items[0]) if axis.items[0] is not None else "1", str(axis.items[1]))
                              for axis in item.items[1].items]
                    # One place in the source, one piece of storage per shape it is asked for.
                    site = int.from_bytes(sha256(f"{scope.qualified}::{name.lower()}".encode()).digest()[:8],
                                          "big") & ((1 << 63) - 1)
                    calls.append(f"call {domain.object}%work({site}_c_int64_t, &\n"
                                 f"{margin(node)}    [integer :: {', '.join(lower for lower, _upper in bounds)}], &\n"
                                 f"{margin(node)}    [integer :: {', '.join(upper for _lower, upper in bounds)}], {name})")
                    found["allocated"].add(name.lower())
                local.append((*span(node), block(node, calls)))
            elif kind == "Deallocate_Stmt":
                listed = node.items[0]
                parts = [listed] if type(listed).__name__ == "Name" else list(listed.items)
                if not any(str(item).lower() in names for item in parts):
                    continue
                local.append((*span(node), block(node, [f"call {domain.object}%work_done({item})" for item in parts])))
                found["deallocated"].update(str(item).lower() for item in parts)
        if not (found["declared"] == found["allocated"] == found["deallocated"] == names):
            raise ValueError("The temporary arrays of " + scope.qualified + " are not where the export found them")
        changes += local
        done[scope.qualified] = sorted(names)
    if set(done) != {procedure for procedure, names in wanted.items() if names}:
        raise ValueError("A procedure with temporary arrays is not in its source")
    for first, last, replacement in sorted(changes, reverse=True):
        lines[first - 1:last] = [replacement]
    return "".join(lines)


def addressable_arrays(text):
    """Add TARGET to the arrays a residency domain can bind, so their C addresses are stable.

    Covers REAL, INTEGER and LOGICAL arrays and arrays of derived type declared
    at module or main-program level, and dummy or allocatable arrays inside
    procedures. Components of derived types, parameters and pointers are left
    alone, and so are the dummies of an external procedure, which its callers
    know without an interface.
    """
    tree = parse_fortran(text)
    lines = text.splitlines(keepends=True)
    changes = []
    for node in walk(tree):
        if type(node).__name__ != "Type_Declaration_Stmt":
            continue
        dtype, attributes, entities = node.items
        attrs = str(attributes).lower()
        if (not re.match(r"(?i)^(real|integer|logical|type\s*\()", str(dtype))
                or re.search(r"\b(target|parameter|pointer)\b", attrs)):
            continue
        is_array = "dimension" in attrs or any(entity.items[1] is not None for entity in entities.items)
        parent = node.parent
        while parent is not None and type(parent).__name__ not in {
                "Module", "Subroutine_Subprogram", "Function_Subprogram", "Derived_Type_Def", "Main_Program"}:
            parent = getattr(parent, "parent", None)
        if not is_array or parent is None or type(parent).__name__ == "Derived_Type_Def":
            continue
        if type(parent).__name__ not in {"Module", "Main_Program"} and "intent" not in attrs and "allocatable" not in attrs:
            continue
        if (type(parent).__name__ in {"Subroutine_Subprogram", "Function_Subprogram"} and "allocatable" not in attrs
                and type(getattr(parent, "parent", None)).__name__ == "Program"):
            # A dummy of an external procedure: with TARGET, every call would need an explicit interface.
            continue
        first, last = node.item.span
        before, after = str(node).split("::", 1)
        changes.append((first, last, before.rstrip() + ", TARGET ::" + after + "\n"))
    for first, last, replacement in sorted(changes, reverse=True):
        lines[first - 1:last] = [replacement]
    return "".join(lines)
