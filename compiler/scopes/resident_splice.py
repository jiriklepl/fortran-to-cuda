"""Put calls to compiled resident entries back into the source they came from.

The caller owns a residency domain: a Fortran object that extends the
runtime's ``resident_domain`` and says whether it is active. Each compiled
entry replaces its source span with a block that binds the arrays to that
domain and calls the entry while the domain is active, and runs the original
statements otherwise. A span the compiler rejected stays native; the domain
publishes its device values first unless the caller supplies its own text.
This module infers no effects and grants no activation: both stay with the
caller.
"""
import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

_BIND = {("real", 8): "bind_real64", ("real", 4): "bind_real32", ("integer", 4): "bind_integer32"}


@dataclass(frozen=True)
class ResidentDomain:
    """How generated code reaches the caller's residency domain.

    ``module`` exports ``object``, the domain variable, and ``check``, a
    subroutine that takes an entry's status. ``prefix`` starts the local names
    of the generated block.
    """

    module: str
    object: str
    check: str
    prefix: str = "fort"


def _identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


_FIELD = {("real", 8): "FORT_SCOPE_REAL64", ("real", 4): "FORT_SCOPE_REAL32", ("integer", 4): "FORT_SCOPE_INTEGER32"}


def _distinct_check(requirements, number, domain, indent):
    """Fortran that clears the ``distinct`` flag unless the subscript tuples of one array name each element once.

    ``requirements`` are all those of the array: its tuples are marked in one
    mask, so an element that two of them name, in any nests, is found too.
    """
    array, rank = requirements[0]["array"], len(requirements[0]["indices"])
    seen = f"{domain.prefix}_seen_{number}"
    lines = [f"allocate({seen}(" + ",".join(f"lbound({array},{axis}):ubound({array},{axis})"
                                           for axis in range(1, rank + 1)) + "))",
             f"{seen} = .false."]
    for requirement in requirements:
        indices = requirement["indices"]
        tuple_text = "[" + ", ".join(indices) + "]"
        lines += requirement["loops"]
        lines += [f"if (any({tuple_text} < lbound({array})) .or. any({tuple_text} > ubound({array}))) then",
                  f"  {domain.prefix}_distinct = .false.",
                  f"else if ({seen}({', '.join(indices)})) then",
                  f"  {domain.prefix}_distinct = .false.",
                  "else",
                  f"  {seen}({', '.join(indices)}) = .true.",
                  "endif"]
        lines += ["END DO"] * len(requirement["loops"])
    lines.append(f"deallocate({seen})")
    return [indent + line for line in lines]


def entry_call_block(region, entry, domain, held=None):
    """The block that replaces one compiled span; ``@ORIGINAL@`` marks the native statements.

    ``held`` maps ``(procedure, array name)`` to an expression for a handle the
    domain already holds; such an array is passed by that handle, not bound.
    A field of an array of derived type is bound through the domain's copy of
    it. Where the entry writes through index values, the block first checks,
    once per publication, that the subscript tuples of each such array name
    every element at most once, across all its nests; if they do not, the
    device values are published and the original statements run. The same
    happens where an entry that was compiled for some values of its scalars
    only (its ``conditions``) says that it was called with others, and where
    an assumed-shape dummy that is not declared CONTIGUOUS is associated with
    storage that is not: an array is bound by the address of its first
    element and its extents.
    """
    bindings = {binding["resource"]: binding for binding in region["resources"]}
    written = set(region.get("written_resources", ()))
    structures, checks = region.get("structures", ()), region.get("injective", ())
    # One mask per array written through index values, shared by all its subscript tuples.
    checked_arrays = {}
    for check in checks:
        checked_arrays.setdefault(check["resource"], []).append(check)
    actual, registration, handles, by_resource, kinds = {}, [], [], {}, set()
    # Dummies that may be associated with a strided section, by what their declaration says.
    strided = sorted({binding["name"] for binding in bindings.values()
                      if binding.get("intent") and "contiguous" not in binding.get("attributes", ())
                      and any(axis["kind"] == "Assumed_Shape_Spec" for axis in binding.get("shape", ()))
                      and "%" not in binding["name"]
                      and any(parameter["resource"] == binding["resource"] and parameter["rank"]
                              for parameter in region["parameters"])})
    spelled = region.get("spelled", {})
    for parameter in region["parameters"]:
        binding = bindings[parameter["resource"]]
        declared = binding["name"]
        element = "element_field" in binding.get("attributes", ())
        # The array as the nests name it: a USE may have renamed it, and its own name may be another's here.
        owner = spelled.get(parameter["resource"].split("%", 1)[0], declared.split("%", 1)[0])
        name = owner + declared[len(declared.split("%", 1)[0]):]
        if parameter["lower_bound_dimension"] is not None:
            actual[parameter["name"]] = f"lbound({owner},{parameter['lower_bound_dimension']})"
        elif parameter["rank"] and (region["procedure"], declared.lower()) in (held or {}):
            actual[parameter["name"]] = held[region["procedure"], declared.lower()]
        elif parameter["rank"]:
            handle = f"{domain.prefix}_handle_{parameter['name']}"
            actual[parameter["name"]] = handle
            by_resource[parameter["resource"]] = handle
            handles.append(handle)
            origin = region["procedure"] + "::" + declared
            if element:
                kind = _FIELD.get((binding["type"], binding["kind"]))
                if kind is None:
                    raise ValueError("Unsupported resident field ABI: " + str(binding))
                kinds.add(kind)
                member = name.split("%", 1)[1]
                registration.append(
                    f"        {handle}={domain.object}%bind_field(c_loc({owner}(lbound({owner},1))%{member}), &\n"
                    f"          int(storage_size({owner})/8,c_size_t),size({owner},kind=c_size_t),{kind}, &\n"
                    f"          {'.true.' if parameter['resource'] in written else '.false.'},origin='{origin}')")
                continue
            method = _BIND.get((binding["type"], binding["kind"]))
            if method is None:
                raise ValueError("Unsupported resident array ABI: " + str(binding))
            registration.append(f"        {handle}={domain.object}%{method}({name},lbound({name}),.true., &\n"
                                f"          origin='{origin}')")
        else:
            actual[parameter["name"]] = name
    run = f"{domain.prefix}_device_run"
    arguments = [f"{domain.object}%context", "FORT_SCOPE_GPU"] + [actual[name] for name in entry["argument_order"][2:]]
    guards = ["        if (.not. (" + guard + ")) error stop 'resident numerical geometry changed'"
              for guard in region["runtime_guards"]]
    conditional = bool(entry.get("conditions"))
    resident, status = f"{domain.prefix}_resident", f"{domain.prefix}_status"
    passed = "          " + ", &\n          ".join(arguments)
    # The statements run on the device; DO variables that are read afterwards get their final values here.
    after = ["        " + line for line in region.get("final_iterators", ())]
    if conditional:
        call = [f"        {status} = {run}( &", passed + ")",
                f"        {resident} = {status} /= FORT_SCOPE_DECLINED", f"        if ({resident}) then",
                f"          call {domain.check}({status})", *("  " + line for line in after), "        endif"]
    else:
        call = [f"        call {domain.check}({run}( &", passed + "))", *after]
    code = ["    block", "      use, intrinsic :: iso_c_binding",
            "      use fort_scoped_memory, only: " + ", ".join(
                ["FORT_SCOPE_GPU", *(["FORT_SCOPE_DECLINED"] if conditional else []), *sorted(kinds)]),
            f"      use {domain.module}, only: {domain.object}, {domain.check}",
            "      use " + entry["fortran_module"] + f", only: {run} => run",
            *(["      integer(c_int64_t) :: " + ",".join(handles)] if handles else [])]
    # Where no nest of the entry has an iteration, the original statements have nothing to do either.
    active = f"{domain.object}%active" + (f" .and. ({region['runs']})" if region.get("runs") else "")
    if not structures and not checks and not conditional and not strided:
        code += [f"      if ({active}) then", *guards, *registration, *call,
                 "      else", "@ORIGINAL@", "      endif", "    end block"]
        return "\n".join(code) + "\n"

    code.append(f"      logical :: {resident}")
    if conditional:
        code.append(f"      integer(c_int) :: {status}")
    if checks:
        code += [f"      logical, save :: {domain.prefix}_distinct = .false.",
                 f"      integer(c_int64_t), save :: {domain.prefix}_checked(2) = -1_c_int64_t"]
        code += [f"      logical, allocatable :: {domain.prefix}_seen_{number}({','.join(':' * len(group[0]['indices']))})"
                 for number, group in enumerate(checked_arrays.values(), 1)]
    # An empty array of derived type has no element whose fields could be bound.
    code.append(f"      {resident} = {active}"
                + "".join(f" .and. size({item['name']}) > 0" for item in structures)
                + "".join(f" .and. is_contiguous({name})" for name in strided))
    code += [f"      if ({resident}) then", *guards, *registration]
    if checks:
        indices = sorted({by_resource[root] for check in checks for root in check["index_resources"]
                          if root in by_resource})
        code += [f"        if (any({domain.prefix}_checked /= [{domain.object}%generation, {domain.object}%stamp])) then"]
        if indices:
            code.append(f"          call {domain.object}%observe([" + ", ".join(indices) + "])")
        code.append(f"          {domain.prefix}_distinct = .true.")
        for number, group in enumerate(checked_arrays.values(), 1):
            code += _distinct_check(group, number, domain, "          ")
        code += [f"          {domain.prefix}_checked = [{domain.object}%generation, {domain.object}%stamp]",
                 "        endif", f"        {resident} = {domain.prefix}_distinct"]
    code += ["      endif", f"      if ({resident}) then", *call,
             # An entry that declines leaves everything as it was, for the original statements.
             *(["      endif", f"      if (.not. {resident}) then"] if conditional else ["      else"]),
             f"        if ({active}) call {domain.object}%publish()", "@ORIGINAL@", "      endif"]
    code.append("    end block")
    return "\n".join(code) + "\n"


_OPENING = {"group", "If_Then_Stmt", "Else_If_Stmt", "Associate_Stmt"}


def published_native_block(domain):
    """A rejected span: publish the domain's device values, then run the original statements."""
    return (f"    block\n      use {domain.module}, only: {domain.object}\n"
            f"      if ({domain.object}%active) call {domain.object}%publish()\n@ORIGINAL@\n    end block\n")


def may_be_kept(resource, bound):
    """Whether an array that host code names may be one that entries keep on the device.

    ``bound`` holds the resources entries bind. A dummy argument is known by
    its name only, and the array it is associated with may be bound under any
    other: whether the domain holds it is asked when the statement runs. An
    array of derived type counts when entries bind one of its fields.
    """
    return resource in bound or resource.startswith("argument::") or _holds_fields(resource, bound)


def _holds_fields(resource, bound):
    """Whether entries bind a field of the array of derived type ``resource``."""
    return any(item.startswith(resource + "%") for item in bound)


def host_access_block(code, bound, domain):
    """The block around host statements that touch arrays entries keep on the device.

    ``code`` describes the statements as the export does: their ``arrays``
    with how each is used, their ``kind``, whether control ``leaves`` from
    inside, a ``runs`` condition of a loop, and for a one-line IF its
    ``condition`` and ``action``. ``bound`` holds the resources entries bind;
    other arrays need nothing. Arrays that are only read are brought up to
    date and stay on the device; arrays that may change are handed to the
    host until the statements are through; a deallocated array is released
    first. Returns None when no array is bound, and a block that publishes
    everything where the precise way is not open: a field of an array of
    derived type, or statements that control can leave before the arrays are
    handed back. The block asks the domain which of the arrays it holds when
    the statements run, so an array that is empty, or that no entry has used
    yet, costs nothing. ``@ORIGINAL@`` marks the statements.
    """
    arrays = [item for item in code.get("arrays", ()) if may_be_kept(item["resource"], bound)]
    if "arrays" not in code:
        return published_native_block(domain)
    if not arrays and not code.get("leave") and not code.get("publish"):
        return None
    if code.get("leave"):
        # Entries run somewhere inside these statements, beside host code that uses the same arrays: no
        # hand-over of arrays covers that. The values come back and stay until the application enters again.
        runs = f" .and. ({code['runs']})" if code.get("runs") else ""
        return (f"    block\n      use {domain.module}, only: {domain.object}\n"
                f"      if ({domain.object}%active{runs}) call {domain.object}%end_epoch()\n    end block\n")
    modes = {mode: [item for item in arrays if item["access"] == mode] for mode in ("read", "write", "release")}
    # The opening statements of a construct take their access before the construct, and cannot hand
    # anything back after it.
    opening = code.get("kind") in _OPENING
    # The fields of an array of derived type come back to its elements only with everything else.
    precise = not code.get("publish") \
        and all(not item["field"] and not _holds_fields(item["resource"], bound) for item in arrays) \
        and not (modes["write"] and (code.get("leaves") or opening)) \
        and all(item["type"] == "real" for item in modes["release"]) \
        and all(len(item.get("exists", ())) <= 1 for item in arrays)
    condition = code.get("condition")
    inner = condition is not None and not {item["name"].lower() for item in arrays} & set(code["condition_names"])
    margin = "      " if inner else "    "
    if not precise:
        body = [f"{margin}block", f"{margin}  use {domain.module}, only: {domain.object}",
                f"{margin}  if ({domain.object}%active) call {domain.object}%publish()", "@ORIGINAL@", f"{margin}end block"]
        if opening:
            body = [*body[:3], body[4]]
    else:
        def handles(items):
            # Asked of the domain, not registered: an array no entry has used yet has nothing on the device.
            return ", &\n".join(f"{margin}      {domain.object}%held({item['name']})" for item in items)

        def calls(action, items, when):
            """One call for the arrays that always exist, and one each for those that may not."""
            groups = [[item for item in items if not item.get("exists")],
                      *([item] for item in items if item.get("exists"))]
            return [f"{margin}  if ({when[id(group[0])] if group[0].get('exists') else when[None]}) "
                    f"call {domain.object}%{action}([ &\n{handles(group)}])" for group in groups if group]

        active = f"{domain.object}%active" + (f" .and. ({code['runs']})" if code.get("runs") else "")
        # An array that is not allocated, associated or present has nothing on the device and cannot be bound.
        # Whether it exists is asked once: the statements may allocate it.
        maybe = [item for item in arrays if item.get("exists") and item["access"] != "release"]
        when = {None: f"{domain.prefix}_host",
                **{id(item): f"{domain.prefix}_host{number}" for number, item in enumerate(maybe, 1)}}
        body = [f"{margin}block", f"{margin}  use {domain.module}, only: {domain.object}",
                # Module arrays that host procedures called from here use, under names of their own.
                *(f"{margin}  use {item['use'][0]}, only: {item['name']} => {item['use'][1]}"
                  for item in arrays if item.get("use")),
                f"{margin}  logical :: {', '.join(when.values())}",
                f"{margin}  {domain.prefix}_host = {active}",
                *(f"{margin}  {when[id(item)]} = {item['exists'][0]}({item['name']})" for item in maybe),
                *(f"{margin}  {when[id(item)]} = {when[id(item)]} .and. {domain.prefix}_host" for item in maybe)]
        # RELEASE takes the arrays themselves; the other two take their handles.
        body += [f"{margin}  if ({domain.prefix}_host"
                 + "".join(f" .and. {inquiry}({item['name']})" for inquiry in item.get("exists", ()))
                 + f") call {domain.object}%release_real({item['name']})" for item in modes["release"]]
        body += calls("observe", modes["read"], when) + calls("host_modify", modes["write"], when)
        body += [] if opening else ["@ORIGINAL@"]
        body += calls("host_done", modes["write"], when)
        body += [f"{margin}end block"]
    if inner:
        # The access belongs to the action: a condition that is rarely true must not bring arrays back.
        body = [f"    if ({condition}) then", *[line.replace("@ORIGINAL@", "@ACTION@") for line in body], "    end if"]
    return "\n".join(body) + "\n"


def _on_shared_lines(statements):
    """Host statements that stand on the same source lines as one: their block goes around those lines once."""
    order = ("read", "write", "release")
    merged = []
    for statement in sorted(statements, key=lambda item: (item["source"], item["first_line"], item["last_line"])):
        last = merged[-1] if merged else None
        if last is None or last["source"] != statement["source"] or statement["first_line"] > last["last_line"]:
            merged.append(statement)
            continue
        both = {**last, "last_line": max(last["last_line"], statement["last_line"]),
                # A statement that opens a construct still cannot have anything handed back after it.
                "kind": next((item["kind"] for item in (last, statement) if item.get("kind") in _OPENING), "statements"),
                "leaves": bool(last.get("leaves") or statement.get("leaves")),
                "leave": bool(last.get("leave") or statement.get("leave")),
                "publish": bool(last.get("publish") or statement.get("publish"))}
        for key in ("condition", "action", "condition_names", "runs", "arrays"):
            both.pop(key, None)
        if "arrays" in last and "arrays" in statement:
            arrays = {}
            for item in (*last["arrays"], *statement["arrays"]):
                known = arrays.get(item["resource"])
                if known is None or order.index(item["access"]) > order.index(known["access"]):
                    arrays[item["resource"]] = item
            both["arrays"] = [arrays[resource] for resource in sorted(arrays)]
        merged[-1] = both
    return merged


def splice_resident_entries(packages, directory, *, domain, native=None, held=None, host=()):
    """Write each source with its compiled spans replaced by entry calls.

    ``packages`` are records from ``compile_resident_entries``. A rejected span
    gets the host access its arrays need (see ``host_access_block``), also in
    a procedure where no span compiled: its arrays may be ones that entries
    of other procedures keep on the device. ``native(region,
    original)`` may return the complete replacement for a rejected span, or
    None for the default. ``held`` names arrays the domain already holds a
    handle for (see ``entry_call_block``). ``host`` lists host statements, as
    the export records them, that get the same access around them; those that
    share source lines get it together. Sources
    are checked against the hashes the export recorded. Writes and returns
    ``source-edits.json``.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    edits, expected = {}, {}
    bound = {resource["resource"] for package in packages for item in package["entries"]
             for resource in item["region"]["resources"] if resource.get("rank", 1)}
    for statement in _on_shared_lines(host):
        block = host_access_block(statement, bound, domain)
        if block is not None:
            # A block without the statements goes in front of them.
            last = statement["last_line"] if "@ORIGINAL@" in block or "@ACTION@" in block else statement["first_line"] - 1
            edits.setdefault(statement["source"], []).append(
                (statement["first_line"], last, block, None, statement.get("action")))
    for package in packages:
        expected.update(package["source_inputs"])
        for item in package["entries"]:
            region = item["region"]
            edits.setdefault(region["source"], []).append(
                (region["first_line"], region["last_line"], entry_call_block(region, item["entry"], domain, held), None,
                 None))
        for region in package["rejected"]:
            block = host_access_block(region, bound, domain)
            if block is not None and "@ORIGINAL@" not in block and "@ACTION@" not in block:
                # A block without the statements goes in front of them.
                block += "@ORIGINAL@\n"
            if block is not None or native is not None:
                edits.setdefault(region["source"], []).append(
                    (region["first_line"], region["last_line"], block or "@ORIGINAL@\n", region, region.get("action")))
    records = []
    for source, replacements in edits.items():
        path = Path(source)
        if _identity(path)["sha256"] != expected[source]:
            raise ValueError("Resident numerical source changed: " + source)
        lines = path.read_text().splitlines(keepends=True)
        replacements.sort(key=lambda item: item[:3])
        if any(a[1] >= b[0] for a, b in zip(replacements, replacements[1:], strict=False)):
            raise ValueError("Overlapping resident source replacements")
        for first, last, code, rejected, action in reversed(replacements):
            original = "".join(lines[first - 1:last])
            custom = native(rejected, original) if rejected is not None and native is not None else None
            if action is not None:
                code = code.replace("@ACTION@", "      " + action)
            lines[first - 1:last] = [custom if custom is not None else code.replace("@ORIGINAL@", original.rstrip())]
        target = directory / path.name
        target.write_text("".join(lines))
        records.append({"original": _identity(path), "replacement": _identity(target),
                        "region_count": len(replacements)})
    result = {"version": 1, "kind": "resident_source_edits", "sources": records,
              "activation_authority": "the caller's residency domain; native call effects are not inferred here"}
    (directory / "source-edits.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


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
    Returns the new text and the names it changed, per procedure.
    """
    from fparser.two.utils import walk

    from compiler.resident.anchors import parse_fortran
    from compiler.resident.reductions import _procedures

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

    for scope in _procedures(parse_fortran(text)):
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
    return "".join(lines), done


def addressable_arrays(text):
    """Add TARGET to the arrays a residency domain can bind, so their C addresses are stable.

    Covers REAL, INTEGER and LOGICAL arrays and arrays of derived type declared
    at module or main-program level, and dummy or allocatable arrays inside
    procedures. Components of derived types, parameters and pointers are left
    alone, and so are the dummies of an external procedure, which its callers
    know without an interface.
    """
    from fparser.common.readfortran import FortranStringReader
    from fparser.two.parser import ParserFactory
    from fparser.two.utils import walk

    tree = ParserFactory().create(std="f2008")(FortranStringReader(text, ignore_comments=False))
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
