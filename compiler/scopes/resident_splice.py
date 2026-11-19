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


def entry_call_block(region, entry, domain):
    """The block that replaces one compiled span; ``@ORIGINAL@`` marks the native statements."""
    bindings = {binding["resource"]: binding for binding in region["resources"]}
    actual, registration, handles = {}, [], []
    for parameter in region["parameters"]:
        binding = bindings[parameter["resource"]]
        name = binding["name"]
        if parameter["lower_bound_dimension"] is not None:
            actual[parameter["name"]] = f"lbound({name},{parameter['lower_bound_dimension']})"
        elif parameter["rank"]:
            handle = f"{domain.prefix}_handle_{parameter['name']}"
            actual[parameter["name"]] = handle
            method = _BIND.get((binding["type"], binding["kind"]))
            if method is None:
                raise ValueError("Unsupported resident array ABI: " + str(binding))
            handles.append(handle)
            origin = region["procedure"] + "::" + name
            registration.append(f"        {handle}={domain.object}%{method}({name},lbound({name}),.true., &\n"
                                f"          origin='{origin}')")
        else:
            actual[parameter["name"]] = name
    run = f"{domain.prefix}_device_run"
    arguments = [f"{domain.object}%context", "FORT_SCOPE_GPU"] + [actual[name] for name in entry["argument_order"][2:]]
    code = ["    block", "      use, intrinsic :: iso_c_binding",
            "      use fort_scoped_memory, only: FORT_SCOPE_GPU",
            f"      use {domain.module}, only: {domain.object}, {domain.check}",
            "      use " + entry["fortran_module"] + f", only: {run} => run",
            "      integer(c_int64_t) :: " + ",".join(handles),
            f"      if ({domain.object}%active) then"]
    for guard in region["runtime_guards"]:
        code.append("        if (.not. (" + guard + ")) error stop 'resident numerical geometry changed'")
    code += [*registration, f"        call {domain.check}({run}( &", "          " + ", &\n          ".join(arguments) + "))",
             "      else", "@ORIGINAL@", "      endif", "    end block"]
    return "\n".join(code) + "\n"


def published_native_block(domain):
    """A rejected span: publish the domain's device values, then run the original statements."""
    return (f"    block\n      use {domain.module}, only: {domain.object}\n"
            f"      if ({domain.object}%active) call {domain.object}%publish()\n@ORIGINAL@\n    end block\n")


def splice_resident_entries(packages, directory, *, domain, native=None):
    """Write each source with its compiled spans replaced by entry calls.

    ``packages`` are records from ``compile_resident_entries``. A rejected span
    is edited only when another span of the same procedure compiled; a
    procedure with no entry at all is left to its caller. ``native(region,
    original)`` may return the complete replacement for a rejected span, or
    None for the default. Sources are checked against the hashes the export
    recorded. Writes and returns ``source-edits.json``.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    edits, expected = {}, {}
    for package in packages:
        expected.update(package["source_inputs"])
        for item in package["entries"]:
            region = item["region"]
            edits.setdefault(region["source"], []).append(
                (region["first_line"], region["last_line"], entry_call_block(region, item["entry"], domain), None))
        for region in package["rejected"]:
            sources = {entry["region"]["source"] for entry in package["entries"]
                       if entry["region"]["procedure"] == region["procedure"]}
            if not sources:
                continue
            if len(sources) != 1:
                raise ValueError("Ambiguous resident native source")
            edits.setdefault(next(iter(sources)), []).append(
                (region["first_line"], region["last_line"], published_native_block(domain), region))
    records = []
    for source, replacements in edits.items():
        path = Path(source)
        if _identity(path)["sha256"] != expected[source]:
            raise ValueError("Resident numerical source changed: " + source)
        lines = path.read_text().splitlines(keepends=True)
        replacements.sort(key=lambda item: item[:3])
        if any(a[1] >= b[0] for a, b in zip(replacements, replacements[1:], strict=False)):
            raise ValueError("Overlapping resident source replacements")
        for first, last, code, rejected in reversed(replacements):
            original = "".join(lines[first - 1:last])
            custom = native(rejected, original) if rejected is not None and native is not None else None
            lines[first - 1:last] = [custom if custom is not None else code.replace("@ORIGINAL@", original.rstrip())]
        target = directory / path.name
        target.write_text("".join(lines))
        records.append({"original": _identity(path), "replacement": _identity(target),
                        "region_count": len(replacements)})
    result = {"version": 1, "kind": "resident_source_edits", "sources": records,
              "activation_authority": "the caller's residency domain; native call effects are not inferred here"}
    (directory / "source-edits.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def addressable_arrays(text):
    """Add TARGET to the arrays a residency domain can bind, so their C addresses are stable.

    Covers REAL, INTEGER and LOGICAL arrays declared at module or main-program
    level, and dummy or allocatable arrays inside procedures. Components of
    derived types, parameters and pointers are left alone.
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
        if not re.match(r"(?i)^(real|integer|logical)\b", str(dtype)) or re.search(r"\b(target|parameter|pointer)\b", attrs):
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
        first, last = node.item.span
        before, after = str(node).split("::", 1)
        changes.append((first, last, before.rstrip() + ", TARGET ::" + after + "\n"))
    for first, last, replacement in sorted(changes, reverse=True):
        lines[first - 1:last] = [replacement]
    return "".join(lines)
