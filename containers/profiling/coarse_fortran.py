"""Select coarse Fortran callbacks before optimization, using original source/DWARF."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def declarations(source):
    text = re.sub(r"&\s*\n\s*&?", " ", Path(source).read_text())
    pattern = (r"(?im)^\s*(?:(?:pure|elemental|recursive|impure|module|non_recursive)\s+)*"
               r"(?:(?:integer|real|logical|complex|character|type)\s*(?:\([^)]*\)|\*\d+)?\s+)?"
               r"(?:subroutine|function)\s+([a-z][a-z0-9_]*)")
    return {name.lower() for name in re.findall(pattern, text)}


def dwarf_procedures(text):
    """Read named subprogram DIEs, including abstract/inlined source procedures."""
    header = re.compile(r"(?m)^\s*<(\d+)><[0-9a-f]+>:.*?\((DW_TAG_\w+)\).*$")
    matches, modules, result = list(header.finditer(text)), {}, []

    def attribute(block, name):
        found = re.search(r"(?m)^.*?\b" + name + r"\s*:\s*(.+)$", block)
        if not found:
            return None
        value = found[1].split("): ")[-1]
        value = re.sub(r"^\([^)]*\)\s*", "", value)
        return value.strip()

    for index, item in enumerate(matches):
        depth, tag = int(item[1]), item[2]
        modules = {level: name for level, name in modules.items() if level < depth}
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[item.end():end]
        name = attribute(block, "DW_AT_name")
        if tag == "DW_TAG_module" and name:
            modules[depth] = name
        if tag != "DW_TAG_subprogram" or not name or not re.fullmatch(r"[A-Za-z_]\w*", name):
            continue
        module = modules[max(modules)] if modules else None
        result.append({"name": name.lower(), "linkage": attribute(block, "DW_AT_linkage_name"),
                       "module": module, "inlined": "DW_AT_inline" in block})
    return result


def selection(source, original_object, category):
    command = ["/usr/bin/readelf", "--debug-dump=info", "--wide", str(original_object)]
    observed = subprocess.run(command, capture_output=True, check=True, timeout=120)
    procedures = dwarf_procedures(observed.stdout.decode())
    if not procedures:
        symbols = subprocess.run(["/usr/bin/nm", "-an", str(original_object)], capture_output=True,
                                 check=True, timeout=120)
        if re.search(rb"(?m)^\w+\s+[tTwW]\s+[^.\s]", symbols.stdout):
            raise ValueError("original debug source procedure authority required")
        return {"retained": [], "excluded": [], "substring_collisions_retained": [], "procedures": [],
                "readelf_argv": command, "readelf_stderr": observed.stderr.decode(),
                "compile_instrumentation": False, "original_object_has_no_code": True}
    retained = set()
    for row in procedures:
        symbol = row["linkage"] or ("__" + row["module"] + "_MOD_" + row["name"] if row["module"] else row["name"])
        if category(symbol) or row["name"] in ("main", "main__") or row["name"] == "tmarchrk3":
            retained.add(row["name"])
    text = Path(source).read_text()
    programs = set(re.findall(r"(?im)^\s*program\s+([a-z][a-z0-9_]*)", text))
    if programs:
        retained.update(name.lower() for name in programs)
        retained.update(("main", "main__"))
    names = declarations(source) | {row["name"] for row in procedures}
    # GCC matches bare source-name SUBSTRINGS, not ELF mangled names. Never let an
    # exclusion remove a selected phase through an overlapping source spelling.
    excluded = {name for name in names - retained if not any(name in keep for keep in retained)}
    collisions = names - retained - excluded
    return {"retained": sorted(retained), "excluded": sorted(excluded), "substring_collisions_retained": sorted(collisions),
            "procedures": procedures, "readelf_argv": command, "readelf_stderr": observed.stderr.decode(),
            "compile_instrumentation": bool(retained)}
