"""Preprocess application sources into the text the compiler analyses and edits."""
import hashlib
import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

from compiler.frontend.syntax import parse_fortran
from compiler.resident.anchors import _strip_inline_comment
from compiler.resident.procedures import heading_of, holds_selected, procedures_of
from compiler.scopes.segments import license_line, parse_license, private_marker


def _identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def distinct_name(path):
    """A file name for a copy of ``path`` that copies of other files of the same name do not take."""
    return hashlib.sha256(str(path).encode()).hexdigest()[:12] + "_" + path.name


def configure_sources(paths, procedures, directory, *, fortran, defines=(), include=()):
    """Preprocess each source and serialize the selected procedures.

    Writes one preprocessed copy of every file into ``directory``; the originals
    are untouched. Included files are searched for in each source's own
    directory, then in the ``include`` directories. OpenMP directives are
    blanked only inside the procedures
    named in ``procedures`` (``module::name``, ``module::host::name`` for an
    internal procedure, ``$external::name`` for an external procedure,
    ``$program::name`` for a main program, or globs): their loop nests become entries that one
    serial caller dispatches. A directive with a ``reduction`` clause leaves a
    ``!$fort reduction`` line, which keeps what the clause says about the
    accumulation order. Every statement, branch and expression stays. Returns
    the record of originals, prepared copies and removed directives.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    selected = {name.lower() for name in procedures}
    records = []
    for filename in paths:
        source = Path(filename).resolve(strict=True)
        before = _identity(source)
        command = [str(fortran), "-cpp", "-E", "-P", "-O3", "-ffree-line-length-none", "-fopenmp",
                   *("-D" + value for value in defines), "-I" + str(source.parent),
                   *("-I" + str(Path(item).resolve()) for item in include), str(source)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError("Preprocessing failed: " + str(source) + "\n" + result.stderr)
        lines = result.stdout.splitlines(keepends=True)
        removed = []
        if holds_selected(result.stdout, [name for name in selected if "::" in name]):
            tree = parse_fortran(result.stdout)
            for scope in procedures_of(tree):
                if not any(fnmatchcase(scope.qualified, pattern) for pattern in selected):
                    continue
                first, last = heading_of(scope.node).item.span[0], scope.node.content[-1].item.span[1]
                removed += [{"line": index + 1, "directive": directive, "procedure": scope.qualified}
                            for index, directive in serialize_openmp(lines, first - 1, last)]
        target = directory / distinct_name(source)
        target.write_text("".join(lines))
        if _identity(source) != before:
            raise ValueError("Source changed while it was preprocessed: " + str(source))
        records.append({"original": before, "prepared": _identity(target), "command": command,
                        "serialized_directives": removed})
    return {"version": 1, "kind": "configured_resident_sources", "defines": list(defines),
            "procedures": sorted(selected), "sources": records}


def serialize_openmp(lines, first, last):
    """Blank the OpenMP directives in ``lines[first:last]``; return ``(index, text)`` of each removed line.

    The loop nests of a selected procedure become entries that one serial
    caller dispatches. A directive with a ``reduction`` clause leaves a
    ``!$fort reduction`` line: what the clause says about the order of an
    accumulation outlives the directive. A loop directive with a ``private``
    clause and no reduction leaves ``!$fort private(...)`` at the end of the
    DO statement: the scalars it names have no value when an iteration starts.
    """
    removed, index = [], first
    while index < last:
        if not re.match(r"\s*!\$omp\b", lines[index], re.IGNORECASE):
            index += 1
            continue
        # One directive, with its continuation lines.
        group = [index]
        while lines[group[-1]].rstrip().endswith("&") and group[-1] + 1 < last \
                and re.match(r"\s*!\$omp\b", lines[group[-1] + 1], re.IGNORECASE):
            group.append(group[-1] + 1)
        text = " ".join(re.sub(r"(?i)^\s*!\$omp&?", "", lines[item]).strip().rstrip("&") for item in group)
        indent = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
        for item in group:
            removed.append((item, lines[item].rstrip()))
            lines[item] = "\n"
        stated = license_line(text)
        if stated is not None:
            lines[index] = indent + stated + "\n"
        elif re.search(r"(?i)\bdo\b", text) and parse_license(text)[1]:
            # The scalars a loop's directive makes private have no value when an iteration starts. That is
            # said at the end of the DO statement, where it stays with the loop through every rewrite.
            loop = next((item for item in range(group[-1] + 1, min(last, group[-1] + 8))
                         if re.match(r"(?i)\s*do\b", lines[item])), None)
            if loop is not None and not lines[loop].rstrip().endswith("&"):
                code, comment = _strip_inline_comment(lines[loop].rstrip("\n"))
                lines[loop] = (code.rstrip() + "  " + private_marker(parse_license(text)[1])
                               + (" " + comment if comment else "") + "\n")
        index = group[-1] + 1
    return removed
