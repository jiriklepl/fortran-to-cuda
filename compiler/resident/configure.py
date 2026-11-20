"""Preprocess application sources into the text the compiler analyses and edits."""
import hashlib
import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

from compiler.resident.reductions import _procedures, serialize_openmp


def _identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def configure_sources(paths, procedures, directory, *, fortran, defines=()):
    """Preprocess each source and serialize the selected procedures.

    Writes one preprocessed copy of every file into ``directory``; the originals
    are untouched. OpenMP directives are blanked only inside the procedures
    named in ``procedures`` (``module::name``, ``module::host::name`` for an
    internal procedure, or globs): their loop nests become entries that one
    serial caller dispatches. A directive with a ``reduction`` clause leaves a
    ``!$fort reduction`` line, which keeps what the clause says about the
    accumulation order. Every statement, branch and expression stays. Returns
    the record of originals, prepared copies and removed directives.
    """
    from fparser.common.readfortran import FortranStringReader
    from fparser.two.parser import ParserFactory

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    parser = ParserFactory().create(std="f2008")
    selected = {name.lower() for name in procedures}
    records = []
    for filename in paths:
        source = Path(filename).resolve(strict=True)
        before = _identity(source)
        command = [str(fortran), "-cpp", "-E", "-P", "-O3", "-ffree-line-length-none", "-fopenmp",
                   *("-D" + value for value in defines), "-I" + str(source.parent), str(source)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError("Preprocessing failed: " + str(source) + "\n" + result.stderr)
        lines = result.stdout.splitlines(keepends=True)
        removed = []
        modules = {m.group(1).lower() for m in re.finditer(r"(?im)^\s*module\s+([a-z_]\w*)\b", result.stdout)}
        if any(fnmatchcase(module, name.split("::", 1)[0]) for module in modules for name in selected if "::" in name):
            tree = parser(FortranStringReader(result.stdout, ignore_comments=False))
            for scope in _procedures(tree):
                if not any(fnmatchcase(scope.qualified, pattern) for pattern in selected):
                    continue
                statement = next(node for node in scope.node.content
                                 if type(node).__name__ in {"Subroutine_Stmt", "Function_Stmt"})
                first, last = statement.item.span[0], scope.node.content[-1].item.span[1]
                removed += [{"line": index + 1, "directive": directive, "procedure": scope.qualified}
                            for index, directive in serialize_openmp(lines, first - 1, last)]
        target = directory / (hashlib.sha256(str(source).encode()).hexdigest()[:12] + "_" + source.name)
        target.write_text("".join(lines))
        if _identity(source) != before:
            raise ValueError("Source changed while it was preprocessed: " + str(source))
        records.append({"original": before, "prepared": _identity(target), "command": command,
                        "serialized_directives": removed})
    return {"version": 1, "kind": "configured_resident_sources", "defines": list(defines),
            "procedures": sorted(selected), "sources": records}
