"""Preprocess application sources into the text the compiler analyses and edits."""
import hashlib
import re
import subprocess
from pathlib import Path


def _identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def configure_sources(paths, procedures, directory, *, fortran, defines=()):
    """Preprocess each source and serialize the selected procedures.

    Writes one preprocessed copy of every file into ``directory``; the originals
    are untouched. OpenMP directives are blanked only inside the procedures
    named in ``procedures`` (``module::name``): their loop nests become entries
    that one serial caller dispatches. Every statement, branch and expression
    stays. Returns the record of originals, prepared copies and removed
    directives.
    """
    from fparser.common.readfortran import FortranStringReader
    from fparser.two.parser import ParserFactory
    from fparser.two.utils import walk

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    parser = ParserFactory().create(std="f2008")
    selected = {name.lower() for name in procedures}
    selected_modules = {name.split("::", 1)[0] for name in selected if "::" in name}
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
        modules_in_file = {m.group(1).lower() for m in re.finditer(r"(?im)^\s*module\s+([a-z_]\w*)\b", result.stdout)}
        if modules_in_file & selected_modules:
            tree = parser(FortranStringReader(result.stdout, ignore_comments=False))
            for module in (node for node in walk(tree) if type(node).__name__ == "Module"):
                header = next(node for node in module.content if type(node).__name__ == "Module_Stmt")
                module_name = str(header.items[1]).lower()
                for routine in (node for node in walk(module) if type(node).__name__ == "Subroutine_Subprogram"):
                    statement = next(node for node in routine.content if type(node).__name__ == "Subroutine_Stmt")
                    name = module_name + "::" + str(statement.items[1]).lower()
                    if name not in selected:
                        continue
                    ending = routine.content[-1]
                    first, last = statement.item.span[0], ending.item.span[1]
                    for index in range(first - 1, last):
                        if re.match(r"\s*!\$omp\b", lines[index], re.I):
                            removed.append({"line": index + 1, "directive": lines[index].rstrip(), "procedure": name})
                            lines[index] = "\n"
        target = directory / (hashlib.sha256(str(source).encode()).hexdigest()[:12] + "_" + source.name)
        target.write_text("".join(lines))
        if _identity(source) != before:
            raise ValueError("Source changed while it was preprocessed: " + str(source))
        records.append({"original": before, "prepared": _identity(target), "command": command,
                        "serialized_directives": removed})
    return {"version": 1, "kind": "configured_resident_sources", "defines": list(defines),
            "procedures": sorted(selected), "sources": records}
